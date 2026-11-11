// Moonshine Streaming live (ADR-106): the encoder a window at a time, the decoder re-run, through
// `TranscriptionStream` -- the stream ends on what one call over the clip decodes.
//
//   1. Tensor oracle: the stream's final encoder rows (its `rows` state) against the `encoder` topology run
//      once over the whole clip, framed as `infer` frames it. Windows with the receptive field either side
//      are the one pass to f32 rounding.
//   2. The ids after `finish` are `infer`'s ids exactly, and the text is `transcribe`'s, for every clip and
//      every piece size (20 ms, 100 ms, 0.5 s, the whole clip), at the default cadence and at
//      `update_seconds` 1.0 -- the cadence is a cost knob, not a result.
//   3. Mid-stream updates arrive before the end, and an update that changes earlier text says `revised`.
//   4. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where
//      they end alone.
//   5. With LOOM_MOONSHINE_STREAM_OLD_GGUF, the file exported before the stream functions: the same ids in
//      one call, its encoder bit-identical to this one's at `row_offset` 0, and the live door refuses it.
//
// LOOM_MOONSHINE_STREAM_GGUF: moonshine-streaming-tiny or -small exported with the stream functions.
// LOOM_MOONSHINE_STREAM_CLIPS_DIR: `*.npy`, mono F32 at 16 kHz -- real speech, since a decoder on noise says
// little and a wrong row can hide in it; lengths that end mid-frame and one past 30 s.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr size_t kFrame = 80;
// A window's encoder against one call's: the same rows, summed in another order (the window is shorter).
constexpr double kRowTolerance = 1e-4;

const char* const kProbe = R"(
function probe_encoder(inputs)
    return loom.run_subgraph('encoder', {n_samples = #inputs.waveform, n_past = 0},
                             {waveform = inputs.waveform, row_offset = {0}})
end
)";

// `infer`'s framing, as `transcribe` and 01_frames.lua make it: whole 80-sample frames, the partial one
// zeroed.
std::vector<double> framed(const std::vector<float>& wav) {
    std::vector<double> out(wav.begin(), wav.end());
    const size_t whole = wav.size() / kFrame * kFrame;
    out.resize((wav.size() + kFrame - 1) / kFrame * kFrame, 0.0);
    std::fill(out.begin() + whole, out.end(), 0.0);
    return out;
}

std::vector<int32_t> one_pass_ids(loom::LoomLuaBridge& bridge, const std::vector<float>& wav) {
    const auto got = std::get<std::vector<double>>(bridge.call(
        "infer", {{"waveform", framed(wav)}, {"audio_samples", static_cast<double>(wav.size())}}));
    return std::vector<int32_t>(got.begin(), got.end());
}

struct Run {
    std::vector<int32_t> ids;
    std::string text;
    size_t updates_before_finish = 0;
    bool revised = false;
};

Run streamed(loom::audio::TranscriptionStream& s, const std::vector<float>& wav, size_t piece) {
    Run run;
    for (size_t at = 0; at < wav.size(); at += piece) {
        const loom::audio::TranscriptUpdate u = s.push(wav.data() + at, std::min(piece, wav.size() - at));
        if (!u.delta.empty()) ++run.updates_before_finish;
        run.revised = run.revised || u.revised;
    }
    const loom::audio::TranscriptUpdate last = s.finish();
    run.revised = run.revised || last.revised;
    run.ids = s.tokens();
    run.text = s.text();
    return run;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_MOONSHINE_STREAM_GGUF");
    const char* dir = loom_test::fixture_env("LOOM_MOONSHINE_STREAM_CLIPS_DIR");
    if (!gguf || !dir) {
        std::fprintf(stderr, "SKIP: set LOOM_MOONSHINE_STREAM_GGUF and LOOM_MOONSHINE_STREAM_CLIPS_DIR\n");
        return kSkipReturnCode;
    }
    std::vector<std::string> clips;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.path().extension() == ".npy") clips.push_back(e.path().string());
    }
    std::sort(clips.begin(), clips.end());
    LOOM_CHECK(!clips.empty());

    ggml_backend_ptr backend(loom_test::cpu_backend());
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());
    loom::LoomLuaBridge& bridge = session.bridge();
    bridge.load_script(kProbe);
    LOOM_CHECK(loom::audio::TranscriptionStream::supported(bridge));

    double audio_s = 0.0, stream_s = 0.0;
    for (const std::string& path : clips) {
        std::vector<int64_t> shape;
        const std::vector<float> wav = loom_test::read_npy_f32(path, shape);
        const std::string name = std::filesystem::path(path).stem().string();
        const std::vector<int32_t> want = one_pass_ids(bridge, wav);
        const std::string want_text = loom::audio::transcribe(bridge, *model, wav, {}).text;
        LOOM_CHECK(!want.empty());

        // 1. The rows.
        const auto one = std::get<std::vector<double>>(bridge.call("probe_encoder", {{"waveform", framed(wav)}}));
        loom::audio::TranscriptionStream s(bridge, *model);
        const auto t0 = std::chrono::steady_clock::now();
        const Run base = streamed(s, wav, 1600);
        stream_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        audio_s += wav.size() / 16000.0;
        const loom::StreamState::Value* rows = s.state().value("rows");
        LOOM_CHECK(rows != nullptr && rows->data.size() == one.size());
        double d = 0.0;
        if (rows && rows->data.size() == one.size()) {
            for (size_t i = 0; i < one.size(); ++i) d = std::max(d, std::fabs(rows->data[i] - one[i]));
        }
        LOOM_CHECK(d <= kRowTolerance);

        // 2. The ids, at every cut and two cadences.
        LOOM_CHECK(base.ids == want);
        LOOM_CHECK(base.text == want_text);
        size_t cases = 0, same = 0;
        for (size_t piece : {size_t(320), size_t(8000), wav.size()}) {
            s.reset();
            const Run r = streamed(s, wav, piece);
            ++cases;
            same += r.ids == want;
            LOOM_CHECK(r.ids == want);
        }
        loom::audio::TranscriptionStream slow(bridge, *model, {{"update_seconds", 1.0}});
        const Run r = streamed(slow, wav, 1600);
        ++cases;
        same += r.ids == want;
        LOOM_CHECK(r.ids == want);

        // 3. Text before the end, on anything longer than a decode's cadence plus the look-ahead.
        if (wav.size() > 16000 * 2) LOOM_CHECK(base.updates_before_finish > 0);
        std::printf("%-24s %6.2f s  rows max|d| %.3g  ids %zu  %zu/%zu cuts same  %zu updates%s\n  \"%s\"\n",
                    name.c_str(), wav.size() / 16000.0, d, want.size(), same + (base.ids == want), cases + 1,
                    base.updates_before_finish, base.revised ? ", revised" : "", base.text.c_str());
    }
    std::printf("streamed %.1f s of audio in 100 ms pieces in %.1f s: %.2fx real time, RTF %.3f\n", audio_s,
                stream_s, audio_s / stream_s, stream_s / audio_s);

    // 4. Two streams and a one-pass call between them.
    {
        std::vector<int64_t> shape;
        const std::vector<float> a_wav = loom_test::read_npy_f32(clips.front(), shape);
        const std::vector<float> b_wav = loom_test::read_npy_f32(clips.back(), shape);
        loom::audio::TranscriptionStream alone_a(bridge, *model), alone_b(bridge, *model);
        const Run want_a = streamed(alone_a, a_wav, 1600), want_b = streamed(alone_b, b_wav, 1600);
        loom::audio::TranscriptionStream a(bridge, *model), b(bridge, *model);
        const size_t n = std::max(a_wav.size(), b_wav.size());
        for (size_t at = 0; at < n; at += 1600) {
            if (at < a_wav.size()) a.push(a_wav.data() + at, std::min<size_t>(1600, a_wav.size() - at));
            if (at == 1600 * 20) loom::audio::transcribe(bridge, *model, b_wav, {});
            if (at < b_wav.size()) b.push(b_wav.data() + at, std::min<size_t>(1600, b_wav.size() - at));
        }
        a.finish();
        b.finish();
        LOOM_CHECK(a.tokens() == want_a.ids && a.text() == want_a.text);
        LOOM_CHECK(b.tokens() == want_b.ids && b.text() == want_b.text);
    }

    // 5. The file exported before the door.
    if (const char* old_gguf = loom_test::fixture_env("LOOM_MOONSHINE_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old_gguf, backend.get());
        LOOM_CHECK(old_model != nullptr);
        loom::Session old_session(*old_model, backend.get());
        std::vector<int64_t> shape;
        const std::vector<float> wav = loom_test::read_npy_f32(clips.front(), shape);
        LOOM_CHECK(one_pass_ids(old_session.bridge(), wav) == one_pass_ids(bridge, wav));
        // The encoder at offset 0 is the published encoder, bit for bit: `row_offset` added nothing.
        old_session.bridge().load_script(R"(
function probe_old_encoder(inputs)
    return loom.run_subgraph('encoder', {n_samples = #inputs.waveform, n_past = 0}, {waveform = inputs.waveform})
end
)");
        LOOM_CHECK(old_session.bridge().call("probe_old_encoder", {{"waveform", framed(wav)}}) ==
                   bridge.call("probe_encoder", {{"waveform", framed(wav)}}));
        LOOM_CHECK(!loom::audio::TranscriptionStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(old_session.bridge(), *old_model), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
