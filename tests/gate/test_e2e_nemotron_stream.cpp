// Nemotron 3.5 ASR live (ADR-109): the cache-aware encoder a few attention chunks per call with its streaming
// state carried, the RNN-T decode frame by frame, through `TranscriptionStream` -- the stream ends on what one
// call over the clip decodes.
//
//   1. Tensor oracle: `encoder_step` run over the clip the way the driver runs it, in uneven groups of chunks
//      (1, 3, 2, 7 chunks of 320 ms, the state kept on the device between calls) and then the clip's end,
//      against `encoder` once over the whole clip. The step is the one-call encoder to f32 rounding, past the
//      56-frame attention window and at the clip's last partial chunk.
//   2. The ids after `finish` are `infer`'s ids exactly, and the text `transcribe`'s, at 100 ms pieces and the
//      whole clip on every clip; at 10 ms, one chunk (320 ms) and 0.5 s on the first.
//   3. Text arrives before the end and is never revised: each emission is final.
//   4. A named language: a stream opened with `language_input("en-US")` ends on `transcribe`'s text with
//      `language = "en-US"`; a name the file does not declare is refused.
//   5. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where
//      they end alone.
//   6. With LOOM_NEMOTRON_STREAM_OLD_GGUF, the file exported before the stream functions: the same ids in one
//      call, and the live door refuses it.
//
// LOOM_NEMOTRON_STREAM_GGUF: nemotron-3.5-asr-streaming-0.6b exported with the stream functions.
// LOOM_NEMOTRON_STREAM_CLIPS_DIR: `*.npy`, mono F32 at 16 kHz -- real speech, the longest sorted last.

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
#include <regex>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr size_t kRate = 16000;
// The step against one call: the same rows summed in another order (f64 puts them 2e-14 apart).
constexpr double kRowTolerance = 1e-4;

// `encoder_step`'s state count, off its topology: the driver's own constant is a local of its chunk.
double step_states(const loom::GgufModel& model) {
    const std::string& json = model.topology_json("encoder_step");
    const std::regex state("\"state_([0-9]+)\"");
    int most = 0;
    for (auto it = std::sregex_iterator(json.begin(), json.end(), state); it != std::sregex_iterator(); ++it) {
        most = std::max(most, std::stoi((*it)[1].str()));
    }
    return most;
}

// The driver's window arithmetic (01_stream.lua), with the clip whole in hand: 160-sample hops, chunks of 32
// mel frames reaching 7 frames back, 256 centring zeros, a window of 160 M + 353 samples for M mel frames.
const char* const kProbe = R"(
function probe_encoder(inputs)
    return loom.run_subgraph('encoder', {n_samples = #inputs.waveform, n_past = 0},
                             {waveform = inputs.waveform, prompt = {inputs.prompt}})
end

function probe_step(inputs)
    local x, pieces, n_states = inputs.waveform, inputs.pieces, inputs.n_states
    local HOP, CM, LEAD, PADL, EXTRA = 160, 32, 7, 256, 353
    local n = #x
    local rows, c, i, encoded = {}, 0, 0, false
    local function call(frame0, n_mel, n_valid)
        local a = frame0 * HOP - 1 - PADL
        local w = {}
        for k = 1, HOP * n_mel + EXTRA do
            local s = a + k - 1
            w[k] = (s >= 0 and s < n) and x[s + 1] or 0.0
        end
        local args = {waveform = w, frame0 = {frame0}, n_valid = {n_valid}, prompt = {inputs.prompt}}
        local shift = 1
        if not encoded then
            loom.run_subgraph_and_retain('encoder_init', {n_tokens = 0, n_past = 0}, {zero = {0}})
            loom.keep_state('probe', 'encoder_init')
            shift = 0
            encoded = true
        end
        for k = 1, n_states do args['state_' .. k] = {state = 'probe', index = k + shift} end
        loom.run_subgraph_and_retain('encoder_step', {n_samples = #w, n_tokens = 0, n_past = 0}, args)
        loom.keep_state('probe', 'encoder_step')
        local out = loom.get_output('encoder_step', 1)
        for s = 1, #out do rows[#rows + 1] = out[s] end
    end
    local ready = math.floor(((n - PADL) / HOP + LEAD + 1) / CM)
    while ready > c do
        local g = math.min(ready - c, pieces[i % #pieces + 1])
        call(CM * c - LEAD, CM * g, HOP * CM * g + EXTRA - 1)
        c, i = c + g, i + 1
    end
    local frame0 = CM * c - LEAD
    local frames = math.floor(n / HOP)
    if frames - frame0 > 0 then call(frame0, frames - frame0, n - (frame0 * HOP - 1 - PADL) - 1) end
    return rows
end
)";

std::vector<int32_t> one_pass_ids(loom::LoomLuaBridge& bridge, const std::vector<float>& wav) {
    const auto got = std::get<std::vector<double>>(
        bridge.call("infer", {{"waveform", std::vector<double>(wav.begin(), wav.end())}}));
    return std::vector<int32_t>(got.begin(), got.end());
}

struct Run {
    std::vector<int32_t> ids;
    std::string text;
    size_t updates_before_finish = 0;
    bool revised = false;
    double seconds = 0.0;
};

Run streamed(loom::audio::TranscriptionStream& s, const std::vector<float>& wav, size_t piece) {
    Run run;
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t at = 0; at < wav.size(); at += piece) {
        const loom::audio::TranscriptUpdate u = s.push(wav.data() + at, std::min(piece, wav.size() - at));
        if (!u.delta.empty()) ++run.updates_before_finish;
        run.revised = run.revised || u.revised;
    }
    const loom::audio::TranscriptUpdate last = s.finish();
    run.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    run.revised = run.revised || last.revised;
    run.ids = s.tokens();
    run.text = s.text();
    return run;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_NEMOTRON_STREAM_GGUF");
    const char* dir = loom_test::fixture_env("LOOM_NEMOTRON_STREAM_CLIPS_DIR");
    if (!gguf || !dir) {
        std::fprintf(stderr, "SKIP: set LOOM_NEMOTRON_STREAM_GGUF and LOOM_NEMOTRON_STREAM_CLIPS_DIR\n");
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
    const double n_states = step_states(*model);
    LOOM_CHECK(n_states == 3 + 24 * 3);  // three subsampling stages; per layer keys, values, conv inputs
    // The ids the stream's `language` input takes, resolved as `transcribe` resolves the names.
    const double en_us = loom::audio::TranscriptionStream::language_input(*model, "en-US");
    const double auto_prompt = loom::audio::TranscriptionStream::language_input(*model, "auto");
    LOOM_CHECK_THROWS(loom::audio::TranscriptionStream::language_input(*model, "xx-XX"), loom::LoadError);

    double audio_s = 0.0, stream_s = 0.0;
    std::vector<Run> alone;
    for (size_t c = 0; c < clips.size(); ++c) {
        std::vector<int64_t> shape;
        const std::vector<float> wav = loom_test::read_npy_f32(clips[c], shape);
        const std::string name = std::filesystem::path(clips[c]).stem().string();
        const std::vector<double> whole(wav.begin(), wav.end());

        // 1. The rows.
        const auto one = std::get<std::vector<double>>(
            bridge.call("probe_encoder", {{"waveform", whole}, {"prompt", auto_prompt}}));
        const auto step = std::get<std::vector<double>>(bridge.call(
            "probe_step", {{"waveform", whole}, {"pieces", std::vector<double>{1, 3, 2, 7}},
                           {"n_states", n_states}, {"prompt", auto_prompt}}));
        LOOM_CHECK(step.size() == one.size() && !one.empty());
        double d = 0.0;
        if (step.size() == one.size()) {
            for (size_t i = 0; i < one.size(); ++i) d = std::max(d, std::fabs(step[i] - one[i]));
        }
        LOOM_CHECK(d <= kRowTolerance);

        // 2. The ids, at every cut tried.
        const std::vector<int32_t> want = one_pass_ids(bridge, wav);
        const std::string want_text = loom::audio::transcribe(bridge, *model, wav, {}).text;
        LOOM_CHECK(!want.empty());
        loom::audio::TranscriptionStream s(bridge, *model);
        const Run base = streamed(s, wav, kRate / 10);
        stream_s += base.seconds;
        audio_s += static_cast<double>(wav.size()) / kRate;
        LOOM_CHECK(base.ids == want);
        LOOM_CHECK(base.text == want_text);
        std::vector<size_t> cuts = {wav.size()};
        if (c == 0) cuts.insert(cuts.end(), {kRate / 100, size_t(5120), kRate / 2});
        size_t same = base.ids == want;
        for (size_t piece : cuts) {
            s.reset();
            const Run r = streamed(s, wav, piece);
            same += r.ids == want;
            LOOM_CHECK(r.ids == want);
        }

        // 3. Text before the end, never revised.
        if (wav.size() > kRate * 2) LOOM_CHECK(base.updates_before_finish > 0);
        LOOM_CHECK(!base.revised);
        alone.push_back(base);
        std::printf("%-26s %6.2f s  rows max|d| %.3g  ids %zu  %zu/%zu cuts same  %zu updates  %.2fx real time\n"
                    "  \"%s\"\n",
                    name.c_str(), static_cast<double>(wav.size()) / kRate, d, want.size(), same, cuts.size() + 1,
                    base.updates_before_finish, static_cast<double>(wav.size()) / kRate / base.seconds,
                    base.text.c_str());
    }
    std::printf("streamed %.1f s of audio in 100 ms pieces in %.1f s: %.2fx real time, RTF %.3f\n", audio_s,
                stream_s, audio_s / stream_s, stream_s / audio_s);

    std::vector<int64_t> shape;
    const std::vector<float> first_wav = loom_test::read_npy_f32(clips.front(), shape);

    // 4. A named language.
    {
        loom::audio::TranscribeOptions options;
        options.language = "en-US";
        const auto want = loom::audio::transcribe(bridge, *model, first_wav, options);
        loom::audio::TranscriptionStream s(bridge, *model, {{"language", en_us}});
        const Run r = streamed(s, first_wav, kRate / 10);
        LOOM_CHECK(r.text == want.text);
        std::printf("language en-US: \"%s\"\n", r.text.c_str());
    }

    // 5. Two streams and a one-pass call between them: the first clip and the last (the longest).
    {
        const std::vector<float> b_wav = loom_test::read_npy_f32(clips.back(), shape);
        loom::audio::TranscriptionStream a(bridge, *model), b(bridge, *model);
        const size_t n = std::max(first_wav.size(), b_wav.size()), piece = kRate / 10;
        for (size_t at = 0; at < n; at += piece) {
            if (at < first_wav.size()) a.push(first_wav.data() + at, std::min(piece, first_wav.size() - at));
            if (at == piece * 20) loom::audio::transcribe(bridge, *model, first_wav, {});
            if (at < b_wav.size()) b.push(b_wav.data() + at, std::min(piece, b_wav.size() - at));
        }
        a.finish();
        b.finish();
        LOOM_CHECK(a.tokens() == alone.front().ids && a.text() == alone.front().text);
        LOOM_CHECK(b.tokens() == alone.back().ids && b.text() == alone.back().text);
    }

    // 6. The file exported before the door.
    if (const char* old_gguf = loom_test::fixture_env("LOOM_NEMOTRON_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old_gguf, backend.get());
        LOOM_CHECK(old_model != nullptr);
        loom::Session old_session(*old_model, backend.get());
        LOOM_CHECK(one_pass_ids(old_session.bridge(), first_wav) == alone.front().ids);
        LOOM_CHECK(!loom::audio::TranscriptionStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(old_session.bridge(), *old_model), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
