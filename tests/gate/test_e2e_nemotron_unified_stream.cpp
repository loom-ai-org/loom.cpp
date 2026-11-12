// Nemotron-Labs-Unified-ASR live (ADR-110): NeMo's buffered streaming mode through `TranscriptionStream` -- each
// step the whole model over up to `left + chunk + right` seconds under the chunked attention mask of those sizes,
// the buffer's chunk rows decoded by the RNN-T loop. The stream ends on what NeMo's own streaming script decodes.
//
// The oracle is NeMo, not the one pass: buffered streaming sees less context than the offline encoder, so its
// ids differ from `infer`'s (jfk: 37 against 38). Reference: `loom-exporter/fixture_gen/
// reference_stream_nemotron_unified.py`, one directory per clip, configuration and language,
// `<clip>.<left>-<chunk>-<right>.<lang>/`.
//
//   1. Tensor oracle: `encoder_step` over each of NeMo's buffers (where the reference's own step facts put them),
//      its decoded rows against NeMo's to 1e-3 of their absmax.
//   2. The ids after `finish` are NeMo's streamed ids, at 100 ms pieces and the whole clip on every case; at
//      10 ms, one chunk and 0.5 s on the first.
//   3. Text arrives before the end and is never revised: each emission is final.
//   4. Configurations NeMo has no reference for are refused: no right context, a chunk under one frame, a buffer
//      longer than the step takes. A language the file does not declare is refused too.
//   5. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where they
//      end alone.
//   6. With LOOM_UNIFIED_STREAM_OLD_GGUF, the file exported before the stream functions: the same ids in one
//      call, and the live door refuses it.
//
// LOOM_UNIFIED_STREAM_GGUF: nemotron-labs-unified-asr-0.6b exported with the stream functions.
// LOOM_UNIFIED_STREAM_CLIPS_DIR: `<clip>.npy`, mono F32 at 16 kHz. LOOM_UNIFIED_STREAM_REF_DIR: the reference.

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
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr size_t kRate = 16000;
constexpr size_t kFrameSamples = 1280;  // an encoder frame: 8 hops of 160

// One encoder step over clip samples [stop - total, stop), the decoded rows [row0, row0 + rows) returned.
const char* const kProbe = R"(
function probe_step(inputs)
    local x, w = inputs.waveform, {}
    for k = 1, inputs.total do w[k] = x[inputs.stop - inputs.total + k] end
    loom.run_subgraph_and_retain('encoder_step', {n_samples = #w, n_tokens = 0, n_past = 0},
                                  {waveform = w, length = {#w}, prompt = {inputs.prompt},
                                   att_left = {inputs.att[1]}, att_chunk = {inputs.att[2]},
                                   att_right = {inputs.att[3]}})
    local shape = loom.output_shape('encoder_step', 1)
    -- The step's frame count first, then the rows.
    local out, d, rows = loom.get_output('encoder_step', 1), shape[1], {shape[2]}
    for i = inputs.row0 * d + 1, (inputs.row0 + inputs.rows) * d do rows[#rows + 1] = out[i] end
    return rows
end
)";

struct Case {
    std::string dir, clip, lang;
    double seconds[3];  // left, chunk, right
};

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> parts;
    std::stringstream in(s);
    for (std::string p; std::getline(in, p, sep);) parts.push_back(p);
    return parts;
}

std::vector<float> read_f32(const std::string& path) {
    std::vector<int64_t> shape;
    return loom_test::read_npy_f32(path, shape);
}

std::vector<int32_t> as_ids(const std::vector<float>& v) { return std::vector<int32_t>(v.begin(), v.end()); }

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

loom::audio::TranscriptionStream::Inputs stream_inputs(const loom::GgufModel& model, const Case& c) {
    return {{"language", loom::audio::TranscriptionStream::language_input(model, c.lang)},
            {"left_seconds", c.seconds[0]}, {"chunk_seconds", c.seconds[1]}, {"right_seconds", c.seconds[2]}};
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_UNIFIED_STREAM_GGUF");
    const char* clips_dir = loom_test::fixture_env("LOOM_UNIFIED_STREAM_CLIPS_DIR");
    const char* ref_dir = loom_test::fixture_env("LOOM_UNIFIED_STREAM_REF_DIR");
    if (!gguf || !clips_dir || !ref_dir) {
        std::fprintf(stderr, "SKIP: set LOOM_UNIFIED_STREAM_GGUF, _CLIPS_DIR and _REF_DIR\n");
        return kSkipReturnCode;
    }
    std::vector<Case> cases;
    for (const auto& e : std::filesystem::directory_iterator(ref_dir)) {
        if (!e.is_directory()) continue;
        // `<clip>.<left>-<chunk>-<right>.<lang>`; the sizes may hold dots of their own.
        const std::string name = e.path().filename().string();
        const size_t first = name.find('.'), last = name.rfind('.');
        const std::vector<std::string> sizes = split(name.substr(first + 1, last - first - 1), '-');
        LOOM_CHECK(first != last && sizes.size() == 3);
        if (sizes.size() != 3) continue;
        cases.push_back({e.path().string(), name.substr(0, first), name.substr(last + 1),
                         {std::stod(sizes[0]), std::stod(sizes[1]), std::stod(sizes[2])}});
    }
    std::sort(cases.begin(), cases.end(), [](const Case& a, const Case& b) { return a.dir < b.dir; });
    LOOM_CHECK(!cases.empty());

    ggml_backend_ptr backend(loom_test::cpu_backend());
    auto model = loom::GgufModel::load(gguf, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, backend.get());
    loom::LoomLuaBridge& bridge = session.bridge();
    bridge.load_script(kProbe);
    LOOM_CHECK(loom::audio::TranscriptionStream::supported(bridge));

    double audio_s = 0.0, stream_s = 0.0;
    std::vector<Run> alone;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case& c = cases[ci];
        const std::vector<float> wav = read_f32(std::string(clips_dir) + "/" + c.clip + ".npy");
        const std::vector<double> whole(wav.begin(), wav.end());
        const double prompt = loom::audio::TranscriptionStream::language_input(*model, c.lang);

        // 1. The rows, step by step, against NeMo's. The script's frames: `int(seconds * 100 / 8)`.
        double att[3];
        for (int k = 0; k < 3; ++k) att[k] = std::floor(c.seconds[k] * 100.0 / 8);
        const std::vector<float> steps = read_f32(c.dir + "/steps.npy");
        const size_t n_steps = steps.size() / 3;
        double worst = 0.0;
        for (size_t s = 0; s < n_steps; ++s) {
            const double total = steps[3 * s], row0 = steps[3 * s + 1], rows = steps[3 * s + 2];
            // A step's buffer ends at its right context's end, the last's at the clip's.
            const double stop = s + 1 == n_steps ? static_cast<double>(wav.size())
                                                 : kFrameSamples * (att[1] + att[2] + att[1] * s);
            const auto got = std::get<std::vector<double>>(bridge.call(
                "probe_step", {{"waveform", whole}, {"stop", stop}, {"total", total}, {"prompt", prompt},
                               {"att", std::vector<double>{att[0], att[1], att[2]}}, {"row0", row0},
                               {"rows", rows}}));
            const std::vector<float> want = read_f32(c.dir + "/rows_" + std::to_string(s) + ".npy");
            LOOM_CHECK(got.size() == want.size() + 1);
            if (got.size() != want.size() + 1) break;
            float absmax = 0.0f;
            for (float v : want) absmax = std::max(absmax, std::fabs(v));
            double d = 0.0;
            for (size_t i = 0; i < want.size(); ++i) d = std::max(d, std::fabs(got[i + 1] - want[i]) / absmax);
            worst = std::max(worst, d);
        }
        LOOM_CHECK(worst <= 1e-3);

        // 2. The ids, at every cut tried.
        const std::vector<int32_t> want = as_ids(read_f32(c.dir + "/ids.npy"));
        LOOM_CHECK(!want.empty());
        loom::audio::TranscriptionStream stream(bridge, *model, stream_inputs(*model, c));
        const Run base = streamed(stream, wav, kRate / 10);
        stream_s += base.seconds;
        audio_s += static_cast<double>(wav.size()) / kRate;
        LOOM_CHECK(base.ids == want);
        std::vector<size_t> cuts = {wav.size()};
        if (ci == 0) cuts.insert(cuts.end(), {kRate / 100, static_cast<size_t>(kFrameSamples * att[1]), kRate / 2});
        size_t same = base.ids == want;
        for (size_t piece : cuts) {
            stream.reset();
            const Run r = streamed(stream, wav, piece);
            same += r.ids == want;
            LOOM_CHECK(r.ids == want);
        }

        // 3. Text before the end, never revised.
        if (wav.size() > kRate * 3) LOOM_CHECK(base.updates_before_finish > 0);
        LOOM_CHECK(!base.revised);
        alone.push_back(base);
        std::ifstream text_in(c.dir + "/text.txt");
        const std::string nemo_text((std::istreambuf_iterator<char>(text_in)), std::istreambuf_iterator<char>());
        std::printf("%-44s %6.2f s  %3zu steps  rows max|d|/absmax %.2e  ids %zu (%s)  %zu/%zu cuts same  "
                    "%zu updates  %.2fx real time\n  \"%s\"\n  NeMo \"%s\"\n",
                    std::filesystem::path(c.dir).filename().string().c_str(), static_cast<double>(wav.size()) / kRate,
                    n_steps, worst, want.size(), base.ids == want ? "NeMo's" : "DIFFERENT", same, cuts.size() + 1,
                    base.updates_before_finish, static_cast<double>(wav.size()) / kRate / base.seconds,
                    base.text.c_str(), nemo_text.c_str());
    }
    std::printf("streamed %.1f s of audio in 100 ms pieces in %.1f s: %.2fx real time, RTF %.3f\n", audio_s,
                stream_s, audio_s / stream_s, stream_s / audio_s);

    const Case& first = cases.front();
    const std::vector<float> first_wav = read_f32(std::string(clips_dir) + "/" + first.clip + ".npy");

    // 4. What NeMo has no reference for.
    {
        auto with = [&](const char* key, double value) {
            loom::audio::TranscriptionStream::Inputs inputs = stream_inputs(*model, first);
            inputs[key] = value;
            return inputs;
        };
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, with("right_seconds", 0.0)), loom::Error);
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, with("chunk_seconds", 0.05)), loom::Error);
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, with("left_seconds", 30.0)), loom::Error);
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream::language_input(*model, "xx-XX"), loom::LoadError);
    }

    // 5. Two streams and a one-pass call between them: the first case and the last.
    {
        const Case& last = cases.back();
        const std::vector<float> b_wav = read_f32(std::string(clips_dir) + "/" + last.clip + ".npy");
        loom::audio::TranscriptionStream a(bridge, *model, stream_inputs(*model, first));
        loom::audio::TranscriptionStream b(bridge, *model, stream_inputs(*model, last));
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

    // 6. The file exported before the door: the one pass unchanged, the door refused.
    if (const char* old_gguf = loom_test::fixture_env("LOOM_UNIFIED_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old_gguf, backend.get());
        LOOM_CHECK(old_model != nullptr);
        loom::Session old_session(*old_model, backend.get());
        LOOM_CHECK(loom::audio::transcribe(old_session.bridge(), *old_model, first_wav, {}).text ==
                   loom::audio::transcribe(bridge, *model, first_wav, {}).text);
        LOOM_CHECK(!loom::audio::TranscriptionStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(old_session.bridge(), *old_model), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
