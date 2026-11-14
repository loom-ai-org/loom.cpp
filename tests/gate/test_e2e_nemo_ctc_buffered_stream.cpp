// A NeMo CTC model live (ADR-123): NeMo's `FrameBatchASR` through `TranscriptionStream`. The clip's raw log-mel
// a chunk at a time (`raw_mel` over windows reaching past the chunk), each chunk slid into a mel buffer that is
// normalised over itself and run through the model (`ctc_buffer`), each buffer's middle frames kept, one greedy
// CTC merge over them. The stream ends on what NeMo's buffered CTC script transcribes.
//
// The oracle is NeMo, not the one pass: the buffered transcript differs from the one pass's (jfk at the default:
// "and so am my fellow americans", 29 ids against 28). Reference: `loom-exporter/fixture_gen/
// reference_stream_nemo_ctc_buffered.py`, one directory per clip and configuration, `<clip>.<chunk>-<total>/`.
//
//   1. Tensor oracle, both graphs: `raw_mel` over each chunk's window, its kept frames against NeMo's whole-clip
//      mel; `ctc_buffer` over each of NeMo's buffers (built from NeMo's mel as `FeatureFrameBufferer` builds
//      them), its kept log-prob rows against NeMo's to 1e-3 of their absmax, its frame count NeMo's.
//   2. The ids after `finish` are NeMo's, at 100 ms pieces and the whole clip on every case; at 10 ms, one chunk
//      and 0.5 s on the first.
//   3. Text arrives before the end and is never revised.
//   4. Configurations without a reference are refused: a chunk under a mel frame, a buffer shorter than its
//      chunk, a buffer longer than the step takes.
//   5. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where they
//      end alone.
//   6. With LOOM_NEMO_CTC_STREAM_OLD_GGUF, the file exported before the stream functions: the same one-pass
//      text, and the live door refuses it.
//
// LOOM_NEMO_CTC_STREAM_GGUF: the model exported with the stream functions. LOOM_NEMO_CTC_STREAM_CLIPS_DIR:
// `<clip>.npy`, mono F32 at 16 kHz. LOOM_NEMO_CTC_STREAM_REF_DIR: the reference.

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
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr size_t kRate = 16000;
constexpr int kHop = 160;
constexpr double kFeatureStride = 0.01;
constexpr int kMargin = 4;

// Mel frames [f0, f1) of the padded clip from `raw_mel` over the driver's window; and one buffer through
// `ctc_buffer`, its frame count then its rows [first, first + rows).
const char* const kProbe = R"(
function probe_mel(inputs)
    local x, f0, f1, n = inputs.waveform, inputs.f0, inputs.f1, inputs.n_total
    local s0 = math.max(f0 - inputs.margin, 0) * inputs.hop
    local s1 = math.min((f1 + inputs.margin) * inputs.hop, n)
    local w = {}
    for i = 1, s1 - s0 do w[i] = x[s0 + i] or 0.0 end
    loom.run_subgraph_and_retain('raw_mel', {n_samples = #w, n_tokens = 0, n_past = 0}, {waveform = w, length = {#w}})
    local out, rows, d = loom.get_output('raw_mel', 1), {}, loom.output_shape('raw_mel', 1)[1]
    local j0 = f0 - s0 / inputs.hop
    for i = 1, (f1 - f0) * d do rows[i] = out[j0 * d + i] end
    return rows
end
function probe_buffer(inputs)
    loom.run_subgraph_and_retain('ctc_buffer', {n_tokens = inputs.frames, n_past = 0},
                                  {features = inputs.features, length = {inputs.frames}})
    local shape = loom.output_shape('ctc_buffer', 1)
    local out, v, rows = loom.get_output('ctc_buffer', 1), shape[1], {shape[2]}
    for i = inputs.first * v + 1, (inputs.first + inputs.rows) * v do rows[#rows + 1] = out[i] end
    return rows
end
)";

struct Case {
    std::string dir, clip;
    double chunk = 0.0, total = 0.0;
    std::map<std::string, double> facts;
};

std::vector<float> read_f32(const std::string& path, std::vector<int64_t>* shape_out = nullptr) {
    std::vector<int64_t> shape;
    std::vector<float> v = loom_test::read_npy_f32(path, shape);
    if (shape_out) *shape_out = shape;
    return v;
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

loom::audio::TranscriptionStream::Inputs stream_inputs(const Case& c) {
    return {{"chunk_seconds", c.chunk}, {"total_seconds", c.total}};
}

double max_rel(const std::vector<double>& got, size_t offset, const std::vector<float>& want) {
    float absmax = 0.0f;
    for (float v : want) absmax = std::max(absmax, std::fabs(v));
    double d = 0.0;
    for (size_t i = 0; i < want.size(); ++i) d = std::max(d, std::fabs(got[offset + i] - want[i]) / absmax);
    return d;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_NEMO_CTC_STREAM_GGUF");
    const char* clips_dir = loom_test::fixture_env("LOOM_NEMO_CTC_STREAM_CLIPS_DIR");
    const char* ref_dir = loom_test::fixture_env("LOOM_NEMO_CTC_STREAM_REF_DIR");
    if (!gguf || !clips_dir || !ref_dir) {
        std::fprintf(stderr, "SKIP: set LOOM_NEMO_CTC_STREAM_GGUF, _CLIPS_DIR and _REF_DIR\n");
        return kSkipReturnCode;
    }
    std::vector<Case> cases;
    for (const auto& e : std::filesystem::directory_iterator(ref_dir)) {
        if (!e.is_directory()) continue;
        // `<clip>.<chunk>-<total>`; the sizes may hold dots of their own.
        const std::string name = e.path().filename().string();
        const size_t dot = name.find('.'), dash = name.rfind('-');
        LOOM_CHECK(dot != std::string::npos && dash != std::string::npos && dash > dot);
        if (dot == std::string::npos || dash == std::string::npos || dash < dot) continue;
        Case c{e.path().string(), name.substr(0, dot), std::stod(name.substr(dot + 1, dash - dot - 1)),
               std::stod(name.substr(dash + 1)), {}};
        std::ifstream facts(c.dir + "/facts.txt");
        for (std::string key; facts >> key;) facts >> c.facts[key];
        cases.push_back(c);
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
        // The script's numbers, in its order of operations (as the driver computes them).
        const double stride_s = kFeatureStride * c.facts.at("model_stride");
        const int chunk = static_cast<int>(std::floor(c.chunk / kFeatureStride));
        const int total = static_cast<int>(std::floor(c.total / kFeatureStride));
        const int tokens = static_cast<int>(c.facts.at("tokens_per_chunk"));
        const int delay = static_cast<int>(c.facts.at("delay"));
        const size_t pad = static_cast<size_t>(std::floor(delay * stride_s * kRate));
        LOOM_CHECK(tokens == static_cast<int>(std::ceil(c.chunk / stride_s)));

        // 1a. raw_mel over each chunk's window, against NeMo's whole-clip mel.
        std::vector<int64_t> mel_shape;
        const std::vector<float> mel = read_f32(c.dir + "/mel.npy", &mel_shape);
        const int features = static_cast<int>(mel_shape[0]), n_mel = static_cast<int>(mel_shape[1]);
        std::vector<double> padded(wav.begin(), wav.end());
        padded.resize(wav.size() + pad, 0.0);
        LOOM_CHECK(features == static_cast<int>(padded.size() / kHop));
        double mel_worst = 0.0;
        for (int f0 = 0; f0 < features; f0 += chunk) {
            const int f1 = std::min(f0 + chunk, features);
            const auto got = std::get<std::vector<double>>(bridge.call(
                "probe_mel", {{"waveform", padded}, {"f0", static_cast<double>(f0)}, {"f1", static_cast<double>(f1)},
                              {"n_total", static_cast<double>(padded.size())}, {"hop", static_cast<double>(kHop)},
                              {"margin", static_cast<double>(kMargin)}}));
            LOOM_CHECK(got.size() == static_cast<size_t>((f1 - f0) * n_mel));
            for (size_t i = 0; i < got.size() && i + f0 * n_mel < mel.size(); ++i)
                mel_worst = std::max(mel_worst, std::fabs(got[i] - mel[f0 * n_mel + i]));
        }
        LOOM_CHECK(mel_worst <= 1e-3);

        // 1b. ctc_buffer over NeMo's buffers, built from NeMo's mel as `FeatureFrameBufferer` builds them.
        const std::vector<float> steps = read_f32(c.dir + "/steps.npy");
        const size_t n_steps = steps.size() / 2;
        std::vector<double> buffer(static_cast<size_t>(total) * n_mel, c.facts.at("fill"));
        double rows_worst = 0.0;
        for (size_t s = 0; s < n_steps; ++s) {
            const int f0 = static_cast<int>(s) * chunk;
            std::rotate(buffer.begin(), buffer.begin() + static_cast<long>(chunk) * n_mel, buffer.end());
            for (int i = 0; i < chunk * n_mel; ++i) {
                const size_t at = static_cast<size_t>(f0) * n_mel + i;
                buffer[buffer.size() - static_cast<size_t>(chunk) * n_mel + i] =
                    f0 * n_mel + i < features * n_mel ? mel[at] : 0.0;
            }
            const int first = static_cast<int>(steps[2 * s + 1]);
            const auto got = std::get<std::vector<double>>(bridge.call(
                "probe_buffer", {{"features", buffer}, {"frames", static_cast<double>(total)},
                                 {"first", static_cast<double>(first)}, {"rows", static_cast<double>(tokens)}}));
            LOOM_CHECK(!got.empty() && got[0] == steps[2 * s]);
            const std::vector<float> want = read_f32(c.dir + "/logits_" + std::to_string(s) + ".npy");
            LOOM_CHECK(got.size() == want.size() + 1);
            if (got.size() != want.size() + 1) break;
            rows_worst = std::max(rows_worst, max_rel(got, 1, want));
        }
        LOOM_CHECK(rows_worst <= 1e-3);

        // 2. The ids, at every cut tried.
        const std::vector<int32_t> want = as_ids(read_f32(c.dir + "/ids.npy"));
        LOOM_CHECK(!want.empty());
        loom::audio::TranscriptionStream stream(bridge, *model, stream_inputs(c));
        const Run base = streamed(stream, wav, kRate / 10);
        stream_s += base.seconds;
        audio_s += static_cast<double>(wav.size()) / kRate;
        LOOM_CHECK(base.ids == want);
        std::vector<size_t> cuts = {wav.size()};
        if (ci == 0) cuts.insert(cuts.end(), {kRate / 100, static_cast<size_t>(chunk) * kHop, kRate / 2});
        size_t same = base.ids == want;
        for (size_t piece : cuts) {
            stream.reset();
            const Run r = streamed(stream, wav, piece);
            same += r.ids == want;
            LOOM_CHECK(r.ids == want);
        }

        // 2b. The mel the driver itself builds: its buffer (state `buffer`, raw log-mel, frame-major) after each
        //     step against NeMo's, built from NeMo's mel as 1b builds it. The probe in 1a computes its own
        //     windows, so without this a driver whose window margin no longer covers a frame's reach (1 frame
        //     for 4) still passed: the edge frames move, and jfk's ids happened not to. One hop per push, so a
        //     step is seen as it lands.
        double state_worst = 0.0;
        size_t state_steps = 0;
        {
            stream.reset();
            std::vector<double> expect(static_cast<size_t>(total) * n_mel, c.facts.at("fill"));
            // NeMo's buffer after step s, from NeMo's mel: the chunk slid in, its missing frames 0.
            auto advance = [&](size_t s) {
                const int f0 = static_cast<int>(s) * chunk;
                std::rotate(expect.begin(), expect.begin() + static_cast<long>(chunk) * n_mel, expect.end());
                for (int i = 0; i < chunk * n_mel; ++i)
                    expect[expect.size() - static_cast<size_t>(chunk) * n_mel + i] =
                        f0 * n_mel + i < features * n_mel ? mel[static_cast<size_t>(f0) * n_mel + i] : 0.0;
            };
            auto compare = [&]() {
                const loom::StreamState::Value* got = stream.state().value("buffer");
                LOOM_CHECK(got != nullptr && got->data.size() == expect.size());
                if (got == nullptr || got->data.size() != expect.size()) return;
                for (size_t i = 0; i < expect.size(); ++i)
                    state_worst = std::max(state_worst, std::fabs(got->data[i] - expect[i]));
            };
            auto next_frame = [&]() {
                const loom::StreamState::Value* v = stream.state().value("next_frame");
                return v && !v->data.empty() ? v->data[0] : -1.0;
            };
            double seen = next_frame();
            for (size_t at = 0; at < wav.size(); at += kHop) {
                stream.push(wav.data() + at, std::min(static_cast<size_t>(kHop), wav.size() - at));
                if (next_frame() != seen) {
                    seen = next_frame();
                    advance(state_steps++);
                    compare();
                }
            }
            // `finish` runs the rest at once; only its last buffer is left to read, and it is NeMo's last.
            stream.finish();
            while (state_steps < n_steps) advance(state_steps++);
            compare();
            LOOM_CHECK(state_steps == n_steps);
            LOOM_CHECK(state_worst <= 1e-3);
            LOOM_CHECK(stream.tokens() == want);
        }

        // 3. Text before the end, never revised: a clip that runs on past a buffer's middle by two chunks.
        if (wav.size() > (static_cast<size_t>(total) + 2 * chunk) * kHop) LOOM_CHECK(base.updates_before_finish > 0);
        LOOM_CHECK(!base.revised);
        alone.push_back(base);
        std::ifstream text_in(c.dir + "/text.txt");
        const std::string nemo_text((std::istreambuf_iterator<char>(text_in)), std::istreambuf_iterator<char>());
        std::printf("%-28s %6.2f s  %3zu buffers  mel max|d| %.2e (driver's %.2e)  rows max|d|/absmax %.2e  "
                    "ids %zu (%s)  %zu/%zu cuts same  %zu updates  %.2fx real time\n  \"%s\"\n  NeMo \"%s\"\n",
                    std::filesystem::path(c.dir).filename().string().c_str(), static_cast<double>(wav.size()) / kRate,
                    n_steps, mel_worst, state_worst, rows_worst, want.size(), base.ids == want ? "NeMo's" : "DIFFERENT", same,
                    cuts.size() + 1, base.updates_before_finish, static_cast<double>(wav.size()) / kRate / base.seconds,
                    base.text.c_str(), nemo_text.c_str());
    }
    std::printf("streamed %.1f s of audio in 100 ms pieces in %.1f s: %.2fx real time, RTF %.3f\n", audio_s,
                stream_s, audio_s / stream_s, stream_s / audio_s);

    const Case& first = cases.front();
    const std::vector<float> first_wav = read_f32(std::string(clips_dir) + "/" + first.clip + ".npy");

    // 4. What NeMo has no reference for.
    {
        auto with = [&](const char* key, double value) {
            loom::audio::TranscriptionStream::Inputs inputs = stream_inputs(first);
            inputs[key] = value;
            return inputs;
        };
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, with("chunk_seconds", 0.005)), loom::Error);
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, with("total_seconds", first.chunk / 2)),
                          loom::Error);
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, with("total_seconds", 30.0)), loom::Error);
    }

    // 5. Two streams and a one-pass call between them: the first case and the last.
    {
        const Case& last = cases.back();
        const std::vector<float> b_wav = read_f32(std::string(clips_dir) + "/" + last.clip + ".npy");
        loom::audio::TranscriptionStream a(bridge, *model, stream_inputs(first));
        loom::audio::TranscriptionStream b(bridge, *model, stream_inputs(last));
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
    if (const char* old_gguf = loom_test::fixture_env("LOOM_NEMO_CTC_STREAM_OLD_GGUF")) {
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
