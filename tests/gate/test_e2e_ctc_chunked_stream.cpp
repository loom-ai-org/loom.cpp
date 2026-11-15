// A wav2vec 2.0-style CTC model live (ADR-126): the transformers ASR pipeline's strided chunks through
// `TranscriptionStream`. Each chunk of the raw waveform is the one-pass graph's whole input, its strides' logit
// frames dropped, one greedy CTC merge over what is kept. The stream ends on what
// `pipeline(chunk_length_s, stride_length_s)` transcribes.
//
// The oracle is the pipeline, not the one pass: chunked ids differ from it at short chunks (pocket_tts_ref at
// 2 s: 98 ids against 95). Reference: `loom-exporter/fixture_gen/reference_stream_ctc_chunked.py`, one
// directory per clip and configuration, `<clip>.<chunk>-<stride>/`.
//
//   1. Tensor oracle: the graph over each of the pipeline's chunks, its kept logit rows against the pipeline's
//      to 1e-3 of their absmax, the kept span the pipeline's.
//   2. The ids after `finish` are the pipeline's, at 100 ms pieces and the whole clip on every case; at 10 ms,
//      one chunk and 0.5 s on the first.
//   3. Per chunk, as it runs: the stream's next chunk start and its ids so far are the pipeline's chunks'
//      (catches a stride rounded the wrong way on a clip whose final ids happen not to move).
//   4. Text arrives before the end and is never revised.
//   5. Sizes the pipeline refuses, or the graph cannot take, are refused; `stream_open` with no sizes is the
//      4 s / 0.66 s case.
//   6. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where they
//      end alone.
//   7. With LOOM_CTC_STREAM_OLD_GGUF, the file exported before the stream functions: the same one-pass text,
//      and the live door refuses it.
//
// LOOM_CTC_STREAM_GGUF: the model exported with the stream functions. LOOM_CTC_STREAM_CLIPS_DIR: `<clip>.npy`,
// mono F32 at 16 kHz. LOOM_CTC_STREAM_REF_DIR: the reference.

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
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr size_t kRate = 16000;

// The graph over samples [s, s + n) of `waveform`: its frame count, then rows [first, end) of its logits.
const char* const kProbe = R"(
function probe_chunk(inputs)
    local w = {}
    for i = 1, inputs.n do w[i] = inputs.waveform[inputs.s + i] end
    loom.run_subgraph_and_retain('main_topology', {n_samples = #w, n_past = 0}, {waveform = w})
    local shape = loom.output_shape('main_topology', 1)
    local out, v, rows = loom.get_output('main_topology', 1), shape[1], {shape[2]}
    for i = inputs.first * v + 1, inputs['end'] * v do rows[#rows + 1] = out[i] end
    return rows
end
)";

struct Case {
    std::string dir, clip;
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

// The sizes in seconds that round to the case's samples (whole logit frames, so the rounding is exact).
loom::audio::TranscriptionStream::Inputs stream_inputs(const Case& c) {
    return {{"chunk_seconds", c.facts.at("chunk_len") / kRate},
            {"left_stride_seconds", c.facts.at("stride_left") / kRate},
            {"right_stride_seconds", c.facts.at("stride_right") / kRate}};
}

// The greedy CTC merge over argmax ids, the previous id carried.
void merge(const std::vector<int32_t>& ids, int32_t blank, int32_t& prev, std::vector<int32_t>& out) {
    for (int32_t p : ids) {
        if (p != prev && p != blank) out.push_back(p);
        prev = p;
    }
}

std::vector<int32_t> argmax_rows(const std::vector<float>& rows, size_t width) {
    std::vector<int32_t> ids;
    for (size_t r = 0; r * width < rows.size(); ++r)
        ids.push_back(static_cast<int32_t>(std::max_element(rows.begin() + r * width, rows.begin() + (r + 1) * width) -
                                           (rows.begin() + r * width)));
    return ids;
}

double state_number(const loom::audio::TranscriptionStream& s, const char* key) {
    const loom::StreamState::Value* v = s.state().value(key);
    return v && !v->data.empty() ? v->data[0] : -1.0;
}

} // namespace

int main() {
    const char* gguf = loom_test::fixture_env("LOOM_CTC_STREAM_GGUF");
    const char* clips_dir = loom_test::fixture_env("LOOM_CTC_STREAM_CLIPS_DIR");
    const char* ref_dir = loom_test::fixture_env("LOOM_CTC_STREAM_REF_DIR");
    if (!gguf || !clips_dir || !ref_dir) {
        std::fprintf(stderr, "SKIP: set LOOM_CTC_STREAM_GGUF, _CLIPS_DIR and _REF_DIR\n");
        return kSkipReturnCode;
    }
    std::vector<Case> cases;
    for (const auto& e : std::filesystem::directory_iterator(ref_dir)) {
        if (!e.is_directory()) continue;
        const std::string name = e.path().filename().string();
        const size_t dot = name.find('.');
        LOOM_CHECK(dot != std::string::npos);
        if (dot == std::string::npos) continue;
        Case c{e.path().string(), name.substr(0, dot), {}};
        std::ifstream facts(c.dir + "/facts.txt");
        for (std::string key; facts >> key;) facts >> c.facts[key];
        cases.push_back(c);
    }
    std::sort(cases.begin(), cases.end(), [](const Case& a, const Case& b) { return a.dir < b.dir; });
    LOOM_CHECK(!cases.empty());

    // The CPU, or the GPU under LOOM_TEST_DEVICE=gpu (with the CPU behind it).
    loom::Device device = loom::Device::open(loom_test::test_device_name());
    std::fprintf(stderr, "device: %s\n", device.name().c_str());
    auto model = loom::GgufModel::load(gguf, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, device.backends());
    loom::LoomLuaBridge& bridge = session.bridge();
    bridge.load_script(kProbe);
    LOOM_CHECK(loom::audio::TranscriptionStream::supported(bridge));

    double audio_s = 0.0, stream_s = 0.0;
    std::vector<Run> alone;
    const Case* default_case = nullptr;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const Case& c = cases[ci];
        const std::vector<float> wav = read_f32(std::string(clips_dir) + "/" + c.clip + ".npy");
        const int32_t blank = static_cast<int32_t>(c.facts.at("blank"));
        const size_t chunk_len = static_cast<size_t>(c.facts.at("chunk_len"));
        const size_t step = chunk_len - static_cast<size_t>(c.facts.at("stride_left") + c.facts.at("stride_right"));
        if (chunk_len == 64000 && c.facts.at("stride_left") == 10560 && c.facts.at("stride_right") == 10560)
            default_case = &c;

        // 1. The graph over each of the pipeline's chunks; the merged prefix after each, for 3.
        std::vector<int64_t> steps_shape;
        const std::vector<float> steps = read_f32(c.dir + "/steps.npy", &steps_shape);
        const size_t n_steps = static_cast<size_t>(steps_shape[0]);
        std::vector<std::vector<int32_t>> prefix;  // the merged ids after chunk s
        std::vector<int32_t> merged;
        int32_t prev = blank;
        double rows_worst = 0.0;
        std::vector<double> clip(wav.begin(), wav.end());
        for (size_t s = 0; s < n_steps; ++s) {
            const double start = steps[4 * s], n = steps[4 * s + 1], first = steps[4 * s + 2], end = steps[4 * s + 3];
            LOOM_CHECK(start == static_cast<double>(s * step));
            const std::vector<float> want = read_f32(c.dir + "/logits_" + std::to_string(s) + ".npy");
            const auto got = std::get<std::vector<double>>(bridge.call(
                "probe_chunk", {{"waveform", clip}, {"s", start}, {"n", n}, {"first", first}, {"end", end}}));
            LOOM_CHECK(got.size() == want.size() + 1 && got[0] >= end);
            if (got.size() != want.size() + 1) break;
            float absmax = 0.0f;
            for (float v : want) absmax = std::max(absmax, std::fabs(v));
            for (size_t i = 0; i < want.size(); ++i)
                rows_worst = std::max(rows_worst, std::fabs(got[i + 1] - want[i]) / absmax);
            const size_t width = end > first ? want.size() / static_cast<size_t>(end - first) : 1;
            merge(argmax_rows(want, width), blank, prev, merged);
            prefix.push_back(merged);
        }
        LOOM_CHECK(rows_worst <= 1e-3);

        // 2. The ids, at every cut tried.
        const std::vector<int32_t> want = as_ids(read_f32(c.dir + "/ids.npy"));
        LOOM_CHECK(!want.empty());
        LOOM_CHECK(merged == want);
        loom::audio::TranscriptionStream stream(bridge, *model, stream_inputs(c));
        const Run base = streamed(stream, wav, kRate / 10);
        stream_s += base.seconds;
        audio_s += static_cast<double>(wav.size()) / kRate;
        LOOM_CHECK(base.ids == want);
        std::vector<size_t> cuts = {wav.size()};
        if (ci == 0) cuts.insert(cuts.end(), {kRate / 100, chunk_len, kRate / 2});
        size_t same = base.ids == want;
        for (size_t piece : cuts) {
            stream.reset();
            const Run r = streamed(stream, wav, piece);
            same += r.ids == want;
            LOOM_CHECK(r.ids == want);
        }

        // 3. Each chunk as it runs: 10 ms pushes, so a chunk is seen as it lands.
        size_t seen_steps = 0;
        {
            stream.reset();
            double seen = state_number(stream, "next_start");
            const size_t piece = kRate / 100;
            for (size_t at = 0; at < wav.size(); at += piece) {
                stream.push(wav.data() + at, std::min(piece, wav.size() - at));
                const double next = state_number(stream, "next_start");
                if (next != seen) {
                    seen = next;
                    LOOM_CHECK(seen_steps < n_steps);
                    if (seen_steps >= n_steps) break;
                    LOOM_CHECK(next == static_cast<double>((seen_steps + 1) * step));
                    LOOM_CHECK(stream.tokens() == prefix[seen_steps]);
                    ++seen_steps;
                }
            }
            stream.finish();
            LOOM_CHECK(stream.tokens() == want);
            // Only the last chunk is left for `finish`.
            LOOM_CHECK(seen_steps + 1 == n_steps);
        }

        // 4. Text before the end, never revised: a clip of more than two chunks.
        if (n_steps > 2) LOOM_CHECK(base.updates_before_finish > 0);
        LOOM_CHECK(!base.revised);
        alone.push_back(base);
        std::ifstream text_in(c.dir + "/text.txt");
        const std::string pipe_text((std::istreambuf_iterator<char>(text_in)), std::istreambuf_iterator<char>());
        std::printf("%-28s %6.2f s  %3zu chunks  rows max|d|/absmax %.2e  ids %zu (%s)  %zu/%zu cuts same  "
                    "%zu/%zu chunks seen  %zu updates  %.2fx real time\n  \"%s\"\n  pipeline \"%s\"\n",
                    std::filesystem::path(c.dir).filename().string().c_str(),
                    static_cast<double>(wav.size()) / kRate, n_steps, rows_worst, want.size(),
                    base.ids == want ? "the pipeline's" : "DIFFERENT", same, cuts.size() + 1, seen_steps, n_steps,
                    base.updates_before_finish, static_cast<double>(wav.size()) / kRate / base.seconds,
                    base.text.c_str(), pipe_text.c_str());
    }
    std::printf("streamed %.1f s of audio in 100 ms pieces in %.1f s: %.2fx real time, RTF %.3f\n", audio_s,
                stream_s, audio_s / stream_s, stream_s / audio_s);

    const Case& first = cases.front();
    const std::vector<float> first_wav = read_f32(std::string(clips_dir) + "/" + first.clip + ".npy");

    // 5. Sizes the pipeline refuses or the graph cannot take; and the default sizes.
    {
        using Inputs = loom::audio::TranscriptionStream::Inputs;
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, Inputs{{"chunk_seconds", 2.0},
                                                                                   {"stride_seconds", 1.0}}),
                          loom::Error);
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, Inputs{{"chunk_seconds", 31.0}}),
                          loom::Error);
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, Inputs{{"chunk_seconds", 4.0},
                                                                                   {"stride_seconds", 0.04}}),
                          loom::Error);
        LOOM_CHECK(default_case != nullptr);
        if (default_case) {
            const std::vector<float> wav = read_f32(std::string(clips_dir) + "/" + default_case->clip + ".npy");
            loom::audio::TranscriptionStream s(bridge, *model);
            const Run r = streamed(s, wav, kRate / 10);
            LOOM_CHECK(r.ids == as_ids(read_f32(default_case->dir + "/ids.npy")));
        }
    }

    // 6. Two streams and a one-pass call between them: the first case and the last.
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

    // 7. The file exported before the door: the one pass unchanged, the door refused.
    if (const char* old_gguf = loom_test::fixture_env("LOOM_CTC_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old_gguf, device.backends().primary);
        LOOM_CHECK(old_model != nullptr);
        loom::Session old_session(*old_model, device.backends());
        LOOM_CHECK(loom::audio::transcribe(old_session.bridge(), *old_model, first_wav, {}).text ==
                   loom::audio::transcribe(bridge, *model, first_wav, {}).text);
        LOOM_CHECK(!loom::audio::TranscriptionStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(old_session.bridge(), *old_model), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
