// Canary live (ADR-124): NeMo's streaming AED loop through `TranscriptionStream`, wait-k and AlignAtt. Each step
// the encoder over up to `left + chunk + right` seconds, the decoder shown the left context and the chunk (all of
// it on the last chunk), its KV cache carried from chunk to chunk; wait-k keeps one token a chunk, AlignAtt keeps
// tokens while layer -2's cross-attention peaks far enough from the end of what the decoder may see.
//
// The oracle is NeMo's `speech_to_text_aed_streaming_infer.py` loop at batch 1 (`loom-exporter/fixture_gen/
// reference_stream_canary.py`), one directory per clip, policy, sizes and languages:
// `<clip>.<policy>.<left>-<chunk>-<right>.<src>-<tgt>/`.
//
//   1. Tensor oracle: on the jfk cases, the encoder over each of NeMo's buffers, the rows the decoder sees
//      against NeMo's to 1e-3 of their absmax; on every case, the first decoding step's first forward (the
//      prompt over that step's rows): its last row's logits to 1e-3 and layer -2's head-mean cross-attention,
//      AlignAtt's input, to 1e-3 of its absmax, and its token NeMo's.
//   2. One encoder frame per push: after every step the kept context (prompt and tokens) is as long as NeMo's,
//      so every policy decision lands where NeMo's did; the stream ends on NeMo's ids. The whole clip in one push
//      too, and on the first case 10 ms, one chunk and 0.5 s.
//   3. Text arrives before the end where NeMo's does, and is never revised.
//   4. Refused: no right context, a chunk under a frame, a buffer longer than the encoder takes, a policy that
//      is neither.
//   5. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where they
//      end alone.
//   6. With LOOM_CANARY_STREAM_OLD_GGUF, the file exported before the stream functions: the same one-pass text,
//      and the live door refuses it.
//
// LOOM_CANARY_STREAM_GGUF: canary-1b-v2 exported with the stream functions. LOOM_CANARY_STREAM_CLIPS_DIR:
// `<clip>.npy`, mono F32 at 16 kHz. LOOM_CANARY_STREAM_REF_DIR: the reference.

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
constexpr int kLayers = 8;

// The encoder over clip samples [stop - total, stop) with `length` valid, its first `rows` rows returned (after
// the row count). And the first decode forward: cross-attention K/V over the first `k` rows, the prompt at
// position 0, the last row's logits then every row's layer -2 attention.
const char* const kProbe = R"(
function probe_encoder(inputs)
    local x, w = inputs.waveform, {}
    for i = 1, inputs.total do w[i] = x[inputs.stop - inputs.total + i] end
    loom.run_subgraph_and_retain('encoder', {n_samples = #w, n_past = 0}, {waveform = w, length = {inputs.length}})
    local shape = loom.output_shape('encoder', 1)
    local out, d, rows = loom.get_output('encoder', 1), shape[1], {shape[2]}
    for i = 1, inputs.rows * d do rows[#rows + 1] = out[i] end
    return rows
end
function probe_first(inputs)
    local k = inputs.k
    loom.run_subgraph_and_retain('cross_kv', {n_enc_frames = k, n_past = 0}, {xa = {from = 'encoder', row = 0, rows = k}})
    local p = inputs.prompt
    local args = {tokens = p, position_ids = loom.range(0, #p), attention_mask = loom.causal_mask(#p, 0)}
    for i = 0, inputs.layers - 1 do
        args['xk_' .. i] = {from = 'cross_kv', index = 2 * i + 1}
        args['xv_' .. i] = {from = 'cross_kv', index = 2 * i + 2}
    end
    loom.run_subgraph_and_retain('decoder', {n_tokens = #p, n_past = 0, n_enc_frames = k}, args)
    local logits, v = loom.get_output('decoder', 1), loom.output_shape('decoder', 1)[1]
    local out = {}
    for i = (#p - 1) * v + 1, #p * v do out[#out + 1] = logits[i] end
    local xatt = loom.get_output('decoder', 2)
    for i = 1, #xatt do out[#out + 1] = xatt[i] end
    return out
end
)";

struct Case {
    std::string dir, clip, policy, src, tgt;
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
    return {{"left_seconds", c.seconds[0]}, {"chunk_seconds", c.seconds[1]}, {"right_seconds", c.seconds[2]},
            {"policy", c.policy == "alignatt" ? 1.0 : 0.0},
            {"language", loom::audio::TranscriptionStream::language_input(model, c.src)},
            {"target_language", loom::audio::TranscriptionStream::language_input(model, c.tgt)}};
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
    const char* gguf = loom_test::fixture_env("LOOM_CANARY_STREAM_GGUF");
    const char* clips_dir = loom_test::fixture_env("LOOM_CANARY_STREAM_CLIPS_DIR");
    const char* ref_dir = loom_test::fixture_env("LOOM_CANARY_STREAM_REF_DIR");
    if (!gguf || !clips_dir || !ref_dir) {
        std::fprintf(stderr, "SKIP: set LOOM_CANARY_STREAM_GGUF, _CLIPS_DIR and _REF_DIR\n");
        return kSkipReturnCode;
    }
    std::vector<Case> cases;
    for (const auto& e : std::filesystem::directory_iterator(ref_dir)) {
        if (!e.is_directory()) continue;
        // `<clip>.<policy>.<left>-<chunk>-<right>.<src>-<tgt>`; the sizes may hold dots of their own.
        const std::string name = e.path().filename().string();
        const size_t a = name.find('.'), b = name.find('.', a + 1), z = name.rfind('.');
        const std::vector<std::string> sizes = split(name.substr(b + 1, z - b - 1), '-');
        const std::vector<std::string> langs = split(name.substr(z + 1), '-');
        LOOM_CHECK(sizes.size() == 3 && langs.size() == 2);
        if (sizes.size() != 3 || langs.size() != 2) continue;
        cases.push_back({e.path().string(), name.substr(0, a), name.substr(a + 1, b - a - 1), langs[0], langs[1],
                         {std::stod(sizes[0]), std::stod(sizes[1]), std::stod(sizes[2])}});
    }
    std::sort(cases.begin(), cases.end(), [](const Case& x, const Case& y) { return x.dir < y.dir; });
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
        double ctx[3];
        for (int k = 0; k < 3; ++k) ctx[k] = std::floor(c.seconds[k] * 100.0 / 8);
        const std::vector<float> steps = read_f32(c.dir + "/steps.npy");
        const size_t n_steps = steps.size() / 6;
        // A step's buffer ends at its right context's end, the last's at the clip's.
        auto stop_of = [&](size_t s) {
            return s + 1 == n_steps ? static_cast<double>(wav.size())
                                    : kFrameSamples * (ctx[1] + ctx[2] + ctx[1] * static_cast<double>(s));
        };
        auto encode = [&](size_t s, double rows) {
            return std::get<std::vector<double>>(bridge.call(
                "probe_encoder", {{"waveform", whole}, {"stop", stop_of(s)}, {"total", static_cast<double>(steps[6 * s])},
                                  {"length", static_cast<double>(steps[6 * s + 1])}, {"rows", rows}}));
        };

        // 1a. The encoder rows the decoder sees, step by step, on the jfk cases.
        double rows_worst = 0.0;
        if (c.clip == "jfk") {
            for (size_t s = 0; s < n_steps; ++s) {
                const double k = steps[6 * s + 2];
                const auto got = encode(s, k);
                const std::vector<float> want = read_f32(c.dir + "/rows_" + std::to_string(s) + ".npy");
                LOOM_CHECK(got.size() == want.size() + 1);
                if (got.size() != want.size() + 1) break;
                rows_worst = std::max(rows_worst, max_rel(got, 1, want));
            }
            LOOM_CHECK(rows_worst <= 1e-3);
        }

        // 1b. The first decoding step's first forward: logits and AlignAtt's attention.
        const std::vector<float> first = read_f32(c.dir + "/first.npy");
        const std::vector<float> prompt_f = read_f32(c.dir + "/prompt.npy");
        double logits_worst = 0.0, xatt_worst = 0.0;
        bool first_token = false;
        if (first[0] >= 0) {
            const size_t s = static_cast<size_t>(first[0]);
            const double k = steps[6 * s + 2];
            encode(s, 0);
            const auto got = std::get<std::vector<double>>(bridge.call(
                "probe_first", {{"k", k}, {"prompt", std::vector<double>(prompt_f.begin(), prompt_f.end())},
                                {"layers", static_cast<double>(kLayers)}}));
            const std::vector<float> logits = read_f32(c.dir + "/logits.npy");
            const std::vector<float> xatt = read_f32(c.dir + "/xatt.npy");
            LOOM_CHECK(got.size() == logits.size() + xatt.size());
            if (got.size() == logits.size() + xatt.size()) {
                logits_worst = max_rel(got, 0, logits);
                xatt_worst = max_rel(got, logits.size(), xatt);
                const auto best = std::max_element(got.begin(), got.begin() + static_cast<long>(logits.size()));
                first_token = static_cast<float>(best - got.begin()) == first[1];
            }
            LOOM_CHECK(logits_worst <= 1e-3 && xatt_worst <= 1e-3 && first_token);
        }

        // 2. One encoder frame per push: the kept context after each step, then the ids.
        const std::vector<int32_t> want = as_ids(read_f32(c.dir + "/ids.npy"));
        loom::audio::TranscriptionStream stream(bridge, *model, stream_inputs(*model, c));
        size_t steps_seen = 0, context_same = 0;
        auto context = [&]() {
            const loom::StreamState::Value* v = stream.state().value("seq");
            return v ? static_cast<double>(v->data.size()) : -1.0;
        };
        auto counter = [&]() {
            const loom::StreamState::Value* v = stream.state().value("steps");
            return v && !v->data.empty() ? static_cast<size_t>(v->data[0]) : 0;
        };
        Run base;
        {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t at = 0; at < wav.size(); at += kFrameSamples) {
                const loom::audio::TranscriptUpdate u =
                    stream.push(wav.data() + at, std::min(kFrameSamples, wav.size() - at));
                if (!u.delta.empty()) ++base.updates_before_finish;
                base.revised = base.revised || u.revised;
                while (steps_seen < counter()) {
                    // Only the latest step's context is readable; a push runs at most one step at these sizes.
                    if (steps_seen + 1 == counter()) context_same += context() == steps[6 * steps_seen + 5];
                    ++steps_seen;
                }
            }
            const loom::audio::TranscriptUpdate last = stream.finish();
            base.revised = base.revised || last.revised;
            if (counter() == n_steps) context_same += context() == steps[6 * (n_steps - 1) + 5];
            base.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            base.ids = stream.tokens();
            base.text = stream.text();
        }
        LOOM_CHECK(counter() == n_steps);
        LOOM_CHECK(context_same == n_steps);
        LOOM_CHECK(base.ids == want);
        stream_s += base.seconds;
        audio_s += static_cast<double>(wav.size()) / kRate;
        std::vector<size_t> cuts = {wav.size()};
        if (ci == 0) cuts.insert(cuts.end(), {kRate / 100, static_cast<size_t>(kFrameSamples * ctx[1]), kRate / 2});
        size_t same = base.ids == want;
        for (size_t piece : cuts) {
            stream.reset();
            const Run r = streamed(stream, wav, piece);
            same += r.ids == want;
            LOOM_CHECK(r.ids == want);
        }

        // 3. Text before the end wherever NeMo kept a token before its last step; never revised.
        if (n_steps > 1 && steps[6 * (n_steps - 2) + 5] > prompt_f.size()) LOOM_CHECK(base.updates_before_finish > 0);
        LOOM_CHECK(!base.revised);
        alone.push_back(base);
        std::ifstream text_in(c.dir + "/text.txt");
        const std::string nemo_text((std::istreambuf_iterator<char>(text_in)), std::istreambuf_iterator<char>());
        std::printf("%-44s %6.2f s  %3zu steps  rows %.2e  first logits %.2e xatt %.2e token %s  context %zu/%zu  "
                    "ids %zu (%s)  %zu/%zu cuts same  %zu updates  %.2fx real time\n  \"%s\"\n  NeMo \"%s\"\n",
                    std::filesystem::path(c.dir).filename().string().c_str(), static_cast<double>(wav.size()) / kRate,
                    n_steps, rows_worst, logits_worst, xatt_worst, first_token ? "NeMo's" : "-", context_same, n_steps,
                    want.size(), base.ids == want ? "NeMo's" : "DIFFERENT", same, cuts.size() + 1,
                    base.updates_before_finish, static_cast<double>(wav.size()) / kRate / base.seconds,
                    base.text.c_str(), nemo_text.c_str());
    }
    std::printf("streamed %.1f s of audio in 80 ms pieces in %.1f s: %.2fx real time, RTF %.3f\n", audio_s, stream_s,
                audio_s / stream_s, stream_s / audio_s);

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
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, with("left_seconds", 40.0)), loom::Error);
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(bridge, *model, with("policy", 2.0)), loom::Error);
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
            if (at == piece * 50) loom::audio::transcribe(bridge, *model, first_wav, {});
            if (at < b_wav.size()) b.push(b_wav.data() + at, std::min(piece, b_wav.size() - at));
        }
        a.finish();
        b.finish();
        LOOM_CHECK(a.tokens() == alone.front().ids && a.text() == alone.front().text);
        LOOM_CHECK(b.tokens() == alone.back().ids && b.text() == alone.back().text);
    }

    // 6. The file exported before the door: the one pass unchanged, the door refused.
    if (const char* old_gguf = loom_test::fixture_env("LOOM_CANARY_STREAM_OLD_GGUF")) {
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
