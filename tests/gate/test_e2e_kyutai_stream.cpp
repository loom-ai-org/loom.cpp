// Kyutai STT live (ADR-108): the Mimi encoder a few frames per call with its streaming state carried, the LM
// a step per frame, through `TranscriptionStream` -- the stream ends on what one call over the clip decodes.
//
//   1. Tensor oracle: `mimi_encode_step` run over the clip in uneven pieces (1, 3, 7, 2, 16 and 5 frames,
//      the state kept on the device between calls) against `mimi_encode` once over the whole clip. The
//      step is the one-call encoder to f32 rounding, past the 250-position window's wrap. On the same rows,
//      `rvq_encode` (the RVQ in one graph, the argmax on the device) draws every code `infer`'s chain of
//      a stage per call draws.
//   2. The ids after `finish` are `infer`'s ids exactly, and the text `transcribe`'s, at 100 ms pieces and
//      the whole clip on every clip; at 20 ms, one frame and 0.5 s on the first.
//   3. Text arrives before the end and is never revised: each LM step's id is final.
//   4. Two streams pushed in alternation, with a one-pass `transcribe` between their pushes, each end where
//      they end alone -- the LM's ring KV cache parked and restored, past its 750-step wrap when a clip is
//      longer than 60 s.
//   5. With LOOM_KYUTAI_STREAM_OLD_GGUF, the file exported before the stream functions: the same ids in one
//      call, and the live door refuses it.
//
// LOOM_KYUTAI_STREAM_GGUF: kyutai-stt-1b-en-fr exported with the stream functions.
// LOOM_KYUTAI_STREAM_CLIPS_DIR: `*.npy`, mono F32 at 24 kHz -- real speech, one longer than 60 s sorted last.

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
constexpr size_t kFrame = 1920;
// The step against one call: the same rows summed in another order (f64 puts them 1e-15 apart).
constexpr double kRowTolerance = 1e-4;
// The encoder transformer's window (the checkpoint's `context`): the keys `mimi_encode_step` carries.
constexpr double kMimiContext = 250;

// `mimi_encode_step`'s state count, off its topology: the driver's own constant is a local of its chunk.
double step_states(const loom::GgufModel& model) {
    const std::string& json = model.topology_json("mimi_encode_step");
    const std::regex state("\"state_([0-9]+)\"");
    int most = 0;
    for (auto it = std::sregex_iterator(json.begin(), json.end(), state); it != std::sregex_iterator(); ++it) {
        most = std::max(most, std::stoi((*it)[1].str()));
    }
    return most;
}

const char* const kProbe = R"(
function probe_encoder(inputs)
    local n = math.floor(#inputs.waveform / 1920)
    return loom.run_subgraph('mimi_encode', {n_codes = n, n_past = 0},
                             {waveform = inputs.waveform, positions = loom.range(0, 2 * n)})
end

-- The split RVQ over `mimi_encode_step`'s rows as `infer` runs it, a stage per call: frame-major codes.
local function chain_codes(n)
    local CARD, NQ = 2048, 32
    local zeros = {}
    for i = 1, n do zeros[i] = 0 end
    local drawn = {}
    loom.run_subgraph_and_retain('rvq_project_semantic', {n_codes = n, n_past = 0}, {rows = {from = 'mimi_encode_step'}})
    local first = loom.range(0, CARD)
    loom.run_subgraph_and_retain('rvq_step', {n_codes = n, n_past = 0},
        {rows = {from = 'rvq_project_semantic'}, prev_ids = zeros, prev_codebook = first, next_codebook = first,
         subtract = {0.0}})
    drawn[1] = loom.argmax_rows('rvq_step')
    loom.run_subgraph_and_retain('rvq_project_acoustic', {n_codes = n, n_past = 0}, {rows = {from = 'mimi_encode_step'}})
    local rows, prev, sub = {from = 'rvq_project_acoustic'}, zeros, 0.0
    for q = 1, NQ - 1 do
        loom.run_subgraph_and_retain('rvq_step', {n_codes = n, n_past = 0},
            {rows = rows, prev_ids = prev, prev_codebook = loom.range(math.max(q - 1, 1) * CARD, CARD),
             next_codebook = loom.range(q * CARD, CARD), subtract = {sub}})
        prev = loom.argmax_rows('rvq_step')
        drawn[q + 1] = prev
        rows, sub = {from = 'rvq_step', index = 2}, 1.0
    end
    local codes = {}
    for f = 0, n - 1 do
        for q = 1, NQ do codes[f * NQ + q] = drawn[q][f + 1] end
    end
    return codes
end

-- `mimi_encode_step` the way the driver runs it, over `pieces` frames per call in turn; every call's rows,
-- then two numbers: the codes `rvq_encode` drew from those rows, and how many of them differ from the chain's.
function probe_step(inputs)
    local wave, pieces = inputs.waveform, inputs.pieces
    local MIMI_CONTEXT, MIMI_STATES = inputs.context, inputs.n_states
    local n_all = math.floor(#wave / 1920)
    local rows, f, i = {}, 0, 0
    local n_codes, n_diff = 0, 0
    while f < n_all do
        local n = math.min(pieces[i % #pieces + 1], n_all - f)
        i = i + 1
        local w = {}
        for s = 1, n * 1920 do w[s] = wave[f * 1920 + s] end
        local args = {waveform = w, positions = loom.range(2 * f, 2 * n),
                      key_positions = loom.range(2 * f - MIMI_CONTEXT, MIMI_CONTEXT + 2 * n),
                      first = {f == 0 and 1.0 or 0.0}}
        local shift = 1
        if f == 0 then
            loom.run_subgraph_and_retain('mimi_encode_init', {n_tokens = 0, n_past = 0}, {zero = {0}})
            loom.keep_state('probe', 'mimi_encode_init')
            shift = 0
        end
        for k = 1, MIMI_STATES do args['state_' .. k] = {state = 'probe', index = k + shift} end
        loom.run_subgraph_and_retain('mimi_encode_step', {n_codes = n, n_tokens = 0, n_past = 0}, args)
        loom.keep_state('probe', 'mimi_encode_step')
        local out = loom.get_output('mimi_encode_step', 1)
        for s = 1, #out do rows[#rows + 1] = out[s] end
        local chain = chain_codes(n)
        loom.run_subgraph_and_retain('rvq_encode', {n_codes = n, n_past = 0}, {rows = {from = 'mimi_encode_step'}})
        local fused = loom.get_output('rvq_encode', 1)
        for c = 1, #chain do
            if chain[c] ~= fused[c] then n_diff = n_diff + 1 end
        end
        n_codes = n_codes + #chain
        f = f + n
    end
    rows[#rows + 1] = n_codes
    rows[#rows + 1] = n_diff
    return rows
end
)";

std::vector<int32_t> one_pass_ids(loom::LoomLuaBridge& bridge, const std::vector<float>& wav) {
    const auto got = std::get<std::vector<double>>(bridge.call(
        "infer", {{"waveform", std::vector<double>(wav.begin(), wav.end())},
                  {"length", std::vector<double>{static_cast<double>(wav.size())}}}));
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
    const char* gguf = loom_test::fixture_env("LOOM_KYUTAI_STREAM_GGUF");
    const char* dir = loom_test::fixture_env("LOOM_KYUTAI_STREAM_CLIPS_DIR");
    if (!gguf || !dir) {
        std::fprintf(stderr, "SKIP: set LOOM_KYUTAI_STREAM_GGUF and LOOM_KYUTAI_STREAM_CLIPS_DIR\n");
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
    LOOM_CHECK(n_states > 17);  // the SEANet convs, 8 layers' keys and values, the downsample

    double audio_s = 0.0, stream_s = 0.0;
    std::vector<Run> alone;
    for (size_t c = 0; c < clips.size(); ++c) {
        std::vector<int64_t> shape;
        const std::vector<float> wav = loom_test::read_npy_f32(clips[c], shape);
        const std::string name = std::filesystem::path(clips[c]).stem().string();

        // 1. The rows, on the clip's whole frames.
        std::vector<double> whole(wav.begin(), wav.begin() + wav.size() / kFrame * kFrame);
        const auto one = std::get<std::vector<double>>(bridge.call("probe_encoder", {{"waveform", whole}}));
        auto step = std::get<std::vector<double>>(bridge.call(
            "probe_step", {{"waveform", whole}, {"pieces", std::vector<double>{1, 3, 7, 2, 16, 5}},
                           {"context", kMimiContext}, {"n_states", n_states}}));
        // The last two numbers: the codes drawn, and how many the one graph drew differently from the chain.
        const double code_diff = step.back();
        step.pop_back();
        const double n_codes = step.back();
        step.pop_back();
        LOOM_CHECK(n_codes == static_cast<double>(whole.size() / kFrame * 32));
        LOOM_CHECK(code_diff == 0);
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
        const Run base = streamed(s, wav, 2400);
        stream_s += base.seconds;
        audio_s += wav.size() / 24000.0;
        LOOM_CHECK(base.ids == want);
        LOOM_CHECK(base.text == want_text);
        std::vector<size_t> cuts = {wav.size()};
        if (c == 0) cuts.insert(cuts.end(), {size_t(480), kFrame, size_t(12000)});
        size_t same = base.ids == want;
        for (size_t piece : cuts) {
            s.reset();
            const Run r = streamed(s, wav, piece);
            same += r.ids == want;
            LOOM_CHECK(r.ids == want);
        }

        // 3. Text before the end, never revised.
        if (wav.size() > 24000 * 2) LOOM_CHECK(base.updates_before_finish > 0);
        LOOM_CHECK(!base.revised);
        alone.push_back(base);
        std::printf("%-20s %6.2f s  rows max|d| %.3g  codes %.0f differ of %.0f  ids %zu  %zu/%zu cuts same  %zu updates  %.2fx real time\n"
                    "  \"%s\"\n",
                    name.c_str(), wav.size() / 24000.0, d, code_diff, n_codes, want.size(), same, cuts.size() + 1,
                    base.updates_before_finish, wav.size() / 24000.0 / base.seconds, base.text.c_str());
    }
    std::printf("streamed %.1f s of audio in 100 ms pieces in %.1f s: %.2fx real time, RTF %.3f\n", audio_s,
                stream_s, audio_s / stream_s, stream_s / audio_s);

    // 4. Two streams and a one-pass call between them: the first clip and the last (the longest).
    {
        std::vector<int64_t> shape;
        const std::vector<float> a_wav = loom_test::read_npy_f32(clips.front(), shape);
        const std::vector<float> b_wav = loom_test::read_npy_f32(clips.back(), shape);
        loom::audio::TranscriptionStream a(bridge, *model), b(bridge, *model);
        const size_t n = std::max(a_wav.size(), b_wav.size());
        for (size_t at = 0; at < n; at += 2400) {
            if (at < a_wav.size()) a.push(a_wav.data() + at, std::min<size_t>(2400, a_wav.size() - at));
            if (at == 2400 * 20) loom::audio::transcribe(bridge, *model, a_wav, {});
            if (at < b_wav.size()) b.push(b_wav.data() + at, std::min<size_t>(2400, b_wav.size() - at));
        }
        a.finish();
        b.finish();
        LOOM_CHECK(a.tokens() == alone.front().ids && a.text() == alone.front().text);
        LOOM_CHECK(b.tokens() == alone.back().ids && b.text() == alone.back().text);
    }

    // 5. The file exported before the door.
    if (const char* old_gguf = loom_test::fixture_env("LOOM_KYUTAI_STREAM_OLD_GGUF")) {
        auto old_model = loom::GgufModel::load(old_gguf, backend.get());
        LOOM_CHECK(old_model != nullptr);
        loom::Session old_session(*old_model, backend.get());
        std::vector<int64_t> shape;
        const std::vector<float> wav = loom_test::read_npy_f32(clips.front(), shape);
        LOOM_CHECK(one_pass_ids(old_session.bridge(), wav) == alone.front().ids);
        LOOM_CHECK(!loom::audio::TranscriptionStream::supported(old_session.bridge()));
        LOOM_CHECK_THROWS(loom::audio::TranscriptionStream(old_session.bridge(), *old_model), loom::LoadError);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
