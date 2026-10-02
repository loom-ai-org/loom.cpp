// Validates the Kyutai STT export (loom-exporter's `kyutai_stt_export.py`) end to end against Kyutai's
// own `moshi` (`scripts/kyutai_stt_reference.py`): the Mimi encoder in chunks, the split RVQ, the LM
// over a RING KV cache (ADR-066), greedy, and the transcript's ids.
//
// Four arms, each a tensor or an id sequence, none a transcript alone:
//   1. CODES: the driver's Mimi codes (`return_codes`) against moshi's streamed ones, every frame.
//      Measured 156/156 on jfk.wav (and 2474/2474 on a 196 s clip, chunked).
//   2. LOGITS, teacher-forced: moshi's own inputs, one step at a time, every step's text logits
//      (measured max|d| 5.2e-05), then the same with the first 40 steps in ONE call (2.7e-05) -- the
//      arm that found a two-sided broadcast one-token steps could not show (Retro-070's lesson).
//   3. IDS: the driver's transcript ids against moshi's, id for id.
//   4. SABOTAGE: the codes of the same clip shifted by one sample must differ.
//
// Fixtures:
//   LOOM_KYUTAI_STT_GGUF     kyutai_stt.gguf  -- `loom-export ~/Dev/models/kyutai-stt-1b-en-fr`
//   LOOM_KYUTAI_STT_REF_DIR  kyutai_stt_ref/  -- from scripts/kyutai_stt_reference.py

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int meta_int(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\": ";
    const size_t at = json.find(needle);
    if (at == std::string::npos) throw std::runtime_error("kyutai_stt_ref/meta.json has no \"" + key + "\"");
    return std::stoi(json.substr(at + needle.size()));
}

constexpr int kNq = 32, kVocab = 8000;

// moshi's inputs, step by step: (initial, initial), then (step k-1's text id, frame k-1's codes).
const char* kTeacherForced = R"(
    local steps = #inputs.tokens / 33
    local out = {}
    local function row(k) local t = {} for i = 1, 33 do t[i] = inputs.tokens[k * 33 + i] end return t end
    local p = inputs.prefill
    if p > 1 then
        local t = {}
        for k = 0, p - 1 do local r = row(k) for i = 1, 33 do t[#t + 1] = r[i] end end
        loom.run_subgraph_and_retain('lm', {n_tokens = p, n_past = 0},
            {tokens = t, position_ids = loom.range(0, p), attention_mask = loom.causal_mask(p, 0)})
        local l = loom.get_output('lm', 1) for i = 1, #l do out[#out + 1] = l[i] end
    else p = 0 end
    for k = p, steps - 1 do
        loom.run_subgraph_and_retain('lm', {n_tokens = 1, n_past = k},
            {tokens = row(k), position_ids = {k}, attention_mask = loom.causal_mask(1, math.min(k, 749))})
        local l = loom.get_output('lm', 1) for i = 1, #l do out[#out + 1] = l[i] end
    end
    return out
)";

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_KYUTAI_STT_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_KYUTAI_STT_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_KYUTAI_STT_GGUF and LOOM_KYUTAI_STT_REF_DIR "
                              "(see scripts/kyutai_stt_reference.py)\n");
        return 77;
    }
    const std::string ref_dir = ref_env;
    std::ifstream meta_file(ref_dir + "/meta.json");
    LOOM_CHECK(meta_file.good());
    std::stringstream meta_buf;
    meta_buf << meta_file.rdbuf();
    const int pad_right = meta_int(meta_buf.str(), "pad_right");

    std::vector<int64_t> shape;
    const auto pcm = loom_test::read_npy_f32(ref_dir + "/pcm.npy", shape);
    const auto codes = loom_test::read_npy_f32(ref_dir + "/codes.npy", shape);     // [32, n]
    const size_t n = static_cast<size_t>(shape[1]);
    const auto text_ids = loom_test::read_npy_f32(ref_dir + "/text_ids.npy", shape);
    const auto logits = loom_test::read_npy_f32(ref_dir + "/step_logits.npy", shape);  // [n + 1, 8000]
    LOOM_CHECK(text_ids.size() == n && logits.size() == (n + 1) * kVocab);

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, device.backends());

    // The clip as handed to the driver: the reference's signal without the padding the driver adds.
    const std::vector<double> wave(pcm.begin(), pcm.end() - pad_right);
    const double real = static_cast<double>(wave.size());

    // --- 1. Codes. ---
    {
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"waveform", wave}, {"length", std::vector<double>{real}}, {"return_codes", 1.0}}));
        LOOM_CHECK(got.size() == n * kNq);
        size_t same = 0;
        for (size_t f = 0; f < n && (f + 1) * kNq <= got.size(); ++f) {
            bool all = true;
            for (int q = 0; q < kNq; ++q) all = all && got[f * kNq + q] == codes[q * n + f];
            same += all;
        }
        std::fprintf(stderr, "codes: %zu/%zu frames identical\n", same, n);
        LOOM_CHECK(same == n);
    }

    // --- 2. Logits, teacher-forced. ---
    {
        size_t best = 0;
        for (size_t i = 1; i < kVocab; ++i) best = logits[i] > logits[best] ? i : best;
        std::vector<double> tokens((n + 1) * 33, 0.0);
        tokens[0] = 8000;
        for (int q = 0; q < kNq; ++q) tokens[1 + q] = 2048;
        for (size_t k = 1; k <= n; ++k) {
            tokens[k * 33] = k == 1 ? static_cast<double>(best) : text_ids[k - 2];
            for (int q = 0; q < kNq; ++q) tokens[k * 33 + 1 + q] = codes[q * n + (k - 1)];
        }
        session.bridge().load_script(std::string("function probe(inputs)\n") + kTeacherForced + "\nend\n");
        for (double prefill : {1.0, 40.0}) {
            const auto got = std::get<std::vector<double>>(session.bridge().call("probe", {
                {"tokens", tokens}, {"prefill", prefill}}));
            const size_t first = prefill > 1 ? static_cast<size_t>(prefill) - 1 : 0;
            LOOM_CHECK(got.size() == (n + 1 - first) * kVocab);
            double max_abs = 0.0;
            for (size_t i = 0; i < got.size() && i < (n + 1 - first) * kVocab; ++i) {
                max_abs = std::max(max_abs, std::fabs(got[i] - logits[first * kVocab + i]));
            }
            std::fprintf(stderr, "teacher-forced, first %g step(s) in one call: max|d| %g\n", prefill, max_abs);
            // ~10x the measured 5.2e-05, for other ISAs' accumulation order.
            LOOM_CHECK(max_abs < 5e-4);
        }
    }

    // --- 3. Ids. ---
    {
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"waveform", wave}, {"length", std::vector<double>{real}}}));
        std::vector<double> want;
        for (float id : text_ids) {
            if (id != 0.0f && id != 3.0f) want.push_back(id);
        }
        std::fprintf(stderr, "ids: loom=%zu reference=%zu\n", got.size(), want.size());
        LOOM_CHECK(got == want);
    }

    // --- 4. Sabotage: the clip one sample late. ---
    {
        std::vector<double> late(1, 0.0);
        late.insert(late.end(), wave.begin(), wave.end() - 1);
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"waveform", late}, {"length", std::vector<double>{real}}, {"return_codes", 1.0}}));
        size_t differ = 0;
        for (size_t i = 0; i < std::min(got.size(), n * kNq); ++i) differ += got[i] != codes[(i % kNq) * n + i / kNq];
        std::fprintf(stderr, "sabotage (one sample late): %zu codes differ\n", differ);
        LOOM_CHECK(differ > 0);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
