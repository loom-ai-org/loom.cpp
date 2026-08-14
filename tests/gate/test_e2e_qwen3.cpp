// Real-model end-to-end test: loads a real Qwen3-0.6B-Base checkpoint (converted by
// tools/convert_qwen3/convert_qwen3.py into a GGUF) and runs autoregressive greedy generation via the
// existing Generator (unmodified -- the topology follows the same "tokens"/"positions"/"kq_mask" input
// convention as the toy LLM and GQA fixtures), comparing logits against
// tools/convert_qwen3/reference_forward_qwen3.py's independent numpy computation, plus a round-trip
// check of the real 151k-merge byte-level-BPE vocab (loom::BpeVocab) on real text.
//
// Same "not generated at ctest time, skip cleanly if the real checkpoint isn't prepared" pattern as
// test_e2e_conformer_ctc.cpp -- needs a real ~1.2GB safetensors download + a PyTorch environment.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::vector<float> read_f32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

} // namespace

int main() {
    // LOOM_QWEN3_DIR must contain qwen3.gguf + ref/{expected_logits_step*.bin,expected_generated_tokens.json},
    // e.g. produced via:
    //   python3 tools/convert_qwen3/convert_qwen3.py <hf_checkpoint_dir> $DIR/qwen3.gguf
    //   python3 tools/convert_qwen3/reference_forward_qwen3.py <hf_checkpoint_dir> $DIR/ref \
    //       --prompt 1 2 3 --n-new 4
    const char* dir_env = loom_test::fixture_env("LOOM_QWEN3_DIR");
    const std::string dir = dir_env != nullptr ? dir_env : "/tmp/qwen3_model";

    const std::string gguf_path = dir + "/qwen3.gguf";
    const std::string ref_dir = dir + "/ref";
    if (!path_exists(gguf_path) || !path_exists(ref_dir)) {
        std::fprintf(stderr,
                      "skipping: real Qwen3-0.6B-Base fixture not found at '%s' (set LOOM_QWEN3_DIR "
                      "or see tools/convert_qwen3/ to produce one)\n",
                      dir.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model->hparam_u32("n_head") == 16);
    LOOM_CHECK(model->hparam_u32("n_head_kv") == 8);

    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    const std::vector<int32_t> prompt = {1, 2, 3};
    constexpr uint32_t kNNew = 4;

    std::vector<std::vector<float>> actual_logits;
    loom::GenerationConfig cfg;
    cfg.max_new_tokens = kNNew;
    cfg.n_ctx_max = 64;
    cfg.on_token = [&](const std::vector<float>& row) { actual_logits.push_back(row); };

    loom::Generator gen(*model, topo, cfg, backend.get());
    std::vector<int32_t> tokens = gen.generate(prompt);

    LOOM_CHECK(tokens.size() == kNNew);
    LOOM_CHECK(actual_logits.size() == kNNew);

    nlohmann::json expected_tokens_json = nlohmann::json::parse(read_file(ref_dir + "/expected_generated_tokens.json"));
    const std::vector<int32_t> expected_tokens = expected_tokens_json.get<std::vector<int32_t>>();
    LOOM_CHECK(tokens == expected_tokens);

    for (uint32_t step = 0; step < kNNew; ++step) {
        const std::vector<float> expected = read_f32_binary(ref_dir + "/expected_logits_step" + std::to_string(step) + ".bin");
        LOOM_CHECK(expected.size() == actual_logits[step].size());

        bool close = true;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < expected.size() && i < actual_logits[step].size(); ++i) {
            const float diff = std::fabs(expected[i] - actual_logits[step][i]);
            max_abs_diff = std::max(max_abs_diff, diff);
            // A real 28-layer/1024-wide model accumulates more floating-point summation-order drift
            // (numpy BLAS vs ggml's own kernels) than the tiny toy/GQA fixtures -- a looser absolute
            // tolerance than those tests' 1e-3, still tight enough to catch a genuine correctness bug.
            if (diff > 5e-2f) close = false;
        }
        LOOM_CHECK(close);
        if (!close) {
            std::fprintf(stderr, "step %u: max abs logit diff = %f\n", step, static_cast<double>(max_abs_diff));
        }
    }

    // Real-vocab round-trip: encode/decode real English text through the actual 151k-merge BpeVocab
    // (not the small synthetic fixture test_bpe_vocab.cpp uses), proving the full pipeline (NFC ->
    // pretokenizer scanner -> byte-level map -> greedy merge against 151387 real merge rules) works
    // end to end against the genuine tokenizer.json-derived vocab, not just a hand-built one.
    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const std::string text = "The capital of France is Paris.";
    const auto ids = vocab->encode(text);
    LOOM_CHECK(!ids.empty());
    for (int32_t id : ids) {
        LOOM_CHECK(id >= 0 && static_cast<size_t>(id) < vocab->size());
    }
    const std::string round_tripped = vocab->decode(ids);
    LOOM_CHECK(round_tripped == text);
    std::fprintf(stderr, "encoded '%s' -> %zu tokens -> decoded '%s'\n", text.c_str(), ids.size(), round_tripped.c_str());

    LOOM_TEST_REPORT_AND_RETURN();
}
