// Quantized-weight proof of concept: same real Qwen3-0.6B-Base checkpoint/prompt/reference as
// test_e2e_qwen3.cpp, but the GGUF's matmul weights (attn_q/k/v/output, ffn_gate/up/down, and
// token_embd.weight -- also the tied-embeddings logits projection) have been quantized to Q8_0 by
// tools/quantize/quantize_gguf_q8_0.py. Exercises the BACKLOG.md "quantized weight support" milestone's
// central claim: GgufModel::load, op_mul_mat, and op_get_rows (src/core/gguf_model.cpp,
// src/ops/primitives_basic.cpp) need *no* changes at all to run a genuinely quantized real model --
// ggml's own CPU backend already implements the weights-quantized/activations-F32 dot product these
// primitives are thin wraps around. See BACKLOG.md for how the quantizer selects which tensors to
// quantize (topology-driven: every MUL_MAT node's weight input, not tensor-name pattern matching).
//
// Same "not generated at ctest time, skip cleanly if the fixture isn't prepared" pattern as
// test_e2e_qwen3.cpp/test_e2e_qwen3_generic.cpp. Reuses test_e2e_qwen3.cpp's own reference fixture (same
// weights before quantization, same prompt/seed => same expected logits, just compared at a looser,
// quantization-appropriate tolerance) rather than generating a second one.

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
    // LOOM_QWEN3_Q8_0_DIR must contain qwen3_q8_0.gguf, e.g. produced via:
    //   python3 tools/quantize/quantize_gguf_q8_0.py $LOOM_QWEN3_DIR/qwen3.gguf $DIR/qwen3_q8_0.gguf
    const char* q8_dir_env = loom_test::fixture_env("LOOM_QWEN3_Q8_0_DIR");
    const std::string q8_dir = q8_dir_env != nullptr ? q8_dir_env : "/tmp/qwen3_q8_0";
    const std::string gguf_path = q8_dir + "/qwen3_q8_0.gguf";

    // Reference fixture is shared with test_e2e_qwen3.cpp -- quantization happens after conversion, so
    // the pre-quantization weights (and thus the expected logits) are identical.
    const char* ref_dir_env = loom_test::fixture_env("LOOM_QWEN3_DIR");
    const std::string ref_dir = (ref_dir_env != nullptr ? std::string(ref_dir_env) : "/tmp/qwen3_model") + "/ref";

    if (!path_exists(gguf_path) || !path_exists(ref_dir)) {
        std::fprintf(stderr,
                      "skipping: quantized Qwen3 fixture not found at '%s' or reference not found at '%s' "
                      "(set LOOM_QWEN3_Q8_0_DIR / LOOM_QWEN3_DIR or see tools/quantize/ and "
                      "tools/convert_qwen3/ to produce them)\n",
                      gguf_path.c_str(), ref_dir.c_str());
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
    // Not LOOM_CHECK'd: unlike test_e2e_qwen3(_generic).cpp, Q8_0 quantization is real, applied lossy
    // compression (not a bit-exact reformat), so the argmax token sequence isn't guaranteed to survive --
    // logged instead, and the real check below is the logit-tolerance comparison.
    if (tokens != expected_tokens) {
        std::fprintf(stderr, "note: quantized generation diverged from the F32 reference's argmax tokens "
                              "(expected vs actual printed below) -- checking logit closeness instead\n");
    }

    for (uint32_t step = 0; step < kNNew; ++step) {
        const std::vector<float> expected = read_f32_binary(ref_dir + "/expected_logits_step" + std::to_string(step) + ".bin");
        LOOM_CHECK(expected.size() == actual_logits[step].size());

        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < expected.size() && i < actual_logits[step].size(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::fabs(expected[i] - actual_logits[step][i]));
        }
        std::fprintf(stderr, "step %u: max abs logit diff = %f (token: expected %d, got %d)\n", step,
                     static_cast<double>(max_abs_diff), expected_tokens[step], tokens[step]);
        // Tolerance measured empirically (see BACKLOG.md), not guessed upfront: real max abs diff across
        // all 4 steps ranged 0.45-0.79 (Q8_0's ~8-bit per-block quantization of every matmul weight
        // compounding through 28 layers), so this is deliberately much looser than test_e2e_qwen3.cpp's
        // 5e-2 (that test's own GGUF is unquantized) -- roughly 2x the observed max, tight enough to
        // still catch a genuine correctness bug, not just "doesn't crash."
        LOOM_CHECK(max_abs_diff < 1.5f);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
