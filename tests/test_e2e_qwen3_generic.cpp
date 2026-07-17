// Generic-converter proof of concept, round 2: same idea as test_e2e_toy_llm_generic.cpp, but against a
// real checkpoint (Qwen3-0.6B-Base) instead of the toy fixture -- exercises genuine GQA (16 query / 8 KV
// heads), per-head QK-norm, and tied embeddings through the *same*, unmodified converter/op-mapping table
// (tools/convert_generic/aten_to_loom.py) the toy LLM POC used. See BACKLOG.md's "generic converter"
// section for how much of that table carried over unchanged.
//
// Reuses the real reference fixture already produced for test_e2e_qwen3.cpp (same weights, same prompt,
// same seed => byte-identical expected logits regardless of which conversion pipeline built the GGUF) --
// set LOOM_QWEN3_DIR to that directory for the reference, and LOOM_QWEN3_GENERIC_DIR for this test's own
// auto-converted GGUF. Skips cleanly (exit code 77) if either isn't prepared.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>
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
    // LOOM_QWEN3_GENERIC_DIR must contain qwen3_generic.gguf, e.g. produced via:
    //   python3 tools/convert_generic/make_qwen3_gguf_generic.py <hf_checkpoint_dir> $DIR/qwen3_generic.gguf
    const char* generic_dir_env = std::getenv("LOOM_QWEN3_GENERIC_DIR");
    const std::string generic_dir = generic_dir_env != nullptr ? generic_dir_env : "/tmp/qwen3_generic";
    const std::string gguf_path = generic_dir + "/qwen3_generic.gguf";

    // Reference fixture is shared with test_e2e_qwen3.cpp -- same weights, so byte-identical expected
    // logits regardless of which conversion pipeline built the GGUF being tested.
    const char* ref_dir_env = std::getenv("LOOM_QWEN3_DIR");
    const std::string ref_dir = (ref_dir_env != nullptr ? std::string(ref_dir_env) : "/tmp/qwen3_model") + "/ref";

    if (!path_exists(gguf_path) || !path_exists(ref_dir)) {
        std::fprintf(stderr,
                      "skipping: auto-converted Qwen3 fixture not found at '%s' or reference not found at "
                      "'%s' (set LOOM_QWEN3_GENERIC_DIR / LOOM_QWEN3_DIR or see tools/convert_generic/ and "
                      "tools/convert_qwen3/ to produce them)\n",
                      gguf_path.c_str(), ref_dir.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
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
            if (diff > 5e-2f) close = false; // same tolerance as test_e2e_qwen3.cpp -- 28-layer accumulation
        }
        LOOM_CHECK(close);
        if (!close) {
            std::fprintf(stderr, "step %u: max abs logit diff = %f\n", step, static_cast<double>(max_abs_diff));
        }
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
