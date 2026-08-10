// Generic-converter proof of concept: loads a toy LLM GGUF whose topology JSON was NOT hand-written
// (unlike tools/fixture_gen/toy_llm_common.py's build_topology()) but auto-derived by walking a real
// torch.export() ATen graph of tools/convert_generic/toy_llm_module.py's ToyLLM through
// tools/convert_generic/aten_to_loom.py's generic op-mapping converter -- see BACKLOG.md's "generic
// converter" section for the full design.
//
// Both topologies are built from the exact same weights (tools/fixture_gen/toy_llm_common.py's
// generate_weights(), same seed), so the expected logits are byte-identical to test_e2e_toy_llm.cpp's own
// reference fixture -- this test intentionally reuses that same LOOM_E2E_REF_DIR rather than generating a
// second one, since a mismatch here would mean the *converter*, not the weights, disagrees with the
// known-good hand-written topology.
//
// Requires a torch environment to produce the GGUF (see tools/convert_generic/make_toy_llm_gguf_generic.py),
// which doesn't belong in a default ctest run any more than the real-model tests do -- looks for
// LOOM_TOY_LLM_GENERIC_DIR (default /tmp/toy_llm_generic) and skips cleanly (exit code 77) if the GGUF
// isn't there.

#include "test_util.h"
#include "fixtures.h"

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
    // LOOM_TOY_LLM_GENERIC_DIR must contain toy_llm_generic.gguf, e.g. produced via:
    //   python3 tools/convert_generic/make_toy_llm_gguf_generic.py $DIR/toy_llm_generic.gguf
    const char* dir_env = loom_test::fixture_env("LOOM_TOY_LLM_GENERIC_DIR");
    const std::string dir = dir_env != nullptr ? dir_env : "/tmp/toy_llm_generic";
    const std::string gguf_path = dir + "/toy_llm_generic.gguf";
    if (!path_exists(gguf_path)) {
        std::fprintf(stderr,
                      "skipping: auto-converted toy LLM fixture not found at '%s' (set "
                      "LOOM_TOY_LLM_GENERIC_DIR or see tools/convert_generic/ to produce one)\n",
                      gguf_path.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    const std::string ref_dir = LOOM_TEST_REF_DIR; // shared with test_e2e_toy_llm.cpp -- same weights

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    const std::vector<int32_t> prompt = {1, 2, 3};
    constexpr uint32_t kNNew = 4;

    std::vector<std::vector<float>> actual_logits;
    loom::GenerationConfig cfg;
    cfg.max_new_tokens = kNNew;
    cfg.n_ctx_max = 32;
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
            if (diff > 1e-3f) close = false;
        }
        LOOM_CHECK(close);
        if (!close) {
            std::fprintf(stderr, "step %u: max abs logit diff = %f\n", step, static_cast<double>(max_abs_diff));
        }
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
