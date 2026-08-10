// Numerical-correctness check that VITS's own REL_POS_ATTENTION_SHAW primitive family is directly
// reusable for SupertonicTTS v2's `MultiHeadRelativeAttention` (real source: components.py -- the same
// Shaw et al. lookup-table + rel_to_abs/abs_to_rel skew mechanism as VITS's `attentions.Encoder`,
// channels=64/n_heads=2/window_size=4 here). Verified against the real
// `duration_predictor.pt`'s own `sentence_encoder.attn_layers[0]` module, T=15 (> window_size+1=5,
// exercising the zero-pad branch of `_get_relative_embeddings`). Skips cleanly if the GGUF/reference
// files aren't present.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

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
    const char* dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_DIR");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_DIR (supertonic_relpos_attn.gguf) and "
                              "LOOM_SUPERTONIC_REF_DIR (relpos_attn_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kT = 15;
    constexpr uint32_t kChannels = 64;

    const std::vector<float> x = read_f32_binary(ref_dir + "/relpos_attn_x.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/relpos_attn_out.bin");
    LOOM_CHECK(x.size() == static_cast<size_t>(kT) * kChannels);
    LOOM_CHECK(expected.size() == x.size());

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/supertonic_relpos_attn.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", kT}, {"n_past", 0}});

    ggml_backend_tensor_set(r.input_tensors.at("x"), x.data(), 0, x.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    LOOM_CHECK(static_cast<size_t>(ggml_nelements(r.output)) == expected.size());
    std::vector<float> out(expected.size());
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(static_cast<double>(out[i]) - static_cast<double>(expected[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / out.size();
    std::fprintf(stderr, "mean_diff=%g, max_diff=%g\n", mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-4);
    LOOM_CHECK(max_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
