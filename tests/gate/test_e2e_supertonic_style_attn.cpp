// Numerical-correctness check for SupertonicTTS v2's `StyleEncoderCrossAttention` (style-token-pooling
// cross-attention: a learnable-query first stage, then a 2nd stage refining against the same original
// input, real source: components.py) against the real `dp-style-encoder.pt`'s own `style_token_layer`
// (dim=64, stl_dim=16, n_style=8). Skips cleanly if the GGUF/reference files aren't present.

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
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_DIR (supertonic_style_attn_dp.gguf) and "
                              "LOOM_SUPERTONIC_REF_DIR (style_attn_dp_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kT = 20;
    constexpr uint32_t kDim = 64;
    constexpr uint32_t kStlDim = 16;
    constexpr uint32_t kNStyle = 8;

    const std::vector<float> x = read_f32_binary(ref_dir + "/style_attn_dp_x.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/style_attn_dp_out.bin");
    LOOM_CHECK(x.size() == static_cast<size_t>(kT) * kDim);
    LOOM_CHECK(expected.size() == static_cast<size_t>(kNStyle) * kStlDim);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/supertonic_style_attn_dp.gguf", backend.get());
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
