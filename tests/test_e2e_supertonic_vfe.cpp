// Numerical-correctness check for SupertonicTTS v2's FULL VectorFieldEstimator.compute_velocity (real
// source: vector_field_estimator.py) -- the biggest single assembly in this project's SupertonicTTS
// effort (4 groups x (4 dilated ConvNeXt + time conditioning + ConvNeXt + fractional-RoPE text
// cross-attention + ConvNeXt + style cross-attention) + final ConvNeXt stack) -- against the real
// `vector_estimator.pt` module, ONE velocity call (no ODE loop yet). Skips cleanly if the GGUF/reference
// files aren't present.

#include "test_util.h"

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
    const char* dir_env = std::getenv("LOOM_SUPERTONIC_DIR");
    const char* ref_dir_env = std::getenv("LOOM_SUPERTONIC_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_DIR (supertonic_vfe.gguf) and "
                              "LOOM_SUPERTONIC_REF_DIR (vfe_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kL = 9;
    constexpr uint32_t kT = 6;
    constexpr uint32_t kLatentDim = 144;
    constexpr uint32_t kTxtDim = 256;
    constexpr uint32_t kStlDim = 256;
    constexpr uint32_t kNStyle = 50;

    const std::vector<float> z_t = read_f32_binary(ref_dir + "/vfe_z_t.bin");
    const std::vector<float> txt_emb = read_f32_binary(ref_dir + "/vfe_txt_emb.bin");
    const std::vector<float> stl_emb = read_f32_binary(ref_dir + "/vfe_stl_emb.bin");
    const std::vector<float> t = read_f32_binary(ref_dir + "/vfe_t.bin");
    const std::vector<float> lat_frac = read_f32_binary(ref_dir + "/vfe_lat_frac.bin");
    const std::vector<float> txt_frac = read_f32_binary(ref_dir + "/vfe_txt_frac.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/vfe_expected_v.bin");
    LOOM_CHECK(z_t.size() == static_cast<size_t>(kL) * kLatentDim);
    LOOM_CHECK(txt_emb.size() == static_cast<size_t>(kTxtDim) * kT);
    LOOM_CHECK(stl_emb.size() == static_cast<size_t>(kStlDim) * kNStyle);
    LOOM_CHECK(t.size() == 1);
    LOOM_CHECK(expected.size() == z_t.size());

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/supertonic_vfe.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    loom::GraphBuilder::BuildResult r = builder.build(kL, 0);

    ggml_backend_tensor_set(r.input_tensors.at("z_t"), z_t.data(), 0, z_t.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("txt_emb"), txt_emb.data(), 0, txt_emb.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl_emb.data(), 0, stl_emb.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("t"), t.data(), 0, sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("lat_frac"), lat_frac.data(), 0, lat_frac.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("txt_frac"), txt_frac.data(), 0, txt_frac.size() * sizeof(float));
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
    LOOM_CHECK(mean_diff < 1e-2);
    LOOM_CHECK(max_diff < 1.0);

    LOOM_TEST_REPORT_AND_RETURN();
}
