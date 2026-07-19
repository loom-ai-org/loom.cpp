// Combines the two pieces verified separately so far -- loom::cfm_euler_sample (the generic Euler CFM
// sampling loop) and the real VectorFieldEstimator (styletts... no, supertonic_vfe.gguf, already
// verified against vector_estimator.pt.compute_velocity()) -- into the FULL CFM sampling loop, matching
// real TextToLatentWrapper.predict's own call order. Verified against
// reference_forward_supertonic_cfm_sampler.py, which combines the SAME two independently-verified pieces
// on the Python side (calling the real `ve.solve()` in a loop). Unlike StyleTTS2's own ADPM2 sampler,
// this is fully DETERMINISTIC given z0 -- no ancestral noise to replay, so this is a much simpler
// combination test. Skips cleanly if the GGUF/reference files aren't present.

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
                              "LOOM_SUPERTONIC_REF_DIR (cfm_*.bin) to run this numerical check\n");
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
    constexpr int kNumSteps = 5;

    const std::vector<float> z0 = read_f32_binary(ref_dir + "/cfm_z0.bin");
    const std::vector<float> txt_emb = read_f32_binary(ref_dir + "/cfm_txt_emb.bin");
    const std::vector<float> stl_emb = read_f32_binary(ref_dir + "/cfm_stl_emb.bin");
    const std::vector<float> lat_frac = read_f32_binary(ref_dir + "/cfm_lat_frac.bin");
    const std::vector<float> txt_frac = read_f32_binary(ref_dir + "/cfm_txt_frac.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/cfm_expected_z_final.bin");
    LOOM_CHECK(z0.size() == static_cast<size_t>(kL) * kLatentDim);
    LOOM_CHECK(txt_emb.size() == static_cast<size_t>(kT) * kTxtDim);
    LOOM_CHECK(stl_emb.size() == static_cast<size_t>(kStlDim) * kNStyle);
    LOOM_CHECK(expected.size() == z0.size());

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/supertonic_vfe.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    loom::VelocityFn velocity_fn = [&](const std::vector<float>& z, float t) {
        loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
        loom::GraphBuilder::BuildResult r = builder.build(kL, 0);
        ggml_backend_tensor_set(r.input_tensors.at("z_t"), z.data(), 0, z.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("txt_emb"), txt_emb.data(), 0, txt_emb.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl_emb.data(), 0, stl_emb.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("t"), &t, 0, sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("lat_frac"), lat_frac.data(), 0, lat_frac.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("txt_frac"), txt_frac.data(), 0, txt_frac.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), r.graph);

        LOOM_CHECK(static_cast<size_t>(ggml_nelements(r.output)) == z.size());
        std::vector<float> v(z.size());
        ggml_backend_tensor_get(r.output, v.data(), 0, v.size() * sizeof(float));
        return v;
    };

    const std::vector<float> z_final = loom::cfm_euler_sample(z0, velocity_fn, kNumSteps);
    LOOM_CHECK(z_final.size() == expected.size());

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < z_final.size(); ++i) {
        const double d = std::fabs(static_cast<double>(z_final[i]) - static_cast<double>(expected[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / z_final.size();
    std::fprintf(stderr, "mean_diff=%g, max_diff=%g\n", mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-2);
    LOOM_CHECK(max_diff < 1e-1);

    LOOM_TEST_REPORT_AND_RETURN();
}
