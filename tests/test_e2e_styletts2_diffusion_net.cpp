// Numerical-correctness check for StyleTTS2's style-diffusion denoiser network (the plain `Transformer1d`
// substituted in place of the AudioDiffusionConditional's own default U-Net, see
// tools/convert_styletts2/convert_styletts2_diffusion.py's own module docstring) against the real
// checkpoint's own weights and a hand-rolled PyTorch reference (reference_forward_styletts2_diffusion.py).
// Exercises ONE raw denoise-network call (KDiffusion's c_skip/c_out/c_in/c_noise preconditioning is
// deliberately NOT part of this graph -- see the conversion script's own docstring for why), driven with
// a synthetic embedding (real weights, synthetic driving input, same scope as this project's other
// individual-piece verification tests). Skips cleanly if the GGUF/reference files aren't present.

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
    const char* dir_env = std::getenv("LOOM_STYLETTS2_DIR");
    const char* ref_dir_env = std::getenv("LOOM_STYLETTS2_DIFFUSION_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STYLETTS2_DIR (styletts2_diffusion.gguf) and "
                              "LOOM_STYLETTS2_DIFFUSION_REF_DIR (diff_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kChannels = 256;
    constexpr uint32_t kContextFeatures = 768;

    const std::vector<float> x_in = read_f32_binary(ref_dir + "/diff_x_in.bin");
    const std::vector<float> time = read_f32_binary(ref_dir + "/diff_time.bin");
    const std::vector<float> embedding = read_f32_binary(ref_dir + "/diff_embedding.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/diff_expected_model_out.bin");
    LOOM_CHECK(x_in.size() == kChannels);
    LOOM_CHECK(time.size() == 1);
    LOOM_CHECK(expected.size() == kChannels);
    LOOM_CHECK(embedding.size() % kContextFeatures == 0);
    const uint32_t T = static_cast<uint32_t>(embedding.size() / kContextFeatures);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/styletts2_diffusion.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", T}, {"n_past", 0}});

    ggml_backend_tensor_set(r.input_tensors.at("x_in"), x_in.data(), 0, x_in.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("time"), time.data(), 0, time.size() * sizeof(float));
    // embedding (ref: (T,768) row-major, time-major) is byte-identical to ggml ne=[768,T] -- no reordering.
    ggml_backend_tensor_set(r.input_tensors.at("embedding"), embedding.data(), 0, embedding.size() * sizeof(float));
    std::vector<float> zero_mask(static_cast<size_t>(T) * T, 0.0f);
    ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), zero_mask.data(), 0, zero_mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);

    LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == kChannels);
    std::vector<float> out(kChannels);
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(static_cast<double>(out[i]) - static_cast<double>(expected[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / static_cast<double>(out.size());
    std::fprintf(stderr, "T=%u, mean_diff=%g, max_diff=%g\n", T, mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-4);
    LOOM_CHECK(max_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
