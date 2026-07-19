// Combines the two pieces verified separately so far -- loom::adpm2_sample (the generic ADPM2/Karras
// sampling loop, tests/test_style_diffusion_sampler.cpp) and the real Transformer1d denoiser network
// (styletts2_diffusion.gguf, tests/test_e2e_styletts2_diffusion_net.cpp) -- into the FULL style-diffusion
// sampling loop, wrapping the network in KDiffusion's own c_skip/c_out/c_in/c_noise preconditioning
// exactly like the real DiffusionSampler does. Verified against
// reference_forward_styletts2_diffusion_sampler.py, which combines the SAME two independently-verified
// pieces on the Python side. This is the first point in this project where the sampler loop and the real
// network run together -- everything downstream (StyleTTS2Driver) depends on this being correct. Ancestral
// noise draws are REPLAYED from the fixture (see test_style_diffusion_sampler.cpp's own rationale for why
// an exact-match test needs this rather than a real RNG). Skips cleanly if the GGUF/reference files aren't
// present.

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
                              "LOOM_STYLETTS2_DIFFUSION_REF_DIR (sampler_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kChannels = 256;
    constexpr uint32_t kContextFeatures = 768;
    constexpr int kNumSteps = 5;
    constexpr float kSigmaData = 0.45731624995853165f;

    const std::vector<float> embedding = read_f32_binary(ref_dir + "/sampler_embedding.bin");
    const std::vector<float> sigmas = read_f32_binary(ref_dir + "/sampler_sigmas.bin");
    const std::vector<float> noise0 = read_f32_binary(ref_dir + "/sampler_noise0.bin");
    const std::vector<float> step_noises_flat = read_f32_binary(ref_dir + "/sampler_step_noises.bin");
    const std::vector<float> expected_x_final = read_f32_binary(ref_dir + "/sampler_expected_x_final.bin");

    LOOM_CHECK(embedding.size() % kContextFeatures == 0);
    const uint32_t T = static_cast<uint32_t>(embedding.size() / kContextFeatures);
    LOOM_CHECK(sigmas.size() == static_cast<size_t>(kNumSteps) + 1);
    LOOM_CHECK(noise0.size() == kChannels);
    LOOM_CHECK(step_noises_flat.size() == static_cast<size_t>(kNumSteps - 1) * kChannels);
    LOOM_CHECK(expected_x_final.size() == kChannels);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/styletts2_diffusion.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    const std::vector<float> zero_mask(static_cast<size_t>(T) * T, 0.0f);

    // KDiffusion-preconditioned DenoiseFn: c_in scales x BEFORE the graph call, c_noise is fed in as
    // `time`, c_skip/c_out combine the graph's raw output back into x_denoised AFTER it returns -- the
    // exact same split established in convert_styletts2_diffusion.py's own docstring.
    loom::DenoiseFn denoise_fn = [&](const std::vector<float>& x, float sigma) {
        const float c_skip = (kSigmaData * kSigmaData) / (sigma * sigma + kSigmaData * kSigmaData);
        const float c_out = sigma * kSigmaData / std::sqrt(kSigmaData * kSigmaData + sigma * sigma);
        const float c_in = 1.0f / std::sqrt(sigma * sigma + kSigmaData * kSigmaData);
        const float c_noise = std::log(sigma) * 0.25f;

        std::vector<float> x_scaled(x.size());
        for (size_t i = 0; i < x.size(); ++i) x_scaled[i] = x[i] * c_in;

        loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
        loom::GraphBuilder::BuildResult r = builder.build(T, 0);
        ggml_backend_tensor_set(r.input_tensors.at("x_in"), x_scaled.data(), 0, x_scaled.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("time"), &c_noise, 0, sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("embedding"), embedding.data(), 0, embedding.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), zero_mask.data(), 0, zero_mask.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), r.graph);

        LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == kChannels);
        std::vector<float> model_out(kChannels);
        ggml_backend_tensor_get(r.output, model_out.data(), 0, model_out.size() * sizeof(float));

        std::vector<float> x_denoised(x.size());
        for (size_t i = 0; i < x.size(); ++i) x_denoised[i] = c_skip * x[i] + c_out * model_out[i];
        return x_denoised;
    };

    int step_call = 0;
    loom::GaussianSampleFn replay_noise = [&](std::vector<float>& out) {
        LOOM_CHECK(step_call < kNumSteps - 1);
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = step_noises_flat[static_cast<size_t>(step_call) * kChannels + i];
        ++step_call;
    };

    const std::vector<float> x_final = loom::adpm2_sample(noise0, denoise_fn, sigmas, kNumSteps, replay_noise);
    LOOM_CHECK(x_final.size() == kChannels);
    LOOM_CHECK(step_call == kNumSteps - 1);

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < kChannels; ++i) {
        const double d = std::fabs(static_cast<double>(x_final[i]) - static_cast<double>(expected_x_final[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / kChannels;
    std::fprintf(stderr, "T=%u, mean_diff=%g, max_diff=%g\n", T, mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-3);
    LOOM_CHECK(max_diff < 1e-1);

    LOOM_TEST_REPORT_AND_RETURN();
}
