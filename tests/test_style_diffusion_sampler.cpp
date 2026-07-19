// Verifies loom::karras_schedule/adpm2_sample (StyleTTS2's style-diffusion sampling loop, real source:
// Modules/diffusion/sampler.py's KarrasSchedule/ADPM2Sampler) against an independent numpy reference
// using a TOY affine denoiser (isolates the discretization math from the real Transformer1d network,
// same "verify the mechanism first" discipline as test_hifigan_generator). The ancestral-noise draws are
// REPLAYED from the fixture (not freshly sampled) so this is an exact numerical comparison, not a
// statistical one -- two independently-implemented RNGs would never agree bit-for-bit, so the fixture
// pre-generates the noise and this test's GaussianSampleFn just serves it back in order.

#include "test_util.h"

#include "loom/loom.h"

#include <cstdio>
#include <fstream>
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
    const std::string ref_dir = LOOM_TEST_REF_DIR;

    constexpr size_t kDim = 8;
    constexpr int kNumSteps = 5;

    const std::vector<float> sigmas = read_f32_binary(ref_dir + "/sigmas.bin");
    const std::vector<float> a = read_f32_binary(ref_dir + "/a.bin");
    const std::vector<float> b = read_f32_binary(ref_dir + "/b.bin");
    const std::vector<float> noise0 = read_f32_binary(ref_dir + "/noise0.bin");
    const std::vector<float> step_noises_flat = read_f32_binary(ref_dir + "/step_noises.bin");
    const std::vector<float> expected_x_final = read_f32_binary(ref_dir + "/expected_x_final.bin");
    const std::vector<float> expected_trace_flat = read_f32_binary(ref_dir + "/expected_trace.bin");

    LOOM_CHECK(sigmas.size() == static_cast<size_t>(kNumSteps) + 1);
    LOOM_CHECK(a.size() == kDim && b.size() == kDim && noise0.size() == kDim);
    LOOM_CHECK(step_noises_flat.size() == static_cast<size_t>(kNumSteps - 1) * kDim);
    LOOM_CHECK(expected_x_final.size() == kDim);

    // Sanity-check karras_schedule itself against the fixture's own sigmas (built via the SAME formula
    // in Python) before trusting adpm2_sample on top of it.
    const std::vector<float> computed_sigmas = loom::karras_schedule(kNumSteps, /*sigma_min=*/1e-4f,
                                                                       /*sigma_max=*/3.0f, /*rho=*/9.0f);
    LOOM_CHECK(computed_sigmas.size() == sigmas.size());
    for (size_t i = 0; i < sigmas.size(); ++i) {
        const float diff = std::fabs(computed_sigmas[i] - sigmas[i]);
        LOOM_CHECK(diff < 1e-4f);
    }

    loom::DenoiseFn toy_denoise = [&](const std::vector<float>& x, float /*sigma*/) {
        std::vector<float> out(x.size());
        for (size_t i = 0; i < x.size(); ++i) out[i] = a[i] * x[i] + b[i];
        return out;
    };

    int step_call = 0;
    loom::GaussianSampleFn replay_noise = [&](std::vector<float>& out) {
        LOOM_CHECK(step_call < kNumSteps - 1);
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = step_noises_flat[static_cast<size_t>(step_call) * kDim + i];
        ++step_call;
    };

    const std::vector<float> x_final = loom::adpm2_sample(noise0, toy_denoise, sigmas, kNumSteps, replay_noise);
    LOOM_CHECK(x_final.size() == kDim);
    LOOM_CHECK(step_call == kNumSteps - 1);

    double max_diff = 0.0, mean_diff = 0.0;
    for (size_t i = 0; i < kDim; ++i) {
        const double diff = std::fabs(static_cast<double>(x_final[i]) - static_cast<double>(expected_x_final[i]));
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
    }
    mean_diff /= kDim;
    std::fprintf(stderr, "adpm2_sample: mean_diff=%.3e max_diff=%.3e\n", mean_diff, max_diff);
    LOOM_CHECK(max_diff < 1e-3);

    (void)expected_trace_flat;
    LOOM_TEST_REPORT_AND_RETURN();
}
