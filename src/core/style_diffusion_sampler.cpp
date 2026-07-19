#include "loom/core/style_diffusion_sampler.h"

#include <cmath>

namespace loom {

std::vector<float> karras_schedule(int num_steps, float sigma_min, float sigma_max, float rho) {
    const float rho_inv = 1.0f / rho;
    const float smin_r = std::pow(sigma_min, rho_inv);
    const float smax_r = std::pow(sigma_max, rho_inv);
    const float denom = static_cast<float>(num_steps > 1 ? num_steps - 1 : 1);

    std::vector<float> sigmas(static_cast<size_t>(num_steps) + 1);
    for (int i = 0; i < num_steps; ++i) {
        const float t = static_cast<float>(i) / denom;
        const float v = smax_r + t * (smin_r - smax_r);
        sigmas[static_cast<size_t>(i)] = std::pow(v, rho);
    }
    sigmas[static_cast<size_t>(num_steps)] = 0.0f;
    return sigmas;
}

std::vector<float> adpm2_step(const std::vector<float>& x, const DenoiseFn& denoise_fn, float sigma,
                               float sigma_next, const GaussianSampleFn& gaussian_sample) {
    const float sigma_up = std::sqrt(sigma_next * sigma_next * (sigma * sigma - sigma_next * sigma_next) /
                                      (sigma * sigma));
    const float sigma_down = std::sqrt(sigma_next * sigma_next - sigma_up * sigma_up);
    const float sigma_mid = (sigma + sigma_down) / 2.0f; // r=1.0: ((sigma^(1/r)+sigma_down^(1/r))/2)^r

    const std::vector<float> denoised = denoise_fn(x, sigma);
    std::vector<float> x_mid(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float d = (x[i] - denoised[i]) / sigma;
        x_mid[i] = x[i] + d * (sigma_mid - sigma);
    }

    const std::vector<float> denoised_mid = denoise_fn(x_mid, sigma_mid);
    std::vector<float> x_next(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float d_mid = (x_mid[i] - denoised_mid[i]) / sigma_mid;
        x_next[i] = x[i] + d_mid * (sigma_down - sigma);
    }

    std::vector<float> noise(x.size());
    gaussian_sample(noise);
    for (size_t i = 0; i < x.size(); ++i) x_next[i] += noise[i] * sigma_up;
    return x_next;
}

std::vector<float> adpm2_sample(const std::vector<float>& noise, const DenoiseFn& denoise_fn,
                                 const std::vector<float>& sigmas, int num_steps,
                                 const GaussianSampleFn& gaussian_sample) {
    std::vector<float> x(noise.size());
    for (size_t i = 0; i < noise.size(); ++i) x[i] = sigmas[0] * noise[i];

    for (int i = 0; i < num_steps - 1; ++i) {
        x = adpm2_step(x, denoise_fn, sigmas[static_cast<size_t>(i)], sigmas[static_cast<size_t>(i + 1)],
                        gaussian_sample);
    }
    return x;
}

} // namespace loom
