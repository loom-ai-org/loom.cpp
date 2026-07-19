#pragma once

// StyleTTS2's style-diffusion sampling loop (Karras et al. 2022 "ancestral" DPM-2, real source:
// yl4579/StyleTTS2's Modules/diffusion/sampler.py -- `KarrasSchedule` + `ADPM2Sampler`). This is a
// host-driven iterative loop calling a denoiser function repeatedly with host-updated scalar `sigma`
// values, directly analogous to this project's existing `ode_stepper.cpp` precedent (VITS's normalizing
// flow ODE integration) -- the denoiser itself (KDiffusion preconditioning + the real Transformer1d
// network) is a ggml graph, invoked here only through the `DenoiseFn` callback so this file has no ggml
// dependency at all and can be verified against a hand-rolled Python reference using a toy denoiser
// (see tools/fixture_gen/reference_style_diffusion_sampler.py) before ever touching the real network.

#include <functional>
#include <vector>

namespace loom {

// Karras et al. 2022 eq. 5: returns num_steps+1 sigmas, decreasing from sigma_max to sigma_min at index
// num_steps-1, then a final 0.0 appended. NOTE: `adpm2_sample`'s own loop below never actually reaches
// this trailing 0 (real source behavior, not a bug -- see ADPM2Sampler.forward's own `range(num_steps-1)`
// bound, which only ever indexes sigmas[0..num_steps-1]).
std::vector<float> karras_schedule(int num_steps, float sigma_min, float sigma_max, float rho = 7.0f);

// `denoise_fn(x, sigma)` -- one KDiffusion-preconditioned network call (a full ggml graph invocation in
// real use, a toy affine map in tests). `gaussian_sample(out)` fills `out` (already sized) with fresh
// N(0,1) draws in place -- injectable so tests can replay an externally-generated deterministic sequence
// instead of depending on a real RNG (two independently-implemented RNGs will never agree bit-for-bit,
// so this decouples "is the discretization math right" from "do the RNGs match").
using DenoiseFn = std::function<std::vector<float>(const std::vector<float>& x, float sigma)>;
using GaussianSampleFn = std::function<void(std::vector<float>& out)>;

// One ADPM2 step (rho=1.0, the real ADPM2Sampler class default): a 2nd-order midpoint update from sigma
// to sigma_next, followed by one ancestral-noise injection scaled by sigma_up.
std::vector<float> adpm2_step(const std::vector<float>& x, const DenoiseFn& denoise_fn, float sigma,
                               float sigma_next, const GaussianSampleFn& gaussian_sample);

// Full ADPM2Sampler.forward: x = sigmas[0]*noise, then num_steps-1 calls to adpm2_step over
// sigmas[0..num_steps-1].
std::vector<float> adpm2_sample(const std::vector<float>& noise, const DenoiseFn& denoise_fn,
                                 const std::vector<float>& sigmas, int num_steps,
                                 const GaussianSampleFn& gaussian_sample);

} // namespace loom
