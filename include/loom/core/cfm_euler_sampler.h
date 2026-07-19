#pragma once

// SupertonicTTS v2's conditional-flow-matching (CFM) sampling loop (real source:
// text_to_latent_encoding/latent_encoder.py's `TextToLatentWrapper.predict` + vector_field_estimator.py's
// `VectorFieldEstimator.solve`, default solver="euler" -- the real checkpoint never overrides this). Much
// simpler than StyleTTS2's own ADPM2 sampler (style_diffusion_sampler.h): a DETERMINISTIC forward-Euler
// ODE integration, no ancestral noise injection at any step -- the only randomness in the whole pipeline
// is the INITIAL noise z_0. Host-driven loop calling a `VelocityFn` callback (a full ggml graph
// invocation in real use), same "iterative loop calling a graph via callback" shape as
// `style_diffusion_sampler.h`/`ode_stepper.h`.

#include <functional>
#include <vector>

namespace loom {

// `velocity_fn(z, t)` -- one `VectorFieldEstimator.compute_velocity` call (a full ggml graph invocation
// in real use, conditioned on whatever text/style embeddings the caller closed over).
using VelocityFn = std::function<std::vector<float>(const std::vector<float>& z, float t)>;

// Real `TextToLatentWrapper.predict`: `time_steps = arange(n_steps)/n_steps` (uniform steps in [0,1)),
// `dt = 1/n_steps` (constant). For each step: `z = z + velocity_fn(z, t) * dt`. Returns the final latent
// (same size as `z0`).
std::vector<float> cfm_euler_sample(const std::vector<float>& z0, const VelocityFn& velocity_fn,
                                     int n_steps);

} // namespace loom
