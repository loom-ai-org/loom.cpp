#include "loom/core/ode_stepper.h"

#include <ggml-backend.h>

namespace loom {

OdeStepper::OdeStepper(GgufModel& model, GraphTopology topo, ggml_backend_t backend)
    : model_(model), topo_(std::move(topo)), backend_(backend), builder_(topo_, model_, backend, /*kv_cache=*/nullptr) {}

std::vector<float> OdeStepper::integrate(const std::vector<float>& initial_latent,
                                          const std::vector<float>& conditioning,
                                          const OdeStepConfig& cfg) {
    const auto n_elems = static_cast<uint32_t>(initial_latent.size());
    const auto n_channels = static_cast<uint32_t>(conditioning.size());
    const uint32_t n_tokens = n_elems / n_channels;

    // Built once: the vector field's shape never changes across integration steps, unlike Generator's
    // autoregressive decode where n_kv grows every token. (Hoisting it out of the loop is now belt and
    // braces -- GraphBuilder retains the graph, so building per step would return this same one --
    // but it still says what this loop means.) See the header comment for why every declared input,
    // including "conditioning", is rewritten every step.
    const GraphBuilder::BuildResult& result = builder_.build({{"n_tokens", n_tokens}, {"n_past", /*n_past=*/0}});
    ggml_tensor* latent_t = result.input_tensors.at("latent");
    ggml_tensor* timestep_t = result.input_tensors.at("timestep");
    ggml_tensor* conditioning_t = result.input_tensors.at("conditioning");

    std::vector<float> latent = initial_latent;
    const float dt = (cfg.t_end - cfg.t_start) / static_cast<float>(cfg.n_steps);

    for (uint32_t step = 0; step < cfg.n_steps; ++step) {
        const float t = cfg.t_start + static_cast<float>(step) * dt;

        // All three declared inputs, every step -- see above.
        ggml_backend_tensor_set(latent_t, latent.data(), 0, latent.size() * sizeof(float));
        const std::vector<float> t_broadcast(n_channels, t);
        ggml_backend_tensor_set(timestep_t, t_broadcast.data(), 0, t_broadcast.size() * sizeof(float));
        ggml_backend_tensor_set(conditioning_t, conditioning.data(), 0, conditioning.size() * sizeof(float));

        ggml_backend_graph_compute(backend_, result.graph);

        std::vector<float> velocity(n_elems);
        ggml_backend_tensor_get(result.output, velocity.data(), 0, velocity.size() * sizeof(float));

        for (uint32_t i = 0; i < n_elems; ++i) {
            latent[i] += dt * velocity[i];
        }

        if (cfg.on_step) cfg.on_step(t, velocity);
    }

    return latent;
}

} // namespace loom
