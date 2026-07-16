#pragma once

#include "loom/core/graph_builder.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace loom {

struct OdeStepConfig {
    uint32_t n_steps = 10;
    float t_start = 0.0f;
    float t_end = 1.0f;

    // Optional: invoked once per integration step with (t, velocity) right after the graph computes that
    // step's vector field -- lets callers (tests, debugging tools) inspect intermediate values without
    // OdeStepper needing to accumulate and return them all itself.
    std::function<void(float t, const std::vector<float>& velocity)> on_step;
};

// The flow-matching ODE integration driver (SPECIFICATION.md §4: "the C++ engine handles the while
// loop, repeatedly executing the sub-graph while updating the timestep and noisy latent input tensors
// in-place"). Builds the graph ONCE via GraphBuilder::build() and reuses the same ggml_cgraph for every
// integration step, only rewriting input tensor data and recomputing -- exactly SPECIFICATION.md's
// intent, and safe: see BACKLOG.md's "ggml graph-reuse" finding for the full investigation, but the
// short version is that ggml_gallocr may alias ANY declared input tensor's buffer as scratch storage for
// some node's output (confirmed empirically, not just for this topology), so EVERY declared input must
// be rewritten before EVERY compute() call -- including "conditioning", which never logically changes
// across steps. Skipping that (writing a "constant" input once, outside the loop) is what produced
// numerically wrong results from the second step onward in an earlier version of this class; writing all
// three every step, as integrate() does below, was verified bit-identical to a from-scratch rebuild.
//
// Assumes the topology follows this milestone's fixed input-naming convention: declared graph inputs
// named "latent" (f32, [n_tokens, n_channels] -- n_tokens is the frame/time dimension, resolved from
// GraphBuilder's usual "n_tokens" symbol), "timestep" (f32, [1, n_channels] -- a scalar broadcast across
// channels, not a learned/sinusoidal embedding; see BACKLOG.md for why), and "conditioning" (f32,
// [1, n_channels] -- logically constant across steps, but still rewritten every step, see above). The
// declared topology output (whatever GraphTopology::output names it -- "velocity" by convention) must be
// the same shape as "latent": the vector field d(latent)/dt.
class OdeStepper {
public:
    OdeStepper(GgufModel& model, GraphTopology topo, ggml_backend_t backend);

    // Integrates from `initial_latent` (flat host buffer, size n_tokens*n_channels) using a fixed
    // `conditioning` embedding (size n_channels) via forward Euler, returning the final latent (same
    // size as `initial_latent`). n_tokens is inferred as initial_latent.size() / conditioning.size().
    std::vector<float> integrate(const std::vector<float>& initial_latent,
                                  const std::vector<float>& conditioning,
                                  const OdeStepConfig& cfg);

private:
    GgufModel& model_;
    GraphTopology topo_;
    ggml_backend_t backend_;
    GraphBuilder builder_;
};

} // namespace loom
