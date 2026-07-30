#include "loom/core/bilstm_stepper.h"

#include <ggml-backend.h>

#include <algorithm>

namespace loom {
namespace {

// Runs one direction's LSTM stack (here: a single layer) for one timestep, mirroring TdtDecoder's own
// per-layer stepping code exactly (see tdt_decoder.cpp) -- duplicated rather than shared since the two
// classes' surrounding control flow (autoregressive double-loop vs. a plain single forward/backward
// pass) is genuinely different, even though this one step's mechanics are identical.
std::vector<float> lstm_cell_step(GraphBuilder& h_builder, GraphBuilder& c_builder, const std::vector<float>& layer_input,
                                   std::vector<float>& h, std::vector<float>& c, ggml_backend_t backend) {
    GraphBuilder::BuildResult hr = h_builder.build({{"n_tokens", 0}, {"n_past", 0}});
    ggml_backend_tensor_set(hr.input_tensors.at("layer_input"), layer_input.data(), 0, layer_input.size() * sizeof(float));
    ggml_backend_tensor_set(hr.input_tensors.at("h_prev"), h.data(), 0, h.size() * sizeof(float));
    ggml_backend_tensor_set(hr.input_tensors.at("c_prev"), c.data(), 0, c.size() * sizeof(float));
    ggml_backend_graph_compute(backend, hr.graph);
    std::vector<float> h_new(h.size());
    ggml_backend_tensor_get(hr.output, h_new.data(), 0, h_new.size() * sizeof(float));

    GraphBuilder::BuildResult cr = c_builder.build({{"n_tokens", 0}, {"n_past", 0}});
    ggml_backend_tensor_set(cr.input_tensors.at("layer_input"), layer_input.data(), 0, layer_input.size() * sizeof(float));
    ggml_backend_tensor_set(cr.input_tensors.at("h_prev"), h.data(), 0, h.size() * sizeof(float));
    ggml_backend_tensor_set(cr.input_tensors.at("c_prev"), c.data(), 0, c.size() * sizeof(float));
    ggml_backend_graph_compute(backend, cr.graph);
    std::vector<float> c_new(c.size());
    ggml_backend_tensor_get(cr.output, c_new.data(), 0, c_new.size() * sizeof(float));

    h = h_new;
    c = c_new;
    return h;
}

} // namespace

BiLstmStepper::BiLstmStepper(GgufModel& model, GraphTopology fwd_h_topo, GraphTopology fwd_c_topo,
                              GraphTopology bwd_h_topo, GraphTopology bwd_c_topo, ggml_backend_t backend,
                              uint32_t hidden_dim_per_direction)
    : model_(model),
      backend_(backend),
      hidden_dim_(hidden_dim_per_direction),
      fwd_h_topo_(std::move(fwd_h_topo)),
      fwd_c_topo_(std::move(fwd_c_topo)),
      bwd_h_topo_(std::move(bwd_h_topo)),
      bwd_c_topo_(std::move(bwd_c_topo)) {
    fwd_h_builder_ = std::make_unique<GraphBuilder>(fwd_h_topo_, model_, backend_, /*kv_cache=*/nullptr);
    fwd_c_builder_ = std::make_unique<GraphBuilder>(fwd_c_topo_, model_, backend_, /*kv_cache=*/nullptr);
    bwd_h_builder_ = std::make_unique<GraphBuilder>(bwd_h_topo_, model_, backend_, /*kv_cache=*/nullptr);
    bwd_c_builder_ = std::make_unique<GraphBuilder>(bwd_c_topo_, model_, backend_, /*kv_cache=*/nullptr);
}

std::vector<std::vector<float>> BiLstmStepper::run(const std::vector<std::vector<float>>& sequence) {
    const auto T = static_cast<uint32_t>(sequence.size());
    std::vector<std::vector<float>> out(T, std::vector<float>(2 * hidden_dim_, 0.0f));

    std::vector<float> h_fwd(hidden_dim_, 0.0f);
    std::vector<float> c_fwd(hidden_dim_, 0.0f);
    for (uint32_t t = 0; t < T; ++t) {
        std::vector<float> h = lstm_cell_step(*fwd_h_builder_, *fwd_c_builder_, sequence[t], h_fwd, c_fwd, backend_);
        std::copy(h.begin(), h.end(), out[t].begin());
    }

    std::vector<float> h_bwd(hidden_dim_, 0.0f);
    std::vector<float> c_bwd(hidden_dim_, 0.0f);
    for (uint32_t i = 0; i < T; ++i) {
        const uint32_t t = T - 1 - i;
        std::vector<float> h = lstm_cell_step(*bwd_h_builder_, *bwd_c_builder_, sequence[t], h_bwd, c_bwd, backend_);
        std::copy(h.begin(), h.end(), out[t].begin() + hidden_dim_);
    }

    return out;
}

} // namespace loom
