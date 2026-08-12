#pragma once

#include "loom/core/graph_builder.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace loom {

// Host-side driver for a single BIDIRECTIONAL LSTM layer run over a full, already-known-length
// sequence (Kokoro/StyleTTS2's `nn.LSTM(..., bidirectional=True)` calls -- TextEncoder's own LSTM,
// ProsodyPredictor's DurationEncoder/top `lstm`/`shared`, all the same shape of problem: a plain,
// non-autoregressive BiLSTM over a short phoneme/frame sequence, genuinely different from
// TdtDecoder's own LSTM stepping, which is autoregressive (each step's INPUT depends on the PREVIOUS
// step's sampled output via the joint network) -- here every timestep's input is already known upfront,
// so both directions can be driven by a plain host-side loop with no data dependency between them.
//
// Considered unrolling entirely in-graph (`repeat_for`), but that needs a per-timestep OUTPUT collected
// into a growing sequence tensor, which this engine has no scatter-into-a-preallocated-tensor primitive
// for yet (see BACKLOG.md's Kokoro research notes) -- so this reuses `TdtDecoder`'s already-proven
// "small per-step topology, h/c carried host-side between `GraphBuilder::build()` calls" pattern
// instead, just without the autoregressive feedback, run once forward and once backward.
//
// Each direction needs its own pair of topologies (`h_topo`/`c_topo`, matching `TdtDecoder`'s own
// per-layer split -- `GraphTopology` supports exactly one declared output each) since GGUF weight
// references are static strings baked in at conversion time: the forward direction's topology
// references e.g. `"lstm.weight_ih_l0"`, the backward direction's references
// `"lstm.weight_ih_l0_reverse"` -- genuinely different weight tensors, not the same topology reused.
// Both topologies' declared inputs are ("layer_input" f32 [input_dim], "h_prev"/"c_prev" f32
// [hidden_dim]) -- no embedding lookup (unlike TdtDecoder's own layer-0 case), since every input here is
// already a continuous feature vector, never a raw token id.
class BiLstmStepper {
public:
    BiLstmStepper(GgufModel& model, GraphTopology fwd_h_topo, GraphTopology fwd_c_topo,
                  GraphTopology bwd_h_topo, GraphTopology bwd_c_topo, Backends backends,
                  uint32_t hidden_dim_per_direction);

    // sequence: T rows, each input_dim floats. Returns T rows, each 2*hidden_dim_per_direction floats:
    // [h_fwd_t ; h_bwd_t] concatenated per position, matching PyTorch's own bidirectional-LSTM output
    // convention (forward direction's hidden state first, backward direction's second).
    std::vector<std::vector<float>> run(const std::vector<std::vector<float>>& sequence);

private:
    GgufModel& model_;
    Backends backends_;
    uint32_t hidden_dim_;

    GraphTopology fwd_h_topo_;
    GraphTopology fwd_c_topo_;
    GraphTopology bwd_h_topo_;
    GraphTopology bwd_c_topo_;
    std::unique_ptr<GraphBuilder> fwd_h_builder_;
    std::unique_ptr<GraphBuilder> fwd_c_builder_;
    std::unique_ptr<GraphBuilder> bwd_h_builder_;
    std::unique_ptr<GraphBuilder> bwd_c_builder_;
};

} // namespace loom
