#pragma once

#include "loom/core/graph_builder.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace loom {

struct TdtDecoderConfig {
    int32_t blank_id = -1;
    // e.g. {0,1,2,3,4}; index into this via the joint's duration argmax. Leave EMPTY for a plain
    // (non-TDT) RNN-T model: the joint then has no duration head at all (output width is exactly
    // n_token_classes, not n_token_classes + n_durations), and every blank advances exactly one frame
    // (standard RNN-T greedy decoding) instead of a predicted duration -- see decode_greedy's own
    // comment for the exact branch.
    std::vector<uint32_t> durations;
    uint32_t max_symbols_per_step = 10; // bounds the inner per-frame loop
};

// Autoregressive decode driver for Transducer/TDT models (BACKLOG.md's "Gap 1: the Transducer problem"),
// analogous to Generator/OdeStepper: the JSON topologies describe the static per-step sub-graphs, this
// class drives the double loop (encoder-frame pointer x symbols-per-frame) and carries LSTM state
// host-side between calls, matching SPECIFICATION.md §4's "TTS Catch" pattern.
//
// The prediction network is a STACK of `num_layers` LSTM layers (real NeMo/Parakeet-TDT checkpoints use
// 2, confirmed against the real state dict's weight_ih_l0/weight_ih_l1 -- not simplified to 1), chained
// per step: layer 0 embeds the last emitted token; layer i>0 takes layer i-1's h_new directly, no
// embedding lookup. The TOP layer's h_new feeds the joint network.
//
// Needs 2*num_layers + 1 topologies since GraphTopology only supports one declared output each (see
// BACKLOG.md for why a schema extension to support multiple outputs was deliberately deferred):
//   - lstm_h_topos[i]/lstm_c_topos[i]: layer i's declared inputs are ("last_label" i32 [1]) for i==0 or
//     ("layer_input" f32 [pred_hidden]) for i>0, plus ("h_prev"/"c_prev" f32 [pred_hidden]) always --
//     differing only in declared output ("h_new"/"c_new").
//   - joint_topo: inputs "encoder_frame" (f32 [n_embd]) and "decoder_out" (f32 [pred_hidden] -- the top
//     LSTM layer's h_new) -> declared output a single combined [n_token_classes + n_durations] vector
//     (first n_token_classes are token+blank logits, last n_durations are the duration head) -- or, for
//     plain RNN-T (cfg.durations empty), just [n_token_classes] with no duration head at all.
//
// This driver does NOT run the encoder itself -- `encoder_output` is whatever an ordinary, separate,
// non-autoregressive GraphBuilder::build() call already produced (the FastConformer encoder is its own
// static sub-graph, no different in kind from Conformer-CTC's).
class TdtDecoder {
public:
    TdtDecoder(GgufModel& model, std::vector<GraphTopology> lstm_h_topos, std::vector<GraphTopology> lstm_c_topos,
               GraphTopology joint_topo, TdtDecoderConfig cfg, ggml_backend_t backend, uint32_t pred_hidden);

    struct Result {
        std::vector<int32_t> tokens;
        std::vector<uint32_t> frame_indices; // frame_indices[i] is the encoder frame tokens[i] was emitted at
    };

    // encoder_output: n_frames rows, each n_embd floats.
    Result decode_greedy(const std::vector<std::vector<float>>& encoder_output);

private:
    GgufModel& model_;
    TdtDecoderConfig cfg_;
    ggml_backend_t backend_;
    uint32_t pred_hidden_;
    uint32_t num_layers_;

    // Declaration order matters: each GraphBuilder stores a reference to its corresponding GraphTopology,
    // so the topologies must be fully constructed first (same precedent as Generator's topo_/builder_
    // pair). GraphBuilder itself is neither copyable nor move-assignable (a reference member), so a
    // vector of them needs heap indirection.
    std::vector<GraphTopology> lstm_h_topos_;
    std::vector<GraphTopology> lstm_c_topos_;
    GraphTopology joint_topo_;
    std::vector<std::unique_ptr<GraphBuilder>> lstm_h_builders_;
    std::vector<std::unique_ptr<GraphBuilder>> lstm_c_builders_;
    std::unique_ptr<GraphBuilder> joint_builder_;
};

} // namespace loom
