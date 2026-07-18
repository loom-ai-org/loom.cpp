#pragma once

#include "loom/core/graph_builder.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace loom {

struct VitsConfig {
    uint32_t hidden_channels = 192;  // TextEncoder's own hidden width (also SDP's forced filter_channels)
    uint32_t inter_channels = 192;   // flow/vocoder channel width (m_p/logs_p/z_p's own channel count)
    uint32_t n_heads = 2;
    uint32_t n_text_layers = 6;      // TextEncoder's attention-layer count (how many emb_rel_k/v tables exist)
    uint32_t window_size = 4;        // attentions.Encoder's relative-position window
    float noise_scale = 0.667f;      // z_p's own sampling noise (real default, models.py's infer())
    float noise_scale_w = 0.8f;      // SDP's internal z_noise sampling (real default)
    float length_scale = 1.0f;       // duration multiplier (real default)
};

// Host-side driver for VITS (piper) text-to-speech inference, tying together the three GGUF
// files tools/convert_piper_vits/convert_vits.py produces: `stats` (TextEncoder -> m_p/logs_p),
// `logw` (TextEncoder + StochasticDurationPredictor reverse -> duration logits), and
// `flow_vocoder` (coupling flow reverse + HiFi-GAN vocoder -> waveform).
//
// This is the "TTS Catch" two-phase pattern (SPECIFICATION.md §4), same as TdtDecoder/OdeStepper: the
// duration predictor's own output determines the TOTAL OUTPUT FRAME COUNT (`y_length`) -- a genuinely
// data-dependent value nothing before it in this whole codebase needed (every other dynamic-length case
// is the INPUT's own static length, known before running anything; here it's something the model itself
// computes at runtime). `generate_path` (real code: commons.py) is therefore done HOST-SIDE, not as a
// ggml composite -- it degenerates to a plain "replicate column t of m_p/logs_p for w_ceil[t] consecutive
// output frames" expansion once x_mask/y_mask are dropped (both always all-ones here: single unpadded
// utterance, no batching, matching every other simplification this whole VITS effort has made).
//
// `emb_rel_k`/`emb_rel_v` (TextEncoder's relative-position tables) are genuinely dynamic-length (`2*T-1`
// for the real per-call T, not the fixed `2*window_size+1` the checkpoint stores) -- computed here via
// `pad_crop_relative_embeddings`, a direct C++ port of piper's own `attentions.py::
// _get_relative_embeddings` (verified against the real function in Python -- see
// tools/convert_piper_vits/vits_common.py's `get_relative_embeddings`, this is the same algorithm).
class VitsDriver {
public:
    // Each phase's GgufModel/GraphTopology pair comes from convert_vits.py's three separate output
    // files (vits_stats.gguf, vits_logw.gguf, vits_flow_vocoder.gguf) -- GraphTopology supports only
    // one declared output each, and GgufModel::load requires exactly one "model.graph_topology" KV per
    // file, so there is no single shared model/topology object here (same reasoning recorded in
    // BACKLOG.md). Models are referenced, not owned, matching GraphBuilder's own convention.
    VitsDriver(GgufModel& stats_model, GraphTopology stats_topo, GgufModel& logw_model, GraphTopology logw_topo,
               GgufModel& flow_vocoder_model, GraphTopology flow_vocoder_topo, VitsConfig cfg,
               ggml_backend_t backend);

    // token_ids: phoneme/symbol ids (TextEncoder's own vocabulary), length T. `seed` seeds this call's
    // own RNG (SDP's z_noise sampling and z_p's noise_scale sampling) -- real inference is stochastic by
    // design (StochasticDurationPredictor), so callers that need bit-exact reproducibility must fix a
    // seed; callers that just want speech don't need to.
    //
    // Returns the raw waveform (mono, one sample per element, at the model's own sample rate -- not
    // tracked here, purely a GGUF/hparam concern for whatever writes it to a file).
    std::vector<float> synthesize(const std::vector<int32_t>& token_ids, uint32_t seed);

private:
    GgufModel& stats_model_;
    GgufModel& logw_model_;
    GgufModel& flow_vocoder_model_;
    VitsConfig cfg_;
    ggml_backend_t backend_;

    // Declaration order matters: each GraphBuilder stores a reference to its corresponding
    // GraphTopology, so the topologies must be fully constructed first (same precedent as
    // TdtDecoder/Generator).
    GraphTopology stats_topo_;
    GraphTopology logw_topo_;
    GraphTopology flow_vocoder_topo_;
    std::unique_ptr<GraphBuilder> stats_builder_;
    std::unique_ptr<GraphBuilder> logw_builder_;
    std::unique_ptr<GraphBuilder> flow_vocoder_builder_;
};

} // namespace loom
