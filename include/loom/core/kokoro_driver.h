#pragma once

#include "loom/core/bilstm_stepper.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_builder.h"
#include "loom/core/graph_topology.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace loom {

// Real hyperparameters, confirmed against the real checkpoint's config.json / istftnet.py's Generator
// throughout this milestone (see BACKLOG.md) -- kept as one struct, same convention as VitsConfig.
struct KokoroConfig {
    uint32_t style_dim = 128;         // EACH half of ref_s (256 total: decoder half + predictor half)
    uint32_t d_model = 512;           // ProsodyPredictor/TextEncoder hidden width
    uint32_t hidden_per_dir = 256;    // every BiLSTM's per-direction hidden size
    uint32_t max_dur = 50;
    uint32_t harmonic_num = 8;        // SineGen: dim = harmonic_num+1
    float sampling_rate = 24000.0f;
    uint32_t upsample_scale = 300;    // prod(upsample_rates)*gen_istft_hop_size
    uint32_t gen_istft_n_fft = 20;
    uint32_t gen_istft_hop = 5;
    float voiced_threshold = 10.0f;
    float sine_amp = 0.1f;
    float noise_std = 0.003f;
    float ln_eps = 1e-5f;
};

// Host-side driver for Kokoro TTS (hexgrad/Kokoro-82M), tying together every topology
// tools/convert_kokoro/convert_kokoro_all.py produces from the real checkpoint. Real call order,
// confirmed directly against kokoro/model.py's KModel.forward_with_tokens (see BACKLOG.md):
//   CustomAlbert -> bert_encoder -> DurationEncoder (3x BiLSTM+AdaLayerNorm) -> predictor.lstm (BiLSTM)
//   -> duration_proj -> predict_durations/expand_by_duration (applied to BOTH the 640-ch DurationEncoder
//   output "d" and the 512-ch plain TextEncoder output "t_en") -> F0Ntrain (shared BiLSTM + F0/N
//   AdainResBlk1d stacks + projections) -> Decoder core (encode/decode AdainResBlk1d stack) -> SineGen
//   (host rand_ini/noise draws) -> forward STFT -> Generator core (host-precomputed wsum) -> waveform.
//
// Owns every GgufModel it loads (unlike VitsDriver's reference-only convention) -- Kokoro's real
// pipeline spans ~40 small GGUF files (every BiLSTM direction/gate gets its own, matching
// BiLstmStepper's/TdtDecoder's own per-file convention), too many to reasonably hand to callers to
// construct and own themselves.
class KokoroDriver {
public:
    // Loads every GGUF file this needs from `gguf_dir` (as produced by convert_kokoro_all.py). Throws
    // loom::LoadError if any expected file is missing.
    KokoroDriver(const std::string& gguf_dir, KokoroConfig cfg, ggml_backend_t backend);

    // input_ids: phoneme/symbol ids (CustomAlbert's own vocabulary, real model always wraps with a
    // leading/trailing 0 token -- callers' responsibility, matching real KModel.forward's own
    // `[0, *input_ids, 0]`). ref_s: 256 floats (`ref_s[:128]` = decoder style, `ref_s[128:]` = predictor
    // style, real KModel convention). `seed` seeds SineGen's own host-drawn rand_ini/noise (the only
    // stochastic step in this whole pipeline at inference time).
    std::vector<float> synthesize(const std::vector<int32_t>& input_ids, const std::vector<float>& ref_s,
                                   float speed, uint32_t seed);

private:
    KokoroConfig cfg_;
    ggml_backend_t backend_;

    std::unique_ptr<GgufModel> load(const std::string& gguf_dir, const std::string& filename);
    // Runs a plain (non-recurrent) topology once over a whole [T,C]-convention input/output pair --
    // shared helper for every single-shot piece (AdainResBlk1d blocks, projections, bert_encoder, etc).

    std::unique_ptr<GgufModel> albert_model_;
    GraphTopology albert_topo_;

    std::unique_ptr<GgufModel> bert_encoder_model_;
    GraphTopology bert_encoder_topo_;

    std::unique_ptr<GgufModel> text_encoder_cnn_model_;
    GraphTopology text_encoder_cnn_topo_;
    std::unique_ptr<GgufModel> text_encoder_lstm_model_;
    GraphTopology text_encoder_lstm_h_fwd_topo_;
    GraphTopology text_encoder_lstm_c_fwd_topo_;
    GraphTopology text_encoder_lstm_h_bwd_topo_;
    GraphTopology text_encoder_lstm_c_bwd_topo_;

    struct DurationBlock {
        std::unique_ptr<GgufModel> lstm_model;
        GraphTopology h_fwd_topo, c_fwd_topo, h_bwd_topo, c_bwd_topo;
        std::unique_ptr<GgufModel> adaln_model;
        GraphTopology adaln_topo;
    };
    std::vector<DurationBlock> duration_blocks_;  // 3x (DurationEncoder's own lstms.{0,2,4}/lstms.{1,3,5})

    std::unique_ptr<GgufModel> top_lstm_model_;
    GraphTopology top_lstm_h_fwd_topo_, top_lstm_c_fwd_topo_, top_lstm_h_bwd_topo_, top_lstm_c_bwd_topo_;
    std::unique_ptr<GgufModel> duration_proj_model_;
    GraphTopology duration_proj_topo_;

    std::unique_ptr<GgufModel> f0n_shared_lstm_model_;
    GraphTopology f0n_shared_h_fwd_topo_, f0n_shared_c_fwd_topo_, f0n_shared_h_bwd_topo_, f0n_shared_c_bwd_topo_;
    std::unique_ptr<GgufModel> f0_block_models_[3];
    GraphTopology f0_block_topos_[3];
    std::unique_ptr<GgufModel> n_block_models_[3];
    GraphTopology n_block_topos_[3];
    std::unique_ptr<GgufModel> f0_proj_model_;
    GraphTopology f0_proj_topo_;
    std::unique_ptr<GgufModel> n_proj_model_;
    GraphTopology n_proj_topo_;

    std::unique_ptr<GgufModel> decoder_core_model_;
    GraphTopology decoder_core_topo_;

    std::unique_ptr<GgufModel> sinegen_model_;
    GraphTopology sinegen_topo_;
    std::unique_ptr<GgufModel> stft_forward_model_;
    GraphTopology stft_forward_topo_;
    std::unique_ptr<GgufModel> stft_inverse_model_;
    GraphTopology stft_inverse_topo_;

    std::unique_ptr<GgufModel> generator_model_;
    GraphTopology generator_topo_;

    std::mt19937 rng_;
};

} // namespace loom
