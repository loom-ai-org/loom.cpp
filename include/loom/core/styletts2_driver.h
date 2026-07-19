#pragma once

#include "loom/core/bilstm_stepper.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_builder.h"
#include "loom/core/graph_topology.h"
#include "loom/core/style_diffusion_sampler.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace loom {

// Real hyperparameters confirmed against the real `yl4579/StyleTTS2-LJSpeech` checkpoint's own
// config.yml -- see tools/convert_styletts2/PLAN.md. Every decoder/prosody-predictor hyperparameter is
// BYTE-IDENTICAL to Kokoro's own KokoroConfig (StyleTTS2 and Kokoro share the same architecture there,
// confirmed directly, not assumed) -- the diffusion-specific fields (context_embedding_features,
// sigma_min/max/rho, sigma_data) are the only genuinely new ones.
struct StyleTTS2Config {
    uint32_t style_dim = 128;
    uint32_t d_model = 512;
    uint32_t hidden_per_dir = 256;
    uint32_t max_dur = 50;
    uint32_t harmonic_num = 8;
    float sampling_rate = 24000.0f;
    uint32_t upsample_scale = 300;
    uint32_t gen_istft_n_fft = 20;
    uint32_t gen_istft_hop = 5;
    float ln_eps = 1e-5f;

    // Style-diffusion sampler (KarrasSchedule + ADPM2Sampler + KDiffusion preconditioning wrapping the
    // real Transformer1d denoiser -- see style_diffusion_sampler.h and
    // tools/convert_styletts2/convert_styletts2_diffusion.py).
    uint32_t context_embedding_features = 768;  // PL-BERT hidden_size
    float sigma_min = 1e-4f;
    float sigma_max = 3.0f;
    float rho = 9.0f;
    float sigma_data = 0.45731624995853165f;  // config.yml's model_params.diffusion.dist.sigma_data
};

// Host-side driver for StyleTTS2 (yl4579/StyleTTS2-LJSpeech), tying together every topology
// tools/convert_styletts2/{convert_styletts2_reused.py,convert_styletts2_diffusion.py} produce from the
// real checkpoint. Real call order, confirmed directly against Demo/Inference_LJSpeech.ipynb's own
// `inference()` function (the actual real demo notebook, a stronger source of truth than Kokoro's own
// KModel.forward_with_tokens was):
//   CustomAlbert (raw bert_dur) -> style-diffusion sampler (ADPM2 over the real Transformer1d, conditioned
//   on raw bert_dur, NOT bert_encoder's projection) -> split s_pred into ref(decoder style)/s(predictor
//   style) -> bert_encoder -> predictor.text_encoder (DurationEncoder, 3x BiLSTM+AdaLayerNorm) ->
//   predictor.lstm (BiLSTM) -> duration_proj -> predict_durations (real quirk: pred_dur[-1] += 5) ->
//   expand_by_duration (applied to BOTH the 640-ch DurationEncoder output "d" and the 512-ch plain
//   TextEncoder output "t_en") -> F0Ntrain -> Decoder core -> SineGen (host rand_ini/noise draws) ->
//   forward STFT -> Generator core (host-precomputed wsum) -> waveform.
//
// Every one of these EXCEPT the style-diffusion sampler itself is architecturally identical to (and
// reuses the real weights/topologies produced by) Kokoro's own already-verified pieces -- see
// tools/convert_styletts2/convert_styletts2_reused.py, which imports Kokoro's own conversion scripts
// directly rather than duplicating ~2000 lines of already-verified code.
//
// Owns every GgufModel it loads (same "too many small GGUF files to hand to callers" rationale as
// KokoroDriver).
class StyleTTS2Driver {
public:
    // Loads every GGUF file this needs from `gguf_dir` (as produced by convert_styletts2_reused.py +
    // convert_styletts2_diffusion.py, both writing into the SAME directory). Throws loom::LoadError if
    // any expected file is missing.
    StyleTTS2Driver(const std::string& gguf_dir, StyleTTS2Config cfg, ggml_backend_t backend);

    // input_ids: phoneme/symbol ids (CustomAlbert's own vocabulary). Real StyleTTS2 wraps with a SINGLE
    // LEADING 0 token only (`tokens.insert(0, 0)` in the real demo -- NOT Kokoro's leading+trailing `[0,
    // ..., 0]` convention; callers' responsibility to pass the already-wrapped sequence). `diffusion_steps`
    // is the real ADPM2Sampler's own `num_steps` (the demo's own default is 5). `seed` seeds BOTH the
    // diffusion sampler's own noise (initial + every ancestral step) and SineGen's host-drawn
    // rand_ini/noise (the only two stochastic points in this whole pipeline at inference time).
    std::vector<float> synthesize(const std::vector<int32_t>& input_ids, uint32_t diffusion_steps, uint32_t seed);

private:
    StyleTTS2Config cfg_;
    ggml_backend_t backend_;

    std::unique_ptr<GgufModel> load(const std::string& gguf_dir, const std::string& filename);

    std::unique_ptr<GgufModel> albert_model_;
    GraphTopology albert_topo_;

    std::unique_ptr<GgufModel> diffusion_model_;
    GraphTopology diffusion_topo_;

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
    std::vector<DurationBlock> duration_blocks_;

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
