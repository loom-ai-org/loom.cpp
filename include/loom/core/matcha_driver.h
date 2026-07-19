#pragma once

#include "loom/core/cfm_euler_sampler.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_builder.h"
#include "loom/core/graph_topology.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace loom {

// Real hyperparameters confirmed against the real `matcha_ljspeech.ckpt` (see
// tools/convert_matcha/PLAN.md and BACKLOG.md's 2026-07-19 entry).
//
// KNOWN SCOPE LIMITATION (real, confirmed, not an oversight): the Decoder U-Net topology
// (matcha_decoder.gguf) drops ALL padding-mask handling (every real `x*mask` multiply is a no-op),
// which is only valid when the mel-frame count is an EXACT MULTIPLE OF 4 (real `fix_len_compatibility`'s
// own default `num_downsamplings_in_unet=2` requirement) and every frame is "real" (no padding).
// `synthesize()` enforces this by EXTENDING the last token's predicted duration (not by masking/padding
// with mask=0, which the topology can't express) until the total frame count is a multiple of 4 -- a
// principled approximation (every frame remains a genuine attended `mu_y` row, never contaminated
// padding) rather than a hack, but still a real, documented scope choice.
struct MatchaConfig {
    uint32_t n_feats = 80;
    float mel_mean = -5.536622f;
    float mel_std = 2.116101f;
    uint32_t hop_size = 256;  // vocoder's real cumulative upsample product (8*8*2*2)
};

// Host-side driver for Matcha-TTS, tying together every topology tools/convert_matcha/convert_matcha_*.py
// produces from the real checkpoint. Real call order, confirmed directly against
// `matcha/models/matcha_tts.py`'s own `MatchaTTS.synthesise()`:
//   TextEncoder (mu_x, logw) -> per-token durations (ceil(exp(logw))) -> row-repeat duration expansion
//   (mu_x -> mu_y, degenerate form of real `generate_path`, see BACKLOG.md) -> deterministic Euler CFM
//   sampling loop (Decoder U-Net estimator, n_steps) -> denormalize (mel_mean/mel_std) -> HiFi-GAN v1
//   vocoder -> waveform.
class MatchaDriver {
public:
    MatchaDriver(const std::string& gguf_dir, MatchaConfig cfg, ggml_backend_t backend);

    // `tokens`: phoneme/symbol ids (n_vocab=178, real Matcha-TTS vocabulary). `n_steps`: the CFM Euler
    // sampler's own step count (real demo default is 10). `seed` seeds the ONLY stochastic point in this
    // whole pipeline (the initial CFM noise z_0, `temperature=1.0`).
    std::vector<float> synthesize(const std::vector<int32_t>& tokens, uint32_t n_steps, uint32_t seed);

private:
    MatchaConfig cfg_;
    ggml_backend_t backend_;

    std::unique_ptr<GgufModel> load(const std::string& gguf_dir, const std::string& filename);

    std::unique_ptr<GgufModel> encoder_mu_model_;
    GraphTopology encoder_mu_topo_;
    std::unique_ptr<GgufModel> encoder_logw_model_;
    GraphTopology encoder_logw_topo_;
    std::unique_ptr<GgufModel> decoder_model_;
    GraphTopology decoder_topo_;
    std::unique_ptr<GgufModel> vocoder_model_;
    GraphTopology vocoder_topo_;

    std::mt19937 rng_;
};

} // namespace loom
