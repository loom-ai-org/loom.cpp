#pragma once

// The hyperparameters the five BESPOKE TTS Lua driver tests pass INTO `infer(...)`.
//
// These used to come from the retired C++ drivers' own `VitsConfig`/`MatchaConfig`/... structs, which
// each test default-constructed purely to read a handful of fields off it (P4.0.8, E.3). A per-model
// C++ struct in a shipped public header is exactly what the lean-runtime argument objects to, so it did
// not survive the driver -- but the VALUES are still real checkpoint hyperparameters, so they landed
// here.
//
// **The MIL half is gone, which is what this header was really about.** P4.0.8's first follow-up moved
// every one of these onto the export side for the MIL drivers: a number the DRIVER needs is an
// `ExportConstants` IR local read off the checkpoint at export time, and a number the HOST needs is a
// `loom.*` GGUF hparam (`LoomExportConfig.hparams()`) -- Kokoro's `style_dim`, Supertonic's `txt_len`.
// So `test_e2e_*_mil_lua_driver.cpp` include none of this any more; the five files below do, and they
// drive `tools/convert_*/[a-z]*_driver.lua`, the hand-written pre-MIL drivers that P6 retires. This
// header goes with them.
//
// It is worth being precise about what that means for these five: a bespoke driver still takes these
// as arguments, so a value here that drifted from the checkpoint would silently change what the
// bespoke test computes and nothing would say so. That was true before too. What changed is that it no
// longer also affects the MIL tests, so the two halves of a family are no longer kept in step by this
// file -- they are kept in step by the frozen reference waveforms in
// fixtures/legacy_driver_reference/, which both halves compare against.
//
// Values are carried over verbatim from the structs they replace, comments included; each was confirmed
// against the real checkpoint when the driver was written (see tools/convert_*/PLAN.md and BACKLOG.md's
// dated entries for the derivations). Where the export now derives the same number from the real
// module, the two agree -- that agreement is what each family's MIL commit gated on.

#include <cstdint>

namespace loom_test::tts_inputs {

namespace vits { // was loom::VitsConfig
inline constexpr uint32_t hidden_channels = 192; // TextEncoder's own hidden width (also SDP's forced filter_channels)
inline constexpr uint32_t inter_channels = 192;  // flow/vocoder channel width (m_p/logs_p/z_p's own channel count)
inline constexpr uint32_t n_heads = 2;
inline constexpr uint32_t n_text_layers = 6;     // TextEncoder's attention-layer count (how many emb_rel_k/v tables exist)
inline constexpr uint32_t window_size = 4;       // attentions.Encoder's relative-position window
inline constexpr float noise_scale = 0.667f;     // z_p's own sampling noise (real default, models.py's infer())
inline constexpr float noise_scale_w = 0.8f;     // SDP's internal z_noise sampling (real default)
inline constexpr float length_scale = 1.0f;      // duration multiplier (real default)
} // namespace vits

namespace matcha { // was loom::MatchaConfig
inline constexpr uint32_t n_feats = 80;
inline constexpr float mel_mean = -5.536622f;
inline constexpr float mel_std = 2.116101f;
} // namespace matcha

namespace supertonic { // was loom::SupertonicConfig
// MUST match what convert_supertonic_all.py baked its DPTextEncoder / TTLTextEncoder /
// VectorFieldEstimator topologies for -- that BESPOKE conversion carries a fixed text length, so its
// `txt_ids` is exactly this long. See supertonic_driver.h's own "KNOWN SCOPE LIMITATION" note (retired
// with it).
//
// The MIL export has no such number and deliberately not a copy of one here: `supertonic_export.py`
// traces its text-touching graphs at SEVERAL padded widths and the driver runs the smallest that fits
// (BACKLOG.md P4.6a), so what a caller needs is a ceiling -- `model->hparam_u32("txt_len")`, read off
// the file (P4.0.8's first follow-up) -- and what a TEST needs is the width the driver would pick,
// which `support/supertonic_buckets.h` discovers from the model's own `topology_names()`.
inline constexpr uint32_t txt_len_fixed = 10;
inline constexpr uint32_t latent_dim = 144;
inline constexpr float sample_rate = 44100.0f;
inline constexpr uint32_t base_chunk_size = 512;
inline constexpr uint32_t compression_factor = 6; // latent_dim / compressed_dim (144/24)
} // namespace supertonic

namespace kokoro { // was loom::KokoroConfig
inline constexpr uint32_t style_dim = 128;      // EACH half of ref_s (256 total: decoder half + predictor half)
inline constexpr uint32_t d_model = 512;        // ProsodyPredictor/TextEncoder hidden width
inline constexpr uint32_t hidden_per_dir = 256; // every BiLSTM's per-direction hidden size
inline constexpr uint32_t harmonic_num = 8;     // SineGen: dim = harmonic_num+1
inline constexpr uint32_t upsample_scale = 300; // prod(upsample_rates)*gen_istft_hop_size
inline constexpr uint32_t gen_istft_n_fft = 20;
inline constexpr uint32_t gen_istft_hop = 5;
} // namespace kokoro

namespace styletts2 { // was loom::StyleTTS2Config
inline constexpr uint32_t style_dim = 128;
inline constexpr uint32_t d_model = 512;
inline constexpr uint32_t hidden_per_dir = 256;
inline constexpr uint32_t harmonic_num = 8;
inline constexpr uint32_t upsample_scale = 300;
inline constexpr uint32_t gen_istft_n_fft = 20;
inline constexpr uint32_t gen_istft_hop = 5;
// Style-diffusion sampler (KarrasSchedule + ADPM2Sampler + KDiffusion preconditioning).
inline constexpr float sigma_min = 1e-4f;
inline constexpr float sigma_max = 3.0f;
inline constexpr float rho = 9.0f;
inline constexpr float sigma_data = 0.45731624995853165f; // config.yml's model_params.diffusion.dist.sigma_data
} // namespace styletts2

} // namespace loom_test::tts_inputs
