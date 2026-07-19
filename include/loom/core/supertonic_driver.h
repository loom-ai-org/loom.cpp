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

// Real hyperparameters confirmed against the real `femelo/supertonic-tts` checkpoint -- see
// tools/convert_supertonic/PLAN.md and convert_supertonic_all.py's own module docstring.
//
// KNOWN SCOPE LIMITATION (real, confirmed, not an oversight): `txt_len_fixed` MUST match exactly what
// `convert_supertonic_all.py` baked its DPTextEncoder/TTLTextEncoder/VectorFieldEstimator topologies for
// (their own relative-position tables AND VectorFieldEstimator's text cross-attention length are FIXED
// at conversion time) -- `loom::GraphBuilder` only supports ONE dynamic-length symbol ("$n_tokens") per
// graph, and SupertonicTTS's VectorFieldEstimator genuinely needs TWO independently-varying lengths in
// one graph (the CFM-iterated latent length, bound to "$n_tokens"; and the input utterance's own
// phoneme count) -- the first model in this project needing that. `synthesize()`'s `txt_ids` MUST be
// exactly `txt_len_fixed` long.
struct SupertonicConfig {
    uint32_t txt_len_fixed = 10;
    uint32_t crop_len = 50;
    uint32_t latent_dim = 144;
    uint32_t style_dim_ttl = 256;
    uint32_t n_style_ttl = 50;
    uint32_t style_dim_dp = 16;
    uint32_t n_style_dp = 8;
    float sample_rate = 44100.0f;
    uint32_t base_chunk_size = 512;
    uint32_t compression_factor = 6;  // latent_dim / compressed_dim (144/24)
};

// Host-side driver for SupertonicTTS v2 (femelo/supertonic-tts), tying together every topology
// tools/convert_supertonic/convert_supertonic_all.py produces from the real checkpoint. Real call order,
// confirmed directly against models/speech_generator.py's own `SpeechGenerator.predict()`:
//   DurationPredictor (DPTextEncoder + MLP head, precomputed DP style embedding) -> scalar duration ->
//   get_latent_mask (host: duration*sample_rate -> latent-frame count T_lat) -> TTLTextEncoder
//   (precomputed TTL style embedding) -> deterministic Euler CFM sampling loop (VectorFieldEstimator,
//   n_steps) -> SpeechDecoder -> waveform.
//
// Owns every GgufModel it loads (same "too many small pieces to hand to callers" rationale as
// KokoroDriver/StyleTTS2Driver).
class SupertonicDriver {
public:
    SupertonicDriver(const std::string& gguf_dir, SupertonicConfig cfg, ggml_backend_t backend);

    // `txt_ids`: phoneme/symbol ids (real `TextVectorizer`'s own vocabulary), length MUST equal
    // `cfg.txt_len_fixed` (see SupertonicConfig's own docstring). `style_ttl`: `n_style_ttl *
    // style_dim_ttl` floats (Layout B: style-index-major, channel-minor -- matches a real voice-style
    // JSON's own `style_ttl` field's raw (1,50,256) row-major data directly, no reordering). `style_dp`:
    // `n_style_dp * style_dim_dp` floats, same convention, matches `style_dp`'s own (1,8,16) data.
    // `n_steps`: the real ADPM2-equivalent... no, CFM Euler sampler's own step count (real demo default
    // is 10). `seed` seeds the ONLY stochastic point in this whole pipeline (the initial CFM noise z_0).
    std::vector<float> synthesize(const std::vector<int32_t>& txt_ids, const std::vector<float>& style_ttl,
                                   const std::vector<float>& style_dp, uint32_t n_steps, uint32_t seed);

private:
    SupertonicConfig cfg_;
    ggml_backend_t backend_;

    std::unique_ptr<GgufModel> load(const std::string& gguf_dir, const std::string& filename);

    // NOTE: `supertonic_dp_style.gguf`/`supertonic_ttl_style.gguf` (the style encoders, taking a raw
    // compressed-latent crop) are produced by the conversion script and independently verified (tasks
    // #100/#102/#103), but NOT loaded here -- `synthesize()` takes PRECOMPUTED style embeddings directly
    // (matching Kokoro's/StyleTTS2's own established "skip the reference-audio style encoder, use a
    // precomputed voice style" scope decision -- real `assets/voice_styles/*.json` already ship
    // precomputed `style_ttl`/`style_dp` fields).
    std::unique_ptr<GgufModel> dp_model_;
    GraphTopology dp_topo_;
    std::unique_ptr<GgufModel> ttl_text_model_;
    GraphTopology ttl_text_topo_;
    std::unique_ptr<GgufModel> vfe_model_;
    GraphTopology vfe_topo_;
    std::unique_ptr<GgufModel> decoder_model_;
    GraphTopology decoder_topo_;

    std::mt19937 rng_;
};

} // namespace loom
