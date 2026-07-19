#pragma once

#include "loom/core/graph_builder.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"
#include "loom/core/kv_cache.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace loom {

struct WhisperConfig {
    uint32_t n_audio_state = 384;
    uint32_t n_audio_ctx = 1500;   // fixed: every Whisper checkpoint always processes exactly 30s of audio
    uint32_t n_text_state = 384;
    uint32_t n_text_head = 6;
    uint32_t n_text_layer = 4;
    uint32_t n_text_ctx = 448;     // KvCache capacity (also the real model's own max decode length)
    int32_t eot_token = -1;        // stop early when sampled; negative disables the check
};

// Host-side driver for Whisper (OpenAI) speech-to-text inference, tying together the two GGUF files
// tools/convert_whisper/{convert_whisper_encoder,convert_whisper_decoder}.py produce -- same multi-file
// reasoning as VitsDriver (GraphTopology supports exactly one declared output per topology).
//
// Two-phase, but simpler than VitsDriver's: the encoder runs exactly ONCE per call (fixed 30s input, no
// duration-dependent sizing question at all -- unlike VITS's `generate_path`, nothing here determines a
// downstream shape at runtime), producing `xa` (channel-first, ne=[n_audio_state,n_audio_ctx]), which is
// then fed UNCHANGED as a per-step cross-attention input to every decoder step. The decoder loop itself
// mirrors `Generator::generate`'s own prefill-then-decode-one-at-a-time structure (same persistent
// `KvCache` for causal self-attention, same "tokens"/"positions"/"kq_mask" input convention) -- extended
// with the two additional per-step inputs cross-attention needs, "xa" and "xa_mask" (an all-zero mask,
// since cross-attention has no causal/padding structure at all).
//
// Cross-attention K/V are recomputed from "xa" on EVERY decode step (not cached across steps the way the
// real PyTorch model's `install_kv_cache_hooks` optimizes) -- correct, just not maximally efficient; a
// documented future optimization, not attempted here (see convert_whisper_decoder.py's own note).
class WhisperDriver {
public:
    WhisperDriver(GgufModel& encoder_model, GraphTopology encoder_topo, GgufModel& decoder_model,
                  GraphTopology decoder_topo, WhisperConfig cfg, ggml_backend_t backend);

    // waveform_padded: the HOST-reflect-padded, pad_or_trim'd-to-30s waveform (whisper_common.py's own
    // convention -- length n_samples + 2*reflect_pad, matching the encoder topology's declared
    // "waveform" input exactly). prompt_tokens: initial decode context (e.g. Whisper's own
    // SOT/language/task/notimestamps special-token sequence) -- greedy-decodes one token at a time from
    // there until either max_new_tokens have been generated or eot_token is sampled. Returns just the
    // generated tokens (not the prompt), same convention as Generator::generate.
    std::vector<int32_t> transcribe(const std::vector<float>& waveform_padded,
                                     const std::vector<int32_t>& prompt_tokens, uint32_t max_new_tokens);

private:
    void fill_decoder_inputs(GraphBuilder::BuildResult& r, const std::vector<int32_t>& step_tokens,
                              uint32_t n_past, const std::vector<float>& xa);
    static int32_t argmax(const float* row, uint32_t n);

    GgufModel& encoder_model_;
    GgufModel& decoder_model_;
    WhisperConfig cfg_;
    ggml_backend_t backend_;

    // Declaration order matters: each GraphBuilder stores a reference to its corresponding
    // GraphTopology, so the topologies must be fully constructed first (same precedent as VitsDriver).
    GraphTopology encoder_topo_;
    GraphTopology decoder_topo_;
    KvCache kv_cache_;
    std::unique_ptr<GraphBuilder> encoder_builder_; // no kv_cache: one fixed-shape non-autoregressive pass
    std::unique_ptr<GraphBuilder> decoder_builder_; // wired to &kv_cache_
};

} // namespace loom
