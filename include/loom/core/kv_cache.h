#pragma once

#include <ggml-cpp.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace loom {

// Persistent per-layer K/V storage -- the "Model Context"-adjacent half of the engine's two-context
// paradigm (SPECIFICATION.md §1): allocated once, entirely separate from the ephemeral per-step compute
// graph built by GraphBuilder.
//
// Deliberately simplified relative to llama.cpp's llama_kv_cache: single sequence, contiguous append (no
// ring buffer, no ggml_set_rows index-tensor indirection, no multi-stream/multi-sequence support). A new
// token's K/V always lands at the next free cell, so a plain ggml_view + ggml_cpy suffices for writes,
// and a plain ggml_view over [0, n_kv) suffices for reads. Sufficient for milestone 1's one-sequence
// autoregressive generation; see SPECIFICATION.md §8/the implementation plan for what generalizing back
// to multi-sequence would need.
class KvCache {
public:
    // n_embd_k / n_embd_v are the *flattened* per-token K/V widths (n_head_kv * n_embd_head_k/v, i.e.
    // what a token's K or V vector looks like collapsed across all KV heads) -- matches llama.cpp's
    // n_embd_k_gqa/n_embd_v_gqa naming. kv_size is the cache's total capacity in tokens (its "n_ctx").
    KvCache(uint32_t n_layer, uint32_t n_embd_k, uint32_t n_embd_v, uint32_t kv_size, ggml_backend_t backend);

    // Builds a ggml_cpy node writing `k_cur`/`v_cur` (each shape [n_embd_k/v, n_tokens]) into this
    // layer's cache at cells [n_past, n_past + n_tokens). The returned tensor is the cpy op itself --
    // callers MUST route it through PrimitiveContext::side_effects (or otherwise ensure it's expanded
    // into the graph) since nothing else in the graph references it via a data dependency.
    ggml_tensor* write_k(ggml_context* ctx, ggml_tensor* k_cur, uint32_t layer, uint32_t n_past, uint32_t n_tokens);
    ggml_tensor* write_v(ggml_context* ctx, ggml_tensor* v_cur, uint32_t layer, uint32_t n_past, uint32_t n_tokens);

    // Views over this layer's valid prefix [0, n_kv), shape [n_embd_k/v, n_kv].
    ggml_tensor* read_k(ggml_context* ctx, uint32_t layer, uint32_t n_kv) const;
    ggml_tensor* read_v(ggml_context* ctx, uint32_t layer, uint32_t n_kv) const;

    uint32_t n_layer() const { return static_cast<uint32_t>(k_layers_.size()); }
    uint32_t kv_size() const { return kv_size_; }

    // Zeroes the whole cache. Doesn't reset any "current length" bookkeeping -- that's the caller's
    // (Generator's) responsibility via its own n_past counter; KvCache itself is a dumb storage/view
    // provider with no notion of "how full" it currently is.
    void reset();

    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;

private:
    ggml_context_ptr store_ctx_;
    ggml_backend_buffer_ptr store_buf_;
    std::vector<ggml_tensor*> k_layers_;
    std::vector<ggml_tensor*> v_layers_;
    uint32_t n_embd_k_;
    uint32_t n_embd_v_;
    uint32_t kv_size_;
};

class GgufModel;

// Builds a KvCache sized entirely from `model`'s own declared hparams, so a host never has to carry a
// per-model C++ struct just to allocate one (KV-CACHE.md stage 1). Before this, the only two callers
// that needed a cache -- WhisperDriver and test_e2e_whisper_lua_driver.cpp -- both sized it from a
// hardcoded `WhisperConfig`, which is what made "the GGUF is self-contained" false for the one model
// on this roadmap that has a KV cache at all.
//
// Reads the five geometry facts from the "loom.*" hparam namespace the converters already write:
//
//   loom.n_layer         cache depth
//   loom.n_head_kv       KV heads (n_head_kv, NOT n_head -- GQA stores the un-repeated K/V)
//   loom.n_embd_head_k   per-head K width; the flat per-token width is n_head_kv * this
//   loom.n_embd_head_v   per-head V width, same
//   loom.kv_cache_size   capacity in tokens
//
// The first four already existed (convert_whisper_all.py has written them since the Lua port, with a
// comment saying they are for "KvCache sizing") and were simply never read; only the capacity is new.
// Deliberately NOT a second `loom.kv_cache.*` namespace duplicating them: two spellings of n_layer that
// can disagree is exactly the failure this project keeps removing elsewhere.
//
// Throws loom::LoadError naming the missing key if any of the five is absent -- a model whose topology
// reports `uses_kv_cache()` and whose file does not say how big the cache is cannot be run, and saying
// which key is missing is the difference between a fixable error and a mystery.
std::unique_ptr<KvCache> make_kv_cache(const GgufModel& model, ggml_backend_t backend);

} // namespace loom
