#pragma once

#include "loom/core/backend.h"

#include <ggml-cpp.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace loom {

// Persistent per-layer K/V storage -- the "Model Context"-adjacent half of the engine's two-context
// paradigm (SPECIFICATION.md §1): allocated once, entirely separate from the ephemeral per-step compute
// graph built by GraphBuilder.
//
// Still deliberately simplified relative to llama.cpp's llama_kv_cache -- single sequence, no ring
// buffer, no multi-stream/multi-sequence support -- but no longer simplified in the one way that made a
// decode loop rebuild its graph every step. **Writes go through ggml_set_rows and a cell-index tensor**
// (BACKLOG.md P4.0.15), the indirection this comment used to name as absent: the destination cells are
// a runtime *value* rather than a byte offset baked into a ggml_view at build time, so step N and step
// N+1 produce the identical graph. Reads are still a plain ggml_view over [0, n_kv), which is all a
// contiguous single-sequence cache needs. See SPECIFICATION.md §8/the implementation plan for what
// generalizing back to multi-sequence would need -- the index tensor is the half of it that now exists.
class KvCache {
public:
    // n_embd_k / n_embd_v are the *flattened* per-token K/V widths (n_head_kv * n_embd_head_k/v, i.e.
    // what a token's K or V vector looks like collapsed across all KV heads) -- matches llama.cpp's
    // n_embd_k_gqa/n_embd_v_gqa naming. kv_size is the cache's total capacity in tokens (its "n_ctx").
    KvCache(uint32_t n_layer, uint32_t n_embd_k, uint32_t n_embd_v, uint32_t kv_size, ggml_backend_t backend);

    // Builds a ggml_set_rows node writing `k_cur`/`v_cur` (each shape [n_embd_k/v, n_tokens]) into this
    // layer's cache at the cells named by `cells` (an I64 [n_tokens] tensor -- see new_cell_index). The
    // returned tensor is the set_rows op itself -- callers MUST route it through
    // PrimitiveContext::side_effects (or otherwise ensure it's expanded into the graph) since nothing
    // else in the graph references it via a data dependency.
    //
    // `cells` carries what used to be the `n_past` parameter, and that is the entire point: as a tensor
    // it is data the host rewrites between steps, not structure the graph is rebuilt around.
    ggml_tensor* write_k(ggml_context* ctx, ggml_tensor* k_cur, uint32_t layer, ggml_tensor* cells);
    ggml_tensor* write_v(ggml_context* ctx, ggml_tensor* v_cur, uint32_t layer, ggml_tensor* cells);

    // The cell-index tensor write_k/write_v expect: I64, one entry per token being written. Created here
    // rather than at the call site so the index dtype ggml_set_rows accepts is stated in exactly one
    // place. Allocate it OUTSIDE the compute graph's gallocr pool (GraphBuilder puts it in the same
    // persistent buffer as the declared inputs) -- a graph that is reused must be able to have its cell
    // indices rewritten without being rebuilt, which is only true of memory the allocator never moves.
    static ggml_tensor* new_cell_index(ggml_context* ctx, uint32_t n_tokens);

    // Fills an already-allocated `cells` with the contiguous append [n_past, n_past + n_tokens), the one
    // pattern this single-sequence cache ever writes. A ring buffer or a multi-sequence scheduler would
    // add a second filler here and change nothing else.
    static void fill_cell_index(ggml_tensor* cells, uint32_t n_past);

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
// that needed a cache -- the since-retired WhisperDriver and its Lua test -- both sized it from a
// hardcoded `WhisperConfig`, which is what made "the GGUF is self-contained" false for the one model
// on this roadmap that had a KV cache at all.
//
// Reads the five geometry facts from the "loom.*" hparam namespace the converters already write:
//
//   loom.n_layer         cache depth
//   loom.n_head_kv       KV heads (n_head_kv, NOT n_head -- GQA stores the un-repeated K/V)
//   loom.n_embd_head_k   per-head K width; the flat per-token width is n_head_kv * this
//   loom.n_embd_head_v   per-head V width, same
//   loom.kv_cache_size   capacity in tokens
//
// The first four already existed (the bespoke Whisper converter wrote them from the Lua port onwards,
// with a comment saying they are for "KvCache sizing") and were simply never read; only the capacity
// was new. Both are now written by the exporter itself, off the fused ATTENTION nodes.
// Deliberately NOT a second `loom.kv_cache.*` namespace duplicating them: two spellings of n_layer that
// can disagree is exactly the failure this project keeps removing elsewhere.
//
// Throws loom::LoadError naming the missing key if any of the five is absent -- a model whose topology
// reports `uses_kv_cache()` and whose file does not say how big the cache is cannot be run, and saying
// which key is missing is the difference between a fixable error and a mystery.
std::unique_ptr<KvCache> make_kv_cache(const GgufModel& model, Backends backends);

} // namespace loom
