#pragma once

#include "loom/core/backend.h"

#include <ggml-cpp.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace loom {

// Persistent per-layer convolution state -- the conv/SSM family's half of what `KvCache` is for
// attention, and the piece whose absence made a hybrid architecture prefill-only (BACKLOG.md P4.0.10,
// KV-CACHE.md "What stage 3 found", third bullet).
//
// Why this is a SECOND class rather than two more slots on KvCache: the two hold different shapes of
// history and are indexed differently. A KV slot is a GROWING prefix `[0, n_kv)` addressed by `n_past`,
// which is why its capacity is the context length. A conv slot is a FIXED-SIZE rolling window -- the
// last `kernel - 1` input columns, all a causal depthwise convolution needs to produce the next output
// -- so its capacity is a property of the kernel and never of the context. Sharing one class would mean
// one `kv_size` meaning two things.
//
// Same seam as KvCache, deliberately: storage in its own ggml_context outside the compute graph, so
// ggml-alloc never sees it; addressed by the `layer` attr on the node; writes returned as cpy nodes the
// caller routes through PrimitiveContext::side_effects; reads returned as plain views. A host that can
// allocate one can allocate the other.
class ConvStateCache {
public:
    // n_state is the per-channel history depth (`kernel - 1`); n_embd_conv is the channel count, i.e.
    // the width of one time step of the convolution's input.
    ConvStateCache(uint32_t n_layer, uint32_t n_state, uint32_t n_embd_conv, ggml_backend_t backend);

    // A view over this layer's whole stored window, shape [n_state, n_embd_conv] -- the `n_state`
    // columns that precede this step's first token.
    ggml_tensor* read(ggml_context* ctx, uint32_t layer) const;

    // A view over a slot that is zeroed at construction and that nothing ever writes -- the history a
    // step at n_past == 0 has. It exists as real backing storage rather than as a `ggml_scale(x, 0)`
    // trick because the compute context is no_alloc: a tensor created there holds whatever gallocr
    // hands it, and 0 * NaN is NaN. One extra [n_state, n_embd_conv] slot buys a single uniform code
    // path in op_short_conv, where prefill is the first iteration and not a special case.
    ggml_tensor* read_zeros(ggml_context* ctx) const;

    // Builds a ggml_cpy node writing `src` (shape [n_state, n_embd_conv]) into this layer's slot.
    // The returned tensor is the cpy op itself; callers MUST route it through
    // PrimitiveContext::side_effects. Unlike KvCache's writes this one is *also* ordered by a real data
    // dependency -- `src` is a view of the concatenated [state, x] buffer, which reads the slot -- so
    // the read cannot be scheduled after the write that clobbers it.
    ggml_tensor* write(ggml_context* ctx, ggml_tensor* src, uint32_t layer);

    uint32_t n_layer() const { return static_cast<uint32_t>(layers_.size()); }
    uint32_t n_state() const { return n_state_; }
    uint32_t n_embd_conv() const { return n_embd_conv_; }

    // Zeroes every slot. A step at n_past == 0 does not need this -- SHORT_CONV treats "no past" as
    // "no history" and never reads the slot there (see op_short_conv) -- but a host reusing one cache
    // across unrelated sequences gets the same guarantee KvCache::reset() gives.
    void reset();

    ConvStateCache(const ConvStateCache&) = delete;
    ConvStateCache& operator=(const ConvStateCache&) = delete;

private:
    ggml_context_ptr store_ctx_;
    ggml_backend_buffer_ptr store_buf_;
    std::vector<ggml_tensor*> layers_;
    ggml_tensor* zeros_ = nullptr;
    uint32_t n_state_;
    uint32_t n_embd_conv_;
};

class GgufModel;

// Builds a ConvStateCache sized entirely from `model`'s own declared hparams, exactly as
// `make_kv_cache` does for K/V, and for the same reason: a host must never need a per-model C++ struct
// to allocate one. Reads three facts from the "loom.*" namespace:
//
//   loom.n_conv_layer    number of stateful conv blocks (NOT model depth -- LFM2-350M declares
//                        num_hidden_layers 16 and has 10 conv blocks and 6 attention ones)
//   loom.n_conv_state    per-channel history depth, `kernel - 1`
//   loom.n_embd_conv     channel count
//
// Throws loom::LoadError naming the missing key if any is absent.
std::unique_ptr<ConvStateCache> make_conv_state_cache(const GgufModel& model, Backends backends);

} // namespace loom
