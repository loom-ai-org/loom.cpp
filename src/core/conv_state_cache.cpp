#include "loom/core/conv_state_cache.h"

#include "loom/core/gguf_model.h"
#include "loom/loom_errors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>

namespace loom {

ConvStateCache::ConvStateCache(uint32_t n_layer, uint32_t n_state, uint32_t n_embd_conv, ggml_backend_t backend)
    : n_state_(n_state), n_embd_conv_(n_embd_conv) {
    // n_layer slots plus the one permanently-zero slot read_zeros() hands out.
    const size_t mem_size = static_cast<size_t>(n_layer + 1) * ggml_tensor_overhead() + 4096;
    store_ctx_.reset(ggml_init(ggml_init_params{mem_size, nullptr, /*no_alloc=*/true}));
    if (!store_ctx_) {
        throw Error("ConvStateCache: ggml_init failed for the persistent conv-state store context");
    }

    layers_.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        // [n_state, n_embd_conv] matches the layout SHORT_CONV's input arrives in: ne[0] is time,
        // ne[1] is channels (LFM2's `Bx` is [n_tokens, 1024]), so a slot is a contiguous prefix of the
        // same shape and the concat below is along ne[0] with no transposes anywhere.
        ggml_tensor* s = ggml_new_tensor_2d(store_ctx_.get(), GGML_TYPE_F32, n_state, n_embd_conv);
        ggml_format_name(s, "conv_state_l%d", il);
        layers_[il] = s;
    }
    zeros_ = ggml_new_tensor_2d(store_ctx_.get(), GGML_TYPE_F32, n_state, n_embd_conv);
    ggml_set_name(zeros_, "conv_state_zeros");

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(store_ctx_.get(), backend);
    if (!buf) {
        throw Error("ConvStateCache: failed to allocate the persistent conv-state backend buffer");
    }
    store_buf_.reset(buf);

    reset();
}

ggml_tensor* ConvStateCache::read(ggml_context* ctx, uint32_t layer) const {
    ggml_tensor* base = layers_.at(layer);
    return ggml_view_2d(ctx, base, n_state_, n_embd_conv_, base->nb[1], 0);
}

ggml_tensor* ConvStateCache::read_zeros(ggml_context* ctx) const {
    return ggml_view_2d(ctx, zeros_, n_state_, n_embd_conv_, zeros_->nb[1], 0);
}

ggml_tensor* ConvStateCache::write(ggml_context* ctx, ggml_tensor* src, uint32_t layer) {
    ggml_tensor* base = layers_.at(layer);
    ggml_tensor* dst = ggml_view_2d(ctx, base, n_state_, n_embd_conv_, base->nb[1], 0);
    return ggml_cpy(ctx, src, dst);
}

void ConvStateCache::reset() {
    if (store_buf_) {
        ggml_backend_buffer_clear(store_buf_.get(), 0);
    }
}

std::unique_ptr<ConvStateCache> make_conv_state_cache(const GgufModel& model, Backends backends) {
    ggml_backend_t backend = backends.primary;
    // Same shape of message as make_kv_cache's: a host reaching this has a topology containing a
    // SHORT_CONV node and a file that does not say how big its state is, and the fix is on the
    // converter side rather than in the host.
    const auto read = [&model](const char* key) {
        try {
            return model.hparam_u32(key);
        } catch (const LoadError&) {
            throw LoadError(std::string("make_conv_state_cache: this model's topology has stateful "
                                        "convolutions, but the GGUF does not declare 'loom.") + key +
                                        "'. Every one of loom.{n_conv_layer,n_conv_state,n_embd_conv} "
                                        "must be written by the converter for a hybrid model to decode "
                                        "incrementally without a per-model host struct.");
        }
    };

    const uint32_t n_layer = read("n_conv_layer");
    const uint32_t n_state = read("n_conv_state");
    const uint32_t n_embd_conv = read("n_embd_conv");

    if (n_state == 0) {
        throw LoadError("make_conv_state_cache: 'loom.n_conv_state' is 0, so a convolution over it "
                        "would carry no history at all -- a kernel of width 1 is position-wise and "
                        "should not have been emitted as SHORT_CONV.");
    }
    return std::make_unique<ConvStateCache>(n_layer, n_state, n_embd_conv, backend);
}

} // namespace loom
