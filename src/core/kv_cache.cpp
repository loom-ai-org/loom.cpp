#include "loom/core/kv_cache.h"
#include "loom/loom_errors.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>

namespace loom {

KvCache::KvCache(uint32_t n_layer, uint32_t n_embd_k, uint32_t n_embd_v, uint32_t kv_size, ggml_backend_t backend)
    : n_embd_k_(n_embd_k), n_embd_v_(n_embd_v), kv_size_(kv_size) {
    const size_t mem_size = static_cast<size_t>(n_layer) * 2 * ggml_tensor_overhead() + 4096;
    store_ctx_.reset(ggml_init(ggml_init_params{mem_size, nullptr, /*no_alloc=*/true}));
    if (!store_ctx_) {
        throw Error("KvCache: ggml_init failed for the persistent K/V store context");
    }

    k_layers_.resize(n_layer);
    v_layers_.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        ggml_tensor* k = ggml_new_tensor_2d(store_ctx_.get(), GGML_TYPE_F32, n_embd_k, kv_size);
        ggml_format_name(k, "cache_k_l%d", il);
        k_layers_[il] = k;

        ggml_tensor* v = ggml_new_tensor_2d(store_ctx_.get(), GGML_TYPE_F32, n_embd_v, kv_size);
        ggml_format_name(v, "cache_v_l%d", il);
        v_layers_[il] = v;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(store_ctx_.get(), backend);
    if (!buf) {
        throw Error("KvCache: failed to allocate the persistent K/V backend buffer");
    }
    store_buf_.reset(buf);

    reset();
}

ggml_tensor* KvCache::write_k(ggml_context* ctx, ggml_tensor* k_cur, uint32_t layer, uint32_t n_past, uint32_t n_tokens) {
    ggml_tensor* base = k_layers_.at(layer);
    ggml_tensor* dst = ggml_view_2d(ctx, base, n_embd_k_, n_tokens, base->nb[1], static_cast<size_t>(n_past) * base->nb[1]);
    return ggml_cpy(ctx, k_cur, dst);
}

ggml_tensor* KvCache::write_v(ggml_context* ctx, ggml_tensor* v_cur, uint32_t layer, uint32_t n_past, uint32_t n_tokens) {
    ggml_tensor* base = v_layers_.at(layer);
    ggml_tensor* dst = ggml_view_2d(ctx, base, n_embd_v_, n_tokens, base->nb[1], static_cast<size_t>(n_past) * base->nb[1]);
    return ggml_cpy(ctx, v_cur, dst);
}

ggml_tensor* KvCache::read_k(ggml_context* ctx, uint32_t layer, uint32_t n_kv) const {
    ggml_tensor* base = k_layers_.at(layer);
    return ggml_view_2d(ctx, base, n_embd_k_, n_kv, base->nb[1], 0);
}

ggml_tensor* KvCache::read_v(ggml_context* ctx, uint32_t layer, uint32_t n_kv) const {
    ggml_tensor* base = v_layers_.at(layer);
    return ggml_view_2d(ctx, base, n_embd_v_, n_kv, base->nb[1], 0);
}

void KvCache::reset() {
    if (store_buf_) {
        ggml_backend_buffer_clear(store_buf_.get(), 0);
    }
}

} // namespace loom
