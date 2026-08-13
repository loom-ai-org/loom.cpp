#include "loom/core/kv_cache.h"

#include "loom/core/gguf_model.h"
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

ggml_tensor* KvCache::write_k(ggml_context* ctx, ggml_tensor* k_cur, uint32_t layer, ggml_tensor* cells) {
    return ggml_set_rows(ctx, k_layers_.at(layer), k_cur, cells);
}

ggml_tensor* KvCache::write_v(ggml_context* ctx, ggml_tensor* v_cur, uint32_t layer, ggml_tensor* cells) {
    return ggml_set_rows(ctx, v_layers_.at(layer), v_cur, cells);
}

ggml_tensor* KvCache::new_cell_index(ggml_context* ctx, uint32_t n_tokens) {
    // I64 rather than I32: ggml_set_rows accepts either, and llama_kv_cache's own index tensors are
    // I64, so matching it costs 4 bytes per token and removes a difference that would otherwise have to
    // be re-derived by anyone reading the two side by side.
    ggml_tensor* cells = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_name(cells, "kv_cells");
    ggml_set_input(cells);
    return cells;
}

void KvCache::fill_cell_index(ggml_tensor* cells, uint32_t n_past) {
    const auto n_tokens = static_cast<size_t>(cells->ne[0]);
    std::vector<int64_t> idx(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) idx[i] = static_cast<int64_t>(n_past) + static_cast<int64_t>(i);
    ggml_backend_tensor_set(cells, idx.data(), 0, idx.size() * sizeof(int64_t));
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

std::unique_ptr<KvCache> make_kv_cache(const GgufModel& model, Backends backends) {
    ggml_backend_t backend = backends.primary;
    // hparam_u32 already names the missing key, but not why anything wanted it. A host reaching this
    // has a topology that reports uses_kv_cache() and a file that does not say how big to make one,
    // and the fix is on the CONVERTER side -- so the message has to carry that, not just the key.
    const auto read = [&model](const char* key) {
        try {
            return model.hparam_u32(key);
        } catch (const LoadError&) {
            throw LoadError(std::string("make_kv_cache: this model's topology uses a KV cache, but the "
                                        "GGUF does not declare 'loom.") + key + "'. Every one of "
                                        "loom.{n_layer,n_head_kv,n_embd_head_k,n_embd_head_v,"
                                        "kv_cache_size} must be written by the converter for a cached "
                                        "model to be loadable without a per-model host struct.");
        }
    };

    const uint32_t n_layer  = read("n_layer");
    const uint32_t n_head_kv = read("n_head_kv");
    // The KvCache stores the FLATTENED per-token width (llama.cpp's n_embd_k_gqa), and stores K/V for
    // the un-repeated KV heads -- so this is n_head_kv, never n_head. Getting that wrong is silently
    // survivable (the cache is merely too large) in one direction and corrupting in the other, which is
    // why the derivation lives here once rather than at each call site.
    const uint32_t n_embd_k = n_head_kv * read("n_embd_head_k");
    const uint32_t n_embd_v = n_head_kv * read("n_embd_head_v");
    const uint32_t kv_size  = read("kv_cache_size");

    if (kv_size == 0) {
        throw LoadError("make_kv_cache: 'loom.kv_cache_size' is 0, so no token could ever be cached");
    }
    return std::make_unique<KvCache>(n_layer, n_embd_k, n_embd_v, kv_size, backend);
}

} // namespace loom
