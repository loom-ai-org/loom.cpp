// One model cache, several streams (ADR-099): a lease parks the owner's rows when another stream takes
// the cache and brings them back when it returns, one stream alone never copies, a dropped stream
// leaves nothing to park into, and `StreamCaches` resolves per backend.
//
// Rows are written with KvCache::set_rows, the host write that also feeds `rows_used` -- the extent a
// park copies -- and read back through read_k/read_v views, whose data pointers are valid as soon as
// they exist (test_kv_cache.cpp says why no graph is needed for a read).

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

namespace {

constexpr uint32_t kEmbd = 4;

std::vector<float> rows_of(float base, uint32_t n_rows) {
    std::vector<float> out(static_cast<size_t>(n_rows) * kEmbd);
    for (size_t i = 0; i < out.size(); ++i) out[i] = base + static_cast<float>(i);
    return out;
}

void write(loom::KvCache& cache, float base, uint32_t n_rows) {
    for (uint32_t layer = 0; layer < cache.n_layer(); ++layer) {
        cache.set_rows(layer, false, rows_of(base + 100.0f * layer, n_rows).data(), 0, n_rows);
        cache.set_rows(layer, true, rows_of(base + 100.0f * layer + 50.0f, n_rows).data(), 0, n_rows);
    }
}

bool holds(loom::KvCache& cache, float base, uint32_t n_rows) {
    ggml_context_ptr ctx(ggml_init(ggml_init_params{64 * 1024, nullptr, /*no_alloc=*/true}));
    for (uint32_t layer = 0; layer < cache.n_layer(); ++layer) {
        for (bool value : {false, true}) {
            ggml_tensor* view = value ? cache.read_v(ctx.get(), layer, n_rows) : cache.read_k(ctx.get(), layer, n_rows);
            std::vector<float> got(static_cast<size_t>(n_rows) * kEmbd);
            ggml_backend_tensor_get(view, got.data(), 0, got.size() * sizeof(float));
            if (got != rows_of(base + 100.0f * layer + (value ? 50.0f : 0.0f), n_rows)) return false;
        }
    }
    return true;
}

void test_two_streams_take_turns(ggml_backend_t backend) {
    loom::KvCache cache(/*n_layer=*/2, kEmbd, kEmbd, /*kv_size=*/8, backend);
    loom::StreamState a, b;

    cache.lease(a.parked(cache.serial()));
    write(cache, 1.0f, 3);
    LOOM_CHECK(cache.rows_used() == 3);

    cache.lease(b.parked(cache.serial()));
    LOOM_CHECK(cache.parks() == 1);
    LOOM_CHECK(cache.rows_used() == 0); // b never wrote: nothing of a's is b's extent
    LOOM_CHECK(a.parked_bytes() == 3 * 2 * 2 * kEmbd * sizeof(float));
    write(cache, 1000.0f, 2);

    cache.lease(a.parked(cache.serial()));
    LOOM_CHECK(cache.parks() == 2);
    LOOM_CHECK(cache.rows_used() == 3);
    LOOM_CHECK(holds(cache, 1.0f, 3));
    LOOM_CHECK(a.parked_bytes() == 0); // restored, so not held twice

    cache.lease(b.parked(cache.serial()));
    LOOM_CHECK(holds(cache, 1000.0f, 2));
}

void test_one_stream_never_copies(ggml_backend_t backend) {
    loom::KvCache cache(2, kEmbd, kEmbd, 8, backend);
    loom::StreamState a;
    for (uint32_t step = 1; step <= 4; ++step) {
        cache.lease(a.parked(cache.serial()));
        write(cache, 1.0f, step);
    }
    LOOM_CHECK(cache.parks() == 0);
    LOOM_CHECK(holds(cache, 1.0f, 4));
}

void test_a_dropped_stream_is_not_parked(ggml_backend_t backend) {
    loom::KvCache cache(2, kEmbd, kEmbd, 8, backend);
    {
        loom::StreamState gone;
        cache.lease(gone.parked(cache.serial()));
        write(cache, 1.0f, 3);
    }
    loom::StreamState b;
    cache.lease(b.parked(cache.serial()));
    LOOM_CHECK(cache.parks() == 0);
    LOOM_CHECK(cache.rows_used() == 0);
}

void test_conv_state_round_trip(ggml_backend_t backend) {
    loom::ConvStateCache cache(/*n_layer=*/2, /*n_state=*/2, /*n_embd_conv=*/3, backend);
    ggml_context_ptr ctx(ggml_init(ggml_init_params{64 * 1024, nullptr, /*no_alloc=*/true}));
    const auto put = [&](float base) {
        for (uint32_t layer = 0; layer < 2; ++layer) {
            std::vector<float> v(6);
            for (size_t i = 0; i < v.size(); ++i) v[i] = base + 10.0f * layer + static_cast<float>(i);
            ggml_backend_tensor_set(cache.read(ctx.get(), layer), v.data(), 0, v.size() * sizeof(float));
        }
    };
    const auto has = [&](float base) {
        for (uint32_t layer = 0; layer < 2; ++layer) {
            std::vector<float> v(6), want(6);
            for (size_t i = 0; i < want.size(); ++i) want[i] = base + 10.0f * layer + static_cast<float>(i);
            ggml_backend_tensor_get(cache.read(ctx.get(), layer), v.data(), 0, v.size() * sizeof(float));
            if (v != want) return false;
        }
        return true;
    };
    loom::StreamState a, b;
    cache.lease(a.parked(cache.serial()));
    put(1.0f);
    cache.lease(b.parked(cache.serial()));
    put(500.0f);
    cache.lease(a.parked(cache.serial()));
    LOOM_CHECK(has(1.0f));
    cache.lease(b.parked(cache.serial()));
    LOOM_CHECK(has(500.0f));
}

void test_own_caches(ggml_backend_t backend) {
    loom::KvCache shared(2, kEmbd, kEmbd, 8, backend, /*ring=*/true);
    loom::StreamState a;
    loom::KvCache& own = a.own_cache(shared);
    LOOM_CHECK(&own == &a.own_cache(shared)); // allocated once
    LOOM_CHECK(own.serial() != shared.serial());
    LOOM_CHECK(own.n_layer() == 2 && own.kv_size() == 8 && own.ring());
    LOOM_CHECK(a.cache_bytes() == 2 * (kEmbd + kEmbd) * 8 * sizeof(float));
}

void test_mode_resolution(ggml_backend_t backend) {
    LOOM_CHECK(!loom::stream_owns_cache(loom::StreamCaches::Park, backend));
    LOOM_CHECK(loom::stream_owns_cache(loom::StreamCaches::PerStream, backend));
    LOOM_CHECK(!loom::stream_owns_cache(loom::StreamCaches::Auto, backend)); // the CPU parks
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    test_two_streams_take_turns(backend.get());
    test_one_stream_never_copies(backend.get());
    test_a_dropped_stream_is_not_parked(backend.get());
    test_conv_state_round_trip(backend.get());
    test_own_caches(backend.get());
    test_mode_resolution(backend.get());

    LOOM_TEST_REPORT_AND_RETURN();
}
