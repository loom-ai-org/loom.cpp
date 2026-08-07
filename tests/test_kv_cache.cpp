// Exercises KvCache directly: append-writes actually land the right bytes without corrupting earlier
// cells, layers are independent of each other, K and V storage don't cross-contaminate, and reset()
// zeroes everything.
//
// Reads use KvCache::read_k/v()'s views directly via ggml_backend_tensor_get with no graph compute step
// -- a ggml view's `data` pointer is valid the instant it's created (plain pointer arithmetic off its
// already-allocated base tensor), unlike write_k/v()'s ggml_set_rows node, which is a real compute op
// and must be run through an actual graph for the write to happen.
//
// Since BACKLOG.md P4.0.15 a write is addressed by a CELL-INDEX TENSOR rather than an n_past offset, so
// each helper below builds one -- and the "append" the tests are named for is now one particular
// filling of that tensor rather than the only thing the cache can do.

#include "ggml_test_helpers.h"
#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

using loom_test::GgmlScratch;
using loom_test::set_f32;

namespace {

void do_write_k(loom::KvCache& cache, ggml_backend_t backend, uint32_t layer, uint32_t n_past,
                 const std::vector<float>& data, uint32_t n_embd_k) {
    const uint32_t n_tokens = static_cast<uint32_t>(data.size() / n_embd_k);
    GgmlScratch s(backend);
    ggml_tensor* k_cur = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, n_embd_k, n_tokens);
    ggml_set_input(k_cur);
    ggml_tensor* cells = loom::KvCache::new_cell_index(s.ctx.get(), n_tokens);
    ggml_tensor* write_node = cache.write_k(s.ctx.get(), k_cur, layer, cells);
    ggml_cgraph* gf = s.expand(write_node);
    set_f32(k_cur, data);
    loom::KvCache::fill_cell_index(cells, n_past);
    s.compute(gf);
}

void do_write_v(loom::KvCache& cache, ggml_backend_t backend, uint32_t layer, uint32_t n_past,
                 const std::vector<float>& data, uint32_t n_embd_v) {
    const uint32_t n_tokens = static_cast<uint32_t>(data.size() / n_embd_v);
    GgmlScratch s(backend);
    ggml_tensor* v_cur = ggml_new_tensor_2d(s.ctx.get(), GGML_TYPE_F32, n_embd_v, n_tokens);
    ggml_set_input(v_cur);
    ggml_tensor* cells = loom::KvCache::new_cell_index(s.ctx.get(), n_tokens);
    ggml_tensor* write_node = cache.write_v(s.ctx.get(), v_cur, layer, cells);
    ggml_cgraph* gf = s.expand(write_node);
    set_f32(v_cur, data);
    loom::KvCache::fill_cell_index(cells, n_past);
    s.compute(gf);
}

std::vector<float> read_k_direct(loom::KvCache& cache, uint32_t layer, uint32_t n_kv, uint32_t n_embd_k) {
    ggml_context_ptr read_ctx(ggml_init(ggml_init_params{16 * 1024, nullptr, /*no_alloc=*/true}));
    ggml_tensor* view = cache.read_k(read_ctx.get(), layer, n_kv);
    std::vector<float> result(static_cast<size_t>(n_embd_k) * n_kv);
    ggml_backend_tensor_get(view, result.data(), 0, result.size() * sizeof(float));
    return result;
}

void test_append_does_not_corrupt_earlier_cells(ggml_backend_t backend) {
    constexpr uint32_t n_embd = 4;
    loom::KvCache cache(/*n_layer=*/2, n_embd, n_embd, /*kv_size=*/8, backend);

    do_write_k(cache, backend, 0, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, n_embd); // 3 tokens
    do_write_k(cache, backend, 0, 3, {13, 14, 15, 16, 17, 18, 19, 20}, n_embd);        // 2 more tokens

    const std::vector<float> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    LOOM_CHECK(read_k_direct(cache, 0, 5, n_embd) == expected);
}

void test_layers_are_independent(ggml_backend_t backend) {
    constexpr uint32_t n_embd = 4;
    loom::KvCache cache(2, n_embd, n_embd, 8, backend);
    do_write_k(cache, backend, 0, 0, {1, 1, 1, 1}, n_embd);
    do_write_k(cache, backend, 1, 0, {2, 2, 2, 2}, n_embd);

    LOOM_CHECK((read_k_direct(cache, 0, 1, n_embd) == std::vector<float>{1, 1, 1, 1}));
    LOOM_CHECK((read_k_direct(cache, 1, 1, n_embd) == std::vector<float>{2, 2, 2, 2}));
}

void test_k_and_v_do_not_cross_contaminate(ggml_backend_t backend) {
    constexpr uint32_t n_embd = 4;
    loom::KvCache cache(1, n_embd, n_embd, 4, backend);
    do_write_v(cache, backend, 0, 0, {9, 9, 9, 9}, n_embd);

    LOOM_CHECK((read_k_direct(cache, 0, 1, n_embd) == std::vector<float>{0, 0, 0, 0}));
}

void test_reset_zeroes_everything(ggml_backend_t backend) {
    constexpr uint32_t n_embd = 4;
    loom::KvCache cache(1, n_embd, n_embd, 4, backend);
    do_write_k(cache, backend, 0, 0, {1, 2, 3, 4}, n_embd);
    cache.reset();

    LOOM_CHECK((read_k_direct(cache, 0, 1, n_embd) == std::vector<float>{0, 0, 0, 0}));
}

} // namespace

int main() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    test_append_does_not_corrupt_earlier_cells(backend.get());
    test_layers_are_independent(backend.get());
    test_k_and_v_do_not_cross_contaminate(backend.get());
    test_reset_zeroes_everything(backend.get());

    LOOM_TEST_REPORT_AND_RETURN();
}
