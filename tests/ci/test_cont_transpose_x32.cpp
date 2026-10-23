// ggml-0046 (P4.36): a materialised transpose of 4-byte elements -- `ggml_cont` of a view whose second
// dimension is the contiguous one -- copies in 4x4 tiles instead of element by element. A copy has one
// right answer, so this checks it byte for byte against an index-by-index reference: whisper's V^T and a
// set of ragged shapes, each at one to four threads (the threads split the second dimension, so a
// thread's row range starts and ends off a multiple of four), plus a 4-D view with `ne03 > 1`.
//
// What a wrong tile looks like here: swapped lanes, a row remainder skipped (the output buffer is
// pre-filled with a pattern that no source holds), a block boundary off by one, or the wrong head's
// plane. Each of those is a byte mismatch; none of them is a tolerance question.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool compute(ggml_backend_t backend, ggml_cgraph* gf, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

// value_states [D, H, T, B] -> permute to [T, D, H, B] -> cont: the chain whisper's export emits for V^T
// (two permutes; ggml composes them into one view with nb0 = D*H*4 and nb1 = 4).
void check(ggml_backend_t backend, int64_t D, int64_t H, int64_t T, int64_t Bt) {
    ggml_init_params ip = { (size_t) D * H * T * Bt * 4 * 2 + (size_t) 4 * 1024 * 1024, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, H, T, Bt);
    uint32_t* pv = (uint32_t*) v->data;
    for (int64_t i = 0; i < ggml_nelements(v); ++i) pv[i] = (uint32_t) (i * 2654435761u) | 1u;   // never 0xffffffff
    ggml_tensor* p = ggml_permute(ctx, ggml_permute(ctx, v, 0, 2, 1, 3), 1, 0, 2, 3);
    ggml_tensor* c = ggml_cont(ctx, p);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    std::vector<uint32_t> ref((size_t) ggml_nelements(v));
    for (int64_t b = 0; b < Bt; ++b)
        for (int64_t h = 0; h < H; ++h)
            for (int64_t d = 0; d < D; ++d)
                for (int64_t t = 0; t < T; ++t)
                    ref[((b * H + h) * D + d) * T + t] = pv[((b * T + t) * H + h) * D + d];

    for (int threads = 1; threads <= 4; ++threads) {
        std::memset(c->data, 0xff, ggml_nbytes(c));
        LOOM_CHECK(compute(backend, gf, threads));
        const bool same = std::memcmp(c->data, ref.data(), ggml_nbytes(c)) == 0;
        if (!same) std::printf("D=%lld H=%lld T=%lld B=%lld threads=%d: transpose differs from the reference\n",
                               (long long) D, (long long) H, (long long) T, (long long) Bt, threads);
        LOOM_CHECK(same);
    }
    ggml_free(ctx);
}

} // namespace

int main() {
    ggml_backend_t backend = loom_test::cpu_backend();
    LOOM_CHECK(backend != nullptr);
    if (backend == nullptr) LOOM_TEST_REPORT_AND_RETURN();
    check(backend, 64, 12, 1500, 1);   // whisper's V^T
    check(backend, 64, 12, 7, 1);      // T below one 4-wide tile
    check(backend, 5, 3, 9, 1);        // rows not a multiple of 4 at any thread count
    check(backend, 32, 12, 130, 1);    // T across two 64-blocks plus a remainder
    check(backend, 6, 2, 67, 3);       // ne03 > 1
    check(backend, 64, 1, 1, 1);       // T = 1
    LOOM_TEST_REPORT_AND_RETURN();
}
