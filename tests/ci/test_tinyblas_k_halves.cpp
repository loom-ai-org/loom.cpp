// ggml-0045 (P4.35): an F32 GEMM whose K lies between one cache chunk and two is cut in two halves when
// threads share the L2 and the tile's operand strips spill L1 -- and must still be the SAME BYTES.
//
// The chunked path carries the tile's vector accumulators across the halves and reduces once at the
// end (ggml-0030's construction), so a chunked output is bit-identical to an unchunked one. That is
// the claim pinned here, with the thread count as the switch: at one thread the rule never chunks, at
// two and four it does on aarch64 with a 32 KB L1 (k > 1024 at the 4x3 tile, k < 1536). So every
// shape below is computed at 1, 2 and 4 threads and the three outputs must match byte for byte, and
// the one-thread output must match a double reference. A chunk that dropped, re-zeroed or double-added
// its accumulators differs from the one-thread bytes in the first check; the reference catches a
// tiling bug that all three thread counts would share.
//
// The boundary shapes are deliberate: k = 1024 and 1536 sit just outside the rule (no chunk at any
// thread count), 1028 and 1532 just inside, 1500 is whisper's A@V. On x86 and on any core whose L1 is
// not 32 KB the rule may not fire at all, and the test then pins only the thread-count identity,
// which tinyBLAS gives anyway at these shapes.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct Shape { int64_t m, n, k; };

const Shape kShapes[] = {
    {64, 48, 1500},   // whisper's A@V rows, fewer columns
    {32, 36, 1028}, {32, 36, 1532},     // just inside the rule
    {32, 36, 1024}, {32, 36, 1536},     // just outside it
    {48, 27, 1152}, {16, 13, 1344},     // ragged column tiles (RN - 1), and the m % 16 == 0 floor
};

bool compute(ggml_backend_t backend, ggml_cgraph* gf, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

} // namespace

int main() {
    ggml_backend_t backend = loom_test::cpu_backend();
    LOOM_CHECK(backend != nullptr);
    if (backend == nullptr) LOOM_TEST_REPORT_AND_RETURN();
    unsigned st = 11;
    auto u = [&] { st = st * 1664525u + 1013904223u; return (float) (st >> 8) / 16777216.0f * 2 - 1; };

    for (const Shape& s : kShapes) {
        ggml_init_params ip = { (size_t) (s.k * s.m + s.k * s.n + s.m * s.n) * 4 + (size_t) 16 * 1024 * 1024,
                                nullptr, false };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.m);
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.k, s.n);
        float* A = (float*) a->data; float* B = (float*) b->data;
        for (int64_t i = 0; i < s.k * s.m; ++i) A[i] = u();
        for (int64_t i = 0; i < s.k * s.n; ++i) B[i] = u();
        ggml_tensor* c = ggml_mul_mat(ctx, a, b);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);

        std::vector<float> one((size_t) (s.m * s.n));
        for (int threads : {1, 2, 4}) {
            std::memset(c->data, 0xff, ggml_nbytes(c));        // a skipped tile reads as NaN, not as stale bytes
            LOOM_CHECK(compute(backend, gf, threads));
            const float* C = (const float*) c->data;
            if (threads == 1) {
                std::memcpy(one.data(), C, ggml_nbytes(c));
                // Error over sum |a b|, the scale f32 rounding of a k-term dot product is bounded by: about
                // 1e-6 here, where a dropped or doubled chunk is O(0.1). Against |ref| instead, k = 1500
                // random +-1 products put honest rounding at 1.3e-5 on the A72, past any tight bound.
                double worst = 0;
                for (int64_t j = 0; j < s.n; ++j)
                    for (int64_t i = 0; i < s.m; ++i) {
                        double ref = 0, mag = 0;
                        for (int64_t l = 0; l < s.k; ++l) {
                            const double p = (double) A[i * s.k + l] * (double) B[j * s.k + l];
                            ref += p;
                            mag += std::fabs(p);
                        }
                        const double err = std::fabs(C[j * s.m + i] - ref) / mag;
                        if (!(err <= worst)) worst = err;              // NaN-aware: a NaN makes worst NaN
                    }
                if (!(worst < 1e-5)) std::printf("m=%lld n=%lld k=%lld: worst rel err %g\n",
                                                 (long long) s.m, (long long) s.n, (long long) s.k, worst);
                LOOM_CHECK(worst < 1e-5);
            } else {
                const bool same = std::memcmp(one.data(), C, ggml_nbytes(c)) == 0;
                if (!same) std::printf("m=%lld n=%lld k=%lld: %d threads differ from 1 thread\n",
                                       (long long) s.m, (long long) s.n, (long long) s.k, threads);
                LOOM_CHECK(same);
            }
        }
        ggml_free(ctx);
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
