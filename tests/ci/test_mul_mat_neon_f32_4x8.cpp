// ggml-0048 (P4.38): an F32 x F32 mul_mat on an AArch64 core without DOTPROD takes an outer-product 4x8
// tile under an L2-blocked schedule when it has at least 128 activation rows. Every output is then ONE
// sequential fused sum over k -- the panels carry the sum through C between blocks of k -- so where the
// path runs the result must equal, bit for bit, a scalar `fmaf` loop over k from 0. That is the claim
// pinned here, at 1, 2 and 4 threads, which must also agree byte for byte with each other.
//
// The shapes put every edge of the schedule under the check: a weight-row count that is not a multiple
// of 8 (the edge panel, whose columns may end at the edge of C's allocation), more rows than one packed
// block holds (256 on a 1 MB L2), k past one block of k (256) and not a multiple of 4 (the scalar tail),
// an activation-row count that is not a multiple of 4 (the aliased rows), a weight whose rows are a
// strided view, and a batch of three products that reuse the same work buffer one after another.
//
// Elsewhere -- x86, a DOTPROD core, or GGML_CPU_NEON_F32_4X8=0, which the `_off` registration sets --
// tinyBLAS computes these: the test pins the thread-count identity and an error bound against a double
// product, AND that some output is not the sequential sum. tinyBLAS adds lane partials, so across
// these shapes it cannot reproduce the loop everywhere; if the `_off` run does, the switch is dead.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Shape { int64_t m, n, k, pad, batch; };

const Shape kShapes[] = {
    {300, 130, 600, 0, 1},    // two row blocks (256 + 44, an edge panel of 4), three blocks of k, n % 4 = 2
    {64, 128, 768, 0, 1},     // the threshold row count exactly, one of whisper's k
    {13, 131, 257, 0, 1},     // one edge panel only, k % 4 = 1 across a block boundary, n % 4 = 3
    {8, 200, 3, 0, 1},        // k shorter than one vector step: the tail loop alone
    {40, 129, 260, 5, 1},     // weight rows strided (lda = k + 5)
    {24, 136, 100, 0, 3},     // a batch: three products through the same work buffer
};

bool compute(ggml_backend_t backend, ggml_cgraph * gf, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

bool path_expected() {
#if defined(__aarch64__)
    const char * e = std::getenv("GGML_CPU_NEON_F32_4X8");
    return !(e && e[0] == '0') && !ggml_cpu_has_dotprod();
#else
    return false;
#endif
}

} // namespace

int main() {
    ggml_backend_t backend = loom_test::cpu_backend();
    LOOM_CHECK(backend != nullptr);
    if (backend == nullptr) LOOM_TEST_REPORT_AND_RETURN();
    const bool exact = path_expected();
    std::printf("outer-product path expected: %s\n", exact ? "yes (bit-exact check)" : "no (bound check)");
    int64_t mismatches_total = 0;   // over shapes with k >= 64, against the sequential loop
    unsigned st = 7;
    auto u = [&] { st = st * 1664525u + 1013904223u; return (float) (st >> 8) / 16777216.0f * 2 - 1; };

    for (const Shape & s : kShapes) {
        const int64_t lda = s.k + s.pad;
        ggml_init_params ip = { (size_t) (lda * s.m + s.k * s.n + s.m * s.n) * 4 * s.batch + (size_t) 16 * 1024 * 1024,
                                nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * wide = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, lda, s.m, s.batch);
        ggml_tensor * a = s.pad ? ggml_view_3d(ctx, wide, s.k, s.m, s.batch, wide->nb[1], wide->nb[2], 0) : wide;
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.k, s.n, s.batch);
        float * A = (float *) wide->data;
        float * B = (float *) b->data;
        for (int64_t i = 0; i < lda * s.m * s.batch; ++i) A[i] = u();
        for (int64_t i = 0; i < s.k * s.n * s.batch; ++i) B[i] = u();
        ggml_tensor * c = ggml_mul_mat(ctx, a, b);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, c);

        std::vector<float> one((size_t) (s.m * s.n * s.batch));
        for (int threads : {1, 2, 4}) {
            std::memset(c->data, 0xff, ggml_nbytes(c));   // a skipped tile reads as NaN, not as stale bytes
            LOOM_CHECK(compute(backend, gf, threads));
            const float * C = (const float *) c->data;
            if (threads > 1) {
                const bool same = std::memcmp(one.data(), C, ggml_nbytes(c)) == 0;
                if (!same) std::printf("m=%lld n=%lld k=%lld: %d threads differ from 1 thread\n",
                                       (long long) s.m, (long long) s.n, (long long) s.k, threads);
                LOOM_CHECK(same);
                continue;
            }
            std::memcpy(one.data(), C, ggml_nbytes(c));
            int64_t mismatches = 0;
            double worst = 0;
            for (int64_t q = 0; q < s.batch; ++q)
                for (int64_t j = 0; j < s.n; ++j)
                    for (int64_t i = 0; i < s.m; ++i) {
                        const float * ar = A + (q * s.m + i) * lda;
                        const float * br = B + (q * s.n + j) * s.k;
                        const float got = C[(q * s.n + j) * s.m + i];
                        float acc = 0.0f;
                        for (int64_t l = 0; l < s.k; ++l) acc = std::fmaf(ar[l], br[l], acc);
                        if (std::memcmp(&acc, &got, sizeof acc) != 0) ++mismatches;
                        // over sum |a b|, the scale k-term f32 rounding is bounded by (see test_tinyblas_k_halves)
                        double ref = 0, mag = 0;
                        for (int64_t l = 0; l < s.k; ++l) {
                            const double p = (double) ar[l] * (double) br[l];
                            ref += p;
                            mag += std::fabs(p);
                        }
                        const double err = std::fabs(got - ref) / mag;
                        if (!(err <= worst)) worst = err;              // NaN-aware
                    }
            if (s.k >= 64) mismatches_total += mismatches;
            if ((exact && mismatches) || !(worst < 1e-5))
                std::printf("m=%lld n=%lld k=%lld batch=%lld: %lld outputs differ from the fmaf loop, worst rel err %g\n",
                            (long long) s.m, (long long) s.n, (long long) s.k, (long long) s.batch,
                            (long long) mismatches, worst);
            if (exact) LOOM_CHECK(mismatches == 0);
            LOOM_CHECK(worst < 1e-5);
        }
        ggml_free(ctx);
    }
    if (!exact) {
        std::printf("%lld outputs differ from the sequential loop (must be > 0 off the path)\n",
                    (long long) mismatches_total);
        LOOM_CHECK(mismatches_total > 0);
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
