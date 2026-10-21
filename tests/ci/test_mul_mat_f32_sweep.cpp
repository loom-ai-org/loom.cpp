// F32 MUL_MAT against a double reference, over the shapes that reach each of the ARMv6 paths and their
// edges (ADR-079): ggml-0033's one-column GEMV, which must give exactly its own contract's bits (float
// products summed in float, k in order, one chain per row), and ggml-0034's short-vector GEMM for n >= 24, which blocks K in chunks, rows by
// 32 and columns by quads -- the shapes below reach each edge and each fallback. Every
// CPU build runs its own paths through here; the ARMv6 build is checked in qemu (loom-armv6-build:bookworm)
// and on the board.
#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct Shape { int64_t k, m, n; };

// (K, M = rows of the weight, N = columns of the activation)
const Shape kShapes[] = {
    {288, 2304, 1}, {512, 2048, 1}, {7, 13, 1}, {33, 21, 1},            // GEMV, incl. tails
    {288, 1152, 457}, {1152, 288, 457}, {256, 256, 551},                // Moonshine, Citrinet-sized
    {100, 132, 24}, {33, 4, 25}, {1153, 36, 27}, {64, 260, 31},         // chunks, ragged blocks, partial quads
    {64, 64, 7}, {64, 66, 32}, {64, 64, 16}, {5, 8, 2},                 // fall back: n < 24, M % 4, n = 2
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
    unsigned st = 3;
    auto u = [&] { st = st * 1664525u + 1013904223u; return (float) (st >> 8) / 16777216.0f * 2 - 1; };

    for (const Shape& s : kShapes) {
        for (int threads : {1, 2}) {
            ggml_init_params ip = { (size_t) (s.k * s.m + s.k * s.n + s.m * s.n) * 4 + (size_t) 64 * 1024 * 1024,
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
            LOOM_CHECK(compute(backend, gf, threads));
            const float* C = (const float*) c->data;
            double worst = 0;
            bool exact = true;   // the ARMv6 GEMV contract: float products summed in float, in k order
            for (int64_t n = 0; n < s.n; ++n)
                for (int64_t m = 0; m < s.m; ++m) {
                    double r = 0, mag = 0;
                    float sf = 0;
                    for (int64_t k = 0; k < s.k; ++k) {
                        const double p = (double) A[m * s.k + k] * B[n * s.k + k];
                        r += p; mag += std::fabs(p);
                        sf += A[m * s.k + k] * B[n * s.k + k];
                    }
                    worst = std::max(worst, std::fabs(C[n * s.m + m] - r) / (mag + 1e-30));
                    if (C[n * s.m + m] != sf) exact = false;
                }
            LOOM_CHECK(worst < 1e-5);
            if (worst >= 1e-5)
                std::fprintf(stderr, "  k %lld m %lld n %lld, %d threads: %.3e\n", (long long) s.k, (long long) s.m,
                             (long long) s.n, threads, worst);
#if defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON) && !defined(__aarch64__)
            if (s.n == 1) {
                LOOM_CHECK(exact);
                if (!exact) std::fprintf(stderr, "  GEMV k %lld m %lld does not give its float-sum contract's bits\n",
                                         (long long) s.k, (long long) s.m);
            }
#else
            (void) exact;
#endif
            ggml_free(ctx);
        }
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
