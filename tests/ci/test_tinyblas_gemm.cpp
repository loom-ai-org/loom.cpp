// tinyBLAS is compiled in, and it computes the right answer (BACKLOG.md P4.15).
//
// Two claims, and the first is the reason this file exists. `GGML_LLAMAFILE` is a ggml build option
// that defaults OFF for a standalone ggml, and this engine turns it on (cmake/Dependencies.cmake)
// because it is worth ~2x on x86-64 and -- with the two aarch64 patches alongside it -- 1.6x on ARM
// at the F32 GEMM shapes a convolutional vocoder actually runs. Nothing about a build with the option
// silently lost is observable from its output: it produces correct audio, slightly different in the
// last few bits, at half the speed. So the flag itself is asserted here, where a wrong answer is loud.
//
//   1. **The CPU backend reports the LLAMAFILE feature.** Asked through the registry's
//      `ggml_backend_get_features` rather than `ggml_cpu_has_llamafile()`, because the latter is a
//      symbol inside the CPU backend and a GGML_BACKEND_DL build -- the one the wheels ship -- does not
//      link it (the same trap tests/support/cpu_backend.h documents for `ggml_backend_cpu_init`).
//   2. **mul_mat is right, at the shapes that take the tinyBLAS path and at the ones that fall off
//      it.** tinyBLAS is selected by shape: `n >= 4` on NEON, `k >= KN`, and one of `m % 16/8/4`, with
//      BOTH remainders -- rows past `m % 4` and the `k % KN` leftovers -- handled separately from the
//      tiled body. Both sides of each of those conditions are covered below, so a patch that got a
//      tail block wrong -- the failure mode of ANY change to the tiling, which is exactly what P4.15
//      and P4.18 changed -- shows up as an arithmetic error rather than as a performance change
//      nobody notices.
//
//      **`k % KN` used to be a rejection and is now a tail** (cmake/patches/ggml-0011, P4.18): the
//      contraction is a sequence length in every attention matmul, so whisper's `k = 1500` missed
//      `KN = 8` on AVX2 and `KN = 16` on AVX-512 and sent that whole GEMM to the generic kernel. The
//      residue sweep below is the analogue of the `m % 4` one above it, and it is the check that the
//      leftover elements are ACCUMULATED rather than dropped -- a dropped term is O(1) relative,
//      four orders of magnitude past the 1e-5 this file already tolerates.
//
// Compared against a double-precision reference within a relative tolerance, not bit-exactly: tinyBLAS
// accumulates in a different order from ggml's own path (~3e-7 relative), which is the whole point of
// the "re-baseline byte-identity gates" note in P4.15 and not something to pin here.

#include "test_util.h"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// C[n*M + m] = sum_k A[m*K + k] * B[n*K + k], in double, which is what ggml_mul_mat computes for two
// K-contiguous F32 operands -- the layout every im2col-lowered convolution in this engine produces.
double reference_element(const std::vector<float>& A, const std::vector<float>& B,
                         int64_t K, int64_t m, int64_t n) {
    double acc = 0.0;
    for (int64_t k = 0; k < K; ++k) acc += (double)A[m * K + k] * (double)B[n * K + k];
    return acc;
}

// Deterministic, non-degenerate, and never left unwritten: an untouched ggml buffer reads as the
// shared zero page, which turns a wrong kernel into a passing test (BACKLOG.md P4.14's first
// benchmarking trap, which cost a full write-up).
void fill(std::vector<float>& v, float base, float step, int period) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = base + step * (float)(i % period);
}

void check_mul_mat(ggml_backend_t backend, int64_t K, int64_t M, int64_t N) {
    std::vector<float> A((size_t)M * K), B((size_t)N * K);
    fill(A, 0.01f, 0.001f, 97);
    fill(B, 0.02f, -0.001f, 53);

    ggml_init_params ip = { (size_t)512 * 1024 * 1024, nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    LOOM_CHECK(ctx != nullptr);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_tensor* ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
    ggml_tensor* tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_input(ta);
    ggml_set_input(tb);
    ggml_tensor* td = ggml_mul_mat(ctx, ta, tb);
    ggml_build_forward_expand(gf, td);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    LOOM_CHECK(ggml_gallocr_alloc_graph(alloc, gf));
    ggml_backend_tensor_set(ta, A.data(), 0, A.size() * sizeof(float));
    ggml_backend_tensor_set(tb, B.data(), 0, B.size() * sizeof(float));
    LOOM_CHECK(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float> out((size_t)ggml_nelements(td));
    ggml_backend_tensor_get(td, out.data(), 0, out.size() * sizeof(float));

    // Every element of the first and last row-block and every column: the tail blocks a tiling change
    // gets wrong are at the edges, and checking a diagonal would walk straight past them.
    double worst = 0.0;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t m = 0; m < M; ++m) {
            if (m > 3 && m < M - 4) continue;
            const double ref = reference_element(A, B, K, m, n);
            const double got = (double)out[(size_t)n * M + m];
            const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-6);
            if (rel > worst) worst = rel;
        }
    }
    LOOM_CHECK(worst < 1e-5);
    if (worst >= 1e-5) {
        std::fprintf(stderr, "  shape K=%lld M=%lld N=%lld worst relative error %.3e\n",
                     (long long)K, (long long)M, (long long)N, worst);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
}

} // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    if (backend == nullptr) LOOM_TEST_REPORT_AND_RETURN();

    ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
    LOOM_CHECK(dev != nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    LOOM_CHECK(reg != nullptr);

    auto get_features = (ggml_backend_get_features_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
    LOOM_CHECK(get_features != nullptr);

    bool llamafile = false;
    if (get_features != nullptr) {
        for (ggml_backend_feature* f = get_features(reg); f != nullptr && f->name != nullptr; ++f) {
            if (std::strcmp(f->name, "LLAMAFILE") == 0 && std::strcmp(f->value, "1") == 0) {
                llamafile = true;
            }
        }
    }
#if LOOM_EXPECT_TINYBLAS
    // Red when a build loses GGML_LLAMAFILE -- a ggml bump that renames the option, a stale cache
    // entry from before this engine started asking for it, an accelerator package configured by hand.
    LOOM_CHECK(llamafile);
    if (!llamafile) {
        std::fprintf(stderr,
                     "the CPU backend was built without tinyBLAS: configure with -DLOOM_TINYBLAS=ON, "
                     "and check that ggml still spells the option GGML_LLAMAFILE\n");
    }
#else
    // The A/B direction. -DLOOM_TINYBLAS=OFF is how the measurements in P4.15 were taken, and it has
    // to keep meaning what it says.
    LOOM_CHECK(!llamafile);
#endif

    // K % 4 == 0 and n >= 4: the tinyBLAS path proper, at the three `m % 16 / 8 / 4` branches whose
    // tile widths the aarch64 patch changes. M and N are the vocoder's own aspect ratio -- long
    // activation, few channels -- shrunk until the whole file runs in well under a second.
    check_mul_mat(backend.get(), 224, 256, 32);   // m % 16 == 0
    check_mul_mat(backend.get(), 160, 264, 64);   // m % 8  == 0, m % 16 != 0
    check_mul_mat(backend.get(), 96,  260, 128);  // m % 4  == 0, m % 8  != 0
    check_mul_mat(backend.get(), 1344, 288, 384); // large K, wide N -- the two small-M shapes

    // Every remainder of the row split, because those rows take a DIFFERENT path from the ones above
    // them -- the tiled kernel handles `m - m % 4` and a separate loop finishes the rest, so a bug
    // there shows up in one to three rows of a matrix whose other 256 are right. The reference check
    // covers the last four rows of every shape precisely so this is visible.
    check_mul_mat(backend.get(), 224, 257, 32);   // m % 4 == 1
    check_mul_mat(backend.get(), 224, 258, 64);   // m % 4 == 2
    check_mul_mat(backend.get(), 960, 287, 384);  // m % 4 == 3 -- the real VITS vocoder shape
    check_mul_mat(backend.get(), 224, 3,   32);   // fewer rows than one tile: no tiled part at all

    // And the fall-off, where ggml's own kernel has to produce the same answer: n < 4 (NEON bails).
    check_mul_mat(backend.get(), 224, 256, 3);

    // Every remainder of the CONTRACTION split, for the same reason as the row split above: the tiled
    // body computes `k - k % KN` and a scalar loop inside the tile finishes the rest, so a bug there
    // is a small systematic error in EVERY element rather than a wrong row. 16 covers KN of 4, 8 and
    // 16 with one sweep, and 97 is kept because it was the shape this file already had.
    for (int64_t r = 1; r < 16; ++r) {
        check_mul_mat(backend.get(), 224 + r, 256, 32);
    }
    check_mul_mat(backend.get(), 97,  256, 32);

    // k below one vector, which still declines -- there is no aligned prefix to tile.
    check_mul_mat(backend.get(), 3,   256, 32);

    // And the real shape the tail exists for: whisper-small's encoder A@V contracts over 1500 mel
    // frames. 1500 % 8 == 4, 1500 % 16 == 12, 1500 % 4 == 0 -- so this is a tail on x86 and not on
    // ARM, which is exactly how it stayed invisible. Small M and N keep it under a second.
    check_mul_mat(backend.get(), 1500, 64, 48);

    LOOM_TEST_REPORT_AND_RETURN();
}
