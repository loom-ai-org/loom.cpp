// ggml-0049 (P4.39): the batched im2col convolution (CONV_2D, ggml_compute_forward_conv_2d_impl) hands its
// GEMM to P4.38's outer-product path on an AArch64 core without DOTPROD, with the path's packed block carved
// off the end of the convolution's own work buffer, past its patches. Every output is then ONE sequential
// fused sum over the patch, in im2col order (input channel, then tap) -- so a plain convolution must equal,
// bit for bit, a scalar `fmaf` loop over that order, padding taps included. That is pinned here at 1, 2 and
// 4 threads, which must also agree byte for byte with each other.
//
// The shapes reach every GEMM call the convolution makes, in both of its operand orders:
//   * a single image with a contiguous output: the GEMM writes straight into dst (patches packed, the
//     kernel's rows walked four at a time), one batch and many batches of patches, strided and dilated;
//   * two images: the result goes through a scratch buffer and a permute (kernel packed, patches walked);
//   * a resblock through a graph allocator, where the output is placed over the input and each batch's
//     result is held back in the buffer one batch longer (the deferred flush) -- with a bias and the
//     residual, `(sum + bias) + residual` in float, which is exactly what that flush computes.
// and CONV_TRANSPOSE_1D's GEMM, which already has the path's operand order (kernel packed, positions walked).
// Every CONV_2D shape keeps off the direct path (weights over 512 KB, or too few positions per channel tile).
//
// Elsewhere -- x86, a DOTPROD core, or either switch at "0", which the `_off` registration sets -- the
// test pins the thread-count identity and an error bound, AND that some output is NOT the sequential sum:
// tinyBLAS adds lane partials, so if the `_off` run reproduces the loop everywhere the switch is dead.
#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Shape { int64_t ic, oc, k, d, il, pad, st, images; bool resblock; };

const Shape kShapes[] = {
    {96, 138, 5, 1, 300, 2, 1, 1, false},    // one batch, two blocks of patches (256 + 44), two of k, oc % 4 = 2
    {192, 200, 7, 2, 1100, 6, 1, 1, false},  // 14 batches of 80 patches (the last 60), five blocks of k
    {32, 144, 4, 1, 301, 1, 2, 1, false},    // strided: the general gather, k = 128 exactly
    {64, 40, 3, 1, 200, 1, 1, 2, false},     // two images: kernel packed (40 rows, an edge panel), patches walked
    {160, 160, 3, 1, 120, 1, 1, 1, true},    // a resblock whose output lands on its input: the deferred flush
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
    const char * c = std::getenv("GGML_CPU_NEON_F32_4X8_CONV");
    return !(e && e[0] == '0') && !(c && c[0] == '0') && !ggml_cpu_has_dotprod();
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
    std::printf("outer-product conv path expected: %s\n", exact ? "yes (bit-exact check)" : "no (bound check)");
    int64_t mismatches_total = 0;
    int aliased = 0;
    unsigned st = 11;
    auto u = [&] { st = st * 1664525u + 1013904223u; return (float) (st >> 8) / 16777216.0f * 2 - 1; };

    for (const Shape & s : kShapes) {
        const int64_t ol = (s.il + 2 * s.pad - (s.k - 1) * s.d - 1) / s.st + 1;
        std::vector<float> K((size_t) (s.k * s.ic * s.oc)), X((size_t) (s.il * s.ic * s.images)), B((size_t) s.oc);
        for (float & v : K) v = u() * 0.2f;
        for (float & v : X) v = u();
        for (float & v : B) v = u() * 0.5f;

        ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.k, 1, s.ic, s.oc);
        ggml_tensor * tx = s.resblock ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.il, s.ic)
                                      : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.il, 1, s.ic, s.images);
        ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, s.oc);
        ggml_set_input(tk); ggml_set_input(tx); ggml_set_input(tb);
        ggml_tensor * out = nullptr;
        ggml_tensor * h = nullptr;
        if (s.resblock) {
            // h -> conv -> + bias -> + h, which ggml-0005/0007 fuse into the convolution; the allocator
            // gives the fused result h's own memory (counted below, so the deferred case is real)
            h = ggml_scale(ctx, tx, 1.0f);              // an intermediate, so its memory is reusable
            ggml_tensor * conv = ggml_conv_2d_direct(ctx, tk, ggml_reshape_4d(ctx, h, s.il, 1, s.ic, 1), 1, 1,
                                                     (int) s.pad, 0, (int) s.d, 1);
            out = ggml_add(ctx, ggml_reshape_2d(ctx, conv, ol, s.oc), tb);
            out = ggml_add(ctx, h, out);                // h first: the allocator reuses src[0] in place
        } else {
            out = ggml_conv_2d_direct(ctx, tk, tx, (int) s.st, 1, (int) s.pad, 0, (int) s.d, 1);
        }
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        LOOM_CHECK(ggml_gallocr_alloc_graph(ga, gf));
        if (h != nullptr && out->data == h->data) ++aliased;

        const size_t n_out = (size_t) (ol * s.oc * s.images);
        std::vector<float> one(n_out), got(n_out);
        for (int threads : {1, 2, 4}) {
            ggml_backend_tensor_set(tk, K.data(), 0, ggml_nbytes(tk));
            ggml_backend_tensor_set(tx, X.data(), 0, ggml_nbytes(tx));
            if (s.resblock) ggml_backend_tensor_set(tb, B.data(), 0, ggml_nbytes(tb));   // a graph node only there
            LOOM_CHECK(compute(backend, gf, threads));
            ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
            if (threads > 1) {
                const bool same = std::memcmp(one.data(), got.data(), n_out * sizeof(float)) == 0;
                if (!same) std::printf("ic %lld oc %lld k %lld: %d threads differ from 1 thread\n",
                                       (long long) s.ic, (long long) s.oc, (long long) s.k, threads);
                LOOM_CHECK(same);
                continue;
            }
            one = got;
            int64_t mismatches = 0;
            double worst = 0;
            for (int64_t q = 0; q < s.images; ++q)
                for (int64_t o = 0; o < s.oc; ++o)
                    for (int64_t t = 0; t < ol; ++t) {
                        float acc = 0.0f;
                        double ref = 0, mag = 0;
                        for (int64_t c = 0; c < s.ic; ++c)
                            for (int64_t j = 0; j < s.k; ++j) {
                                const int64_t p = t * s.st + j * s.d - s.pad;
                                const float x = (p < 0 || p >= s.il) ? 0.0f : X[(q * s.ic + c) * s.il + p];
                                const float w = K[(o * s.ic + c) * s.k + j];
                                acc = std::fmaf(x, w, acc);
                                ref += (double) x * w;
                                mag += std::fabs((double) x * w);
                            }
                        if (s.resblock) {
                            const float r = X[o * s.il + t];
                            acc = (acc + B[o]) + r;
                            ref += (double) B[o] + r;
                            mag += std::fabs(B[o]) + std::fabs(r);
                        }
                        const float g = got[(q * s.oc + o) * ol + t];
                        if (std::memcmp(&acc, &g, sizeof acc) != 0) ++mismatches;
                        const double err = std::fabs(g - ref) / (mag + 1e-30);
                        if (!(err <= worst)) worst = err;               // NaN-aware
                    }
            mismatches_total += mismatches;
            if ((exact && mismatches) || !(worst < 1e-5))
                std::printf("ic %lld oc %lld k %lld il %lld images %lld%s: %lld outputs differ from the fmaf loop, "
                            "worst rel err %g\n", (long long) s.ic, (long long) s.oc, (long long) s.k,
                            (long long) s.il, (long long) s.images, s.resblock ? " resblock" : "",
                            (long long) mismatches, worst);
            if (exact) LOOM_CHECK(mismatches == 0);
            LOOM_CHECK(worst < 1e-5);
        }
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
    // CONV_TRANSPOSE_1D: [Cout*K, Cin] x [Cin, L] blocked over positions, then an overlap-add. Two blocks of
    // positions, 655 and 200 (both >= 128, so both on the path), Cout*K = 100 (an edge panel), Cin = 70 (a
    // k tail); the output is 0 plus each position's sum in increasing order, which is what the overlap-add
    // does.
    {
        const int64_t K = 5, oc = 20, ic = 70, L = 855, s0 = 3, ol = (L - 1) * s0 + K;
        std::vector<float> W((size_t) (K * oc * ic)), X((size_t) (L * ic));
        for (float & v : W) v = u() * 0.2f;
        for (float & v : X) v = u();
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * tw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, oc, ic);
        ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, ic);
        ggml_set_input(tw); ggml_set_input(tx);
        ggml_tensor * out = ggml_conv_transpose_1d(ctx, tw, tx, (int) s0, 0, 1);
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        LOOM_CHECK(ggml_gallocr_alloc_graph(ga, gf));
        std::vector<float> want((size_t) (ol * oc), 0.0f), ref((size_t) (ol * oc), 0.0), mag((size_t) (ol * oc), 0.0);
        for (int64_t l = 0; l < L; ++l)
            for (int64_t o = 0; o < oc; ++o)
                for (int64_t k = 0; k < K; ++k) {
                    float acc = 0.0f;
                    double r = 0, m = 0;
                    for (int64_t c = 0; c < ic; ++c) {
                        const float w = W[(c * oc + o) * K + k], x = X[c * L + l];
                        acc = std::fmaf(w, x, acc);
                        r += (double) w * x;
                        m += std::fabs((double) w * x);
                    }
                    want[o * ol + l * s0 + k] += acc;
                    ref[o * ol + l * s0 + k] += (float) r;
                    mag[o * ol + l * s0 + k] += (float) m;
                }
        std::vector<float> one(want.size()), got(want.size());
        for (int threads : {1, 2, 4}) {
            ggml_backend_tensor_set(tw, W.data(), 0, ggml_nbytes(tw));
            ggml_backend_tensor_set(tx, X.data(), 0, ggml_nbytes(tx));
            LOOM_CHECK(compute(backend, gf, threads));
            ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
            if (threads > 1) {
                LOOM_CHECK(std::memcmp(one.data(), got.data(), got.size() * sizeof(float)) == 0);
                continue;
            }
            one = got;
            int64_t mismatches = 0;
            double worst = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                if (std::memcmp(&want[i], &got[i], sizeof(float)) != 0) ++mismatches;
                const double err = std::fabs(got[i] - ref[i]) / (mag[i] + 1e-30);
                if (!(err <= worst)) worst = err;
            }
            mismatches_total += mismatches;
            if ((exact && mismatches) || !(worst < 1e-5))
                std::printf("conv_transpose_1d: %lld outputs differ from the fmaf loop, worst rel err %g\n",
                            (long long) mismatches, worst);
            if (exact) LOOM_CHECK(mismatches == 0);
            LOOM_CHECK(worst < 1e-5);
        }
        ggml_gallocr_free(ga);
        ggml_free(ctx);
    }
    std::printf("resblock output placed over its input: %d\n", aliased);
    LOOM_CHECK(aliased > 0);    // otherwise the deferred flush went untested
    if (!exact) {
        std::printf("%lld outputs differ from the sequential loop (must be > 0 off the path)\n",
                    (long long) mismatches_total);
        LOOM_CHECK(mismatches_total > 0);
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
