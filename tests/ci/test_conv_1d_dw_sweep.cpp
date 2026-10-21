// 1-D depthwise convolutions (CONV_2D_DW with one row) against a double reference, over the shapes that
// reach each edge of ggml-0035 (ADR-079 P4.32e): the ARMv6 build sends every stride-1, dilation-1 F32
// depthwise row with K >= 2 and at least 8 outputs to a short-vector kernel generated per K % 8, with the
// first 8 taps peeled when K >= 8, 8 outputs a block and a ragged last block staged. There it must give
// exactly its own contract's bits: the even taps and the odd taps, each summed in float in k order,
// then added. The shapes:
//   * every K % 8 on both sides of 8 (K = 2..17), at a length that is not a multiple of 8;
//   * Citrinet-256-ls's and conformer-ctc's widths at their lengths, a 'valid' conv (no pad), a pad wider
//     than the kernel, and an output shorter and longer than its input;
//   * what the kernel declines (K = 1, fewer than 8 outputs, stride 2, dilation 2), which ggml's sweep runs.
// Each through the fused chain the engine emits (PAD left -> CONV_2D_DW -> ADD bias -> RELU, ggml-0022)
// with the output placed over the PAD's input, the aliasing a graph allocator can produce, and bare. On
// 1 and 2 threads. Every CPU build runs its own paths through here; the ARMv6 build is checked in qemu
// (loom-armv6-build:bookworm) and on the board.
#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct Shape { int64_t c, il, k, pad, s = 1, d = 1; };

const Shape kShapes[] = {
    {3, 37, 2, 1},  {3, 37, 3, 1},  {3, 37, 4, 2},  {3, 37, 5, 2},  {3, 37, 6, 3},  {3, 37, 7, 3},
    {3, 37, 8, 4},  {3, 37, 9, 4},  {3, 37, 10, 5}, {3, 37, 11, 5}, {3, 37, 12, 6}, {3, 37, 13, 6},
    {3, 37, 14, 7}, {3, 37, 15, 7}, {3, 37, 16, 8}, {3, 37, 17, 8},
    {16, 1101, 11, 5}, {8, 551, 21, 10}, {8, 138, 41, 20}, {6, 200, 31, 15},   // Citrinet, conformer
    {4, 64, 5, 0}, {4, 50, 7, 9}, {5, 8, 3, 1}, {5, 16, 9, 0},                  // valid, wide pad, 8 and 8 outputs
    {4, 40, 1, 0}, {4, 9, 5, 0}, {4, 61, 11, 5, 2}, {4, 61, 5, 4, 1, 2},        // declined
};

bool compute(ggml_cgraph* gf, int n_threads) {
    static ggml_backend_t backend = loom_test::cpu_backend();
    if (backend == nullptr) return false;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

float u(unsigned& st) { st = st * 1664525u + 1013904223u; return (float) (st >> 8) / 16777216.0f * 2 - 1; }

// x at padded position j of channel c: x[c][j - pad], zero outside
float at(const std::vector<float>& X, const Shape& s, int64_t c, int64_t j) {
    const int64_t i = j - s.pad;
    return i >= 0 && i < s.il ? X[c * s.il + i] : 0.0f;
}

} // namespace

int main() {
    ggml_init_params ip = { (size_t) 64 * 1024 * 1024, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    LOOM_CHECK(ctx != nullptr);
    unsigned st = 11;
    int contract_rows = 0;

    for (const Shape& s : kShapes) {
        const int64_t ol = (s.il + 2 * s.pad - s.d * (s.k - 1) - 1) / s.s + 1;
#if defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON) && !defined(__aarch64__)
        const bool kernel = s.s == 1 && s.d == 1 && s.k >= 2 && ol >= 8;
#else
        const bool kernel = false;
#endif
        std::vector<float> W((size_t) s.k * s.c), X((size_t) s.il * s.c), B((size_t) s.c);
        for (float& v : W) v = u(st) * 0.3f;
        for (float& v : X) v = u(st);
        for (float& v : B) v = u(st) * 0.2f;

        for (int threads : {1, 2}) {
            for (int fused = 0; fused < 2; ++fused) {
                // fused: the padding moves into a PAD node (both sides) and the conv pads nothing, so the
                // geometry is the bare conv's; ggml-0022 folds the PAD back into the kernel's left pad.
                if (fused && (s.s != 1 || s.pad == 0)) continue;
                ggml_tensor* w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.k, 1, 1, s.c);
                ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.il, s.c, 1);
                ggml_tensor* b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, s.c, 1);
                std::memcpy(w->data, W.data(), ggml_nbytes(w));
                std::memcpy(x->data, X.data(), ggml_nbytes(x));
                std::memcpy(b->data, B.data(), ggml_nbytes(b));
                ggml_tensor* out;
                if (fused) {
                    ggml_tensor* p = ggml_pad_ext(ctx, x, (int) s.pad, (int) s.pad, 0, 0, 0, 0, 0, 0);
                    ggml_tensor* p4 = ggml_reshape_4d(ctx, p, p->ne[0], 1, s.c, 1);
                    ggml_tensor* conv = ggml_conv_2d_dw_direct(ctx, w, p4, 1, 1, 0, 0, (int) s.d, 1);
                    ggml_tensor* add = ggml_add(ctx, ggml_reshape_3d(ctx, conv, conv->ne[0], s.c, 1), b);
                    out = ggml_relu(ctx, add);
                    if (ggml_nbytes(out) == ggml_nbytes(x)) out->data = x->data;   // over the PAD's input
                } else {
                    ggml_tensor* x4 = ggml_reshape_4d(ctx, x, s.il, 1, s.c, 1);
                    out = ggml_conv_2d_dw_direct(ctx, w, x4, (int) s.s, 1, (int) s.pad, 0, (int) s.d, 1);
                }
                ggml_cgraph* gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out);
                LOOM_CHECK(compute(gf, threads));
                LOOM_CHECK(out->ne[0] == ol);
                const float* y = (const float*) out->data;

                double worst = 0;
                bool exact = true;
                for (int64_t c = 0; c < s.c; ++c)
                    for (int64_t t = 0; t < ol; ++t) {
                        double r = 0, mag = 0;
                        for (int64_t k = 0; k < s.k; ++k) {
                            const double m = (double) W[c * s.k + k] * at(X, s, c, t * s.s + k * s.d);
                            r += m; mag += std::fabs(m);
                        }
                        float want = 0;                       // the kernel's contract, in float
                        if (kernel) {
                            float e = W[c * s.k] * at(X, s, c, t), o = W[c * s.k + 1] * at(X, s, c, t + 1);
                            for (int64_t k = 2; k < s.k; ++k) (k % 2 ? o : e) += W[c * s.k + k] * at(X, s, c, t + k);
                            want = e + o;
                        }
                        if (fused) {
                            r += B[c]; mag += std::fabs(B[c]);
                            r = r > 0 ? r : 0;
                            want = want + B[c];
                            want = want > 0.f ? want : 0.f;
                        }
                        worst = std::max(worst, std::fabs(y[c * ol + t] - r) / (mag + 1e-30));
                        if (kernel && std::memcmp(&y[c * ol + t], &want, 4) != 0) exact = false;
                    }
                LOOM_CHECK(worst < 1e-5);
                if (kernel) { LOOM_CHECK(exact); ++contract_rows; }
                if (worst >= 1e-5 || (kernel && !exact))
                    std::fprintf(stderr, "  %s c %lld il %lld k %lld pad %lld s %lld d %lld, %d threads: %.3e%s\n",
                                 fused ? "fused" : "bare", (long long) s.c, (long long) s.il, (long long) s.k,
                                 (long long) s.pad, (long long) s.s, (long long) s.d, threads, worst,
                                 kernel && !exact ? ", not the kernel contract's bits" : "");
            }
        }
    }
#if defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON) && !defined(__aarch64__)
    std::printf("ARMv6: %d runs held to the depthwise kernel's contract bits\n", contract_rows);
    LOOM_CHECK(contract_rows > 0);
#endif
    ggml_free(ctx);
    LOOM_TEST_REPORT_AND_RETURN();
}
