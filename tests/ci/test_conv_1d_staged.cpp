// ggml-0053 (P4.40): on AArch64 the direct 1-D convolution (ggml_conv_1d_direct_ok's shapes, eight output
// channels at a time) runs an 8 x 12 tile over per-thread staged slices of the activation, with no padded
// copy of the whole of it. It sums every output exactly as the 4 x 16 direct kernel and the phase-major path
// do -- the bias, then each input channel's taps in order, one fused multiply-add each, then the residual --
// so with or without it (`GGML_CPU_CONV1D_STAGED=0`, the `_off` registration) a convolution must equal, bit
// for bit, that `fmaf` loop, the fused LEAKY_RELU included, at 1 to 4 threads. By default it is on only on a
// core without DOTPROD (the A72 it was measured on); on any other AArch64 core both registrations run the
// 4 x 16 kernel, and `GGML_CPU_CONV1D_STAGED=1` forces this one there.
//
// The shapes reach what the staging has to get right:
//   * a tail of 1-11 positions past the last 12-position block (the overlapping tile, stored from its floor);
//   * halos wider than a slice (dilation 12 at kernel 7 is 72 positions; 128 channels stage 12 at a time),
//     so the carry between slices overlaps itself, and asymmetric and zero padding;
//   * more threads than position blocks, so one thread has none;
//   * a resblock through a graph allocator whose output is placed over its input (counted), where each
//     thread's halos belong to its neighbours and its own earlier slices have overwritten what the next
//     one needs.
//
// Off AArch64 the direct kernel has the same order with an FMA (AVX2) and not without one, so there it is a
// bound check.
#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr float kSlope = 0.1f;

struct Shape { int64_t ic, oc, k, d, il, pad; bool leaky, bias, res; };

const Shape kShapes[] = {
    {8, 8, 3, 1, 251, 1, true, true, true},       // tail of 11
    {24, 24, 7, 3, 700, 9, true, true, true},     // tail of 4
    {32, 32, 7, 12, 1000, 36, true, true, true},  // a 72-position halo against 48-position slices
    {128, 128, 7, 12, 600, 36, true, true, true}, // one block per slice: the carry overlaps itself
    {16, 32, 5, 2, 400, 4, true, true, false},    // ic != oc
    {8, 8, 4, 1, 100, 1, false, true, false},     // asymmetric padding (1 left, 2 right), output 99
    {8, 16, 3, 1, 200, 0, false, false, false},   // no padding, no bias, nothing fused
    {8, 8, 3, 1, 64, 1, true, true, true},        // five blocks: at 4 threads one thread has none
};

bool compute(ggml_backend_t backend, ggml_cgraph * gf, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend, n_threads);
    return ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
}

} // namespace

int main() {
#if defined(__aarch64__)
    const bool exact = true;
#else
    const bool exact = false;
#endif
    ggml_backend_t backend = loom_test::cpu_backend();
    LOOM_CHECK(backend != nullptr);
    if (backend == nullptr) LOOM_TEST_REPORT_AND_RETURN();
    std::printf("bit-exact check: %s\n", exact ? "yes" : "no (bound check)");
    int aliased = 0;
    unsigned st = 23;
    auto u = [&] { st = st * 1664525u + 1013904223u; return (float) (st >> 8) / 16777216.0f * 2 - 1; };

    for (const Shape & s : kShapes) {
        const int64_t ol = s.il + 2 * s.pad - (s.k - 1) * s.d;
        std::vector<float> K((size_t) (s.k * s.ic * s.oc)), X((size_t) (s.il * s.ic)), B((size_t) s.oc), R((size_t) (ol * s.oc));
        for (float & v : K) v = u() * 0.2f;
        for (float & v : X) v = u();
        for (float & v : B) v = u() * 0.5f;
        for (float & v : R) v = u();

        // in place: the resblock h -> leaky -> conv -> + bias -> + h through the allocator, h's memory
        // reused for the result; otherwise the same chain (or less of it) with a residual of its own
        const bool inplace_case = s.res && s.ic == s.oc && ol == s.il;
        for (int variant = 0; variant < (inplace_case ? 2 : 1); ++variant) {
            const bool inplace = variant == 1;
            ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.k, 1, s.ic, s.oc);
            ggml_tensor * tx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.il, s.ic);
            ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, s.oc);
            ggml_tensor * tr = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ol, s.oc);
            ggml_set_input(tk); ggml_set_input(tx); ggml_set_input(tb); ggml_set_input(tr);
            ggml_tensor * h = ggml_scale(ctx, tx, 1.0f);              // an intermediate, so its memory is reusable
            ggml_tensor * a = s.leaky ? ggml_leaky_relu(ctx, h, kSlope, false) : h;
            ggml_tensor * conv = ggml_conv_2d_direct(ctx, tk, ggml_reshape_4d(ctx, a, s.il, 1, s.ic, 1), 1, 1,
                                                     (int) s.pad, 0, (int) s.d, 1);
            ggml_tensor * out = ggml_reshape_2d(ctx, conv, ol, s.oc);
            if (s.bias) out = ggml_add(ctx, out, tb);
            if (inplace)    out = ggml_add(ctx, h, out);              // h first: the allocator reuses src[0] in place
            else if (s.res) out = ggml_add(ctx, out, tr);
            ggml_set_output(out);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            LOOM_CHECK(ggml_gallocr_alloc_graph(ga, gf));
            if (inplace && out->data == h->data) ++aliased;

            // the direct kernel's arithmetic: bias first, then fmaf per (channel, tap), then the residual. When
            // the bias ADD is not fused into the convolution (whether it is, is ggml-0005/0007's call, not this
            // kernel's) the sum starts at zero and the bias is added after it, so that form is `alt`; a whole
            // output must be one form or the other.
            const float * res = inplace ? X.data() : (s.res ? R.data() : nullptr);
            std::vector<float> want((size_t) (ol * s.oc)), alt((size_t) (ol * s.oc));
            double worst = 0;
            std::vector<float> got(want.size()), one(want.size());
            for (int64_t o = 0; o < s.oc; ++o)
                for (int64_t t = 0; t < ol; ++t) {
                    float acc = s.bias ? B[o] : 0.0f, a0 = 0.0f;
                    for (int64_t c = 0; c < s.ic; ++c)
                        for (int64_t j = 0; j < s.k; ++j) {
                            const int64_t p = t + j * s.d - s.pad;
                            float x = (p < 0 || p >= s.il) ? 0.0f : X[c * s.il + p];
                            if (s.leaky) x = std::fmaf(kSlope, x < 0.0f ? x : 0.0f, x > 0.0f ? x : 0.0f);
                            acc = std::fmaf(x, K[(o * s.ic + c) * s.k + j], acc);
                            a0  = std::fmaf(x, K[(o * s.ic + c) * s.k + j], a0);
                        }
                    if (s.bias) a0 = a0 + B[o];
                    if (res) { acc = acc + res[o * ol + t]; a0 = a0 + res[o * ol + t]; }
                    want[o * ol + t] = acc;
                    alt[o * ol + t]  = a0;
                }
            for (int threads : {1, 2, 3, 4}) {
                ggml_backend_tensor_set(tk, K.data(), 0, ggml_nbytes(tk));
                ggml_backend_tensor_set(tx, X.data(), 0, ggml_nbytes(tx));
                if (s.bias) ggml_backend_tensor_set(tb, B.data(), 0, ggml_nbytes(tb));
                if (s.res && !inplace) ggml_backend_tensor_set(tr, R.data(), 0, ggml_nbytes(tr));
                LOOM_CHECK(compute(backend, gf, threads));
                ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
                int64_t mismatches = 0, mismatches_alt = 0;
                for (size_t i = 0; i < got.size(); ++i) {
                    if (std::memcmp(&want[i], &got[i], sizeof(float)) != 0) ++mismatches;
                    if (std::memcmp(&alt[i], &got[i], sizeof(float)) != 0) ++mismatches_alt;
                    const double err = std::fabs((double) got[i] - want[i]) / (std::fabs((double) want[i]) + 1.0);
                    if (!(err <= worst)) worst = err;                  // NaN-aware
                }
                mismatches = std::min(mismatches, mismatches_alt);
                if (threads == 1) one = got;
                else LOOM_CHECK(std::memcmp(one.data(), got.data(), got.size() * sizeof(float)) == 0);
                if ((exact && mismatches) || !(worst < 1e-5))
                    std::printf("ic %lld oc %lld k %lld d %lld il %lld pad %lld%s, %d threads: %lld outputs differ "
                                "from the fmaf loop, worst %g\n", (long long) s.ic, (long long) s.oc,
                                (long long) s.k, (long long) s.d, (long long) s.il, (long long) s.pad,
                                inplace ? " (output over input)" : "", threads, (long long) mismatches, worst);
                if (exact) LOOM_CHECK(mismatches == 0);
                LOOM_CHECK(worst < 1e-5);
            }
            ggml_gallocr_free(ga);
            ggml_free(ctx);
        }
    }
    std::printf("resblock output placed over its input: %d\n", aliased);
    LOOM_CHECK(aliased > 0);    // otherwise the halos and the carry went untested
    LOOM_TEST_REPORT_AND_RETURN();
}
