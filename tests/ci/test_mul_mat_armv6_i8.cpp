// The ARMv6 dynamic int8 mul_mat (ADR-079 P4.33, ggml-0041, ggml-0042 and ggml-0043): CPU_ARMV6_I8, a buffer type for
// Q8_0 weights whose blocks carry one scale per row (ADR-076's Q8_0_PC). Below 16 columns one kernel runs
// a channel quad against one column; from 16, int16 GEMM tiles. Both keep the same contract.
//
// Its contract is integer arithmetic plus a fixed float epilogue, so it is checked BIT FOR BIT against a
// reference written here: each column of x quantized with its absmax to the widest range R whose K
// products cannot overflow an int32 (32767, or 2^31 / (128 K) above K = 516), rounded to nearest; the
// int32 dot product with the int8 weights; then acc * (sx * sw[n]). Registered twice, GGML_CPU_ARMV6_I8=0
// and =1 (read once). On ARMv6 the first must find no such buffer type and the second must find it and
// pass every check below. Elsewhere the type never exists, and both runs check only that.
//
//   1. Every (K, N, columns) below, on both paths, including N not a multiple of four, K above 516 (a
//      narrower x range), K in several chunks, an odd column count and a batched src1, on one and two threads.
//   2. The overflow bound: weights of -128 and a constant x at full range, K = 3072, against int64.
//   3. A NaN in one column makes that column's outputs NaN and leaves the others exact.
//   4. A tensor whose blocks do NOT share a scale is not packed: it computes exactly as the same bytes
//      do in a plain CPU buffer (ggml's own Q8_0 mul_mat).
//   5. A pointwise convolution (ggml-0043), at stride 1 and 2, on both paths, followed by a bias ADD that
//      ggml's conv fusion must not take; and a three-tap convolution on the same kernel is declined.

#include "cpu_backend.h"
#include "test_util.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

ggml_backend_t backend() {
    static ggml_backend_t b = loom_test::cpu_backend();
    return b;
}

bool compute(ggml_cgraph* gf, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend()));
    auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_n_threads != nullptr) set_n_threads(backend(), n_threads);
    return ggml_backend_graph_compute(backend(), gf) == GGML_STATUS_SUCCESS;
}

ggml_backend_buffer_type_t i8_buft() {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend());
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra == nullptr) return nullptr;
    for (ggml_backend_buffer_type_t* p = get_extra(dev); p != nullptr && *p != nullptr; ++p) {
        if (std::strcmp(ggml_backend_buft_name(*p), "CPU_ARMV6_I8") == 0) return *p;
    }
    return nullptr;
}

struct Block { uint16_t d; int8_t qs[32]; };
static_assert(sizeof(Block) == 34, "block_q8_0");

// A Q8_0 weight [N][K] with one F16 scale per row (or, `uniform` false, a different one per block).
std::vector<Block> make_weight(int64_t K, int64_t N, int seed, bool uniform) {
    std::vector<Block> w((size_t) (N * K / 32));
    uint32_t s = 2166136261u ^ (uint32_t) seed;
    auto next = [&] { s = s * 1664525u + 1013904223u; return s >> 8; };
    for (int64_t n = 0; n < N; ++n) {
        const float scale = 0.002f + 0.0005f * (float) (n % 13);
        for (int64_t b = 0; b < K / 32; ++b) {
            Block& blk = w[(size_t) (n * (K / 32) + b)];
            blk.d = ggml_fp32_to_fp16(uniform ? scale : scale * (1.0f + 0.25f * (float) (b % 3)));
            for (int i = 0; i < 32; ++i) blk.qs[i] = (int8_t) ((int) (next() % 255) - 127);
        }
    }
    return w;
}

int x_range(int64_t K) {
    const int64_t r = INT64_C(2147483647) / (128 * K);
    return r > 32767 ? 32767 : (int) r;
}

// The contract, for one column: what CPU_ARMV6_I8 must write, to the bit.
void reference(const std::vector<Block>& w, int64_t K, int64_t N, const float* x, float* y) {
    float m = 0.0f, poison = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const float a = std::fabs(x[k]);
        m = a > m ? a : m;
        poison += x[k] * 0.0f;
    }
    const int R = x_range(K);
    const float inv = m > 0.0f ? R / m : 0.0f;
    const float sx = m / R + poison;
    std::vector<int32_t> q((size_t) K);
    for (int64_t k = 0; k < K; ++k) q[(size_t) k] = (int32_t) std::nearbyint(x[k] * inv);
    for (int64_t n = 0; n < N; ++n) {
        int64_t acc = 0;
        for (int64_t k = 0; k < K; ++k) acc += (int64_t) w[(size_t) (n * (K / 32) + k / 32)].qs[k % 32] * q[(size_t) k];
        const float sw = ggml_fp16_to_fp32(w[(size_t) (n * (K / 32))].d);
        y[n] = (float) (int32_t) acc * (sx * sw);
    }
}

struct Run {
    std::vector<float> y;
    bool ok = false;
};

// y = mul_mat(w, x) with w in `wbuft`, x [K][T][B] (filled by `fill`), on `n_threads`.
template <typename Fill>
Run run(ggml_backend_buffer_type_t wbuft, const std::vector<Block>& w, int64_t K, int64_t N, int64_t T, int64_t B,
        int n_threads, Fill fill) {
    Run r;
    ggml_context* cw = ggml_init({ ggml_tensor_overhead() * 2, nullptr, true });
    ggml_context* cx = ggml_init({ ggml_tensor_overhead() * 4 + ggml_graph_overhead(), nullptr, true });
    ggml_tensor* wt = ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, K, N);
    ggml_tensor* xt = ggml_new_tensor_3d(cx, GGML_TYPE_F32, K, T, B);
    ggml_tensor* yt = ggml_mul_mat(cx, wt, xt);
    ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, wbuft);
    ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_cpu_buffer_type());
    if (bw != nullptr && bx != nullptr) {
        ggml_backend_tensor_set(wt, w.data(), 0, ggml_nbytes(wt));
        fill((float*) xt->data);
        ggml_cgraph* gf = ggml_new_graph(cx);
        ggml_build_forward_expand(gf, yt);
        r.ok = compute(gf, n_threads);
        r.y.assign((const float*) yt->data, (const float*) yt->data + ggml_nelements(yt));
    }
    ggml_backend_buffer_free(bw);
    ggml_backend_buffer_free(bx);
    ggml_free(cw);
    ggml_free(cx);
    return r;
}

void fill_x(float* x, int64_t n, int seed) {
    for (int64_t i = 0; i < n; ++i) x[i] = 0.01f * (float) ((((i + seed) * 7919) % 211) - 105) + 0.003f * (float) (i % 7);
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

}  // namespace

int main() {
    std::fesetround(FE_TONEAREST);
    const char* e = std::getenv("GGML_CPU_ARMV6_I8");
    const bool want_on = !(e != nullptr && std::atoi(e) == 0);
#if defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON) && !defined(__aarch64__)
    const bool armv6 = true;
#else
    const bool armv6 = false;
#endif
    LOOM_CHECK(backend() != nullptr);
    ggml_backend_buffer_type_t buft = i8_buft();
    LOOM_CHECK((buft != nullptr) == (armv6 && want_on));
    if (buft == nullptr) {
        std::printf("CPU_ARMV6_I8 %s: nothing more to check here\n", armv6 ? "switched off" : "absent on this ISA");
        LOOM_TEST_REPORT_AND_RETURN();
    }

    // 1. The shapes.
    struct Shape { int64_t K, N, T, B; };
    // Below 16 columns the one-column kernel; from 16 the GEMM tiles (ggml-0042), whose K runs in chunks of
    // 512 (3072, 704) with the last one short (704), on an odd column count (17, 33) and a batched src1.
    const Shape shapes[] = { {32, 4, 1, 1}, {64, 6, 1, 1}, {288, 288, 1, 1}, {288, 13, 3, 1}, {1152, 20, 5, 1},
                             {3072, 12, 2, 1}, {96, 36, 7, 2}, {256, 8, 17, 1}, {288, 64, 40, 1}, {32, 4, 16, 1},
                             {3072, 12, 17, 1}, {704, 13, 33, 1}, {192, 24, 9, 2} };
    for (const Shape& s : shapes) {
        const std::vector<Block> w = make_weight(s.K, s.N, (int) (s.K + s.N), true);
        for (int threads : {1, 2}) {
            std::vector<float> x;
            Run r = run(buft, w, s.K, s.N, s.T, s.B, threads, [&](float* p) {
                fill_x(p, s.K * s.T * s.B, (int) s.N);
                x.assign(p, p + s.K * s.T * s.B);
            });
            LOOM_CHECK(r.ok);
            int bad = 0;
            std::vector<float> ref((size_t) s.N);
            for (int64_t c = 0; c < s.T * s.B; ++c) {
                reference(w, s.K, s.N, x.data() + c * s.K, ref.data());
                for (int64_t n = 0; n < s.N; ++n) bad += !same_bits(r.y[(size_t) (c * s.N + n)], ref[(size_t) n]);
            }
            if (bad) std::printf("K=%lld N=%lld T=%lld B=%lld threads=%d: %d outputs differ from the contract\n",
                                 (long long) s.K, (long long) s.N, (long long) s.T, (long long) s.B, threads, bad);
            LOOM_CHECK(bad == 0);
        }
    }

    // 2. The overflow bound: every product at its most negative, every x at full range, on both paths.
    for (int64_t T : {1, 16}) {
        const int64_t K = 3072, N = 4;
        std::vector<Block> w((size_t) (N * K / 32));
        for (Block& b : w) { b.d = ggml_fp32_to_fp16(0.001f); std::memset(b.qs, 0x80, 32); }
        Run r = run(buft, w, K, N, T, 1, 1, [&](float* p) { for (int64_t k = 0; k < K * T; ++k) p[k] = 1.5f; });
        LOOM_CHECK(r.ok);
        const int R = x_range(K);
        const float want = (float) (int32_t) (INT64_C(-128) * R * K) * ((1.5f / R) * ggml_fp16_to_fp32(w[0].d));
        for (int64_t i = 0; i < N * T; ++i) LOOM_CHECK(same_bits(r.y[(size_t) i], want));
    }

    // 3. A NaN poisons its own column only, on both paths.
    for (int64_t T : {3, 17}) {
        const int64_t K = 288, N = 8;
        const std::vector<Block> w = make_weight(K, N, 3, true);
        std::vector<float> x;
        Run r = run(buft, w, K, N, T, 1, 1, [&](float* p) {
            fill_x(p, K * T, 5);
            p[K + 17] = NAN;
            x.assign(p, p + K * T);
        });
        LOOM_CHECK(r.ok);
        std::vector<float> ref((size_t) N);
        for (int64_t c = 0; c < T; ++c) {
            if (c != 1) reference(w, K, N, x.data() + c * K, ref.data());
            for (int64_t n = 0; n < N; ++n) {
                const float got = r.y[(size_t) (c * N + n)];
                LOOM_CHECK(c == 1 ? std::isnan(got) : same_bits(got, ref[(size_t) n]));
            }
        }
    }

    // 5. A pointwise convolution (ggml-0043): ggml_conv_2d_direct_packed with a 1x1 kernel on channels-first
    //    data is the same GEMM, each output position a column read across the channels. Followed by a bias
    //    ADD, the pattern ggml's conv fusion would otherwise take, and read the packed kernel as Q8_0.
    for (const int64_t S : {1, 2})
    for (int64_t T : {5, 40, 71}) {
        const int64_t IC = 256, OC = 12, IT = (T - 1) * S + 1;   // a stride-S 1x1 conv reads every S-th input
        const std::vector<Block> w = make_weight(IC, OC, 21, true);
        ggml_context* cw = ggml_init({ ggml_tensor_overhead() * 2, nullptr, true });
        ggml_context* cx = ggml_init({ ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true });
        ggml_tensor* wt = ggml_new_tensor_2d(cw, GGML_TYPE_Q8_0, IC, OC);
        ggml_tensor* xt = ggml_new_tensor_4d(cx, GGML_TYPE_F32, IT, 1, IC, 1);
        ggml_tensor* bt = ggml_new_tensor_2d(cx, GGML_TYPE_F32, 1, OC);
        ggml_tensor* conv = ggml_conv_2d_direct_packed(cx, wt, xt, (int) S, 1, 0, 0, 1, 1, 1, 1, (int) IC);
        // [T, OC] + [1, OC]: the shape loom's lowering gives the bias ADD, which is what the fusion matches
        ggml_tensor* yt = ggml_add(cx, ggml_reshape_2d(cx, conv, T, OC), bt);
        ggml_backend_buffer_t bw = ggml_backend_alloc_ctx_tensors_from_buft(cw, buft);
        ggml_backend_buffer_t bx = ggml_backend_alloc_ctx_tensors_from_buft(cx, ggml_backend_cpu_buffer_type());
        LOOM_CHECK(bw != nullptr && bx != nullptr);
        ggml_backend_tensor_set(wt, w.data(), 0, ggml_nbytes(wt));
        fill_x((float*) xt->data, IC * IT, 31);
        for (int64_t o = 0; o < OC; ++o) ((float*) bt->data)[o] = 0.25f * (float) o - 1.0f;
        ggml_backend_dev_t dev = ggml_backend_get_device(backend());
        LOOM_CHECK(ggml_backend_dev_supports_op(dev, conv));
        ggml_cgraph* gf = ggml_new_graph(cx);
        ggml_build_forward_expand(gf, yt);
        for (int threads : {1, 2}) {   // the GEMM path quantizes runs of 32 positions, split across threads
            std::memset(yt->data, 0xff, ggml_nbytes(yt));
            LOOM_CHECK(compute(gf, threads));
            std::vector<float> col((size_t) IC), ref((size_t) OC);
            int bad = 0;
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t ic = 0; ic < IC; ++ic) col[(size_t) ic] = ((const float*) xt->data)[ic * IT + t * S];
                reference(w, IC, OC, col.data(), ref.data());
                for (int64_t o = 0; o < OC; ++o) {
                    bad += !same_bits(((const float*) yt->data)[o * T + t], ref[(size_t) o] + ((const float*) bt->data)[o]);
                }
            }
            if (bad) std::printf("pointwise conv S=%lld T=%lld threads=%d: %d outputs differ from the contract\n",
                                 (long long) S, (long long) T, threads, bad);
            LOOM_CHECK(bad == 0);
        }
        // A kernel of three taps is not this GEMM: declined, so the engine never builds it on the packed kernel.
        ggml_tensor* x3 = ggml_new_tensor_4d(cx, GGML_TYPE_F32, T + 2, 1, IC / 4, 1);
        ggml_tensor* k3 = ggml_conv_2d_direct_packed(cx, wt, x3, 1, 1, 0, 0, 1, 1, 4, 1, (int) (IC / 4));
        LOOM_CHECK(!ggml_backend_dev_supports_op(dev, k3));
        ggml_backend_buffer_free(bw);
        ggml_backend_buffer_free(bx);
        ggml_free(cw);
        ggml_free(cx);
    }

    // 4. Blocks with different scales in a row: stored as given, computed by ggml's Q8_0 mul_mat.
    {
        const int64_t K = 256, N = 12, T = 5;
        const std::vector<Block> w = make_weight(K, N, 9, false);
        auto fill = [&](float* p) { fill_x(p, K * T, 11); };
        Run a = run(buft, w, K, N, T, 1, 1, fill);
        Run b = run(ggml_backend_cpu_buffer_type(), w, K, N, T, 1, 1, fill);
        LOOM_CHECK(a.ok && b.ok);
        LOOM_CHECK(a.y.size() == b.y.size() && std::memcmp(a.y.data(), b.y.data(), a.y.size() * 4) == 0);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
