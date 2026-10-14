// ADR-077 GATE 1: CITRINET-256'S DEPTHWISE CONVOLUTIONS ON THE ARM1176, INT8 AGAINST A FAIR F32.
//
// The gate asks whether any op in Citrinet-256's non-GEMM time gets faster as int8 than as VFP F32.
// The profile (rc16, Q8_0, jfk, one thread) puts 4.90 s of the 8.4 s non-GEMM in CONV_2D_DW; the
// rest is the padding-mask MUL (1.31 s, identity at batch 1), the 1x1 conv's bias ADD (0.78 s) and
// RELU (0.59 s) -- all three fusable in either precision -- and ~0.8 s of front end. So this replays
// the whole jfk depthwise schedule (107 calls, K = 5..41, T = 1101/551/276/138, stride-2 at the three
// block boundaries) in three arms, verified before timing:
//
//   ggml  ggml_conv_2d_dw_direct through a graph, one thread: what loom runs (ggml-0021/0022, whose
//         interior is a scalar ggml_vec_mad_f32 sweep per tap on a core with no NEON)
//   f32   8 outputs per register block, the window RENAMED by an 8-tap unroll rather than moved: a
//         first version that slid it with 7 vmov per tap ran 0.99 s instead of 0.67 s. Equal to ggml
//         bit for bit (same accumulation order, VFPv2's vmla is not fused)
//   i8    int8 x int8 -> int32 by SMLAD over int16 pairs: E/O rows (even/odd-aligned pairs) built
//         straight from the int8 row with SXTB16 + PKHBT/PKHTB, the input zero point folded into the
//         accumulator's start, SMMULR + rounding shift + SSAT requantization. Exact against a scalar
//         int64 reference (0 / 9227280). A first version that widened with a scalar loop and
//         requantized in int64 ran 0.91 s, prep-bound.
//
// Pi Zero W, 2026-10-14, best of 5, two rounds agreeing within 2% (default gcc tuning = the wheel's;
// -mtune=arm1176jzf-s measured too and buys nothing):
//
//     ggml                       3.96 s    0.044 GMAC/s
//     f32                        0.67 s    0.262          5.9x ggml
//     f32, RunFast (FZ|DN)       0.50 s    0.348          7.9x ggml
//     i8                         0.68 s    0.258          0.74x of f32 + RunFast
//
// The ceilings (scripts/bench44.c): SMLAD with the loads a kernel needs ~1.07 GMAC/s, VFP vmla 0.41
// GMAC/s in IEEE mode and 0.59-0.68 in RunFast. The i8 kernel is at ~24% of its ceiling and f32 at
// ~55%, so a tuned int8 kernel could plausibly reach ~2x f32 here -- worth at most ~0.25 s of jfk.
//
//   build in loom-armv6-build:bookworm against an ARMv6 ggml (GGML_NATIVE=OFF):
//     g++ -O3 -std=c++17 bench43.cpp -I$GGML/include -L$GGML_BUILD/src -lggml -lggml-base -lggml-cpu \
//         -Wl,-rpath,'$ORIGIN/lib' -o bench43
//   ./bench43 [reps=3] [arms=gfi] [runfast=0]
#include "ggml.h"
#include "ggml-cpu.h"
#include <arm_acle.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Call { int C, T, K, s; int To() const { return (T + 2 * (K / 2) - K) / s + 1; } };

static std::vector<Call> jfk_schedule() {
    std::vector<Call> v;
    v.push_back({80, 1101, 5, 1});                                          // prolog
    for (int r = 0; r < 4; ++r) v.push_back({256, 1101, 11, 1});
    v.push_back({256, 1101, 11, 2});                                       // -> 551
    for (int k : {13, 15, 17, 19, 21}) for (int r = 0; r < 5; ++r) v.push_back({256, 551, k, 1});
    for (int r = 0; r < 4; ++r) v.push_back({256, 551, 13, 1});
    v.push_back({256, 551, 13, 2});                                        // -> 276
    for (int k : {15, 17, 19, 21, 23, 25}) for (int r = 0; r < 5; ++r) v.push_back({256, 276, k, 1});
    for (int r = 0; r < 4; ++r) v.push_back({256, 276, 25, 1});
    v.push_back({256, 276, 25, 2});                                        // -> 138
    for (int k : {27, 29, 31, 33, 35, 37, 39}) for (int r = 0; r < 5; ++r) v.push_back({256, 138, k, 1});
    v.push_back({256, 138, 41, 1});                                        // epilog
    return v;
}

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

// ---------------------------------------------------------------- ggml arm
struct GgmlCall { ggml_context * ctx; ggml_cgraph * gf; ggml_tensor * y; };
static GgmlCall ggml_build(const Call & c, const float * x, const float * w) {
    const size_t mem = (size_t) c.C * (c.T + c.To() + c.K) * 4 + 1024 * 1024;
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * kn = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.K, 1, 1, c.C);
    ggml_tensor * in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.T, 1, c.C, 1);
    memcpy(kn->data, w, (size_t) c.K * c.C * 4);
    memcpy(in->data, x, (size_t) c.T * c.C * 4);
    ggml_tensor * y = ggml_conv_2d_dw_direct(ctx, kn, in, c.s, 1, c.K / 2, 0, 1, 1);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    return {ctx, gf, y};
}

// ---------------------------------------------------------------- f32 arm
// One channel. xp: the row with K/2 zeros each side. 8 accumulators so VFP11's 8-cycle FMAC latency is
// covered; the window x0..x7 slides one element per tap (one new load per tap per 8 outputs).
static void dw_f32_row(const float * xp, const float * w, int K, int s, int To, float * y) {
    int t = 0;
    if (s == 1) {
        for (; t + 8 <= To; t += 8) {
            const float * p = xp + t;
            // slot m holds p[k + j + i] where m = (i + j) & 7 at tap j of an 8-tap group starting at k
            // (k % 8 == 0): the newest element goes into the slot the oldest one just vacated, so the
            // window is renamed by the unroll and never moved.
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
            float s0 = p[0], s1 = p[1], s2 = p[2], s3 = p[3], s4 = p[4], s5 = p[5], s6 = p[6], s7;
            int k = 0;
#define TAP(J, N, X0, X1, X2, X3, X4, X5, X6, X7) { \
                N = p[k + (J) + 7]; const float wk = w[k + (J)]; \
                a0 += wk * X0; a1 += wk * X1; a2 += wk * X2; a3 += wk * X3; \
                a4 += wk * X4; a5 += wk * X5; a6 += wk * X6; a7 += wk * X7; }
            for (; k + 8 <= K; k += 8) {
                TAP(0, s7, s0, s1, s2, s3, s4, s5, s6, s7)
                TAP(1, s0, s1, s2, s3, s4, s5, s6, s7, s0)
                TAP(2, s1, s2, s3, s4, s5, s6, s7, s0, s1)
                TAP(3, s2, s3, s4, s5, s6, s7, s0, s1, s2)
                TAP(4, s3, s4, s5, s6, s7, s0, s1, s2, s3)
                TAP(5, s4, s5, s6, s7, s0, s1, s2, s3, s4)
                TAP(6, s5, s6, s7, s0, s1, s2, s3, s4, s5)
                TAP(7, s6, s7, s0, s1, s2, s3, s4, s5, s6)
            }
#undef TAP
            for (; k < K; ++k) {
                const float wk = w[k]; const float * q = p + k;
                a0 += wk * q[0]; a1 += wk * q[1]; a2 += wk * q[2]; a3 += wk * q[3];
                a4 += wk * q[4]; a5 += wk * q[5]; a6 += wk * q[6]; a7 += wk * q[7];
            }
            y[t] = a0; y[t + 1] = a1; y[t + 2] = a2; y[t + 3] = a3;
            y[t + 4] = a4; y[t + 5] = a5; y[t + 6] = a6; y[t + 7] = a7;
        }
    } else {
        for (; t + 4 <= To; t += 4) {
            const float * p = xp + t * s;
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int k = 0; k < K; ++k) {
                const float wk = w[k];
                a0 += wk * p[k]; a1 += wk * p[k + 2]; a2 += wk * p[k + 4]; a3 += wk * p[k + 6];
            }
            y[t] = a0; y[t + 1] = a1; y[t + 2] = a2; y[t + 3] = a3;
        }
    }
    for (; t < To; ++t) {
        float a = 0;
        for (int k = 0; k < K; ++k) a += w[k] * xp[t * s + k];
        y[t] = a;
    }
}

static void dw_f32(const Call & c, const float * x, const float * w, float * y, std::vector<float> & xp) {
    const int p = c.K / 2, To = c.To();
    xp.assign(c.T + 2 * p + 16, 0.0f);
    for (int ch = 0; ch < c.C; ++ch) {
        memcpy(xp.data() + p, x + (size_t) ch * c.T, (size_t) c.T * 4);
        dw_f32_row(xp.data(), w + (size_t) ch * c.K, c.K, c.s, To, y + (size_t) ch * To);
    }
}

// ---------------------------------------------------------------- i8 arm
// Per channel: x int8 [T] -> E = padded int16 row packed as words (E[2j], E[2j+1]) and O = E shifted by
// one. Weights pre-packed per channel as Kp/2 words (w[2m], w[2m+1]), Kp = K rounded up to even.
// out[2a] = sum_m W[m] . E32[a+m], out[2a+1] = sum_m W[m] . O32[a+m]  (SMLAD: lo*lo + hi*hi).
struct I8Weights { std::vector<uint32_t> w; int Kp; std::vector<int32_t> mult, sh, bias; };

static inline uint32_t pack16(int16_t lo, int16_t hi) { return (uint16_t) lo | ((uint32_t) (uint16_t) hi << 16); }

// y = sat8(round(round(acc * M / 2^32) / 2^sh) + zp): SMMULR, a rounding shift, SSAT.
static inline int32_t smmulr(int32_t a, int32_t b) {
    int32_t r;
    __asm__("smmulr %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
static inline int8_t requant(int32_t acc, int32_t mult, int sh, int32_t zp) {
    const int32_t v = (smmulr(acc, mult) + (1 << (sh - 1))) >> sh;
    return (int8_t) __ssat(v + zp, 8);
}
static inline int8_t requant_ref(int64_t acc, int32_t mult, int sh, int32_t zp) {
    const int64_t v = (((acc * mult + ((int64_t) 1 << 31)) >> 32) + ((int64_t) 1 << (sh - 1))) >> sh;
    const int64_t y = v + zp;
    return (int8_t) (y < -128 ? -128 : y > 127 ? 127 : y);
}

static void dw_i8_row(const uint32_t * E, const uint32_t * O, const uint32_t * W, int M, int s, int To,
                      int32_t mult, int shift, int32_t zp, int32_t bias, int8_t * y) {
    int t = 0;
    if (s == 1) {
        for (; t + 4 <= To; t += 4) {
            const int a = t >> 1;                     // t = 4b -> a = 2b
            const uint32_t * e = E + a;
            const uint32_t * o = O + a;
            int32_t c0 = bias, c1 = bias, c2 = bias, c3 = bias;
            uint32_t e0 = e[0], o0 = o[0], e1, o1;
            int m = 0;
            for (; m + 2 <= M; m += 2) {              // unrolled by 2 so (e0,o0) and (e1,o1) swap roles
                const uint32_t w0 = W[m], w1 = W[m + 1];
                e1 = e[m + 1]; o1 = o[m + 1];
                c0 = __smlad(w0, e0, c0); c1 = __smlad(w0, o0, c1);
                c2 = __smlad(w0, e1, c2); c3 = __smlad(w0, o1, c3);
                e0 = e[m + 2]; o0 = o[m + 2];
                c0 = __smlad(w1, e1, c0); c1 = __smlad(w1, o1, c1);
                c2 = __smlad(w1, e0, c2); c3 = __smlad(w1, o0, c3);
            }
            if (m < M) {
                const uint32_t w0 = W[m];
                e1 = e[m + 1]; o1 = o[m + 1];
                c0 = __smlad(w0, e0, c0); c1 = __smlad(w0, o0, c1);
                c2 = __smlad(w0, e1, c2); c3 = __smlad(w0, o1, c3);
            }
            y[t] = requant(c0, mult, shift, zp); y[t + 1] = requant(c1, mult, shift, zp);
            y[t + 2] = requant(c2, mult, shift, zp); y[t + 3] = requant(c3, mult, shift, zp);
        }
    } else {                                          // s == 2: every output starts on an even element
        for (; t + 4 <= To; t += 4) {
            const uint32_t * e = E + t;
            int32_t c0 = bias, c1 = bias, c2 = bias, c3 = bias;
            uint32_t e0 = e[0], e1 = e[1], e2 = e[2], e3;
            int m = 0;
#define STEP(J, N, X0, X1, X2, X3) { N = e[m + (J) + 3]; const uint32_t wm = W[m + (J)]; \
                c0 = __smlad(wm, X0, c0); c1 = __smlad(wm, X1, c1); c2 = __smlad(wm, X2, c2); c3 = __smlad(wm, X3, c3); }
            for (; m + 4 <= M; m += 4) {
                STEP(0, e3, e0, e1, e2, e3)
                STEP(1, e0, e1, e2, e3, e0)
                STEP(2, e1, e2, e3, e0, e1)
                STEP(3, e2, e3, e0, e1, e2)
            }
#undef STEP
            for (; m < M; ++m) {
                const uint32_t wm = W[m]; const uint32_t * q = e + m;
                c0 = __smlad(wm, q[0], c0); c1 = __smlad(wm, q[1], c1); c2 = __smlad(wm, q[2], c2); c3 = __smlad(wm, q[3], c3);
            }
            y[t] = requant(c0, mult, shift, zp); y[t + 1] = requant(c1, mult, shift, zp);
            y[t + 2] = requant(c2, mult, shift, zp); y[t + 3] = requant(c3, mult, shift, zp);
        }
    }
    for (; t < To; ++t) {
        const int st = t * s;
        const uint32_t * src = (st & 1) ? O + (st >> 1) : E + (st >> 1);
        int32_t c = bias;
        for (int m = 0; m < M; ++m) c = __smlad(W[m], src[m], c);
        y[t] = requant(c, mult, shift, zp);
    }
}

static void dw_i8(const Call & c, const int8_t * x, const I8Weights & q, int32_t zp_in, int32_t zp_out, int8_t * y,
                  std::vector<uint32_t> & xbw, std::vector<uint32_t> & E, std::vector<uint32_t> & O) {
    const int p = c.K / 2, To = c.To(), M = q.Kp / 2;
    const int L = (c.T + 2 * p + 2 * M + 16 + 3) & ~3;    // bytes; E/O words read past the row stay zp
    xbw.resize(L / 4 + 1);
    int8_t * xb = (int8_t *) xbw.data();
    memset(xb, zp_in, (size_t) L + 4);                // pads = zp_in, i.e. a real zero
    E.resize(L / 2 + 2); O.resize(L / 2 + 2);
    for (int ch = 0; ch < c.C; ++ch) {
        memcpy(xb + p, x + (size_t) ch * c.T, (size_t) c.T);
        // bytes b0..b3 of word j -> E[2j] = (b0,b1), E[2j+1] = (b2,b3), O[2j] = (b1,b2), O[2j+1] = (b3,b4)
        uint32_t prev13 = 0;
        for (int j = 0; j <= L / 4; ++j) {
            const uint32_t u = xbw[j];
            const uint32_t s02 = __sxtb16(u), s13 = __sxtb16((u >> 8) | (u << 24));
            if (j > 0) O[2 * j - 1] = (prev13 >> 16) | (s02 << 16);
            if (j == L / 4) break;
            E[2 * j]     = (s02 & 0xFFFFu) | (s13 << 16);
            E[2 * j + 1] = (s02 >> 16) | (s13 & 0xFFFF0000u);
            O[2 * j]     = (s13 & 0xFFFFu) | (s02 & 0xFFFF0000u);
            prev13 = s13;
        }
        dw_i8_row(E.data(), O.data(), q.w.data() + (size_t) ch * M, M, c.s, To, q.mult[ch], q.sh[ch], zp_out,
                  q.bias[ch], y + (size_t) ch * To);
    }
}

// Scalar int64 reference with the same requantization.
static void dw_i8_ref(const Call & c, const int8_t * x, const int8_t * w8, const I8Weights & q, int32_t zp_in, int32_t zp_out, int8_t * y) {
    const int p = c.K / 2, To = c.To();
    for (int ch = 0; ch < c.C; ++ch)
        for (int t = 0; t < To; ++t) {
            int64_t a = 0;
            for (int k = 0; k < c.K; ++k) {
                const int i = t * c.s + k - p;
                if (i >= 0 && i < c.T) a += (int64_t) w8[(size_t) ch * c.K + k] * (x[(size_t) ch * c.T + i] - zp_in);
            }
            y[(size_t) ch * To + t] = requant_ref(a, q.mult[ch], q.sh[ch], zp_out);
        }
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 3;
    const char * arms = argc > 2 ? argv[2] : "gfi";
    const bool doG = strchr(arms, 'g'), doF = strchr(arms, 'f'), doI = strchr(arms, 'i');
    if (argc > 3 && atoi(argv[3])) {                  // RunFast: FZ | DN (exceptions are already off)
        unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); f |= (1u << 24) | (1u << 25);
        __asm__ volatile("vmsr fpscr, %0" : : "r"(f));
        printf("RunFast on\n");
    }
    const auto sched = jfk_schedule();
    double macs = 0;
    for (auto & c : sched) macs += (double) c.C * c.To() * c.K;
    printf("jfk depthwise schedule: %zu calls, %.1f MMAC\n", sched.size(), macs / 1e6);

    std::mt19937 rng(42);
    std::normal_distribution<float> nx(0, 1), nw(0, 0.15f);
    double tg = 0, tf = 0, ti = 0, worst_rel = 0;
    long i8_mismatch = 0, i8_total = 0;
    std::vector<float> xp; std::vector<uint32_t> xbw, E, O;

    printf("%4s %5s %3s %2s %5s | %10s %10s %10s\n", "C", "T", "K", "s", "To", "ggml ms", "f32 ms", "i8 ms");
    for (size_t ci = 0; ci < sched.size(); ++ci) {
        const Call & c = sched[ci];
        const int To = c.To();
        std::vector<float> x((size_t) c.C * c.T), w((size_t) c.C * c.K), yf((size_t) c.C * To);
        for (auto & v : x) v = nx(rng);
        for (auto & v : w) v = nw(rng);

        double bg = 1e30, bf = 1e30, bi = 1e30;
        if (doG) {
            GgmlCall g = ggml_build(c, x.data(), w.data());
            for (int r = 0; r < reps; ++r) { auto t0 = clk::now(); ggml_graph_compute_with_ctx(g.ctx, g.gf, 1); bg = std::min(bg, ms_since(t0)); }
            if (doF) {
                dw_f32(c, x.data(), w.data(), yf.data(), xp);
                const float * yg = (const float *) g.y->data;
                double mx = 0, num = 0;
                for (size_t k = 0; k < yf.size(); ++k) { mx = std::max(mx, (double) fabsf(yg[k])); num = std::max(num, (double) fabsf(yg[k] - yf[k])); }
                worst_rel = std::max(worst_rel, num / mx);
            }
            ggml_free(g.ctx);
        }
        if (doF) for (int r = 0; r < reps; ++r) { auto t0 = clk::now(); dw_f32(c, x.data(), w.data(), yf.data(), xp); bf = std::min(bf, ms_since(t0)); }
        if (doI) {
            // Per-tensor asymmetric activations, per-channel symmetric weights; scales from the data.
            const float sx = 8.0f / 255.0f; const int32_t zx = -3, zy = 5;
            std::vector<int8_t> x8(x.size()), w8(w.size()), y8((size_t) c.C * To), yr(y8.size());
            for (size_t k = 0; k < x.size(); ++k) x8[k] = (int8_t) std::max(-128.0f, std::min(127.0f, roundf(x[k] / sx) + zx));
            I8Weights q; q.Kp = (c.K + 1) & ~1;
            q.w.assign((size_t) c.C * q.Kp / 2, 0); q.mult.resize(c.C); q.sh.resize(c.C); q.bias.resize(c.C);
            for (int ch = 0; ch < c.C; ++ch) {
                float amax = 0;
                for (int k = 0; k < c.K; ++k) amax = std::max(amax, fabsf(w[(size_t) ch * c.K + k]));
                const float sw = amax / 127.0f;
                std::vector<int16_t> wk(q.Kp, 0);
                for (int k = 0; k < c.K; ++k) { w8[(size_t) ch * c.K + k] = (int8_t) roundf(w[(size_t) ch * c.K + k] / sw); wk[k] = w8[(size_t) ch * c.K + k]; }
                for (int m = 0; m < q.Kp / 2; ++m) q.w[(size_t) ch * q.Kp / 2 + m] = pack16(wk[2 * m], wk[2 * m + 1]);
                const double sy = 6.0 / 255.0;                       // output scale
                int e; const double fr = frexp(sx * sw / sy, &e);    // m = fr * 2^e, fr in [0.5, 1)
                q.mult[ch] = (int32_t) std::min((long long) INT32_MAX, llround(fr * 2147483648.0));
                q.sh[ch] = -e - 1;                                   // smmulr already divides by 2^32 = 2^31 * 2
                int32_t sumw = 0;
                for (int k = 0; k < c.K; ++k) sumw += w8[(size_t) ch * c.K + k];
                q.bias[ch] = -zx * sumw;
            }
            for (int r = 0; r < reps; ++r) { auto t0 = clk::now(); dw_i8(c, x8.data(), q, zx, zy, y8.data(), xbw, E, O); bi = std::min(bi, ms_since(t0)); }
            dw_i8_ref(c, x8.data(), w8.data(), q, zx, zy, yr.data());
            for (size_t k = 0; k < y8.size(); ++k) i8_mismatch += y8[k] != yr[k];
            i8_total += (long) y8.size();
        }
        if (doG) tg += bg;
        if (doF) tf += bf;
        if (doI) ti += bi;
        printf("%4d %5d %3d %2d %5d | %10.2f %10.2f %10.2f\n", c.C, c.T, c.K, c.s, To, doG ? bg : 0, doF ? bf : 0, doI ? bi : 0);
        fflush(stdout);
    }
    printf("TOTAL ms   ggml %.1f   f32 %.1f   i8 %.1f\n", tg, tf, ti);
    if (doG) printf("ggml  %.3f GMAC/s\n", macs / tg / 1e6);
    if (doF) printf("f32   %.3f GMAC/s   (%.2fx ggml)\n", macs / tf / 1e6, doG ? tg / tf : 0);
    if (doI) printf("i8    %.3f GMAC/s   (%.2fx ggml, %.2fx f32)\n", macs / ti / 1e6, doG ? tg / ti : 0, doF ? tf / ti : 0);
    if (doG && doF) printf("f32 vs ggml: max |diff| / max |y| = %.3g\n", worst_rel);
    if (doI) printf("i8 vs int64 reference: %ld / %ld mismatches\n", i8_mismatch, i8_total);
    return 0;
}
