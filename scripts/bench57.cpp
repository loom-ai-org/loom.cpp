// ADR-079 P4.32f f2 step 3a (ggml-0039): the transcendental activations on the VFP11 (ARMv6), per element,
// data in L1 (2048 floats, 256 passes).
//
//   GELU_ERF   ggml-0010's scalar rational erf (what ARMv6 runs: there is no SIMD body), the same arithmetic
//              on four or eight elements interleaved in C (clamps as VFP compares or integer tests), and
//              the generated short-vector kernel. All must be bit-identical to the scalar tail.
//   SIGMOID, SILU, TANH   glibc (expf, tanhf), an own expf in C, and the generated kernels, which must match
//              the scalar reference below (Cephes' construction, the kernels' exact operation order) bit for
//              bit. Their accuracy against f64 and glibc is scripts/bench57_accuracy.cpp's (x86).
//
// The sweeps cover every 61st float bit pattern (~70M). In IEEE mode the VFP11 bounces denormal operands to
// the kernel's support code, which rounds them differently (Retro-086): run with `runfast` (ggml-0031 sets
// it on every ARMv6 compute thread) for the bit-identity verdict.
//
// Build (ARMv6, loom-armv6-build image), with <ggml> a tree carrying ggml-0039:
//   c++ -O3 -std=c++17 -ffp-contract=off -I<ggml>/src/ggml-cpu bench57.cpp -o bench57
// Run on the board: ./bench57 [reps] [runfast]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define NOINL __attribute__((noinline))
#if defined(__arm__)
#include "arch/arm/armv6-act-f32.inc"
NOINL static void gelu_asm(int n, float * y, const float * x);
#endif
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static inline uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// ---- GELU_ERF, ggml-0010's scalar tail --------------------------------------------------------------
static const float SQRT_2_INV = 0.70710678118654752440084436210484f;
#define GGML_ERF_CLAMP 4.0f
static const float P[6] = {1.128379046e+00f, 1.923043769e-01f, 5.298133013e-02f,
                           3.844561124e-03f, 2.861880010e-04f, 2.027359838e-06f};
static const float Q[6] = {1.000000000e+00f, 5.037572164e-01f, 1.148780232e-01f,
                           1.512413709e-02f, 1.181028205e-03f, 3.792931260e-05f};

NOINL static void gelu_stock(int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        const float xi = x[i];
        float z = xi*SQRT_2_INV;
        z = z <  GGML_ERF_CLAMP ? z :  GGML_ERF_CLAMP;
        z = z > -GGML_ERF_CLAMP ? z : -GGML_ERF_CLAMP;
        const float w = z*z;
        const float p = ((((P[5]*w + P[4])*w + P[3])*w + P[2])*w + P[1])*w + P[0];
        const float q = ((((Q[5]*w + Q[4])*w + Q[3])*w + Q[2])*w + Q[1])*w + Q[0];
        float e = z*(p/q);
        e = e <  1.0f ? e :  1.0f;
        e = e > -1.0f ? e : -1.0f;
        y[i] = 0.5f*xi*(1.0f + e);
    }
}

// clamp to [-c, c] exactly as `v < c ? v : c` then `v > -c ? v : -c` decide it: NaN (either sign) -> c
static inline float clamp_bits(float v, uint32_t cb) {
    const uint32_t u = fbits(v), a = u & 0x7fffffffu;
    if (a > 0x7f800000u) return bitsf(cb);
    if (a >= cb) return bitsf(cb | (u & 0x80000000u));
    return v;
}

template <int U, bool INT>
NOINL static void gelu_il(int n, float * y, const float * x) {
    int i = 0;
    for (; i + U <= n; i += U) {
        float z[U], w[U], p[U], q[U], e[U];
        for (int k = 0; k < U; ++k) {
            z[k] = x[i + k]*SQRT_2_INV;
            if (INT) z[k] = clamp_bits(z[k], 0x40800000u);
            else { z[k] = z[k] < GGML_ERF_CLAMP ? z[k] : GGML_ERF_CLAMP; z[k] = z[k] > -GGML_ERF_CLAMP ? z[k] : -GGML_ERF_CLAMP; }
        }
        for (int k = 0; k < U; ++k) w[k] = z[k]*z[k];
        for (int k = 0; k < U; ++k) { p[k] = P[5]*w[k] + P[4]; q[k] = Q[5]*w[k] + Q[4]; }
        for (int k = 0; k < U; ++k) { p[k] = p[k]*w[k] + P[3]; q[k] = q[k]*w[k] + Q[3]; }
        for (int k = 0; k < U; ++k) { p[k] = p[k]*w[k] + P[2]; q[k] = q[k]*w[k] + Q[2]; }
        for (int k = 0; k < U; ++k) { p[k] = p[k]*w[k] + P[1]; q[k] = q[k]*w[k] + Q[1]; }
        for (int k = 0; k < U; ++k) { p[k] = p[k]*w[k] + P[0]; q[k] = q[k]*w[k] + Q[0]; }
        for (int k = 0; k < U; ++k) e[k] = z[k]*(p[k]/q[k]);
        for (int k = 0; k < U; ++k) {
            if (INT) e[k] = clamp_bits(e[k], 0x3f800000u);
            else { e[k] = e[k] < 1.0f ? e[k] : 1.0f; e[k] = e[k] > -1.0f ? e[k] : -1.0f; }
        }
        for (int k = 0; k < U; ++k) y[i + k] = 0.5f*x[i + k]*(1.0f + e[k]);
    }
    gelu_stock(n - i, y + i, x + i);
}

#if defined(__arm__)
NOINL static void gelu_asm(int n, float * y, const float * x) {
    float sc[8];
    ggml_armv6_act_args a = { x, y, sc, n / 8 };
    if (n >= 8) ggml_armv6_gelu_erf_f32(&a);
    gelu_stock(n % 8, y + n / 8 * 8, x + n / 8 * 8);
}
#endif
// ---- the reference the asm must match bit for bit (act_ref.cpp)
static inline uint32_t fb(float f) { return fbits(f); }
static inline float bf(uint32_t u) { return bitsf(u); }
// e^v for v in any range; the integer pass handles |v| beyond the core's reach
static const float LOG2E = 1.44269504088896341f, MAGIC = 12582912.0f;   // 1.5 * 2^23
static const float LN2HI = 0.693359375f, LN2LO = -2.12194440e-4f;        // Cephes' split
static const float C0 = 1.9875691500e-4f, C1 = 1.3981999507e-3f, C2 = 8.3334519073e-3f,
                   C3 = 4.1665795894e-2f, C4 = 1.6666665459e-1f, C5 = 5.0000001201e-1f;
static inline float exp_core(float v) {
    const float t  = v * LOG2E + MAGIC;
    const float nf = t - MAGIC;
    float r = v - nf * LN2HI;
    r = r - nf * LN2LO;
    float p = C0 * r + C1;
    p = p * r + C2; p = p * r + C3; p = p * r + C4; p = p * r + C5;
    const float r2 = r * r;
    p = p * r2;
    p = p + r;
    p = p + 1.0f;
    // integer pass
    const uint32_t a = fb(v) & 0x7fffffffu;
    if (a > 0x7f800000u) return v;                               // NaN
    if ((fb(v) >> 31) && a >= fb(86.98f)) return 0.0f;  // n >= -125 above this (a sum past p's exponent field's bottom wraps into the sign)
    if (!(fb(v) >> 31) && a >= fb(88.37f)) return INFINITY;  // n <= 127 below this
    const int32_t n = (int32_t) (fb(t) - fb(MAGIC));
    return bf(fb(p) + ((uint32_t) n << 23));
}
static inline float sigm(float x) { return 1.0f / (exp_core(-x) + 1.0f); }
static inline float silu(float x) { return x / (exp_core(-x) + 1.0f); }

static const float T0 = -5.70498872745e-3f, T1 = 2.06390887954e-2f, T2 = -5.37397155531e-2f,
                   T3 = 1.33314422036e-1f, T4 = -3.33332819422e-1f;
static inline float tanh_(float x) {
    const float ax = fabsf(x);
    // small: x + x^3 P(x^2)
    const float z = x * x;
    float s = T0 * z + T1; s = s * z + T2; s = s * z + T3; s = s * z + T4;
    s = s * z; s = s * x; s = s + x;
    // large: 1 - 2 / (e^{2|x|} + 1), sign of x
    const float e = exp_core(ax + ax);
    float l = 2.0f / (e + 1.0f);
    l = 1.0f - l;
    const uint32_t a = fb(x) & 0x7fffffffu;
    if (a > 0x7f800000u) return x;
    if (a < fb(0.625f)) return bf(fb(s) | (fb(x) & 0x80000000u));
    return bf(fb(l) | (fb(x) & 0x80000000u));
}

NOINL static void tanh_ref(int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = tanh_(x[i]); }
NOINL static void sigm_ref(int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sigm(x[i]); }
NOINL static void silu_ref(int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = silu(x[i]); }
#if defined(__arm__)
NOINL static void sigm_asm(int n, float * y, const float * x) {
    float sc[32]; ggml_armv6_act_args a = { x, y, sc, n / 8 };
    if (n >= 8) ggml_armv6_sigmoid_f32(&a);
    sigm_ref(n % 8, y + n / 8 * 8, x + n / 8 * 8);
}
NOINL static void tanh_asm(int n, float * y, const float * x) {
    float sc[32]; ggml_armv6_act_args a = { x, y, sc, n / 8 };
    if (n >= 8) ggml_armv6_tanh_f32(&a);
    tanh_ref(n % 8, y + n / 8 * 8, x + n / 8 * 8);
}
NOINL static void silu_asm(int n, float * y, const float * x) {
    float sc[32]; ggml_armv6_act_args a = { x, y, sc, n / 8 };
    if (n >= 8) ggml_armv6_silu_f32(&a);
    silu_ref(n % 8, y + n / 8 * 8, x + n / 8 * 8);
}
#endif
// ---- exp-based: libm ----------------------------------------------------------------------------------
NOINL static void expf_libm(int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = expf(x[i]); }
NOINL static void tanh_libm(int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = tanhf(x[i]); }
NOINL static void sigm_libm(int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
NOINL static void silu_libm(int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i] / (1.0f + expf(-x[i])); }

// ---- an own expf: n = round(x log2 e), r = x - n ln2 (Cody-Waite), e^r by Taylor to r^6, times 2^n --
static inline float fexp(float x) {
    x = x < 88.37f ? x : 88.37f;
    x = x > -87.3f ? x : -87.3f;   // below: e^x is under the smallest normal; FZ flushes it anyway
    const float t = x * 1.44269504f + 12582912.0f;
    const int32_t n = (int32_t) (fbits(t) - fbits(12582912.0f));
    const float nf = t - 12582912.0f;
    const float r = (x - nf * 0.693145752f) - nf * 1.42860677e-06f;
    const float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.66666667e-1f + r * (4.16666667e-2f + r * (8.33333333e-3f + r * 1.38888889e-3f)))));
    return bitsf(fbits(p) + ((uint32_t) n << 23));
}
template <int U> NOINL static void expf_own(int n, float * y, const float * x) {
    int i = 0;
    for (; i + U <= n; i += U) { float a[U]; for (int k = 0; k < U; ++k) a[k] = fexp(x[i + k]); for (int k = 0; k < U; ++k) y[i + k] = a[k]; }
    for (; i < n; ++i) y[i] = fexp(x[i]);
}
template <int U> NOINL static void sigm_own(int n, float * y, const float * x) {
    int i = 0;
    for (; i + U <= n; i += U) { float a[U]; for (int k = 0; k < U; ++k) a[k] = 1.f / (1.f + fexp(-x[i + k])); for (int k = 0; k < U; ++k) y[i + k] = a[k]; }
    for (; i < n; ++i) y[i] = 1.f / (1.f + fexp(-x[i]));
}
template <int U> NOINL static void silu_own(int n, float * y, const float * x) {
    int i = 0;
    for (; i + U <= n; i += U) { float a[U]; for (int k = 0; k < U; ++k) a[k] = x[i + k] / (1.f + fexp(-x[i + k])); for (int k = 0; k < U; ++k) y[i + k] = a[k]; }
    for (; i < n; ++i) y[i] = x[i] / (1.f + fexp(-x[i]));
}
// tanh(x) = 1 - 2 / (e^{2x} + 1): speed only here (it cancels near 0; a real kernel needs a polynomial there)
template <int U> NOINL static void tanh_own(int n, float * y, const float * x) {
    int i = 0;
    for (; i + U <= n; i += U) { float a[U]; for (int k = 0; k < U; ++k) a[k] = 1.f - 2.f / (fexp(2.f * x[i + k]) + 1.f); for (int k = 0; k < U; ++k) y[i + k] = a[k]; }
    for (; i < n; ++i) y[i] = 1.f - 2.f / (fexp(2.f * x[i]) + 1.f);
}

template <class F> static double ns_per(F f, int n_total, int reps) {
    f();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) { const double t0 = now(); f(); t.push_back((now() - t0) * 1e9 / n_total); }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 7;
#if defined(__arm__)
    if (argc > 2 && !strcmp(argv[2], "runfast")) {
        uint32_t f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); f |= (3u << 24); __asm__ volatile("vmsr fpscr, %0" : : "r"(f));
    }
#endif
    const int N = 2048, R = 256;              // 2048 floats in L1, 256 passes
    std::vector<float> x(N), y(N), y2(N);
    for (int i = 0; i < N; ++i) x[i] = -6.0f + 12.0f * (float) ((i * 7919) % N) / N;   // activations' range
    printf("ns/element, %d floats x %d passes in L1, reps %d\n", N, R, reps);
    auto arm = [&](const char * name, auto fn) {
        printf("%-28s %7.1f\n", name, ns_per([&] { for (int r = 0; r < R; ++r) fn(N, y.data(), x.data()); }, N * R, reps));
    };
    arm("gelu_erf stock (0010 tail)", gelu_stock);
    arm("gelu_erf x4, vfp clamps", gelu_il<4, false>);
    arm("gelu_erf x4, int clamps", gelu_il<4, true>);
    arm("gelu_erf x8, int clamps", gelu_il<8, true>);
#if defined(__arm__)
    arm("gelu_erf asm (short vectors)", gelu_asm);
#endif
    arm("expf libm", expf_libm);
    arm("expf own x1", expf_own<1>);
    arm("expf own x4", expf_own<4>);
    arm("tanhf libm", tanh_libm);
    arm("tanh own x4 (1-2/(e+1))", tanh_own<4>);
    arm("sigmoid libm", sigm_libm);
    arm("sigmoid own x4", sigm_own<4>);
    arm("silu libm", silu_libm);
    arm("silu own x4", silu_own<4>);
#if defined(__arm__)
    arm("sigmoid asm", sigm_asm);
    arm("silu asm", silu_asm);
    arm("tanh asm", tanh_asm);
#endif

    // GELU_ERF bit-identity over a sweep of bit patterns (every 61st, both signs: ~70M)
    long bad = 0, cnt = 0, asm_bad = 0;
    std::vector<float> in(4096), a(4096), b(4096);
    for (uint64_t u0 = 0; u0 < (1ull << 32); u0 += 61ull * 4096) {
        for (int k = 0; k < 4096; ++k) in[k] = bitsf((uint32_t) (u0 + 61ull * k));
        gelu_stock(4096, a.data(), in.data());
        gelu_il<8, true>(4096, b.data(), in.data());
        for (int k = 0; k < 4096; ++k) bad += fbits(a[k]) != fbits(b[k]);
        gelu_il<4, false>(4096, b.data(), in.data());
        for (int k = 0; k < 4096; ++k) bad += fbits(a[k]) != fbits(b[k]);
#if defined(__arm__)
        gelu_asm(4093, b.data(), in.data());   // a ragged count: the tail goes scalar
        for (int k = 0; k < 4093; ++k) asm_bad += fbits(a[k]) != fbits(b[k]);
        gelu_asm(4096, in.data(), in.data());  // in place
        for (int k = 0; k < 4096; ++k) asm_bad += fbits(a[k]) != fbits(in[k]);
#endif
        cnt += 4096;
    }
#if defined(__arm__)
    long sbad = 0, lbad = 0, tbad = 0;
    for (uint64_t u0 = 0; u0 < (1ull << 32); u0 += 61ull * 4096) {
        for (int k = 0; k < 4096; ++k) in[k] = bitsf((uint32_t) (u0 + 61ull * k));
        sigm_ref(4096, a.data(), in.data()); sigm_asm(4093, b.data(), in.data());
        for (int k = 0; k < 4093; ++k) sbad += fbits(a[k]) != fbits(b[k]);
        silu_ref(4096, a.data(), in.data()); silu_asm(4093, b.data(), in.data());
        for (int k = 0; k < 4093; ++k) lbad += fbits(a[k]) != fbits(b[k]);
        tanh_ref(4096, a.data(), in.data()); tanh_asm(4093, b.data(), in.data());
        for (int k = 0; k < 4093; ++k) tbad += fbits(a[k]) != fbits(b[k]);
        tanh_asm(4096, in.data(), in.data());   // in place
        for (int k = 0; k < 4096; ++k) tbad += fbits(a[k]) != fbits(in[k]);
    }
    printf("sigmoid asm vs reference: %ld differ; silu: %ld; tanh (ragged + in place): %ld (of %ld each)\n", sbad, lbad, tbad, cnt);
    bad += sbad + lbad + tbad;
#endif
    printf("gelu_erf interleaved vs stock: %ld of %ld differ; asm vs stock (ragged + in place): %ld of %ld\n", bad, 2 * cnt, asm_bad, 2 * cnt);
    bad += asm_bad;

    // the own expf against libm, relative, over [-87, 88]
    double me = 0, ms = 0, mt = 0;
    for (int k = 0; k < 2000000; ++k) {
        const float v = -87.0f + 175.0f * (float) k / 2000000.0f;
        const double e = expf(v);
        if (e > 1.2e-38) me = std::max(me, std::fabs(fexp(v) - e) / e);
        const float s = -20.0f + 40.0f * (float) k / 2000000.0f;
        const double ss = 1.0 / (1.0 + std::exp(-(double) s));
        ms = std::max(ms, std::fabs(1.f / (1.f + fexp(-s)) - ss) / ss);
        const double tt = std::tanh((double) s);
        if (std::fabs(s) > 0.5f) mt = std::max(mt, std::fabs((1.f - 2.f / (fexp(2.f * s) + 1.f)) - tt) / std::fabs(tt));
    }
    printf("own expf max rel err vs libm %.2e; sigmoid vs f64 %.2e; tanh |x|>0.5 vs f64 %.2e\n", me, ms, mt);
    return bad ? 1 : 0;
}
