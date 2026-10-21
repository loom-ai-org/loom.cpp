// ADR-077 gate 1's cross-model analysis, regime R2: THE VOCODERS' DIRECT CONVOLUTIONS, F32 AGAINST INT8 AT
// EQUAL EFFORT. VITS, the piperlite voices and sanoTTS spend 64-83% of a Pi Zero run in CONV_2D with
// k = 3-7, dilation 1-5, 12-64 channels and 2,000-130,000 time steps (scripts/armv6_opmix.py over
// their board profiles), where ggml runs 0.25-0.65 GOP/s. Both arms are asm with the reduction
// unrolled (scripts/gen_armv6_conv_tiles.py), channels-first in and out, bias included:
//
//   f32   the VFP11's short vectors in RunFast, 4 output channels x 4 time steps per tile; per block of
//         TB time steps the input window [IC][TB + (k-1)d] is packed contiguous first (timed).
//   i8    dynamic int8: X quantized per tensor (int8 range) to channels-last int16 pairs (timed: the
//         amax pass and the transposing quantize, in asm and checked word for word against a C
//         one), W per output channel; the SMLAD 2x4 tile over
//         (tap, channel pair); F32 epilogue. `i8 gemm` = i8 minus its quantize, which static int8 skips.
//
//   ./bench49 [reps] [T divisor]        e.g. ./bench49 5 1 on the board, ./bench49 1 64 under qemu
//   $BENCH_TB overrides the F32 arm's time block (default 32).
// Every arm is checked against a double reference on a subset of time steps before it is timed.
//
// Build (loom-armv6-build:bookworm):
//   python3 scripts/gen_armv6_conv_tiles.py conv_tiles.S conv_tiles.h
//   g++ -O2 -std=c++17 -march=armv6+fp -mfpu=vfp -mfloat-abi=hard -I. scripts/bench49.cpp conv_tiles.S -o bench49
#include "conv_tiles.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static unsigned fpscr_get() { unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); return f; }
static void fpscr_set(unsigned f) { __asm__ volatile("vmsr fpscr, %0" : : "r"(f)); }
static inline int32_t vcvtr(float x) {
    float r; __asm__("vcvtr.s32.f32 %0, %1" : "=t"(r) : "t"(x));
    int32_t i; std::memcpy(&i, &r, 4); return i;
}

struct Shape { int ic, oc, k, d, T; const char* who; };

// F32: Wp [OC/4][IC][k][4] (+4 floats of padding for the tile's last preload).
static void conv_f32(const Shape& s, conv_f32_fn fn, const float* X, int Tin, const float* Wp, const float* b,
                     float* Y, int TB, std::vector<uint32_t>& xp_store) {
    const int span = (s.k - 1) * s.d;
    const int Lmax = TB + span;
    xp_store.assign((size_t)(s.ic + 1) * Lmax + 8, 0);   // a spare row for the tile's preload
    uint32_t* Xp = (uint32_t*)(((uintptr_t)xp_store.data() + 7) & ~(uintptr_t)7);
    const uint32_t* Xw = (const uint32_t*)X;
    const unsigned f0 = fpscr_get();
    const unsigned rf = (f0 & ~(7u << 16) & ~(3u << 20)) | (1u << 24) | (1u << 25);
    fpscr_set(rf | (3u << 16));
    for (int t0 = 0; t0 < s.T; t0 += TB) {
        const int nt = std::min(TB, s.T - t0) / 4;
        const int L = nt * 4 + span;
        // Word copies, not memcpy: a library call must not run with LEN != 0 (AAPCS).
        for (int c = 0; c < s.ic; ++c) {
            const uint32_t* src = Xw + (size_t)c * Tin + t0;
            uint32_t* dst = Xp + (size_t)c * L;
            for (int i = 0; i < L; ++i) dst[i] = src[i];
        }
        for (int q = 0; q < s.oc / 4; ++q)
            for (int i = 0; i < nt; ++i)
                fn((const float*)Xp + 4 * i, Wp + (size_t)q * s.ic * s.k * 4, b + 4 * q,
                   Y + (size_t)4 * q * s.T + t0 + 4 * i, s.T * 4, s.ic, L * 4);
    }
    fpscr_set(f0);
}

// int8 quantize: per-tensor scale over X [IC][Tin] (int8 range), to channels-last int16 pairs
// Xq [Tin][IC/2]. The amax pass works on bit patterns (sign cleared; unsigned order is IEEE order for
// non-NaN). The convert is bench_xnnpack's quantize_cf asm (4 time steps x 2 channels per step, eight
// vmul/vcvtr chains in flight -- gcc emits one at a time), with one scale in all four lanes; the
// Tin % 4 tail takes the C path. Equal effort with the GEMM tile: C here cost 1.2-1.7x on Citrinet.
static float quantize(const Shape& s, const float* X, int Tin, uint32_t* Xq) {
    uint32_t mx = 0;
    const uint32_t* xb = (const uint32_t*)X;
    for (size_t i = 0, n = (size_t)s.ic * Tin; i < n; ++i) { const uint32_t v = xb[i] & 0x7fffffffu; mx = v > mx ? v : mx; }
    float m; std::memcpy(&m, &mx, 4);
    const float sx = m > 0 ? m / 127.0f : 1.0f, inv = 1.0f / sx;
    const int pw = s.ic / 2, T4 = Tin & ~3;
    const float inv4[4] = {inv, inv, inv, inv};
    __asm__ volatile("vldmia %0, {s24-s27}" : : "r"(inv4) : "s24", "s25", "s26", "s27");
    for (int t0 = 0; t0 < T4; t0 += 4) {
        const float* x0 = X + t0;
        uint32_t* q = Xq + (size_t)t0 * pw;
        for (int p = 0; p < pw; ++p, x0 += 2 * (size_t)Tin, ++q)
            __asm__ volatile(
                "vldmia %[a], {s16-s19}\n vldmia %[b], {s20-s23}\n"
                "vmul.f32 s16, s16, s24\n vmul.f32 s17, s17, s25\n vmul.f32 s18, s18, s26\n vmul.f32 s19, s19, s27\n"
                "vmul.f32 s20, s20, s24\n vmul.f32 s21, s21, s25\n vmul.f32 s22, s22, s26\n vmul.f32 s23, s23, s27\n"
                "vcvtr.s32.f32 s16, s16\n vcvtr.s32.f32 s17, s17\n vcvtr.s32.f32 s18, s18\n vcvtr.s32.f32 s19, s19\n"
                "vcvtr.s32.f32 s20, s20\n vcvtr.s32.f32 s21, s21\n vcvtr.s32.f32 s22, s22\n vcvtr.s32.f32 s23, s23\n"
                "vmov r0, r1, s16, s17\n vmov r2, r3, s20, s21\n"
                "pkhbt r0, r0, r2, lsl #16\n pkhbt r1, r1, r3, lsl #16\n"
                "str r0, [%[q]]\n str r1, [%[q], %[s1]]\n"
                "vmov r0, r1, s18, s19\n vmov r2, r3, s22, s23\n"
                "pkhbt r0, r0, r2, lsl #16\n pkhbt r1, r1, r3, lsl #16\n"
                "str r0, [%[q], %[s2]]\n str r1, [%[q], %[s3]]\n"
                : : [a] "r"(x0), [b] "r"(x0 + Tin), [q] "r"(q), [s1] "r"(pw * 4), [s2] "r"(pw * 8), [s3] "r"(pw * 12)
                : "r0", "r1", "r2", "r3", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "memory");
    }
    for (int t = T4; t < Tin; ++t)
        for (int p = 0; p < pw; ++p)
            Xq[(size_t)t * pw + p] = (uint16_t)vcvtr(X[(size_t)(2 * p) * Tin + t] * inv) |
                                     ((uint32_t)vcvtr(X[(size_t)(2 * p + 1) * Tin + t] * inv) << 16);
    return sx;
}

// The C quantize the asm replaces, kept as the check that the asm writes the same words.
static void quantize_ref(const Shape& s, const float* X, int Tin, float sx, uint32_t* Xq) {
    const float inv = 1.0f / sx;
    const int pw = s.ic / 2;
    for (int t = 0; t < Tin; ++t)
        for (int p = 0; p < pw; ++p)
            Xq[(size_t)t * pw + p] = (uint16_t)vcvtr(X[(size_t)(2 * p) * Tin + t] * inv) |
                                     ((uint32_t)vcvtr(X[(size_t)(2 * p + 1) * Tin + t] * inv) << 16);
}

// int8 GEMM part: Wq [OC/4][k][4][IC/2] words; Y channels-first.
static void conv_i8(const Shape& s, conv_i8_fn fn, const uint32_t* Xq, float sx, const uint32_t* Wq, const float* sw,
                    const float* b, float* Y) {
    const int pw = s.ic / 2, MC = 24;
    int32_t c[8];
    for (int t0 = 0; t0 < s.T; t0 += MC) {
        const int te = std::min(s.T, t0 + MC);
        for (int q = 0; q < s.oc / 4; ++q) {
            const uint32_t* wq = Wq + (size_t)q * s.k * 4 * pw;
            for (int t = t0; t < te; t += 2) {
                fn(Xq + (size_t)t * pw, wq, c);
                for (int j = 0; j < 4; ++j) {
                    const int o = 4 * q + j;
                    const float sc = sx * sw[o];
                    Y[(size_t)o * s.T + t] = (float)c[j] * sc + b[o];
                    Y[(size_t)o * s.T + t + 1] = (float)c[4 + j] * sc + b[o];
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const int div = argc > 2 ? std::atoi(argv[2]) : 1;
    const int TB = std::getenv("BENCH_TB") ? std::atoi(std::getenv("BENCH_TB")) : 32;
    // From the board profiles (armv6_opmix.py): amy's 24- and 48-channel stages, de051's 12-channel
    // one, VITS's 64-channel one; d = 3 and 5 are the HiFi-GAN resblocks' dilations.
    const Shape shapes[] = {
        {24, 24, 7, 1, 130048, "amy 24ch k7"}, {24, 24, 7, 3, 130048, "amy 24ch k7 d3"},
        {24, 24, 7, 5, 130048, "amy 24ch k7 d5"}, {48, 48, 3, 1, 32512, "amy 48ch k3"},
        {48, 48, 7, 1, 32512, "amy 48ch k7"}, {12, 12, 7, 1, 97024, "de051 12ch k7"},
        {64, 64, 7, 1, 17600, "VITS 64ch k7"},
    };
    {
        unsigned f = fpscr_get(); fpscr_set(f | (1u << 24) | (1u << 25));   // RunFast everywhere, as the engine
    }
    std::printf("reps=%d T/%d TB=%d\n%-16s %8s %9s %9s %9s %9s | %7s %7s %7s | %8s %8s\n", reps, div, TB, "shape", "GFLOP",
                "f32 ms", "i8 ms", "i8 q ms", "i8gemm ms", "f32 G/s", "i8 G/s", "i8g G/s", "f32 err", "i8 err");
    double tot[4] = {0, 0, 0, 0}, totf = 0;
    for (const Shape& s0 : shapes) {
        Shape s = s0;
        s.T = (s.T / div) & ~31;
        const conv_f32_fn ff = [&] { for (auto& t : kConvF32) if (t.k == s.k && t.d == s.d) return t.fn; return (conv_f32_fn) nullptr; }();
        const conv_i8_fn fi = [&] { for (auto& t : kConvI8) if (t.ic == s.ic && t.k == s.k && t.d == s.d) return t.fn; return (conv_i8_fn) nullptr; }();
        if (!ff || !fi) { std::fprintf(stderr, "%s: no tile generated\n", s.who); return 1; }
        const int span = (s.k - 1) * s.d, Tin = s.T + span, pw = s.ic / 2;
        std::mt19937 rng(11);
        std::normal_distribution<float> nx(0, 1), nw(0, 0.08f);
        std::vector<float> X((size_t)s.ic * Tin), W((size_t)s.oc * s.ic * s.k), B(s.oc);
        for (float& v : X) v = nx(rng);
        for (float& v : W) v = nw(rng);
        for (float& v : B) v = nw(rng);
        // Packing (weights are packed once, at load, in either precision).
        std::vector<float> Wp((size_t)s.oc * s.ic * s.k + 4, 0.0f);
        for (int o = 0; o < s.oc; ++o) for (int c = 0; c < s.ic; ++c) for (int j = 0; j < s.k; ++j)
            Wp[(((size_t)(o / 4) * s.ic + c) * s.k + j) * 4 + o % 4] = W[((size_t)o * s.ic + c) * s.k + j];
        std::vector<float> sw(s.oc);
        std::vector<uint32_t> Wq((size_t)s.oc / 4 * s.k * 4 * pw);
        for (int o = 0; o < s.oc; ++o) {
            float m = 0;
            for (int i = 0; i < s.ic * s.k; ++i) m = std::max(m, std::fabs(W[(size_t)o * s.ic * s.k + i]));
            sw[o] = m / 127.0f;
            auto q = [&](int c, int j) { return (uint16_t)(int16_t)std::lrint(W[((size_t)o * s.ic + c) * s.k + j] / sw[o]); };
            for (int j = 0; j < s.k; ++j) for (int p = 0; p < pw; ++p)
                Wq[(((size_t)(o / 4) * s.k + j) * 4 + o % 4) * pw + p] = q(2 * p, j) | ((uint32_t)q(2 * p + 1, j) << 16);
        }
        std::vector<float> Yf((size_t)s.oc * s.T), Yi((size_t)s.oc * s.T);
        std::vector<uint32_t> xp_store, Xq((size_t)Tin * pw + 64);
        float sx = 1;
        auto run_f = [&] { conv_f32(s, ff, X.data(), Tin, Wp.data(), B.data(), Yf.data(), TB, xp_store); };
        auto run_q = [&] { sx = quantize(s, X.data(), Tin, Xq.data()); };
        auto run_g = [&] { conv_i8(s, fi, Xq.data(), sx, Wq.data(), sw.data(), B.data(), Yi.data()); };
        run_f(); run_q(); run_g();
        {
            std::vector<uint32_t> ref(Xq.size(), 0);
            quantize_ref(s, X.data(), Tin, sx, ref.data());
            if (std::memcmp(ref.data(), Xq.data(), (size_t)Tin * pw * 4)) { std::fprintf(stderr, "%s: asm quantize differs\n", s.who); return 1; }
        }
        // Double reference on every 7th time step.
        double ef = 0, ei = 0, mr = 0;
        for (int o = 0; o < s.oc; ++o)
            for (int t = 0; t < s.T; t += 7) {
                double r = B[o];
                for (int c = 0; c < s.ic; ++c) for (int j = 0; j < s.k; ++j)
                    r += (double)W[((size_t)o * s.ic + c) * s.k + j] * X[(size_t)c * Tin + t + j * s.d];
                mr = std::max(mr, std::fabs(r));
                ef = std::max(ef, std::fabs(Yf[(size_t)o * s.T + t] - r));
                ei = std::max(ei, std::fabs(Yi[(size_t)o * s.T + t] - r));
            }
        ef /= mr; ei /= mr;
        if (ef > 1e-5 || ei > 3e-2) { std::fprintf(stderr, "%s: error f32 %.2e i8 %.2e\n", s.who, ef, ei); return 1; }
        std::vector<double> tf, tq, tg;
        for (int r = 0; r < reps; ++r) {   // alternate the arms, median of each
            double t0 = now(); run_f(); tf.push_back(now() - t0);
            t0 = now(); run_q(); tq.push_back(now() - t0);
            t0 = now(); run_g(); tg.push_back(now() - t0);
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        const double mf = med(tf), mq = med(tq), mg = med(tg), flop = 2.0 * s.oc * s.ic * s.k * (double)s.T;
        std::printf("%-16s %8.3f %9.2f %9.2f %9.2f %9.2f | %7.3f %7.3f %7.3f | %8.1e %8.1e\n", s.who, flop / 1e9, mf * 1e3,
                    (mq + mg) * 1e3, mq * 1e3, mg * 1e3, flop / mf / 1e9, flop / (mq + mg) / 1e9, flop / mg / 1e9, ef, ei);
        tot[0] += mf; tot[1] += mq + mg; tot[2] += mq; tot[3] += mg; totf += flop;
    }
    std::printf("%-16s %8.3f %9.2f %9.2f %9.2f %9.2f | %7.3f %7.3f %7.3f\n", "all", totf / 1e9, tot[0] * 1e3, tot[1] * 1e3,
                tot[2] * 1e3, tot[3] * 1e3, totf / tot[0] / 1e9, totf / tot[1] / 1e9, totf / tot[3] / 1e9);
    return 0;
}
