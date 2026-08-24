// P4.18's two elementwise candidates, measured before either is written as a patch: is whisper's
// encoder losing 18% of its gap to onnxruntime in GELU and SOFT_MAX because those two ggml kernels
// are, respectively, NOT VECTORISED AT ALL and MAKING FIVE PASSES over every row? NOT part of the
// build: a standalone measurement, kept because the numbers it produces are the justification for
// cmake/patches/ggml-0010 and ggml-0011 and have to stay reproducible.
//
//   g++ -O3 -std=c++17 -march=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench12.cpp -o bench12 \
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//   ./bench12
//
// Single-threaded on purpose: P4.18's profile is a one-thread profile (P4.14's floor trap), and both
// of these kernels are pure per-row maps, so a thread count only scales both arms together.
//
// WHAT THE BASELINES ARE. Neither is transcribed -- `ggml_vec_gelu_erf_f32` is included from ggml's
// own `vec.h`, and the SOFT_MAX baseline is the row body of `ggml_compute_forward_soft_max_f32`
// (ops.cpp) built out of the same `ggml_vec_*` calls it makes, including the real linked
// `ggml_vec_soft_max_f32`. If either arm is edited, edit it to match ggml, not to win.
//
// WHY GELU IS THE BIGGER OF THE TWO. loom emits the EXACT-erf GELU on purpose (op_gelu,
// src/ops/primitives_basic.cpp -- reproducibility against a numpy reference), which lands on
// `ggml_vec_gelu_erf_f32`, and that function is a scalar `erff()` libm call per element with no SIMD
// path on any architecture. The tanh-approximation GELU next to it in vec.h has a 128 KB f16 lookup
// table and is not usable here: it is a different function, and the exporter distinguishes them
// (loom-exporter/topology_ops.py, MIL `gelu` mode EXACT vs TANH_APPROXIMATION).
//
// The candidate keeps the exact-erf FUNCTION and replaces only the libm call: erf(z) ~ z*P(z^2)/Q(z^2)
// with degree 5 over degree 5, on z clamped to [-4, 4], then saturated to [-1, 1]. It shipped as
// cmake/patches/ggml-0010-gelu-erf-simd.patch, whose header carries the derivation and the accuracy
// argument; the short version is that the error was bounded EXHAUSTIVELY -- over all 2^32 float32
// inputs, not a sample -- at 2.64e-07 relative to max(|x|,1) against the libm path's own 1.08e-07,
// i.e. 2.4x, or about two f32 ulps of the value's own scale.
//
// The accuracy number this bench prints is a DIFFERENT and weaker thing: the largest disagreement
// between the two arms over one normal(0,2) tensor. It is here to catch a transcription error in the
// arm, not to justify the kernel. The justification is the exhaustive sweep.
//
// WHY SOFT_MAX IS THE SMALLER ONE. Its inner `ggml_vec_soft_max_f32` IS vectorised, so the cost is
// not the exp -- it is the shape of the row body around it. Per row ggml makes FIVE passes (copy to
// scratch, scale scratch, max-reduce, exp+sum, normalise) where three suffice, and its AVX2 loop
// does a full horizontal reduction of the accumulator EVERY 8 ELEMENTS. The candidate folds the
// scale into the exp, drops the scratch copy, and keeps four vector accumulators reduced once at the
// end. It is guarded on the case that makes the folding valid -- no mask, no sinks, scale >= 0 --
// which is exactly whisper's encoder self-attention.
#include "vec.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Both GELU arms are ggml's OWN functions, so what is measured is the patch and not a copy of it:
// `ggml_vec_gelu_erf_f32_libm` is the reference path ggml-0010 preserves, and `ggml_vec_gelu_erf_f32`
// is the patched one. That means this bench only builds against a PATCHED ggml checkout -- which is
// the right dependency, because the number it exists to defend is the patch's.
// ---------------------------------------------------------------------------------------------
// SOFT_MAX baseline: the row body of ggml_compute_forward_soft_max_f32, no mask / no sinks.
// ---------------------------------------------------------------------------------------------
static void base_soft_max_row(const int n, float * dp, const float * sp, float * wp, float scale) {
    ggml_vec_cpy_f32  (n, wp, sp);
    ggml_vec_scale_f32(n, wp, scale);
    float max = -INFINITY;
    ggml_vec_max_f32(n, &max, wp);
    ggml_float sum = ggml_vec_soft_max_f32(n, dp, wp, max);
    sum = 1.0/sum;
    ggml_vec_scale_f32(n, dp, (float) sum);
}

// SOFT_MAX candidate: three passes, scale folded into the exp, accumulators reduced once.
static void cand_soft_max_row(const int n, float * dp, const float * sp, float scale) {
    int i = 0;
    float max = -INFINITY;
    for (; i < n; ++i) max = sp[i] > max ? sp[i] : max;   // -O3 vectorises a max-reduce
    const float maxs = scale*max;

    ggml_float sum = 0;
    i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    {
        const __m512 vs = _mm512_set1_ps(scale), vm = _mm512_set1_ps(maxs);
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        for (; i + 31 < n; i += 32) {
            __m512 v0 = ggml_v_expf(_mm512_fmsub_ps(_mm512_loadu_ps(sp + i     ), vs, vm));
            __m512 v1 = ggml_v_expf(_mm512_fmsub_ps(_mm512_loadu_ps(sp + i + 16), vs, vm));
            _mm512_storeu_ps(dp + i,      v0);
            _mm512_storeu_ps(dp + i + 16, v1);
            a0 = _mm512_add_ps(a0, v0);
            a1 = _mm512_add_ps(a1, v1);
        }
        for (; i + 15 < n; i += 16) {
            __m512 v0 = ggml_v_expf(_mm512_fmsub_ps(_mm512_loadu_ps(sp + i), vs, vm));
            _mm512_storeu_ps(dp + i, v0);
            a0 = _mm512_add_ps(a0, v0);
        }
        sum += (ggml_float) _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    {
        const __m256 vs = _mm256_set1_ps(scale), vm = _mm256_set1_ps(maxs);
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (; i + 31 < n; i += 32) {
            __m256 v0 = ggml_v_expf(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i     ), vs, vm));
            __m256 v1 = ggml_v_expf(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i +  8), vs, vm));
            __m256 v2 = ggml_v_expf(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i + 16), vs, vm));
            __m256 v3 = ggml_v_expf(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i + 24), vs, vm));
            _mm256_storeu_ps(dp + i,      v0);  _mm256_storeu_ps(dp + i +  8, v1);
            _mm256_storeu_ps(dp + i + 16, v2);  _mm256_storeu_ps(dp + i + 24, v3);
            a0 = _mm256_add_ps(a0, v0);  a1 = _mm256_add_ps(a1, v1);
            a2 = _mm256_add_ps(a2, v2);  a3 = _mm256_add_ps(a3, v3);
        }
        for (; i + 7 < n; i += 8) {
            __m256 v0 = ggml_v_expf(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i), vs, vm));
            _mm256_storeu_ps(dp + i, v0);
            a0 = _mm256_add_ps(a0, v0);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        __m128 h = _mm_add_ps(_mm256_extractf128_ps(acc, 1), _mm256_castps256_ps128(acc));
        h = _mm_add_ps(h, _mm_movehl_ps(h, h));
        h = _mm_add_ss(h, _mm_movehdup_ps(h));
        sum += (ggml_float) _mm_cvtss_f32(h);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    {
        const float32x4_t vs = vdupq_n_f32(scale), vm = vdupq_n_f32(maxs);
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        for (; i + 15 < n; i += 16) {
            float32x4_t v0 = ggml_v_expf(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i     ), vs));
            float32x4_t v1 = ggml_v_expf(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i +  4), vs));
            float32x4_t v2 = ggml_v_expf(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i +  8), vs));
            float32x4_t v3 = ggml_v_expf(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i + 12), vs));
            vst1q_f32(dp + i, v0);      vst1q_f32(dp + i +  4, v1);
            vst1q_f32(dp + i + 8, v2);  vst1q_f32(dp + i + 12, v3);
            a0 = vaddq_f32(a0, v0);  a1 = vaddq_f32(a1, v1);
            a2 = vaddq_f32(a2, v2);  a3 = vaddq_f32(a3, v3);
        }
        for (; i + 3 < n; i += 4) {
            float32x4_t v0 = ggml_v_expf(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i), vs));
            vst1q_f32(dp + i, v0);
            a0 = vaddq_f32(a0, v0);
        }
        sum += (ggml_float) vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    }
#endif
    for (; i < n; ++i) {
        const float v = expf(scale*sp[i] - maxs);
        dp[i] = v;
        sum += (ggml_float) v;
    }
    const float inv = (float) (1.0/sum);
    for (i = 0; i < n; ++i) dp[i] *= inv;
}

// A THIRD soft_max arm, which exists only to bound what is left: identical 3-pass structure, but the
// exp replaced by the Schraudolph bit-trick (build 2^x by writing the exponent field directly, ~2e-2
// relative). FAR too inaccurate to ship -- it is here to answer "is the remaining time the exp, or the
// traffic?", and nothing else. If this arm is not much faster than the candidate, the exp is not what
// soft_max is spending its time on and a cheaper exp is not the lever.
static inline float fast_expf(float a) {
    union { float f; int32_t i; } u;
    const float k = 12102203.16156148f;      // 2^23 / ln 2
    const float b = 1064986816.0f;           // 127*2^23 - a small bias correction
    float t = k*a + b;
    t = t < 0.0f ? 0.0f : t;
    u.i = (int32_t) t;
    return u.f;
}

static void probe_soft_max_row_fastexp(const int n, float * dp, const float * sp, float scale) {
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) max = sp[i] > max ? sp[i] : max;
    const float maxs = scale*max;
    float sum = 0;
    for (int i = 0; i < n; ++i) { const float v = fast_expf(scale*sp[i] - maxs); dp[i] = v; sum += v; }
    const float inv = 1.0f/sum;
    for (int i = 0; i < n; ++i) dp[i] *= inv;
}

// And a FOURTH: the candidate's 3-pass traffic with NO exp at all (a plain copy in its place), which
// puts a hard floor under the structure -- whatever this costs is what three passes over the row
// cost, and everything above it is arithmetic.
static void probe_soft_max_row_noexp(const int n, float * dp, const float * sp, float scale) {
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) max = sp[i] > max ? sp[i] : max;
    const float maxs = scale*max;
    float sum = 0;
    for (int i = 0; i < n; ++i) { const float v = scale*sp[i] - maxs; dp[i] = v; sum += v; }
    const float inv = 1.0f/(sum == 0.0f ? 1.0f : sum);
    for (int i = 0; i < n; ++i) dp[i] *= inv;
}

// ---------------------------------------------------------------------------------------------

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 7;

    // Shapes are P4.18's, from $LOOM_PROFILE on whisper-small's encoder (jfk.wav, 11 s).
    const int GN = 3072, GR = 1500, GCALLS = 12;     // UNARY  3072 x 1500, 12 calls, 273 ms
    const int SN = 1500, SR = 1500*12, SCALLS = 12;  // SOFT_MAX 1500 x 1500 x 12 heads, 12 calls, 396 ms

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 2.0f);

    // ---- GELU ----
    {
        std::vector<float> x((size_t) GN*GR), ya((size_t) GN*GR), yb((size_t) GN*GR);
        for (auto & v : x) v = nd(rng);

        double ta = 0, tb = 0;
        std::vector<double> va, vb;
        for (int r = 0; r < reps; ++r) {
            // ABBA within the pair, and the pair order flipped on odd reps
            for (int order = 0; order < 2; ++order) {
                const bool a_first = (r % 2 == 0) ? (order == 0) : (order == 1);
                double t0, t1;
                if (a_first) {
                    t0 = now();
                    for (int i = 0; i < GR; ++i) ggml_vec_gelu_erf_f32_libm(GN, ya.data() + (size_t) i*GN, x.data() + (size_t) i*GN);
                    t1 = now(); ta = t1 - t0; va.push_back(ta);
                    t0 = now();
                    for (int i = 0; i < GR; ++i) ggml_vec_gelu_erf_f32(GN, yb.data() + (size_t) i*GN, x.data() + (size_t) i*GN);
                    t1 = now(); tb = t1 - t0; vb.push_back(tb);
                } else {
                    t0 = now();
                    for (int i = 0; i < GR; ++i) ggml_vec_gelu_erf_f32(GN, yb.data() + (size_t) i*GN, x.data() + (size_t) i*GN);
                    t1 = now(); tb = t1 - t0; vb.push_back(tb);
                    t0 = now();
                    for (int i = 0; i < GR; ++i) ggml_vec_gelu_erf_f32_libm(GN, ya.data() + (size_t) i*GN, x.data() + (size_t) i*GN);
                    t1 = now(); ta = t1 - t0; va.push_back(ta);
                }
            }
        }
        double ma = median(va), mb = median(vb);
        double maxabs = 0, maxrel = 0;
        for (size_t i = 0; i < ya.size(); ++i) {
            const double d = std::fabs((double) ya[i] - (double) yb[i]);
            maxabs = std::max(maxabs, d);
            if (std::fabs((double) ya[i]) > 1e-3) maxrel = std::max(maxrel, d/std::fabs((double) ya[i]));
        }
        printf("GELU (exact erf)  %d x %d, one call\n", GN, GR);
        printf("  ggml  libm erff     %8.2f ms   -> %7.1f ms over %d calls\n", ma*1e3, ma*1e3*GCALLS, GCALLS);
        printf("  0010  rational P/Q  %8.2f ms   -> %7.1f ms over %d calls\n", mb*1e3, mb*1e3*GCALLS, GCALLS);
        printf("  speedup %.2fx        max abs diff %.3e   max rel diff (|y|>1e-3) %.3e\n\n", ma/mb, maxabs, maxrel);
    }

    // ---- SOFT_MAX ----
    {
        const float scale = 1.0f/std::sqrt(64.0f);
        std::vector<float> x((size_t) SN*SR), ya((size_t) SN*SR), yb((size_t) SN*SR);
        std::vector<float> wp(SN + 64);
        for (auto & v : x) v = nd(rng);

        std::vector<double> va, vb;
        for (int r = 0; r < reps; ++r) {
            for (int order = 0; order < 2; ++order) {
                const bool a_first = (r % 2 == 0) ? (order == 0) : (order == 1);
                double t0, t1;
                if (a_first) {
                    t0 = now();
                    for (int i = 0; i < SR; ++i) base_soft_max_row(SN, ya.data() + (size_t) i*SN, x.data() + (size_t) i*SN, wp.data(), scale);
                    t1 = now(); va.push_back(t1 - t0);
                    t0 = now();
                    for (int i = 0; i < SR; ++i) cand_soft_max_row(SN, yb.data() + (size_t) i*SN, x.data() + (size_t) i*SN, scale);
                    t1 = now(); vb.push_back(t1 - t0);
                } else {
                    t0 = now();
                    for (int i = 0; i < SR; ++i) cand_soft_max_row(SN, yb.data() + (size_t) i*SN, x.data() + (size_t) i*SN, scale);
                    t1 = now(); vb.push_back(t1 - t0);
                    t0 = now();
                    for (int i = 0; i < SR; ++i) base_soft_max_row(SN, ya.data() + (size_t) i*SN, x.data() + (size_t) i*SN, wp.data(), scale);
                    t1 = now(); va.push_back(t1 - t0);
                }
            }
        }
        double ma = median(va), mb = median(vb);
        double maxabs = 0, maxrel = 0;
        for (size_t i = 0; i < ya.size(); ++i) {
            const double d = std::fabs((double) ya[i] - (double) yb[i]);
            maxabs = std::max(maxabs, d);
            if (std::fabs((double) ya[i]) > 1e-6) maxrel = std::max(maxrel, d/std::fabs((double) ya[i]));
        }
        std::vector<double> vc, vd;
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
            for (int i = 0; i < SR; ++i) probe_soft_max_row_fastexp(SN, yb.data() + (size_t) i*SN, x.data() + (size_t) i*SN, scale);
            vc.push_back(now() - t0);
            t0 = now();
            for (int i = 0; i < SR; ++i) probe_soft_max_row_noexp(SN, yb.data() + (size_t) i*SN, x.data() + (size_t) i*SN, scale);
            vd.push_back(now() - t0);
        }
        double mc = median(vc), md = median(vd);
        printf("SOFT_MAX  %d x %d rows (12 heads), one call\n", SN, SR);
        printf("  ggml  5-pass row    %8.2f ms   -> %7.1f ms over %d calls\n", ma*1e3, ma*1e3*SCALLS, SCALLS);
        printf("  cand  3-pass fused  %8.2f ms   -> %7.1f ms over %d calls\n", mb*1e3, mb*1e3*SCALLS, SCALLS);
        printf("  speedup %.2fx        max abs diff %.3e   max rel diff (|y|>1e-6) %.3e\n", ma/mb, maxabs, maxrel);
        printf("  -- bounds, not candidates --\n");
        printf("  probe 3-pass, fast exp  %8.2f ms  (%.2fx over ggml)   <- all that a cheaper exp could buy\n", mc*1e3, ma/mc);
        printf("  probe 3-pass, NO exp    %8.2f ms  (%.2fx over ggml)   <- the floor the row structure sets\n", md*1e3, ma/md);
    }
    return 0;
}
