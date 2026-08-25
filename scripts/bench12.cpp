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

// ---------------------------------------------------------------------------------------------
// The exp itself: `ggml_v_expf` against one specialised to SOFT_MAX's domain.
// ---------------------------------------------------------------------------------------------
//
// WHY THIS ARM EXISTS. With the floor built honestly (below), the exp is the largest single piece of
// the candidate row -- so P4.18's item stopped being "rewrite soft_max" and became "look at
// `ggml_v_expf`", the same shape of finding as ggml-0010's GELU and in the same file.
//
// WHAT IS ACTUALLY SPECIALISABLE. `ggml_v_expf` is ARM's optimized-routines exp: a general
// `expf` that must be right for every float, including the ones whose result is subnormal or
// infinite. It pays for that with a mask (`|n| > 126`), a `movemask` and a branch around a slow path
// that reconstructs the scale in two halves. SOFT_MAX's argument is `x - max <= 0` and its result is
// in `(0, 1]`, so **none of that path can ever be reached** -- the argument only has to be clamped
// below at the point the answer stops being a normal float.
//
// So the candidate is ggml's own fast path, verbatim, with the mask/movemask/branch replaced by one
// `max`. Same range reduction, same degree-5 polynomial, same `2^n` by integer add -- which is the
// point: it isolates the general-case handling and nothing else. Anything that changed the polynomial
// would be measuring two things at once.
//
// -87.0 rather than -103.97 (where ggml flushes to zero) or -87.3 (where f32 stops being normal):
// exp(-87) = 1.6e-38 is the smallest answer that is still a NORMAL float, and stopping there keeps
// subnormals -- which cost microcode on some cores -- out of a kernel that would otherwise never see
// one. The elements this pins at 1.6e-38 instead of 0 are elements whose true weight is 1e-38 against
// a sum that is at least 1, so the difference is not representable in the output.
//
// A REDUCTION USING ONE MULTIPLY (`t = x*log2e; f = t - round(t)`) instead of the hi/lo `ln2` split
// was worked through and is NOT here: it removes one fnmadd and adds a multiply and a subtract, so it
// is a net one operation of fourteen, in exchange for carrying the rounding error of `x*log2e`
// (~1e-5 relative at the bottom of the range) into every element. Priced before writing, not after.
//
// `VW`/`VLOAD`/`VSTORE` go with it: the exp-alone bench below drives both arms through one loop, and
// the loop has to be the same loop for the difference between them to be the exp.
#define LOOM_HAVE_SM_EXPF 1
#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define VW 16
#define VLOAD(p)     _mm512_loadu_ps(p)
#define VSTORE(p, v) _mm512_storeu_ps(p, v)
template <int POLY = 5>
static inline __m512 sm_expf(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    const __m512 r = _mm512_set1_ps(0x1.8p23f);
    const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
    const __m512 n = _mm512_sub_ps(z, r);
    const __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f),
                                      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
    const __m512i e = _mm512_slli_epi32(_mm512_castps_si512(z), 23);
    const __m512 k = _mm512_castsi512_ps(
        _mm512_add_epi32(e, _mm512_castps_si512(_mm512_set1_ps(1))));
    if constexpr (POLY == 0) return _mm512_fmadd_ps(b, k, k);
    const __m512 u = _mm512_mul_ps(b, b);
    const __m512 j = _mm512_fmadd_ps(
        _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b,
                                        _mm512_set1_ps(0x1.573e2ep-5f)), u,
                        _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b,
                                        _mm512_set1_ps(0x1.fffdb6p-2f))),
        u, _mm512_mul_ps(_mm512_set1_ps(0x1.ffffecp-1f), b));
    return _mm512_fmadd_ps(j, k, k);
}
#elif defined(__AVX2__) && defined(__FMA__)
#define VW 8
#define VLOAD(p)     _mm256_loadu_ps(p)
#define VSTORE(p, v) _mm256_storeu_ps(p, v)
template <int POLY = 5>
static inline __m256 sm_expf(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    const __m256 r = _mm256_set1_ps(0x1.8p23f);
    const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
    const __m256 n = _mm256_sub_ps(z, r);
    const __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
    const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    const __m256 k = _mm256_castsi256_ps(
        _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
    if constexpr (POLY == 0) return _mm256_fmadd_ps(b, k, k);
    const __m256 u = _mm256_mul_ps(b, b);
    const __m256 j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                        _mm256_set1_ps(0x1.573e2ep-5f)), u,
                        _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                        _mm256_set1_ps(0x1.fffdb6p-2f))),
        u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
    return _mm256_fmadd_ps(j, k, k);
}
#elif defined(__ARM_NEON) && defined(__aarch64__)
#define VW 4
#define VLOAD(p)     vld1q_f32(p)
#define VSTORE(p, v) vst1q_f32(p, v)
template <int POLY = 5>
static inline float32x4_t sm_expf(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)),
                                    n, vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    if constexpr (POLY == 0) return vfmaq_f32(k, b, k);
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  u, vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b)),
        u);
    return vfmaq_f32(k, j, k);
}
#else
#undef LOOM_HAVE_SM_EXPF
#endif

// Which exp `cand_soft_max_row` uses, chosen at compile time so the three instantiations differ in
// NOTHING else -- same intrinsics, same unroll, same stores, same reduction.
//   0 = the identity, i.e. the floor: what the row costs with no exp in it at all.
//   1 = `ggml_v_expf`, which is what ggml's own soft_max calls.
//   2 = `sm_expf` above, the same fast path with the general-case mask and branch removed.
template <int KIND, typename V> static inline V EXPV(V x) {
    if (KIND == 0) return x;
#if defined(LOOM_HAVE_SM_EXPF)
    if (KIND == 2) return sm_expf<5>(x);
#endif
    return ggml_v_expf(x);
}

// SOFT_MAX candidate: three passes, scale folded into the exp, accumulators reduced once.
//
// TEMPLATED ON WHICH EXP IT USES, and that is the whole point. `EXP_KIND = 0` replaces
// `ggml_v_expf` with the identity and changes NOTHING else -- same intrinsics, same unroll, same
// stores, same reduction -- so the difference between the two instantiations is the exp and only the
// exp. The two scalar-C probes this file used to carry could not say that: they were plain loops left
// to the auto-vectoriser, measured against a hand-written AVX2/AVX-512 arm, so what they compared was
// two compilers' output. On a Ryzen 3 3250U the `noexp` one came out SLOWER than the arm it was
// supposed to bound -- impossible for a floor, and the tell that it was never one. Retro-012 read
// those probes as "deleting the exp buys nothing (0.99x)" and concluded SOFT_MAX was DRAM-bound; with
// the floor built this way it is 1.63x, and the memcpy arm below says the bytes are a quarter of it.
template <int EXP_KIND = 1>
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
            __m512 v0 = EXPV<EXP_KIND>(_mm512_fmsub_ps(_mm512_loadu_ps(sp + i     ), vs, vm));
            __m512 v1 = EXPV<EXP_KIND>(_mm512_fmsub_ps(_mm512_loadu_ps(sp + i + 16), vs, vm));
            _mm512_storeu_ps(dp + i,      v0);
            _mm512_storeu_ps(dp + i + 16, v1);
            a0 = _mm512_add_ps(a0, v0);
            a1 = _mm512_add_ps(a1, v1);
        }
        for (; i + 15 < n; i += 16) {
            __m512 v0 = EXPV<EXP_KIND>(_mm512_fmsub_ps(_mm512_loadu_ps(sp + i), vs, vm));
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
            __m256 v0 = EXPV<EXP_KIND>(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i     ), vs, vm));
            __m256 v1 = EXPV<EXP_KIND>(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i +  8), vs, vm));
            __m256 v2 = EXPV<EXP_KIND>(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i + 16), vs, vm));
            __m256 v3 = EXPV<EXP_KIND>(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i + 24), vs, vm));
            _mm256_storeu_ps(dp + i,      v0);  _mm256_storeu_ps(dp + i +  8, v1);
            _mm256_storeu_ps(dp + i + 16, v2);  _mm256_storeu_ps(dp + i + 24, v3);
            a0 = _mm256_add_ps(a0, v0);  a1 = _mm256_add_ps(a1, v1);
            a2 = _mm256_add_ps(a2, v2);  a3 = _mm256_add_ps(a3, v3);
        }
        for (; i + 7 < n; i += 8) {
            __m256 v0 = EXPV<EXP_KIND>(_mm256_fmsub_ps(_mm256_loadu_ps(sp + i), vs, vm));
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
            float32x4_t v0 = EXPV<EXP_KIND>(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i     ), vs));
            float32x4_t v1 = EXPV<EXP_KIND>(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i +  4), vs));
            float32x4_t v2 = EXPV<EXP_KIND>(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i +  8), vs));
            float32x4_t v3 = EXPV<EXP_KIND>(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i + 12), vs));
            vst1q_f32(dp + i, v0);      vst1q_f32(dp + i +  4, v1);
            vst1q_f32(dp + i + 8, v2);  vst1q_f32(dp + i + 12, v3);
            a0 = vaddq_f32(a0, v0);  a1 = vaddq_f32(a1, v1);
            a2 = vaddq_f32(a2, v2);  a3 = vaddq_f32(a3, v3);
        }
        for (; i + 3 < n; i += 4) {
            float32x4_t v0 = EXPV<EXP_KIND>(vfmsq_f32(vnegq_f32(vm), vld1q_f32(sp + i), vs));
            vst1q_f32(dp + i, v0);
            a0 = vaddq_f32(a0, v0);
        }
        sum += (ggml_float) vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    }
#endif
    for (; i < n; ++i) {
        const float v = EXP_KIND != 0 ? expf(scale*sp[i] - maxs) : (scale*sp[i] - maxs);
        dp[i] = v;
        sum += (ggml_float) v;
    }
    const float inv = (float) (1.0/sum);
    for (i = 0; i < n; ++i) dp[i] *= inv;
}

// THE TWO FLOORS, and why the ones that used to be here were not floors.
//
// This file previously carried two probes -- a Schraudolph bit-trick exp and a no-exp copy -- both
// written as plain scalar C and left to the auto-vectoriser, and both compared against a candidate
// written in hand intrinsics. What they measured was two compilers' output, not two amounts of work.
// On a Ryzen 3 3250U the no-exp one came out at 47.2 ms against the candidate's 42.6 -- SLOWER than
// the thing it claimed to put a floor under, which cannot happen and is the tell. Retro-012 read them
// as "deleting the exp entirely is 0.99x" and concluded SOFT_MAX was DRAM-bandwidth bound.
//
// Both floors below are honest ones:
//   * `cand_soft_max_row<0>` is the SAME FUNCTION with the exp switched off at compile time, so the
//     only difference is the exp. On the dev box that is 1.63x, not 1.00x.
//   * `memcpy` of the same bytes is what "bandwidth bound" is a claim ABOUT, and neither Retro-012
//     nor its predecessor ever measured it. On the dev box ggml's row body is 3.9x above it.
//
// A floor arm has to be built the same way as the arm it bounds. That is the lesson, and it is
// cheaper to restate here than to re-derive.

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
        std::vector<double> vc, vd, ve;
        for (int r = 0; r < reps; ++r) {
            double t0 = now();
            for (int i = 0; i < SR; ++i) cand_soft_max_row<0>(SN, yb.data() + (size_t) i*SN, x.data() + (size_t) i*SN, scale);
            vc.push_back(now() - t0);
            t0 = now();
            std::memcpy(yb.data(), x.data(), (size_t) SN*SR*sizeof(float));
            vd.push_back(now() - t0);
        }
        double mc = median(vc), md = median(vd);
        const double MB = (double) SN*SR*4 / (1024*1024);
        printf("SOFT_MAX  %d x %d rows (12 heads), one call -- %.0f MB in + %.0f MB out\n", SN, SR, MB, MB);
        printf("  ggml  5-pass row    %8.2f ms   -> %7.1f ms over %d calls\n", ma*1e3, ma*1e3*SCALLS, SCALLS);
        printf("  cand  3-pass fused  %8.2f ms   -> %7.1f ms over %d calls\n", mb*1e3, mb*1e3*SCALLS, SCALLS);
        printf("  speedup %.2fx        max abs diff %.3e   max rel diff (|y|>1e-6) %.3e\n", ma/mb, maxabs, maxrel);
        printf("  -- floors, not candidates; both built the same way as the arm above them --\n");
        printf("  same arm, exp switched off  %8.2f ms  (%.2fx of the candidate)  <- what the exp costs\n", mc*1e3, mb/mc);
        printf("  memcpy of the same bytes    %8.2f ms  (%.2f GB/s)               <- what the BYTES cost\n",
               md*1e3, 2*MB/1024.0/md);
        printf("  ggml is %.1fx the memcpy floor: if that is far above 1, this op is not bandwidth bound\n", ma/md);
#if defined(LOOM_HAVE_SM_EXPF)
        // The two exps IN THE ROW, as their own ABBA pair.
        //
        // Measured separately from the pair above and not by comparing against `mb`, because at this
        // shape the row streams 206 MB and a 3% ALU difference is far under what an un-paired position
        // in the round is worth: the first draft of this arm ran it after the memcpy floor and it came
        // out 32% SLOWER than the same function with a more expensive exp in it, which is the same
        // "compare two things measured different ways" error the deleted probes made.
        std::vector<double> vg, vh;
        for (int r = 0; r < reps; ++r) {
            for (int order = 0; order < 2; ++order) {
                const bool g_first = (r % 2 == 0) ? (order == 0) : (order == 1);
                for (int which = 0; which < 2; ++which) {
                    const bool ggml_now = (which == 0) == g_first;
                    const double t0 = now();
                    if (ggml_now) {
                        for (int i = 0; i < SR; ++i) cand_soft_max_row<1>(SN, yb.data() + (size_t) i*SN, x.data() + (size_t) i*SN, scale);
                    } else {
                        for (int i = 0; i < SR; ++i) cand_soft_max_row<2>(SN, yb.data() + (size_t) i*SN, x.data() + (size_t) i*SN, scale);
                    }
                    (ggml_now ? vg : vh).push_back(now() - t0);
                }
            }
        }
        const double mg = median(vg), mh = median(vh);
        printf("  -- the two exps in the SAME row, as their own ABBA pair --\n");
        printf("  cand, ggml_v_expf   %8.2f ms\n", mg*1e3);
        printf("  cand, sm_expf       %8.2f ms   %.2fx the row\n", mh*1e3, mg/mh);
#else
        (void) ve;
#endif
    }

#if defined(LOOM_HAVE_SM_EXPF)
    // ---- the exp on its own ----
    //
    // CACHE-RESIDENT ON PURPOSE, unlike the SOFT_MAX arm above. The question here is how many
    // OPERATIONS the two exps cost, and a 108 MB working set answers a different question -- it
    // measures the memory system, under which a 2-operation difference is invisible. 1 MB sits in L2
    // or L3 on every machine this is run on, so what is left is the ALU.
    //
    // The domain is [-30, 0]: SOFT_MAX's argument is `x - max`, which is <= 0 by construction, and 30
    // is well past where an attention weight stops mattering. **In this domain ggml's slow-path branch
    // is never taken** -- |n| <= 44 against its threshold of 126 -- so what the candidate removes is
    // the COMPUTATION of the mask and the branch on it, not a taken branch. Anyone reading a win here
    // as "the general case was being hit" has it backwards.
    {
        // 512 KB of input and ONE 512 KB output that all three arms share. Separate destinations were
        // the first draft and they cost the result: x + three 1 MB buffers is 4 MB against this box's
        // 4 MB L3, so which arm ran last decided what was resident, and the two arms disagreed with the
        // in-row pair by 1.4x. One destination, and the only thing that differs is the exp.
        const int EN = 131072;
        std::vector<float> x(EN), y(EN), ref_ggml(EN), ref_sm(EN);
        std::mt19937 erng(99);
        std::uniform_real_distribution<float> ud(-30.0f, 0.0f);
        for (auto & v : x) v = ud(erng);

        const int EITER = 400;
        std::vector<double> va, vb, vf, vp;
        for (int r = 0; r < reps; ++r) {
            for (int order = 0; order < 2; ++order) {
                const bool a_first = (r % 2 == 0) ? (order == 0) : (order == 1);
                for (int which = 0; which < 2; ++which) {
                    const bool ggml_now = (which == 0) == a_first;
                    // The branch is OUTSIDE the timed loop -- inside it, both bodies live in one loop
                    // and neither gets the schedule it would have alone.
                    double t0;
                    if (ggml_now) {
                        t0 = now();
                        for (int it = 0; it < EITER; ++it)
                            for (int i = 0; i + VW - 1 < EN; i += VW) VSTORE(y.data() + i, ggml_v_expf(VLOAD(x.data() + i)));
                    } else {
                        t0 = now();
                        for (int it = 0; it < EITER; ++it)
                            for (int i = 0; i + VW - 1 < EN; i += VW) VSTORE(y.data() + i, sm_expf<5>(VLOAD(x.data() + i)));
                    }
                    (ggml_now ? va : vb).push_back(now() - t0);
                }
            }
            // The load/store floor, built the same way as the two arms: same loop, same width, no exp.
            double t0 = now();
            for (int it = 0; it < EITER; ++it)
                for (int i = 0; i + VW - 1 < EN; i += VW) VSTORE(y.data() + i, VLOAD(x.data() + i));
            vf.push_back(now() - t0);
            // THE FLOOR THAT BOUNDS EVERY OTHER EXP, not just this one. Same function again, with the
            // degree-5 polynomial replaced by `exp(b) ~ 1 + b` -- six of its fourteen operations gone,
            // and an answer wrong in the second decimal place. Nothing that still computes an exp can
            // be faster than this, so whatever it does NOT return is unavailable to any rewrite.
            t0 = now();
            for (int it = 0; it < EITER; ++it)
                for (int i = 0; i + VW - 1 < EN; i += VW) VSTORE(y.data() + i, sm_expf<0>(VLOAD(x.data() + i)));
            vp.push_back(now() - t0);
        }
        const double ma = median(va), mb = median(vb), mf = median(vf), mp = median(vp);

        // Accuracy against a double-precision `exp`, over the same domain, for both arms -- the
        // candidate's claim is that it is not worse HERE, and the only way to say that is to hold
        // ggml's own error up beside it in the same sweep.
        double rel_a = 0, rel_b = 0;
        for (int i = 0; i + VW - 1 < EN; i += VW) {
            VSTORE(ref_ggml.data() + i, ggml_v_expf(VLOAD(x.data() + i)));
            VSTORE(ref_sm.data()   + i, sm_expf<5>(VLOAD(x.data() + i)));
        }
        for (int i = 0; i + VW - 1 < EN; ++i) {
            const double ref = std::exp((double) x[i]);
            rel_a = std::max(rel_a, std::fabs((double) ref_ggml[i] - ref)/ref);
            rel_b = std::max(rel_b, std::fabs((double) ref_sm[i]   - ref)/ref);
        }

        const double elems = (double) (EN/VW*VW) * EITER;
        printf("\nexp alone, %d floats in [-30,0] x %d passes (%.1f MB working set, ALU-bound)\n",
               EN, EITER, (double) EN*4/(1024*1024));
        printf("  ggml_v_expf         %8.2f ms   %6.2f ns/elem\n", ma*1e3, ma*1e9/elems);
        printf("  sm_expf             %8.2f ms   %6.2f ns/elem   %.2fx\n", mb*1e3, mb*1e9/elems, ma/mb);
        printf("  floor: load+store   %8.2f ms   %6.2f ns/elem\n", mf*1e3, mf*1e9/elems);
        printf("  above the floor:    ggml %.2f ns/elem, sm %.2f ns/elem -> %.2fx on the exp ITSELF\n",
               (ma - mf)*1e9/elems, (mb - mf)*1e9/elems, (ma - mf)/(mb - mf));
        printf("  max rel err vs double exp:  ggml %.3e   sm %.3e\n", rel_a, rel_b);
        printf("  floor: 1+b, no poly %8.2f ms   %6.2f ns/elem   %.2fx  <- the MOST any exp rewrite\n",
               mp*1e3, mp*1e9/elems, ma/mp);
        printf("                                                      can return, at any accuracy\n");
    }
#else
    printf("\nexp alone: skipped -- no sm_expf for this ISA\n");
#endif
    return 0;
}
