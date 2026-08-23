// P4.15's last open question: is a DIRECT 1-D convolution -- no im2col at all -- faster than the
// im2col + GEMM this engine runs? NOT part of the build; a standalone measurement, kept because its
// answer is "in one regime, by a lot, and in the other it is four times slower", and the next person
// to have this idea should get both halves.
//
//   g++ -O3 -std=c++17 -fopenmp -march=armv8-a -I <ggml>/include -I <ggml>/src scripts/bench10.cpp \
//       -o bench10 -L<build>/_deps/ggml-build/src -lggml -lggml-base -lggml-cpu -lgomp -lpthread -lm
//   ./bench10 4
//
// WHY ASK. With the tinyBLAS patches the GEMM has nothing left in it -- 23.5 GFLOP/s in-model against
// 25.1 measured standalone -- but the im2col that feeds it is 37% of all convolution time (phase-timed
// inside ggml's conv: 396 ms of 1070 ms per VITS synthesis). That is not the gather's implementation.
// Making its inner copy branch-free and width-specialised is worth ~10% of it, twice measured. It is
// the gather's EXISTENCE: im2col writes every input element `kw` times, 137 M element-writes per
// synthesis. A direct convolution writes none of them -- it holds a tile of the OUTPUT in registers
// and sweeps the input in place, contiguous loads, contiguous stores, one lane-broadcast weight per
// (channel, tap). That is MLAS's kernel shape minus its assembly.
//
// THE ANSWER, on a Pi 4 at 4 threads over the eleven convolution shapes of a VITS vocoder, best tile
// (4 output channels x 16 positions), padded copy included, against ggml's cache-blocked CONV_2D:
//
//   long activation, few channels   32x32 kw7 L73472   37.7 ms vs 59.3   **1.57x faster**
//                                   32x32 kw5 L73472   30.9 ms vs 46.6     1.51x
//                                   64x64 kw7 L18368   41.0 ms vs 56.3     1.37x
//                                  128x128 kw7 L2296   24.7 ms vs 27.5     1.11x
//   weight-heavy, short activation 192x384 kw5 L287    40.8 ms vs 10.3     0.25x
//                                  768x768 kw3 L100   167   ms vs 24.8     0.15x
//
//   weighted by how often each appears in one synthesis: 0.37x overall -- and 1.15x if each shape
//   takes whichever is faster, worth ~174 ms of the 1156 ms this model spends in convolution.
//
// The split is not subtle and it is not about the kernel. A direct convolution re-reads the WEIGHTS
// once per position block; a GEMM blocks both operands. When the weights are 1.5 MB and the activation
// is 287 positions long, re-reading them 17 times is the whole cost, and no amount of register tuning
// touches it. When the activation is 73472 long and the weights are 28 KB, the direct form wins by not
// materialising 66 MB nobody needed.
//
// TWO THINGS THAT LOOKED LIKE THE ANSWER AND WERE NOT, both worth 3x on their own:
//   * The loop nesting. With the position block INSIDE the output-channel loop -- the obvious way to
//     write it -- every channel block re-streams the whole input from DRAM (8 passes over 9.4 MB for
//     the first shape), and the kernel then measures the SAME time whatever `kw` is, which is the tell:
//     it is not doing arithmetic, it is waiting for memory. Position block outermost: 0.12x -> 0.36x.
//   * The edges. Testing "is this block interior" and sending whole blocks to a scalar path costs a
//     third of a short convolution. Copying the input once into a zero-padded buffer makes every block
//     interior, and that copy is one pass against im2col's `kw`.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#if defined(__aarch64__)
#  include <arm_neon.h>
#elif defined(__AVX2__)
#  include <immintrin.h>
#endif
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>
#include <vector>
static double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}

// Weights packed [ic][kx][oc] so that the four output channels a tile covers are CONTIGUOUS, which is
// what lets the inner loop use one vector load and four lane-broadcast FMAs instead of four scalar
// loads. Constant per model, so a real engine packs it once at load time; timed separately below.
static void pack_weights(const float* w, float* wp, int64_t IC, int64_t OC, int64_t KW) {
    for (int64_t ic = 0; ic < IC; ++ic)
        for (int64_t kx = 0; kx < KW; ++kx)
            for (int64_t oc = 0; oc < OC; ++oc)
                wp[(ic*KW + kx)*OC + oc] = w[oc*(IC*KW) + ic*KW + kx];
}

// The input, copied once into a zero-padded buffer so that EVERY position block is interior and the
// inner loop needs no bounds test at all. One pass over the activations (~9 MB for the largest shape
// here) against im2col's kw passes -- this is the whole structural argument for a direct convolution.
static void pad_input(const float* x, float* xp, int64_t IC, int64_t L, int64_t pad, int nth) {
    const int64_t LP = L + 2*pad;
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t ic = 0; ic < IC; ++ic) {
        float* dst = xp + ic*LP;
        memset(dst, 0, (size_t)pad*sizeof(float));
        memcpy(dst + pad, x + ic*L, (size_t)L*sizeof(float));
        memset(dst + pad + L, 0, (size_t)pad*sizeof(float));
    }
}

#if defined(__aarch64__)
// OCB output channels x (VEC*4) output positions held in registers, over the padded input.
template <int OCB, int VEC>
static void conv1d_direct(const float* xp, const float* wp, float* y,
                          int64_t IC, int64_t OC, int64_t KW, int64_t L, int64_t pad, int nth, int64_t dil = 1) {
    const int64_t OL = L, LP = L + 2*pad, P = VEC*4;
    const int64_t nblk = OL / P;                 // whole blocks; the ragged tail is handled after
    // POSITION BLOCK OUTERMOST, output channels inside it. The other way round -- which is the obvious
    // way to write it -- every output-channel block re-streams the entire input from DRAM: for a
    // 32-channel, 73472-long convolution that is 8 passes over 9.4 MB, and the kernel then measures
    // the same time whatever `kw` is, because it is not doing arithmetic, it is waiting for memory.
    // With the position block outside, the slice it needs (P + kw - 1 positions x IC channels, a few
    // KB) stays in L1 across every output channel.
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t b = 0; b < nblk; ++b) {
        for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB) {
            const int64_t p0 = b * P;
            float32x4_t acc[OCB][VEC];
            for (int i = 0; i < OCB; ++i)
                for (int j = 0; j < VEC; ++j) acc[i][j] = vdupq_n_f32(0.0f);
            for (int64_t ic = 0; ic < IC; ++ic) {
                const float* xrow = xp + ic*LP + p0;
                for (int64_t kx = 0; kx < KW; ++kx) {
                    const float* q = xrow + kx*dil;
                    float32x4_t xv[VEC];
                    for (int j = 0; j < VEC; ++j) xv[j] = vld1q_f32(q + j*4);
                    const float32x4_t wv = vld1q_f32(wp + (ic*KW + kx)*OC + oc0);
                    for (int j = 0; j < VEC; ++j) {
                        if (OCB > 0) acc[0][j] = vfmaq_laneq_f32(acc[0][j], xv[j], wv, 0);
                        if (OCB > 1) acc[1][j] = vfmaq_laneq_f32(acc[1][j], xv[j], wv, 1);
                        if (OCB > 2) acc[2][j] = vfmaq_laneq_f32(acc[2][j], xv[j], wv, 2);
                        if (OCB > 3) acc[3][j] = vfmaq_laneq_f32(acc[3][j], xv[j], wv, 3);
                    }
                }
            }
            for (int i = 0; i < OCB; ++i)
                for (int j = 0; j < VEC; ++j) vst1q_f32(y + (oc0+i)*OL + p0 + j*4, acc[i][j]);
        }
    }
    // Ragged tail as ONE MORE OVERLAPPING tile, matching the shipped kernel. A scalar tail here biases
    // every comparison in this bench by up to 2x on any shape whose length is not a whole number of
    // blocks -- which is most of them, and which is how it hid the phase-major comparison for a while.
    if (nblk * P < OL && OL >= P) {
        const int64_t p0 = OL - P;
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB) {
            float32x4_t acc[OCB][VEC];
            for (int i = 0; i < OCB; ++i)
                for (int v = 0; v < VEC; ++v) acc[i][v] = vdupq_n_f32(0.0f);
            for (int64_t ic = 0; ic < IC; ++ic) {
                const float* xrow = xp + ic*LP + p0;
                for (int64_t kx = 0; kx < KW; ++kx) {
                    const float* q = xrow + kx*dil;
                    float32x4_t xv[VEC];
                    for (int v = 0; v < VEC; ++v) xv[v] = vld1q_f32(q + v*4);
                    const float32x4_t wv = vld1q_f32(wp + (ic*KW + kx)*OC + oc0);
                    for (int v = 0; v < VEC; ++v) {
                        if (OCB > 0) acc[0][v] = vfmaq_laneq_f32(acc[0][v], xv[v], wv, 0);
                        if (OCB > 1) acc[1][v] = vfmaq_laneq_f32(acc[1][v], xv[v], wv, 1);
                        if (OCB > 2) acc[2][v] = vfmaq_laneq_f32(acc[2][v], xv[v], wv, 2);
                        if (OCB > 3) acc[3][v] = vfmaq_laneq_f32(acc[3][v], xv[v], wv, 3);
                    }
                }
            }
            for (int i = 0; i < OCB; ++i)
                for (int v = 0; v < VEC; ++v) vst1q_f32(y + (oc0+i)*OL + p0 + v*4, acc[i][v]);
        }
    }
}

#elif defined(__AVX2__)

// The same kernel for AVX2. Two differences that matter: the vector is 8 floats rather than 4, and
// there is no lane-broadcast FMA, so each weight costs its own broadcast load -- 0.5 loads/FMA at
// 4x2 against NEON's 0.31. And there are 16 vector registers, not 32, so the accumulator tile has to
// be half the size: 4 channels x 16 positions is 8 accumulators here, where aarch64 affords 16.
template <int OCB, int VEC>
static void conv1d_direct(const float* xp, const float* wp, float* y,
                          int64_t IC, int64_t OC, int64_t KW, int64_t L, int64_t pad, int nth) {
    const int64_t OL = L, LP = L + 2*pad, P = VEC*8;
    const int64_t nblk = OL / P;
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t b = 0; b < nblk; ++b) {
        for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB) {
            const int64_t p0 = b * P;
            __m256 acc[OCB][VEC];
            for (int i = 0; i < OCB; ++i)
                for (int j = 0; j < VEC; ++j) acc[i][j] = _mm256_setzero_ps();
            for (int64_t ic = 0; ic < IC; ++ic) {
                const float* xrow = xp + ic*LP + p0;
                for (int64_t kx = 0; kx < KW; ++kx) {
                    const float* q = xrow + kx;
                    __m256 xv[VEC];
                    for (int j = 0; j < VEC; ++j) xv[j] = _mm256_loadu_ps(q + j*8);
                    const float* w = wp + (ic*KW + kx)*OC + oc0;
                    for (int i = 0; i < OCB; ++i) {
                        const __m256 wv = _mm256_set1_ps(w[i]);
                        for (int j = 0; j < VEC; ++j) acc[i][j] = _mm256_fmadd_ps(xv[j], wv, acc[i][j]);
                    }
                }
            }
            for (int i = 0; i < OCB; ++i)
                for (int j = 0; j < VEC; ++j) _mm256_storeu_ps(y + (oc0+i)*OL + p0 + j*8, acc[i][j]);
        }
    }
    const int64_t done = nblk * P;
    if (done < OL) {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int64_t oc = 0; oc < OC; ++oc)
            for (int64_t p = done; p < OL; ++p) {
                float acc = 0.0f;
                for (int64_t ic = 0; ic < IC; ++ic)
                    for (int64_t kx = 0; kx < KW; ++kx)
                        acc += wp[(ic*KW + kx)*OC + oc] * xp[ic*LP + p + kx*dil];
                y[oc*OL + p] = acc;
            }
    }
}

#endif

// The same kernel WITHOUT the padded copy: interior blocks read the activation where it lies, and the
// two blocks per convolution that hang off an edge fall back per (channel, tap) -- a tap whose whole
// vector is in range takes the vector path, one that is not takes a scalar loop. The copy it removes
// is 5.5 ms on the largest shape here, ~22% of what the direct form wins.
#if defined(__aarch64__)
template <int OCB, int VEC>
static void conv1d_direct_nocopy(const float* x, const float* wp, float* y,
                                 int64_t IC, int64_t OC, int64_t KW, int64_t L, int64_t pad, int64_t dil, int nth) {
    const int64_t OL = L, P = VEC*4;
    const int64_t nblk = OL / P;
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t b = 0; b < nblk; ++b) {
        for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB) {
            const int64_t p0 = b * P;
            float32x4_t acc[OCB][VEC];
            for (int i = 0; i < OCB; ++i)
                for (int j = 0; j < VEC; ++j) acc[i][j] = vdupq_n_f32(0.0f);
            for (int64_t ic = 0; ic < IC; ++ic) {
                const float* xrow = x + ic*L;
                for (int64_t kx = 0; kx < KW; ++kx) {
                    const int64_t lo = p0 + kx*dil - pad;
                    const float32x4_t wv = vld1q_f32(wp + (ic*KW + kx)*OC + oc0);
                    if (lo >= 0 && lo + P <= L) {
                        const float* q = xrow + lo;
                        float32x4_t xv[VEC];
                        for (int j = 0; j < VEC; ++j) xv[j] = vld1q_f32(q + j*4);
                        for (int j = 0; j < VEC; ++j) {
                            if (OCB > 0) acc[0][j] = vfmaq_laneq_f32(acc[0][j], xv[j], wv, 0);
                            if (OCB > 1) acc[1][j] = vfmaq_laneq_f32(acc[1][j], xv[j], wv, 1);
                            if (OCB > 2) acc[2][j] = vfmaq_laneq_f32(acc[2][j], xv[j], wv, 2);
                            if (OCB > 3) acc[3][j] = vfmaq_laneq_f32(acc[3][j], xv[j], wv, 3);
                        }
                    } else {
                        float tmp[VEC*4];
                        for (int64_t t = 0; t < P; ++t) {
                            const int64_t sx = lo + t;
                            tmp[t] = (sx < 0 || sx >= L) ? 0.0f : xrow[sx];
                        }
                        for (int j = 0; j < VEC; ++j) {
                            const float32x4_t xv = vld1q_f32(tmp + j*4);
                            if (OCB > 0) acc[0][j] = vfmaq_laneq_f32(acc[0][j], xv, wv, 0);
                            if (OCB > 1) acc[1][j] = vfmaq_laneq_f32(acc[1][j], xv, wv, 1);
                            if (OCB > 2) acc[2][j] = vfmaq_laneq_f32(acc[2][j], xv, wv, 2);
                            if (OCB > 3) acc[3][j] = vfmaq_laneq_f32(acc[3][j], xv, wv, 3);
                        }
                    }
                }
            }
            for (int i = 0; i < OCB; ++i)
                for (int j = 0; j < VEC; ++j) vst1q_f32(y + (oc0+i)*OL + p0 + j*4, acc[i][j]);
        }
    }
    const int64_t done = nblk * P;
    if (done < OL) {
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int64_t oc = 0; oc < OC; ++oc)
            for (int64_t p = done; p < OL; ++p) {
                float a = 0.0f;
                for (int64_t ic = 0; ic < IC; ++ic)
                    for (int64_t kx = 0; kx < KW; ++kx) {
                        const int64_t sx = p + kx*dil - pad;
                        if (sx >= 0 && sx < L) a += wp[(ic*KW + kx)*OC + oc] * x[ic*L + sx];
                    }
                y[oc*OL + p] = a;
            }
    }
}
#endif

// PHASE-MAJOR ("a trous") variant, for dilated convolutions. A convolution with dilation d is d
// independent DENSE convolutions, one over each subsequence p = j*d + r. Laid out that way, the taps a
// channel needs for one output block stop being d floats apart: at d = 12 and kw = 7 the kernel reads
// seven separate 64-byte runs spread over 352 bytes per channel -- 448 streams for a prefetcher that
// tracks a dozen -- where the phase-major form reads one contiguous 88-byte run per channel, exactly
// as an undilated convolution does. The input transform is free: this path already copies the input
// once for padding, so it copies it phase-major instead.
//
// What is NOT free is the output, which comes out phase-major and has to be interleaved back. Timed
// separately below, because whether it eats the win is the entire question.
#if defined(__aarch64__)
// De-interleave `n` floats: dst[r*stride + j] = src[j*f + r], for f = 2, 3 or 4, which is exactly what
// NEON's vld2q/vld3q/vld4q do in one instruction. Everything else falls back to the scalar gather this
// replaces -- which runs at about 0.5 GB/s, where these run near memcpy.
static void deint(const float* src, float* dst, int64_t stride, int64_t n, int f) {
    int64_t j = 0;
    if (f == 2) {
        for (; j + 4 <= n/2; j += 4) {
            const float32x4x2_t v = vld2q_f32(src + j*2);
            vst1q_f32(dst + 0*stride + j, v.val[0]);
            vst1q_f32(dst + 1*stride + j, v.val[1]);
        }
    } else if (f == 3) {
        for (; j + 4 <= n/3; j += 4) {
            const float32x4x3_t v = vld3q_f32(src + j*3);
            vst1q_f32(dst + 0*stride + j, v.val[0]);
            vst1q_f32(dst + 1*stride + j, v.val[1]);
            vst1q_f32(dst + 2*stride + j, v.val[2]);
        }
    } else if (f == 4) {
        for (; j + 4 <= n/4; j += 4) {
            const float32x4x4_t v = vld4q_f32(src + j*4);
            vst1q_f32(dst + 0*stride + j, v.val[0]);
            vst1q_f32(dst + 1*stride + j, v.val[1]);
            vst1q_f32(dst + 2*stride + j, v.val[2]);
            vst1q_f32(dst + 3*stride + j, v.val[3]);
        }
    }
    for (int64_t jj = j; jj * f < n; ++jj)                 // remainder, and every other f
        for (int r = 0; r < f; ++r)
            if (jj*f + r < n) dst[r*stride + jj] = src[jj*f + r];
}

// The inverse: dst[j*f + r] = src[r*stride + j], i.e. vst2q/vst3q/vst4q.
static void inter(const float* src, float* dst, int64_t stride, int64_t n, int f) {
    int64_t j = 0;
    if (f == 2) {
        for (; j + 4 <= n/2; j += 4) {
            float32x4x2_t v; v.val[0] = vld1q_f32(src + 0*stride + j); v.val[1] = vld1q_f32(src + 1*stride + j);
            vst2q_f32(dst + j*2, v);
        }
    } else if (f == 3) {
        for (; j + 4 <= n/3; j += 4) {
            float32x4x3_t v; v.val[0] = vld1q_f32(src + 0*stride + j); v.val[1] = vld1q_f32(src + 1*stride + j);
            v.val[2] = vld1q_f32(src + 2*stride + j);
            vst3q_f32(dst + j*3, v);
        }
    } else if (f == 4) {
        for (; j + 4 <= n/4; j += 4) {
            float32x4x4_t v; v.val[0] = vld1q_f32(src + 0*stride + j); v.val[1] = vld1q_f32(src + 1*stride + j);
            v.val[2] = vld1q_f32(src + 2*stride + j); v.val[3] = vld1q_f32(src + 3*stride + j);
            vst4q_f32(dst + j*4, v);
        }
    }
    for (int64_t jj = j; jj * f < n; ++jj)
        for (int r = 0; r < f; ++r)
            if (jj*f + r < n) dst[jj*f + r] = src[r*stride + jj];
}

// NEON only splits by 2, 3 or 4, and the model wants 6 and 12 as well. d = a*b decomposes: splitting
// first by `a` and then by `b` lands element p = j*d + r in phase r = r1 + a*r2, because
// p = a*(j*b + r2) + r1. Two vectorised passes still beat one scalar one.
static void phase_factors(int64_t d, int* a, int* b) {
    *a = (int) d; *b = 1;
    if (d == 6)  { *a = 2; *b = 3; }
    if (d == 8)  { *a = 2; *b = 4; }
    if (d == 9)  { *a = 3; *b = 3; }
    if (d == 12) { *a = 4; *b = 3; }
    if (d == 16) { *a = 4; *b = 4; }
}

static void pad_input_phase(const float* x, float* xp, int64_t IC, int64_t L, int64_t KW,
                            int64_t dil, int64_t J, int nth) {
    const int64_t H = (KW - 1) / 2;
    const int64_t JP = J + 2*H;
    int a, b; phase_factors(dil, &a, &b);
    const bool vec = (a == 2 || a == 3 || a == 4) && (b == 1 || b == 2 || b == 3 || b == 4);
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t ic = 0; ic < IC; ++ic) {
        float* base = xp + ic * dil * JP;
        memset(base, 0, (size_t) dil * JP * sizeof(float));
        if (dil == 1) { memcpy(base + H, x + ic*L, (size_t) L * sizeof(float)); continue; }
        if (!vec) {
            for (int64_t r = 0; r < dil; ++r) {
                float* dst = base + r * JP + H;
                for (int64_t j = 0; j < J; ++j) { const int64_t p = j*dil + r; if (p < L) dst[j] = x[ic*L + p]; }
            }
            continue;
        }
        if (b == 1) {                                   // one vectorised pass
            deint(x + ic*L, base + H, JP, L, a);
        } else {
            // Two vectorised passes: split by `a`, then split each of those streams by `b`. The second
            // call lands phase r2 of stream r1 at base + (r1 + a*r2)*JP, which is just a stride of
            // a*JP -- so it is the same de-interleave, not a scalar walk. Vectorising only the first
            // pass (the first thing tried) left dilation 6 and 12 exactly where they started.
            static thread_local std::vector<float> tmp;
            const int64_t na = (L + a - 1) / a;
            if ((int64_t) tmp.size() < a * na) tmp.assign(a * na, 0.0f);
            deint(x + ic*L, tmp.data(), na, L, a);
            for (int r1 = 0; r1 < a; ++r1)
                deint(tmp.data() + (int64_t) r1*na, base + (int64_t) r1*JP + H, (int64_t) a*JP, na, b);
        }
    }
}

template <int OCB, int VEC>
static void conv1d_direct_phase(const float* xp, const float* wp, float* yp,
                                int64_t IC, int64_t OC, int64_t KW, int64_t dil, int64_t J, int nth) {
    const int64_t H = (KW - 1) / 2, JP = J + 2*H, P = VEC*4;
    const int64_t nblk = J / P;
#pragma omp parallel for collapse(2) num_threads(nth) schedule(static)
    for (int64_t r = 0; r < dil; ++r) {
        for (int64_t b = 0; b < nblk; ++b) {
            for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB) {
                const int64_t j0 = b * P;
                float32x4_t acc[OCB][VEC];
                for (int i = 0; i < OCB; ++i)
                    for (int v = 0; v < VEC; ++v) acc[i][v] = vdupq_n_f32(0.0f);
                for (int64_t ic = 0; ic < IC; ++ic) {
                    const float* xrow = xp + (ic * dil + r) * JP + j0;
                    for (int64_t kx = 0; kx < KW; ++kx) {
                        const float* q = xrow + kx;      // dense in j-space
                        float32x4_t xv[VEC];
                        for (int v = 0; v < VEC; ++v) xv[v] = vld1q_f32(q + v*4);
                        const float32x4_t wv = vld1q_f32(wp + (ic*KW + kx)*OC + oc0);
                        for (int v = 0; v < VEC; ++v) {
                            acc[0][v] = vfmaq_laneq_f32(acc[0][v], xv[v], wv, 0);
                            acc[1][v] = vfmaq_laneq_f32(acc[1][v], xv[v], wv, 1);
                            acc[2][v] = vfmaq_laneq_f32(acc[2][v], xv[v], wv, 2);
                            acc[3][v] = vfmaq_laneq_f32(acc[3][v], xv[v], wv, 3);
                        }
                    }
                }
                for (int i = 0; i < OCB; ++i)
                    for (int v = 0; v < VEC; ++v)
                        vst1q_f32(yp + ((oc0+i) * dil + r) * J + j0 + v*4, acc[i][v]);
            }
        }
    }
    // The ragged j-tail, as ONE MORE OVERLAPPING BLOCK ending at J rather than a scalar loop. It
    // recomputes up to P-1 positions that the last full block already did, which is cheaper than doing
    // a handful of them at scalar speed: with a scalar tail this kernel measured 42 ms on a shape it
    // does in 20 when the phase length happens to be a whole number of blocks, and that artefact was
    // large enough to hide the comparison this bench exists for.
    if (nblk * P < J && J >= P) {
        const int64_t j0 = J - P;
#pragma omp parallel for collapse(2) num_threads(nth) schedule(static)
        for (int64_t r = 0; r < dil; ++r)
            for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB) {
                float32x4_t acc[OCB][VEC];
                for (int i = 0; i < OCB; ++i)
                    for (int v = 0; v < VEC; ++v) acc[i][v] = vdupq_n_f32(0.0f);
                for (int64_t ic = 0; ic < IC; ++ic) {
                    const float* xrow = xp + (ic * dil + r) * JP + j0;
                    for (int64_t kx = 0; kx < KW; ++kx) {
                        const float* q = xrow + kx;
                        float32x4_t xv[VEC];
                        for (int v = 0; v < VEC; ++v) xv[v] = vld1q_f32(q + v*4);
                        const float32x4_t wv = vld1q_f32(wp + (ic*KW + kx)*OC + oc0);
                        for (int v = 0; v < VEC; ++v) {
                            acc[0][v] = vfmaq_laneq_f32(acc[0][v], xv[v], wv, 0);
                            acc[1][v] = vfmaq_laneq_f32(acc[1][v], xv[v], wv, 1);
                            acc[2][v] = vfmaq_laneq_f32(acc[2][v], xv[v], wv, 2);
                            acc[3][v] = vfmaq_laneq_f32(acc[3][v], xv[v], wv, 3);
                        }
                    }
                }
                for (int i = 0; i < OCB; ++i)
                    for (int v = 0; v < VEC; ++v)
                        vst1q_f32(yp + ((oc0+i) * dil + r) * J + j0 + v*4, acc[i][v]);
            }
    } else if (nblk * P < J) {
        const int64_t done = nblk * P;
#pragma omp parallel for collapse(2) num_threads(nth) schedule(static)
        for (int64_t r = 0; r < dil; ++r)
            for (int64_t oc = 0; oc < OC; ++oc)
                for (int64_t j = done; j < J; ++j) {
                    float a = 0.0f;
                    for (int64_t ic = 0; ic < IC; ++ic)
                        for (int64_t kx = 0; kx < KW; ++kx)
                            a += wp[(ic*KW + kx)*OC + oc] * xp[(ic*dil + r)*JP + j + kx];
                    yp[(oc*dil + r)*J + j] = a;
                }
    }
}

// phase-major [OC][dil][J] back to [OC][OL], interleaved with vst2/3/4 where the dilation allows.
static void unphase_output(const float* yp, float* y, int64_t OC, int64_t OL, int64_t dil, int64_t J, int nth) {
    int a, b; phase_factors(dil, &a, &b);
    const bool vec1 = b == 1 && (a == 2 || a == 3 || a == 4);
    const bool vec2 = (a == 2 || a == 3 || a == 4) && (b == 2 || b == 3 || b == 4);
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t oc = 0; oc < OC; ++oc) {
        if (dil == 1) { memcpy(y + oc*OL, yp + oc*J, (size_t) OL * sizeof(float)); continue; }
        if (vec1) { inter(yp + oc*dil*J, y + oc*OL, J, OL, a); continue; }
        if (vec2) {                                    // the mirror image of the two-pass split
            static thread_local std::vector<float> tmp;
            const int64_t na = (OL + a - 1) / a;
            if ((int64_t) tmp.size() < a * na) tmp.assign(a * na, 0.0f);
            for (int r1 = 0; r1 < a; ++r1)
                inter(yp + oc*dil*J + (int64_t) r1*J, tmp.data() + (int64_t) r1*na, (int64_t) a*J, na, b);
            inter(tmp.data(), y + oc*OL, na, OL, a);
            continue;
        }
        for (int64_t r = 0; r < dil; ++r) {
            const float* src = yp + (oc*dil + r) * J;
            for (int64_t j = 0; j < J; ++j) { const int64_t p = j*dil + r; if (p < OL) y[oc*OL + p] = src[j]; }
        }
    }
}
#endif

struct Conf { int64_t IC, OC, kw, L; int calls; int64_t dil; };

int main(int argc,char**argv){
    ggml_backend_load_all();
    ggml_backend_t B = ggml_backend_cpu_init();
    int nth = argc>1?atoi(argv[1]):4;
    ggml_backend_cpu_set_n_threads(B,nth);
    printf("threads=%d\n\n%-22s %5s %10s %9s %9s %9s %9s   %s\n", nth,
           "IC x OC x kw x L","calls","ggml conv","4x16 pad","phase tot","4x16 nocopy","ph kern","ph xform");

    Conf cs[] = {
        // dilations are the model's own -- a HiFi-GAN resblock runs the same shape at several, and
        // they matter: the taps of a dilated convolution are 12 floats apart, not adjacent.
        { 32,  32, 7, 73472, 1, 3}, { 32,  32, 7, 73472, 1,12},
        { 32,  32, 5, 73472, 1, 2}, { 32,  32, 5, 73472, 1, 6},
        { 32,  32, 3, 73472, 1, 1}, { 32,  32, 3, 73472, 1, 2},
        { 64,  64, 7, 18368, 1, 3}, { 64,  64, 7, 18368, 1,12},
        { 64,  64, 5, 18368, 1, 2}, { 64,  64, 5, 18368, 1, 6},
        { 64,  64, 3, 18368, 1, 1}, { 64,  64, 3, 18368, 1, 2},
        {128, 128, 7,  2296, 1, 3}, {128, 128, 7,  2296, 1,12},
        {128, 128, 5,  2296, 1, 2}, {128, 128, 5,  2296, 1, 6},
        {128, 128, 3,  2296, 1, 1}, {128, 128, 3,  2296, 1, 2},
        {192, 384, 5,   287,16, 1}, {768, 768, 3,   100,12, 1},
    };
    double tot_g=0, tot_a=0, tot_b=0, tot_c=0, tot_pack=0, tot_f=0;
    for (auto& c : cs) {
        const int64_t pad = (c.kw - 1) * c.dil / 2;
        std::vector<float> K((size_t)c.kw*c.IC*c.OC), X((size_t)c.L*c.IC), WP(K.size());
        for (size_t i=0;i<K.size();++i) K[i] = 0.02f - 0.001f*(float)(i%53);
        for (size_t i=0;i<X.size();++i) X[i] = 0.01f + 0.001f*(float)(i%97);

        // ---- ggml's current lowering: GGML_OP_CONV_2D with KH=1 (cache-blocked, in-place write)
        ggml_init_params ip={(size_t)1024*1024*1024,nullptr,true};
        ggml_context* ctx=ggml_init(ip);
        ggml_cgraph* gf=ggml_new_graph(ctx);
        ggml_tensor* tk = ggml_new_tensor_4d(ctx,GGML_TYPE_F32,c.kw,1,c.IC,c.OC);
        ggml_tensor* tx = ggml_new_tensor_4d(ctx,GGML_TYPE_F32,c.L,1,c.IC,1);
        ggml_set_input(tk); ggml_set_input(tx);
        ggml_tensor* r = ggml_conv_2d_direct(ctx,tk,tx,1,1,(int)pad,0,(int)c.dil,1);
        ggml_build_forward_expand(gf,r);
        ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(B));
        if(!ggml_gallocr_alloc_graph(ga,gf)){fprintf(stderr,"alloc fail\n");exit(1);}
        ggml_backend_tensor_set(tk,K.data(),0,K.size()*sizeof(float));
        ggml_backend_tensor_set(tx,X.data(),0,X.size()*sizeof(float));
        ggml_backend_graph_compute(B,gf);
        std::vector<float> ref(ggml_nelements(r));
        ggml_backend_tensor_get(r,ref.data(),0,ref.size()*sizeof(float));
        double t0=now(); for(int i=0;i<3;i++) ggml_backend_graph_compute(B,gf);
        const double tg=(now()-t0)/3;
        ggml_gallocr_free(ga); ggml_free(ctx);

        t0=now(); for(int i=0;i<3;i++) pack_weights(K.data(), WP.data(), c.IC, c.OC, c.kw);
        const double tp=(now()-t0)/3;

        std::vector<float> Y(ref.size());
        std::vector<float> XP((size_t)c.IC*(c.L + 2*pad + 2*c.kw*c.dil));
        double t1=now(); for(int i=0;i<3;i++) pad_input(X.data(), XP.data(), c.IC, c.L, pad, nth);
        const double tpad=(now()-t1)/3;
        double ta, tb, tc, md = 0, phase_kernel = 0, phase_xform = 0;
        auto run = [&](auto fn) {
            std::fill(Y.begin(), Y.end(), 0.0f);
            fn();
            double t=now(); for(int i=0;i<3;i++) fn(); t=(now()-t)/3;
            for (size_t i=0;i<Y.size();++i) {
                const double d = std::fabs((double)Y[i]-(double)ref[i]);
                const double rr = std::fabs((double)ref[i]) + 1e-6;
                if (d/rr > md) md = d/rr;
            }
            return t;
        };
#if defined(__aarch64__)
        ta = run([&]{ conv1d_direct<4,4>(XP.data(),WP.data(),Y.data(),c.IC,c.OC,c.kw,c.L,pad,nth,c.dil); });
        {   // phase-major: transform, kernel, and the interleave back, each timed
            const int64_t J = (c.L + c.dil - 1) / c.dil;
            const int64_t H = (c.kw - 1) / 2, JP = J + 2*H;
            std::vector<float> XPP((size_t)c.IC*c.dil*JP), YP((size_t)c.OC*c.dil*J);
            double t2=now(); for(int i=0;i<3;i++) pad_input_phase(X.data(),XPP.data(),c.IC,c.L,c.kw,c.dil,J,nth);
            const double t_tr=(now()-t2)/3;
            std::fill(Y.begin(), Y.end(), 0.0f);
            auto k = [&]{ conv1d_direct_phase<4,4>(XPP.data(),WP.data(),YP.data(),c.IC,c.OC,c.kw,c.dil,J,nth); };
            k(); t2=now(); for(int i=0;i<3;i++) k(); const double t_k=(now()-t2)/3;
            t2=now(); for(int i=0;i<3;i++) unphase_output(YP.data(),Y.data(),c.OC,c.L,c.dil,J,nth);
            const double t_un=(now()-t2)/3;
            double m=0;
            for (size_t i=0;i<Y.size();++i){ double d=std::fabs((double)Y[i]-(double)ref[i]);
                double rr=std::fabs((double)ref[i])+1e-6; if(d/rr>m) m=d/rr; }
            if (m > 1e-4) printf("[PHASE MISMATCH %.1e]", m);
            tb = t_k + t_tr + t_un;
            phase_kernel = t_k; phase_xform = t_tr + t_un;
        }
        tc = run([&]{ conv1d_direct_nocopy<4,4>(X.data(),WP.data(),Y.data(),c.IC,c.OC,c.kw,c.L,pad,c.dil,nth); }) - tpad;
#else
        ta = run([&]{ conv1d_direct<4,2>(XP.data(),WP.data(),Y.data(),c.IC,c.OC,c.kw,c.L,pad,nth,c.dil); });
        tb = run([&]{ conv1d_direct<2,4>(XP.data(),WP.data(),Y.data(),c.IC,c.OC,c.kw,c.L,pad,nth,c.dil); });
        tc = run([&]{ conv1d_direct<4,1>(XP.data(),WP.data(),Y.data(),c.IC,c.OC,c.kw,c.L,pad,nth,c.dil); });
#endif

        const double gf_=2.0*c.kw*c.IC*c.OC*c.L/1e9;
        ta+=tpad; tb+=tpad; tc+=tpad;   // the padded copy is part of what a direct conv costs
        tot_g+=tg*c.calls; tot_a+=ta*c.calls; tot_b+=tb*c.calls; tot_c+=tc*c.calls;
        tot_pack+=tp; tot_f+=gf_*c.calls;
        char buf[64]; snprintf(buf,sizeof buf,"%lldx%lld k%lld L%lld d%lld",(long long)c.IC,(long long)c.OC,(long long)c.kw,(long long)c.L,(long long)c.dil);
        printf("%-22s %5d %7.2f ms %6.2f ms %6.2f ms %6.2f ms %6.2f ms %6.2f ms\n",
               buf,c.calls,tg*1e3,ta*1e3,tb*1e3,tc*1e3,phase_kernel*1e3,phase_xform*1e3);
    }
    printf("\nweighted total: ggml conv %.3f s | direct A %.3f s (%.2fx) | B %.3f s (%.2fx) | C %.3f s (%.2fx)\n",
           tot_g, tot_a, tot_g/tot_a, tot_b, tot_g/tot_b, tot_c, tot_g/tot_c);
    printf("arithmetic: %.3f GFLOP -> ggml %.1f GFLOP/s, best direct %.1f GFLOP/s   (weight packing, once per model: %.1f ms total)\n",
           tot_f, tot_f/tot_g, tot_f/std::min(std::min(tot_a,tot_b),tot_c), tot_pack*1e3);
    return 0;
}
