// P4.15b step 2: is it worth running a resblock's TWO convolutions as one tiled chain, so the
// activation between them never leaves cache? NOT part of the build; a standalone measurement, kept
// whatever the answer, because the next person to want this needs the number and not the idea.
//
//   g++ -O3 -std=c++17 -fopenmp -march=armv8-a -I <ggml>/include -I <ggml>/src scripts/bench11.cpp \
//       -o bench11 -lgomp -lpthread -lm
//   ./bench11 4 [T]
//
// WHY ASK. After step 1 (ggml-0007) a resblock layer is two fused convolutions, and each of them makes
// FIVE passes over a full activation: read the input, write the padded copy, read the padded copy,
// read the residual, write the output. Ten passes per layer, of which only three are unavoidable --
// the layer reads one activation, reads one residual and writes one result. Phase-timed inside the
// shipped kernel on a Pi 4, the padded copy alone (two of the ten) is 77 ms of a 1.34 s synthesis, so
// the ten look like around 190 ms and the three like around 58.
//
// THE ANSWER IS NO: 1.05x on the nine chains with an ORACLE choosing T per shape, and below 1.0x at
// 128 channels for every T. That is under 2% of a synthesis, for a kernel spanning four graph nodes.
//
// The reason is the part to keep, and it corrects the estimate above. Sweep T and the optimum is the
// SMALLEST tile, degrading monotonically to 0.39x at T = 2048 -- the intermediate wants L1, not the L2
// this was scoped around. And shape by shape, what the chain recovers is close to exactly what that
// layer's two PADDED COPIES cost and no more (32x73472 kw3: 12.3 ms saved against 16.4 ms of copy).
// The sweep's own loads were never on the clock: they are already overlapped with its FMAs, and
// removing a read the arithmetic was hiding removes no time. Only the padded copy, a memcpy with
// nothing to hide behind, is exposed. **Count only the passes with no arithmetic over them.**
//
// Which is where the win actually was: the padded copy was zeroing a whole row and then copying over
// all but 2*pad of it, writing every element of a 9.4 MB buffer twice. Fixing that (ggml-0006) is
// ~30 ms, more than this whole kernel would have been.
//
// THE SHAPE OF THE FIX. Tile the chain over the sequence: hold `mid` for a window of T positions in
// L2, have the second convolution consume it as the first produces it. The obvious way costs
// recomputation -- the second convolution's output tile needs `hB = (kw-1)*dB/2` positions of `mid` on
// each side -- but it does not have to: a thread walking its tiles IN ORDER already has the previous
// tile's trailing halo, so it appends exactly T new positions per step and recomputes only a 2*hB
// prologue once per thread. That is 0.4% of this model's longest activation.
//
// WHAT BOUNDS T. `mid` must hold ALL channels -- the second convolution's every output channel reads
// every input channel -- so the per-thread window is 2 * (T + 2*hB) * C floats (the residual copy and
// the leaky'd copy), and four threads share 1 MB of L2 on a Pi 4. That is T ~ 512 at 32 channels and
// T ~ 128 at 128, which is the whole reason this is measured per stage rather than assumed.
//
// THE MODEL'S OWN CHAINS, which is what the table below runs (trap #5 in BACKLOG.md P4.15b: a bench
// without the model's dilations measures a convolution the model does not have). Nine resblocks, three
// per upsample stage, each two convolutions with its own kernel size and dilation pair:
//
//   stage 3   32 ch  L 73472   kw 3 d(1,2)   kw 5 d(2,6)   kw 7 d(3,12)
//   stage 2   64 ch  L 18368   same three
//   stage 1  128 ch  L  2296   same three
#if defined(__aarch64__)
#  include <arm_neon.h>
#else
#  error "aarch64 only -- this measures a Cortex-A72's L2, and the shapes are chosen for its 1 MB"
#endif
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

#define OCB 4
#define VEC 4
#define P   (VEC * 4)

static void pack_weights(const float* w, float* wp, int64_t IC, int64_t OC, int64_t KW) {
    for (int64_t ic = 0; ic < IC; ++ic)
        for (int64_t kx = 0; kx < KW; ++kx)
            for (int64_t oc = 0; oc < OC; ++oc)
                wp[(ic*KW + kx)*OC + oc] = w[oc*(IC*KW) + ic*KW + kx];
}

static inline float leaky_s(float v, float s) {
    return (v > 0.0f ? v : 0.0f) + s * (v < 0.0f ? v : 0.0f);
}
static inline float32x4_t leaky_v(float32x4_t v, float32x4_t z, float32x4_t s) {
    return vfmaq_f32(vbslq_f32(vcgtq_f32(v, z), v, z), s, vbslq_f32(vcltq_f32(v, z), v, z));
}

// ---------------------------------------------------------------------------------------------
// ARM A: the shipped kernel's shape -- one padded copy of the whole activation, then one sweep.

static void pad_copy(const float* x, float* xp, int64_t C, int64_t L, int64_t pad, int nth,
                     const float* leaky) {
    const int64_t LP = L + 2*pad;
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t c = 0; c < C; ++c) {
        float* d = xp + c*LP;
        memset(d, 0, (size_t)pad*sizeof(float));
        if (leaky) {
            const float32x4_t z = vdupq_n_f32(0.0f), s = vdupq_n_f32(*leaky);
            int64_t i = 0;
            for (; i + 4 <= L; i += 4) vst1q_f32(d + pad + i, leaky_v(vld1q_f32(x + c*L + i), z, s));
            for (; i < L; ++i) d[pad + i] = leaky_s(x[c*L + i], *leaky);
        } else {
            memcpy(d + pad, x + c*L, (size_t)L*sizeof(float));
        }
        memset(d + pad + L, 0, (size_t)pad*sizeof(float));
    }
}

// [oc0, oc0+OCB) x [p0, p0+P) held in registers, over a zero-padded activation, with the bias as the
// accumulators' starting value and the residual added at the store. This is ggml-0006 + ggml-0007.
// The output, the input and the residual are each given their OWN offset: in the chained arm the
// intermediate is a rolling window, so a buffer position and an absolute sequence position are not the
// same number, and every one of the three lives in a different frame.
static inline void tile(const float* xp, const float* wp, const float* bias, const float* res,
                        float* y, int64_t IC, int64_t OC, int64_t KW, int64_t LP, int64_t OL,
                        int64_t dil, int64_t oc0, int64_t yoff, int64_t xoff,
                        int64_t roff, int64_t rstride) {
    float32x4_t acc[OCB][VEC];
    for (int i = 0; i < OCB; ++i) {
        const float32x4_t b = vdupq_n_f32(bias ? bias[oc0 + i] : 0.0f);
        for (int j = 0; j < VEC; ++j) acc[i][j] = b;
    }
    for (int64_t ic = 0; ic < IC; ++ic) {
        const float* xrow = xp + ic*LP + xoff;
        for (int64_t kx = 0; kx < KW; ++kx) {
            const float* q = xrow + kx*dil;
            float32x4_t xv[VEC];
            for (int j = 0; j < VEC; ++j) xv[j] = vld1q_f32(q + j*4);
            const float32x4_t wv = vld1q_f32(wp + (ic*KW + kx)*OC + oc0);
            for (int j = 0; j < VEC; ++j) {
                acc[0][j] = vfmaq_laneq_f32(acc[0][j], xv[j], wv, 0);
                acc[1][j] = vfmaq_laneq_f32(acc[1][j], xv[j], wv, 1);
                acc[2][j] = vfmaq_laneq_f32(acc[2][j], xv[j], wv, 2);
                acc[3][j] = vfmaq_laneq_f32(acc[3][j], xv[j], wv, 3);
            }
        }
    }
    for (int i = 0; i < OCB; ++i)
        for (int j = 0; j < VEC; ++j) {
            float32x4_t v = acc[i][j];
            if (res) v = vaddq_f32(v, vld1q_f32(res + (oc0 + i)*rstride + roff + j*4));
            vst1q_f32(y + (oc0 + i)*OL + yoff + j*4, v);
        }
}

static void sweep(const float* xp, const float* wp, const float* bias, const float* res, float* y,
                  int64_t C, int64_t KW, int64_t LP, int64_t OL, int64_t dil, int nth) {
    const int64_t nblk = OL / P;
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t b = 0; b < nblk; ++b)
        for (int64_t oc0 = 0; oc0 < C; oc0 += OCB)
            tile(xp, wp, bias, res, y, C, C, KW, LP, OL, dil, oc0, b*P, b*P, b*P, OL);
    if (nblk*P < OL) {
        const int64_t p0 = nblk*P;                      // no overlap: this bench's L is a multiple of 8
        for (int64_t oc = 0; oc < C; ++oc)
            for (int64_t p = p0; p < OL; ++p) {
                float a = bias ? bias[oc] : 0.0f;
                for (int64_t ic = 0; ic < C; ++ic)
                    for (int64_t kx = 0; kx < KW; ++kx) a += wp[(ic*KW+kx)*C+oc] * xp[ic*LP + p + kx*dil];
                y[oc*OL + p] = a + (res ? res[oc*OL + p] : 0.0f);
            }
    }
}

// ---------------------------------------------------------------------------------------------
// ARM B: the chain, tiled over the sequence.
//
// Per thread, over a contiguous run of output tiles. `mid` and `midh` are rolling windows of
// W = T + 2*hB positions: each step keeps the trailing 2*hB positions and appends T new ones, so the
// first convolution is computed exactly once per position after a 2*hB prologue.
static void chain(const float* x, const float* r, const float* wpA, const float* bA,
                  const float* wpB, const float* bB, float* out,
                  int64_t C, int64_t L, int64_t KW, int64_t dA, int64_t dB, int64_t T, float slope,
                  int nth) {
    const int64_t hA = (KW - 1)*dA/2, hB = (KW - 1)*dB/2;
    const int64_t W  = T + 2*hB;
    const int64_t ntile = (L + T - 1) / T;
    const int64_t per = (ntile + nth - 1) / nth;

#pragma omp parallel num_threads(nth)
    {
        const int t = omp_get_thread_num();
        std::vector<float> midv((size_t)C*W), midhv((size_t)C*W);
        float* mid  = midv.data();
        float* midh = midhv.data();
        const float32x4_t z = vdupq_n_f32(0.0f), sv = vdupq_n_f32(slope);

        // conv A for absolute output positions [a0, a1), landing at mid[c*W + (a0 - base)]
        auto convA = [&](int64_t a0, int64_t a1, int64_t base) {
            for (int64_t p = a0; p < a1; ) {
                const bool whole = (p + P <= a1) && (p + P <= L) && (p - hA >= 0) &&
                                   (p + P - 1 + (KW - 1)*dA - hA < L);
                if (whole) {
                    for (int64_t oc0 = 0; oc0 < C; oc0 += OCB)
                        tile(x, wpA, bA, r, mid, C, C, KW, L, W, dA, oc0,
                             /*yoff=*/p - base, /*xoff=*/p - hA, /*roff=*/p, /*rstride=*/L);
                    p += P;
                } else {
                    for (int64_t oc = 0; oc < C; ++oc) {
                        float a = bA[oc];
                        for (int64_t ic = 0; ic < C; ++ic)
                            for (int64_t kx = 0; kx < KW; ++kx) {
                                const int64_t q = p + kx*dA - hA;
                                if (q >= 0 && q < L) a += wpA[(ic*KW+kx)*C+oc] * x[ic*L + q];
                            }
                        mid[oc*W + (p - base)] = a + ((p >= 0 && p < L) ? r[oc*L + p] : 0.0f);
                    }
                    p += 1;
                }
            }
        };
        auto leakyrange = [&](int64_t n0, int64_t n1) {      // buffer offsets
            for (int64_t c = 0; c < C; ++c) {
                int64_t i = n0;
                for (; i + 4 <= n1; i += 4)
                    vst1q_f32(midh + c*W + i, leaky_v(vld1q_f32(mid + c*W + i), z, sv));
                for (; i < n1; ++i) midh[c*W + i] = leaky_s(mid[c*W + i], slope);
            }
        };

        for (int64_t k = t*per; k < (t + 1)*per && k < ntile; ++k) {
            const int64_t p0 = k*T, p1 = (p0 + T < L) ? p0 + T : L;
            const int64_t base = p0 - hB;                 // mid[.][0] is absolute position `base`
            if (k == t*per) {                             // prologue: the whole window
                memset(mid, 0, (size_t)C*W*sizeof(float));
                convA(p0 - hB < 0 ? 0 : p0 - hB, (p1 + hB < L ? p1 + hB : L), base);
                leakyrange(0, W);
            } else {                                      // steady state: shift and append T
                for (int64_t c = 0; c < C; ++c) {
                    memmove(mid  + c*W, mid  + c*W + T, (size_t)(2*hB)*sizeof(float));
                    memmove(midh + c*W, midh + c*W + T, (size_t)(2*hB)*sizeof(float));
                }
                const int64_t n0 = 2*hB;
                const int64_t a0 = base + n0, a1 = (base + W < L) ? base + W : L;
                if (a1 > a0) convA(a0, a1, base);
                if (base + W > L)
                    for (int64_t c = 0; c < C; ++c)
                        memset(mid + c*W + (L - base), 0, (size_t)(base + W - L)*sizeof(float));
                leakyrange(n0, W);
            }
            // conv B: output [p0, p1), reading midh, residual mid at buffer offset hB
            int64_t p = p0;
            for (; p + P <= p1; p += P)
                for (int64_t oc0 = 0; oc0 < C; oc0 += OCB)
                    tile(midh, wpB, bB, mid, out, C, C, KW, W, L, dB, oc0,
                         /*yoff=*/p, /*xoff=*/p - base - hB, /*roff=*/p - base, /*rstride=*/W);
            for (; p < p1; ++p)
                for (int64_t oc = 0; oc < C; ++oc) {
                    float a = bB[oc];
                    for (int64_t ic = 0; ic < C; ++ic)
                        for (int64_t kx = 0; kx < KW; ++kx)
                            a += wpB[(ic*KW+kx)*C+oc] * midh[ic*W + (p - base - hB) + kx*dB];
                    out[oc*L + p] = a + mid[oc*W + (p - base)];
                }
        }
    }
}

// ---------------------------------------------------------------------------------------------
static void reference(const float* x, const float* r, const float* wA, const float* bA,
                      const float* wB, const float* bB, float* out,
                      int64_t C, int64_t L, int64_t KW, int64_t dA, int64_t dB, float slope) {
    const int64_t hA = (KW-1)*dA/2, hB = (KW-1)*dB/2;
    std::vector<double> mid((size_t)C*L);
    for (int64_t oc = 0; oc < C; ++oc)
        for (int64_t p = 0; p < L; ++p) {
            double a = bA[oc];
            for (int64_t ic = 0; ic < C; ++ic)
                for (int64_t kx = 0; kx < KW; ++kx) {
                    const int64_t q = p + kx*dA - hA;
                    if (q >= 0 && q < L) a += (double)wA[oc*(C*KW) + ic*KW + kx] * x[ic*L + q];
                }
            mid[oc*L + p] = a + r[oc*L + p];
        }
    for (int64_t oc = 0; oc < C; ++oc)
        for (int64_t p = 0; p < L; ++p) {
            double a = bB[oc];
            for (int64_t ic = 0; ic < C; ++ic)
                for (int64_t kx = 0; kx < KW; ++kx) {
                    const int64_t q = p + kx*dB - hB;
                    if (q >= 0 && q < L) a += (double)wB[oc*(C*KW) + ic*KW + kx] * leaky_s((float)mid[ic*L + q], slope);
                }
            out[oc*L + p] = (float)(a + mid[oc*L + p]);
        }
}

struct Chain { int64_t C, L, kw, dA, dB, T; };

int main(int argc, char** argv) {
    const int nth = argc > 1 ? atoi(argv[1]) : 4;
    const int64_t Tover = argc > 2 ? atoll(argv[2]) : 0;
    const float slope = 0.1f;
    printf("threads=%d\n\n%-28s %5s %10s %10s %8s %9s %9s\n", nth,
           "C x L  kw d(A,B)", "T", "two convs", "chained", "ratio", "vs ref", "vs armA");

    Chain cs[] = {
        { 32, 73472, 3, 1,  2, 512}, { 32, 73472, 5, 2,  6, 512}, { 32, 73472, 7, 3, 12, 512},
        { 64, 18368, 3, 1,  2, 256}, { 64, 18368, 5, 2,  6, 256}, { 64, 18368, 7, 3, 12, 256},
        {128,  2296, 3, 1,  2, 128}, {128,  2296, 5, 2,  6, 128}, {128,  2296, 7, 3, 12, 128},
    };
    double tot_a = 0, tot_b = 0;
    for (auto& c : cs) {
        const int64_t T = Tover ? Tover : c.T;
        const int64_t hA = (c.kw-1)*c.dA/2, hB = (c.kw-1)*c.dB/2;
        std::vector<float> WA((size_t)c.kw*c.C*c.C), WB(WA.size()), BA(c.C), BB(c.C);
        std::vector<float> X((size_t)c.C*c.L), R(X.size()), OUT(X.size()), REF(X.size());
        std::vector<float> WPA(WA.size()), WPB(WA.size());
        for (size_t i = 0; i < WA.size(); ++i) { WA[i] = 0.02f - 0.001f*(float)(i%53); WB[i] = 0.015f - 0.0009f*(float)(i%47); }
        for (int64_t i = 0; i < c.C; ++i) { BA[i] = 0.01f*(float)(i%7) - 0.02f; BB[i] = 0.008f*(float)(i%5) - 0.01f; }
        for (size_t i = 0; i < X.size(); ++i) { X[i] = 0.01f + 0.001f*(float)(i%97) - 0.05f; R[i] = 0.02f - 0.0013f*(float)(i%61); }
        pack_weights(WA.data(), WPA.data(), c.C, c.C, c.kw);
        pack_weights(WB.data(), WPB.data(), c.C, c.C, c.kw);
        reference(X.data(), R.data(), WA.data(), BA.data(), WB.data(), BB.data(), REF.data(),
                  c.C, c.L, c.kw, c.dA, c.dB, slope);

        std::vector<float> MID((size_t)c.C*c.L);
        std::vector<float> XPA((size_t)c.C*(c.L + 2*hA + P)), XPB((size_t)c.C*(c.L + 2*hB + P));
        auto armA = [&] {
            pad_copy(X.data(), XPA.data(), c.C, c.L, hA, nth, nullptr);
            sweep(XPA.data(), WPA.data(), BA.data(), R.data(), MID.data(), c.C, c.kw, c.L + 2*hA, c.L, c.dA, nth);
            pad_copy(MID.data(), XPB.data(), c.C, c.L, hB, nth, &slope);
            sweep(XPB.data(), WPB.data(), BB.data(), MID.data(), OUT.data(), c.C, c.kw, c.L + 2*hB, c.L, c.dB, nth);
        };
        auto armB = [&] {
            chain(X.data(), R.data(), WPA.data(), BA.data(), WPB.data(), BB.data(), OUT.data(),
                  c.C, c.L, c.kw, c.dA, c.dB, T, slope, nth);
        };
        // Error as max|diff| against the PEAK, not per element. A resblock's output passes through
        // zero constantly, and a per-element relative error there reports the cancellation, not the
        // kernel: it reads 1e-2 for both arms at once, which is the tell that it is measuring the
        // reference's own conditioning.
        double peak = 0;
        for (float v : REF) peak = std::max(peak, (double)std::fabs(v));
        std::vector<float> SAVE;
        auto run = [&](auto fn) {
            std::fill(OUT.begin(), OUT.end(), 0.0f);
            fn();
            double m = 0;
            for (size_t i = 0; i < OUT.size(); ++i)
                m = std::max(m, std::fabs((double)OUT[i] - (double)REF[i]));
            if (SAVE.empty()) SAVE = OUT;                 // arm A's result, to compare the arms directly
            else for (size_t i = 0; i < OUT.size(); ++i)
                m = std::max(m, std::fabs((double)OUT[i] - (double)SAVE[i]));
            double t = now(); for (int i = 0; i < 3; ++i) fn(); t = (now() - t)/3;
            return std::make_pair(t, m/peak);
        };
        auto [ta, ma] = run(armA);
        auto [tb, mb] = run(armB);
        tot_a += ta; tot_b += tb;
        char buf[64];
        snprintf(buf, sizeof buf, "%lldx%lld k%lld d(%lld,%lld)", (long long)c.C, (long long)c.L,
                 (long long)c.kw, (long long)c.dA, (long long)c.dB);
        printf("%-28s %5lld %7.2f ms %7.2f ms %7.2fx %9.1e %9.1e%s\n", buf, (long long)T,
               ta*1e3, tb*1e3, ta/tb, ma, mb, (ma > 1e-5 || mb > 1e-5) ? "  <-- MISMATCH" : "");
    }
    printf("\ntotal: two convs %.1f ms | chained %.1f ms | %.2fx\n", tot_a*1e3, tot_b*1e3, tot_a/tot_b);
    return 0;
}
