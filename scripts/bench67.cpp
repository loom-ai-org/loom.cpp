// P4.40: VITS's vocoder convolutions on the Raspberry Pi 4 -- the direct 1-D kernel ggml-0006/0007 ship
// against variants of it, at the eighteen resblock shapes of a piper (VITS) synthesis, every arm in ONE
// process on the same inputs.
//
// NOT part of the build.
//
// WHY IT EXISTS. After P4.39 the vocoder's resblocks are the largest share of a VITS synthesis left on the
// board (~37%), and they run the direct kernel at ~20-23 GFLOP/s against the 57 of the A72's FMA roofline
// (4 cores x 1.8 GHz x one 4-lane FMA per cycle). The im2col route was measured out in P4.39 (ADR-083), so
// the question is what the direct kernel itself is losing:
//
//   base         the shipped kernel, transcribed: a zero-padded, LEAKY_RELU'd copy of the whole activation,
//                weights packed [ic][kx][oc], position blocks outermost, a 4 x 16 tile, bias in the
//                accumulators, the residual added on the store
//   ...:pk       the same with each tile's weights packed contiguously ([oc/OCB][ic][kx][OCB])
//   t8x12        8 output channels x 12 positions (24 accumulators): 5 loads per 24 FMAs, not 5 per 16
//   g<N>         N position blocks per output-channel sweep (weights stay in L1 across them)
//   s<C>         no global padded copy: each thread stages C positions at a time into an L1/L2 buffer
//
//   b<KB>        the same, the chunk sized so the staged slice is about KB kilobytes
//
// Every arm sums each output in the same order (bias, then input channel, then tap), so every arm must
// match `base` BYTE FOR BYTE; the bench checks it on every shape and says so.
//
// THE ANSWER (Pi 4, 4 threads, sum of the eighteen per-shape medians, every arm byte-identical): base 580
// ms; base:pk 1.29x; t8x12:pk 1.40x (t8x8 1.30x, t4x24 1.35x, :g4 no better); t8x12:pk with staging
// 1.69x at s252, 1.78x at s96, 1.84x at s36, **1.91x at b16** (309 ms), s12 1.82x. Shipped as ggml-0053
// (ADR-086), b16 and all.
//
// WHAT IT DOES NOT MEASURE: the library's own code generation. Here `tile` is inlined into `sweep` and
// GCC keeps its `acc[OCB][NV]` array in registers; pasted into ops.cpp, the same source stayed out of line
// and stored all 24 accumulators to the stack on every tap, and VITS got 0.89x slower instead of 1.30x
// faster (Retro-093). Disassemble the library's copy, not this one.
//
//   g++ -O3 -std=c++17 -mcpu=cortex-a72 -fopenmp scripts/bench67.cpp -o bench67
//   ./bench67 [reps] [arm ...]          # default: 7 reps, every arm
#include <arm_neon.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static inline float32x4_t leaky_v(float32x4_t v, float32x4_t z, float32x4_t s) {
    return vfmaq_f32(vbslq_f32(vcgtq_f32(v, z), v, z), s, vbslq_f32(vcltq_f32(v, z), v, z));
}
static inline float leaky_s(float v, float slope) {
    return (v > 0.0f ? v : 0.0f) + slope * (v < 0.0f ? v : 0.0f);
}
static void leaky_copy(float * dst, const float * src, int64_t n, float slope) {
    int64_t i = 0;
    const float32x4_t z = vdupq_n_f32(0.0f), s = vdupq_n_f32(slope);
    for (; i + 4 <= n; i += 4) vst1q_f32(dst + i, leaky_v(vld1q_f32(src + i), z, s));
    for (; i < n; ++i) dst[i] = leaky_s(src[i], slope);
}

// One OCB x (4*NV) tile. `w` points at this tile's weight for (ic = 0, kx = 0); successive (ic, kx) are
// `ws` floats apart (OC for the shipped [ic][kx][oc] pack, OCB for a per-tile pack). `xoff` is the tile's
// first position in `xp`'s own coordinates, `p0` in the output's.
template <int OCB, int NV, bool RES, bool CLAMP>
static inline void tile(const float * xp, int64_t LP, int64_t xoff, const float * w, int64_t ws,
                        const float * bias, float * y, const float * res, int64_t IC, int64_t KW,
                        int64_t OL, int64_t dil, int64_t oc0, int64_t p0, int64_t pmin) {
    static_assert(OCB % 4 == 0, "lane-broadcast weights");
    float32x4_t acc[OCB][NV];
    for (int i = 0; i < OCB; ++i) {
        const float32x4_t b = vdupq_n_f32(bias[oc0 + i]);
        for (int j = 0; j < NV; ++j) acc[i][j] = b;
    }
    for (int64_t ic = 0; ic < IC; ++ic) {
        const float * xrow = xp + ic * LP + xoff;
        const float * wr   = w + ic * KW * ws;
        for (int64_t kx = 0; kx < KW; ++kx) {
            const float * q = xrow + kx * dil;
            float32x4_t xv[NV];
            for (int j = 0; j < NV; ++j) xv[j] = vld1q_f32(q + j * 4);
            for (int g = 0; g < OCB / 4; ++g) {
                const float32x4_t wv = vld1q_f32(wr + kx * ws + 4 * g);
                for (int j = 0; j < NV; ++j) {
                    acc[4*g+0][j] = vfmaq_laneq_f32(acc[4*g+0][j], xv[j], wv, 0);
                    acc[4*g+1][j] = vfmaq_laneq_f32(acc[4*g+1][j], xv[j], wv, 1);
                    acc[4*g+2][j] = vfmaq_laneq_f32(acc[4*g+2][j], xv[j], wv, 2);
                    acc[4*g+3][j] = vfmaq_laneq_f32(acc[4*g+3][j], xv[j], wv, 3);
                }
            }
        }
    }
    if (CLAMP) {
        float t[OCB][NV * 4];
        for (int i = 0; i < OCB; ++i)
            for (int j = 0; j < NV; ++j) vst1q_f32(&t[i][j * 4], acc[i][j]);
        for (int i = 0; i < OCB; ++i) {
            const int64_t off = (oc0 + i) * OL + p0;
            for (int64_t k = pmin - p0; k < NV * 4; ++k) y[off + k] = t[i][k] + (RES ? res[off + k] : 0.0f);
        }
    } else if (RES) {
        for (int i = 0; i < OCB; ++i)
            for (int j = 0; j < NV; ++j) {
                const int64_t off = (oc0 + i) * OL + p0 + j * 4;
                vst1q_f32(y + off, vaddq_f32(acc[i][j], vld1q_f32(res + off)));
            }
    } else {
        for (int i = 0; i < OCB; ++i)
            for (int j = 0; j < NV; ++j) vst1q_f32(y + (oc0 + i) * OL + p0 + j * 4, acc[i][j]);
    }
}

struct Prob {
    int64_t C, L, KW, dil, pad, LP, OL;
    const float * x; const float * w; const float * bias; float * y;
    float * xp; float * wp; std::vector<float *> sbuf;   // global padded copy / per-thread staging
    float slope;
};

struct Arm {
    std::string name;
    int ocb = 4, nv = 4;       // tile
    bool tpack = false;        // per-tile weight pack
    int group = 1;             // position blocks per output-channel sweep
    int stage = 0;             // 0 = global padded copy; else positions per staged chunk
    int kb = 0;                // else: a chunk sized so the staged slice is about this many KB
};

template <int OCB, int NV>
static void sweep(const Arm & a, Prob & p, int ith, int nth) {
    const int64_t IC = p.C, OC = p.C, KW = p.KW, OL = p.OL, dil = p.dil, L = p.L;
    const int64_t P = 4 * NV;
    const int64_t ws = a.tpack ? OCB : OC;
    auto wt = [&](int64_t oc0) -> const float * {
        return a.tpack ? p.wp + (oc0 / OCB) * IC * KW * OCB : p.wp + oc0;
    };

    // weights: the shipped pack, or one contiguous run per tile
    if (a.tpack) {
        for (int64_t t = ith; t < OC / OCB; t += nth)
            for (int64_t ic = 0; ic < IC; ++ic)
                for (int64_t kx = 0; kx < KW; ++kx)
                    for (int i = 0; i < OCB; ++i)
                        p.wp[((t * IC + ic) * KW + kx) * OCB + i] = p.w[(t * OCB + i) * (IC * KW) + ic * KW + kx];
    } else {
        for (int64_t ic = ith; ic < IC; ic += nth)
            for (int64_t kx = 0; kx < KW; ++kx)
                for (int64_t oc = 0; oc < OC; ++oc)
                    p.wp[(ic * KW + kx) * OC + oc] = p.w[oc * (IC * KW) + ic * KW + kx];
    }
    if (!a.stage && !a.kb) {
        for (int64_t ic = ith; ic < IC; ic += nth) {
            float * row = p.xp + ic * p.LP;
            const int64_t lo = p.pad, n = std::min(L, p.LP - lo);
            memset(row, 0, lo * sizeof(float));
            if (p.LP > lo + n) memset(row + lo + n, 0, (p.LP - lo - n) * sizeof(float));
            leaky_copy(row + lo, p.x + ic * L, n, p.slope);
        }
    }
#pragma omp barrier
    const int64_t nblk = OL / P;
    const int64_t per  = (nblk + nth - 1) / nth;
    const int64_t b0   = std::min<int64_t>(ith * per, nblk), b1 = std::min(b0 + per, nblk);
    const bool tail    = nblk * P < OL;
    const float * res  = p.x;

    if (!a.stage && !a.kb) {
        for (int64_t g0 = b0; g0 < b1; g0 += a.group) {
            const int64_t g1 = std::min<int64_t>(g0 + a.group, b1);
            for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB)
                for (int64_t b = g0; b < g1; ++b)
                    tile<OCB, NV, true, false>(p.xp, p.LP, b * P, wt(oc0), ws, p.bias, p.y, res, IC, KW, OL,
                                               dil, oc0, b * P, 0);
        }
        if (tail)
            for (int64_t oc0 = ith * OCB; oc0 < OC; oc0 += nth * OCB)
                tile<OCB, NV, true, true>(p.xp, p.LP, OL - P, wt(oc0), ws, p.bias, p.y, res, IC, KW, OL, dil,
                                          oc0, OL - P, nblk * P);
    } else {
        // per-thread staging: positions [s0, s1) need input [s0 - pad, s1 - pad + (KW-1)*dil)
        float * buf = p.sbuf[ith];
        const int64_t H = (KW - 1) * dil;
        const int64_t cb = a.kb ? std::max<int64_t>(1, ((int64_t) a.kb * 1024 / (4 * IC) - H) / P)
                                : std::max<int64_t>(1, a.stage / P);
        for (int64_t c0 = b0; c0 < b1; c0 += cb) {
            const int64_t c1 = std::min(c0 + cb, b1);
            const bool last = tail && c1 == nblk;
            const int64_t s0 = c0 * P, s1 = last ? OL : c1 * P;
            const int64_t LPc = s1 - s0 + H;
            for (int64_t ic = 0; ic < IC; ++ic) {
                float * row = buf + ic * LPc;
                const int64_t in0 = s0 - p.pad;                 // input position of row[0]
                const int64_t lo  = std::max<int64_t>(0, -in0);  // first row index with a real input
                const int64_t hi  = std::min<int64_t>(LPc, L - in0);
                if (lo > 0) memset(row, 0, lo * sizeof(float));
                if (hi < LPc) memset(row + std::max(hi, lo), 0, (LPc - std::max(hi, lo)) * sizeof(float));
                if (hi > lo) leaky_copy(row + lo, p.x + ic * L + in0 + lo, hi - lo, p.slope);
            }
            for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB) {
                for (int64_t b = c0; b < c1; ++b)
                    tile<OCB, NV, true, false>(buf, LPc, b * P - s0, wt(oc0), ws, p.bias, p.y, res, IC, KW, OL,
                                               dil, oc0, b * P, 0);
                if (last)
                    tile<OCB, NV, true, true>(buf, LPc, OL - P - s0, wt(oc0), ws, p.bias, p.y, res, IC, KW, OL,
                                              dil, oc0, OL - P, nblk * P);
            }
        }
    }
#pragma omp barrier
}

static void run(const Arm & a, Prob & p, int ith, int nth) {
    if (a.ocb == 4 && a.nv == 4)  return sweep<4, 4>(a, p, ith, nth);
    if (a.ocb == 8 && a.nv == 3)  return sweep<8, 3>(a, p, ith, nth);
    if (a.ocb == 8 && a.nv == 2)  return sweep<8, 2>(a, p, ith, nth);
    if (a.ocb == 4 && a.nv == 6)  return sweep<4, 6>(a, p, ith, nth);
    std::abort();
}

static Arm parse(const std::string & s) {
    // base | t8x12 | t8x8 | t4x24, then any of :pk :g<N> :s<C>
    Arm a; a.name = s;
    size_t i = s.find(':');
    const std::string head = s.substr(0, i);
    if (head == "t8x12") { a.ocb = 8; a.nv = 3; }
    else if (head == "t8x8") { a.ocb = 8; a.nv = 2; }
    else if (head == "t4x24") { a.ocb = 4; a.nv = 6; }
    else if (head != "base") { fprintf(stderr, "bad arm %s\n", s.c_str()); exit(2); }
    while (i != std::string::npos) {
        const size_t j = s.find(':', i + 1);
        const std::string m = s.substr(i + 1, j == std::string::npos ? std::string::npos : j - i - 1);
        if (m == "pk") a.tpack = true;
        else if (m[0] == 'g') a.group = atoi(m.c_str() + 1);
        else if (m[0] == 's') a.stage = atoi(m.c_str() + 1);
        else if (m[0] == 'b') a.kb = atoi(m.c_str() + 1);
        else { fprintf(stderr, "bad modifier %s\n", m.c_str()); exit(2); }
        i = j;
    }
    return a;
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 7;
    std::vector<Arm> arms;
    for (int i = 2; i < argc; ++i) arms.push_back(parse(argv[i]));
    if (arms.empty())
        for (const char * s : {"base", "base:pk", "t8x12:pk", "t8x8:pk", "t4x24:pk", "base:pk:g4", "t8x12:pk:g4",
                               "base:pk:s256", "t8x12:pk:s252", "t8x12:pk:s504"})
            arms.push_back(parse(s));
    if (arms[0].name != "base") arms.insert(arms.begin(), parse("base"));

    // piper's resblock "2": kernels 3/5/7 at dilations (1,2) / (2,6) / (3,12), at the three upsampled lengths
    struct S { int C, L, KW, dil; };
    std::vector<S> shapes;
    for (auto cl : {std::pair<int,int>{128, 2288}, {64, 18304}, {32, 73216}})
        for (auto kd : {std::pair<int,int>{3, 1}, {3, 2}, {5, 2}, {5, 6}, {7, 3}, {7, 12}})
            shapes.push_back({cl.first, cl.second, kd.first, kd.second});

    const int nth = omp_get_max_threads();
    std::vector<float> flush(8 << 20);
    printf("threads %d, reps %d (median); GF/s = 2*C*C*KW*L / t\n", nth, reps);
    std::vector<double> tot(arms.size(), 0.0);
    for (const S & s : shapes) {
        const int64_t C = s.C, L = s.L, KW = s.KW, dil = s.dil, H = (KW - 1) * dil;
        std::vector<float> x(C * L), w(C * C * KW), bias(C);
        uint32_t r = 12345u + (uint32_t) (C * 7 + KW * 13 + dil);
        auto rnd = [&]() { r = r * 1664525u + 1013904223u; return (float) ((r >> 8) & 0xffff) / 32768.0f - 1.0f; };
        for (auto & v : x) v = rnd();
        for (auto & v : w) v = rnd() * 0.1f;
        for (auto & v : bias) v = rnd() * 0.1f;
        Prob p{C, L, KW, dil, H / 2, L + H, L, x.data(), w.data(), bias.data(), nullptr, nullptr, nullptr, {}, 0.1f};
        std::vector<float> xp(C * p.LP), wp(C * C * KW);
        std::vector<std::vector<float>> ys(arms.size(), std::vector<float>(C * L));
        std::vector<std::vector<float>> sb(nth, std::vector<float>(C * (1024 + H + 64)));
        p.xp = xp.data(); p.wp = wp.data();
        for (auto & b : sb) p.sbuf.push_back(b.data());

        std::vector<std::vector<double>> ms(arms.size());
        for (int rep = 0; rep < reps + 1; ++rep)
            for (size_t ai = 0; ai < arms.size(); ++ai) {
                const size_t k = (ai + rep) % arms.size();      // rotate the order every round
                for (size_t i = 0; i < flush.size(); i += 16) flush[i] += 1.0f;
                p.y = ys[k].data();
                double t = 0;
#pragma omp parallel num_threads(nth)
                {
#pragma omp barrier
                    const auto t0 = std::chrono::steady_clock::now();
                    run(arms[k], p, omp_get_thread_num(), nth);
                    if (omp_get_thread_num() == 0)
                        t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                }
                if (rep > 0) ms[k].push_back(t);
            }
        const double gf = 2.0 * C * C * KW * L * 1e-6;
        printf("C%-3lld L%-6lld kw%lld d%-2lld", (long long) C, (long long) L, (long long) KW, (long long) dil);
        double base_ms = 0;
        for (size_t k = 0; k < arms.size(); ++k) {
            std::sort(ms[k].begin(), ms[k].end());
            const double m = ms[k][ms[k].size() / 2];
            if (k == 0) base_ms = m;
            tot[k] += m;
            const bool same = memcmp(ys[k].data(), ys[0].data(), C * L * sizeof(float)) == 0;
            printf("  %s %.2f (%.1f, %.2fx)%s", arms[k].name.c_str(), m, gf / m, base_ms / m, same ? "" : " MISMATCH");
        }
        printf("\n");
    }
    printf("TOTAL");
    for (size_t k = 0; k < arms.size(); ++k) printf("  %s %.1f ms (%.3fx)", arms[k].name.c_str(), tot[k], tot[0] / tot[k]);
    printf("\n");
}
