// ADR-079 P4.33 step 2: THE INT8 GEMM TILE AGAINST ggml-0034's F32 GEMM AND STEP 1's KERNEL, at the zoo's
// n > 1 shapes. bench59 found step 1's one-quad-one-column kernel compute-bound at 0.70-0.76 GOP/s as a
// GEMM, a tie with F32 at n >= 32. This arm is ADR-077 gate 2's tile (1.95 GOP/s on L1-resident operands,
// int16 W) generalized to any K by gen_armv6_i8_gemm.py: the int8 W of a group of channel quads is
// unpacked to int16 pairs into an L1-sized block, and x (int16 pairs, two columns interleaved) streams
// past it with a PLD per line. bench61 measured why: an x or w line from the Pi Zero's L2 costs ~100
// cycles, as much as DRAM, and the PLD recovers most of it (x from L2 0.78 -> 1.45 GOP/s).
//
//   ./bench60 [reps] [w_block_bytes] [k_chunk] [col_pairs_per_block]
//       the unpacked int16 W kept in L1 per group of quads (default 8192), the K chunk (default 512), and
//       the column pairs whose accumulators stay in L1 (default 16)
//
// Checked before timing: both int8 arms' int32 accumulators equal an int64 reference on sampled columns,
// and both are within their quantization error of ggml's F32 output. The quantize is timed in both.
//
// Build (loom-armv6-build:bookworm), against a static ggml built from the patched tree:
//   python3 scripts/gen_armv6_i8_gemm.py tiles.S
//   g++ -O3 -std=c++17 -march=armv6+fp -mfloat-abi=hard -Wno-psabi -I<ggml>/include scripts/bench60.cpp tiles.S
//       <build>/src/libggml.a <build>/src/libggml-cpu.a <build>/src/libggml-base.a -fopenmp -lpthread -latomic -o bench60
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

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

extern "C" void ggml_armv6_i8_tile_p128(const uint32_t* x, const uint32_t* w, int32_t* acc);
extern "C" void ggml_armv6_i8_tile_p16(const uint32_t* x, const uint32_t* w, int32_t* acc);

// Step 1's kernel (ggml-0041), verbatim.
__asm__(
    "    .text\n    .arm\n    .align 4\n    .global gemv4_i8u2\n    .type gemv4_i8u2, %function\n"
    "gemv4_i8u2:\n"
    "    push {r3-r11, lr}\n"
    "    mov r4, #0\n    mov r5, #0\n    mov r6, #0\n    mov r7, #0\n"
    "1:\n"
    "    pld [r0, #128]\n"
    "    ldmia r1!, {r8, r9}\n"
    "    ldmia r0!, {r10, r11, r12, lr}\n"
    "    sxtb16 r3, r10\n    smlad r4, r3, r8, r4\n    sxtb16 r3, r10, ror #8\n    smlad r4, r3, r9, r4\n"
    "    sxtb16 r3, r11\n    smlad r5, r3, r8, r5\n    sxtb16 r3, r11, ror #8\n    smlad r5, r3, r9, r5\n"
    "    sxtb16 r3, r12\n    smlad r6, r3, r8, r6\n    sxtb16 r3, r12, ror #8\n    smlad r6, r3, r9, r6\n"
    "    sxtb16 r3, lr\n    smlad r7, r3, r8, r7\n    sxtb16 r3, lr, ror #8\n    smlad r7, r3, r9, r7\n"
    "    ldmia r1!, {r8, r9}\n"
    "    ldmia r0!, {r10, r11, r12, lr}\n"
    "    sxtb16 r3, r10\n    smlad r4, r3, r8, r4\n    sxtb16 r3, r10, ror #8\n    smlad r4, r3, r9, r4\n"
    "    sxtb16 r3, r11\n    smlad r5, r3, r8, r5\n    sxtb16 r3, r11, ror #8\n    smlad r5, r3, r9, r5\n"
    "    sxtb16 r3, r12\n    smlad r6, r3, r8, r6\n    sxtb16 r3, r12, ror #8\n    smlad r6, r3, r9, r6\n"
    "    sxtb16 r3, lr\n    smlad r7, r3, r8, r7\n    sxtb16 r3, lr, ror #8\n    smlad r7, r3, r9, r7\n"
    "    subs r2, r2, #2\n    bne 1b\n"
    "    pop {r3}\n"
    "    stmia r3, {r4-r7}\n"
    "    pop {r4-r11, pc}\n"
    "    .size gemv4_i8u2, .-gemv4_i8u2\n");
extern "C" void gemv4_i8u2(const uint32_t* w, const uint32_t* x, int count, int32_t* acc);

static int x_range(int K) {
    const long r = 2147483647L / (128L * K);
    return r > 32767 ? 32767 : (int) r;
}

static inline int32_t round_s32(float v) {
    float r;
    __asm__("vcvtr.s32.f32 %0, %1" : "=t"(r) : "t"(v));
    int32_t i;
    std::memcpy(&i, &r, 4);
    return i;
}

static inline uint32_t sxtb16(uint32_t v) { uint32_t r; __asm__("sxtb16 %0, %1" : "=r"(r) : "r"(v)); return r; }
static inline uint32_t sxtb16_ror8(uint32_t v) { uint32_t r; __asm__("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(v)); return r; }

static float absmax(const float* x, int K, int* R) {
    float m = 0.0f;
    for (int k = 0; k < K; ++k) m = std::fmax(m, std::fabs(x[k]));
    *R = x_range(K);
    return m;
}

// Step 1's layout: (x0, x2), (x1, x3) pairs per four k.
static float quantize_gemv(const float* x, int K, uint32_t* x2) {
    int R; const float m = absmax(x, K, &R);
    const float inv = m > 0 ? R / m : 0.0f;
    for (int k = 0; k < K; k += 4) {
        const int32_t q0 = round_s32(x[k] * inv), q1 = round_s32(x[k + 1] * inv);
        const int32_t q2 = round_s32(x[k + 2] * inv), q3 = round_s32(x[k + 3] * inv);
        x2[k / 2]     = (uint16_t) q0 | ((uint32_t) (uint16_t) q2 << 16);
        x2[k / 2 + 1] = (uint16_t) q1 | ((uint32_t) (uint16_t) q3 << 16);
    }
    return m / R;
}

// The tile's layout: pairs (x_k, x_k+1) in k order, the column pair's two columns interleaved word by word.
static float quantize_tile(const float* x, int K, uint32_t* xp /* stride 2 words */) {
    int R; const float m = absmax(x, K, &R);
    const float inv = m > 0 ? R / m : 0.0f;
    for (int k = 0; k < K; k += 2) {
        const int32_t q0 = round_s32(x[k] * inv), q1 = round_s32(x[k + 1] * inv);
        xp[k] = (uint16_t) q0 | ((uint32_t) (uint16_t) q1 << 16);
    }
    return m / R;
}

static void gemm_step1(const uint32_t* Wp, const float* sw, const float* X, uint32_t* x2, float* sx, float* y,
                       int K, int N, int T) {
    for (int c = 0; c < T; ++c) sx[c] = quantize_gemv(X + (size_t) c * K, K, x2 + (size_t) c * (K / 2));
    const int cb = std::max(1, 8192 / (2 * K)), qb = std::max(1, 65536 / (4 * K));
    for (int q0 = 0; q0 < N / 4; q0 += qb) {
        const int q1 = std::min(N / 4, q0 + qb);
        for (int c0 = 0; c0 < T; c0 += cb) {
            const int c1 = std::min(T, c0 + cb);
            for (int q = q0; q < q1; ++q)
                for (int c = c0; c < c1; ++c) {
                    int32_t acc[4];
                    gemv4_i8u2(Wp + (size_t) q * K, x2 + (size_t) c * (K / 2), K / 4, acc);
                    float* yc = y + (size_t) c * N + 4 * q;
                    for (int j = 0; j < 4; ++j) yc[j] = (float) acc[j] * (sx[c] * sw[4 * q + j]);
                }
        }
    }
}

struct TileCfg { int wb, kc, cpb; };

// The tile driver. W: a group of channel quads, unpacked to int16 pairs for a chunk of K, sized to stay in
// L1 (wb bytes); x (two columns interleaved) streams past it, the tile PLDs it, and each column pair's
// chunk is reused across the group's quads from L1. The accumulators of a block of cpb column pairs stay
// in L1 too ([cpb][qb][8], <= 2 KB): a buffer over all T, walked per call, evicted the W block (bench60 v2).
// When K needs more than one chunk, the W chunk is unpacked again per column block.
// x2t [T2][K/2][2] words; ws [qb][kc/2][4] words; acc [cpb][qb][8].
static void unpack(const uint32_t* Wp, int K, int g0, int g1, int k0, int kn, int kc, uint32_t* ws) {
    for (int q = g0; q < g1; ++q) {   // (b0, b2), (b1, b3) -> (b0, b1), (b2, b3)
        const uint32_t* src = Wp + (size_t) q * K + (size_t) (k0 / 4) * 4;
        uint32_t* dst = ws + (size_t) (q - g0) * (kc / 2) * 4;
        for (int kw = 0; kw < kn / 4; ++kw) {
            if ((kw & 1) == 0) __builtin_prefetch(src + kw * 4 + 32);
            for (int j = 0; j < 4; ++j) {
                const uint32_t v = src[kw * 4 + j], ev = sxtb16(v), od = sxtb16_ror8(v);
                dst[(2 * kw) * 4 + j]     = (ev & 0xffff) | (od << 16);
                dst[(2 * kw + 1) * 4 + j] = (ev >> 16) | (od & 0xffff0000u);
            }
        }
    }
}

static void gemm_tile(const uint32_t* Wp, const float* sw, const float* X, uint32_t* x2t, float* sx, float* y,
                      int K, int N, int T, uint32_t* ws, int32_t* acc, TileCfg cfg) {
    const int T2 = (T + 1) / 2;
    for (int c = 0; c < T; ++c) sx[c] = quantize_tile(X + (size_t) c * K, K, x2t + (size_t) (c / 2) * K + (c & 1));
    if (T & 1) {
        for (int p = 0; p < K / 2; ++p) x2t[(size_t) (T2 - 1) * K + 2 * p + 1] = 0;
        sx[T] = 0;
    }
    const int Q = N / 4, kc = std::min(K, cfg.kc), qb = std::max(1, std::min(Q, cfg.wb / (8 * kc)));
    const bool one_chunk = kc == K;
    for (int g0 = 0; g0 < Q; g0 += qb) {
        const int g1 = std::min(Q, g0 + qb);
        if (one_chunk) unpack(Wp, K, g0, g1, 0, K, kc, ws);
        for (int cp0 = 0; cp0 < T2; cp0 += cfg.cpb) {
            const int cp1 = std::min(T2, cp0 + cfg.cpb);
            std::memset(acc, 0, sizeof(int32_t) * 8 * qb * (cp1 - cp0));
            for (int k0 = 0; k0 < K; k0 += kc) {
                const int kn = std::min(kc, K - k0);
                if (!one_chunk) unpack(Wp, K, g0, g1, k0, kn, kc, ws);
                for (int cp = cp0; cp < cp1; ++cp) {
                    const uint32_t* xp = x2t + (size_t) cp * K + k0;
                    for (int q = g0; q < g1; ++q) {
                        const uint32_t* wq = ws + (size_t) (q - g0) * (kc / 2) * 4;
                        int32_t* a = acc + ((cp - cp0) * qb + (q - g0)) * 8;
                        int p = 0;
                        for (; p + 128 <= kn / 2; p += 128) ggml_armv6_i8_tile_p128(xp + 2 * p, wq + 4 * p, a);
                        for (; p < kn / 2; p += 16) ggml_armv6_i8_tile_p16(xp + 2 * p, wq + 4 * p, a);
                    }
                }
            }
            for (int cp = cp0; cp < cp1; ++cp)
                for (int t = 0; t < 2 && 2 * cp + t < T; ++t) {
                    const int c = 2 * cp + t;
                    float* yc = y + (size_t) c * N;
                    for (int q = g0; q < g1; ++q) {
                        const int32_t* a = acc + ((cp - cp0) * qb + (q - g0)) * 8 + 4 * t;
                        for (int j = 0; j < 4; ++j) yc[4 * q + j] = (float) a[j] * (sx[c] * sw[4 * q + j]);
                    }
                }
        }
    }
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    TileCfg cfg = { argc > 2 ? std::atoi(argv[2]) : 8192, argc > 3 ? std::atoi(argv[3]) : 512,
                    argc > 4 ? std::atoi(argv[4]) : 16 };
    {
        unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); f |= (1u << 24) | (1u << 25);   // RunFast
        __asm__ volatile("vmsr fpscr, %0" : : "r"(f));
    }
    struct Shape { int k, n, cols; const char* who; };
    const Shape shapes[] = {
        {288, 288, 457, "mt attn proj"}, {288, 1152, 457, "mt enc ffn up"}, {1152, 288, 457, "mt enc ffn down"},
        {704, 176, 75, "conformer ffn down"}, {256, 256, 550, "wakehubert"}, {256, 256, 1101, "citrinet 1x1"},
        {768, 768, 16, "ner attn"}, {768, 3072, 16, "ner ffn up"}, {3072, 768, 16, "ner ffn down"},
        {192, 576, 100, "heart"}, {288, 1152, 8, "n=8"}, {288, 1152, 16, "n=16"}, {288, 1152, 24, "n=24"},
        {288, 1152, 32, "n=32"}, {288, 1152, 64, "n=64"},
    };
    std::printf("reps=%d w_block=%d k_chunk=%d col_pairs/block=%d\n%-20s %5s %5s %4s | %8s %8s %8s | %6s %6s %6s | %5s %5s\n",
                reps, cfg.wb, cfg.kc, cfg.cpb, "shape", "K", "N", "T", "f32 ms", "s1 ms", "tile ms", "f32", "s1", "tile",
                "s1/f", "t/f");
    double tf = 0, t1 = 0, tt = 0;
    for (const Shape& s : shapes) {
        const int K = s.k, N = s.n, T = s.cols, T2 = (T + 1) / 2;
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
        ggml_context* ctx = ggml_init(ip);
        ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
        ggml_tensor* y = ggml_mul_mat(ctx, w, x);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0, 1);
        float* W = (float*) w->data;
        for (int64_t i = 0; i < ggml_nelements(w); ++i) W[i] = nd(rng) * 0.05f;
        for (int64_t i = 0; i < ggml_nelements(x); ++i) ((float*) x->data)[i] = nd(rng);
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_cplan plan = ggml_graph_plan(gf, 1, nullptr);
        std::vector<uint8_t> work(plan.work_size + 1);
        plan.work_data = work.data();

        std::vector<int8_t> Wq((size_t) N * K);
        float* sw = (float*) aligned_alloc(32, N * 4);
        for (int n = 0; n < N; ++n) {
            float m = 0;
            for (int k = 0; k < K; ++k) m = std::fmax(m, std::fabs(W[(size_t) n * K + k]));
            sw[n] = m / 127;
            for (int k = 0; k < K; ++k) Wq[(size_t) n * K + k] = (int8_t) std::lrint(W[(size_t) n * K + k] / sw[n]);
        }
        uint32_t* Wp = (uint32_t*) aligned_alloc(32, (size_t) N * K);
        for (int n = 0; n < N; n += 4)
            for (int kw = 0; kw < K / 4; ++kw)
                for (int j = 0; j < 4; ++j)
                    std::memcpy(Wp + ((size_t) n * (K / 4) + (size_t) kw * 4 + j), &Wq[(size_t) (n + j) * K + kw * 4], 4);
        uint32_t* x2 = (uint32_t*) aligned_alloc(32, (size_t) T * K * 2);
        uint32_t* x2t = (uint32_t*) aligned_alloc(32, (size_t) T2 * K * 4);
        float* sx = (float*) aligned_alloc(32, (T + 2) * 4);
        float* y1 = (float*) aligned_alloc(32, (size_t) T * N * 4);
        float* yt = (float*) aligned_alloc(32, (size_t) T * N * 4);
        const int qbmax = std::max(1, std::min(N / 4, cfg.wb / (8 * std::min(K, cfg.kc))));
        uint32_t* ws = (uint32_t*) aligned_alloc(32, (size_t) qbmax * std::min(K, cfg.kc) * 8 + 32);
        int32_t* acc = (int32_t*) aligned_alloc(32, (size_t) cfg.cpb * qbmax * 32 + 32);

        ggml_graph_compute(gf, &plan);
        gemm_step1(Wp, sw, (const float*) x->data, x2, sx, y1, K, N, T);
        gemm_tile(Wp, sw, (const float*) x->data, x2t, sx, yt, K, N, T, ws, acc, cfg);
        for (int c = 0; c < T; c += std::max(1, T / 7))
            for (int n = 0; n < N; ++n) {   // the tile's accumulators, through its own quantized x
                int64_t r = 0;
                const uint32_t* xc = x2t + (size_t) (c / 2) * K + (c & 1);
                for (int k = 0; k < K; ++k) {
                    const uint32_t v = xc[(k / 2) * 2];
                    r += (int64_t) Wq[(size_t) n * K + k] * (int16_t) ((k & 1) ? (v >> 16) : (v & 0xffff));
                }
                const float want = (float) (int32_t) r * (sx[c] * sw[n]);
                if (yt[(size_t) c * N + n] != want) { std::printf("%s: tile mismatch at c=%d n=%d\n", s.who, c, n); return 1; }
            }
        double e1 = 0, et = 0, ref = 0;
        for (size_t i = 0; i < (size_t) T * N; ++i) {
            const float f = ((float*) y->data)[i];
            e1 = std::fmax(e1, std::fabs(y1[i] - f)); et = std::fmax(et, std::fabs(yt[i] - f)); ref = std::fmax(ref, std::fabs(f));
        }
        if (!(e1 / ref < 2e-2) || !(et / ref < 2e-2)) { std::printf("%s: int8 error %.2e / %.2e\n", s.who, e1 / ref, et / ref); return 1; }

        std::vector<double> af, a1, at;
        for (int r = 0; r < reps; ++r) {   // arms interleaved
            double t0 = now(); ggml_graph_compute(gf, &plan); af.push_back(now() - t0);
            t0 = now(); gemm_step1(Wp, sw, (const float*) x->data, x2, sx, y1, K, N, T); a1.push_back(now() - t0);
            t0 = now(); gemm_tile(Wp, sw, (const float*) x->data, x2t, sx, yt, K, N, T, ws, acc, cfg); at.push_back(now() - t0);
        }
        std::sort(af.begin(), af.end()); std::sort(a1.begin(), a1.end()); std::sort(at.begin(), at.end());
        const double mf = af[af.size() / 2], m1 = a1[a1.size() / 2], mt = at[at.size() / 2], op = 2.0 * K * N * T / 1e9;
        tf += mf; t1 += m1; tt += mt;
        std::printf("%-20s %5d %5d %4d | %8.2f %8.2f %8.2f | %6.3f %6.3f %6.3f | %4.2fx %4.2fx\n", s.who, K, N, T, mf * 1e3,
                    m1 * 1e3, mt * 1e3, op / mf, op / m1, op / mt, mf / m1, mf / mt);
        std::free(sw); std::free(Wp); std::free(x2); std::free(x2t); std::free(sx); std::free(y1); std::free(yt);
        std::free(ws); std::free(acc);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    std::printf("%-20s %16s | %8.2f %8.2f %8.2f | %20s | %4.2fx %4.2fx\n", "sum of medians", "", tf * 1e3, t1 * 1e3, tt * 1e3, "",
                tf / t1, tf / tt);
    return 0;
}
