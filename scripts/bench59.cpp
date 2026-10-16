// ADR-079 P4.33 step 1, the other half: A WEIGHT IN THE INT8 BUFFER MUST BE SERVED AT EVERY COLUMN COUNT.
// Moonshine's decoder weights run one column per step (bench58: 4.29x over ggml-0033) but its encoder and
// cross-attention run the same kind of weights at n = T_enc ~ 457, where ggml-0034's F32 GEMM is the
// incumbent (0.76-0.79 GOP/s). This times bench58's int8 kernel (gemv4_i8u2: one channel quad x one
// column) driven over blocks of columns and rows, against ggml's F32 mul_mat as the engine runs it, at
// the zoo's n > 1 shapes. The dynamic quantize of every column is inside the timed region.
//
//   ./bench59 [reps] [xblock_bytes] [wblock_bytes]
//       xblock: the int16 X columns kept in L1 per pass (default 8192)
//       wblock: the int8 W rows kept in L2 per pass (default 65536)
//
// Checked before timing: every int32 accumulator equals an int64 reference, and the int8 result is
// within its quantization error of ggml's F32 output.
//
// Build (loom-armv6-build:bookworm), against a static ggml built from the patched tree:
//   g++ -O3 -std=c++17 -march=armv6+fp -mfloat-abi=hard -I<ggml>/include scripts/bench59.cpp
//       <build>/src/libggml.a <build>/src/libggml-cpu.a <build>/src/libggml-base.a -fopenmp -lpthread -latomic -o bench59
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

// bench58's gemv4_i8u2, verbatim: one channel quad, `count` (even) k-words; acc[4] out.
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
    const long r = 2147483647L / (127L * K);
    return r > 32767 ? 32767 : (int) r;
}

static inline int32_t round_s32(float v) {
    float r;
    __asm__("vcvtr.s32.f32 %0, %1" : "=t"(r) : "t"(v));
    int32_t i;
    std::memcpy(&i, &r, 4);
    return i;
}

static float quantize_x(const float* x, int K, uint32_t* x2) {
    float m = 0.0f;
    for (int k = 0; k < K; ++k) m = std::fmax(m, std::fabs(x[k]));
    const int R = x_range(K);
    const float sx = m / R, inv = m > 0 ? R / m : 0.0f;
    for (int k = 0; k < K; k += 4) {
        const int32_t q0 = round_s32(x[k] * inv), q1 = round_s32(x[k + 1] * inv);
        const int32_t q2 = round_s32(x[k + 2] * inv), q3 = round_s32(x[k + 3] * inv);
        x2[k / 2]     = (uint16_t) q0 | ((uint32_t) (uint16_t) q2 << 16);
        x2[k / 2 + 1] = (uint16_t) q1 | ((uint32_t) (uint16_t) q3 << 16);
    }
    return sx;
}

// y[c][n] for c < T, n < N. X [T][K] (ggml's src1, K contiguous), Wp [N/4][K/4][4] words, y [T][N].
static void gemm_i8(const uint32_t* Wp, const float* sw, const float* X, uint32_t* x2, float* sx, float* y,
                    int K, int N, int T, int xblock, int wblock) {
    for (int c = 0; c < T; ++c) sx[c] = quantize_x(X + (size_t) c * K, K, x2 + (size_t) c * (K / 2));
    const int cb = std::max(1, xblock / (2 * K));          // columns of int16 X per L1 pass
    const int qb = std::max(1, wblock / (4 * K));          // channel quads of int8 W per L2 pass
    for (int q0 = 0; q0 < N / 4; q0 += qb) {
        const int q1 = std::min(N / 4, q0 + qb);
        for (int c0 = 0; c0 < T; c0 += cb) {
            const int c1 = std::min(T, c0 + cb);
            for (int q = q0; q < q1; ++q) {
                const uint32_t* wq = Wp + (size_t) q * K;
                const float s0 = sw[4 * q], s1 = sw[4 * q + 1], s2 = sw[4 * q + 2], s3 = sw[4 * q + 3];
                for (int c = c0; c < c1; ++c) {
                    int32_t acc[4];
                    gemv4_i8u2(wq, x2 + (size_t) c * (K / 2), K / 4, acc);
                    float* yc = y + (size_t) c * N + 4 * q;
                    const float s = sx[c];
                    yc[0] = (float) acc[0] * (s * s0); yc[1] = (float) acc[1] * (s * s1);
                    yc[2] = (float) acc[2] * (s * s2); yc[3] = (float) acc[3] * (s * s3);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const int xblock = argc > 2 ? std::atoi(argv[2]) : 8192, wblock = argc > 3 ? std::atoi(argv[3]) : 65536;
    {
        unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); f |= (1u << 24) | (1u << 25);   // RunFast
        __asm__ volatile("vmsr fpscr, %0" : : "r"(f));
    }
    struct Shape { int k, n, cols; const char* who; };
    const Shape shapes[] = {
        {288, 288, 457, "mt attn proj"}, {288, 1152, 457, "mt enc ffn up"}, {1152, 288, 457, "mt enc ffn down"},
        {704, 176, 75, "conformer ffn down"}, {256, 256, 550, "wakehubert"}, {768, 768, 16, "ner attn"},
        {768, 3072, 16, "ner ffn up"}, {3072, 768, 16, "ner ffn down"}, {192, 576, 100, "heart"},
        {288, 1152, 2, "n=2"}, {288, 1152, 4, "n=4"}, {288, 1152, 8, "n=8"}, {288, 1152, 16, "n=16"},
        {288, 1152, 24, "n=24"}, {288, 1152, 32, "n=32"},
    };
    std::printf("reps=%d xblock=%d wblock=%d\n%-20s %5s %5s %4s | %9s %9s | %7s %7s | %6s | %s\n", reps, xblock, wblock,
                "shape", "K", "N", "T", "f32 ms", "i8 ms", "f32 GOP", "i8 GOP", "x", "rel err");
    double tf = 0, ti = 0;
    for (const Shape& s : shapes) {
        const int K = s.k, N = s.n, T = s.cols;
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
        float* sx = (float*) aligned_alloc(32, T * 4);
        float* yi = (float*) aligned_alloc(32, (size_t) T * N * 4);

        ggml_graph_compute(gf, &plan);
        gemm_i8(Wp, sw, (const float*) x->data, x2, sx, yi, K, N, T, xblock, wblock);
        for (int c = 0; c < T; c += std::max(1, T / 7))           // accumulators, sampled columns, every row
            for (int n = 0; n < N; ++n) {
                int64_t r = 0;
                const uint32_t* xc = x2 + (size_t) c * (K / 2);
                for (int k = 0; k < K; ++k) {
                    const uint32_t v = xc[(k / 4) * 2 + (k & 1)];
                    r += (int64_t) Wq[(size_t) n * K + k] * (int16_t) ((k & 2) ? (v >> 16) : (v & 0xffff));
                }
                const float want = (float) (int32_t) r * (sx[c] * sw[n]);
                if (yi[(size_t) c * N + n] != want) { std::printf("%s: mismatch at c=%d n=%d\n", s.who, c, n); return 1; }
            }
        double err = 0, ref = 0;
        for (size_t i = 0; i < (size_t) T * N; ++i) {
            err = std::fmax(err, std::fabs(yi[i] - ((float*) y->data)[i]));
            ref = std::fmax(ref, std::fabs(((float*) y->data)[i]));
        }
        if (!(err / ref < 2e-2)) { std::printf("%s: int8 error %.2e of peak\n", s.who, err / ref); return 1; }

        std::vector<double> af, ai;
        for (int r = 0; r < reps; ++r) {   // arms interleaved
            double t0 = now(); ggml_graph_compute(gf, &plan); af.push_back(now() - t0);
            t0 = now(); gemm_i8(Wp, sw, (const float*) x->data, x2, sx, yi, K, N, T, xblock, wblock); ai.push_back(now() - t0);
        }
        std::sort(af.begin(), af.end()); std::sort(ai.begin(), ai.end());
        const double mf = af[af.size() / 2], mi = ai[ai.size() / 2], op = 2.0 * K * N * T / 1e9;
        tf += mf; ti += mi;
        std::printf("%-20s %5d %5d %4d | %9.3f %9.3f | %7.3f %7.3f | %5.2fx | %.1e\n", s.who, K, N, T, mf * 1e3, mi * 1e3,
                    op / mf, op / mi, mf / mi, err / ref);
        std::free(sw); std::free(Wp); std::free(x2); std::free(sx); std::free(yi);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }
    std::printf("%-20s %16s | %9.3f %9.3f | %25.2fx\n", "sum of medians", "", tf * 1e3, ti * 1e3, tf / ti);
    return 0;
}
