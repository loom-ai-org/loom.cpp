// ADR-079 P4.33 step 1: THE DYNAMIC INT8 GEMV AGAINST THE SHIPPED F32 ONE, at the zoo's decode shapes.
// ADR-077's cross-model analysis put bench48's int8 GEMV at 2.14x its F32 GEMV on Moonshine's 288 x 32768
// vocabulary projection. The F32 arm has since shipped as ggml-0033 (a float sum, four rows a pass), so
// this re-asks the question against that kernel, on every GEMV shape a Pi Zero model runs, with the
// activation quantize in the timed region and the weights cold, as a decode step sees them.
//
//   ./bench58 [reps]
//
//   f32     ggml-0033's ggml_armv6_gemv_rows, copied verbatim: y[n] = float sum of W[n][k] x[k] in k order
//   i8      x quantized per call (absmax, symmetric, to the widest range SMLAD's int32 sum can hold at this
//           K), W per-channel int8 repacked by channel quad ([N/4][K/4][4] words, as ADR-076's Q8_0_PC
//           rows become at load), SXTB16 + SMLAD in asm, F32 out (acc * sx * sw[n])
//   i8u2    the same loop unrolled to 32 B of W per pass, one PLD per line
//
// Every shape is checked before it is timed: the asm accumulators equal a C int64 reference exactly,
// and the int8 result is within its quantization error of F32. Weights are cold: each arm cycles through
// enough copies of its matrix to cover 8 MB, more than the L2 (128 KB) many times over.
//
//   gcc -O3 -march=armv6+fp -mfloat-abi=hard -o bench58 bench58.c -lm      (loom-armv6-build:bookworm)
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

// ggml-0033 (ggml-cpu.c), verbatim.
static void ggml_armv6_gemv_rows(const char * a, size_t lda, const float * x, float * y,
                                 int64_t k, int64_t r0, int64_t r1) {
    int64_t n = r0;
    for (; n + 4 <= r1; n += 4) {
        const float * w0 = (const float *) (a + n * lda);
        const float * w1 = (const float *) (a + (n + 1) * lda);
        const float * w2 = (const float *) (a + (n + 2) * lda);
        const float * w3 = (const float *) (a + (n + 3) * lda);
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int64_t i = 0; i < k; ++i) {
            const float v = x[i];
            s0 += w0[i] * v; s1 += w1[i] * v; s2 += w2[i] * v; s3 += w3[i] * v;
        }
        y[n] = s0; y[n + 1] = s1; y[n + 2] = s2; y[n + 3] = s3;
    }
    for (; n < r1; ++n) {
        const float * w = (const float *) (a + n * lda);
        float s = 0.0f;
        for (int64_t i = 0; i < k; ++i) s += w[i] * x[i];
        y[n] = s;
    }
}

// bench48's gemv4_i8: one channel quad, `count` k-words. r0 = w, r1 = x pairs (xe, xo), r2 = count,
// acc on the stack; r4-r7 accumulators, r8/r9 = xe/xo, r10-r12/lr the quad's four words.
__asm__(
    "    .text\n    .arm\n    .align 4\n    .global gemv4_i8\n    .type gemv4_i8, %function\n"
    "gemv4_i8:\n"
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
    "    subs r2, r2, #1\n    bne 1b\n"
    "    pop {r3}\n"
    "    stmia r3, {r4-r7}\n"
    "    pop {r4-r11, pc}\n"
    "    .size gemv4_i8, .-gemv4_i8\n");
void gemv4_i8(const uint32_t* w, const uint32_t* x, int count, int32_t* acc);

// The same, two k-words a pass (32 B of W, one line), one PLD per line. count is even.
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
void gemv4_i8u2(const uint32_t* w, const uint32_t* x, int count, int32_t* acc);

// The widest symmetric range for x such that K products of an int8 weight and an x value cannot
// overflow SMLAD's int32 sum: K * 127 * R <= 2^31 - 1, capped at int16.
static int x_range(int K) {
    const long r = 2147483647L / (127L * K);
    return r > 32767 ? 32767 : (int)r;
}

static inline int32_t round_s32(float v) {   // vcvtr: the FPSCR's mode, round to nearest (even)
    float r;
    __asm__("vcvtr.s32.f32 %0, %1" : "=t"(r) : "t"(v));
    int32_t i;
    memcpy(&i, &r, 4);
    return i;
}

// x -> (xe, xo) word pairs as SXTB16 unpacks a weight word: xe = (x0, x2), xo = (x1, x3). Returns sx.
static float quantize_x(const float* x, int K, uint32_t* x2) {
    float m = 0.0f;
    for (int k = 0; k < K; ++k) m = fmaxf(m, fabsf(x[k]));
    const int R = x_range(K);
    const float sx = m / R, inv = m > 0 ? R / m : 0.0f;
    for (int k = 0; k < K; k += 4) {
        const int32_t q0 = round_s32(x[k] * inv), q1 = round_s32(x[k + 1] * inv);
        const int32_t q2 = round_s32(x[k + 2] * inv), q3 = round_s32(x[k + 3] * inv);
        x2[k / 2]     = (uint16_t)q0 | ((uint32_t)(uint16_t)q2 << 16);
        x2[k / 2 + 1] = (uint16_t)q1 | ((uint32_t)(uint16_t)q3 << 16);
    }
    return sx;
}

static void gemv_i8(const uint32_t* Wp, const float* sw, const float* x, uint32_t* x2, float* y, int K, int N, int u2) {
    const float sx = quantize_x(x, K, x2);
    for (int n = 0; n < N; n += 4) {
        int32_t acc[4];
        if (u2) gemv4_i8u2(Wp + (size_t)n * (K / 4), x2, K / 4, acc);
        else    gemv4_i8(Wp + (size_t)n * (K / 4), x2, K / 4, acc);
        for (int j = 0; j < 4; ++j) y[n + j] = (float)acc[j] * (sx * sw[n + j]);
    }
}

static int cmp_d(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 7;
    {
        unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); f |= (1u << 24) | (1u << 25);   // RunFast
        __asm__ volatile("vmsr fpscr, %0" : : "r"(f));
    }
    // (K, N, what): every one-column weight mul_mat a Pi Zero model runs.
    static const struct { int K, N; const char* what; } shapes[] = {
        {288, 288, "moonshine attn q/k/v/o"},
        {288, 2304, "moonshine dec ffn up (swiglu)"},
        {1152, 288, "moonshine dec ffn down"},
        {288, 32768, "moonshine vocab"},
        {128, 512, "silero lstm"},
        {256, 256, "citrinet SE / generic"},
        {768, 768, "distilbert-sized"},
    };
    printf("%-30s %6s %6s | %9s %9s %9s | %6s %6s | %8s\n", "shape", "K", "N", "f32 ms", "i8 ms", "i8u2 ms",
           "x i8", "x u2", "rel err");
    double tf = 0, ti = 0, tu = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof *shapes; ++s) {
        const int K = shapes[s].K, N = shapes[s].N;
        const size_t f32b = (size_t)N * K * 4, i8b = (size_t)N * K;
        const int cf = (int)((8u << 20) / f32b) + 1, ci = (int)((8u << 20) / i8b) + 1;
        float* W = aligned_alloc(32, f32b * cf);
        uint32_t* Wp = aligned_alloc(32, i8b * ci);
        float* sw = aligned_alloc(32, N * 4);
        float *x = aligned_alloc(32, K * 4), *yf = aligned_alloc(32, N * 4), *yi = aligned_alloc(32, N * 4);
        uint32_t* x2 = aligned_alloc(32, K * 2);
        srand(7 + s);
        for (int k = 0; k < K; ++k) x[k] = (rand() / (float)RAND_MAX - 0.5f) * 2;
        for (size_t i = 0; i < (size_t)N * K; ++i) W[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        for (int c = 1; c < cf; ++c) memcpy(W + (size_t)c * N * K, W, f32b);
        int8_t* Wq = malloc(i8b);
        for (int n = 0; n < N; ++n) {
            float m = 0;
            for (int k = 0; k < K; ++k) m = fmaxf(m, fabsf(W[(size_t)n * K + k]));
            sw[n] = m / 127;
            for (int k = 0; k < K; ++k) Wq[(size_t)n * K + k] = (int8_t)lrintf(W[(size_t)n * K + k] / sw[n]);
        }
        for (int n = 0; n < N; n += 4)
            for (int kw = 0; kw < K / 4; ++kw)
                for (int j = 0; j < 4; ++j)
                    memcpy(Wp + ((size_t)n * (K / 4) + (size_t)kw * 4 + j), Wq + (size_t)(n + j) * K + kw * 4, 4);
        for (int c = 1; c < ci; ++c) memcpy((char*)Wp + (size_t)c * i8b, Wp, i8b);

        // Check: the asm accumulators against an int64 reference, then the result against F32.
        const float sx = quantize_x(x, K, x2);
        for (int n = 0; n < N; n += 4) {
            int32_t a1[4], a2[4];
            gemv4_i8(Wp + (size_t)n * (K / 4), x2, K / 4, a1);
            gemv4_i8u2(Wp + (size_t)n * (K / 4), x2, K / 4, a2);
            for (int j = 0; j < 4; ++j) {
                int64_t r = 0;
                for (int k = 0; k < K; ++k) {
                    const uint32_t w = x2[(k / 4) * 2 + (k & 1)];
                    const int16_t xq = (int16_t)((k & 2) ? (w >> 16) : (w & 0xffff));
                    r += (int64_t)Wq[(size_t)(n + j) * K + k] * xq;
                }
                if (r != a1[j] || r != a2[j]) { printf("%s: accumulator mismatch at row %d\n", shapes[s].what, n + j); return 1; }
            }
        }
        (void)sx;
        ggml_armv6_gemv_rows((const char*)W, K * 4, x, yf, K, 0, N);
        gemv_i8(Wp, sw, x, x2, yi, K, N, 0);
        double err = 0, ref = 0;
        for (int n = 0; n < N; ++n) { err = fmax(err, fabs(yi[n] - yf[n])); ref = fmax(ref, fabs(yf[n])); }
        if (!(err / ref < 2e-2)) { printf("%s: int8 error %.2e of peak\n", shapes[s].what, err / ref); return 1; }

        double bf[64], bi[64], bu[64];
        int jf = 0, ji = 0, ju = 0;
        for (int r = 0; r < reps && r < 64; ++r) {   // arms interleaved; each call on a cold copy
            double t0 = now();
            ggml_armv6_gemv_rows((const char*)(W + (size_t)(jf++ % cf) * N * K), K * 4, x, yf, K, 0, N);
            bf[r] = now() - t0;
            t0 = now();
            gemv_i8((const uint32_t*)((const char*)Wp + (size_t)(ji++ % ci) * i8b), sw, x, x2, yi, K, N, 0);
            bi[r] = now() - t0;
            t0 = now();
            gemv_i8((const uint32_t*)((const char*)Wp + (size_t)(ju++ % ci) * i8b), sw, x, x2, yi, K, N, 1);
            bu[r] = now() - t0;
        }
        const int nr = reps < 64 ? reps : 64;
        qsort(bf, nr, sizeof *bf, cmp_d); qsort(bi, nr, sizeof *bi, cmp_d); qsort(bu, nr, sizeof *bu, cmp_d);
        const double mf = bf[nr / 2], mi = bi[nr / 2], mu = bu[nr / 2];
        tf += mf; ti += mi; tu += mu;
        printf("%-30s %6d %6d | %9.3f %9.3f %9.3f | %5.2fx %5.2fx | %8.1e\n", shapes[s].what, K, N, mf * 1e3, mi * 1e3,
               mu * 1e3, mf / mi, mf / mu, err / ref);
        free(W); free(Wp); free(sw); free(x); free(yf); free(yi); free(x2); free(Wq);
    }
    printf("%-30s %13s | %9.3f %9.3f %9.3f | %5.2fx %5.2fx\n", "sum of medians", "", tf * 1e3, ti * 1e3, tu * 1e3, tf / ti, tf / tu);
    return 0;
}
