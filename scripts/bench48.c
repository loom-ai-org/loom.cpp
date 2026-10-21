// ADR-077 gate 1's cross-model analysis: THE GEMV CLASS ON THE ARM1176. A weight GEMM with one column
// (a decoder's step, an LSTM's step) reads every weight once per call, so it is bound by DRAM, not by
// the MACs -- Moonshine tiny's vocabulary projection (288 x 32768, F32) is 0.415 s per token in ggml,
// 91 MB/s. This measures what the board can stream and what an F32 and an int8 GEMV reach against it,
// on that shape, both in C at the same effort (4 rows per pass, RunFast for the VFP).
//
//   ./bench48 [K] [N]     default 288 32768
//
//   read     ldmia of 8 words per 32 B line over a 64 MB buffer: the stream ceiling
//   read+pld the same with a PLD 4 lines ahead (the ARM1176 has no hardware prefetcher)
//   f32      y[n] = sum_k W[n][k] x[k], W F32 [N][K]
//   i8       the same with W int8 per-channel [N][K] (SXTB16 unpack, SMLAD), x int16 pairs, F32 out
//   i8asm    the same arithmetic in asm (W repacked by channel quad, PLD), checked equal to i8
//
//   gcc -O2 -march=armv6+fp -o bench48 bench48.c        (loom-armv6-build:bookworm)
#include <arm_acle.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static uint32_t read_stream(const uint32_t* p, size_t words, int pld) {
    uint32_t s = 0;
    const uint32_t* end = p + words;
    if (pld)
        __asm__ volatile("1:\n pld [%1, #128]\n ldmia %1!, {r4-r11}\n eor %0, %0, r4\n cmp %1, %2\n blo 1b\n"
                         : "+r"(s), "+r"(p) : "r"(end) : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "cc", "memory");
    else
        __asm__ volatile("1:\n ldmia %1!, {r4-r11}\n eor %0, %0, r4\n cmp %1, %2\n blo 1b\n"
                         : "+r"(s), "+r"(p) : "r"(end) : "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "cc", "memory");
    return s;
}

static void gemv_f32(const float* W, const float* x, float* y, int K, int N) {
    for (int n = 0; n < N; n += 4) {
        const float *w0 = W + (size_t)n * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int k = 0; k < K; ++k) {
            const float v = x[k];
            a0 += w0[k] * v; a1 += w1[k] * v; a2 += w2[k] * v; a3 += w3[k] * v;
        }
        y[n] = a0; y[n + 1] = a1; y[n + 2] = a2; y[n + 3] = a3;
    }
}

// W int8 [N][K], read as words of four; x as int16 pairs in the order SXTB16 unpacks a word:
// (b0, b2) then (b1, b3), so xe[k/2] holds x[k], x[k+2] and xo[k/2] holds x[k+1], x[k+3] per 4-group.
static void gemv_i8(const int8_t* W, const uint32_t* xe, const uint32_t* xo, const float* sw, float sx,
                    float* y, int K, int N) {
    const int KW = K / 4;
    for (int n = 0; n < N; n += 4) {
        const uint32_t *w0 = (const uint32_t*)(W + (size_t)n * K), *w1 = w0 + KW, *w2 = w1 + KW, *w3 = w2 + KW;
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int k = 0; k < KW; ++k) {
            const uint32_t e = xe[k], o = xo[k];
            uint32_t w;
            w = w0[k]; a0 = __smlad(__sxtb16(w), e, a0); a0 = __smlad(__sxtb16((w >> 8) | (w << 24)), o, a0);
            w = w1[k]; a1 = __smlad(__sxtb16(w), e, a1); a1 = __smlad(__sxtb16((w >> 8) | (w << 24)), o, a1);
            w = w2[k]; a2 = __smlad(__sxtb16(w), e, a2); a2 = __smlad(__sxtb16((w >> 8) | (w << 24)), o, a2);
            w = w3[k]; a3 = __smlad(__sxtb16(w), e, a3); a3 = __smlad(__sxtb16((w >> 8) | (w << 24)), o, a3);
        }
        y[n] = a0 * sx * sw[n]; y[n + 1] = a1 * sx * sw[n + 1]; y[n + 2] = a2 * sx * sw[n + 2]; y[n + 3] = a3 * sx * sw[n + 3];
    }
}

// The same int8 GEMV in asm, the int8 arm's equal effort: W repacked so a channel quad's four words
// at one k-word sit together ([N/4][K/4][4] words), one ldmia per quad per four k, a PLD per 32 B line,
// and SXTB16's ROR #8 operand instead of a separate rotate. x interleaved as (xe, xo) word pairs.
// gemv4_i8(w, x, count, acc): count k-words of one channel quad; acc[4] out. Plain asm because the
// loop needs every register: r0 = w, r1 = x, r2 = count, r3 = scratch (acc's pointer is on the stack),
// r4-r7 the accumulators, r8/r9 = xe/xo, r10-r12/lr the quad's four words.
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

// The same int8 GEMV in asm, the int8 arm's equal effort: W repacked so a channel quad's four words
// at one k-word sit together ([N/4][K/4][4] words), one ldmia per quad per four k, a PLD per 32 B line,
// and SXTB16's ROR #8 operand instead of a separate rotate. x interleaved as (xe, xo) word pairs.
static void gemv_i8_asm(const uint32_t* Wp, const uint32_t* x2, const float* sw, float sx, float* y, int K, int N) {
    for (int n = 0; n < N; n += 4) {
        int32_t acc[4];
        gemv4_i8(Wp + (size_t)n * (K / 4), x2, K / 4, acc);
        for (int j = 0; j < 4; ++j) y[n + j] = acc[j] * sx * sw[n + j];
    }
}

int main(int argc, char** argv) {
    const int K = argc > 1 ? atoi(argv[1]) : 288, N = argc > 2 ? atoi(argv[2]) : 32768;
    {
        unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); f |= (1u << 24) | (1u << 25);   // RunFast
        __asm__ volatile("vmsr fpscr, %0" : : "r"(f));
    }
    const size_t big = 64u << 20;
    uint32_t* buf = malloc(big);
    for (size_t i = 0; i < big / 4; ++i) buf[i] = (uint32_t)i * 2654435761u;
    for (int pld = 0; pld < 2; ++pld) {
        double best = 1e9; uint32_t s = 0;
        for (int r = 0; r < 3; ++r) { double t0 = now(); s += read_stream(buf, big / 4, pld); best = fmin(best, now() - t0); }
        printf("read%s       %7.1f MB/s  [%u]\n", pld ? "+pld" : "    ", big / best / 1e6, s);
    }
    free(buf);

    float* W = malloc((size_t)N * K * 4);
    int8_t* Wq = malloc((size_t)N * K);
    float *x = malloc(K * 4), *y = malloc(N * 4), *yq = malloc(N * 4), *sw = malloc(N * 4);
    uint32_t *xe = malloc(K / 4 * 4), *xo = malloc(K / 4 * 4);
    srand(7);
    for (int k = 0; k < K; ++k) x[k] = (rand() / (float)RAND_MAX - 0.5f) * 2;
    for (size_t i = 0; i < (size_t)N * K; ++i) W[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    for (int n = 0; n < N; ++n) {
        float m = 0;
        for (int k = 0; k < K; ++k) m = fmaxf(m, fabsf(W[(size_t)n * K + k]));
        sw[n] = m / 127;
        for (int k = 0; k < K; ++k) Wq[(size_t)n * K + k] = (int8_t)lrintf(W[(size_t)n * K + k] / sw[n]);
    }
    float mx = 0;
    for (int k = 0; k < K; ++k) mx = fmaxf(mx, fabsf(x[k]));
    const float sx = mx / 32767;   // x in the int16 range: SMLAD has the headroom at these K
    for (int k = 0; k < K; k += 4) {
        const int16_t q0 = lrintf(x[k] / sx), q1 = lrintf(x[k + 1] / sx), q2 = lrintf(x[k + 2] / sx), q3 = lrintf(x[k + 3] / sx);
        xe[k / 4] = (uint16_t)q0 | ((uint32_t)(uint16_t)q2 << 16);
        xo[k / 4] = (uint16_t)q1 | ((uint32_t)(uint16_t)q3 << 16);
    }
    uint32_t* Wp = malloc((size_t)N * K);
    uint32_t* x2 = malloc(K / 2 * 4);
    for (int n = 0; n < N; n += 4)
        for (int kw = 0; kw < K / 4; ++kw)
            for (int j = 0; j < 4; ++j) memcpy(Wp + ((size_t)n * (K / 4) + (size_t)kw * 4 + j), Wq + (size_t)(n + j) * K + kw * 4, 4);
    for (int kw = 0; kw < K / 4; ++kw) { x2[2 * kw] = xe[kw]; x2[2 * kw + 1] = xo[kw]; }
    float* ya = malloc(N * 4);
    const double mb32 = (double)N * K * 4 / 1e6, mb8 = (double)N * K / 1e6;
    double bf = 1e9, bq = 1e9, ba = 1e9;
    for (int r = 0; r < 5; ++r) {   // interleaved, best of 5
        double t0 = now(); gemv_f32(W, x, y, K, N); bf = fmin(bf, now() - t0);
        t0 = now(); gemv_i8(Wq, xe, xo, sw, sx, yq, K, N); bq = fmin(bq, now() - t0);
        t0 = now(); gemv_i8_asm(Wp, x2, sw, sx, ya, K, N); ba = fmin(ba, now() - t0);
    }
    if (memcmp(ya, yq, N * 4)) { printf("i8 asm differs from i8 C\n"); return 1; }
    double err = 0, ref = 0;
    for (int n = 0; n < N; ++n) { err = fmax(err, fabs(yq[n] - y[n])); ref = fmax(ref, fabs(y[n])); }
    printf("K=%d N=%d\n", K, N);
    printf("f32  %8.1f ms  %6.1f MB/s  %.3f GOP/s\n", bf * 1e3, mb32 / bf, 2.0 * N * K / bf / 1e9);
    printf("i8   %8.1f ms  %6.1f MB/s  %.3f GOP/s  (%.2fx f32; max |err| / max |y| %.1e)\n", bq * 1e3, mb8 / bq,
           2.0 * N * K / bq / 1e9, bf / bq, err / ref);
    printf("i8asm%8.1f ms  %6.1f MB/s  %.3f GOP/s  (%.2fx f32; identical to i8)\n", ba * 1e3, mb8 / ba,
           2.0 * N * K / ba / 1e9, bf / ba);
    return 0;
}
