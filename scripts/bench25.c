// P7: ggml's fp16->f32 lookup table (65536 floats, 256 KB) against six instructions of bit
// arithmetic, on a board with 16 KB of L1 D-cache. Shaped as a GEMV so the scale reads scatter
// like a real weight tensor's. Pi Zero W: 1.17-1.19x, bit-identical. Epic-08 6.6.
//
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK 32
#define K   768
#define NB  (K / QK)
#define M   64
typedef struct { uint16_t d; uint8_t qs[QK/2]; } block_q4_0;
typedef struct { uint16_t d; int8_t  qs[QK];   } block_q8_0;

static float *table;                                    // ggml_table_f32_f16, 256 KB
static inline float via_table(uint16_t h) { return table[h]; }

// Branch-free, no memory. Handles normals; zero/denormal fall out as ~0, which is what a scale is not.
static inline float via_math(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t em = h & 0x7FFFu;
    uint32_t bits = s | ((em + 0x1C000u) << 13);        // exponent rebias 15 -> 127
    if (em == 0) bits = s;
    float f; memcpy(&f, &bits, 4); return f;
}

#define DOTBODY(XQ, YQ, ACC)                                                           \
    for (int j = 0; j < QK/2; j += 4) {                                                \
        uint32_t xv, y0, y1;                                                           \
        memcpy(&xv, (XQ)+j, 4); memcpy(&y0, (YQ)+j, 4); memcpy(&y1, (YQ)+QK/2+j, 4);   \
        const uint32_t lo = xv & 0x0F0F0F0Fu, hi = (xv>>4) & 0x0F0F0F0Fu, e = 0x00080008u; \
        ACC = __smlad(__ssub16(__uxtb16(lo),      e), __sxtb16(y0),      ACC);         \
        ACC = __smlad(__ssub16(__uxtb16(lo >> 8), e), __sxtb16(y0 >> 8), ACC);         \
        ACC = __smlad(__ssub16(__uxtb16(hi),      e), __sxtb16(y1),      ACC);         \
        ACC = __smlad(__ssub16(__uxtb16(hi >> 8), e), __sxtb16(y1 >> 8), ACC);         \
    }

#define GEMV(NAME, CVT)                                                                \
static float NAME(const block_q4_0 *x, const block_q8_0 *y) {                          \
    float t = 0;                                                                       \
    for (int r = 0; r < M; r++) {                                                      \
        const block_q4_0 *xr = x + (size_t)r * NB; float sumf = 0;                     \
        for (int i = 0; i < NB; i++) {                                                 \
            int32_t a = 0; DOTBODY(xr[i].qs, y[i].qs, a)                               \
            sumf += a * CVT(xr[i].d) * CVT(y[i].d);                                    \
        }                                                                              \
        t += sumf;                                                                     \
    }                                                                                  \
    return t;                                                                          \
}
GEMV(gemv_table, via_table)
GEMV(gemv_math,  via_math)

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }

int main(void) {
    table = malloc(65536 * sizeof(float));
    for (int i = 0; i < 65536; i++) table[i] = via_math((uint16_t)i);

    block_q4_0 *x = malloc(sizeof *x * (size_t)M * NB);
    block_q8_0 *y = malloc(sizeof *y * NB);
    srand(7);
    for (size_t i = 0; i < (size_t)M * NB; i++) {
        x[i].d = 0x3000 + (rand() & 0x0FFF);            // scattered, ~4096 distinct scales
        for (int j = 0; j < QK/2; j++) x[i].qs[j] = rand() & 0xFF;
    }
    for (int i = 0; i < NB; i++) {
        y[i].d = 0x3400 + (rand() & 0x00FF);
        for (int j = 0; j < QK; j++) y[i].qs[j] = (int8_t)(rand() & 0xFF);
    }
    const float a = gemv_table(x, y), b = gemv_math(x, y);
    printf("agree: %d  (%.4f vs %.4f)\n", a == b, a, b);

    const long reps = 3000; volatile float sink = 0; double t;
    t = now(); for (long r = 0; r < reps; r++) sink += gemv_table(x, y); const double tt = now()-t;
    t = now(); for (long r = 0; r < reps; r++) sink += gemv_math (x, y); const double tm = now()-t;
    const double macs = (double)M * K * reps;
    printf("fp16 via 256 KB table  %6.1f MMAC/s\n", macs/tt/1e6);
    printf("fp16 via arithmetic    %6.1f MMAC/s   %.2fx\n", macs/tm/1e6, tt/tm);
    (void)sink; return 0;
}
