// P7: ggml_vec_dot_q4_0_q8_0's generic arm against an ARMv6 SIMD32 rewrite.
//
// NOT part of the build -- a standalone measurement, and the one that says whether the __smlad
// idea in Epic-08 6.6 is worth building. Build ON THE TARGET (an ARM1176; no NEON):
//   gcc -O2 -o bench23 bench23.c && ./bench23
// Measured on a Pi Zero W: 101.8 -> 149.6 MMAC/s at K=768, and the results compare `==`.
//
// The block layout is ggml's exactly, INCLUDING the 2-byte fp16 scale that puts `qs` at offset 2 and
// so makes every word load of it unaligned -- which bench24 then shows costs nothing here.
//
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK 32
typedef struct { uint16_t d; uint8_t qs[QK / 2]; } block_q4_0;   // 18 bytes
typedef struct { uint16_t d; int8_t  qs[QK];     } block_q8_0;   // 34 bytes

static float half_to_float(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, bits;
    if (e == 0)       bits = m ? (s | ((127 - 15 + 1) << 23) | (m << 13)) : s;  // approx; unused range
    else if (e == 31) bits = s | 0x7F800000u | (m << 13);
    else              bits = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

// ---- as shipped (src/ggml-cpu/quants.c, ggml_vec_dot_q4_0_q8_0_generic) -------------------------
static float dot_generic(int nb, const block_q4_0 *x, const block_q8_0 *y) {
    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        int sumi0 = 0, sumi1 = 0;
        for (int j = 0; j < QK / 2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;
            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + QK / 2]);
        }
        sumf += (sumi0 + sumi1) * half_to_float(x[ib].d) * half_to_float(y[ib].d);
    }
    return sumf;
}

// ---- ARMv6 SIMD32: __uxtb16/__sxtb16 to unpack, __smlad for two MACs per instruction ------------
// Eight products per 4 bytes of `x`: four low nibbles against y[j..j+3], four high against
// y[j+16..j+19] -- the same set of products the loop above forms, so the integer sum is identical.
static float dot_smlad(int nb, const block_q4_0 *x, const block_q8_0 *y) {
    float sumf = 0;
    for (int ib = 0; ib < nb; ++ib) {
        const uint8_t *xq = x[ib].qs;
        const int8_t  *yq = y[ib].qs;
        int32_t acc = 0;
        for (int j = 0; j < QK / 2; j += 4) {
            uint32_t xv, y0, y1;
            memcpy(&xv, xq + j, 4);            // unaligned by construction (qs is at offset 2)
            memcpy(&y0, yq + j, 4);
            memcpy(&y1, yq + QK / 2 + j, 4);

            const uint32_t lo = xv & 0x0F0F0F0Fu, hi = (xv >> 4) & 0x0F0F0F0Fu;
            const uint32_t eight = 0x00080008u;
            acc = __smlad(__ssub16(__uxtb16(lo),      eight), __sxtb16(y0),      acc);
            acc = __smlad(__ssub16(__uxtb16(lo >> 8), eight), __sxtb16(y0 >> 8), acc);
            acc = __smlad(__ssub16(__uxtb16(hi),      eight), __sxtb16(y1),      acc);
            acc = __smlad(__ssub16(__uxtb16(hi >> 8), eight), __sxtb16(y1 >> 8), acc);
        }
        sumf += acc * half_to_float(x[ib].d) * half_to_float(y[ib].d);
    }
    return sumf;
}

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }

static void run(int K, long reps) {
    const int nb = K / QK;
    block_q4_0 *x = malloc(sizeof(block_q4_0) * nb);
    block_q8_0 *y = malloc(sizeof(block_q8_0) * nb);
    srand(1234);
    for (int i = 0; i < nb; i++) {
        x[i].d = 0x3400; y[i].d = 0x3000;                       // 0.25, 0.125
        for (int j = 0; j < QK/2; j++) x[i].qs[j] = rand() & 0xFF;
        for (int j = 0; j < QK;   j++) y[i].qs[j] = (int8_t)(rand() & 0xFF);
    }
    const float a = dot_generic(nb, x, y), b = dot_smlad(nb, x, y);
    if (a != b) { printf("K=%-5d MISMATCH generic=%.6f smlad=%.6f\n", K, a, b); free(x); free(y); return; }

    volatile float sink = 0;
    double t = now(); for (long r = 0; r < reps; r++) sink += dot_generic(nb, x, y); double tg = now() - t;
    t = now();        for (long r = 0; r < reps; r++) sink += dot_smlad  (nb, x, y); double ts = now() - t;
    const double macs = (double)K * reps;
    printf("K=%-5d generic %6.1f MMAC/s   smlad %6.1f MMAC/s   %.2fx   (identical: %.4f)\n",
           K, macs/tg/1e6, macs/ts/1e6, tg/ts, a);
    (void)sink; free(x); free(y);
}

int main(void) { run(768, 40000); run(3072, 10000); run(64, 400000); return 0; }
