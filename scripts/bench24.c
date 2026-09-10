// P7: two follow-ups to bench23 -- does the unaligned `qs` cost anything, and what would hoisting
// the -8 bias out of the inner loop buy (a matmul can, since the activation block sum is one per
// column). Pi Zero W: alignment 1.01x/0.95x (nothing), bias hoist 1.14-1.16x. Epic-08 6.6.
//
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK 32
typedef struct { uint16_t d; uint8_t qs[QK/2]; } block_q4_0;              // qs at offset 2
typedef struct { uint16_t d; int8_t  qs[QK];   } block_q8_0;
typedef struct { uint16_t d; uint16_t pad; uint8_t qs[QK/2]; } blk4_al;   // qs at offset 4
typedef struct { uint16_t d; uint16_t pad; int8_t  qs[QK];   } blk8_al;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }

#define BODY(XQ, YQ, ACC)                                                              \
    for (int j = 0; j < QK/2; j += 4) {                                                \
        uint32_t xv, y0, y1;                                                           \
        memcpy(&xv, (XQ) + j, 4); memcpy(&y0, (YQ) + j, 4);                            \
        memcpy(&y1, (YQ) + QK/2 + j, 4);                                               \
        const uint32_t lo = xv & 0x0F0F0F0Fu, hi = (xv >> 4) & 0x0F0F0F0Fu, e = 0x00080008u; \
        ACC = __smlad(__ssub16(__uxtb16(lo),      e), __sxtb16(y0),      ACC);         \
        ACC = __smlad(__ssub16(__uxtb16(lo >> 8), e), __sxtb16(y0 >> 8), ACC);         \
        ACC = __smlad(__ssub16(__uxtb16(hi),      e), __sxtb16(y1),      ACC);         \
        ACC = __smlad(__ssub16(__uxtb16(hi >> 8), e), __sxtb16(y1 >> 8), ACC);         \
    }

// bias hoisted: sum(v*y) with v in 0..15, then correct by -8 * sum(y) once per block
#define BODY_NB(XQ, YQ, ACC)                                                           \
    for (int j = 0; j < QK/2; j += 4) {                                                \
        uint32_t xv, y0, y1;                                                           \
        memcpy(&xv, (XQ) + j, 4); memcpy(&y0, (YQ) + j, 4);                            \
        memcpy(&y1, (YQ) + QK/2 + j, 4);                                               \
        const uint32_t lo = xv & 0x0F0F0F0Fu, hi = (xv >> 4) & 0x0F0F0F0Fu;            \
        ACC = __smlad(__uxtb16(lo),      __sxtb16(y0),      ACC);                      \
        ACC = __smlad(__uxtb16(lo >> 8), __sxtb16(y0 >> 8), ACC);                      \
        ACC = __smlad(__uxtb16(hi),      __sxtb16(y1),      ACC);                      \
        ACC = __smlad(__uxtb16(hi >> 8), __sxtb16(y1 >> 8), ACC);                      \
    }

static int32_t dot_un(int nb, const block_q4_0 *x, const block_q8_0 *y) {
    int32_t t = 0; for (int i = 0; i < nb; i++) { int32_t a = 0; BODY(x[i].qs, y[i].qs, a) t += a; } return t; }
static int32_t dot_al(int nb, const blk4_al *x, const blk8_al *y) {
    int32_t t = 0; for (int i = 0; i < nb; i++) { int32_t a = 0; BODY(x[i].qs, y[i].qs, a) t += a; } return t; }
static int32_t dot_nb(int nb, const block_q4_0 *x, const block_q8_0 *y, const int32_t *ysum) {
    int32_t t = 0; for (int i = 0; i < nb; i++) { int32_t a = 0; BODY_NB(x[i].qs, y[i].qs, a) t += a - 8*ysum[i]; } return t; }

int main(void) {
    const int K = 768, nb = K / QK; const long reps = 40000;
    block_q4_0 *x = malloc(sizeof *x * nb); block_q8_0 *y = malloc(sizeof *y * nb);
    blk4_al *xa = malloc(sizeof *xa * nb);  blk8_al *ya = malloc(sizeof *ya * nb);
    int32_t *ysum = malloc(sizeof *ysum * nb);
    srand(1234);
    for (int i = 0; i < nb; i++) {
        int s = 0;
        for (int j = 0; j < QK/2; j++) xa[i].qs[j] = x[i].qs[j] = rand() & 0xFF;
        for (int j = 0; j < QK;   j++) { ya[i].qs[j] = y[i].qs[j] = (int8_t)(rand() & 0xFF); s += y[i].qs[j]; }
        ysum[i] = s;
    }
    const int32_t ref = dot_un(nb, x, y);
    printf("aligned  matches: %d    bias-hoisted matches: %d\n",
           dot_al(nb, xa, ya) == ref, dot_nb(nb, x, y, ysum) == ref);

    volatile int32_t sink = 0; double t;
    t = now(); for (long r = 0; r < reps; r++) sink += dot_un(nb, x, y);        const double tu = now()-t;
    t = now(); for (long r = 0; r < reps; r++) sink += dot_al(nb, xa, ya);      const double ta = now()-t;
    t = now(); for (long r = 0; r < reps; r++) sink += dot_nb(nb, x, y, ysum);  const double tn = now()-t;
    const double m = (double)K * reps;
    printf("smlad, ggml layout (qs unaligned)   %6.1f MMAC/s\n", m/tu/1e6);
    printf("smlad, qs 4-byte aligned            %6.1f MMAC/s   %.2fx\n", m/ta/1e6, tu/ta);
    printf("smlad, -8 bias hoisted per block    %6.1f MMAC/s   %.2fx\n", m/tn/1e6, tu/tn);
    (void)sink; return 0;
}
