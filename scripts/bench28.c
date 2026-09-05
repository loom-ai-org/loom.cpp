// P7: a DEDICATED ARMv6 q4_0 x q8_0 GEMM, tiled, against the one-element-at-a-time vec_dot.
//
// bench23 showed the dot product alone reaches 149.6 MMAC/s; bench27 showed it delivers only 78.3
// inside a GEMM. The difference is that vec_dot recomputes the nibble unpack -- mask, shift, 4x
// uxtb16, 4x ssub16, about ten instructions per eight MACs -- for EVERY output column, and re-reads
// the weight row with it. Tiling N columns against one weight row amortises both over N.
//
// Instruction budget per 4 weight bytes (32 MACs at 1xN), counting the unpacks:
//   1x1  10 (w) + 6 (y) +  4 smlad =  20 for  8 MACs   2.50 instr/MAC
//   1x2  10     + 12    +  8       =  30 for 16 MACs   1.88
//   1x4  10     + 24    + 16       =  50 for 32 MACs   1.56
// Shape is bench27's: IC*K=960, OC=384, OL=275.  gcc -O2 -o bench28 bench28.c && ./bench28
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK 32
#define KDIM 960
#define OC   384
#define OL   276          /* 275 rounded to a multiple of 4 so every arm does identical work */
#define NB   (KDIM / QK)

typedef struct { uint16_t d; uint8_t qs[QK/2]; } blk4;
typedef struct { uint16_t d; int8_t  qs[QK];   } blk8;
static float h2f(uint16_t h){ uint32_t s=(uint32_t)(h&0x8000u)<<16, em=h&0x7FFFu;
    uint32_t b = em ? (s|((em+0x1C000u)<<13)) : s; float f; memcpy(&f,&b,4); return f; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

#define UNPACK_W(XQ, J)                                                        \
    uint32_t xv; memcpy(&xv, (XQ)+(J), 4);                                     \
    const uint32_t lo = xv & 0x0F0F0F0Fu, hi = (xv>>4) & 0x0F0F0F0Fu, e = 0x00080008u; \
    const uint32_t L0 = __ssub16(__uxtb16(lo),    e), L1 = __ssub16(__uxtb16(lo>>8), e); \
    const uint32_t H0 = __ssub16(__uxtb16(hi),    e), H1 = __ssub16(__uxtb16(hi>>8), e);
#define COL(Q, J, A) { uint32_t p, q; memcpy(&p,(Q)+(J),4); memcpy(&q,(Q)+QK/2+(J),4);   \
    A = __smlad(L0, __sxtb16(p),    A); A = __smlad(L1, __sxtb16(p>>8), A);              \
    A = __smlad(H0, __sxtb16(q),    A); A = __smlad(H1, __sxtb16(q>>8), A); }

static void tile1(const blk4 *w, const blk8 *const *y, float *out) {
    float s0 = 0;
    for (int ib = 0; ib < NB; ib++) { int32_t a0 = 0;
        for (int j = 0; j < QK/2; j += 4) { UNPACK_W(w[ib].qs, j) COL(y[0][ib].qs, j, a0) }
        s0 += a0 * h2f(w[ib].d) * h2f(y[0][ib].d); }
    out[0] = s0;
}
static void tile2(const blk4 *w, const blk8 *const *y, float *out) {
    float s0 = 0, s1 = 0;
    for (int ib = 0; ib < NB; ib++) { int32_t a0 = 0, a1 = 0;
        for (int j = 0; j < QK/2; j += 4) { UNPACK_W(w[ib].qs, j) COL(y[0][ib].qs,j,a0) COL(y[1][ib].qs,j,a1) }
        const float dw = h2f(w[ib].d);
        s0 += a0*dw*h2f(y[0][ib].d); s1 += a1*dw*h2f(y[1][ib].d); }
    out[0] = s0; out[1] = s1;
}
static void tile4(const blk4 *w, const blk8 *const *y, float *out) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int ib = 0; ib < NB; ib++) { int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int j = 0; j < QK/2; j += 4) { UNPACK_W(w[ib].qs, j)
            COL(y[0][ib].qs,j,a0) COL(y[1][ib].qs,j,a1) COL(y[2][ib].qs,j,a2) COL(y[3][ib].qs,j,a3) }
        const float dw = h2f(w[ib].d);
        s0 += a0*dw*h2f(y[0][ib].d); s1 += a1*dw*h2f(y[1][ib].d);
        s2 += a2*dw*h2f(y[2][ib].d); s3 += a3*dw*h2f(y[3][ib].d); }
    out[0]=s0; out[1]=s1; out[2]=s2; out[3]=s3;
}

static blk4 *W; static blk8 *X; static float *C;
static const blk8 *cols[4];

#define GEMM(NAME, TILE, N)                                                         \
static void NAME(void) {                                                            \
    for (int oc = 0; oc < OC; oc++) {                                               \
        const blk4 *w = W + (size_t)oc*NB;                                          \
        for (int ol = 0; ol < OL; ol += (N)) {                                      \
            for (int c = 0; c < (N); c++) cols[c] = X + (size_t)(ol+c)*NB;          \
            TILE(w, cols, C + (size_t)oc*OL + ol);                                  \
        }                                                                           \
    }                                                                               \
}
GEMM(gemm1, tile1, 1) GEMM(gemm2, tile2, 2) GEMM(gemm4, tile4, 4)

int main(void) {
    W = malloc(sizeof(blk4)*(size_t)NB*OC); X = malloc(sizeof(blk8)*(size_t)NB*OL);
    C = malloc(sizeof(float)*(size_t)OC*OL);
    srand(11);
    for (size_t i = 0; i < (size_t)NB*OC; i++) { W[i].d = 0x3000; for (int j=0;j<QK/2;j++) W[i].qs[j]=rand()&0xFF; }
    for (size_t i = 0; i < (size_t)NB*OL; i++) { X[i].d = 0x3400; for (int j=0;j<QK;j++) X[i].qs[j]=(int8_t)(rand()&0xFF); }

    gemm1(); float *ref = malloc(sizeof(float)*(size_t)OC*OL); memcpy(ref, C, sizeof(float)*(size_t)OC*OL);
    memset(C, 0, sizeof(float)*(size_t)OC*OL); gemm2();
    int ok2 = !memcmp(ref, C, sizeof(float)*(size_t)OC*OL);
    memset(C, 0, sizeof(float)*(size_t)OC*OL); gemm4();
    int ok4 = !memcmp(ref, C, sizeof(float)*(size_t)OC*OL);
    printf("identical to 1x1:  1x2 %s   1x4 %s\n\n", ok2?"yes":"NO", ok4?"yes":"NO");

    const double macs = (double)OC*OL*KDIM; const int reps = 3; double t;
    t = now(); for (int r=0;r<reps;r++) gemm1(); const double t1 = now()-t;
    t = now(); for (int r=0;r<reps;r++) gemm2(); const double t2 = now()-t;
    t = now(); for (int r=0;r<reps;r++) gemm4(); const double t4 = now()-t;
    printf("1x1  vec_dot shape   %6.1f MMAC/s   %5.2f s/call\n", macs*reps/t1/1e6, t1/reps);
    printf("1x2  tiled           %6.1f MMAC/s   %5.2f s/call   %.2fx\n", macs*reps/t2/1e6, t2/reps, t1/t2);
    printf("1x4  tiled           %6.1f MMAC/s   %5.2f s/call   %.2fx\n", macs*reps/t4/1e6, t4/reps, t1/t4);
    printf("\n(bench23's isolated dot reached 149.6; bench27's F32 GEMM at this shape, 29.7)\n");
    return 0;
}
