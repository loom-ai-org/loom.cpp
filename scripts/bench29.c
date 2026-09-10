// P7: the loop order and the tile shape for a dedicated ARMv6 q4_0 x q8_0 GEMM.
//
// bench28 tiled COLUMNS and got 1.20x, then plateaued -- because in `for oc { for ol { dot } }` the
// weight row is already L1-resident and it is the ACTIVATIONS that stream: 384 sweeps of 281 KB =
// 108 MB per call. The other order sweeps 207 KB of weights per column instead, 57 MB. And a 2-D tile
// amortises BOTH unpacks: per 4 weight bytes the nibble unpack is ~10 instructions and the activation
// unpack ~6, so RxC pays 10R + 6C for 32*R*C/4 MACs.
//
//   1x1  20 instr /  8 MACs  2.50/MAC      2x2  48 / 32  1.50/MAC
//   1x4  50 instr / 32 MACs  1.56/MAC      2x4  76 / 64  1.19/MAC
//
// Shape is bench27/28's: IC*K=960, OC=384, OL=276.  gcc -O2 -o bench29 bench29.c && ./bench29
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK 32
#define KDIM 960
#define OC   384
#define OL   276
#define NB   (KDIM / QK)
typedef struct { uint16_t d; uint8_t qs[QK/2]; } blk4;
typedef struct { uint16_t d; int8_t  qs[QK];   } blk8;
static float h2f(uint16_t h){ uint32_t s=(uint32_t)(h&0x8000u)<<16, em=h&0x7FFFu;
    uint32_t b = em ? (s|((em+0x1C000u)<<13)) : s; float f; memcpy(&f,&b,4); return f; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

#define WUNPACK(P, J, L0,L1,H0,H1)                                              \
    uint32_t xv##L0; memcpy(&xv##L0, (P)+(J), 4);                               \
    const uint32_t lo##L0 = xv##L0 & 0x0F0F0F0Fu, hi##L0 = (xv##L0>>4) & 0x0F0F0F0Fu; \
    const uint32_t L0 = __ssub16(__uxtb16(lo##L0),    0x00080008u),             \
                   L1 = __ssub16(__uxtb16(lo##L0>>8), 0x00080008u),             \
                   H0 = __ssub16(__uxtb16(hi##L0),    0x00080008u),             \
                   H1 = __ssub16(__uxtb16(hi##L0>>8), 0x00080008u);
#define YUNPACK(P, J, A0,A1,B0,B1)                                              \
    uint32_t p##A0, q##A0; memcpy(&p##A0,(P)+(J),4); memcpy(&q##A0,(P)+QK/2+(J),4); \
    const uint32_t A0 = __sxtb16(p##A0), A1 = __sxtb16(p##A0>>8),               \
                   B0 = __sxtb16(q##A0), B1 = __sxtb16(q##A0>>8);
#define MAC4(L0,L1,H0,H1, A0,A1,B0,B1, ACC)                                     \
    ACC = __smlad(L0,A0,ACC); ACC = __smlad(L1,A1,ACC);                         \
    ACC = __smlad(H0,B0,ACC); ACC = __smlad(H1,B1,ACC);

static blk4 *W; static blk8 *X; static float *C;

// order B, no tile: activation column outer, weight rows inner
static void gemmB11(void) {
    for (int ol = 0; ol < OL; ol++) { const blk8 *y = X + (size_t)ol*NB;
        for (int oc = 0; oc < OC; oc++) { const blk4 *w = W + (size_t)oc*NB; float s = 0;
            for (int ib = 0; ib < NB; ib++) { int32_t a = 0;
                for (int j = 0; j < QK/2; j += 4) {
                    WUNPACK(w[ib].qs, j, L0,L1,H0,H1) YUNPACK(y[ib].qs, j, A0,A1,B0,B1)
                    MAC4(L0,L1,H0,H1, A0,A1,B0,B1, a) }
                s += a * h2f(w[ib].d) * h2f(y[ib].d); }
            C[(size_t)oc*OL+ol] = s; } }
}
// order B, 2 weight rows x 2 activation columns
static void gemmB22(void) {
    for (int ol = 0; ol < OL; ol += 2) {
        const blk8 *y0 = X + (size_t)ol*NB, *y1 = X + (size_t)(ol+1)*NB;
        for (int oc = 0; oc < OC; oc += 2) {
            const blk4 *w0 = W + (size_t)oc*NB, *w1 = W + (size_t)(oc+1)*NB;
            float s00=0,s01=0,s10=0,s11=0;
            for (int ib = 0; ib < NB; ib++) {
                int32_t a00=0,a01=0,a10=0,a11=0;
                for (int j = 0; j < QK/2; j += 4) {
                    WUNPACK(w0[ib].qs, j, P0,P1,P2,P3) WUNPACK(w1[ib].qs, j, Q0,Q1,Q2,Q3)
                    YUNPACK(y0[ib].qs, j, C0,C1,C2,C3) YUNPACK(y1[ib].qs, j, D0,D1,D2,D3)
                    MAC4(P0,P1,P2,P3, C0,C1,C2,C3, a00) MAC4(P0,P1,P2,P3, D0,D1,D2,D3, a01)
                    MAC4(Q0,Q1,Q2,Q3, C0,C1,C2,C3, a10) MAC4(Q0,Q1,Q2,Q3, D0,D1,D2,D3, a11) }
                const float dw0 = h2f(w0[ib].d), dw1 = h2f(w1[ib].d);
                const float dy0 = h2f(y0[ib].d), dy1 = h2f(y1[ib].d);
                s00 += a00*dw0*dy0; s01 += a01*dw0*dy1; s10 += a10*dw1*dy0; s11 += a11*dw1*dy1; }
            C[(size_t)oc*OL+ol] = s00; C[(size_t)oc*OL+ol+1] = s01;
            C[(size_t)(oc+1)*OL+ol] = s10; C[(size_t)(oc+1)*OL+ol+1] = s11; } }
}
// order A control (bench28's best): weight row outer, 4 columns tiled
static void gemmA14(void) {
    for (int oc = 0; oc < OC; oc++) { const blk4 *w = W + (size_t)oc*NB;
        for (int ol = 0; ol < OL; ol += 4) {
            const blk8 *y0=X+(size_t)ol*NB,*y1=X+(size_t)(ol+1)*NB,*y2=X+(size_t)(ol+2)*NB,*y3=X+(size_t)(ol+3)*NB;
            float s0=0,s1=0,s2=0,s3=0;
            for (int ib = 0; ib < NB; ib++) { int32_t a0=0,a1=0,a2=0,a3=0;
                for (int j = 0; j < QK/2; j += 4) {
                    WUNPACK(w[ib].qs, j, L0,L1,H0,H1)
                    { YUNPACK(y0[ib].qs,j,A0,A1,B0,B1) MAC4(L0,L1,H0,H1,A0,A1,B0,B1,a0) }
                    { YUNPACK(y1[ib].qs,j,A0,A1,B0,B1) MAC4(L0,L1,H0,H1,A0,A1,B0,B1,a1) }
                    { YUNPACK(y2[ib].qs,j,A0,A1,B0,B1) MAC4(L0,L1,H0,H1,A0,A1,B0,B1,a2) }
                    { YUNPACK(y3[ib].qs,j,A0,A1,B0,B1) MAC4(L0,L1,H0,H1,A0,A1,B0,B1,a3) } }
                const float dw = h2f(w[ib].d);
                s0+=a0*dw*h2f(y0[ib].d); s1+=a1*dw*h2f(y1[ib].d);
                s2+=a2*dw*h2f(y2[ib].d); s3+=a3*dw*h2f(y3[ib].d); }
            C[(size_t)oc*OL+ol]=s0; C[(size_t)oc*OL+ol+1]=s1;
            C[(size_t)oc*OL+ol+2]=s2; C[(size_t)oc*OL+ol+3]=s3; } }
}

int main(void) {
    W = malloc(sizeof(blk4)*(size_t)NB*OC); X = malloc(sizeof(blk8)*(size_t)NB*OL);
    C = malloc(sizeof(float)*(size_t)OC*OL);
    srand(11);
    for (size_t i=0;i<(size_t)NB*OC;i++){ W[i].d=0x3000; for(int j=0;j<QK/2;j++) W[i].qs[j]=rand()&0xFF; }
    for (size_t i=0;i<(size_t)NB*OL;i++){ X[i].d=0x3400; for(int j=0;j<QK;j++) X[i].qs[j]=(int8_t)(rand()&0xFF); }

    const size_t bytes = sizeof(float)*(size_t)OC*OL;
    float *ref = malloc(bytes);
    gemmA14(); memcpy(ref, C, bytes);
    memset(C,0,bytes); gemmB11(); const int okB11 = !memcmp(ref,C,bytes);
    memset(C,0,bytes); gemmB22(); const int okB22 = !memcmp(ref,C,bytes);
    printf("identical to order-A 1x4:  B 1x1 %s   B 2x2 %s\n\n", okB11?"yes":"NO", okB22?"yes":"NO");

    const double macs = (double)OC*OL*KDIM; const int reps = 3; double t;
    t=now(); for(int r=0;r<reps;r++) gemmA14(); const double ta=now()-t;
    t=now(); for(int r=0;r<reps;r++) gemmB11(); const double tb=now()-t;
    t=now(); for(int r=0;r<reps;r++) gemmB22(); const double tc=now()-t;
    printf("order A, 1x4 cols (bench28)  %6.1f MMAC/s  %5.2f s\n", macs*reps/ta/1e6, ta/reps);
    printf("order B, 1x1                 %6.1f MMAC/s  %5.2f s   %.2fx\n", macs*reps/tb/1e6, tb/reps, ta/tb);
    printf("order B, 2x2 tile            %6.1f MMAC/s  %5.2f s   %.2fx\n", macs*reps/tc/1e6, tc/reps, ta/tc);
    return 0;
}
