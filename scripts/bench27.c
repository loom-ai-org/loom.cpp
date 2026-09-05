// P7: the lowering VITS's DOMINANT convolutions actually take on ARMv6, against the one ggml already
// has. Shape is the real one -- flow_vocoder's IC=192 OC=384 K=5 at OL=275, 16 calls and the largest
// bucket in LOOM_PROFILE (35.8% of a synthesis).
//
// ggml_conv_1d_direct_ok declines it twice over (weights 1.47 MB against a 512 KB budget; OL/4=68 <
// OC/4=96), so ggml_compute_forward_conv_2d_impl dequantizes the kernel to F32 and takes the batched
// im2col + F32 GEMM fallback -- arm (a). Arms (b) and (c) are what the SAME im2col could feed if the
// kernel stayed quantized: ggml's existing q4_0 x q8_0 GEMM, generic and __smlad.
//   gcc -O2 -o bench27 bench27.c && ./bench27
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define QK 32
#define KDIM 960          /* IC*K = 192*5 */
#define OC   384
#define OL   275
#define NB   (KDIM / QK)

typedef struct { uint16_t d; uint8_t qs[QK/2]; } blk4;
typedef struct { uint16_t d; int8_t  qs[QK];   } blk8;

static float h2f(uint16_t h){ uint32_t s=(uint32_t)(h&0x8000u)<<16, em=h&0x7FFFu;
    uint32_t b = em ? (s | ((em + 0x1C000u) << 13)) : s; float f; memcpy(&f,&b,4); return f; }
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

// (a) what ggml_vec_dot_f32's generic arm is on a target with no GGML_SIMD
static float dot_f32(int n, const float *x, const float *y) {
    float s = 0; for (int i = 0; i < n; i++) s += x[i]*y[i]; return s;
}
// (b) ggml_vec_dot_q4_0_q8_0_generic
static float dot_q_generic(const blk4 *x, const blk8 *y) {
    float sumf = 0;
    for (int ib = 0; ib < NB; ib++) {
        int s0 = 0, s1 = 0;
        for (int j = 0; j < QK/2; j++) {
            s0 += ((x[ib].qs[j] & 0x0F) - 8) * y[ib].qs[j];
            s1 += ((x[ib].qs[j] >>   4) - 8) * y[ib].qs[j + QK/2];
        }
        sumf += (s0 + s1) * h2f(x[ib].d) * h2f(y[ib].d);
    }
    return sumf;
}
// (c) the same, on ARMv6 SIMD32
static float dot_q_smlad(const blk4 *x, const blk8 *y) {
    float sumf = 0;
    for (int ib = 0; ib < NB; ib++) {
        int32_t a = 0;
        for (int j = 0; j < QK/2; j += 4) {
            uint32_t xv,y0,y1;
            memcpy(&xv,x[ib].qs+j,4); memcpy(&y0,y[ib].qs+j,4); memcpy(&y1,y[ib].qs+QK/2+j,4);
            const uint32_t lo=xv&0x0F0F0F0Fu, hi=(xv>>4)&0x0F0F0F0Fu, e=0x00080008u;
            a = __smlad(__ssub16(__uxtb16(lo),     e), __sxtb16(y0),     a);
            a = __smlad(__ssub16(__uxtb16(lo>>8),  e), __sxtb16(y0>>8),  a);
            a = __smlad(__ssub16(__uxtb16(hi),     e), __sxtb16(y1),     a);
            a = __smlad(__ssub16(__uxtb16(hi>>8),  e), __sxtb16(y1>>8),  a);
        }
        sumf += a * h2f(x[ib].d) * h2f(y[ib].d);
    }
    return sumf;
}

static float *Wf, *Xf, *C;  static blk4 *Wq; static blk8 *Xq;

static void quantize_cols(void) {                 // the pass arms (b)/(c) must pay and (a) must not
    for (int c = 0; c < OL; c++)
        for (int ib = 0; ib < NB; ib++) {
            const float *src = Xf + (size_t)c*KDIM + ib*QK;
            float amax = 0; for (int j = 0; j < QK; j++) { float a = src[j] < 0 ? -src[j] : src[j]; if (a > amax) amax = a; }
            const float d = amax / 127.0f, id = d ? 1.0f/d : 0.0f;
            blk8 *b = &Xq[(size_t)c*NB + ib];
            b->d = 0x3400;
            for (int j = 0; j < QK; j++) { int v = (int)(src[j]*id + (src[j] >= 0 ? 0.5f : -0.5f)); b->qs[j] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v); }
        }
}

int main(void) {
    Wf = malloc(sizeof(float)*(size_t)KDIM*OC); Xf = malloc(sizeof(float)*(size_t)KDIM*OL);
    C  = malloc(sizeof(float)*(size_t)OC*OL);
    Wq = malloc(sizeof(blk4)*(size_t)NB*OC);     Xq = malloc(sizeof(blk8)*(size_t)NB*OL);
    srand(3);
    for (size_t i = 0; i < (size_t)KDIM*OC; i++) Wf[i] = (rand()%2001-1000)/10000.0f;
    for (size_t i = 0; i < (size_t)KDIM*OL; i++) Xf[i] = (rand()%2001-1000)/1000.0f;
    for (size_t i = 0; i < (size_t)NB*OC; i++) { Wq[i].d = 0x3000; for (int j=0;j<QK/2;j++) Wq[i].qs[j]=rand()&0xFF; }

    const double macs = (double)OC*OL*KDIM;
    const int reps = 3; double t; volatile float sink = 0;

    t = now();
    for (int r = 0; r < reps; r++) for (int oc = 0; oc < OC; oc++) for (int ol = 0; ol < OL; ol++)
        C[(size_t)oc*OL+ol] = dot_f32(KDIM, Wf + (size_t)oc*KDIM, Xf + (size_t)ol*KDIM);
    const double ta = now()-t; sink += C[0];

    t = now(); for (int r = 0; r < reps; r++) quantize_cols(); const double tq = (now()-t)/reps;

    t = now();
    for (int r = 0; r < reps; r++) for (int oc = 0; oc < OC; oc++) for (int ol = 0; ol < OL; ol++)
        C[(size_t)oc*OL+ol] = dot_q_generic(Wq + (size_t)oc*NB, Xq + (size_t)ol*NB);
    const double tb = now()-t; sink += C[0];

    t = now();
    for (int r = 0; r < reps; r++) for (int oc = 0; oc < OC; oc++) for (int ol = 0; ol < OL; ol++)
        C[(size_t)oc*OL+ol] = dot_q_smlad(Wq + (size_t)oc*NB, Xq + (size_t)ol*NB);
    const double tc = now()-t; sink += C[0];

    printf("shape IC*K=%d OC=%d OL=%d   %.1f MMAC per call\n\n", KDIM, OC, OL, macs/1e6);
    printf("(a) F32 GEMM   (what it does today) %7.1f MMAC/s   %6.2f s/call\n", macs*reps/ta/1e6, ta/reps);
    printf("(b) q4_0 GEMM, shipped vec_dot      %7.1f MMAC/s   %6.2f s/call   %.2fx\n", macs*reps/tb/1e6, tb/reps, ta/tb);
    printf("(c) q4_0 GEMM, __smlad vec_dot      %7.1f MMAC/s   %6.2f s/call   %.2fx\n", macs*reps/tc/1e6, tc/reps, ta/tc);
    printf("\n    activation quantize pass          %6.3f s/call  (%.1f%% of arm c)\n", tq, 100.0*tq/(tc/reps));
    printf("    including it, (c) is                %.2fx\n", ta/reps/(tc/reps + tq));
    (void)sink; return 0;
}
