// P7.1: WHAT A PHASE-MAJOR LAYOUT WOULD BE WORTH TO THE ARMv6 DIRECT SWEEP.
//
// The last convolution bucket still taking the direct sweep is VITS's OC=32 resblocks, and its rate
// tracks one thing: the BYTE SPAN of the tap window, `(k-1)*dilation` floats. Per-node, on an idle
// Pi Zero W:
//
//     k=3 d=1    8 B span   207.7 MMAC/s        k=7 d=3    72 B span   136.3
//     k=3 d=2   16 B span   198.1               k=5 d=6    96 B span   135.8
//     k=5 d=2   32 B span   154.1               k=7 d=12  288 B span   107.9
//
// A tile touches IC * ceil(span/32) lines, so at k=7 d=12 that is 320 of this core's 512, plus the
// weights. Phase-major ("a trous") makes each channel's taps CONTIGUOUS -- span k floats, one line --
// which is what the d=1 row already measures. ggml has that path (ggml_conv1d_phase_ok) and it is
// compile-time gated to aarch64, where its de-interleave is one instruction; it also requires
// IC*KW >= 768, and these shapes are 224.
//
// Rather than port it to find out, this BOUNDS it. Arm A is the shipped tile at the real dilation.
// Arm B is the same tile on a dense layout with the same MAC count -- the ceiling phase-major could
// reach. Arm C is the de-interleave and interleave the transform would actually pay. Phase-major is
// worth building iff A > B + C, and by how much.
//
//   gcc -O2 -o bench38 bench38.c && ./bench38
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

#define IC 32
#define OC 32
#define OL 70400
#define OCB 4
#define VEC 4

static float *xp, *wp, *y, *ph_x, *ph_y;

/* ggml_conv_1d_direct_tile_impl's ARMv6 arm: explicit scalars, 4 channels x 4 positions. */
static void sweep(int64_t KW, int64_t dil, int64_t LP) {
    for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB)
        for (int64_t p0 = 0; p0 + VEC <= OL; p0 += VEC) {
            float a00=0,a01=0,a02=0,a03=0, a10=0,a11=0,a12=0,a13=0;
            float a20=0,a21=0,a22=0,a23=0, a30=0,a31=0,a32=0,a33=0;
            for (int64_t ic = 0; ic < IC; ++ic) {
                const float *xr = xp + ic*LP + p0;
                const float *wr = wp + ic*KW*OC + oc0;
                for (int64_t kx = 0; kx < KW; ++kx, xr += dil, wr += OC) {
                    const float x0=xr[0], x1=xr[1], x2=xr[2], x3=xr[3];
                    const float w0=wr[0], w1=wr[1], w2=wr[2], w3=wr[3];
                    a00+=w0*x0; a01+=w0*x1; a02+=w0*x2; a03+=w0*x3;
                    a10+=w1*x0; a11+=w1*x1; a12+=w1*x2; a13+=w1*x3;
                    a20+=w2*x0; a21+=w2*x1; a22+=w2*x2; a23+=w2*x3;
                    a30+=w3*x0; a31+=w3*x1; a32+=w3*x2; a33+=w3*x3;
                }
            }
            const float st[4][4]={{a00,a01,a02,a03},{a10,a11,a12,a13},
                                  {a20,a21,a22,a23},{a30,a31,a32,a33}};
            for (int i=0;i<OCB;i++) for (int j=0;j<VEC;j++) y[(oc0+i)*OL+p0+j]=st[i][j];
        }
}

/* the transform's real cost: de-interleave IC channels into `dil` phases, and interleave OC back */
static void transform(int64_t dil, int64_t LP, int64_t J) {
    for (int64_t ic = 0; ic < IC; ++ic) {
        const float *src = xp + ic*LP;
        for (int64_t r = 0; r < dil; ++r) {
            float *d = ph_x + (ic*dil + r)*J;
            for (int64_t j = 0; j < J; ++j) {
                const int64_t p = j*dil + r;
                d[j] = (p < LP) ? src[p] : 0.0f;
            }
        }
    }
    for (int64_t oc = 0; oc < OC; ++oc) {
        float *dst = y + oc*OL;
        for (int64_t r = 0; r < dil; ++r) {
            const float *s = ph_y + (oc*dil + r)*J;
            for (int64_t j = 0; j < J; ++j) {
                const int64_t p = j*dil + r;
                if (p < OL) dst[p] = s[j];
            }
        }
    }
}

static void run(int64_t KW, int64_t dil) {
    const int64_t LP = OL + (KW-1)*dil;
    const int64_t J  = (OL + dil - 1)/dil + KW;      /* one phase, with its halo */
    const double mmac = (double)OL*IC*OC*KW/1e6;
    const int64_t span = (KW-1)*dil*4;

    double t0 = now(); sweep(KW, dil, LP);          double tA = now()-t0;   /* as shipped */
    t0 = now();        sweep(KW, 1,  LP);           double tB = now()-t0;   /* dense: the ceiling */
    t0 = now();        transform(dil, LP, J);       double tC = now()-t0;   /* what it would cost */

    printf("  k=%lld d=%2lld  span %4lld B  %7.1f MMAC\n",
           (long long)KW, (long long)dil, (long long)span, mmac);
    printf("      A shipped, dilated   %8.1f ms  %6.1f MMAC/s\n", tA*1e3, mmac/tA);
    printf("      B dense (ceiling)    %8.1f ms  %6.1f MMAC/s\n", tB*1e3, mmac/tB);
    printf("      C transform          %8.1f ms\n", tC*1e3);
    printf("      B+C                  %8.1f ms  -> %s by %.2fx\n\n", (tB+tC)*1e3,
           (tB+tC) < tA ? "WINS" : "loses", tA/(tB+tC));
}

int main(void) {
    const int64_t LPMAX = OL + 6*12;
    const int64_t JMAX  = (OL + 1)/1 + 7;
    xp = malloc(sizeof(float)*(size_t)IC*LPMAX);
    wp = malloc(sizeof(float)*(size_t)IC*7*OC);
    y  = malloc(sizeof(float)*(size_t)OC*OL);
    ph_x = malloc(sizeof(float)*(size_t)IC*JMAX + 4096);
    ph_y = malloc(sizeof(float)*(size_t)OC*JMAX + 4096);
    if (!xp||!wp||!y||!ph_x||!ph_y) { puts("alloc failed"); return 1; }
    for (size_t i=0;i<(size_t)IC*LPMAX;++i) xp[i]=(float)(i%13)*0.01f;
    for (size_t i=0;i<(size_t)IC*7*OC;++i)  wp[i]=(float)(i%7)*0.01f;

    puts("VITS's OC=32 resblock convolutions, one call each:\n");
    run(3, 1); run(3, 2); run(5, 2); run(5, 6); run(7, 3); run(7, 12);
    return 0;
}
