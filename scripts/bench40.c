// P7.1: THE im2col GEMM RE-READS ITS PATCH PANEL ONCE PER FOUR OUTPUT CHANNELS.
//
// With the direct sweep and conv_transpose_1d closed, three im2col buckets are 38.5 s of a 72 s VITS
// synthesis and run at 132.5 MMAC/s where the same kernel reaches 226-286 elsewhere. `tinyBLAS`'s
// `mnpack` is a 4x4 REGISTER tile with no CACHE blocking: it sweeps the whole (m, n) space flat, so
// with `col_outer` set the patch panel A is streamed once per 4-column tile. A is 18-56 KB at these
// shapes and the L1 is 16 KB, so none of that is free.
//
//     bucket        per-batch GEMM        A       B      batches   A re-read
//     17600 k=7     m=32 n= 64 k=448    56 KB   112 KB     550        16x
//     2200  k=7     m=16 n=128 k=896    56 KB   448 KB     138        32x
//     275   k=5     m= 8 n=384 k=960    30 KB  1440 KB      35        96x
//
// Arm A is the shipped flat tile. Arm B blocks the reduction so A's slice fits L1 and accumulates C
// across chunks -- the same 4x4 inner kernel, only the traversal changes. If the gap is panel traffic
// this closes it; if it is not, arm B measures the same and the idea is dead for one bench.
//
//   gcc -O2 -o bench40 bench40.c && ./bench40
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static float *A, *B, *C;

/* the shipped inner kernel: 4x4 register tile, col_outer (n outer, m inner) */
static void gemm44_flat(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
    const int64_t ytiles = m/4, xtiles = n/4, tiles = xtiles*ytiles;
    for (int64_t job = 0; job < tiles; ++job) {
        const int64_t jj = job / ytiles * 4, ii = job % ytiles * 4;
        const float *a0=A+lda*(ii+0), *a1=A+lda*(ii+1), *a2=A+lda*(ii+2), *a3=A+lda*(ii+3);
        const float *b0=B+ldb*(jj+0), *b1=B+ldb*(jj+1), *b2=B+ldb*(jj+2), *b3=B+ldb*(jj+3);
        float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;
        float c20=0,c21=0,c22=0,c23=0, c30=0,c31=0,c32=0,c33=0;
        for (int64_t l = 0; l < k; ++l) {
            const float x0=*a0++, x1=*a1++, x2=*a2++, x3=*a3++;
            const float y0=*b0++, y1=*b1++, y2=*b2++, y3=*b3++;
            c00+=x0*y0; c01+=x1*y0; c02+=x2*y0; c03+=x3*y0;
            c10+=x0*y1; c11+=x1*y1; c12+=x2*y1; c13+=x3*y1;
            c20+=x0*y2; c21+=x1*y2; c22+=x2*y2; c23+=x3*y2;
            c30+=x0*y3; c31+=x1*y3; c32+=x2*y3; c33+=x3*y3;
        }
        float *r0=C+ldc*(jj+0)+ii, *r1=C+ldc*(jj+1)+ii;
        float *r2=C+ldc*(jj+2)+ii, *r3=C+ldc*(jj+3)+ii;
        r0[0]=c00;r0[1]=c01;r0[2]=c02;r0[3]=c03; r1[0]=c10;r1[1]=c11;r1[2]=c12;r1[3]=c13;
        r2[0]=c20;r2[1]=c21;r2[2]=c22;r2[3]=c23; r3[0]=c30;r3[1]=c31;r3[2]=c32;r3[3]=c33;
    }
}

/* the same inner kernel, with the reduction blocked so A's slice stays resident. C is accumulated
   across chunks, so it is read and written per chunk instead of written once. */
static void gemm44_kblocked(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc,
                            int64_t kc) {
    const int64_t ytiles = m/4, xtiles = n/4, tiles = xtiles*ytiles;
    for (int64_t k0 = 0; k0 < k; k0 += kc) {
        const int64_t kn = (k0 + kc > k) ? (k - k0) : kc;
        const int first = (k0 == 0);
        for (int64_t job = 0; job < tiles; ++job) {
            const int64_t jj = job / ytiles * 4, ii = job % ytiles * 4;
            const float *a0=A+lda*(ii+0)+k0, *a1=A+lda*(ii+1)+k0;
            const float *a2=A+lda*(ii+2)+k0, *a3=A+lda*(ii+3)+k0;
            const float *b0=B+ldb*(jj+0)+k0, *b1=B+ldb*(jj+1)+k0;
            const float *b2=B+ldb*(jj+2)+k0, *b3=B+ldb*(jj+3)+k0;
            float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;
            float c20=0,c21=0,c22=0,c23=0, c30=0,c31=0,c32=0,c33=0;
            for (int64_t l = 0; l < kn; ++l) {
                const float x0=*a0++, x1=*a1++, x2=*a2++, x3=*a3++;
                const float y0=*b0++, y1=*b1++, y2=*b2++, y3=*b3++;
                c00+=x0*y0; c01+=x1*y0; c02+=x2*y0; c03+=x3*y0;
                c10+=x0*y1; c11+=x1*y1; c12+=x2*y1; c13+=x3*y1;
                c20+=x0*y2; c21+=x1*y2; c22+=x2*y2; c23+=x3*y2;
                c30+=x0*y3; c31+=x1*y3; c32+=x2*y3; c33+=x3*y3;
            }
            float *r0=C+ldc*(jj+0)+ii, *r1=C+ldc*(jj+1)+ii;
            float *r2=C+ldc*(jj+2)+ii, *r3=C+ldc*(jj+3)+ii;
            if (first) {
                r0[0]=c00;r0[1]=c01;r0[2]=c02;r0[3]=c03; r1[0]=c10;r1[1]=c11;r1[2]=c12;r1[3]=c13;
                r2[0]=c20;r2[1]=c21;r2[2]=c22;r2[3]=c23; r3[0]=c30;r3[1]=c31;r3[2]=c32;r3[3]=c33;
            } else {
                r0[0]+=c00;r0[1]+=c01;r0[2]+=c02;r0[3]+=c03; r1[0]+=c10;r1[1]+=c11;r1[2]+=c12;r1[3]+=c13;
                r2[0]+=c20;r2[1]+=c21;r2[2]+=c22;r2[3]+=c23; r3[0]+=c30;r3[1]+=c31;r3[2]+=c32;r3[3]+=c33;
            }
        }
    }
}

static const int64_t KCS[] = {32, 64, 128, 256, 512};
static int g_reverse = 0;

static void run(const char *name, int64_t m, int64_t n, int64_t k, int64_t batches, int64_t calls) {
    const double mmac = (double)m*n*k*batches*calls/1e6;

    double t0 = now();
    for (int64_t b = 0; b < batches; ++b) gemm44_flat(m, n, k, k, k, m);
    const double tf = now() - t0;

    printf("  %-11s m=%3lld n=%3lld k=%4lld  A=%5.1f KB C=%5.1f KB  x%4lld batches x%2lld calls\n",
           name, (long long)m, (long long)n, (long long)k,
           m*k*4/1024.0, m*n*4/1024.0, (long long)batches, (long long)calls);
    printf("        flat            %8.1f ms  %6.1f MMAC/s  -> %6.2f s/synth\n",
           tf*1e3, mmac/calls/tf, tf*calls);
    const size_t NK = sizeof(KCS)/sizeof(KCS[0]);
    for (size_t ii = 0; ii < NK; ++ii) {
        const size_t i = g_reverse ? (NK - 1 - ii) : ii;
        const int64_t kc = KCS[i];
        if (kc >= k) continue;                       /* one chunk is the flat path */
        t0 = now();
        for (int64_t b = 0; b < batches; ++b) gemm44_kblocked(m, n, k, k, k, m, kc);
        const double tb = now() - t0;
        printf("        kc=%-4lld        %8.1f ms  %6.1f MMAC/s  -> %6.2f s/synth   %.2fx%s\n",
               (long long)kc, tb*1e3, mmac/calls/tb, tb*calls, tf/tb, tf/tb > 1.0 ? "" : "   (loses)");
    }
    /* the flat arm again, last, so any drift across the arms above is visible rather than folded in */
    t0 = now();
    for (int64_t b = 0; b < batches; ++b) gemm44_flat(m, n, k, k, k, m);
    const double tf2 = now() - t0;
    printf("        flat again      %8.1f ms  %6.1f MMAC/s   (first pass %.1f, drift %+.1f%%)\n\n",
           tf2*1e3, mmac/calls/tf2, tf*1e3, (tf2/tf - 1.0)*100.0);
}

int main(void) {
    A = malloc(sizeof(float)*64*1024);
    B = malloc(sizeof(float)*384*960);
    C = malloc(sizeof(float)*64*384);
    if (!A||!B||!C) { puts("alloc failed"); return 1; }
    for (size_t i=0;i<64*1024;++i)   A[i]=(float)(i%13)*0.01f;
    for (size_t i=0;i<384*960;++i)   B[i]=(float)(i%7)*0.01f;

    for (int pass = 0; pass < 2; ++pass) {
    g_reverse = pass;
    printf("======== pass %d, kc order %s ========\n\n", pass, pass ? "descending" : "ascending");
    run("17600 k=7", 32,  64, 448, 550, 2);
    run("17600 k=5", 40,  64, 320, 440, 2);
    run("17600 k=3", 64,  64, 192, 275, 2);
    run("2200  k=7", 16, 128, 896, 138, 2);
    run("2200  k=3", 32, 128, 384,  69, 2);
    run("275   k=5",  8, 384, 960,  35, 16);
    run("275   k=1", 24, 384, 192,  12, 12);
    }
    return 0;
}
