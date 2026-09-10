// P7.1: WHY CONV_TRANSPOSE_1D'S FIRST NODE RUNS ITS GEMM AT 40% OF THE ROOFLINE.
//
// `tinyBLAS_F32_ARMV6::gemm44` reaches 226.7 MMAC/s on the shapes it was tuned on, and
// CONV_TRANSPOSE_1D's three nodes get 91.3, 209.7 and 219.5 (per-node profile minus bench35's
// non-GEMM phases). The three differ in both n and k, so this separates them: gemm44 is reproduced
// verbatim, single-threaded, and each shape is run at its own n and at the n the other nodes get.
//
//     node 1   m=2048 k=256   n=32   <- what GGML_CONV_TRANSPOSE_1D_TILE (64 KB) / mk gives it
//     node 2   m=1024 k=128   n=64
//     node 3   m= 256 k= 64   n=256
//
// If the slow one is slow because of n, raising that tile constant fixes it and the constant is the
// same kind of cache-sized guess as the other two P7.1 items. If it is k, the kernel is the problem.
//
//   gcc -O2 -o bench36 bench36.c && ./bench36
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static float *A, *B, *C;

/* tinyBLAS_F32_ARMV6::gemm44, verbatim, one thread, col_outer == false (which is what all three
   nodes select: n*k < m*k for every one of them). */
static void gemm44(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
    const int64_t ytiles = m / 4;
    const int64_t xtiles = n / 4;
    const int64_t tiles  = xtiles * ytiles;
    for (int64_t job = 0; job < tiles; ++job) {
        const int64_t jj = job % xtiles * 4;
        const int64_t ii = job / xtiles * 4;
        const float *a0 = A + lda*(ii+0), *a1 = A + lda*(ii+1);
        const float *a2 = A + lda*(ii+2), *a3 = A + lda*(ii+3);
        const float *b0 = B + ldb*(jj+0), *b1 = B + ldb*(jj+1);
        const float *b2 = B + ldb*(jj+2), *b3 = B + ldb*(jj+3);
        float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;
        float c20=0,c21=0,c22=0,c23=0, c30=0,c31=0,c32=0,c33=0;
        for (int64_t l = 0; l < k; ++l) {
            const float x0 = *a0++, x1 = *a1++, x2 = *a2++, x3 = *a3++;
            const float y0 = *b0++, y1 = *b1++, y2 = *b2++, y3 = *b3++;
            c00 += x0*y0; c01 += x1*y0; c02 += x2*y0; c03 += x3*y0;
            c10 += x0*y1; c11 += x1*y1; c12 += x2*y1; c13 += x3*y1;
            c20 += x0*y2; c21 += x1*y2; c22 += x2*y2; c23 += x3*y2;
            c30 += x0*y3; c31 += x1*y3; c32 += x2*y3; c33 += x3*y3;
        }
        float *r0 = C + ldc*(jj+0) + ii, *r1 = C + ldc*(jj+1) + ii;
        float *r2 = C + ldc*(jj+2) + ii, *r3 = C + ldc*(jj+3) + ii;
        r0[0]=c00; r0[1]=c01; r0[2]=c02; r0[3]=c03;
        r1[0]=c10; r1[1]=c11; r1[2]=c12; r1[3]=c13;
        r2[0]=c20; r2[1]=c21; r2[2]=c22; r2[3]=c23;
        r3[0]=c30; r3[1]=c31; r3[2]=c32; r3[3]=c33;
    }
}

/* One node's whole GEMM workload: L positions in ceil(L/n) calls of width n. */
static void run(const char *name, int64_t m, int64_t k, int64_t n, int64_t L) {
    const double mmac = (double)m * L * k / 1e6;
    double t0 = now();
    for (int64_t l0 = 0; l0 < L; l0 += n) {
        int64_t nl = (l0 + n > L) ? (L - l0) : n;
        nl = nl / 4 * 4;                 /* the 4x4 path; the ragged tail is a different kernel */
        if (nl) gemm44(m, nl, k, k, k, m);
    }
    double dt = now() - t0;
    printf("  %-7s m=%4lld k=%3lld n=%5lld L=%5lld   %7.1f ms   %6.1f MMAC/s\n",
           name, (long long)m, (long long)k, (long long)n, (long long)L, dt*1e3, mmac/dt);
}

int main(void) {
    /* big enough for the largest m*k, n*k and m*n used below */
    A = malloc(sizeof(float)*2048*256);
    B = malloc(sizeof(float)*2200*256);
    C = malloc(sizeof(float)*2048*2200);
    if (!A||!B||!C) { puts("alloc failed"); return 1; }
    for (long i = 0; i < 2048L*256; ++i) A[i] = (float)(i%13)*0.01f;
    for (long i = 0; i < 2200L*256; ++i) B[i] = (float)(i%7)*0.01f;

    puts("node 1 shape (m=2048 k=256), sweeping n:");
    run("n=32",  2048, 256,  32, 275);          /* what it gets today */
    run("n=64",  2048, 256,  64, 275);
    run("n=128", 2048, 256, 128, 275);
    run("n=272", 2048, 256, 272, 275);          /* one call, whole layer */

    puts("\nnode 2 shape (m=1024 k=128), sweeping n:");
    run("n=64",  1024, 128,  64, 2200);         /* what it gets today */
    run("n=256", 1024, 128, 256, 2200);
    run("n=512", 1024, 128, 512, 2200);

    puts("\nnode 3 shape (m=256 k=64), sweeping n:");
    run("n=256",  256,  64, 256, 2200);         /* what it gets today (L truncated to fit B) */
    run("n=1024", 256,  64,1024, 2200);

    puts("\nsame n, different k -- is the axis k rather than n?");
    run("k=256", 2048, 256, 64, 275);
    run("k=128", 2048, 128, 64, 275);
    run("k=64",  2048,  64, 64, 275);
    return 0;
}
