// P7.1: IS CONV_TRANSPOSE_1D'S FIRST NODE HITTING 4 KB CACHE ALIASING?
//
// bench36 found the axis is k, and that it is a cliff rather than a slope: at m=2048 n=64 the
// ARMv6 4x4 GEMM does 280.4 MMAC/s at k=64, 220.9 at k=128 and 81.4 at k=256.
//
// The hypothesis. `gemm44` walks EIGHT streams -- a0..a3 and b0..b3 -- with a0..a3 at lda apart and
// b0..b3 at ldb apart, and here lda = ldb = k. An ARM1176 has a 16 KB 4-way L1, so its way is 4 KB.
// At k=256 a row is exactly 1024 bytes, four rows tile a way exactly, and the eight streams land in
// four sets with four ways between them -- every access evicts one of the others. At k=128 a row is
// 512 bytes and the streams spread over twice as many sets.
//
// If that is what it is, the fix is a stride and not a kernel: pad lda/ldb so the rows stop tiling
// the way. That costs nothing -- the layout is produced by this op's own repack -- and it is testable
// here before any of it is built.
//
// Arms: the baseline; padding A only; padding B only; padding both, by one line and by two; and
// k-blocking, which halves the live window instead of moving it.
//
//   gcc -O2 -o bench37 bench37.c && ./bench37
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static float *A, *B, *C;

/* gemm44, with lda and ldb free rather than pinned to k. */
static void gemm44(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc) {
    const int64_t ytiles = m / 4, xtiles = n / 4, tiles = xtiles * ytiles;
    for (int64_t job = 0; job < tiles; ++job) {
        const int64_t jj = job % xtiles * 4, ii = job / xtiles * 4;
        const float *a0 = A + lda*(ii+0), *a1 = A + lda*(ii+1);
        const float *a2 = A + lda*(ii+2), *a3 = A + lda*(ii+3);
        const float *b0 = B + ldb*(jj+0), *b1 = B + ldb*(jj+1);
        const float *b2 = B + ldb*(jj+2), *b3 = B + ldb*(jj+3);
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

/* the same, with the k reduction split into blocks of KB so the live window is smaller */
static void gemm44_kblocked(int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb,
                            int64_t ldc, int64_t KB) {
    const int64_t ytiles = m / 4, xtiles = n / 4, tiles = xtiles * ytiles;
    for (int64_t job = 0; job < tiles; ++job) {
        const int64_t jj = job % xtiles * 4, ii = job / xtiles * 4;
        float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;
        float c20=0,c21=0,c22=0,c23=0, c30=0,c31=0,c32=0,c33=0;
        for (int64_t l0 = 0; l0 < k; l0 += KB) {
            const int64_t le = (l0+KB > k) ? k : l0+KB;
            const float *a0=A+lda*(ii+0)+l0, *a1=A+lda*(ii+1)+l0;
            const float *a2=A+lda*(ii+2)+l0, *a3=A+lda*(ii+3)+l0;
            const float *b0=B+ldb*(jj+0)+l0, *b1=B+ldb*(jj+1)+l0;
            const float *b2=B+ldb*(jj+2)+l0, *b3=B+ldb*(jj+3)+l0;
            for (int64_t l = l0; l < le; ++l) {
                const float x0=*a0++, x1=*a1++, x2=*a2++, x3=*a3++;
                const float y0=*b0++, y1=*b1++, y2=*b2++, y3=*b3++;
                c00+=x0*y0; c01+=x1*y0; c02+=x2*y0; c03+=x3*y0;
                c10+=x0*y1; c11+=x1*y1; c12+=x2*y1; c13+=x3*y1;
                c20+=x0*y2; c21+=x1*y2; c22+=x2*y2; c23+=x3*y2;
                c30+=x0*y3; c31+=x1*y3; c32+=x2*y3; c33+=x3*y3;
            }
        }
        float *r0=C+ldc*(jj+0)+ii, *r1=C+ldc*(jj+1)+ii;
        float *r2=C+ldc*(jj+2)+ii, *r3=C+ldc*(jj+3)+ii;
        r0[0]=c00;r0[1]=c01;r0[2]=c02;r0[3]=c03; r1[0]=c10;r1[1]=c11;r1[2]=c12;r1[3]=c13;
        r2[0]=c20;r2[1]=c21;r2[2]=c22;r2[3]=c23; r3[0]=c30;r3[1]=c31;r3[2]=c32;r3[3]=c33;
    }
}

#define M 2048
#define K 256
#define N 64
#define L 275

static float *Abase, *Bbase;

static void arm_off(const char *what, int64_t lda, int64_t ldb, int64_t KB, int aoff, int boff);
static void arm(const char *what, int64_t lda, int64_t ldb, int64_t KB) { arm_off(what, lda, ldb, KB, 0, 0); }

static void arm_off(const char *what, int64_t lda, int64_t ldb, int64_t KB, int aoff, int boff) {
    const double mmac = (double)M * L * K / 1e6;
    A = Abase + aoff; B = Bbase + boff;
    double t0 = now();
    for (int64_t l0 = 0; l0 < L; l0 += N) {
        int64_t nl = ((l0 + N > L) ? (L - l0) : N) / 4 * 4;
        if (!nl) continue;
        if (KB) gemm44_kblocked(M, nl, K, lda, ldb, M, KB);
        else    gemm44(M, nl, K, lda, ldb, M);
    }
    double dt = now() - t0;
    printf("  %-30s lda=%4lld ldb=%4lld  %7.1f ms  %6.1f MMAC/s\n",
           what, (long long)lda, (long long)ldb, dt*1e3, mmac/dt);
}

int main(void) {
    /* padded to the largest stride used */
    Abase = malloc(sizeof(float)*(size_t)M*(K+16));
    Bbase = malloc(sizeof(float)*(size_t)2200*(K+16));
    A = Abase; B = Bbase;
    C = malloc(sizeof(float)*(size_t)M*N);
    if (!Abase||!Bbase||!C) { puts("alloc failed"); return 1; }
    for (size_t i = 0; i < (size_t)M*(K+16); ++i) A[i] = (float)(i%13)*0.01f;
    for (size_t i = 0; i < (size_t)2200*(K+16); ++i) B[i] = (float)(i%7)*0.01f;

    printf("node 1's GEMM, m=%d k=%d n=%d, L=%d\n", M, K, N, L);
    arm("baseline (lda=ldb=k)",      K,    K,    0);
    arm("pad A by one line (+8)",    K+8,  K,    0);
    arm("pad B by one line (+8)",    K,    K+8,  0);
    arm("pad both by one line",      K+8,  K+8,  0);
    arm("pad both by two lines",     K+16, K+16, 0);
    arm("pad both by half a line",   K+4,  K+4,  0);
    arm("k-blocked 128, no padding", K,    K,    128);
    arm("k-blocked 64,  no padding", K,    K,    64);
    arm("k-blocked 64 + pad both",   K+8,  K+8,  64);

    /* THE CHEAP FIX, if the mechanism is what it looks like. In the engine the two panels sit at
       wdata and wdata + nk, and node 1's nk is 524288 floats -- exactly 2 MB, a whole number of
       4 KB ways -- so the panels are placed on top of each other by construction. Moving the second
       one by a single line needs no stride plumbing at all: one addend in conv_transpose_1d. */
    puts("");
    arm_off("offset B base by +4 floats", K, K, 0, 0, 4);
    arm_off("offset B base by +8 floats", K, K, 0, 0, 8);
    arm_off("offset B base by +16 floats",K, K, 0, 0, 16);
    arm_off("offset B base by +260",      K, K, 0, 0, 260);
    return 0;
}
