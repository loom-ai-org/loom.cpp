// P7: two questions about the ARMv6 tile, at conformer's feed-forward shape (m=704 n=75 k=176).
//
// 1. WHY DOES 4x4 SPILL? bench30 measured 2x2 at 153.7 MMAC/s and 4x4 at 27.2. Sixteen accumulators
//    plus eight operands is 24 of VFP's 32 single registers, which should fit -- so the suspect is not
//    the float file. Two candidates: the `av[RM]`/`bv[RN]` ARRAYS in the inner loop (an aggregate GCC
//    may leave on the stack), and the GPR file (RM+RN running pointers plus counters against 14
//    registers). Arms B/C rewrite the tile with explicit scalars and no arrays.
//
// 2. WOULD __smlad HELP A FLOAT MATMUL? Not directly -- it is an integer instruction. But the reason
//    it did not pay for Q4_0 is the UNPACKING (12 setup instructions per 8 MACs). With int16 operands
//    already packed two-per-word there is none: one smlad IS two MACs. Arm D is that ceiling.
//   gcc -O2 -o bench31 bench31.c && ./bench31
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 704
#define N 75
#define K 176
static float  A[M*K], B[N*K], C[M*N];
static int16_t Ai[M*K], Bi[N*K];
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static void f32_11(void) {
    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        float s = 0; const float *a = A+(size_t)i*K, *b = B+(size_t)j*K;
        for (int l = 0; l < K; l++) s += a[l]*b[l];
        C[(size_t)i*N+j] = s; }
}
// 2x2, arrays (what bench30 measured at 153.7)
static void f32_22_arr(void) {
    for (int i = 0; i < M; i += 2) for (int j = 0; j < N-1; j += 2) {
        float s[2][2] = {{0,0},{0,0}};
        for (int l = 0; l < K; l++) { float av[2], bv[2];
            for (int x=0;x<2;x++) av[x]=A[(size_t)(i+x)*K+l];
            for (int y=0;y<2;y++) bv[y]=B[(size_t)(j+y)*K+l];
            for (int x=0;x<2;x++) for (int y=0;y<2;y++) s[x][y]+=av[x]*bv[y]; }
        for (int x=0;x<2;x++) for (int y=0;y<2;y++) C[(size_t)(i+x)*N+(j+y)]=s[x][y]; }
}
// 4x4, explicit scalars and running pointers -- no arrays, no index arithmetic in the loop
static void f32_44_scalar(void) {
    for (int i = 0; i < M-3; i += 4) for (int j = 0; j < N-3; j += 4) {
        const float *a0=A+(size_t)i*K, *a1=a0+K, *a2=a1+K, *a3=a2+K;
        const float *b0=B+(size_t)j*K, *b1=b0+K, *b2=b1+K, *b3=b2+K;
        float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0,
              c20=0,c21=0,c22=0,c23=0, c30=0,c31=0,c32=0,c33=0;
        for (int l = 0; l < K; l++) {
            const float x0=*a0++, x1=*a1++, x2=*a2++, x3=*a3++;
            const float y0=*b0++, y1=*b1++, y2=*b2++, y3=*b3++;
            c00+=x0*y0; c01+=x0*y1; c02+=x0*y2; c03+=x0*y3;
            c10+=x1*y0; c11+=x1*y1; c12+=x1*y2; c13+=x1*y3;
            c20+=x2*y0; c21+=x2*y1; c22+=x2*y2; c23+=x2*y3;
            c30+=x3*y0; c31+=x3*y1; c32+=x3*y2; c33+=x3*y3; }
        float *r0=C+(size_t)i*N+j, *r1=r0+N, *r2=r1+N, *r3=r2+N;
        r0[0]=c00;r0[1]=c01;r0[2]=c02;r0[3]=c03; r1[0]=c10;r1[1]=c11;r1[2]=c12;r1[3]=c13;
        r2[0]=c20;r2[1]=c21;r2[2]=c22;r2[3]=c23; r3[0]=c30;r3[1]=c31;r3[2]=c32;r3[3]=c33; }
}
// 2x2, explicit scalars
static void f32_22_scalar(void) {
    for (int i = 0; i < M-1; i += 2) for (int j = 0; j < N-1; j += 2) {
        const float *a0=A+(size_t)i*K, *a1=a0+K, *b0=B+(size_t)j*K, *b1=b0+K;
        float c00=0,c01=0,c10=0,c11=0;
        for (int l = 0; l < K; l++) { const float x0=*a0++, x1=*a1++, y0=*b0++, y1=*b1++;
            c00+=x0*y0; c01+=x0*y1; c10+=x1*y0; c11+=x1*y1; }
        C[(size_t)i*N+j]=c00; C[(size_t)i*N+j+1]=c01;
        C[(size_t)(i+1)*N+j]=c10; C[(size_t)(i+1)*N+j+1]=c11; }
}
// int16 x int16 -> int32 via __smlad, operands already packed two per word: no unpacking at all
static void i16_22_smlad(void) {
    for (int i = 0; i < M-1; i += 2) for (int j = 0; j < N-1; j += 2) {
        const uint32_t *a0=(const uint32_t*)(Ai+(size_t)i*K), *a1=(const uint32_t*)(Ai+(size_t)(i+1)*K);
        const uint32_t *b0=(const uint32_t*)(Bi+(size_t)j*K), *b1=(const uint32_t*)(Bi+(size_t)(j+1)*K);
        int32_t c00=0,c01=0,c10=0,c11=0;
        for (int l = 0; l < K/2; l++) { const uint32_t x0=*a0++, x1=*a1++, y0=*b0++, y1=*b1++;
            c00=__smlad(x0,y0,c00); c01=__smlad(x0,y1,c01);
            c10=__smlad(x1,y0,c10); c11=__smlad(x1,y1,c11); }
        C[(size_t)i*N+j]=c00*1e-6f; C[(size_t)i*N+j+1]=c01*1e-6f;
        C[(size_t)(i+1)*N+j]=c10*1e-6f; C[(size_t)(i+1)*N+j+1]=c11*1e-6f; }
}

int main(void) {
    srand(5);
    for (size_t i=0;i<(size_t)M*K;i++){ A[i]=(rand()%2001-1000)/1000.0f; Ai[i]=(int16_t)(A[i]*1000); }
    for (size_t i=0;i<(size_t)N*K;i++){ B[i]=(rand()%2001-1000)/1000.0f; Bi[i]=(int16_t)(B[i]*1000); }
    const double macs=(double)M*N*K; const int reps=20; double t;
    t=now(); for(int r=0;r<reps;r++) f32_11();        const double t1=now()-t; const float ref=C[0];
    t=now(); for(int r=0;r<reps;r++) f32_22_arr();    const double t2=now()-t; const int ok2=C[0]==ref;
    t=now(); for(int r=0;r<reps;r++) f32_22_scalar(); const double t3=now()-t; const int ok3=C[0]==ref;
    t=now(); for(int r=0;r<reps;r++) f32_44_scalar(); const double t4=now()-t; const int ok4=C[0]==ref;
    t=now(); for(int r=0;r<reps;r++) i16_22_smlad();  const double t5=now()-t;
    printf("m=%d n=%d k=%d\n\n", M,N,K);
    printf("F32 1x1  (ggml's shape)      %6.1f MMAC/s\n", macs*reps/t1/1e6);
    printf("F32 2x2  arrays              %6.1f MMAC/s  %.2fx  exact:%d\n", macs*reps/t2/1e6, t1/t2, ok2);
    printf("F32 2x2  explicit scalars    %6.1f MMAC/s  %.2fx  exact:%d\n", macs*reps/t3/1e6, t1/t3, ok3);
    printf("F32 4x4  explicit scalars    %6.1f MMAC/s  %.2fx  exact:%d\n", macs*reps/t4/1e6, t1/t4, ok4);
    printf("int16 2x2 __smlad, no unpack %6.1f MMAC/s  %.2fx\n", macs*reps/t5/1e6, t1/t5);
    return 0;
}
