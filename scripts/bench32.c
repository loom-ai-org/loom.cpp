// P7: the right tile SHAPE for each ARMv6 kernel, now that bench31 showed the 4x4 collapse was
// arrays-and-index-arithmetic rather than register pressure. Two questions left:
//   * does int16 __smlad keep its 1.29x when the tile grows to 4x4, or does 2 MACs/instr spill first?
//   * does the shipped q4_0 kernel (arrays + A[lda*(ii+i)+l]) gain from the same rewrite, and at
//     which shape -- its unpack needs 4 live registers per ROW and 4 per COLUMN, so it may not reach
//     4x4 even written well.
// Shape: conformer's feed-forward, m=704 n=75 k=176.   gcc -O2 -o bench32 bench32.c && ./bench32
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 704
#define N 75
#define K 176
static int16_t Ai[M*K], Bi[N*K];
static float C[M*N];
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static void i16_22(void) {
    for (int i=0;i<M-1;i+=2) for (int j=0;j<N-1;j+=2) {
        const uint32_t *a0=(const uint32_t*)(Ai+(size_t)i*K),*a1=(const uint32_t*)(Ai+(size_t)(i+1)*K);
        const uint32_t *b0=(const uint32_t*)(Bi+(size_t)j*K),*b1=(const uint32_t*)(Bi+(size_t)(j+1)*K);
        int32_t c00=0,c01=0,c10=0,c11=0;
        for (int l=0;l<K/2;l++){ const uint32_t x0=*a0++,x1=*a1++,y0=*b0++,y1=*b1++;
            c00=__smlad(x0,y0,c00); c01=__smlad(x0,y1,c01);
            c10=__smlad(x1,y0,c10); c11=__smlad(x1,y1,c11); }
        float *r0=C+(size_t)i*N+j, *r1=r0+N;
        r0[0]=c00*1e-6f; r0[1]=c01*1e-6f; r1[0]=c10*1e-6f; r1[1]=c11*1e-6f; }
}
static void i16_42(void) {
    for (int i=0;i<M-3;i+=4) for (int j=0;j<N-1;j+=2) {
        const uint32_t *a0=(const uint32_t*)(Ai+(size_t)i*K),*a1=a0+K/2,*a2=a1+K/2,*a3=a2+K/2;
        const uint32_t *b0=(const uint32_t*)(Bi+(size_t)j*K),*b1=b0+K/2;
        int32_t c00=0,c01=0,c10=0,c11=0,c20=0,c21=0,c30=0,c31=0;
        for (int l=0;l<K/2;l++){ const uint32_t x0=*a0++,x1=*a1++,x2=*a2++,x3=*a3++,y0=*b0++,y1=*b1++;
            c00=__smlad(x0,y0,c00); c01=__smlad(x0,y1,c01); c10=__smlad(x1,y0,c10); c11=__smlad(x1,y1,c11);
            c20=__smlad(x2,y0,c20); c21=__smlad(x2,y1,c21); c30=__smlad(x3,y0,c30); c31=__smlad(x3,y1,c31); }
        float *r0=C+(size_t)i*N+j,*r1=r0+N,*r2=r1+N,*r3=r2+N;
        r0[0]=c00*1e-6f;r0[1]=c01*1e-6f; r1[0]=c10*1e-6f;r1[1]=c11*1e-6f;
        r2[0]=c20*1e-6f;r2[1]=c21*1e-6f; r3[0]=c30*1e-6f;r3[1]=c31*1e-6f; }
}
static void i16_44(void) {
    for (int i=0;i<M-3;i+=4) for (int j=0;j<N-3;j+=4) {
        const uint32_t *a0=(const uint32_t*)(Ai+(size_t)i*K),*a1=a0+K/2,*a2=a1+K/2,*a3=a2+K/2;
        const uint32_t *b0=(const uint32_t*)(Bi+(size_t)j*K),*b1=b0+K/2,*b2=b1+K/2,*b3=b2+K/2;
        int32_t c00=0,c01=0,c02=0,c03=0,c10=0,c11=0,c12=0,c13=0,
                c20=0,c21=0,c22=0,c23=0,c30=0,c31=0,c32=0,c33=0;
        for (int l=0;l<K/2;l++){
            const uint32_t x0=*a0++,x1=*a1++,x2=*a2++,x3=*a3++,y0=*b0++,y1=*b1++,y2=*b2++,y3=*b3++;
            c00=__smlad(x0,y0,c00);c01=__smlad(x0,y1,c01);c02=__smlad(x0,y2,c02);c03=__smlad(x0,y3,c03);
            c10=__smlad(x1,y0,c10);c11=__smlad(x1,y1,c11);c12=__smlad(x1,y2,c12);c13=__smlad(x1,y3,c13);
            c20=__smlad(x2,y0,c20);c21=__smlad(x2,y1,c21);c22=__smlad(x2,y2,c22);c23=__smlad(x2,y3,c23);
            c30=__smlad(x3,y0,c30);c31=__smlad(x3,y1,c31);c32=__smlad(x3,y2,c32);c33=__smlad(x3,y3,c33); }
        float *r0=C+(size_t)i*N+j,*r1=r0+N,*r2=r1+N,*r3=r2+N;
        r0[0]=c00*1e-6f;r0[1]=c01*1e-6f;r0[2]=c02*1e-6f;r0[3]=c03*1e-6f;
        r1[0]=c10*1e-6f;r1[1]=c11*1e-6f;r1[2]=c12*1e-6f;r1[3]=c13*1e-6f;
        r2[0]=c20*1e-6f;r2[1]=c21*1e-6f;r2[2]=c22*1e-6f;r2[3]=c23*1e-6f;
        r3[0]=c30*1e-6f;r3[1]=c31*1e-6f;r3[2]=c32*1e-6f;r3[3]=c33*1e-6f; }
}
int main(void){
    srand(5);
    for(size_t i=0;i<(size_t)M*K;i++) Ai[i]=(int16_t)(rand()%2001-1000);
    for(size_t i=0;i<(size_t)N*K;i++) Bi[i]=(int16_t)(rand()%2001-1000);
    const double macs=(double)M*N*K; const int reps=20; double t;
    t=now(); for(int r=0;r<reps;r++) i16_22(); const double t2=now()-t;
    t=now(); for(int r=0;r<reps;r++) i16_42(); const double t3=now()-t;
    t=now(); for(int r=0;r<reps;r++) i16_44(); const double t4=now()-t;
    printf("int16 __smlad 2x2  %6.1f MMAC/s\n", macs*reps/t2/1e6);
    printf("int16 __smlad 4x2  %6.1f MMAC/s\n", macs*reps/t3/1e6);
    printf("int16 __smlad 4x4  %6.1f MMAC/s\n", macs*reps/t4/1e6);
    double sum=0; for(size_t i=0;i<(size_t)M*N;i+=997) sum+=C[i];
    printf("\n(F32 4x4 explicit scalars, bench31: 226.7 MMAC/s)   checksum %.3f\n", sum);
    return 0; }
