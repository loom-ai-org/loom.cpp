// P7: does the TEMPLATE form (arrays + running pointers) reach the explicit-scalar 4x4 number?
// It does not -- 27.2 against 173.9 MMAC/s on a Pi Zero W -- which is why tinyBLAS_F32_ARMV6's 4x4
// case is written out longhand in llamafile/sgemm.cpp instead of instantiated. Epic-08 6.6.
//   gcc -O2 -o bench33 bench33.c && ./bench33
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 704
#define N 75
#define K 176
static float A[M*K], B[N*K], C[M*N];
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

template <int RM, int RN>
static void tmpl(void) {
    for (int i = 0; i <= M-RM; i += RM) for (int j = 0; j <= N-RN; j += RN) {
        const float *ap[RM]; const float *bp[RN];
        for (int x=0;x<RM;x++) ap[x]=A+(size_t)(i+x)*K;
        for (int y=0;y<RN;y++) bp[y]=B+(size_t)(j+y)*K;
        float acc[RN][RM] = {};
        for (int l=0;l<K;l++) { float av[RM], bv[RN];
            for (int x=0;x<RM;x++) av[x]=*ap[x]++;
            for (int y=0;y<RN;y++) bv[y]=*bp[y]++;
            for (int y=0;y<RN;y++) for (int x=0;x<RM;x++) acc[y][x]+=av[x]*bv[y]; }
        for (int y=0;y<RN;y++) for (int x=0;x<RM;x++) C[(size_t)(i+x)*N+(j+y)]=acc[y][x]; }
}
static void scalar44(void) {
    for (int i=0;i<=M-4;i+=4) for (int j=0;j<=N-4;j+=4) {
        const float *a0=A+(size_t)i*K,*a1=a0+K,*a2=a1+K,*a3=a2+K;
        const float *b0=B+(size_t)j*K,*b1=b0+K,*b2=b1+K,*b3=b2+K;
        float c00=0,c01=0,c02=0,c03=0,c10=0,c11=0,c12=0,c13=0,
              c20=0,c21=0,c22=0,c23=0,c30=0,c31=0,c32=0,c33=0;
        for (int l=0;l<K;l++){ const float x0=*a0++,x1=*a1++,x2=*a2++,x3=*a3++;
            const float y0=*b0++,y1=*b1++,y2=*b2++,y3=*b3++;
            c00+=x0*y0;c01+=x0*y1;c02+=x0*y2;c03+=x0*y3; c10+=x1*y0;c11+=x1*y1;c12+=x1*y2;c13+=x1*y3;
            c20+=x2*y0;c21+=x2*y1;c22+=x2*y2;c23+=x2*y3; c30+=x3*y0;c31+=x3*y1;c32+=x3*y2;c33+=x3*y3; }
        float *r0=C+(size_t)i*N+j,*r1=r0+N,*r2=r1+N,*r3=r2+N;
        r0[0]=c00;r0[1]=c01;r0[2]=c02;r0[3]=c03; r1[0]=c10;r1[1]=c11;r1[2]=c12;r1[3]=c13;
        r2[0]=c20;r2[1]=c21;r2[2]=c22;r2[3]=c23; r3[0]=c30;r3[1]=c31;r3[2]=c32;r3[3]=c33; }
}
int main(void){
    srand(5);
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=(rand()%2001-1000)/1000.0f;
    for (size_t i=0;i<(size_t)N*K;i++) B[i]=(rand()%2001-1000)/1000.0f;
    const double macs=(double)M*N*K; const int reps=20; double t;
    t=now(); for(int r=0;r<reps;r++) scalar44();   const double ts=now()-t; const float ref=C[0];
    t=now(); for(int r=0;r<reps;r++) tmpl<4,4>();  const double t44=now()-t; const int ok44=C[0]==ref;
    t=now(); for(int r=0;r<reps;r++) tmpl<2,2>();  const double t22=now()-t; const int ok22=C[0]==ref;
    t=now(); for(int r=0;r<reps;r++) tmpl<4,2>();  const double t42=now()-t; const int ok42=C[0]==ref;
    printf("explicit scalars 4x4        %6.1f MMAC/s\n", macs*reps/ts/1e6);
    printf("template arrays+ptrs 4x4    %6.1f MMAC/s   exact:%d\n", macs*reps/t44/1e6, ok44);
    printf("template arrays+ptrs 4x2    %6.1f MMAC/s   exact:%d\n", macs*reps/t42/1e6, ok42);
    printf("template arrays+ptrs 2x2    %6.1f MMAC/s   exact:%d\n", macs*reps/t22/1e6, ok22);
    return 0; }
