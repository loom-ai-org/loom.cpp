// P7: the DIRECT CONVOLUTION tile on ARMv6, `float acc[4][4]` against explicit scalars. 22.5 vs
// 51.8 MMAC/s on a Pi Zero W, bit-identical -- the same stack-spill defect as bench33's, and the
// reason ggml_conv_1d_direct_tile_impl has an ARMv6 arm (loom's ggml-0019). Epic-08 6.6.
//   gcc -O2 -o bench34 bench34.c && ./bench34
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define IC 128
#define OC 128
#define KW 7
#define OL 2200
#define LP (OL + KW - 1)
#define OCB 4
#define VEC 4
static float xp[IC*LP], wp[IC*KW*OC], y[OC*OL];
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

// faithful copy of the generic arm in ggml_conv_1d_direct_tile_impl
static void tile_array(void) {
    for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB)
        for (int64_t p0 = 0; p0 + VEC <= OL; p0 += VEC) {
            float acc[OCB][VEC];
            for (int i=0;i<OCB;i++) for (int j=0;j<VEC;j++) acc[i][j]=0.0f;
            for (int64_t ic=0; ic<IC; ++ic)
                for (int64_t kx=0; kx<KW; ++kx)
                    for (int i=0;i<OCB;i++) {
                        const float w = wp[(ic*KW+kx)*OC + oc0 + i];
                        for (int j=0;j<VEC;j++) acc[i][j] += w * xp[ic*LP + p0 + j + kx];
                    }
            for (int i=0;i<OCB;i++) for (int j=0;j<VEC;j++) y[(oc0+i)*OL + p0 + j] = acc[i][j];
        }
}
// same arithmetic, explicit scalars
static void tile_scalar(void) {
    for (int64_t oc0 = 0; oc0 < OC; oc0 += 4)
        for (int64_t p0 = 0; p0 + 4 <= OL; p0 += 4) {
            float a00=0,a01=0,a02=0,a03=0,a10=0,a11=0,a12=0,a13=0,
                  a20=0,a21=0,a22=0,a23=0,a30=0,a31=0,a32=0,a33=0;
            for (int64_t ic=0; ic<IC; ++ic) {
                const float *x = xp + ic*LP + p0;
                const float *w = wp + ic*KW*OC + oc0;
                for (int64_t kx=0; kx<KW; ++kx, x++, w+=OC) {
                    const float x0=x[0], x1=x[1], x2=x[2], x3=x[3];
                    const float w0=w[0], w1=w[1], w2=w[2], w3=w[3];
                    a00+=w0*x0; a01+=w0*x1; a02+=w0*x2; a03+=w0*x3;
                    a10+=w1*x0; a11+=w1*x1; a12+=w1*x2; a13+=w1*x3;
                    a20+=w2*x0; a21+=w2*x1; a22+=w2*x2; a23+=w2*x3;
                    a30+=w3*x0; a31+=w3*x1; a32+=w3*x2; a33+=w3*x3; } }
            float *r0=y+(oc0+0)*OL+p0,*r1=y+(oc0+1)*OL+p0,*r2=y+(oc0+2)*OL+p0,*r3=y+(oc0+3)*OL+p0;
            r0[0]=a00;r0[1]=a01;r0[2]=a02;r0[3]=a03; r1[0]=a10;r1[1]=a11;r1[2]=a12;r1[3]=a13;
            r2[0]=a20;r2[1]=a21;r2[2]=a22;r2[3]=a23; r3[0]=a30;r3[1]=a31;r3[2]=a32;r3[3]=a33; }
}
int main(void){
    srand(9);
    for (size_t i=0;i<sizeof xp/sizeof*xp;i++) xp[i]=(rand()%2001-1000)/1000.0f;
    for (size_t i=0;i<sizeof wp/sizeof*wp;i++) wp[i]=(rand()%2001-1000)/1000.0f;
    const double macs=(double)OC*OL*IC*KW; const int reps=3; double t;
    t=now(); for(int r=0;r<reps;r++) tile_array();  const double ta=now()-t;
    float *ref=malloc(sizeof y); memcpy(ref,y,sizeof y);
    t=now(); for(int r=0;r<reps;r++) tile_scalar(); const double ts=now()-t;
    int ok = memcmp(ref,y,sizeof y)==0;
    printf("IC=%d OC=%d KW=%d OL=%d   %.1f MMAC per call   (profile says 43.0 MMAC/s here)\n\n",
           IC,OC,KW,OL, macs/1e6);
    printf("generic arm, float acc[4][4]  %6.1f MMAC/s\n", macs*reps/ta/1e6);
    printf("explicit scalars              %6.1f MMAC/s   %.2fx   identical:%d\n", macs*reps/ts/1e6, ta/ts, ok);
    return 0; }
