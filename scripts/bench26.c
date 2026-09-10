// P7: what a MAC costs on an ARM1176, three ways -- VFP, scalar integer, and ARMv6's dual-16
// __smlad. Pi Zero W: 43.4 / 70.0 / 156.7 MMAC/s.
//
// READ bench30.c BEFORE QUOTING THE F32 NUMBER. This arm's 43.4 MMAC/s is an ARTEFACT of the loop,
// not a property of VFP: a real 2x2-tiled F32 GEMM on the same board reaches 153.7 MMAC/s. The
// '3.61x for integer' this file was used to argue is therefore wrong -- tiled int (167.9) and tiled
// float (153.7) are within 10% of each other on this core, because __smlad's two MACs per
// instruction are spent on the nibble unpacking around them. The smlad instructions ARE emitted
// (192 of them in the shipped libggml-cpu.so); they are just not where the win came from. The win
// was the register tile, in both kernels. Epic-08 6.5/6.6.
//
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }
int main(void) {
    const long it = 20000000L;
    { float a[8],b[8],c[8]; for (int i=0;i<8;i++){a[i]=1.0f+i;b[i]=1.000001f;c[i]=0.5f;}
      double t=now(); for(long n=0;n<it;n++) for(int i=0;i<8;i++) a[i]=a[i]*b[i]+c[i];
      double s=now()-t; printf("f32 VFP MAC        %7.1f MMAC/s   sink %g\n", 8.0*it/s/1e6, (double)(a[0]+a[7])); }
    { uint32_t a[8],b[8],c[8]; for (int i=0;i<8;i++){a[i]=1u+i;b[i]=3u;c[i]=1u;}
      double t=now(); for(long n=0;n<it;n++) for(int i=0;i<8;i++) a[i]=a[i]*b[i]+c[i];
      double s=now()-t; printf("i32 scalar MAC     %7.1f MMAC/s   sink %u\n", 8.0*it/s/1e6, a[0]+a[7]); }
    { uint32_t x[8],y[8]; int32_t acc[8]; for (int i=0;i<8;i++){x[i]=0x00020003u;y[i]=0x00040005u;acc[i]=0;}
      double t=now(); for(long n=0;n<it;n++) for(int i=0;i<8;i++) acc[i]=__smlad(x[i],y[i],acc[i]);
      double s=now()-t; printf("smlad 2x16 MAC     %7.1f MMAC/s   sink %d\n", 16.0*it/s/1e6, acc[0]+acc[7]); }
    return 0;
}
