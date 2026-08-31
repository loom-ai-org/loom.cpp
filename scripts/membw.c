// The box's streaming memory bandwidth, measured, for the shape a vocoder's elementwise chain actually
// has: c[i] = a[i] + b[i] over a 8.9 MB tensor (two reads and a write, far past any cache).
//
// NOT part of the build. It exists because "loom's elementwise ops do not scale past one core" is only
// worth an item if that one core is not already at the bus limit -- and on a Raspberry Pi 4B it is:
// 4.56 GB/s at one thread, 4.13 at two, 3.64 at four, i.e. threading a streaming op there makes it
// SLOWER. That measurement is what closed the elementwise half of P4.16 (Epic-05 §5).
//
//   gcc -O3 -fopenmp -march=native scripts/membw.c -o membw && for t in 1 2 4; do ./membw $t; done
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
    long n=73216L*32;                       // exactly VITS's ADD 73216x32 bucket, one tensor
    int nt=argc>1?atoi(argv[1]):1, rep=30;
    float*a=aligned_alloc(64,n*4),*b=aligned_alloc(64,n*4),*c=aligned_alloc(64,n*4);
    for(long i=0;i<n;i++){a[i]=i;b[i]=2*i;c[i]=0;}
    double best=1e9;
    for(int r=0;r<rep;r++){
        double t0=now();
        #pragma omp parallel for num_threads(nt) schedule(static)
        for(long i=0;i<n;i++) c[i]=a[i]+b[i];
        double dt=now()-t0; if(dt<best)best=dt;
    }
    printf("threads %d   add over %.1f MB: %.3f ms   %.2f GB/s (2r+1w)\n",
           nt,n*4/1048576.0,best*1e3,3.0*n*4/best/1e9);
    if(c[7]<0)puts("");
    return 0;
}
