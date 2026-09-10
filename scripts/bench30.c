// P7: what a register-tiled F32 GEMM is worth on ARMv6, before one is written into ggml.
//
// conformer-ctc-small keeps 99.7% of its weights in F32 because d_model=176 and 32 does not divide it,
// so the q4_0 kernel (bench23-29) never sees them and `llamafile_sgemm`'s F32 case returns false here.
// Shape is its feed-forward: m=704 (linear1 out), k=176, n=75 frames for 3 s of audio.
//
// Unlike the integer case there is no wider instruction to reach for -- VFP is scalar, and bench26 put
// its multiply-add ceiling at 43.4 MMAC/s. All a tile can buy is load amortisation: 1x1 does two loads
// per MAC, RMxRN does (RM+RN) loads per RM*RN MACs.
//   gcc -O2 -o bench30 bench30.c && ./bench30
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 704
#define N 75
#define K 176
static float A[M*K], B[N*K], C[M*N];
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

// what ggml_vec_dot_f32's generic arm is with no GGML_SIMD
static void gemm11(void) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0; const float *a = A + (size_t)i*K, *b = B + (size_t)j*K;
            for (int l = 0; l < K; l++) s += a[l]*b[l];
            C[(size_t)i*N+j] = s;
        }
}
#define TILE(NAME, RM, RN)                                                            \
static void NAME(void) {                                                              \
    for (int i = 0; i < M; i += RM)                                                   \
        for (int j = 0; j < N - (RN-1); j += RN) {                                    \
            float s[RM][RN];                                                          \
            for (int x=0;x<RM;x++) for (int y=0;y<RN;y++) s[x][y]=0;                  \
            for (int l = 0; l < K; l++) {                                             \
                float av[RM], bv[RN];                                                 \
                for (int x=0;x<RM;x++) av[x] = A[(size_t)(i+x)*K + l];                \
                for (int y=0;y<RN;y++) bv[y] = B[(size_t)(j+y)*K + l];                \
                for (int x=0;x<RM;x++) for (int y=0;y<RN;y++) s[x][y] += av[x]*bv[y]; \
            }                                                                         \
            for (int x=0;x<RM;x++) for (int y=0;y<RN;y++) C[(size_t)(i+x)*N+(j+y)]=s[x][y]; \
        }                                                                             \
}
TILE(gemm22, 2, 2) TILE(gemm42, 4, 2) TILE(gemm44, 4, 4)

int main(void) {
    srand(5);
    for (size_t i=0;i<(size_t)M*K;i++) A[i]=(rand()%2001-1000)/1000.0f;
    for (size_t i=0;i<(size_t)N*K;i++) B[i]=(rand()%2001-1000)/1000.0f;
    const double macs = (double)M*N*K; const int reps = 20; double t;
    t=now(); for(int r=0;r<reps;r++) gemm11(); const double t1=now()-t; const float ref=C[0];
    t=now(); for(int r=0;r<reps;r++) gemm22(); const double t2=now()-t; const int ok2 = C[0]==ref;
    t=now(); for(int r=0;r<reps;r++) gemm42(); const double t4=now()-t; const int ok4 = C[0]==ref;
    t=now(); for(int r=0;r<reps;r++) gemm44(); const double t5=now()-t; const int ok5 = C[0]==ref;
    printf("m=%d n=%d k=%d   %.1f MMAC per call   (bench26's VFP MAC ceiling: 43.4 MMAC/s)\n\n", M,N,K, macs/1e6);
    printf("1x1  ggml's shape  %6.1f MMAC/s\n", macs*reps/t1/1e6);
    printf("2x2  tiled         %6.1f MMAC/s   %.2fx   same result: %d\n", macs*reps/t2/1e6, t1/t2, ok2);
    printf("4x2  tiled         %6.1f MMAC/s   %.2fx   same result: %d\n", macs*reps/t4/1e6, t1/t4, ok4);
    printf("4x4  tiled         %6.1f MMAC/s   %.2fx   same result: %d\n", macs*reps/t5/1e6, t1/t5, ok5);
    return 0;
}
