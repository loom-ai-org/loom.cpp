// P7.1: CONV_TRANSPOSE_1D'S REMAINING DATA MOVEMENT -- is the overlap-add worth restructuring?
//
// After the panel skew, the op is 3235.7 ms of a 72 s synthesis and 545 ms of that is not the GEMM
// (loom scripts/bench35.c, on the board):
//
//     kernel repack    28.4 +  6.1 + 0.6 =  35.1 ms
//     src transpose     3.6 + 33.1 + 83.8 = 120.5 ms
//     overlap-add      27.0 +107.0 +255.2 = 389.2 ms   <- 71% of it
//
// The overlap-add is a scatter: for every (channel, position, tap) it does `out[l*s0 + i00] +=
// gemm[...]`, so each output element is READ, MODIFIED and WRITTEN once per contribution, and with
// K = 2*s0 -- which every one of these three nodes has -- that is twice.
//
// The gather does each output element once. Blocked by s0 it keeps both reads contiguous:
//
//     out[b*s0 + t] = gemm[b*mk + i1*K + t] + gemm[(b-1)*mk + i1*K + s0 + t]
//
// 12 bytes per output element against 24 per output element for the scatter, so the ceiling is 2x on
// this phase -- about 0.27% of a synthesis. This measures whether it is even that, before anything
// that touches an accumulation loop gets written.
//
//   gcc -O2 -o bench39 bench39.c && ./bench39
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static float *gemm, *dst, *ref;

/* as shipped: scatter-add, one tile of positions at a time */
static void scatter(int K, int Cout, int s0, int Tout, int L, int mk, int tile) {
    memset(dst, 0, (size_t)Tout*Cout*sizeof(float));
    for (int l0 = 0; l0 < L; l0 += tile) {
        const int nl = (l0 + tile > L) ? (L - l0) : tile;
        for (int i1 = 0; i1 < Cout; ++i1) {
            float *dd = dst + (size_t)i1*Tout;
            for (int l = 0; l < nl; ++l) {
                const float *s = gemm + (size_t)l*mk + (size_t)i1*K;
                float *o = dd + (size_t)(l0 + l)*s0;
                for (int i00 = 0; i00 < K; ++i00) o[i00] += s[i00];
            }
        }
    }
}

/* the gather, blocked by s0. Only valid for K == 2*s0, which is what these three nodes have; the
   general form sums K/s0 terms. Writes each output element exactly once. */
static void gather(int K, int Cout, int s0, int Tout, int L, int mk, int tile) {
    memset(dst, 0, (size_t)Tout*Cout*sizeof(float));
    for (int l0 = 0; l0 < L; l0 += tile) {
        const int nl = (l0 + tile > L) ? (L - l0) : tile;
        for (int i1 = 0; i1 < Cout; ++i1) {
            float *dd = dst + (size_t)i1*Tout;
            for (int l = 0; l < nl; ++l) {
                const int b = l0 + l;
                float *o = dd + (size_t)b*s0;
                const float *cur = gemm + (size_t)l*mk + (size_t)i1*K;
                if (l > 0) {
                    const float *prv = gemm + (size_t)(l-1)*mk + (size_t)i1*K + s0;
                    for (int t = 0; t < s0; ++t) o[t] = cur[t] + prv[t];
                } else {
                    for (int t = 0; t < s0; ++t) o[t] = cur[t];
                }
                /* the tail half of the last position in the tile has no successor yet */
                for (int t = s0; t < K; ++t) o[t] = cur[t];
            }
        }
    }
}

static void run(const char *name, int K, int Cout, int Cin, int s0, int L) {
    const int Tout = (L - 1)*s0 + K;
    const int mk   = Cout*K;
    int tile = 64*1024/mk; if (tile < 1) tile = 1; if (tile > L) tile = L;
    for (size_t i = 0; i < (size_t)tile*mk; ++i) gemm[i] = (float)(i % 11) * 0.01f;

    double t0 = now(); scatter(K, Cout, s0, Tout, L, mk, tile); double ts = now()-t0;
    t0 = now();        gather (K, Cout, s0, Tout, L, mk, tile); double tg = now()-t0;

    const double adds = (double)L*Cout*K/1e6;
    printf("  %-6s K=%2d Cout=%3d s0=%d L=%5d  %6.2f M contributions\n", name, K, Cout, s0, L, adds);
    printf("         scatter (shipped) %8.1f ms\n", ts*1e3);
    printf("         gather, blocked   %8.1f ms   %.2fx\n\n", tg*1e3, ts/tg);
}

int main(void) {
    gemm = malloc(sizeof(float)*64*1024);
    dst  = malloc(sizeof(float)*(size_t)70404*32);
    ref  = malloc(sizeof(float)*16);
    if (!gemm||!dst||!ref) { puts("alloc failed"); return 1; }
    puts("CONV_TRANSPOSE_1D's overlap-add, VITS's three nodes:\n");
    run("node1", 16, 128, 256, 8,   275);
    run("node2", 16,  64, 128, 8,  2200);
    run("node3",  8,  32,  64, 4, 17600);
    return 0;
}
