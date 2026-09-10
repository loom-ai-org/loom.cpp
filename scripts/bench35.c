// P7.1: WHERE CONV_TRANSPOSE_1D'S TIME GOES ON ARMv6, phase by phase.
//
// The op is already a GEMM (loom's ggml-0009): it repacks the kernel, transposes the activation,
// calls mul_mat over tiles of input positions, and overlap-adds each tile into dst. Only the GEMM is
// arithmetic; the other three phases are pure data movement, and this measures them at VITS's three
// real shapes so the GEMM's share can be got by subtraction from a per-node profile.
//
// The shapes are read off the model, not invented -- kernel [K, Cout, Cin], stride s0:
//     node 1   K=16 Cout=128 Cin=256 s0=8  L=275      144.2 MMAC
//     node 2   K=16 Cout=64  Cin=128 s0=8  L=2200     288.4 MMAC
//     node 3   K=8  Cout=32  Cin=64  s0=4  L=17600    288.4 MMAC
//
// The suspicion this is written to test: node 1 repacks 2 MB of kernel with a 1 KB-strided scatter to
// feed a GEMM of only 144 MMAC, on a core with a 16 KB L1 -- the same "large weights, few positions"
// shape that made the direct-convolution predicate wrong here.
//
//   gcc -O2 -o bench35 bench35.c && ./bench35
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

#define TILE (64*1024)
#define MAXW (16*256*128)      /* nk, node 1 */
#define MAXS (17600*64)        /* L*Cin, node 3 */
#define MAXD (70404*32)        /* dst, node 3: T_out * Cout, the largest of the three */
#define MAXG (TILE)            /* gemm scratch, tile*mk <= TILE by construction */

static float *src0, *src1, *wdata, *wsrc, *gemm, *dst;

/* ggml_compute_forward_conv_transpose_1d_f32's kernel repack, verbatim in shape:
   (K x Cout x Cin) -> (Cin x K x Cout), one thread. */
static void repack_kernel(int K, int Cout, int Cin) {
    for (int i01 = 0; i01 < Cout; ++i01) {
        float *d = wdata + (long)i01*K*Cin;
        for (int i02 = 0; i02 < Cin; ++i02) {
            const float *s = src0 + (long)i02*Cout*K + (long)i01*K;
            for (int i00 = 0; i00 < K; ++i00) d[(long)i00*Cin + i02] = s[i00];
        }
    }
}

/* the activation transpose: (L x Cin) -> (Cin x L) */
static void transpose_src(int L, int Cin) {
    for (int i10 = 0; i10 < L; ++i10) {
        float *d = wsrc + (long)i10*Cin;
        for (int i11 = 0; i11 < Cin; ++i11) d[i11] = src1[(long)i11*L + i10];
    }
}

/* the overlap-add, over one tile of nl positions */
static void overlap_add(int K, int Cout, int s0, int Tout, int l0, int nl) {
    for (int i1 = 0; i1 < Cout; ++i1) {
        float *dd = dst + (long)i1*Tout;         /* dst is [T_out, Cout], row stride T_out */
        for (int l = 0; l < nl; ++l) {
            const float *s = gemm + (long)l*Cout*K + (long)i1*K;
            float *o = dd + (long)(l0 + l)*s0;
            for (int i00 = 0; i00 < K; ++i00) o[i00] += s[i00];
        }
    }
}

static void run(const char *name, int K, int Cout, int Cin, int s0, int L, double mmac) {
    const long nk = (long)K*Cout*Cin;
    const int  Tout = (L - 1)*s0 + K;
    const int  mk = Cout*K;
    const int  tile = TILE/mk < 1 ? 1 : (TILE/mk > L ? L : TILE/mk);

    for (long i = 0; i < nk; ++i)   src0[i] = (float)(i % 13) * 0.01f;
    for (long i = 0; i < (long)L*Cin; ++i) src1[i] = (float)(i % 7) * 0.01f;
    memset(dst, 0, sizeof(float)*(long)Tout*Cout);
    for (long i = 0; i < (long)tile*mk; ++i) gemm[i] = 0.001f;

    double t0 = now(); repack_kernel(K, Cout, Cin);          double t_k = now() - t0;
    t0 = now();        transpose_src(L, Cin);                double t_s = now() - t0;

    t0 = now();
    for (int l0 = 0; l0 < L; l0 += tile) {
        int nl = (l0 + tile > L) ? (L - l0) : tile;
        overlap_add(K, Cout, s0, Tout, l0, nl);
    }
    double t_o = now() - t0;

    printf("%-8s K=%2d Cout=%3d Cin=%3d s0=%d L=%5d  tile=%4d  %6.1f MMAC\n",
           name, K, Cout, Cin, s0, L, tile, mmac);
    printf("           kernel repack %7.1f ms   (%ld elems, stride %d floats)\n",
           t_k*1e3, nk, Cin);
    printf("           src transpose %7.1f ms   (%ld elems)\n", t_s*1e3, (long)L*Cin);
    printf("           overlap-add   %7.1f ms   (%ld adds)\n", t_o*1e3, (long)L*Cout*K);
    printf("           non-GEMM total%7.1f ms   | GEMM at 226.7 MMAC/s would be %6.1f ms\n\n",
           (t_k+t_s+t_o)*1e3, mmac/226.7*1e3);
}

int main(void) {
    src0  = malloc(sizeof(float)*MAXW);
    src1  = malloc(sizeof(float)*MAXS);
    wdata = malloc(sizeof(float)*MAXW);
    wsrc  = malloc(sizeof(float)*MAXS);
    gemm  = malloc(sizeof(float)*MAXG);
    dst   = malloc(sizeof(float)*MAXD);
    if (!src0||!src1||!wdata||!wsrc||!gemm||!dst) { puts("alloc failed"); return 1; }

    run("node1", 16, 128, 256, 8,   275, 144.2);
    run("node2", 16,  64, 128, 8,  2200, 288.4);
    run("node3",  8,  32,  64, 4, 17600, 288.4);
    return 0;
}
