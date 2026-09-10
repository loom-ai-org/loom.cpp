// P7.1: HOW MUCH OF THE im2col BUCKETS IS THE PATCH GATHER?
//
// The three im2col buckets are 38.5 s of a 72 s synthesis and an idealised 4x4 GEMM at their exact
// per-batch shapes accounts for 28.1 s of it (loom scripts/bench40.c). This measures the other half's
// leading candidate: the gather that builds each patch, reproduced exactly as
// ggml_compute_forward_conv_2d_impl writes it -- one element at a time, with the bounds test and the
// three-term address computation per element.
//
// Arm B is the specialisation the 1-D case allows. With knl_h == 1, stride 1 and a contiguous source,
// the elements a fixed (ic, kx) contributes across consecutive patches are a CONTIGUOUS RUN of the
// input, and the only thing that varies is where they land: a strided store of stride knl_n. The
// bounds test becomes two ends and a body, computed once per run rather than once per element.
//
//   gcc -O2 -o bench41 bench41.c && ./bench41
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static float *src, *tmp;

/* verbatim shape of the shipped gather, for one batch of `ppb` patches */
static void gather_generic(int64_t p0, int64_t ppb, int64_t c_in, int64_t knl_w, int64_t knl_n,
                           int64_t dil, int64_t pad, int64_t src_w) {
    for (int64_t p = p0; p < p0 + ppb; ++p) {
        float * dst_row = tmp + (p % ppb) * knl_n;
        for (int64_t ic = 0; ic < c_in; ++ic) {
            for (int64_t kx = 0; kx < knl_w; ++kx) {
                const int64_t sx = p + kx*dil - pad;
                const int64_t dst_idx = ic*knl_w + kx;
                float v;
                if (sx < 0 || sx >= src_w) v = 0.0f;
                else v = src[ic*src_w + sx];
                dst_row[dst_idx] = v;
            }
        }
    }
}

/* the run-at-a-time form: contiguous read, strided store, bounds resolved per run */
static void gather_runs(int64_t p0, int64_t ppb, int64_t c_in, int64_t knl_w, int64_t knl_n,
                        int64_t dil, int64_t pad, int64_t src_w) {
    for (int64_t ic = 0; ic < c_in; ++ic) {
        const float * s = src + ic*src_w;
        for (int64_t kx = 0; kx < knl_w; ++kx) {
            float * d = tmp + ic*knl_w + kx;
            const int64_t off = kx*dil - pad;
            /* p ranges over [p0, p0+ppb); sx = p + off must lie in [0, src_w) */
            const int64_t pe = p0 + ppb;
            int64_t lo = -off;         if (lo < p0) lo = p0;  if (lo > pe) lo = pe;
            int64_t hi = src_w - off;  if (hi > pe) hi = pe;   if (hi < lo) hi = lo;
            for (int64_t p = p0; p < lo;  ++p) d[(p - p0)*knl_n] = 0.0f;
            for (int64_t p = lo;  p < hi; ++p) d[(p - p0)*knl_n] = s[p + off];
            for (int64_t p = hi;  p < pe; ++p) d[(p - p0)*knl_n] = 0.0f;
        }
    }
}

static void run(const char *name, int64_t OL, int64_t c_in, int64_t c_out, int64_t knl_w,
                int64_t dil, int64_t calls) {
    const int64_t knl_n = c_in*knl_w;
    const int64_t pad   = (knl_w - 1)*dil/2;
    const int64_t sp    = knl_n*4 + c_out*4;
    int64_t bs = 65536/sp, ppb = bs > 8 ? (bs/8)*8 : bs;
    if (ppb > OL) ppb = OL;
    const int64_t batches = (OL + ppb - 1)/ppb;
    const double elems = (double)OL*knl_n*calls/1e6;

    double t0 = now();
    for (int64_t b = 0; b < batches; ++b) gather_generic(b*ppb, (b+1)*ppb <= OL ? ppb : OL - b*ppb,
                                                         c_in, knl_w, knl_n, dil, pad, OL);
    const double tg = now() - t0;
    t0 = now();
    for (int64_t b = 0; b < batches; ++b) gather_runs(b*ppb, (b+1)*ppb <= OL ? ppb : OL - b*ppb,
                                                      c_in, knl_w, knl_n, dil, pad, OL);
    const double tr = now() - t0;

    printf("  %-11s knl_n=%4lld ppb=%3lld batches=%4lld  %6.2f M elements over %lld calls\n",
           name, (long long)knl_n, (long long)ppb, (long long)batches, elems, (long long)calls);
    printf("        generic (shipped) %7.1f ms/call  %5.1f ns/elem  -> %5.2f s/synth\n",
           tg*1e3, tg*1e9/(OL*knl_n), tg*calls);
    printf("        run-at-a-time     %7.1f ms/call  %5.1f ns/elem  -> %5.2f s/synth   %.2fx\n\n",
           tr*1e3, tr*1e9/(OL*knl_n), tr*calls, tg/tr);
}

int main(void) {
    src = malloc(sizeof(float)*(size_t)192*17600);
    tmp = malloc(sizeof(float)*(size_t)64*1024);
    if (!src||!tmp) { puts("alloc failed"); return 1; }
    for (size_t i=0;i<(size_t)192*17600;++i) src[i]=(float)(i%13)*0.01f;

    puts("The im2col patch gather, VITS's seven convolution shapes:\n");
    run("17600 k=7", 17600,  64,  64, 7, 12, 2);
    run("17600 k=5", 17600,  64,  64, 5,  6, 2);
    run("17600 k=3", 17600,  64,  64, 3,  2, 2);
    run("2200  k=7",  2200, 128, 128, 7, 12, 2);
    run("2200  k=3",  2200, 128, 128, 3,  2, 2);
    run("275   k=5",   275, 192, 384, 5,  1, 16);
    run("275   k=1",   275, 192, 384, 1,  1, 12);
    return 0;
}
