// P7.1: THE PERMUTE-BACK AFTER THE im2col GEMM.
//
// The GEMM writes `gemm_output[patch, channel]` and `dst` wants `[channel, position]`, so the batch
// is transposed one element at a time with a write of stride `dst_w*dst_h` floats:
//
//     for i in patches:  for oc in channels:  dst[oc*OL + p] = gemm_output[i*c_out + oc] + bias
//
// Three arms. A is that. B walks channels on the outside so the WRITES are contiguous and the reads
// are strided instead. C is what the op could do instead of either: `ggml_call_mul_mat_ldc` can write
// C[ldc*col + row] directly, and with ldc = dst_w*dst_h that IS the destination layout -- ggml-0004's
// own header says so and the `defer` path uses it, but `defer` requires an aliasing input AND
// `patches_per_batch > (knl_w-1)*dilation`, which is false for every high-dilation convolution here
// (72 wanted, 32 available). So C measures only the contiguous bias pass that would remain.
//
//   gcc -O2 -o bench42 bench42.c && ./bench42
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static float *gemm_output, *dst, *bias;

/* as shipped: patch outer, channel inner, strided write */
static void permute_shipped(int64_t p0, int64_t patch_n, int64_t c_out, int64_t OL) {
    for (int64_t i = 0; i < patch_n; ++i) {
        const int64_t p = p0 + i;
        for (int64_t oc = 0; oc < c_out; ++oc) {
            dst[oc*OL + p] = gemm_output[i*c_out + oc] + bias[oc];
        }
    }
}

/* channel outer: contiguous writes, strided reads */
static void permute_channel_outer(int64_t p0, int64_t patch_n, int64_t c_out, int64_t OL) {
    for (int64_t oc = 0; oc < c_out; ++oc) {
        float * d = dst + oc*OL + p0;
        const float * s = gemm_output + oc;
        const float b = bias[oc];
        for (int64_t i = 0; i < patch_n; ++i) d[i] = s[i*c_out] + b;
    }
}

/* what remains if the GEMM writes channel-major straight into dst: a contiguous bias add */
static void bias_only(int64_t p0, int64_t patch_n, int64_t c_out, int64_t OL) {
    for (int64_t oc = 0; oc < c_out; ++oc) {
        float * d = dst + oc*OL + p0;
        const float b = bias[oc];
        for (int64_t i = 0; i < patch_n; ++i) d[i] += b;
    }
}

static void run(const char *name, int64_t OL, int64_t c_in, int64_t c_out, int64_t knl_w,
                int64_t calls) {
    const int64_t knl_n = c_in*knl_w;
    const int64_t sp    = knl_n*4 + c_out*4;
    int64_t bs = 65536/sp, ppb = bs > 8 ? (bs/8)*8 : bs;
    if (ppb > OL) ppb = OL;
    const int64_t batches = (OL + ppb - 1)/ppb;
    const double elems = (double)OL*c_out*calls/1e6;

    double t0 = now();
    for (int64_t b = 0; b < batches; ++b)
        permute_shipped(b*ppb, (b+1)*ppb <= OL ? ppb : OL - b*ppb, c_out, OL);
    const double ta = now() - t0;

    t0 = now();
    for (int64_t b = 0; b < batches; ++b)
        permute_channel_outer(b*ppb, (b+1)*ppb <= OL ? ppb : OL - b*ppb, c_out, OL);
    const double tb = now() - t0;

    t0 = now();
    for (int64_t b = 0; b < batches; ++b)
        bias_only(b*ppb, (b+1)*ppb <= OL ? ppb : OL - b*ppb, c_out, OL);
    const double tc = now() - t0;

    printf("  %-11s OL=%5lld c_out=%3lld ppb=%3lld  %6.2f M outputs over %lld calls\n",
           name, (long long)OL, (long long)c_out, (long long)ppb, elems, (long long)calls);
    printf("        A shipped (strided write) %7.1f ms/call  -> %5.2f s/synth\n", ta*1e3, ta*calls);
    printf("        B channel outer           %7.1f ms/call  -> %5.2f s/synth   %.2fx\n",
           tb*1e3, tb*calls, ta/tb);
    printf("        C bias only (no permute)  %7.1f ms/call  -> %5.2f s/synth   %.2fx\n\n",
           tc*1e3, tc*calls, ta/tc);
}

int main(void) {
    gemm_output = malloc(sizeof(float)*64*1024);
    dst  = malloc(sizeof(float)*(size_t)384*17600);
    bias = malloc(sizeof(float)*512);
    if (!gemm_output||!dst||!bias) { puts("alloc failed"); return 1; }
    for (size_t i=0;i<64*1024;++i) gemm_output[i]=(float)(i%11)*0.01f;
    for (size_t i=0;i<512;++i) bias[i]=0.5f;

    puts("The im2col permute-back, VITS's seven convolution shapes:\n");
    run("17600 k=7", 17600,  64,  64, 7, 2);
    run("17600 k=5", 17600,  64,  64, 5, 2);
    run("17600 k=3", 17600,  64,  64, 3, 2);
    run("2200  k=7",  2200, 128, 128, 7, 2);
    run("2200  k=3",  2200, 128, 128, 3, 2);
    run("275   k=5",   275, 192, 384, 5, 16);
    run("275   k=1",   275, 192, 384, 1, 12);
    return 0;
}
