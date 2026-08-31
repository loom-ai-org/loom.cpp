// P4.29: what a quantized direct convolution would newly cost, and nothing else.
//
// NOT part of the build -- a standalone measurement, kept because it is the number the whole item
// rests on and it has to stay re-runnable on the next machine.
//
//   gcc -O3 -I build/_deps/ggml-src/include scripts/bench21.c -o bench21 \
//       -L build/_deps/ggml-build/src -lggml -lggml-base -lggml-cpu -lm
//   ./bench21
//
// THE POINT. `ggml_conv_1d_direct_run` (ggml-0006) does not read the convolution kernel in its stored
// layout: it repacks the whole thing into an F32 scratch buffer `wp[(ic*KW + kx)*OC + oc]` first, and
// the register-tiled inner kernel reads only that. So making it accept a quantized kernel does not need
// a quantized inner loop -- it needs the pack loop to dequantize instead of copy. This measures that
// substitution at the size that matters: all 59.5 MB of a VITS vocoder's convolution weights, which is
// the whole model's per-synthesis packing rather than one convolution.
//
// What it prints is the dequantize against the plain F32 copy it would REPLACE, so the meaningful
// number is the delta, not the dequantize on its own. Measured on a Ryzen 3 3250U: Q4_0 10.3 ms against
// 8.5 ms, i.e. +1.8 ms on a ~500 ms synthesis, to recover ~160 ms. The quantized case reads LESS memory
// (8.4 MB rather than 59.5 MB) and writes the same, so the delta is arithmetic and not traffic.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ggml.h"
#include "ggml-cpu.h"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(void) {
    const size_t n = 59500000 / 4;   // VITS's convolution weights, as an F32 element count
    float * f32 = malloc(n * sizeof(float));
    float * out = malloc(n * sizeof(float));
    if (!f32 || !out) { fprintf(stderr, "oom\n"); return 1; }
    // Real-ish values rather than zeros: a quantizer given all-zero input picks a degenerate scale and
    // the dequantize then reads a block whose branch behaviour is not the one a model produces.
    for (size_t i = 0; i < n; ++i) f32[i] = (float) ((i % 251) - 125) * 0.01f;

    const enum ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 };
    for (int t = 0; t < 2; ++t) {
        const enum ggml_type ty = types[t];
        const struct ggml_type_traits * tr = ggml_get_type_traits(ty);
        void * q = malloc(ggml_row_size(ty, n));
        if (!q) { fprintf(stderr, "oom\n"); return 1; }
        ggml_quantize_chunk(ty, f32, q, 0, 1, n, NULL);

        // Best-of, and the two arms alternate inside the loop so they see the same cache state.
        double best_deq = 1e9, best_cpy = 1e9;
        for (int r = 0; r < 5; ++r) {
            const double t0 = now();
            tr->to_float(q, out, n);
            const double t1 = now();
            memcpy(out, f32, n * sizeof(float));
            const double t2 = now();
            if (t1 - t0 < best_deq) best_deq = t1 - t0;
            if (t2 - t1 < best_cpy) best_cpy = t2 - t1;
        }
        printf("%-6s  dequantize %6.1f ms (%5.2f GB/s out)   the F32 copy it replaces %6.1f ms   "
               "delta %+.1f ms\n",
               ggml_type_name(ty), best_deq * 1e3, n * 4.0 / best_deq / 1e9, best_cpy * 1e3,
               (best_deq - best_cpy) * 1e3);
        free(q);
    }
    printf("(59.5 MB of F32 conv weights; Q4_0 stores them in %.1f MB, Q8_0 in %.1f MB)\n",
           ggml_row_size(GGML_TYPE_Q4_0, n) / 1e6, ggml_row_size(GGML_TYPE_Q8_0, n) / 1e6);
    free(f32); free(out);
    return 0;
}
