// ADR-079 P4.33 step 2, a diagnostic: WHERE THE INT8 GEMM TILE'S TIME GOES. bench60 measured the generalized
// tile at 0.78-0.90 GOP/s inside its driver, where ADR-077 gate 2 measured the same arithmetic at 1.95 on
// L1-resident operands. Each mode here adds one thing the driver does, so the step that costs is visible.
//
//   ./bench61 [calls]
//
//   l1          tile_p128_nopld on one 1 KB x chunk and one 2 KB w chunk, over and over
//   l1-2q       the same, alternating two w chunks (the driver's two quads a group)
//   x-stream    x walks a 4 MB buffer (each chunk read once), w in L1
//   x-l2        x walks a 64 KB buffer (the driver's column block, L2-resident), w in L1
//   w-l2        x in L1, w walks a 64 KB buffer
//   then l1 / x-stream / x-l2 again with the tiles that PLD x 64, 128 and 256 bytes ahead
//
//   gcc -O2 -march=armv6+fp -mfloat-abi=hard bench61.c tiles.S -o bench61     (tiles.S: gen_armv6_i8_gemm.py --pld)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

typedef void (*tile_fn)(const uint32_t* x, const uint32_t* w, int32_t* acc);
void ggml_armv6_i8_tile_p128_nopld(const uint32_t* x, const uint32_t* w, int32_t* acc);
void ggml_armv6_i8_tile_p128_pld64(const uint32_t* x, const uint32_t* w, int32_t* acc);
void ggml_armv6_i8_tile_p128(const uint32_t* x, const uint32_t* w, int32_t* acc);   // PLD 128 bytes ahead
void ggml_armv6_i8_tile_p128_pld256(const uint32_t* x, const uint32_t* w, int32_t* acc);
static tile_fn tile = ggml_armv6_i8_tile_p128_nopld;

static double run(const char* name, const uint32_t* xb, size_t xwords, const uint32_t* wb, size_t wwords, int calls) {
    int32_t acc[8] = { 0 };
    size_t xo = 0, wo = 0;
    double best = 1e9;
    for (int r = 0; r < 5; ++r) {
        const double t0 = now();
        for (int i = 0; i < calls; ++i) {
            tile(xb + xo, wb + wo, acc);
            xo += 256; if (xo + 256 > xwords) xo = 0;
            wo += 512; if (wo + 512 > wwords) wo = 0;
        }
        const double t = now() - t0;
        if (t < best) best = t;
    }
    const double gop = 2.0 * 8 * 256 * calls / best / 1e9;   // 128 pairs x 2 MACs x 8 accumulators
    printf("%-10s %8.3f ms  %.3f GOP/s  %.2f cycles/SMLAD at 1 GHz  [%d]\n", name, best * 1e3, gop,
           best * 1e9 / (128.0 * 8 * calls), acc[0]);
    return gop;
}

int main(int argc, char** argv) {
    const int calls = argc > 1 ? atoi(argv[1]) : 20000;
    uint32_t* big = aligned_alloc(32, 4u << 20);
    uint32_t* l2 = aligned_alloc(32, 64u << 10);
    for (size_t i = 0; i < (4u << 20) / 4; ++i) big[i] = (uint32_t) i * 2654435761u;
    for (size_t i = 0; i < (64u << 10) / 4; ++i) l2[i] = (uint32_t) i * 40503u;
    run("l1", l2, 256, l2 + 1024, 512, calls);
    run("l1-2q", l2, 256, l2 + 1024, 1024, calls);
    run("x-stream", big, (4u << 20) / 4, l2 + 1024, 512, calls);
    run("x-l2", l2, (64u << 10) / 4, big, 512, calls);
    run("w-l2", l2 + 8192, 256, l2, (64u << 10) / 4, calls);
    const tile_fn plds[] = { ggml_armv6_i8_tile_p128_pld64, ggml_armv6_i8_tile_p128, ggml_armv6_i8_tile_p128_pld256 };
    const char* names[] = { "pld 64", "pld 128", "pld 256" };
    for (int i = 0; i < 3; ++i) {
        tile = plds[i];
        printf("-- x prefetched, %s\n", names[i]);
        run("l1", l2, 256, l2 + 1024, 512, calls);
        run("x-stream", big, (4u << 20) / 4, l2 + 1024, 512, calls);
        run("x-l2", l2, (64u << 10) / 4, big, 512, calls);
    }
    return 0;
}
