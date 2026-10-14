// ADR-077: THE ARM1176'S TWO MAC CEILINGS, F32 ON THE VFP AGAINST INT8 ON SMLAD.
//
//   ./bench44 vfp 0     vmla.f32, 8 independent accumulators, operands in registers, IEEE mode
//   ./bench44 vfp 1     the same in RunFast (FPSCR FZ | DN; exceptions are off by default)
//   ./bench44 smlad 0   SMLAD (two 16x16 MACs), 8 independent accumulators, operands in registers
//   ./bench44 smlad 1   SMLAD with the two word loads per four SMLADs a dw/GEMM inner loop needs
//
// Pi Zero W, 2026-10-14 (second round; the first ran during the governor's clock ramp):
//
//     vfp 0     0.41 GMAC/s   2.4 ns per vmla -- IEEE mode does not pipeline back-to-back vmla
//     vfp 1     0.59-0.68     1.5-1.7 ns      -- RunFast: 1.65x on the same instruction stream
//     smlad 0   1.24-1.68
//     smlad 1   1.07          1.9 ns per smlad
//
// So int8's ceiling on this core is ~1.6x F32's in RunFast and ~2.6x in IEEE mode -- not the 4x a
// "two MACs per instruction" reading suggests (4 GOP/s at 1 GHz is an issue bound, not a rate).
//
//   gcc -O2 -o bench44 bench44.c        (in loom-armv6-build:bookworm: -march=armv6+fp, hard float)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void vfp(int mode) {

    if (mode) {
        unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); f |= (1u << 24) | (1u << 25);   // FZ | DN
        __asm__ volatile("vmsr fpscr, %0" : : "r"(f));
    }
    const long N = 50000000;
    float out[8], init[8] = {1, 0.5f, 1, 1, 1, 1, 1, 1};
    double t0 = now();
    __asm__ volatile(
        "vldmia %2, {s8-s15}\n vldr s0, [%2]\n vldr s1, [%2, #4]\n"
        "mov r4, %1\n"
        "1:\n"
        "vmla.f32 s8, s0, s1\n vmla.f32 s9, s0, s1\n vmla.f32 s10, s0, s1\n vmla.f32 s11, s0, s1\n"
        "vmla.f32 s12, s0, s1\n vmla.f32 s13, s0, s1\n vmla.f32 s14, s0, s1\n vmla.f32 s15, s0, s1\n"
        "subs r4, r4, #1\n bne 1b\n"
        "vstmia %0, {s8-s15}\n"
        : : "r"(out), "r"(N / 8), "r"(init) : "r4", "s0", "s1", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "memory", "cc");
    double dt = now() - t0;
    printf("runfast=%d  %.3f GMAC/s  (%.2f ns per vmla)  [%g]\n", mode, N / dt / 1e9, dt / N * 1e9, out[0]);
    }

static void smlad(int mode) {

        const long N = 80000000;                        // SMLADs
    static uint32_t buf[1024];
    for (int i = 0; i < 1024; ++i) buf[i] = 0x00010001u;
    uint32_t acc[8] = {0};
    double t0 = now();
    if (mode == 0) {
        __asm__ volatile(
            "mov r0, #0\n mov r1, #0\n mov r2, #0\n mov r3, #0\n mov r4, #0\n mov r5, #0\n mov r6, #0\n mov r7, #0\n"
            "ldr r8, =0x00030002\n ldr r9, =0x00050004\n mov r10, %1\n"
            "1:\n"
            "smlad r0, r8, r9, r0\n smlad r1, r8, r9, r1\n smlad r2, r8, r9, r2\n smlad r3, r8, r9, r3\n"
            "smlad r4, r8, r9, r4\n smlad r5, r8, r9, r5\n smlad r6, r8, r9, r6\n smlad r7, r8, r9, r7\n"
            "subs r10, r10, #1\n bne 1b\n"
            "stmia %0, {r0-r7}\n"
            : : "r"(acc), "r"(N / 8) : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "memory", "cc");
    } else {
        __asm__ volatile(
            "mov r0, #0\n mov r1, #0\n mov r2, #0\n mov r3, #0\n"
            "mov r10, %1\n"
            "2:\n mov r11, %2\n ldr r8, [r11], #4\n ldr r9, [r11], #4\n mov r12, #127\n"
            "1:\n"
            "smlad r0, r8, r9, r0\n ldr r4, [r11], #4\n smlad r1, r8, r9, r1\n ldr r5, [r11], #4\n"
            "smlad r2, r8, r9, r2\n smlad r3, r8, r9, r3\n"
            "smlad r0, r4, r5, r0\n ldr r8, [r11], #4\n smlad r1, r4, r5, r1\n ldr r9, [r11], #4\n"
            "smlad r2, r4, r5, r2\n smlad r3, r4, r5, r3\n"
            "subs r12, r12, #1\n bne 1b\n"
            "subs r10, r10, #1\n bne 2b\n"
            "stmia %0, {r0-r3}\n"
            : : "r"(acc), "r"(N / (8 * 127)), "r"(buf) : "r0", "r1", "r2", "r3", "r4", "r5", "r8", "r9", "r10", "r11", "r12", "memory", "cc");
    }
    double dt = now() - t0;
    printf("mode=%d  %.3f G-SMLAD/s = %.3f GMAC/s  (%.2f ns per smlad)  [%u]\n", mode, N / dt / 1e9, 2 * N / dt / 1e9, dt / N * 1e9, acc[0]);
    }

int main(int argc, char ** argv) {
    const int mode = argc > 2 ? atoi(argv[2]) : 0;
    if (argc > 1 && !strcmp(argv[1], "smlad")) smlad(mode); else vfp(mode);
    return 0;
}
