// Where does the ARM1176's VFP11 start bouncing a multiply to the kernel's support code? (ADR-078.)
//
// Retro-085 blamed Citrinet's F32 slowdown on products that underflow. But gate 3's flag census found
// only 5 of its 3,869 nodes raise even the underflow flag, and a weight of 1e-33 times an activation of
// ~1 is a normal float. This times `vmla.f32` (and `vmul.f32`) against the magnitude of the product, with
// every operand and every result normal, in IEEE mode and in RunFast. A bounce costs microseconds, so the
// threshold shows as a cliff.
//
//   gcc -O2 -o bench47 scripts/bench47.c     (on the board; qemu has no cliff to find)
#include <math.h>
#include <stdio.h>
#include <time.h>

#define RUNFAST ((1u << 24) | (1u << 25))
static inline unsigned fpscr_get(void) { unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); return f; }
static inline void fpscr_set(unsigned f) { __asm__ volatile("vmsr fpscr, %0" : : "r"(f) : "memory"); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

// n back-to-back vmla's: acc += a*b, with acc chosen so the sum stays at the product's scale.
static double time_vmla(float a, float b, float acc0, int n) {
    float acc = acc0;
    const double t = now();
    for (int i = 0; i < n; ++i) __asm__ volatile("vmla.f32 %0, %1, %2" : "+t"(acc) : "t"(a), "t"(b));
    const double dt = now() - t;
    __asm__ volatile("" : : "t"(acc));
    return dt / n * 1e9;
}
static double time_vmul(float a, float b, int n) {
    float r;
    const double t = now();
    for (int i = 0; i < n; ++i) __asm__ volatile("vmul.f32 %0, %1, %2" : "=t"(r) : "t"(a), "t"(b));
    const double dt = now() - t;
    __asm__ volatile("" : : "t"(r));
    return dt / n * 1e9;
}

// Pairs of vmla that add and then subtract the same product, so the accumulator stays at 2^e and every
// second instruction is an effective subtraction between two values of that size: the case gate 1's
// IEEE-mode mismatches all shared.
static double time_vmla_cancel(float acc0, float a, float b, int n) {
    float acc = acc0; const float nb = -b;
    const double t = now();
    for (int i = 0; i < n; ++i)
        __asm__ volatile("vmla.f32 %0, %1, %2\n\tvmla.f32 %0, %1, %3" : "+t"(acc) : "t"(a), "t"(b), "t"(nb));
    const double dt = now() - t;
    __asm__ volatile("" : : "t"(acc));
    return dt / (2.0 * n) * 1e9;
}

int main(void) {
    const unsigned base = fpscr_get() & ~RUNFAST;
    const int n = 20000;
    printf("ns per instruction; a = 2^e, b = 1.5 (so the product is 1.5 * 2^e, always normal)\n");
    printf("%6s %12s %12s %12s %12s\n", "e", "vmul ieee", "vmul rf", "vmla ieee", "vmla rf");
    for (int e = -126; e <= -20; e += 2) {
        const float a = ldexpf(1.0f, e), b = 1.5f;
        double r[4];
        for (int m = 0; m < 2; ++m) {
            fpscr_set(base | (m ? RUNFAST : 0));
            r[m] = time_vmul(a, b, n);
            r[2 + m] = time_vmla(a, b, 0.0f, n);
        }
        fpscr_set(base);
        printf("%6d %12.1f %12.1f %12.1f %12.1f\n", e, r[0], r[1], r[2], r[3]);
    }
    // The other operand order matters to a bounce rule built on exponents: a weight of 2^e times an
    // activation of 2^-k, with the product held at 2^-60 (normal by a wide margin).
    printf("\nproduct fixed at 2^-60: a = 2^e, b = 2^(-60-e)\n");
    for (int e = -100; e <= -30; e += 10) {
        const float a = ldexpf(1.0f, e), b = ldexpf(1.0f, -60 - e);
        fpscr_set(base);
        const double ie = time_vmla(a, b, 0.0f, n);
        fpscr_set(base | RUNFAST);
        const double rf = time_vmla(a, b, 0.0f, n);
        fpscr_set(base);
        printf("%6d %12.1f %12.1f\n", e, ie, rf);
    }
    printf("\nacc = 2^e, product = +-0.75 * 2^e alternately (half the vmla's are effective subtractions)\n");
    printf("%6s %12s %12s\n", "e", "vmla ieee", "vmla rf");
    for (int e = -126; e <= 0; e += (e >= -102 && e < -90) ? 1 : 4) {
        const float acc = ldexpf(1.0f, e), a = ldexpf(0.75f, e), b = 1.0f;
        fpscr_set(base);
        const double ie = time_vmla_cancel(acc, a, b, n);
        fpscr_set(base | RUNFAST);
        const double rf = time_vmla_cancel(acc, a, b, n);
        fpscr_set(base);
        printf("%6d %12.1f %12.1f\n", e, ie, rf);
    }
    // NaN and infinity operands: ggml-0029's softmax replaces a masked -inf before its exp because
    // inf - inf bounced there. Does RunFast (DN) keep these in hardware too?
    printf("\nNaN / infinity operands, vmla.f32\n");
    const float qnan = __builtin_nanf(""), inf = __builtin_inff();
    const struct { const char* what; float acc, a, b; } cases[] = {
        {"acc += 1 * 1 (control)", 1.0f, 1.0f, 1.0f}, {"acc = qNaN", qnan, 1.0f, 1.0f},
        {"a = qNaN", 1.0f, qnan, 1.0f}, {"acc = +inf, a*b = -inf", inf, -inf, 1.0f},
        {"acc = +inf", inf, 1.0f, 1.0f},
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; ++c) {
        fpscr_set(base);
        const double ie = time_vmla(cases[c].a, cases[c].b, cases[c].acc, n);
        fpscr_set(base | RUNFAST);
        const double rf = time_vmla(cases[c].a, cases[c].b, cases[c].acc, n);
        fpscr_set(base);
        printf("%-26s %10.1f %10.1f\n", cases[c].what, ie, rf);
    }
    return 0;
}
