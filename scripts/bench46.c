// ADR-078 gate 3: does any libm function ggml's CPU ops call depend on IEEE mode beyond the flush itself?
//
// RunFast (FPSCR FZ | DN) is meant to change a result only where a denormal is involved: a denormal
// input reads as +0 and a denormal result becomes +0. A libm routine can do worse than that, because
// it is written for IEEE arithmetic: glibc's logf, for one, spots a subnormal argument by its bits
// and rescales it by 2^23 in floating point, which under FZ multiplies a zero. So every function the
// CPU ops call (counted in ggml-cpu: expf 45, sqrtf 15, powf 13, exp 10, roundf 8, erff 8, logf 7, ...)
// is run in both modes on operands that reach the denormal range from either side, and each RunFast
// result is checked against what the flush alone predicts:
//
//     expected = flush(f_ieee(flush(x)))     with flush(d) = +0 for a denormal d, and NaN as any NaN
//
// A mismatch that is only the sign of a zero is counted apart (the VFP11 flushes to +0, gate 1).
// Anything else is a dependence on IEEE mode, and is printed.
//
//   gcc -O2 -o bench46 scripts/bench46.c -lm      (on the board: this is about glibc there)
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RUNFAST ((1u << 24) | (1u << 25))
static inline unsigned fpscr_get(void) { unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); return f; }
static inline void fpscr_set(unsigned f) { __asm__ volatile("vmsr fpscr, %0" : : "r"(f) : "memory"); }

static uint64_t rng = 0x2545f4914f6cdd1dull;
static uint32_t next32(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }

static float f_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t u_of(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// Operands: denormals, the bottom binades, ordinary values, and the arguments that send exp/pow/erf
// results into the denormal range (exp(-87.4) is the last normal float, exp(-103.9) the last denormal).
static float draw(void) {
    const uint32_t s = next32() & 0x80000000u, m = next32() & 0x7fffffu;
    switch (next32() % 8) {
    case 0: return f_of(s | m);
    case 1: return f_of(s | ((1u + next32() % 24) << 23) | m);
    case 2: return -(80.0f + 30.0f * (next32() / 4294967296.0f));
    case 3: return f_of(s | ((100u + next32() % 54) << 23) | m);
    case 4: return 12.0f * (next32() / 4294967296.0f) - 6.0f;
    case 5: return f_of(s | ((60u + next32() % 40) << 23) | m);
    default: return f_of(next32());
    }
}

static float flush(float x) {
    const uint32_t u = u_of(x);
    return ((u & 0x7f800000u) == 0 && (u & 0x7fffffu)) ? 0.0f : x;
}

typedef float (*f1)(float);
static float t_powf(float x) { return powf(fabsf(x), 1.5f); }   // ggml: powf(x, p) on |x|-ish inputs
static float t_powf2(float x) { return powf(0.75f, x); }
static float t_exp(float x) { return (float)exp((double)x); }
static float t_log(float x) { return (float)log((double)x); }
static float t_erf(float x) { return (float)erf((double)x); }
static float t_sqrt(float x) { return (float)sqrt((double)x); }
static float t_expm1f(float x) { return expm1f(x); }

static const struct { const char* name; f1 fn; } fns[] = {
    {"expf", expf}, {"sqrtf", sqrtf}, {"powf(|x|,1.5)", t_powf}, {"powf(0.75,x)", t_powf2},
    {"exp", t_exp}, {"roundf", roundf}, {"erff", erff}, {"logf", logf}, {"tanhf", tanhf},
    {"sinf", sinf}, {"log", t_log}, {"floorf", floorf}, {"cosf", cosf}, {"expm1f", t_expm1f},
    {"erf", t_erf}, {"ceilf", ceilf}, {"sqrt", t_sqrt},
};

// The double-precision ones see their float argument widened by `vcvt.f64.f32`, which flushes a
// denormal float to +0 itself (bench45), so the same expectation holds for them.
int main(int argc, char** argv) {
    const long n = argc > 1 ? atol(argv[1]) : 100000;
    const unsigned base = fpscr_get() & ~RUNFAST;
    int bad = 0;
    printf("%-14s %8s %8s %8s %8s\n", "function", "cases", "same", "flushed", "unexpect");
    for (size_t k = 0; k < sizeof fns / sizeof fns[0]; ++k) {
        long same = 0, flushed = 0, sign0 = 0, unexpected = 0, shown = 0;
        for (long i = 0; i < n; ++i) {
            volatile float x = draw();
            fpscr_set(base);
            const float ieee = fns[k].fn(x);
            const float ieee_fl = fns[k].fn(flush(x));
            fpscr_set(base | RUNFAST);
            const float rf = fns[k].fn(x);
            fpscr_set(base);
            const float expect = flush(ieee_fl);
            const uint32_t ur = u_of(rf), ue = u_of(expect);
            if (u_of(rf) == u_of(ieee)) { ++same; continue; }
            if (ur == ue || (isnan(rf) && isnan(expect))) { ++flushed; continue; }
            if ((ur & 0x7fffffffu) == 0 && (ue & 0x7fffffffu) == 0) { ++sign0; ++flushed; continue; }
            ++unexpected;
            if (shown++ < 4)
                printf("    %s(%a [%08x]): ieee %a, ieee(flush x) %a, runfast %a [%08x]\n", fns[k].name, (double)x,
                       u_of(x), (double)ieee, (double)ieee_fl, (double)rf, ur);
        }
        printf("%-14s %8ld %8ld %8ld %8ld   (sign of zero only: %ld)\n", fns[k].name, n, same, flushed, unexpected, sign0);
        if (unexpected) bad = 1;
    }
    return bad;
}
