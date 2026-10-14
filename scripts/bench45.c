// ADR-078 gate 1: does qemu-user honour the VFP's RunFast mode (FPSCR FZ | DN) bit for bit, as the
// ARM1176 does? The ARMv6 numerics gate runs in the emulated build container because qemu was
// calibrated bit-identical to the board in IEEE mode (env note, P7.1's VITS hash). That calibration
// says nothing about FZ/DN, which qemu implements in its own softfloat flags, so it is checked here.
//
// Every VFP instruction loom's ARMv6 build can emit on a float or double is run, through inline asm so
// the compiler can neither fold nor fuse it, on operands drawn to land in the cases RunFast changes:
// denormal inputs, products and sums that underflow, quiet and signalling NaNs with payloads, and
// the ordinary values around them. Each mode prints one hash over the result bits and one over the
// FPSCR cumulative flags, per instruction. The emulator agrees when the two runs print the same lines.
//
// The test can fail: each instruction also prints how many results and how many flag sets differ
// between IEEE and RunFast in the same run, since a mode that changed nothing would make the
// comparison vacuous (ADR-015).
//
//   gcc -O2 -o bench45 scripts/bench45.c      (inside loom-armv6-build:bookworm, and on the board)
//   ./bench45 [cases per instruction, default 200000] [dump file]
// The dump holds, per instruction and case, the three operands and each mode's result and flags, so
// two runs that disagree can be compared case by case (scripts/bench45_diff.py).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FZ (1u << 24)
#define DN (1u << 25)
#define FLAGS 0x9fu  // IDC IXC UFC OFC DZC IOC

static inline unsigned fpscr_get(void) { unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); return f; }
static inline void fpscr_set(unsigned f) { __asm__ volatile("vmsr fpscr, %0" : : "r"(f) : "memory"); }

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t next32(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

// A float drawn to sit near the cases FZ and DN change. Exponents cluster at the bottom of the range,
// so products of two of them underflow, and at 1 so sums against them are ordinary.
static uint32_t draw_f32(void) {
    uint32_t r = next32(), s = r & 0x80000000u, m = next32() & 0x7fffffu;
    switch (next32() % 10) {
    case 0: return s | m;                                            // denormal (or zero)
    case 1: return s | (1u << 23) | m;                               // smallest normal binade
    case 2: return s | ((1u + next32() % 40) << 23) | m;             // 2^-126 .. 2^-87
    case 3: return s | ((50u + next32() % 30) << 23) | m;            // ~1e-23 .. 1e-14: products underflow
    case 4: return s | ((127u - 8 + next32() % 16) << 23) | m;       // around 1
    case 5: return s | ((200u + next32() % 54) << 23) | m;           // large: products overflow
    case 6: return s | 0x7fc00000u | (m & 0x3fffffu);                // quiet NaN with a payload
    case 7: return s | 0x7f800000u | ((m & 0x3fffffu) | 1u);         // signalling NaN with a payload
    case 8: return (next32() & 1) ? (s | 0x7f800000u) : s;           // +-inf, +-0
    default: return next32();                                        // anything
    }
}

static uint64_t draw_f64(void) {
    uint64_t s = (uint64_t)(next32() & 1) << 63, m = ((uint64_t)next32() << 32 | next32()) & 0xfffffffffffffull;
    switch (next32() % 8) {
    case 0: return s | m;
    case 1: return s | (1ull << 52) | m;
    case 2: return s | ((uint64_t)(400 + next32() % 200) << 52) | m;  // products underflow
    case 3: return s | ((uint64_t)(1023 - 8 + next32() % 16) << 52) | m;
    case 4: return s | 0x7ff8000000000000ull | (m & 0x7ffffffffffffull);
    case 5: return s | 0x7ff0000000000000ull | ((m & 0x7ffffffffffffull) | 1);
    case 6: return (next32() & 1) ? (s | 0x7ff0000000000000ull) : s;
    default: return (uint64_t)next32() << 32 | next32();
    }
}

static float f32_of(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static double f64_of(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }

// One instruction on (a, b, c); the result's bits come back as 64 bits either way.
typedef uint64_t (*op_fn)(uint64_t a, uint64_t b, uint64_t c);
#define F32OP(name, body) \
    static uint64_t name(uint64_t ua, uint64_t ub, uint64_t uc) { \
        float a = f32_of((uint32_t)ua), b = f32_of((uint32_t)ub), d = f32_of((uint32_t)uc); \
        (void)a; (void)b; (void)d; body; uint32_t r; memcpy(&r, &d, 4); return r; }
#define F64OP(name, body) \
    static uint64_t name(uint64_t ua, uint64_t ub, uint64_t uc) { \
        double a = f64_of(ua), b = f64_of(ub), d = f64_of(uc); \
        (void)a; (void)b; (void)d; body; uint64_t r; memcpy(&r, &d, 8); return r; }

F32OP(vadd_f32, __asm__ volatile("vadd.f32 %0, %1, %2" : "=t"(d) : "t"(a), "t"(b)))
F32OP(vsub_f32, __asm__ volatile("vsub.f32 %0, %1, %2" : "=t"(d) : "t"(a), "t"(b)))
F32OP(vmul_f32, __asm__ volatile("vmul.f32 %0, %1, %2" : "=t"(d) : "t"(a), "t"(b)))
F32OP(vnmul_f32, __asm__ volatile("vnmul.f32 %0, %1, %2" : "=t"(d) : "t"(a), "t"(b)))
F32OP(vdiv_f32, __asm__ volatile("vdiv.f32 %0, %1, %2" : "=t"(d) : "t"(a), "t"(b)))
F32OP(vmla_f32, __asm__ volatile("vmla.f32 %0, %1, %2" : "+t"(d) : "t"(a), "t"(b)))
F32OP(vmls_f32, __asm__ volatile("vmls.f32 %0, %1, %2" : "+t"(d) : "t"(a), "t"(b)))
F32OP(vnmla_f32, __asm__ volatile("vnmla.f32 %0, %1, %2" : "+t"(d) : "t"(a), "t"(b)))
F32OP(vsqrt_f32, __asm__ volatile("vsqrt.f32 %0, %1" : "=t"(d) : "t"(a)))
F32OP(vabs_f32, __asm__ volatile("vabs.f32 %0, %1" : "=t"(d) : "t"(a)))
F32OP(vneg_f32, __asm__ volatile("vneg.f32 %0, %1" : "=t"(d) : "t"(a)))
F32OP(vcvtr_s32, __asm__ volatile("vcvtr.s32.f32 %0, %1" : "=t"(d) : "t"(a)))
F32OP(vcvt_s32, __asm__ volatile("vcvt.s32.f32 %0, %1" : "=t"(d) : "t"(a)))
F32OP(vcmp_f32, { unsigned n; __asm__ volatile("vcmp.f32 %1, %2\n\tvmrs %0, fpscr" : "=r"(n) : "t"(a), "t"(b)); uint32_t u = n >> 28; memcpy(&d, &u, 4); })
F32OP(vcmpe_f32, { unsigned n; __asm__ volatile("vcmpe.f32 %1, %2\n\tvmrs %0, fpscr" : "=r"(n) : "t"(a), "t"(b)); uint32_t u = n >> 28; memcpy(&d, &u, 4); })
F32OP(vcvt_f64_f32, { double w; __asm__ volatile("vcvt.f64.f32 %P0, %1" : "=w"(w) : "t"(a)); uint64_t u; memcpy(&u, &w, 8); uint32_t h = (uint32_t)(u ^ (u >> 32)) * 2654435761u + (uint32_t)(u >> 32); memcpy(&d, &h, 4); })
F64OP(vcvt_f32_f64, { float n; __asm__ volatile("vcvt.f32.f64 %0, %P1" : "=t"(n) : "w"(a)); uint32_t u; memcpy(&u, &n, 4); uint64_t w = u; memcpy(&d, &w, 8); })
F64OP(vadd_f64, __asm__ volatile("vadd.f64 %P0, %P1, %P2" : "=w"(d) : "w"(a), "w"(b)))
F64OP(vmul_f64, __asm__ volatile("vmul.f64 %P0, %P1, %P2" : "=w"(d) : "w"(a), "w"(b)))
F64OP(vdiv_f64, __asm__ volatile("vdiv.f64 %P0, %P1, %P2" : "=w"(d) : "w"(a), "w"(b)))
F64OP(vmla_f64, __asm__ volatile("vmla.f64 %P0, %P1, %P2" : "+w"(d) : "w"(a), "w"(b)))
F64OP(vsqrt_f64, __asm__ volatile("vsqrt.f64 %P0, %P1" : "=w"(d) : "w"(a)))

static const struct { const char* name; op_fn fn; int f64; } ops[] = {
    {"vadd.f32", vadd_f32, 0}, {"vsub.f32", vsub_f32, 0}, {"vmul.f32", vmul_f32, 0},
    {"vnmul.f32", vnmul_f32, 0}, {"vdiv.f32", vdiv_f32, 0}, {"vmla.f32", vmla_f32, 0},
    {"vmls.f32", vmls_f32, 0}, {"vnmla.f32", vnmla_f32, 0}, {"vsqrt.f32", vsqrt_f32, 0},
    {"vabs.f32", vabs_f32, 0}, {"vneg.f32", vneg_f32, 0}, {"vcvtr.s32.f32", vcvtr_s32, 0},
    {"vcvt.s32.f32", vcvt_s32, 0}, {"vcmp.f32", vcmp_f32, 0}, {"vcmpe.f32", vcmpe_f32, 0},
    {"vcvt.f64.f32", vcvt_f64_f32, 0}, {"vcvt.f32.f64", vcvt_f32_f64, 1}, {"vadd.f64", vadd_f64, 1},
    {"vmul.f64", vmul_f64, 1}, {"vdiv.f64", vdiv_f64, 1}, {"vmla.f64", vmla_f64, 1},
    {"vsqrt.f64", vsqrt_f64, 1},
};

static uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull; }

int main(int argc, char** argv) {
    const long n = argc > 1 ? atol(argv[1]) : 200000;
    const unsigned base = fpscr_get() & ~(FZ | DN | FLAGS);
    uint64_t* res[2]; unsigned* flg[2];
    for (int m = 0; m < 2; ++m) { res[m] = malloc(n * 8); flg[m] = malloc(n * sizeof(unsigned)); }
    uint64_t* in = malloc(n * 3 * 8);
    int bad = 0;
    FILE* dump = argc > 2 ? fopen(argv[2], "wb") : NULL;
    printf("cases=%ld fpscr=%#x\n", n, fpscr_get());
    printf("%-14s %18s %18s %18s %18s %8s %8s\n", "insn", "ieee results", "ieee flags", "runfast results",
           "runfast flags", "changed", "flags");
    for (size_t o = 0; o < sizeof ops / sizeof ops[0]; ++o) {
        for (long i = 0; i < 3 * n; ++i) in[i] = ops[o].f64 ? draw_f64() : draw_f32();
        uint64_t hr[2], hf[2];
        for (int m = 0; m < 2; ++m) {
            const unsigned mode = base | (m ? FZ | DN : 0);
            for (long i = 0; i < n; ++i) {
                fpscr_set(mode);
                res[m][i] = ops[o].fn(in[3 * i], in[3 * i + 1], in[3 * i + 2]);
                flg[m][i] = fpscr_get() & FLAGS;
            }
            fpscr_set(base);
            hr[m] = hf[m] = 0xcbf29ce484222325ull;
            for (long i = 0; i < n; ++i) { hr[m] = mix(hr[m], res[m][i]); hf[m] = mix(hf[m], flg[m][i]); }
        }
        if (dump) {
            fwrite(in, 8, 3 * n, dump);
            for (int m = 0; m < 2; ++m) { fwrite(res[m], 8, n, dump); fwrite(flg[m], sizeof(unsigned), n, dump); }
        }
        long changed = 0, fchanged = 0;
        for (long i = 0; i < n; ++i) { changed += res[0][i] != res[1][i]; fchanged += flg[0][i] != flg[1][i]; }
        printf("%-14s %018llx %018llx %018llx %018llx %8ld %8ld\n", ops[o].name, (unsigned long long)hr[0],
               (unsigned long long)hf[0], (unsigned long long)hr[1], (unsigned long long)hf[1], changed,
               fchanged);
        // vabs and vneg only move a sign bit and need not see RunFast at all. Every other instruction
        // must, in its results or its flags, or this run cannot tell the two modes apart.
        if (!changed && !fchanged && strcmp(ops[o].name, "vabs.f32") && strcmp(ops[o].name, "vneg.f32")) bad = 1;
    }
    if (dump) fclose(dump);
    if (bad) printf("WARNING: an arithmetic instruction gave the same results in both modes\n");
    return bad;
}
