// ADR-079 P4.32f f2: what ggml's elementwise passes cost on the VFP11, and what a tuned loop costs.
//
// Standalone (no ggml): the "stock" arms are ggml's own loops, copied, with the same shape of call
// (binary-ops.cpp's apply_binary_op calls vec_binary_op_contiguous once per src1 row, so a [1, C]
// bias over [T, C] is T calls of n = 1); the "v" arms are candidates. Per arm: median ns per element.
// Two sizes: amy's largest stage ([130048, 24], 12.5 MB a tensor, cold every pass) and an L1-resident
// 2048 floats repeated (the compute floor).
//
// Build (ARMv6, loom-armv6-build image): c++ -O3 -std=c++17 bench55.cpp -o bench55
// Run on the board: ./bench55 [reps] [runfast]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#define NOINL __attribute__((noinline))

// ---- stock: ggml's loops ---------------------------------------------------------------------------
template <float (*op)(float, float)>
static inline void vec_binary_op_contiguous(const int64_t n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; i++) z[i] = op(x[i], y[i]);
}
static inline float op_add(float a, float b) { return a + b; }
static inline float op_mul(float a, float b) { return a * b; }

// one ggml row of a binary op: src1 row of ne10 floats over a src0 row of ne00
template <float (*op)(float, float)>
NOINL static void stock_row(int64_t ne00, int64_t ne10, float * z, const float * x, const float * y) {
    const int64_t nr0 = ne00 / ne10;
    for (int64_t r = 0; r < nr0; ++r) vec_binary_op_contiguous<op>(ne10, z + r * ne10, x + r * ne10, y);
}

NOINL static void stock_leaky(const int n, float * y, const float * x, const float ns) {
    for (int i = 0; i < n; ++i) y[i] = ((x[i] > 0.f) ? x[i] : 0.f) + ns * ((x[i] < 0.0f) ? x[i] : 0.f);
}

// ---- candidates --------------------------------------------------------------------------------
#define PLD(p) __builtin_prefetch((const char *) (p))

// z = x + y, unrolled by 8, PLD `D` bytes ahead on both inputs
template <int D>
NOINL static void v_add(int32_t n, float * z, const float * x, const float * y) {
    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        if (D) { PLD((const char *) (x + i) + D); PLD((const char *) (y + i) + D); }
        const float a0 = x[i] + y[i], a1 = x[i + 1] + y[i + 1], a2 = x[i + 2] + y[i + 2], a3 = x[i + 3] + y[i + 3];
        const float a4 = x[i + 4] + y[i + 4], a5 = x[i + 5] + y[i + 5], a6 = x[i + 6] + y[i + 6], a7 = x[i + 7] + y[i + 7];
        z[i] = a0; z[i + 1] = a1; z[i + 2] = a2; z[i + 3] = a3; z[i + 4] = a4; z[i + 5] = a5; z[i + 6] = a6; z[i + 7] = a7;
    }
    for (; i < n; ++i) z[i] = x[i] + y[i];
}

// z = x op s (a scalar), unrolled by 8, PLD ahead
template <int D, bool MUL>
NOINL static void v_scalar(int32_t n, float * z, const float * x, float s) {
    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        if (D) PLD((const char *) (x + i) + D);
        float a[8];
        for (int k = 0; k < 8; ++k) a[k] = MUL ? x[i + k] * s : x[i + k] + s;
        for (int k = 0; k < 8; ++k) z[i + k] = a[k];
    }
    for (; i < n; ++i) z[i] = MUL ? x[i] * s : x[i] + s;
}

// the VFP compares as integer ranges: x > 0 iff lo <= u <= 0x7f800000, x < 0 iff 0x80000000 + lo <= u <=
// 0xff800000 (lo = 1 in IEEE mode, 0x00800000 under flush-to-zero), then ggml's own arithmetic
static inline uint32_t fz_lo() {
#if defined(__arm__)
    uint32_t f;
    __asm__ volatile("vmrs %0, fpscr" : "=r"(f));
    return (f & (1u << 24)) ? 0x00800000u : 1u;
#else
    return 1u;
#endif
}

template <int D>
NOINL static void v_leaky(int32_t n, float * y, const float * x, float ns, uint32_t lo) {
    const uint32_t * u = (const uint32_t *) x;
    const uint32_t span = 0x7f800000u - lo;
    int32_t i = 0;
    auto one = [&](int32_t j) {
        const uint32_t b = u[j];
        uint32_t p = b - lo <= span ? b : 0u;
        uint32_t q = b - (0x80000000u + lo) <= span ? b : 0u;
        float fp, fq;
        memcpy(&fp, &p, 4); memcpy(&fq, &q, 4);
        return fp + ns * fq;
    };
    for (; i + 8 <= n; i += 8) {
        if (D) PLD((const char *) (x + i) + D);
        float a[8];
        for (int k = 0; k < 8; ++k) a[k] = one(i + k);
        for (int k = 0; k < 8; ++k) y[i + k] = a[k];
    }
    for (; i < n; ++i) y[i] = one(i);
}

#if defined(__arm__)
// z = x + y in VFP asm: 8 floats a step through vldmia/vstmia, eight independent scalar vadds
template <int D>
NOINL static void a_add(int32_t n, float * z, const float * x, const float * y) {
    int32_t blocks = n / 8;
    if (blocks) {
        __asm__ volatile(
            "1:\n\t"
            "pld [%[x], %[d]]\n\t"
            "pld [%[y], %[d]]\n\t"
            "vldmia %[x]!, {s0-s7}\n\t"
            "vldmia %[y]!, {s8-s15}\n\t"
            "vadd.f32 s0, s0, s8\n\t"
            "vadd.f32 s1, s1, s9\n\t"
            "vadd.f32 s2, s2, s10\n\t"
            "vadd.f32 s3, s3, s11\n\t"
            "vadd.f32 s4, s4, s12\n\t"
            "vadd.f32 s5, s5, s13\n\t"
            "vadd.f32 s6, s6, s14\n\t"
            "vadd.f32 s7, s7, s15\n\t"
            "subs %[b], %[b], #1\n\t"
            "vstmia %[z]!, {s0-s7}\n\t"
            "bne 1b\n\t"
            : [x] "+r"(x), [y] "+r"(y), [z] "+r"(z), [b] "+r"(blocks)
            : [d] "I"(D)
            : "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14",
              "s15", "cc", "memory");
    }
    for (int32_t i = 0; i < n % 8; ++i) z[i] = x[i] + y[i];
}

// z = x * s or x + s in VFP asm, 8 a step; the scalar sits in s16
template <int D, bool MUL>
NOINL static void a_scalar(int32_t n, float * z, const float * x, float sc) {
    int32_t blocks = n / 8;
    if (blocks) {
        if (MUL) {
            __asm__ volatile(
                "vmov s16, %[s]\n\t"
                "1:\n\t"
                "pld [%[x], %[d]]\n\t"
                "vldmia %[x]!, {s0-s7}\n\t"
                "vmul.f32 s0, s0, s16\n\t" "vmul.f32 s1, s1, s16\n\t" "vmul.f32 s2, s2, s16\n\t" "vmul.f32 s3, s3, s16\n\t"
                "vmul.f32 s4, s4, s16\n\t" "vmul.f32 s5, s5, s16\n\t" "vmul.f32 s6, s6, s16\n\t" "vmul.f32 s7, s7, s16\n\t"
                "subs %[b], %[b], #1\n\t"
                "vstmia %[z]!, {s0-s7}\n\t"
                "bne 1b\n\t"
                : [x] "+r"(x), [z] "+r"(z), [b] "+r"(blocks)
                : [d] "I"(D), [s] "r"(sc)
                : "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s16", "cc", "memory");
        } else {
            __asm__ volatile(
                "vmov s16, %[s]\n\t"
                "1:\n\t"
                "pld [%[x], %[d]]\n\t"
                "vldmia %[x]!, {s0-s7}\n\t"
                "vadd.f32 s0, s0, s16\n\t" "vadd.f32 s1, s1, s16\n\t" "vadd.f32 s2, s2, s16\n\t" "vadd.f32 s3, s3, s16\n\t"
                "vadd.f32 s4, s4, s16\n\t" "vadd.f32 s5, s5, s16\n\t" "vadd.f32 s6, s6, s16\n\t" "vadd.f32 s7, s7, s16\n\t"
                "subs %[b], %[b], #1\n\t"
                "vstmia %[z]!, {s0-s7}\n\t"
                "bne 1b\n\t"
                : [x] "+r"(x), [z] "+r"(z), [b] "+r"(blocks)
                : [d] "I"(D), [s] "r"(sc)
                : "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s16", "cc", "memory");
        }
    }
    for (int32_t i = 0; i < n % 8; ++i) z[i] = MUL ? x[i] * sc : x[i] + sc;
}
#endif

// ---- harness -------------------------------------------------------------------------------------
template <class F>
static double ns_per(F f, int64_t n_elems, int reps) {
    f();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        const double t0 = now();
        f();
        t.push_back((now() - t0) * 1e9 / (double) n_elems);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 7;
#if defined(__arm__)
    // "runfast": FZ | DN, what ggml-0031 sets on every compute thread
    if (argc > 2 && !strcmp(argv[2], "runfast")) {
        uint32_t f;
        __asm__ volatile("vmrs %0, fpscr" : "=r"(f));
        f |= (1u << 24) | (1u << 25);
        __asm__ volatile("vmsr fpscr, %0" : : "r"(f));
    }
#endif
    const int64_t T = 130048, C = 24, N = T * C;
    // 32-byte aligned like ggml's CPU buffers (a std::vector starts 8 bytes into a line, and on the
    // ARM1176 that alone nearly doubles a cold stream's cost: bench55_streams.cpp)
    auto abuf = [&](int64_t n) { return (float *) aligned_alloc(32, (size_t) ((n * 4 + 31) / 32 * 32)); };
    struct Buf { float * p; float * data() { return p; } float & operator[](int64_t i) { return p[i]; } };
    Buf X{abuf(N)}, Y{abuf(N)}, Z{abuf(N)}, Z2{abuf(N)};
    std::vector<float> B(C);
    for (int64_t i = 0; i < N; ++i) {
        X[i] = 0.01f * (float) ((i * 7919) % 113 - 56);
        Y[i] = 0.003f * (float) ((i * 104729) % 97 - 48);
    }
    for (int64_t c = 0; c < C; ++c) B[c] = 0.01f * (float) (c - 12);
    const float third = 1.0f / 3.0f, ns = 0.1f;
    const uint32_t lo = fz_lo();
    printf("lo=0x%08x reps %d; ns/elem, cold = [%lld,%lld], hot = 2048 x %d\n", lo, reps, (long long) T,
           (long long) C, (int) (N / 2048));

    int bad = 0;
    auto same = [&](const char * what) {
        if (memcmp(Z.data(), Z2.data(), N * sizeof(float))) { printf("  MISMATCH %s\n", what); ++bad; }
    };

    // each arm over the whole cold tensor, then over a hot 2048-float slice repeated to the same count
    auto both = [&](const char * name, auto fn) {
        const double cold = ns_per([&] { fn(0, N); }, N, reps);
        const double hot  = ns_per([&] { for (int64_t o = 0; o + 2048 <= N; o += 2048) fn(-1, 2048); }, N, reps);
        printf("%-30s %7.1f %7.1f\n", name, cold, hot);
    };

    printf("%-30s %7s %7s\n", "arm", "cold", "hot");
    both("memcpy", [&](int64_t o, int64_t n) { memcpy(Z.data() + (o < 0 ? 0 : o), X.data() + (o < 0 ? 0 : o), n * 4); });

    // same-shape ADD
    both("stock add", [&](int64_t, int64_t n) { stock_row<op_add>(n, n, Z.data(), X.data(), Y.data()); });
    both("v_add pld128", [&](int64_t, int64_t n) { v_add<128>(n, Z.data(), X.data(), Y.data()); });
#if defined(__arm__)
    both("a_add pld96", [&](int64_t, int64_t n) { a_add<96>(n, Z.data(), X.data(), Y.data()); });
    both("a_add pld192", [&](int64_t, int64_t n) { a_add<192>(n, Z.data(), X.data(), Y.data()); });
    both("a_add pld320", [&](int64_t, int64_t n) { a_add<320>(n, Z.data(), X.data(), Y.data()); });
    both("a_add pld512", [&](int64_t, int64_t n) { a_add<512>(n, Z.data(), X.data(), Y.data()); });
    both("a_add pld1024", [&](int64_t, int64_t n) { a_add<1024>(n, Z.data(), X.data(), Y.data()); });
#endif
    stock_row<op_add>(N, N, Z2.data(), X.data(), Y.data());
    v_add<128>(N, Z.data(), X.data(), Y.data()); same("v_add");
#if defined(__arm__)
    a_add<320>(N, Z.data(), X.data(), Y.data()); same("a_add");
#endif

    // bias ADD: [T, C] + [1, C], ggml calls the contiguous loop with n = 1 per element
    auto stock_bias = [&](int64_t rows, int64_t t) {
        for (int64_t c = 0; c < rows; ++c) stock_row<op_add>(t, 1, Z.data() + c * t, X.data() + c * t, &B[c % C]);
    };
    both("stock bias", [&](int64_t o, int64_t n) { if (o < 0) stock_bias(1, n); else stock_bias(C, T); });
    both("v_scalar add pld128", [&](int64_t o, int64_t n) {
        if (o < 0) v_scalar<128, false>(n, Z.data(), X.data(), B[0]);
        else for (int64_t c = 0; c < C; ++c) v_scalar<128, false>(T, Z.data() + c * T, X.data() + c * T, B[c]);
    });
    stock_bias(C, T); memcpy(Z2.data(), Z.data(), N * 4);
    for (int64_t c = 0; c < C; ++c) v_scalar<128, false>(T, Z.data() + c * T, X.data() + c * T, B[c]);
    same("bias");

    // MUL by a [1, 1] scalar (the resblock sum's 1/3)
    both("stock mul scalar", [&](int64_t, int64_t n) { stock_row<op_mul>(n, 1, Z.data(), X.data(), &third); });
    both("v_scalar mul pld128", [&](int64_t, int64_t n) { v_scalar<128, true>(n, Z.data(), X.data(), third); });
#if defined(__arm__)
    both("a_scalar mul pld128", [&](int64_t, int64_t n) { a_scalar<128, true>(n, Z.data(), X.data(), third); });
    both("a_scalar mul pld256", [&](int64_t, int64_t n) { a_scalar<256, true>(n, Z.data(), X.data(), third); });
    both("a_scalar mul pld512", [&](int64_t, int64_t n) { a_scalar<512, true>(n, Z.data(), X.data(), third); });
    both("a_scalar mul pld1024", [&](int64_t, int64_t n) { a_scalar<1024, true>(n, Z.data(), X.data(), third); });
#endif
    both("v_scalar mul pld512", [&](int64_t, int64_t n) { v_scalar<512, true>(n, Z.data(), X.data(), third); });
    stock_row<op_mul>(N, 1, Z2.data(), X.data(), &third);
#if defined(__arm__)
    a_scalar<256, true>(N, Z.data(), X.data(), third); same("a_scalar mul");
    stock_row<op_add>(N, 1, Z2.data(), X.data(), &third);
    a_scalar<256, false>(N, Z.data(), X.data(), third); same("a_scalar add");
    stock_row<op_mul>(N, 1, Z2.data(), X.data(), &third);
#endif
    v_scalar<128, true>(N, Z.data(), X.data(), third); same("mul scalar");

    // LEAKY_RELU
    both("stock leaky", [&](int64_t, int64_t n) { stock_leaky(n, Z.data(), X.data(), ns); });
    both("v_leaky pld128", [&](int64_t, int64_t n) { v_leaky<128>(n, Z.data(), X.data(), ns, lo); });
    both("v_leaky pld256", [&](int64_t, int64_t n) { v_leaky<256>(n, Z.data(), X.data(), ns, lo); });
    both("v_leaky pld512", [&](int64_t, int64_t n) { v_leaky<512>(n, Z.data(), X.data(), ns, lo); });
    stock_leaky(N, Z2.data(), X.data(), ns);
    v_leaky<128>(N, Z.data(), X.data(), ns, lo); same("leaky");

    // the leaky's edge bits: zeros, denormals, NaNs, infs
    {
        const uint32_t edge[] = { 0x0u, 0x80000000u, 0x1u, 0x807fffffu, 0x00800000u, 0x80800000u, 0x7f800000u,
                                  0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u, 0x3f800000u, 0xbf800000u,
                                  0x00400000u, 0x80400000u, 0x00000123u, 0x80000123u };
        const int ne = sizeof edge / sizeof edge[0];
        float in[ne], a[ne], b[ne];
        memcpy(in, edge, sizeof in);
        stock_leaky(ne, a, in, ns);
        v_leaky<0>(ne, b, in, ns, lo);
        if (memcmp(a, b, sizeof a)) {
            ++bad;
            for (int k = 0; k < ne; ++k) {
                uint32_t ua, ub; memcpy(&ua, a + k, 4); memcpy(&ub, b + k, 4);
                if (ua != ub) printf("  leaky edge 0x%08x: stock 0x%08x, v 0x%08x\n", edge[k], ua, ub);
            }
        }
    }
    printf(bad ? "FAILED %d\n" : "all arms bit-identical\n", bad);
    return bad ? 1 : 0;
}
