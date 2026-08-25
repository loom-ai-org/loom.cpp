// P4.18's GEMM half, measured: is whisper's encoder attention losing 2x to the GEMM MICRO-KERNEL, to
// the LAYOUT its operands arrive in, or to something about the SHAPE? NOT part of the build; a
// standalone measurement, kept because the numbers it produces are the justification for
// cmake/patches/ggml-0011 and the "measured out" verdicts in Retro-012, and have to stay reproducible.
//
//   g++ -O3 -std=c++17 -march=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench13.cpp -o bench13 \
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//   ./bench13 [rounds]   # default 14
//   ./bench13 --check    # the k-tail's correctness sweep against a double-precision reference
//
// Single-threaded on purpose: P4.18's profile is a one-thread profile (P4.14's floor trap), and the
// question here is kernel efficiency, which a thread count only scales.
//
// THE ESTIMATOR, WHICH COST A WRONG ANSWER BEFORE IT WAS WRITTEN THIS WAY. An earlier version timed
// each shape as a BLOCK of repetitions and took the best of the block. On a 2-core laptop that
// measures the clock: the same binary reported 27.9 GFLOP/s for `qk k=64` and, twenty minutes later
// on a box still warm from a ctest run, 12.6. Every shape now runs ONCE PER ROUND, rounds are
// interleaved, and the figure is the min over rounds -- so a thermal excursion lands on every shape
// rather than on whichever one was running. The first row is a fixed `proj`-shaped GEMM re-run every
// round as a CLOCK WITNESS: its own min is the machine's rate for a shape known to be near peak, and
// the printed spread (worst/best) is the noise floor every other row has to be read against.
//
// AND THE CEILING, because "44 GFLOP/s" means nothing without one. `%peak` is against
// LOOM_BENCH13_PEAK (default 54.0), which is this dev box: **Zen+ has 128-bit FPU datapaths**, so a
// 256-bit FMA is two uops retiring one per cycle and single-core F32 peak is 8 lanes x 2 flops x
// ~3.4 GHz, NOT the ~112 a lane count suggests. Set the variable when running anywhere else. The
// witness landing at 84-88% is what says loom's dense GEMM has nothing left to win, and it is the
// reason the projections are only 1.10x behind MLAS.
//
// WHAT EACH SECTION ANSWERS.
//  1. layout -- loom feeds QK^T a PERMUTED src0 (key_states is [64, 12, 1500] contiguous, and
//     permute(0,2,1,3) leaves lda = 768 floats with only 64 of them used) where onnxruntime
//     materialises a dense transpose per head. Materialising it here too is worth 4%. NOT the
//     mechanism -- this section exists so nobody spends a day on the transposes again.
//  2. shape -- k, m and the ne02 head batch swept independently. The cliff is on k; the batch is free.
//  3. divisibility -- k = 1496 / 1500 / 1504 at A@V's own shape, which is the whole ggml-0011 finding,
//     plus n for completeness (it does not matter). This section is FLAT once ggml-0011 is in the
//     tree, which is what fixed looks like; to see the cliff, build a ggml with that patch removed.
//  4. QK^T at k = 64 -- the one still open, at ~49% of peak with five candidates measured out
//     (Retro-012). The m variants here are the `BM` probe: tinyBLAS picks BM from m % 16 / 8 / 4, so
//     1500 (BM=1), 1496 (BM=2) and 1504 (BM=4) ask whether the row blocking is the cause. It is not.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static ggml_backend_t backend;
static ggml_context * dctx;
static std::mt19937   rng(1234);

static ggml_tensor * mk(int64_t a, int64_t b, int64_t c) {
    ggml_tensor * t = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, a, b, c);
    std::normal_distribution<float> d(0.f, 1.f);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = d(rng);
    return t;
}

static double time_graph(ggml_tensor * (*build)(ggml_context *, void *), void * arg) {
    struct ggml_init_params gp = { (size_t) 64ull*1024*1024, nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_build_forward_expand(gf, build(gctx, arg));
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(al, gf);
    const double t0 = now();
    ggml_backend_graph_compute(backend, gf);
    const double dt = (now() - t0) * 1e3;
    ggml_gallocr_free(al);
    ggml_free(gctx);
    return dt;
}

// One row of the report. `gflop == 0` means the arm is not a GEMM (a `cont`), so no rate is printed.
struct Arm {
    std::string  section;
    std::string  tag;
    double       gflop;
    ggml_tensor * A = nullptr;   // for the plain mul_mat arms
    ggml_tensor * B = nullptr;
    int          kind = 0;       // 0 mul_mat(A,B), 1 mul_mat(permute(A),B), 2 cont(permute(A)), 3 mul_mat(cont(permute(A)),B)
    double       best = 1e30, worst = 0;
};

static ggml_tensor * build_arm(ggml_context * c, void * arg) {
    Arm * a = (Arm *) arg;
    switch (a->kind) {
        case 1:  return ggml_mul_mat(c, ggml_permute(c, a->A, 0, 2, 1, 3), a->B);
        case 2:  return ggml_cont(c, ggml_permute(c, a->A, 0, 2, 1, 3));
        case 3:  return ggml_mul_mat(c, ggml_cont(c, ggml_permute(c, a->A, 0, 2, 1, 3)), a->B);
        default: return ggml_mul_mat(c, a->A, a->B);
    }
}

// dst = [m, n, batch];  A = [k, m, batch], B = [k, n, batch] -- ggml's own operand order, so `k` here
// is ne00 and is what `k % KN` is tested against.
static void add(std::vector<Arm> & v, const char * section, const char * tag,
                int64_t m, int64_t n, int64_t k, int64_t batch) {
    Arm a; a.section = section; a.tag = tag;
    a.gflop = 2.0 * m * n * k * batch / 1e9;
    a.A = mk(k, m, batch);
    a.B = mk(k, n, batch);
    v.push_back(a);
}

// Every k in [1, 40] plus the boundaries that matter, against a double-precision reference. A k tail
// that drops its leftover elements shows up as a large error at exactly the k that are not multiples
// of KN; f32 accumulation noise does not sort that way. m is 68 so the ROW tail (ggml-0003) is
// exercised at the same time, and n is odd.
static int check() {
    double worst = 0; int worst_k = 0;
    std::vector<int64_t> ks;
    for (int64_t k = 1; k <= 40; ++k) ks.push_back(k);
    for (int64_t k : {63, 64, 65, 1496, 1497, 1499, 1500, 1501, 1503, 1504}) ks.push_back(k);
    for (int64_t k : ks) {
        const int64_t m = 68, n = 37;
        ggml_tensor * A = mk(k, m, 1);
        ggml_tensor * B = mk(k, n, 1);
        struct ggml_init_params gp = { (size_t) 16ull*1024*1024, nullptr, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_tensor * C = ggml_mul_mat(gctx, A, B);
        ggml_build_forward_expand(gf, C);
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(al, gf);
        ggml_backend_graph_compute(backend, gf);

        const float * a = (const float *) A->data, * b = (const float *) B->data,
                    * c = (const float *) C->data;
        double mx = 0;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i) {
                double ref = 0;
                for (int64_t l = 0; l < k; ++l) ref += (double) a[i*k + l] * (double) b[j*k + l];
                mx = std::max(mx, std::fabs(c[j*m + i] - ref) / std::max(1.0, std::fabs(ref)));
            }
        if (mx > worst) { worst = mx; worst_k = (int) k; }
        if (k <= 12 || k % 8 == 4 || k > 1490)
            printf("  k=%-5lld  max rel err %.3e%s\n", (long long) k, mx, (k % 8) ? "   <- tail" : "");
        ggml_gallocr_free(al);
        ggml_free(gctx);
    }
    printf("\n  WORST over every k tested: %.3e at k=%d\n", worst, worst_k);
    // A dropped term is O(1) relative, not O(1e-5). This threshold separates the two by four orders of
    // magnitude and is not a tolerance anyone should tighten to "improve" the kernel.
    if (worst > 1e-3) { printf("  FAIL: the k tail is dropping terms\n"); return 1; }
    printf("  OK: f32 accumulation noise, no dropped term\n");
    return 0;
}

int main(int argc, char ** argv) {
    backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    struct ggml_init_params dp = { (size_t) 6144ull*1024*1024, nullptr, false };
    dctx = ggml_init(dp);

    if (argc > 1 && std::strcmp(argv[1], "--check") == 0) {
        printf("== k-tail correctness, against a double-precision reference ==\n");
        const int rc = check();
        ggml_free(dctx);
        ggml_backend_free(backend);
        return rc;
    }
    const int rounds = argc > 1 ? atoi(argv[1]) : 14;
    const char * pk = getenv("LOOM_BENCH13_PEAK");
    const double peak = pk ? atof(pk) : 54.0;

    const int64_t D = 64, H = 12, T = 1500, W = 768, F = 3072;
    const double gf_attn = 2.0 * T * T * D * H / 1e9;

    std::vector<Arm> arms;
    add(arms, "0 witness",      "proj 768x1500x768",  W, T,    W,  1);

    // 1. layout: the same QK^T with a permuted src0 (what loom runs) and with it materialised.
    {
        Arm a; a.section = "1 layout"; a.tag = "QK^T permuted src0 (loom)"; a.gflop = gf_attn;
        a.A = mk(D, H, T); a.B = mk(D, T, H); a.kind = 1; arms.push_back(a);
        Arm b = a; b.tag = "  the cont on its own";   b.gflop = 0; b.kind = 2; arms.push_back(b);
        Arm c = a; c.tag = "  QK^T dense src0";       c.kind = 3; arms.push_back(c);
    }

    // 2. shape
    for (int64_t k : {64, 96, 128, 192, 256, 512, 768})
        add(arms, "2a k sweep",  ("m=n=1500 k=" + std::to_string(k)).c_str(), T, T, k, 1);
    for (int64_t m : {64, 128, 256, 512, 768})
        add(arms, "2b m sweep",  ("n=1500 k=1500 m=" + std::to_string(m)).c_str(), m, T, T, 1);
    add(arms, "2c head batch",   "QK^T x1",  T, T,  D,  1);
    add(arms, "2c head batch",   "QK^T x12", T, T,  D, 12);
    add(arms, "2c head batch",   "A@V  x1",  D, T,  T,  1);
    add(arms, "2c head batch",   "A@V  x12", D, T,  T, 12);

    // 3. k divisibility at A@V's own shape -- KN is 8 for AVX2 F32, 16 for AVX-512, 4 for NEON
    for (int64_t k : {1496, 1497, 1500, 1502, 1504})
        add(arms, "3 k%KN",      ("m=64 n=1500 k=" + std::to_string(k)).c_str(), D, T, k, 1);
    for (int64_t n : {1496, 1500, 1504})
        add(arms, "3 n (control)", ("m=64 k=1504 n=" + std::to_string(n)).c_str(), D, n, 1504, 1);

    // 4. QK^T at k=64, and the BM probe: 1500 -> BM=1, 1496 -> BM=2, 1504 -> BM=4
    add(arms, "4 k=64 BM probe", "m=1500 (BM=1)", 1500, T, D, 1);
    add(arms, "4 k=64 BM probe", "m=1496 (BM=2)", 1496, T, D, 1);
    add(arms, "4 k=64 BM probe", "m=1504 (BM=4)", 1504, T, D, 1);
    add(arms, "4 k=64 BM probe", "m=1492 (BM=1)", 1492, T, D, 1);

    // 5. reference
    add(arms, "5 reference",     "mlp1 3072x1500x768", F, T, W, 1);
    add(arms, "5 reference",     "mlp2 768x1500x3072", W, T, F, 1);

    for (int r = 0; r < rounds; ++r)
        for (auto & a : arms) {
            const double dt = time_graph(build_arm, &a);
            a.best = std::min(a.best, dt);
            a.worst = std::max(a.worst, dt);
        }

    printf("%d interleaved rounds, 1 thread, %%peak against %.1f GFLOP/s\n\n", rounds, peak);
    printf("%-14s %-24s %9s %8s %7s %8s\n", "section", "shape", "best ms", "GFLOP/s", "%peak", "spread");
    std::string last;
    for (auto & a : arms) {
        if (a.section != last) { printf("\n"); last = a.section; }
        const double rate = a.gflop > 0 ? a.gflop / (a.best * 1e-3) : 0;
        if (rate > 0)
            printf("%-14s %-24s %9.2f %8.1f %6.0f%% %7.2fx\n",
                   a.section.c_str(), a.tag.c_str(), a.best, rate, 100*rate/peak, a.worst/a.best);
        else
            printf("%-14s %-24s %9.2f %8s %7s %7.2fx\n",
                   a.section.c_str(), a.tag.c_str(), a.best, "-", "-", a.worst/a.best);
    }
    ggml_free(dctx);
    ggml_backend_free(backend);
    return 0;
}
