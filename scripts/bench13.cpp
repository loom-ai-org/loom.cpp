// P4.18's GEMM half, measured: is whisper's encoder attention losing 2x to the GEMM MICRO-KERNEL, to
// the LAYOUT its operands arrive in, or to something about the SHAPE? NOT part of the build; a
// standalone measurement, kept because the numbers it produces are the justification for
// cmake/patches/ggml-0011 and have to stay reproducible.
//
//   g++ -O3 -std=c++17 -march=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench13.cpp -o bench13 \
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//   ./bench13            # the four measurement sections
//   ./bench13 --check    # the k-tail's correctness sweep against a double-precision reference
//
// Single-threaded on purpose: P4.18's profile is a one-thread profile (P4.14's floor trap), and the
// question here is kernel efficiency, which a thread count only scales.
//
// UNLIKE scripts/bench6.cpp, THIS IS NOT AARCH64-ONLY, and that is the point. tinyBLAS's contraction
// step KN is 4 on NEON, 8 on AVX2 and 16 on AVX-512, and its `k % KN != 0` rejection therefore fires
// on x86 for shapes that are perfectly aligned on ARM. whisper's encoder contracts A@V over 1500 mel
// frames: 1500 % 4 == 0 and 1500 % 8 == 4. Every earlier GEMM measurement in this epic ran on the one
// ISA where that matmul happened to be accepted.
//
// WHAT EACH SECTION ANSWERS.
//  1. layout -- loom feeds QK^T a PERMUTED src0 (key_states is [64, 12, 1500] contiguous, and
//     permute(0,2,1,3) leaves lda = 768 floats with only 64 of them used) where onnxruntime
//     materialises a dense transpose per head. Materialising it here too is worth 4%. NOT the
//     mechanism -- this section exists so nobody spends a day on the transposes again.
//  2. shape -- k, m and the ne02 head batch swept independently. The cliff is on k, and the batch is
//     free.
//  3. divisibility -- k = 1496 / 1500 / 1504 at A@V's own shape, which is the whole finding, plus n
//     for completeness (it does not matter). NOTE that section 3 is FLAT once ggml-0011 is in the
//     tree, which is what "fixed" looks like; to see the cliff again, build a ggml with that patch
//     removed. The recorded numbers are in Epic-05 P4.18.
//  4. reference -- the projection and MLP shapes, so section 2's numbers have a same-core ceiling to
//     be read against rather than a remembered one.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
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

// Times one already-built graph. `best of nrep` rather than a median: every arm here is a single ggml
// node on an idle core, so the distribution is one-sided and the minimum is the least noisy estimator
// of it -- but BOTH ARMS MUST USE THE SAME ONE (P4.15's estimator trap), which is why nothing in this
// file is timed any other way.
template <typename Build>
static double time_graph(Build build, int nrep = 5) {
    double best = 1e30;
    for (int r = 0; r < nrep; ++r) {
        struct ggml_init_params gp = { (size_t) 64ull*1024*1024, nullptr, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, build(gctx));
        ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(al, gf);
        const double t0 = now();
        ggml_backend_graph_compute(backend, gf);
        best = std::min(best, (now() - t0) * 1e3);
        ggml_gallocr_free(al);
        ggml_free(gctx);
    }
    return best;
}

// dst = [m, n, batch];  A = [k, m, batch], B = [k, n, batch] -- ggml's own operand order, so `k` here
// is ne00 and is what `k % KN` is tested against.
static void bench(const char * tag, int64_t m, int64_t n, int64_t k, int64_t batch) {
    ggml_tensor * A = mk(k, m, batch);
    ggml_tensor * B = mk(k, n, batch);
    const double ms = time_graph([&](ggml_context * c) { return ggml_mul_mat(c, A, B); });
    const double gflop = 2.0 * m * n * k * batch / 1e9;
    printf("  %-10s m=%-5lld n=%-5lld k=%-5lld b=%-3lld  %9.2f ms  %6.1f GFLOP/s\n",
           tag, (long long)m, (long long)n, (long long)k, (long long)batch, ms, gflop / (ms*1e-3));
    fflush(stdout);
}

// Every k in [1, 40] plus the boundaries that matter, against a double-precision reference. A k tail
// that drops its leftover elements shows up as a large error at exactly the k that are not multiples
// of KN; f32 accumulation noise does not sort that way. m is 68 so that the ROW tail (ggml-0003) is
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
        ggml_tensor * C = nullptr;
        struct ggml_init_params gp = { (size_t) 16ull*1024*1024, nullptr, true };
        ggml_context * gctx = ggml_init(gp);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        C = ggml_mul_mat(gctx, A, B);
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
            printf("  k=%-5lld  max rel err %.3e%s\n", (long long) k, mx,
                   (k % 8) ? "   <- tail" : "");
        ggml_gallocr_free(al);
        ggml_free(gctx);
    }
    printf("\n  WORST over every k tested: %.3e at k=%d\n", worst, worst_k);
    // A dropped term is O(1) relative, not O(1e-5). This threshold separates the two by four orders of
    // magnitude and is not a tolerance anyone should tighten to "improve" the kernel.
    if (worst > 1e-3) { printf("  FAIL: k tail is dropping terms\n"); return 1; }
    printf("  OK: f32 accumulation noise, no dropped term\n");
    return 0;
}

int main(int argc, char ** argv) {
    backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    struct ggml_init_params dp = { (size_t) 6144ull*1024*1024, nullptr, false };
    dctx = ggml_init(dp);

    if (argc > 1 && std::strcmp(argv[1], "--check") == 0) {
        printf("== k-tail correctness, vs a double-precision reference ==\n");
        const int rc = check();
        ggml_free(dctx);
        ggml_backend_free(backend);
        return rc;
    }

    const int64_t D = 64, H = 12, T = 1500, W = 768, F = 3072;
    const double gf_attn = 2.0 * T * T * D * H / 1e9;

    ggml_tensor * K  = mk(D, H, T);   // key_states, as the exported graph holds it: [64, 12, 1500]
    ggml_tensor * Q  = mk(D, T, H);   // query, already packed:                      [64, 1500, 12]

    printf("== 1. layout: does the permuted src0 explain QK^T? (it does not) ==\n");
    {
        const double strided = time_graph([&](ggml_context * c) {
            return ggml_mul_mat(c, ggml_permute(c, K, 0, 2, 1, 3), Q);   // lda = 768, what loom runs
        });
        const double cont = time_graph([&](ggml_context * c) {
            return ggml_cont(c, ggml_permute(c, K, 0, 2, 1, 3));
        });
        const double dense = time_graph([&](ggml_context * c) {
            return ggml_mul_mat(c, ggml_cont(c, ggml_permute(c, K, 0, 2, 1, 3)), Q);  // lda = 64
        });
        printf("  %-10s %9.2f ms  %6.1f GFLOP/s\n", "strided",  strided, gf_attn / (strided*1e-3));
        printf("  %-10s %9.2f ms\n",                "cont",     cont);
        printf("  %-10s %9.2f ms  %6.1f GFLOP/s\n", "dense",    dense,   gf_attn / (dense*1e-3));
        printf("  dense + cont = %.2f ms vs strided %.2f ms\n", dense + cont, strided);
    }

    printf("\n== 2a. k sweep, m=n=1500, unbatched (whisper's QK^T is k=64) ==\n");
    for (int64_t k : {64, 128, 256, 512, 768}) bench("k", 1500, 1500, k, 1);

    printf("\n== 2b. m sweep, n=1500, k=1500 (whisper's A@V is m=64) ==\n");
    for (int64_t m : {64, 128, 256, 512, 768}) bench("m", m, 1500, 1500, 1);

    printf("\n== 2c. the ne02 head batch: same flops, one call vs twelve ==\n");
    bench("qk-b1",  1500, 1500,  64,  1);
    bench("qk-b12", 1500, 1500,  64, 12);
    bench("av-b1",    64, 1500, 1500,  1);
    bench("av-b12",   64, 1500, 1500, 12);

    printf("\n== 3. k divisibility at A@V's own shape -- KN is 8 for AVX2 F32, 16 for AVX-512, 4 for NEON ==\n");
    for (int64_t k : {1496, 1497, 1500, 1502, 1504}) bench("k%KN", 64, 1500, k, 1);
    printf("  (and n, which does not matter:)\n");
    for (int64_t n : {1496, 1500, 1504}) bench("n", 64, n, 1504, 1);

    printf("\n== 4. reference: the projection and MLP shapes, same core ==\n");
    bench("proj",  W, T, W, 1);
    bench("mlp1",  F, T, W, 1);
    bench("mlp2",  W, T, F, 1);

    ggml_free(dctx);
    ggml_backend_free(backend);
    return 0;
}
