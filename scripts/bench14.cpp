// The estimator P4.18 needed and did not have: a PAIRED ratio test for two GEMM shapes.
//
// NOT part of the build; a standalone measurement, kept because two verdicts in Retro-012 were written
// from inside this dev box's noise before it was characterised, and both were wrong.
//
//   g++ -O3 -std=c++17 -march=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench14.cpp -o bench14 \
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//   ./bench14 [rounds] [threads]     # default 31 1
//
// **THREADS DEFAULTS TO 1 AND THE PAIRS BELOW WERE ALL READ AT 1** -- P4.18 asked a single-core
// question. It is an argument because P4.30c's `ldc` item is the same pairs at a thread count: after
// `ggml-0012` gave every job a full 64-byte line of `C`, the residue is that `m = 1500` floats is 6000
// bytes and `6000 % 64 = 16`, so that line is not ALIGNED to one and straddles two on odd columns.
// That is invisible at one thread by construction -- see Retro-019 -- and the m=1500-against-m=1504
// pair below is exactly the control it needs.
//
// WHY NOT scripts/bench13.cpp's min-over-interleaved-rounds. That already fixed the worst failure --
// timing each shape as a BLOCK of repetitions, which on a 2-core laptop measures the clock (the same
// binary reported 27.9 GFLOP/s for QK^T at k=64 and, twenty minutes later on a box still warm from a
// ctest run, 12.6). But it still compares two INDEPENDENT MINIMA, and on this machine those are drawn
// from different parts of a thermal excursion: bench13's own clock witness has a worst/best spread of
// 1.4-2.5x, and simply making its table longer depresses every row in it by ~15% as the laptop heats.
//
// Here the two arms of a pair run BACK TO BACK inside one round and the RATIO OF RATES is recorded per
// round. Drift moves both halves together and cancels. The report is the median with p10/p90, because
// the interval is the result as much as the median is: **nothing under about 1.2x is resolvable on
// this box even so**, and a pair whose p10 crosses 1.0 is "weak", not a number.
//
// WHAT THE PAIRS ARE. The BM question: tinyBLAS picks its row block from m % 16 / 8 / 4 (BM = 4/2/1),
// and whisper's QK^T has m = 1500, so 1500 % 16 == 12 gets it the LEAST blocked schedule -- the same
// "a frame count is a number nothing rounds" that made 1500 miss KN and cost A@V the whole file
// (ggml-0011). Measured this way BM=4 is worth ~1.15x at k=64 and 1.02x at k=768, which is a real but
// partial effect with the control behaving exactly as a blocking story predicts. The last pair is the
// scale check: QK^T against a projection-shaped GEMM, ~2.1x.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <random>
#include <vector>
#include <string>

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
static ggml_backend_t backend;
static ggml_context * dctx;
static std::mt19937 rng(17);

static ggml_tensor * mk(int64_t a, int64_t b, int64_t c) {
    ggml_tensor * t = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, a, b, c);
    std::normal_distribution<float> d(0.f, 1.f);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = d(rng);
    return t;
}
static double once(ggml_tensor * A, ggml_tensor * B) {
    struct ggml_init_params gp = { (size_t) 64ull*1024*1024, nullptr, true };
    ggml_context * gc = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gc);
    ggml_build_forward_expand(gf, ggml_mul_mat(gc, A, B));
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(al, gf);
    const double t0 = now();
    ggml_backend_graph_compute(backend, gf);
    const double dt = (now() - t0) * 1e3;
    ggml_gallocr_free(al); ggml_free(gc);
    return dt;
}
struct Pair { std::string name; ggml_tensor *A1,*B1,*A2,*B2; double f1, f2; std::vector<double> r; };

int main(int argc, char ** argv) {
    const int rounds  = argc > 1 ? atoi(argv[1]) : 31;
    const int nthreads = argc > 2 ? atoi(argv[2]) : 1;
    backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, nthreads);
    struct ggml_init_params dp = { (size_t) 6144ull*1024*1024, nullptr, false };
    dctx = ggml_init(dp);

    const int64_t T = 1500, D = 64, W = 768;
    std::vector<Pair> ps;
    auto add = [&](const char * name, int64_t m1, int64_t n1, int64_t k1,
                                     int64_t m2, int64_t n2, int64_t k2) {
        Pair p; p.name = name;
        p.A1 = mk(k1, m1, 1); p.B1 = mk(k1, n1, 1); p.f1 = 2.0*m1*n1*k1/1e9;
        p.A2 = mk(k2, m2, 1); p.B2 = mk(k2, n2, 1); p.f2 = 2.0*m2*n2*k2/1e9;
        ps.push_back(p);
    };
    // The BM question: does tinyBLAS's row blocking explain QK^T at k=64? m picks BM: %16 -> 4,
    // %8 -> 2, else %4 -> 1. Normalised per flop, so the 0.5% size difference does not count.
    add("k=64  BM=1 (m=1500) vs BM=4 (m=1504)", 1500, T, D, 1504, T, D);
    add("k=64  BM=1 (m=1500) vs BM=2 (m=1496)", 1500, T, D, 1496, T, D);
    add("k=64  BM=1 (m=1492) vs BM=4 (m=1504)", 1492, T, D, 1504, T, D);
    // Controls: the same m pair where k is large, where BM should not matter much either way.
    add("k=768 BM=1 (m=1500) vs BM=4 (m=1504)", 1500, T, 768, 1504, T, 768);
    // And the headline ratio, as a scale check: QK^T against a projection-shaped GEMM.
    add("k=64  m=n=1500     vs proj 768x1500x768", T, T, D, W, T, W);

    for (int r = 0; r < rounds; ++r)
        for (auto & p : ps) {
            const double a = once(p.A1, p.B1);
            const double b = once(p.A2, p.B2);
            // ratio of RATES: (f2/b) / (f1/a)  -- >1 means the second arm is faster per flop
            p.r.push_back((p.f2/b) / (p.f1/a));
        }
    printf("%d paired rounds, %d thread(s). Ratio > 1 means the SECOND arm is faster per flop.\n\n",
           rounds, nthreads);
    printf("%-44s %8s %8s %8s %8s\n", "pair", "p10", "median", "p90", "min");
    for (auto & p : ps) {
        std::sort(p.r.begin(), p.r.end());
        const size_t n = p.r.size();
        printf("%-44s %8.3f %8.3f %8.3f %8.3f\n", p.name.c_str(),
               p.r[n/10], p.r[n/2], p.r[(9*n)/10], p.r.front());
    }
    return 0;
}
