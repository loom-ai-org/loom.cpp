// P4.26: `ggml-0012`'s prefix branch, per SHAPE and per THREAD COUNT, with both arms in ONE process.
//
// NOT part of the build.
//
// WHY IT EXISTS. `ggml-0012` was measured on aarch64 at exactly one shape (whisper's `m = 1500`,
// neutral) and on x86 at that same shape (2.75x), and it costs a Cortex-A72 2.4% on VITS -- whose
// matmuls are `m = 100 / 196 / 284`, every one of which takes the branch the patch changed. The rule
// Retro-019 wrote is "measure every ISA a patch is enabled for"; the amendment P4.26 adds is **every
// SHAPE CLASS it is enabled for**, and this is the harness for that.
//
// It needs `scripts/probes/ggml-p426-sgemm-policy-probe.patch`, which is NOT carried -- see
// `scripts/probes/README.md` for how to apply one. That turns `matmul_aligned`'s predicate into a
// run-time switch, so the two arms below are the same binary, the same allocation and the same
// page-cache state, differing by one branch. Copying a whole tree and deleting the patch -- how P4.22
// and P4.26's first pass were both measured -- cannot make that claim.
//
//   GGML_SGEMM_POLICY / loom_sgemm_set_policy:
//     0  pre-0012              m % 16 == 0 && m/16 >= nth
//     1  0012 as FIRST shipped m16 > 0 && nth > 1 && m16/16 >= nth
//     6  as shipped NOW        0, plus a ragged prefix when nth > 1 && k <= 256   (the default)
//     7, 8  the same with k <= 512 / k <= 128
//     2, 3, 4  job-count floors instead: m16/16 >= 2*nth / 4*nth / 8*nth
//     5  0012 with no ragged prefix (m % 16 == 0, plus the nth > 1 guard)
//
// BUILD (same recipe as bench15/bench18):
//
//   g++ -O3 -std=c++17 -march=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench19.cpp -o bench19 \
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//
//   ./bench19 [threads] [policy_a] [policy_b] [rounds]     # default: 4 1 0 5
//
// ESTIMATOR. Per the operating notes: the two arms run back to back inside one round, ABBA over the
// round pair, and the RATIO is taken per round and then medianed -- min-over-interleaved-rounds
// compares two minima drawn from different parts of a thermal excursion, which on a laptop is worth
// 2x. A fixed reference shape runs in every round as a clock witness and its own spread is printed
// with the table, so a reader can see the floor under which nothing here is measurable.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void loom_sgemm_set_policy(int p);   // the probe patch's setter

static double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct Shape {
    int64_t m, n, k;
    int     calls;          // per VITS synthesis, from the census; 0 = not a VITS shape
    const char * note;
};

// VITS's own matmuls, from `GGML_SGEMM_CENSUS=1` on one synthesis of the reference utterance
// (`bench_vits_loom`, 4 threads). `m` is post row-tail, i.e. what `matmul_aligned` sees.
// The three that CHANGE BRANCH under 0012 are m = 100, 196, 284 -- 76% of the sgemm work.
static const Shape kShapes[] = {
    {  284,  384,  960, 32, "VITS  vocoder, the big one   -- CHANGES BRANCH" },
    {  284,  384,  192, 24, "VITS                          -- CHANGES BRANCH" },
    {  284,  192,  192,  8, "VITS                          -- CHANGES BRANCH" },
    {  284,  192,   96,  8, "VITS                          -- CHANGES BRANCH" },
    {  284,   96,  192,  8, "VITS                          -- CHANGES BRANCH" },
    {  100,  192, 2304, 12, "VITS                          -- CHANGES BRANCH" },
    {  100,  768,  576, 12, "VITS                          -- CHANGES BRANCH" },
    {  100,  192,  192, 76, "VITS                          -- CHANGES BRANCH" },
    {  100,  100,   96, 24, "VITS                          -- CHANGES BRANCH" },
    {  196,  100,   96, 24, "VITS                          -- CHANGES BRANCH" },
    {   96,  100,  199, 24, "VITS  m % 16 == 0             -- same branch both arms" },
    {   96,  100,  100, 24, "VITS  m % 16 == 0             -- same branch both arms" },
    {  256,  256,   64,142, "VITS  m % 16 == 0             -- same branch both arms" },
    { 1024,   64,  128, 70, "VITS  m % 16 == 0             -- same branch both arms" },
    { 2048,   32,  256, 16, "VITS  m % 16 == 0             -- same branch both arms" },
    {   80,  256, 1344,  6, "VITS  m % 16 == 0             -- same branch both arms" },
    { 1500, 1500,   64,  0, "whisper QK^T, the shape 0012 was written for -- CHANGES BRANCH" },
    { 1504, 1500,   64,  0, "whisper QK^T padded          -- same branch both arms" },
    {   64, 1500, 1500,  0, "whisper A@V                  -- m = 64, 4 jobs at BM = 4" },
    {  768, 1500,  768,  0, "whisper dense projection     -- same branch both arms" },
};
static const int kNShapes = (int) (sizeof(kShapes) / sizeof(kShapes[0]));

// The clock witness: a shape neither policy touches, run in every round so that the machine's own
// drift is visible beside the result.
static const Shape kWitness = { 1024, 64, 128, 0, "witness (same branch in both arms)" };

struct Bench {
    ggml_backend_t   backend = nullptr;
    ggml_context *   dctx    = nullptr;
    ggml_context *   gctx    = nullptr;
    ggml_cgraph *    gf      = nullptr;
    ggml_gallocr_t   alloc   = nullptr;
    int              nodes   = 0;

    void build(const Shape & s, int threads) {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, threads);

        // Enough nodes that the once-per-graph pool wake is amortised the way a real graph amortises
        // it, capped so that this measures the GEMM and not the allocator.
        nodes = (int) ((8 << 20) / (s.m * s.n));
        nodes = std::max(2, std::min(16, nodes));

        const size_t bytes = (size_t) (s.m + s.n) * s.k * sizeof(float) + (16u << 20);
        ggml_init_params dp = { bytes, nullptr, false };
        dctx = ggml_init(dp);
        ggml_tensor * a = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, s.k, s.m);
        ggml_tensor * b = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, s.k, s.n);
        float * pa = (float *) a->data;
        float * pb = (float *) b->data;
        for (int64_t i = 0; i < s.k * s.m; ++i) pa[i] = (float) ((i % 251) - 125) / 125.f;
        for (int64_t i = 0; i < s.k * s.n; ++i) pb[i] = (float) ((i % 241) - 120) / 120.f;

        ggml_init_params gp = { (size_t) 64u << 20, nullptr, true };
        gctx = ggml_init(gp);
        gf = ggml_new_graph_custom(gctx, (size_t) nodes * 2 + 16, false);
        for (int j = 0; j < nodes; ++j) {
            ggml_build_forward_expand(gf, ggml_mul_mat(gctx, a, b));
        }
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(alloc, gf);
    }

    // One arm: `inner` graph computes under one policy, returning seconds per matmul.
    double arm(int policy, int inner) {
        loom_sgemm_set_policy(policy);
        const double t0 = now();
        for (int i = 0; i < inner; ++i) ggml_backend_graph_compute(backend, gf);
        return (now() - t0) / inner / nodes;
    }

    void free_all() {
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        ggml_free(dctx);
        ggml_backend_free(backend);
    }
};

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2]
                                           : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

// One shape: `rounds` ABBA pairs, ratio per round.
static void run_shape(const Shape & s, int threads, int pa, int pb, int rounds,
                      std::vector<double> * ratios_out) {
    Bench bh;
    bh.build(s, threads);

    const double flop  = 2.0 * (double) s.m * (double) s.n * (double) s.k;
    const int    inner = std::max(2, (int) (2.5e8 / flop / bh.nodes));

    bh.arm(pa, inner);                       // warm-up, both arms, outside the timing
    bh.arm(pb, inner);

    std::vector<double> ratios, ta, tb;
    for (int r = 0; r < rounds; ++r) {
        double a1, b1, b2, a2;
        if (r % 2 == 0) {                    // A B B A, then B A A B
            a1 = bh.arm(pa, inner); b1 = bh.arm(pb, inner);
            b2 = bh.arm(pb, inner); a2 = bh.arm(pa, inner);
        } else {
            b1 = bh.arm(pb, inner); a1 = bh.arm(pa, inner);
            a2 = bh.arm(pa, inner); b2 = bh.arm(pb, inner);
        }
        const double A = 0.5 * (a1 + a2), B = 0.5 * (b1 + b2);
        ta.push_back(A);
        tb.push_back(B);
        ratios.push_back(B / A);             // > 1 means policy A is FASTER
    }
    bh.free_all();

    const double ma = median(ta), mb = median(tb), mr = median(ratios);
    std::vector<double> sorted = ratios;
    std::sort(sorted.begin(), sorted.end());
    std::printf("%6lld %6lld %6lld %5d %11.3f %11.3f %8.2f %8.2f %7.3f  %6.3f-%.3f  %s\n",
                (long long) s.m, (long long) s.n, (long long) s.k, s.calls,
                ma * 1e6, mb * 1e6, flop / ma / 1e9, flop / mb / 1e9,
                mr, sorted.front(), sorted.back(), s.note);
    if (ratios_out) *ratios_out = ratios;
}

// The two sweeps that separate the two costs of the wider row block, each varying ONE thing.
//
//   mode 1  k sweep at m = 284, n = 384. A job's A-slice is `RM * BM * k * 4` bytes -- 16k*4 at
//           BM = 4 against 4k*4 at BM = 1 -- so if the cost is that block outgrowing a core's L1D,
//           it appears as k crosses 512 on a 32 KB L1 and not before.
//   mode 2  n sweep at m = 284, k = 192, which moves NB_BN and therefore the JOB COUNT without
//           touching the footprint. If the cost is load balance, it appears when nb_job/nth is small.
//   mode 3  m sweep at n = 384, k = 192: the other way to move the job count, through ytiles.
static void run_sweep(int mode, int threads, int pa, int pb, int rounds) {
    static const int64_t ks[] = {64, 128, 192, 256, 384, 512, 768, 960, 1536, 2304};
    static const int64_t ns[] = {48, 96, 144, 192, 288, 384, 576, 768, 1536};
    static const int64_t ms[] = {52, 100, 148, 196, 284, 388, 580, 772, 1540};
    const int n_k = (int) (sizeof(ks) / sizeof(ks[0]));
    const int n_n = (int) (sizeof(ns) / sizeof(ns[0]));
    const int n_m = (int) (sizeof(ms) / sizeof(ms[0]));
    const int count = mode == 1 ? n_k : (mode == 2 ? n_n : n_m);
    for (int i = 0; i < count; ++i) {
        Shape s;
        if (mode == 1) s = Shape{284, 384, ks[i], 0, "k sweep -- footprint of one job's row block"};
        if (mode == 2) s = Shape{284, ns[i], 192, 0, "n sweep -- job count through NB_BN"};
        if (mode == 3) s = Shape{ms[i], 384, 192, 0, "m sweep -- job count through ytiles"};
        run_shape(s, threads, pa, pb, rounds, nullptr);
    }
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    const int pa      = argc > 2 ? std::atoi(argv[2]) : 1;
    const int pb      = argc > 3 ? std::atoi(argv[3]) : 0;
    const int rounds  = argc > 4 ? std::atoi(argv[4]) : 5;
    const int mode    = argc > 5 ? std::atoi(argv[5]) : 0;

    std::printf("bench19: sgemm prefix policy A=%d vs B=%d, %d threads, %d ABBA rounds, mode %d\n",
                pa, pb, threads, rounds, mode);
    std::printf("ratio > 1 means policy A (%d) is FASTER than policy B (%d)\n\n", pa, pb);
    std::printf("%6s %6s %6s %5s %11s %11s %8s %8s %7s  %13s  %s\n",
                "m", "n", "k", "calls", "A us/mm", "B us/mm", "A GF/s", "B GF/s", "ratio",
                "min-max", "shape");

    std::vector<double> witness;
    run_shape(kWitness, threads, pa, pb, rounds, &witness);
    if (mode == 0) {
        for (int i = 0; i < kNShapes; ++i) {
            run_shape(kShapes[i], threads, pa, pb, rounds, nullptr);
        }
    } else {
        run_sweep(mode, threads, pa, pb, rounds);
    }

    // What a VITS synthesis would pay, weighted by the census call counts. Not a wall-clock
    // prediction -- an isolated op measurement is an upper bound on what a model sees, never an
    // estimate of it (Retro-012) -- but it says which SHAPES carry whatever the model does show.
    std::printf("\nclock witness spread over the rounds: %.3f-%.3f (nothing below that is measurable)\n",
                *std::min_element(witness.begin(), witness.end()),
                *std::max_element(witness.begin(), witness.end()));
    return 0;
}
