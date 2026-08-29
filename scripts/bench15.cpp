// ONE GEMM shape in a loop, so a hardware profiler can count it. The harness P4.18's `QK^T` item
// asked for: `perf stat` counts a process, and every other bench here runs a table of shapes in one
// process, which makes a counter reading a weighted average of things that behave differently.
//
// NOT part of the build.
//
//   g++ -O3 -std=c++17 -march=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench15.cpp -o bench15 \
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//   ./bench15 <k> [iters] [m] [n]        # defaults: iters=200, m=n=1500
//
//   taskset -c 0 perf stat -e cpu_core/cycles/,cpu_core/instructions/ ./bench15 64 200
//
// **PIN IT AND PREFIX THE EVENTS ON A HYBRID PART.** The 285K splits its PMU into `cpu_core` and
// `cpu_atom`; un-prefixed events are counted on both and the numbers mean nothing. `taskset -c 0` is a
// P-core. Basic counters (cycles, instructions, branches, branch-misses) work per-process at
// `perf_event_paranoid = 2`; the `topdown-*` group does NOT ("Invalid event in per-thread mode") and
// needs system-wide collection, which needs privileges this does not assume.
//
// WHAT IT PRINTS, and why it prints work rather than only time: the question this exists for is
// **instructions retired per FMA the shape actually requires**. `m*n*k` multiply-adds over `KN` f32
// lanes is `m*n*k/KN` FMA instructions if the kernel issues nothing else; the ratio of retired instructions
// to that floor is how much of the time is per-tile overhead rather than arithmetic. A shape with a
// short contraction amortises whatever the tile costs over fewer `k` iterations, so if `k=64` retires
// far more instructions per FMA than `k=768` at the same `m`/`n`, the answer is overhead and the fix is
// blocking or unrolling; if it retires about the same, the time is real work and the mechanism is
// somewhere nobody has looked yet.
//
// The setup (allocation and fill) is outside the loop but INSIDE the process perf counts, so keep
// `iters` large enough that it does not matter -- it is ~10 ms against seconds at the default.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <k> [iters] [m] [n]\n", argv[0]); return 2; }
    const int64_t k     = atoll(argv[1]);
    const int     iters = argc > 2 ? atoi(argv[2]) : 200;
    const int64_t m     = argc > 3 ? atoll(argv[3]) : 1500;
    const int64_t n     = argc > 4 ? atoll(argv[4]) : 1500;

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 1);
    // Sized from the shape rather than a fixed slab: this has to run on a 3 GB Pi as well as on a
    // workstation, and a context that cannot be allocated fails as a null dereference rather than as
    // a message.
    const size_t operand_bytes = (size_t)(k * m + k * n) * sizeof(float);
    struct ggml_init_params dp = { operand_bytes + 64ull*1024*1024, nullptr, false };
    ggml_context * dctx = ggml_init(dp);
    if (!dctx) {
        std::fprintf(stderr, "ggml_init failed for %.1f MB of operands\n", operand_bytes / 1048576.0);
        return 1;
    }

    std::mt19937 rng(17);
    std::normal_distribution<float> dist(0.f, 1.f);
    auto mk = [&](int64_t a, int64_t b) {
        ggml_tensor * t = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, a, b);
        float * p = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = dist(rng);
        return t;
    };
    // ggml_mul_mat contracts over ne0 of both operands, so `k` is ne0 on each -- the same layout
    // bench13/bench14 build, and the same one whisper's QK^T reaches the kernel with.
    ggml_tensor * A = mk(k, m);
    ggml_tensor * B = mk(k, n);

    // One graph, reused: rebuilding it per iteration would put graph construction inside the counters,
    // and it is the kernel that is being counted.
    struct ggml_init_params gp = { (size_t) 16ull*1024*1024, nullptr, true };
    ggml_context * gc = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gc);
    ggml_build_forward_expand(gf, ggml_mul_mat(gc, A, B));
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(al, gf);

    ggml_backend_graph_compute(backend, gf);            // warm: first touch of the output buffer
    const double t0 = now();
    for (int i = 0; i < iters; ++i) ggml_backend_graph_compute(backend, gf);
    const double dt = now() - t0;

    // tinyBLAS's `KN` for f32, which is the lane count the contraction is vectorised over. Not a
    // constant: NEON is 4, AVX2 is 8, AVX-512 is 16, and quoting the wrong one turns the floor below
    // into a number that means nothing on the machine printing it.
#if defined(__AVX512F__)
    const int lanes = 16;
#elif defined(__AVX2__) || defined(__AVX__)
    const int lanes = 8;
#elif defined(__ARM_NEON)
    const int lanes = 4;
#else
    const int lanes = 1;
#endif
    const double flop = 2.0 * m * n * k * iters;
    const double fma_floor = (double) m * n * k * iters / lanes;
    std::printf("m=%lld n=%lld k=%lld iters=%d  %.3f s  %.2f GFLOP/s  %.3f ms/iter\n",
                (long long) m, (long long) n, (long long) k, iters,
                dt, flop / dt / 1e9, dt / iters * 1e3);
    std::printf("FMA floor (m*n*k/%d) = %.4g instructions; divide perf's count by this\n",
                lanes, fma_floor);

    ggml_gallocr_free(al);
    ggml_free(gc);
    ggml_free(dctx);
    ggml_backend_free(backend);
    return 0;
}
