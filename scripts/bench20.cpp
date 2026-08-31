// P4.27: what decides whether a ggml node runs on more than one thread -- and it is NOT the node.
//
// NOT part of the build.
//
// WHY IT EXISTS. P4.25 threaded ggml's transcendental unaries by moving `TANH`/`SIGMOID`/`EXP` from
// `n_tasks = 1` to `n_tasks = n_threads` in `ggml_get_n_tasks`, measured the op 3.92x faster through
// ggml's own threadpool (`scripts/bench18.cpp`), predicted ~26 ms of a 1130 ms VITS synthesis, and
// measured the model at 1.005x over twelve paired ABBA rounds. P4.27 was opened to find where 26 ms
// went. It went nowhere, because it was never there:
//
//   * `ggml_get_n_tasks` is read in exactly ONE place in ggml v0.19.0 -- `ggml_graph_plan`, which uses
//     it for the work-buffer size and for `max_tasks`, and then sets `cplan.n_threads =
//     MIN(max_tasks, n_threads)` (ggml-cpu.c:3018). There is no per-node thread count any more.
//   * `ggml_graph_compute_thread` runs EVERY node on EVERY thread with `params.nth = cplan.n_threads`
//     (ggml-cpu.c:3342-3372). An op that must not split says so itself, by returning early when
//     `ith != 0`; `apply_unary_op` does not -- it splits over rows through `get_thread_range`.
//
// So `n_tasks` is a GRAPH-level clamp: it decides the thread count for the whole graph, and it can
// only ever take it DOWN to 1, when no node in the graph declares more. A `TANH` in a graph that also
// holds a `MUL_MAT` has been threaded all along.
//
// WHICH MAKES `bench18` A BENCH WHOSE GRAPH WAS THE TREATMENT. Its graph is 256 `TANH` nodes and
// nothing else, so unpatched it plans `n_threads = 1` and the "1 thread vs 4 threads" comparison
// inside it is really "this graph cannot thread vs this graph can". The 3.92x is real and it is a
// property of the patch's effect ON THAT GRAPH. VITS's graph contains matmuls and convolutions, so
// both arms of the model A/B planned 4 threads and ran the same code. 1.005x was two identical arms.
//
// WHAT THIS PRINTS. `ggml_graph_plan()` is public, so the mechanism needs no timing at all: the plan's
// own `n_threads` for a graph of N `TANH` nodes, and for the same graph with ONE `MUL_MAT` added.
// The timings underneath it are the consequence.
//
// BUILD (same recipe as bench18/bench19):
//
//   g++ -O3 -std=c++17 -march=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench20.cpp -o bench20 \
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//
//   ./bench20 [threads]        # default 4
//
// It needs nothing patched to show the mechanism. `scripts/probes/ggml-p427-graph-plan-probe.patch`
// (not carried; see scripts/probes/README.md) adds the two switches the rest of P4.27 used:
// `LOOM_PLAN_PROBE` to print what every graph is planned at, and `LOOM_UNARY_SERIAL` to price the
// threading by removing it from a real model.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

static double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// VITS's WN gate, the shape P4.25 was about.
static const int64_t NE0   = 286;
static const int64_t ROWS  = 192;
static const int     NODES = 128;

struct Case {
    ggml_backend_t backend = nullptr;
    ggml_context * dctx    = nullptr;
    ggml_context * gctx    = nullptr;
    ggml_cgraph  * gf      = nullptr;
    ggml_gallocr_t alloc   = nullptr;

    // `companion` adds one small MUL_MAT to the same graph. It is 0.03% of the work and it is the
    // whole experiment.
    void build(int threads, bool companion) {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, threads);

        ggml_init_params dp = { (size_t) NE0 * ROWS * sizeof(float) + (16u << 20), nullptr, false };
        dctx = ggml_init(dp);
        ggml_tensor * x = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, NE0, ROWS);
        float * p = (float *) x->data;
        for (int64_t i = 0; i < NE0 * ROWS; ++i) p[i] = (float) ((i % 2001) - 1000) / 250.f;
        ggml_tensor * a = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, 32, 8);
        ggml_tensor * b = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, 32, 8);
        for (int64_t i = 0; i < 32 * 8; ++i) { ((float *) a->data)[i] = 1.f; ((float *) b->data)[i] = 1.f; }

        ggml_init_params gp = { (size_t) 64u << 20, nullptr, true };
        gctx = ggml_init(gp);
        gf = ggml_new_graph_custom(gctx, (size_t) NODES * 2 + 16, false);
        for (int j = 0; j < NODES; ++j) ggml_build_forward_expand(gf, ggml_tanh(gctx, x));
        if (companion) ggml_build_forward_expand(gf, ggml_mul_mat(gctx, a, b));

        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_gallocr_alloc_graph(alloc, gf);
    }

    int planned(int threads) const { return ggml_graph_plan(gf, threads, nullptr).n_threads; }

    double per_node() {
        double best = 1e9;
        for (int r = 0; r <= 6; ++r) {
            const double t0 = now();
            for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, gf);
            const double dt = (now() - t0) / 3 / NODES;
            if (r > 0 && dt < best) best = dt;
        }
        return best;
    }

    void free_all() {
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        ggml_free(dctx);
        ggml_backend_free(backend);
    }
};

static void row(const char * what, int threads, bool companion) {
    Case c;
    c.build(threads, companion);
    const int planned = c.planned(threads);
    const double t = c.per_node();
    c.free_all();
    std::printf("%-34s %8d %10d %14.2f\n", what, threads, planned, t * 1e6);
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    const int threads = argc > 1 ? std::atoi(argv[1]) : 4;

    std::printf("P4.27: a graph of %d TANH [%lld, %lld] nodes, with and without ONE MUL_MAT [32x8]\n\n",
                NODES, (long long) NE0, (long long) ROWS);
    std::printf("%-34s %8s %10s %14s\n", "graph", "asked", "planned", "us per TANH");
    row("TANH only",                    1,       false);
    row("TANH only",                    threads, false);
    row("TANH + one MUL_MAT",           1,       true);
    row("TANH + one MUL_MAT",           threads, true);
    std::printf("\nIf the two `planned` columns differ, `n_tasks` is a graph-level clamp and the\n"
                "unary was already threaded in any real model graph -- see the header.\n");
    return 0;
}
