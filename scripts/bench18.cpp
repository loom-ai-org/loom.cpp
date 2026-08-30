// Where threading a unary op starts to pay, measured through ggml's OWN threadpool (P4.25).
//
// NOT part of the build. P4.25 needed to decide when a TANH is worth more than one core, and the
// number cannot come from a raw OpenMP loop (scripts/bench17.c): what it trades against is ggml's
// in-graph barrier, and -- far more importantly -- ggml's unary ops split over ROWS.
//
// P4.25 IS CLOSED, MEASURED OUT (Retro-012): the op is 3.92x and the model is 1.005x. This file is
// kept because the two things it found are properties of ggml, not of that patch -- the nrows = 1
// cliff below, and the per-node floor that makes threading a small unary a LOSS at high thread
// counts (~1.9 us at 24 threads on a Core Ultra 9 285K, so anything under ~16K elements loses).
//
// THE THING THIS FOUND, WHICH IS THE WHOLE POINT. `apply_unary_op` takes its slice from
// `get_thread_range` (ggml-cpu/common.h), which divides `ggml_nrows(src0)` by `nth`. A tensor with
// ONE row therefore hands every element to thread 0 and gives the other threads an empty range -- all
// of the barrier, none of the split. That is not hypothetical: Kokoro issues 870 `UNARY [256, 1]`
// nodes per synthesis, one row each, and threading them unconditionally made that bucket 31x worse.
// VITS's gate is `[286, 192]` -- 192 rows -- and threads 3.98x. Same op, same patch, opposite sign,
// and NOTHING about the element count distinguishes them.
//
// So a guard would have to be on ROWS first and on total work second, and both sweeps below exist to
// set it:
//
//   sweep 1  ne0 = 256 fixed, rows 1 .. 512   -- finds the row floor (and the nrows = 1 cliff)
//   sweep 2  rows = 192 fixed, ne0 8 .. 16384 -- finds the work floor at a row count that can split
//
// READ THIS BEFORE PREDICTING AN END-TO-END NUMBER FROM IT. This bench runs its nodes BACK TO BACK,
// which is the right way to price the op and the wrong way to price the op *in a model*. On the Pi it
// puts VITS's gate shape [286, 192] at 3.92x (1107.8 -> 282.7 us), so the model's 32 gate nodes ought
// to be worth ~26 ms of a 1130 ms synthesis -- and twelve paired ABBA rounds of the real model measure
// essentially nothing. Whatever eats it lives between the nodes, not in them; ggml's threadpool
// sleeping between two multi-threaded nodes is the named suspect (loom.cpp Retro-017 is that
// mechanism, from the libgomp side). An isolated op measurement is an upper bound on what a model
// will see, never an estimate of it.
//
// BUILD (same recipe as bench15/bench16):
//
//   g++ -O3 -std=c++17 -march=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench18.cpp -o bench18 \
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//
//   ./bench18 [threads] [op]      # op: 0 = TANH (needs the P4.25 patch to thread), 1 = SILU (always)
//
// TWO HARNESS MISTAKES IT IS WRITTEN TO AVOID, both of which produced confident wrong numbers here:
//  * ONE NODE PER GRAPH. Then every ggml_backend_graph_compute pays a full pool WAKE, and threading
//    reads as a loss at every size up to 2.3M elements. A real graph wakes once and runs hundreds of
//    nodes inside it. Same shape as Retro-012's "a node-by-node profiler cannot price a per-node
//    cost".
//  * A 1-D TENSOR. See above -- it measures the nrows = 1 cliff and calls it the op.
// Both arms are the SAME build, differing only in ggml_backend_cpu_set_n_threads, so nothing here can
// be a build-type difference.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Enough nodes that the once-per-graph pool wake is amortised the way a real graph amortises it --
// VITS's flow_vocoder is 468 nodes -- but capped by footprint, so this measures the op and not the
// allocator.
static int nodes_for(int64_t elems) {
    int nodes = (int) ((4 << 20) / (elems > 0 ? elems : 1));
    if (nodes > 256) nodes = 256;
    if (nodes < 8)   nodes = 8;
    return nodes;
}

static int g_op = 0;   // 0 = TANH, 1 = SILU

static double time_one(int64_t ne0, int64_t rows, int threads, int nodes) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);

    struct ggml_init_params dp = {(size_t) ne0 * rows * sizeof(float) + (64u << 20), nullptr, false};
    ggml_context* dctx = ggml_init(dp);
    ggml_tensor* x = ggml_new_tensor_2d(dctx, GGML_TYPE_F32, ne0, rows);
    float* p = (float*) x->data;
    for (int64_t i = 0; i < ne0 * rows; ++i) p[i] = (float) ((i % 2001) - 1000) / 250.f;

    struct ggml_init_params gp = {(size_t) 64u << 20, nullptr, true};
    ggml_context* gc = ggml_init(gp);
    ggml_cgraph* gf = ggml_new_graph_custom(gc, (size_t) nodes * 2 + 16, false);
    // INDEPENDENT nodes off one input, not a chain: a chain would time the dependency stall too, and
    // a vocoder's unaries are not a chain.
    // SILU is the same apply_unary_op row-split path and is ALREADY n_tasks = n_threads upstream, so
    // it measures that path's scaling on a stock build -- which is how the 285K numbers were taken.
    for (int j = 0; j < nodes; ++j)
        ggml_build_forward_expand(gf, g_op ? ggml_silu(gc, x) : ggml_tanh(gc, x));
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(al, gf);

    double best = 1e9;
    const int inner = ne0 * rows < 65536 ? 20 : 3;
    for (int r = 0; r <= 9; ++r) {
        const double t0 = now();
        for (int i = 0; i < inner; ++i) ggml_backend_graph_compute(backend, gf);
        const double dt = (now() - t0) / inner / nodes;          // per NODE
        if (r > 0 && dt < best) best = dt;                       // round 0 is warm-up
    }
    ggml_gallocr_free(al);
    ggml_free(gc);
    ggml_free(dctx);
    ggml_backend_free(backend);
    return best;
}

static void sweep(const char* title, const int64_t* ne0s, const int64_t* rowss, int n, int threads) {
    std::printf("\n%s   (per node, %d threads against 1)\n", title, threads);
    std::printf("%8s %7s %10s %12s %12s %9s   %s\n",
                "ne0", "rows", "elements", "1 thread us", "N thread us", "speedup", "verdict");
    for (int i = 0; i < n; ++i) {
        const int64_t ne0 = ne0s[i], rows = rowss[i];
        const int nodes = nodes_for(ne0 * rows);
        const double t1 = time_one(ne0, rows, 1, nodes);
        const double tn = time_one(ne0, rows, threads, nodes);
        const double s = t1 / tn;
        std::printf("%8lld %7lld %10lld %12.2f %12.2f %9.2f   %s\n",
                    (long long) ne0, (long long) rows, (long long) (ne0 * rows),
                    t1 * 1e6, tn * 1e6, s,
                    s >= 1.10 ? "thread it" : (s <= 0.95 ? "WORSE THREADED" : "no gain"));
    }
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);            // rows appear as they land, even down a pipe
    const int threads = argc > 1 ? std::atoi(argv[1]) : 4;
    g_op = argc > 2 ? std::atoi(argv[2]) : 0;
    std::printf("op = %s\n", g_op ? "SILU" : "TANH");

    // Sweep 1: the row floor. Kokoro's [256, 1] is the first line.
    const int64_t s1_ne0[]  = {256, 256, 256, 256, 256, 256, 256, 256, 256};
    const int64_t s1_rows[] = {1, 2, 3, 4, 8, 16, 64, 192, 512};
    sweep("SWEEP 1 -- rows, at ne0 = 256", s1_ne0, s1_rows, 9, threads);

    // Sweep 2: the work floor, at a row count that can actually split. VITS's gate is ne0 = 286.
    const int64_t s2_ne0[]  = {8, 16, 32, 64, 128, 286, 512, 1024, 4096, 16384};
    const int64_t s2_rows[] = {192, 192, 192, 192, 192, 192, 192, 192, 192, 192};
    sweep("SWEEP 2 -- ne0, at rows = 192", s2_ne0, s2_rows, 10, threads);
    return 0;
}
