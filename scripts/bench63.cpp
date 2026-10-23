// P4.35: whisper's attention matmuls at the MODEL's shape (12 heads, 3-D), under run-time schedules of
// tinyBLAS's F32 GEMM, with every arm in ONE process.
//
// NOT part of the build.
//
// WHY IT EXISTS. On a Raspberry Pi 4 the encoder's `A@V` (m = 64, n = 1500, k = 1500, 12 heads) scales
// 2.0x from one thread to four, where QK^T scales 3.4x and the dense GEMMs 2.85x (`$LOOM_PROFILE`,
// 2026-10-22), and it is 1.86x behind onnxruntime at four threads. bench19 has the shape too, but as
// one 2-D head on one pair of inputs; this runs the 3-D node the export emits, on DISTINCT inputs per
// node, because the 9 MB per head of attention probabilities is what does not fit and re-reading the
// same tensor would flatter every arm.
//
// It needs `scripts/probes/ggml-p435-sgemm-schedule-probe.patch` (see `scripts/probes/README.md`),
// which turns four of tinyBLAS's F32 schedule choices into run-time knobs, each -1 for "as shipped":
//
//   bm     row blocks per job (2/4/8/16); a job owns 4*bm rows of A
//   bn     column-block width in RN tiles (overrides the L2 cap ggml-0030 derives)
//   kc     K chunk: 0 = off, N = force N, -4 = the rule before ggml-0045 (no chunk below 2 kc, which
//          skipped A@V: k = 1500 < 2 * 768). The sweep that chose ggml-0045's predicate ran an earlier
//          probe against the pre-0045 tree, whose -2 / -3 arms were the candidate rules.
//   order  0 = jobs walk rows first (shipped), 1 = columns first
//
// BUILD (on the board, against a loom.cpp build configured with the probe):
//
//   g++ -O3 -std=c++17 -mcpu=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench63.cpp -o bench63 \
//       -L <ggml-build>/src -lggml -lggml-base -lggml-cpu -lpthread -lm
//
//   ./bench63 <threads> <rounds> <shape: av|qk|fc2|MxNxKxH> <arm> [<arm> ...]
//       arm = "bm,bn,kc,order", the first is the baseline every ratio is against
//
// ESTIMATOR. Each round runs every arm once, the arm order rotated round to round; the ratio to the
// baseline is taken per round and medianed (Retro-012). Every arm's output is hashed after its first
// run and must equal the baseline's: none of these knobs changes the order of any output's sum.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void loom_sgemm_probe_set(int bm, int bn, int kc, int order);

static double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct Arm { int bm, bn, kc, order; std::string label; };

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

static uint64_t fnv(const ggml_tensor * t) {
    const uint8_t * p = (const uint8_t *) t->data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < ggml_nbytes(t); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s threads rounds av|qk|fc2 arm [arm...]\n", argv[0]);
        return 2;
    }
    const int threads = atoi(argv[1]), rounds = atoi(argv[2]);
    const std::string shape = argv[3];
    std::vector<Arm> arms;
    for (int i = 4; i < argc; ++i) {
        Arm a{};
        if (sscanf(argv[i], "%d,%d,%d,%d", &a.bm, &a.bn, &a.kc, &a.order) != 4) return 2;
        a.label = argv[i];
        arms.push_back(a);
    }

    // ggml_mul_mat(a, b): a is [k, m, heads], b is [k, n, heads], out [m, n, heads].
    int64_t m, n, k, heads;
    if (shape == "av")       { m = 64;   n = 1500; k = 1500; heads = 12; }   // V^T x P
    else if (shape == "qk")  { m = 1500; n = 1500; k = 64;   heads = 12; }   // K x Q
    else if (shape == "fc2") { m = 768;  n = 1500; k = 3072; heads = 1;  }
    else if (sscanf(shape.c_str(), "%ldx%ldx%ldx%ld", &m, &n, &k, &heads) != 4) return 2;   // MxNxKxH
    // One node per "layer", each on its own inputs; small shapes get more nodes so a graph is not one
    // barrier and nothing else.
    const int nodes = std::max<int64_t>(4, std::min<int64_t>(32, (int64_t) (64e6 / (2.0 * m * n * k * heads + 1))));

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    const size_t bytes = (size_t) nodes * ((m + n) * k * heads * sizeof(float) + 4096) + (16u << 20);
    ggml_init_params dp = { bytes, nullptr, false };
    ggml_context * dctx = ggml_init(dp);
    ggml_init_params gp = { (size_t) 16u << 20, nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    std::vector<ggml_tensor *> outs;
    uint32_t seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float) (seed >> 8) / 16777216.f; };
    for (int j = 0; j < nodes; ++j) {
        ggml_tensor * a = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, k, m, heads);
        ggml_tensor * b = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, k, n, heads);
        float * pa = (float *) a->data;
        float * pb = (float *) b->data;
        for (int64_t i = 0; i < ggml_nelements(a); ++i) pa[i] = rnd() - 0.5f;
        // For A@V, b is a probability row: positive, summing to ~1, like softmax's output.
        for (int64_t i = 0; i < ggml_nelements(b); ++i) pb[i] = shape == "av" ? rnd() * (2.f / k) : rnd() - 0.5f;
        ggml_tensor * c = ggml_mul_mat(gctx, a, b);
        ggml_set_output(c);
        ggml_build_forward_expand(gf, c);
        outs.push_back(c);
    }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    const double flop = 2.0 * m * n * k * heads;
    auto run = [&](const Arm & a) {
        loom_sgemm_probe_set(a.bm, a.bn, a.kc, a.order);
        const double t0 = now();
        ggml_backend_graph_compute(backend, gf);
        return (now() - t0) / nodes;
    };

    // Warm-up + correctness: every arm must produce the baseline's bytes.
    std::vector<uint64_t> base;
    bool ok = true;
    for (size_t i = 0; i < arms.size(); ++i) {
        run(arms[i]);
        for (int j = 0; j < nodes; ++j) {
            const uint64_t h = fnv(outs[j]);
            if (i == 0) base.push_back(h);
            else if (h != base[j]) { ok = false; std::printf("MISMATCH arm %s node %d\n", arms[i].label.c_str(), j); }
        }
    }
    std::printf("shape %s  m=%lld n=%lld k=%lld heads=%lld  threads=%d  rounds=%d  outputs %s\n",
                shape.c_str(), (long long) m, (long long) n, (long long) k, (long long) heads, threads, rounds,
                ok ? "IDENTICAL across arms" : "DIFFER");

    std::vector<std::vector<double>> t(arms.size()), ratio(arms.size());
    for (int r = 0; r < rounds; ++r) {
        std::vector<double> tr(arms.size());
        for (size_t s = 0; s < arms.size(); ++s) {
            const size_t i = (s + r) % arms.size();
            tr[i] = run(arms[i]);
        }
        for (size_t i = 0; i < arms.size(); ++i) {
            t[i].push_back(tr[i]);
            ratio[i].push_back(tr[0] / tr[i]);
        }
    }
    std::printf("%-18s %10s %9s %8s %15s\n", "arm bm,bn,kc,ord", "ms/node", "GFLOP/s", "x base", "range");
    for (size_t i = 0; i < arms.size(); ++i) {
        std::vector<double> s = ratio[i];
        std::sort(s.begin(), s.end());
        const double mt = median(t[i]);
        std::printf("%-18s %10.2f %9.2f %8.3f %7.3f-%.3f\n", arms[i].label.c_str(), mt * 1e3, flop / mt / 1e9,
                    median(ratio[i]), s.front(), s.back());
    }
    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    ggml_free(dctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
