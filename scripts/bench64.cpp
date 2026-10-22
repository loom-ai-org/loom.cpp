// P4.36: a materialised transpose (`CONT` of a permuted view) under ggml's generic `dup` and under a
// blocked 4x4 one, both arms in ONE process.
//
// NOT part of the build. Needs `scripts/probes/ggml-p436-dup-transpose-probe.patch`, whose
// `loom_dup_probe_set(0|1)` flips `dup_bytes` between ggml's per-element path and the blocked one.
//
// WHY IT EXISTS. whisper-small's V^T copy, `[64, 12, 1500]` permuted to `[1500, 64, 12]`, runs twice per
// layer (the encoder's and the cross-KV phase's) at ~0.4 GB/s on a Raspberry Pi 4 and does not scale with
// threads: 0.58 s of a 25.6 s transcription, against 0.22 s for ALL of onnxruntime's transposes.
//
//   g++ -O3 -std=c++17 -mcpu=native -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench64.cpp -o bench64 -L <ggml-build>/src -lggml -lggml-base -lggml-cpu -lpthread -lm
//   ./bench64 <threads> <rounds> [d t h]       # default 64 1500 12: whisper's V^T
//
// Each round runs both arms (order alternating), the ratio is per round and medianed. Both arms' outputs
// are hashed and must be byte-identical: it is a copy.
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
#include <vector>

extern "C" void loom_dup_probe_set(int v);

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
static uint64_t fnv(const ggml_tensor * t) {
    const uint8_t * p = (const uint8_t *) t->data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < ggml_nbytes(t); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char ** argv) {
    const int threads = argc > 1 ? atoi(argv[1]) : 4, rounds = argc > 2 ? atoi(argv[2]) : 9;
    const int64_t D = argc > 5 ? atoll(argv[3]) : 64, T = argc > 5 ? atoll(argv[4]) : 1500,
                  H = argc > 5 ? atoll(argv[5]) : 12;
    const int nodes = 8;

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_init_params dp = { (size_t) nodes * D * T * H * 4 + (16u << 20), nullptr, false };
    ggml_context * dctx = ggml_init(dp);
    ggml_init_params gp = { (size_t) 16u << 20, nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gctx);
    std::vector<ggml_tensor *> outs;
    uint32_t seed = 7;
    for (int j = 0; j < nodes; ++j) {
        // value_states [D, H, T] -> PERMUTE(0,2,1,3) -> [D, T, H] -> PERMUTE(1,0,2,3) -> [T, D, H] -> CONT,
        // the chain whisper's export emits (encoder nodes 49-51).
        ggml_tensor * v = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, D, H, T);
        float * pv = (float *) v->data;
        for (int64_t i = 0; i < ggml_nelements(v); ++i) { seed = seed * 1664525u + 1013904223u; pv[i] = (float) (seed >> 8); }
        ggml_tensor * p = ggml_permute(gctx, ggml_permute(gctx, v, 0, 2, 1, 3), 1, 0, 2, 3);
        ggml_tensor * c = ggml_cont(gctx, p);
        ggml_set_output(c);
        ggml_build_forward_expand(gf, c);
        outs.push_back(c);
    }
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    auto run = [&](int arm) {
        loom_dup_probe_set(arm);
        const double t0 = now();
        ggml_backend_graph_compute(backend, gf);
        return (now() - t0) / nodes;
    };
    run(0);
    std::vector<uint64_t> h0;
    for (auto * o : outs) h0.push_back(fnv(o));
    for (auto * o : outs) memset(o->data, 0, ggml_nbytes(o));
    run(1);
    bool same = true;
    for (int j = 0; j < nodes; ++j) same &= fnv(outs[j]) == h0[j];

    std::vector<double> t0s, t1s, ratios;
    for (int r = 0; r < rounds; ++r) {
        double a, b;
        if (r % 2 == 0) { a = run(0); b = run(1); } else { b = run(1); a = run(0); }
        t0s.push_back(a); t1s.push_back(b); ratios.push_back(a / b);
    }
    std::sort(ratios.begin(), ratios.end());
    const double bytes = 2.0 * D * T * H * 4;
    const double m0 = median(t0s), m1 = median(t1s);
    std::printf("V^T [%lld,%lld,%lld] threads=%d  outputs %s\n", (long long) D, (long long) H, (long long) T, threads,
                same ? "IDENTICAL" : "DIFFER");
    std::printf("  ggml dup   %8.3f ms  %6.2f GB/s\n", m0 * 1e3, bytes / m0 / 1e9);
    std::printf("  blocked    %8.3f ms  %6.2f GB/s\n", m1 * 1e3, bytes / m1 / 1e9);
    std::printf("  speedup    %8.3f  (%.3f-%.3f)\n", median(ratios), ratios.front(), ratios.back());
    return same ? 0 : 1;
}
