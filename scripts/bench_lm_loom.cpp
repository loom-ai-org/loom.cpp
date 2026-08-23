// What the causal-LM decode loop costs through each of the two entry points a generated driver
// exports. NOT part of the build -- a standalone measurement.
//
//   g++ -O3 -std=c++17 -I include -I tests/support -I build/_deps/ggml-src/include \
//       -I build/_deps/nlohmann_json-src/include scripts/bench_lm_loom.cpp -o bench_lm_loom \
//       -L build -lloom_engine -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread
//   ./bench_lm_loom <causal_lm.gguf> [n_new]
//
// Every generated causal-LM driver exports BOTH `infer` -- one forward over the whole prompt,
// returning one token -- and `infer_with_past`, which runs the decode loop itself against the KV
// cache and returns the whole sequence. `loom::text::generate` calls `infer` unconditionally
// (src/core/text_generate.cpp:58), so the host re-feeds a growing prompt and every step recomputes
// the entire sequence. This times both so the difference is a number rather than an argument.
#include "loom/loom.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <gguf> [n_new]\n", argv[0]); return 2; }
    const std::string path = argv[1];
    const int n_new = argc > 2 ? std::atoi(argv[2]) : 65;

    // "The capital of France is", Qwen3 BPE.
    const std::vector<double> prompt = {785, 6722, 315, 9625, 374};

    // Device::open rather than a bare CPU backend: it is what applies $LOOM_N_THREADS.
    loom::Device device = loom::Device::open("cpu");
    loom::Backends backends = device.backends();

    auto model = loom::GgufModel::load(path, backends.primary);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }

    // A Session, not a bare bridge: it is what ALLOCATES THE KV CACHE the topology declares, and
    // `infer_with_past` throws without one ("no KvCache was provided to GraphBuilder"). This is the
    // same object loom_cli builds, so both timings below are the engine's real configuration.
    loom::Session session(*model, backends);
    loom::LoomLuaBridge& bridge = session.bridge();

    // `infer_with_past`: the driver's own loop, one token of new work per step.
    const auto t0 = std::chrono::steady_clock::now();
    loom::LoomLuaBridge::Value r = bridge.call("infer_with_past", {
        {"tokens", prompt},
        {"max_new_tokens", static_cast<double>(n_new)},
        {"eos_token", -1.0},
    });
    const double with_past = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const size_t got = std::get<std::vector<double>>(r).size();

    // `infer`: what the engine actually calls -- re-fed growing prompt, whole sequence every step.
    std::vector<double> running = prompt;
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_new; ++i) {
        loom::LoomLuaBridge::Value one = bridge.call("infer", {{"tokens", running}});
        running.push_back(std::get<double>(one));
    }
    const double refeed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    std::printf("infer_with_past  %6.2f s  (%zu tokens, %.2f tok/s)\n", with_past, got, got / with_past);
    std::printf("infer (re-fed)   %6.2f s  (%d tokens, %.2f tok/s)\n", refeed, n_new, n_new / refeed);
    std::printf("speedup available: %.2fx\n", refeed / with_past);
    return 0;
}
