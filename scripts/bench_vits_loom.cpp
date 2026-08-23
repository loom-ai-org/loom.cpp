// loom's side of the VITS comparison against onnxruntime: wall time for one synthesis of the SAME
// utterance bench_onnx.py times, with the same three scales PINNED.
//
// NOT part of the build -- a standalone measurement, kept because "what did P4.15 actually buy" is a
// per-machine question and has to stay re-runnable on the next machine.
//
//   g++ -O3 -std=c++17 -I include -I tests/support -I build/_deps/ggml-src/include \
//       scripts/bench_vits_loom.cpp -o bench_vits_loom -L build -lloom_engine \
//       -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread
//   ./bench_vits_loom <dir-with-vits_mil.gguf> [nrun]
//
// THE PINNING IS THE POINT, and it is bench_onnx.py's point too: VITS's duration predictor is
// stochastic, both engines are near-linear in output samples, and an unpinned comparison times two
// different utterances. noise_scale = noise_scale_w = 0 and length_scale = 1 are onnxruntime's
// `scales = [0.0, 1.0, 0.0]`, so both sides synthesise the same number of samples -- which the two
// harnesses print, and which must match before any ratio is believed.
#include "loom/loom.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <dir> [nrun]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const int nrun = argc > 2 ? std::atoi(argv[2]) : 9;

    // The same sentence bench_onnx.py speaks, as piper phoneme ids from miro_en-GB.onnx.json:
    // "hˈeɪ, kæn juː ʃˈʌtdaʊn ðə kəmpjˈuːtɐ, maɪ fɹˈɛnd?"  -- BOS, blank-interleaved, EOS.
    const std::vector<double> token_ids = {
        1,20,0,120,0,18,0,74,0,8,0,3,0,23,0,39,0,26,0,3,0,22,0,33,0,122,0,3,0,96,0,120,0,102,0,32,0,
        17,0,14,0,100,0,26,0,3,0,41,0,59,0,3,0,23,0,59,0,25,0,28,0,22,0,120,0,33,0,122,0,32,0,50,0,8,
        0,3,0,25,0,14,0,74,0,3,0,19,0,88,0,120,0,61,0,26,0,17,0,13,0,2};

    // Device::open rather than a bare CPU backend, because that is what applies $LOOM_N_THREADS --
    // ggml's own default is 4 whatever the machine has, so a bare backend cannot be asked for 24.
    loom::Device device = loom::Device::open("cpu");
    loom::Backends backends = device.backends();

    auto model = loom::GgufModel::load(dir + "/vits_mil.gguf", backends.primary);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }

    loom::LoomLuaBridge bridge(backends);
    bridge.register_module("text", *model, loom::GraphTopology::parse(model->topology_json("text")));
    bridge.register_module("flow_vocoder", *model,
                           loom::GraphTopology::parse(model->topology_json("flow_vocoder")));
    bridge.load_script(model->kv_str("model.driver_script"));

    auto once = [&]() {
        loom::LoomLuaBridge::Value r = bridge.call("infer", {
            {"token_ids", token_ids},
            {"seed", 42.0},
            {"noise_scale", 0.0},          // onnxruntime's scales[0]
            {"length_scale", 1.0},         //                 scales[1]
            {"noise_scale_w", 0.0},        //                 scales[2]
        });
        return std::get<std::vector<double>>(r).size();
    };

    const size_t n_samples = once();        // warm: first call builds the graphs
    std::vector<double> ts;
    for (int i = 0; i < nrun; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        once();
        ts.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    std::printf("loom   vits  samples=%zu  median %.4f s  min %.4f s  (n=%d)\n",
                n_samples, ts[ts.size()/2], ts.front(), nrun);
    return 0;
}
