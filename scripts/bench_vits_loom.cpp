// loom's side of the VITS comparison against onnxruntime: wall time for one synthesis of the SAME
// utterance bench_onnx.py times, with the same three scales PINNED.
//
// NOT part of the build -- a standalone measurement, kept because "what did P4.15 actually buy" is a
// per-machine question and has to stay re-runnable on the next machine.
//
//   g++ -O3 -std=c++17 -I include -I tests/support -I build/_deps/ggml-src/include \
//       -I build/_deps/nlohmann_json-src/include \
//       scripts/bench_vits_loom.cpp -o bench_vits_loom -L build -lloom_engine \
//       -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread
//
// The nlohmann include is not optional and was missing from this line until someone tried to use it:
// loom.h reaches graph_topology.h, which includes <nlohmann/json.hpp>. On macOS, swap `g++` for
// `clang++`, drop `-lpthread`, and add `-Wl,-rpath,<abs build dir>` twice -- once for libloom_engine
// and once for _deps/ggml-build/src -- since there is no LD_LIBRARY_PATH equivalent that survives.
//   ./bench_vits_loom <dir-with-vits_mil.gguf || path.gguf> [nrun] [device]   # device defaults to cpu
//
// THE PINNING IS THE POINT, and it is bench_onnx.py's point too: VITS's duration predictor is
// stochastic, both engines are near-linear in output samples, and an unpinned comparison times two
// different utterances. noise_scale = noise_scale_w = 0 and length_scale = 1 are onnxruntime's
// `scales = [0.0, 1.0, 0.0]`, so both sides synthesise the same number of samples -- which the two
// harnesses print, and which must match before any ratio is believed.
#include "loom/loom.h"
#include "loom/core/profile.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// The audio's identity, printed beside the timing, so a SCHEDULING change (a thread count, a job
// split, a fusion) can be shown not to have moved a bit rather than argued not to have. Two runs that
// disagree here are not two measurements of the same thing, whatever their sample counts say.
static uint64_t fnv1a(const std::vector<double>& v) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
    for (size_t i = 0; i < v.size() * sizeof(double); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <dir-or-gguf> [nrun] [device]\n", argv[0]);
        return 2;
    }
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
    //
    // The device is an ARGUMENT, defaulting to "cpu" so every existing invocation means what it did.
    // It is here because "which device is faster for this model" turned out to be a per-MODEL
    // question and not only a per-machine one: on an M1 Pro, Metal is 1.52x FASTER than the CPU on
    // whisper-small and 5.07x SLOWER on this file (P4.11, P4.30a).
    //
    // Run it with GGML_SCHED_DEBUG=1 and count `## SPLIT` lines per backend before believing a
    // timing -- a first measurement on a new backend measures how much of the graph fell back. But
    // DO NOT THEN READ THE TIMING OFF THE SPLIT COUNT, which is the trap this file has now walked
    // into twice. VITS splits 27 ways onto the CPU because ggml-metal declines a `PAD` with a
    // nonzero leading pad, and collapsing all 27 is worth 1.8%; the time was in two convolution
    // kernels that never fell back at all, one of which was dispatching a single thread per
    // threadgroup (`ggml-0014`, worth 1.77x). Epic-04 SS5.7.
    const std::string dev_spec = argc > 3 ? argv[3] : "cpu";
    loom::Device device = loom::Device::open(dev_spec);
    std::fprintf(stderr, "device: %s (%s)\n", device.name().c_str(), device.description().c_str());
    loom::Backends backends = device.backends();

    // A directory (the original contract, `<dir>/vits_mil.gguf`) or a .gguf path directly -- the
    // second because the file this now gets pointed at is usually an exported `vits-f32-dyn.gguf`
    // sitting beside other models rather than a directory of one.
    const std::string model_path =
        dir.size() > 5 && dir.compare(dir.size() - 5, 5, ".gguf") == 0 ? dir : dir + "/vits_mil.gguf";
    auto model = loom::GgufModel::load(model_path, backends.primary);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }

    loom::LoomLuaBridge bridge(backends);
    bridge.register_module("text", *model, loom::GraphTopology::parse(model->topology_json("text")));
    bridge.register_module("flow_vocoder", *model,
                           loom::GraphTopology::parse(model->topology_json("flow_vocoder")));
    bridge.load_script(model->kv_str("model.driver_script"));

    std::vector<double> last_audio;
    auto once = [&]() {
        loom::LoomLuaBridge::Value r = bridge.call("infer", {
            {"token_ids", token_ids},
            {"seed", 42.0},
            {"noise_scale", 0.0},          // onnxruntime's scales[0]
            {"length_scale", 1.0},         //                 scales[1]
            {"noise_scale_w", 0.0},        //                 scales[2]
        });
        last_audio = std::get<std::vector<double>>(r);
        return last_audio.size();
    };

    const size_t n_samples = once();        // warm: first call builds the graphs
    const uint64_t digest = fnv1a(last_audio);
    // Drop the warm-up's nodes from any `$LOOM_PROFILE` report. The first call builds and allocates
    // the graphs, so every buffer it touches takes a first-touch page fault -- a cost that belongs to
    // neither the op it lands on nor the steady state the rest of this loop measures. A no-op when
    // profiling is off.
    loom::profile::reset();
    std::vector<double> ts;
    for (int i = 0; i < nrun; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        once();
        ts.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    // A real median rather than `ts[ts.size()/2]`, which at an EVEN nrun is the larger of the two
    // middle samples. Retro-018's correction caught that in bench_asr_loom.cpp at nrun=2, where it
    // reported the max of a cold run and a warm one; the same index was here, harmless at the default
    // nrun=9 and wrong the moment anyone passes an even one.
    const double median = ts.size() % 2 == 1 ? ts[ts.size()/2]
                                             : 0.5 * (ts[ts.size()/2 - 1] + ts[ts.size()/2]);
    std::printf("loom   vits  [%s]  samples=%zu  fnv1a=%016llx  median %.4f s  min %.4f s  (n=%d)\n",
                dev_spec.c_str(), n_samples, static_cast<unsigned long long>(digest),
                median, ts.front(), nrun);
    return 0;
}
