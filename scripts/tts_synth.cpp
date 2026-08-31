// Synthesise one utterance from any of loom's four TTS families and write it as a WAV, on any device.
//
// NOT part of the build -- a standalone measurement, kept because it is the FRONT HALF OF THE ASR
// ORACLE and that oracle is the only trustworthy verdict on a TTS change (Retro-006: cosine 0.996
// shipped noise; P4.13/P4.28: correlation 0.025 shipped speech). Without it, "does this still say the
// sentence" has to be re-implemented from scratch every time, which is how it stops being asked.
//
//   g++ -O2 -std=c++17 -I include -I build/_deps/ggml-src/include \
//       -I build/_deps/nlohmann_json-src/include scripts/tts_synth.cpp -o tts_synth \
//       -L build -lloom_engine -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread
//
//   scripts/tts_ids.py "Hey, can you shut down the computer, my friend?" <model.gguf> ids.txt
//   ./tts_synth <model.gguf> <vits|matcha|kokoro|styletts2> ids.txt <rate> out.wav [--device gpu] \
//               [--ref-s kokoro.ref_s.txt]
//   # then resample to 16 kHz and transcribe:
//   ./build/tools/loom_cli/loom_cli --model whisper-small.gguf --wav out16k.wav --language en
//
// The families differ only in what `infer` is called with, which is what the gate tests show and all
// this file really encodes: vits {token_ids, seed, noise_scale, length_scale, noise_scale_w},
// matcha {tokens, n_steps, seed}, kokoro {input_ids, ref_s, speed, seed},
// styletts2 {input_ids, diffusion_steps, seed}. Sample rates: vits/matcha 22050, kokoro/styletts2
// 24000 (kokoro declares `loom.sample_rate`; the others do not, hence the argument).
//
// TWO THINGS IT PRINTS BESIDES THE AUDIO, both of which have caught something:
//
//   * peak and rms. Real speech lands near +-0.3; anything leaving [-1, 1] means the conditioning is
//     wrong rather than the vocoder (see feedback on the Kokoro noise-voice bug).
//   * `device_report()`, on a device build. A scheduler that hands every node back to the CPU produces
//     exactly the same correct audio, so a GPU claim without this line is vacuous -- it is what turned
//     "a folded quantized kernel runs on Vulkan" into a real result in P4.13 (0 fallback nodes).
//
// `synthetic:N` instead of an ids file builds a BOS/blank/EOS sequence of N phonemes with no phonemizer
// involved. That is not speech and must never be transcribed; it exists to sweep LENGTH, which is how
// P4.28 checked that removing VITS's static relative-position pad kept both of the real code's branches
// (n=2 and 4 are the crop branch, 2202 and 5002 are past the bound the old export threw at).
#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

static void w32(FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
static void w16(FILE* f, uint16_t v) { std::fwrite(&v, 2, 1, f); }

static std::vector<double> read_numbers(const std::string& path) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    std::vector<double> v; double x;
    while (in >> x) v.push_back(x);
    return v;
}

// The BOS/blank/EOS shape a piper-style phoneme sequence has, with arbitrary ids in the middle. See the
// header: for sweeping length only.
static std::vector<double> synthetic_ids(int n) {
    std::vector<double> ids{1};
    for (int i = 0; i < n; ++i) { ids.push_back(20 + (i % 90)); ids.push_back(0); }
    ids.push_back(2);
    return ids;
}

static void write_wav(const std::string& path, const std::vector<double>& audio, uint32_t rate) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::perror("fopen"); std::exit(1); }
    const uint32_t n = static_cast<uint32_t>(audio.size());
    std::fwrite("RIFF", 1, 4, f); w32(f, 36 + n * 2); std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); w32(f, 16); w16(f, 1); w16(f, 1);
    w32(f, rate); w32(f, rate * 2); w16(f, 2); w16(f, 16);
    std::fwrite("data", 1, 4, f); w32(f, n * 2);
    for (double s : audio) {
        const double c = std::max(-1.0, std::min(1.0, s));
        w16(f, static_cast<uint16_t>(static_cast<int16_t>(std::lround(c * 32767.0))));
    }
    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: %s <gguf> <vits|matcha|kokoro|styletts2> <ids.txt|synthetic:N> <rate> <out.wav|->\n"
            "          [--device NAME] [--ref-s FILE] [--seed N]\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1], family = argv[2], ids_arg = argv[3], out = argv[5];
    const uint32_t rate = static_cast<uint32_t>(std::atoi(argv[4]));
    std::string device_name = "cpu", ref_s_path;
    double seed = 42.0;
    for (int i = 6; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], "--device") == 0) device_name = argv[++i];
        else if (std::strcmp(argv[i], "--ref-s") == 0) ref_s_path = argv[++i];
        else if (std::strcmp(argv[i], "--seed") == 0) seed = std::atof(argv[++i]);
    }

    const std::vector<double> ids = ids_arg.rfind("synthetic:", 0) == 0
        ? synthetic_ids(std::atoi(ids_arg.c_str() + 10))
        : read_numbers(ids_arg);
    if (ids.empty()) { std::fprintf(stderr, "no ids\n"); return 2; }

    try {
        loom::Device device = loom::Device::open(device_name);
        loom::Backends backends = device.backends();
        auto model = loom::GgufModel::load(gguf, backends.primary);
        if (!model) { std::fprintf(stderr, "load failed: %s\n", gguf.c_str()); return 1; }

        loom::LoomLuaBridge bridge(backends);
        // Every topology the file declares, whatever the family: kokoro has 27 and vits has 2, and a
        // driver that calls one this did not register raises rather than misbehaving.
        for (const std::string& name : model->topology_names()) {
            bridge.register_module(name, *model,
                                    loom::GraphTopology::parse(model->topology_json(name)));
        }
        bridge.load_script(model->kv_str("model.driver_script"));

        std::unordered_map<std::string, loom::LoomLuaBridge::Value> args;
        if (family == "vits") {
            // The three scales PINNED, as bench_vits_loom.cpp pins them and for the same reason: VITS's
            // duration predictor is stochastic, so an unpinned pair of runs synthesises two different
            // utterances and nothing downstream is comparable.
            args = {{"token_ids", ids}, {"seed", seed},
                    {"noise_scale", 0.0}, {"length_scale", 1.0}, {"noise_scale_w", 0.0}};
        } else if (family == "matcha") {
            args = {{"tokens", ids}, {"n_steps", 10.0}, {"seed", seed}};
        } else if (family == "styletts2") {
            args = {{"input_ids", ids}, {"diffusion_steps", 5.0}, {"seed", seed}};
        } else if (family == "kokoro") {
            if (ref_s_path.empty()) {
                std::fprintf(stderr, "kokoro needs --ref-s (scripts/tts_ids.py writes one from the "
                                     "file's own loom.default_style.ref_s)\n");
                return 2;
            }
            args = {{"input_ids", ids}, {"ref_s", read_numbers(ref_s_path)},
                    {"speed", 1.0}, {"seed", seed}};
        } else {
            std::fprintf(stderr, "unknown family '%s'\n", family.c_str());
            return 2;
        }

        const auto& audio = std::get<std::vector<double>>(bridge.call("infer", args));
        double peak = 0.0, energy = 0.0;
        for (double s : audio) { peak = std::max(peak, std::fabs(s)); energy += s * s; }
        std::printf("%-10s n_ids=%zu samples=%zu peak=%.4f rms=%.5f\n", family.c_str(), ids.size(),
                    audio.size(), peak, std::sqrt(energy / std::max<size_t>(1, audio.size())));
        for (const auto& m : bridge.device_report()) {
            std::printf("  module=%-20s splits=%d device_nodes=%zu fallback_nodes=%zu\n",
                        m.module.c_str(), m.splits, m.device_nodes, m.fallback_nodes);
        }
        if (out != "-") write_wav(out, audio, rate);
    } catch (const std::exception& e) {
        // Printed rather than thrown, because a LENGTH sweep wants the message: the engine's VIEW
        // bounds check is what reported VITS's old ~2053-token ceiling, naming the tensor and offset.
        std::printf("THREW: %s\n", e.what());
        return 1;
    }
    return 0;
}
