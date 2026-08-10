// GigaAM v3 decodes through its own embedded Lua driver, and agrees with the checkpoint's own
// `transcribe()` token-for-token (BACKLOG.md P4.2).
//
// **What this test is actually for.** Nothing in the driver, the decode loop or the four phase names is
// GigaAM's -- they are `transducer_export.py`'s, shared verbatim with Parakeet TDT/RNN-T, whose own e2e
// test sits beside this one. What is GigaAM's is the *loader*: an HF directory with a
// `trust_remote_code` modeling file, loaded through `AutoModel.from_pretrained` instead of
// `ASRModel.restore_from`. So the property under test is that a second loader produces an artifact the
// first loader's template can run unchanged -- which is only observable end to end, on real audio.
//
// It also covers the two things `gigaam_export.py` had to do to that checkpoint before it would trace,
// and neither of them is visible from the export succeeding:
//
//   * the mel frontend is a **rewrite** of torchaudio's `MelSpectrogram` (coremltools cannot lower the
//     `complex_shape` op torchaudio's own batch-reshape emits). The export compares the two on a chirp
//     and refuses a mismatch, but only a real transcript shows the rewrite is the *right* frontend for
//     this model rather than merely a self-consistent one.
//   * the prediction network is ONE layer, where Parakeet's is two. `RecurrentPhase` names a single
//     cell `pred_lstm_fwd` unless asked to number it, and the driver's loop composes
//     `'pred_lstm_l' .. (l - 1) .. '_fwd'`. A wrong answer there is a missing topology at load, which is
//     exactly what this test would report first.
//
// The expectation is `RNNTGreedyDecoding._greedy_decode`'s own hypothesis for `samples/jfk.wav`, kept as
// ids rather than text: 80 tokens decoding to "En so my fellow americans noth wat your cuntry can dou
// for you. Can dow for your cuntry." -- GigaAM v3 is a Russian(+EN) model and this is English speech, so
// the transcript is rough. That is not a problem for an oracle: it is what the checkpoint does, and a
// wrong frontend or a wrong blank id moves it immediately.
//
// Set LOOM_GIGAAM_MIL_GGUF (produced by `loom-export <dir> --task automatic-speech-recognition --model
// gigaam-rnnt`); the test skips cleanly if it is absent.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

// Minimal 16-bit PCM mono reader; `tools/loom_cli/wav_file.h` has one but belongs to the CLI target.
std::vector<float> read_wav_pcm16_mono(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    char riff[12] = {};
    f.read(riff, 12);
    if (std::string(riff, 4) != "RIFF" || std::string(riff + 8, 4) != "WAVE") return {};
    uint16_t channels = 0, bits = 0;
    while (f) {
        char id[4] = {};
        uint32_t size = 0;
        f.read(id, 4);
        f.read(reinterpret_cast<char*>(&size), 4);
        if (!f) break;
        if (std::string(id, 4) == "fmt ") {
            std::vector<char> fmt(size);
            f.read(fmt.data(), size);
            if (size >= 16) {
                std::memcpy(&channels, fmt.data() + 2, 2);
                std::memcpy(&bits, fmt.data() + 14, 2);
            }
        } else if (std::string(id, 4) == "data") {
            if (bits != 16 || channels < 1) return {};
            std::vector<int16_t> pcm(size / 2);
            f.read(reinterpret_cast<char*>(pcm.data()), size);
            std::vector<float> out(pcm.size() / channels);
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = static_cast<float>(pcm[i * channels]) / 32768.0f;
            }
            return out;
        } else {
            f.seekg(size, std::ios::cur);
        }
    }
    return {};
}

// GigaAM v3 e2e_rnnt's own greedy hypothesis for `samples/jfk.wav`, verbatim.
const std::vector<int32_t> kExpected = {
    3, 639, 336, 3, 132, 99, 3, 252, 243, 3, 579, 79, 738, 99, 618, 3, 85, 252, 421, 684, 405, 132, 3,
    336, 99, 134, 290, 3, 618, 85, 134, 3, 243, 99, 205, 190, 3, 349, 205, 336, 134, 190, 243, 3, 349,
    405, 3, 225, 99, 205, 3, 579, 702, 3, 243, 99, 205, 1, 413, 405, 3, 225, 99, 618, 3, 579, 702, 3,
    243, 99, 205, 190, 3, 349, 205, 336, 134, 190, 243, 1,
};

} // namespace

int main() {
    const char* samples_env = std::getenv("LOOM_SAMPLES_DIR");
    const std::string jfk = std::string(samples_env != nullptr ? samples_env : "samples") + "/jfk.wav";
    const std::vector<float> waveform = read_wav_pcm16_mono(jfk);

    const char* gguf = loom_test::fixture_env("LOOM_GIGAAM_MIL_GGUF");
    if (waveform.empty() || gguf == nullptr || !path_exists(gguf)) {
        std::fprintf(stderr,
                      "skipping: need '%s' plus LOOM_GIGAAM_MIL_GGUF (loom-export <dir> --task "
                      "automatic-speech-recognition --model gigaam-rnnt)\n",
                      jfk.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf, backend.get());
    // Four, not five: this prediction network is one layer deep where Parakeet's is two, and the
    // per-layer cell naming is what the driver's computed call site has to agree with.
    const std::vector<std::string> names = model->topology_names();
    for (const std::string& name : names) std::fprintf(stderr, "topology: %s\n", name.c_str());
    LOOM_CHECK(names.size() == 4);

    loom::LoomLuaBridge bridge(backend.get());
    for (const std::string& name : names) {
        bridge.register_module(name, *model, loom::GraphTopology::parse(model->topology_json(name)));
    }
    bridge.load_script(model->kv_str("model.driver_script"));

    const std::vector<double> waveform_d(waveform.begin(), waveform.end());
    const std::vector<double> length_d{static_cast<double>(waveform.size())};
    const auto got = std::get<std::vector<double>>(bridge.call(
        "infer", {{"waveform", waveform_d}, {"length", length_d}}));

    std::fprintf(stderr, "gigaam-v3: driver -> %zu token(s), GigaAM -> %zu\n", got.size(),
                  kExpected.size());
    LOOM_CHECK(got.size() == kExpected.size());
    for (size_t i = 0; i < got.size() && i < kExpected.size(); ++i) {
        if (static_cast<int32_t>(got[i]) != kExpected[i]) {
            std::fprintf(stderr, "  token %zu: driver %d, GigaAM %d\n", i,
                          static_cast<int32_t>(got[i]), kExpected[i]);
        }
        LOOM_CHECK(static_cast<int32_t>(got[i]) == kExpected[i]);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
