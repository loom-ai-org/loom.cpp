// Parakeet TDT and RNN-T decode through their own embedded Lua driver, and agree with NeMo
// token-for-token (BACKLOG.md P4.0.17 step 2).
//
// This replaces `test_e2e_parakeet_{tdt,rnnt}.cpp`, and the replacement is not like-for-like -- it is a
// stronger oracle against a different artifact:
//
//   * **The artifact.** Those tests drove six separate bespoke GGUFs (encoder + four LSTM cells + joint)
//     through `loom::TdtDecoder`, C++ that carried the whole transducer double loop. Now there is ONE
//     GGUF holding four traced phases, and the loop is the driver's. No parakeet C++ remains.
//   * **The oracle.** They compared against `reference_forward_parakeet_{tdt,rnnt}.py`, a hand-rolled
//     PyTorch reimplementation run on a synthetic waveform -- and for RNN-T that fixture decodes to an
//     EMPTY token list, so the test could not tell a working decoder from a broken one. Here the
//     expectation is what NeMo's own `model.transcribe()` returns for 11 seconds of real speech.
//
// **That change of oracle found a real defect in what it replaced.** On `samples/jfk.wav` the retired
// C++ path emitted 36 tokens and NeMo emits 38: it dropped two `7877`s (the commas in "And so, my
// fellow Americans,"). The driver reproduces all 38, and the RNN-T model's 26, exactly. Had this test
// been written to match the path being removed -- which is what the migration plan originally said --
// it would have preserved the bug and called it a passing gate.
//
// The ids below are `model.transcribe()`'s own `y_sequence`, verbatim:
//   TDT  : "And so, my fellow Americans, ask not what your country can do for you, ask what you can do
//           for your country."
//   RNN-T: the same words, lowercase and unpunctuated -- a different vocabulary, hence entirely
//           different ids.
//
// Set LOOM_PARAKEET_TDT_MIL_GGUF / LOOM_PARAKEET_RNNT_MIL_GGUF (produced by
// `loom-export <checkpoint> --task automatic-speech-recognition --model parakeet-{tdt,rnnt}`);
// each case skips cleanly if its GGUF is absent.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

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

// `model.transcribe(["samples/jfk.wav"])[0].y_sequence`, for each checkpoint.
const std::vector<int32_t> kTdtExpected = {
    1976, 547, 7877, 1103, 309, 530, 596, 3213, 404, 667, 7877, 279, 583, 1491, 3470, 3629, 867, 331,
    958, 7893, 2059, 458, 509, 1180, 7877, 279, 583, 3470, 1180, 2059, 458, 509, 3629, 867, 331, 958,
    7893, 7883,
};
const std::vector<int32_t> kRnntExpected = {
    25, 75, 173, 281, 714, 988, 313, 719, 108, 130, 149, 606, 347, 117, 97, 69, 38, 719, 130, 38, 117,
    97, 69, 149, 606, 347,
};

bool run_case(const char* label, const std::string& gguf, const std::vector<float>& waveform,
              const std::vector<int32_t>& expected) {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf, backend.get());
    // One GGUF, every phase in it: encoder, embed, both prediction cells, joint. The driver names all
    // five and nothing else has to be loaded alongside.
    const std::vector<std::string> names = model->topology_names();
    LOOM_CHECK(names.size() == 5);

    loom::LoomLuaBridge bridge(backend.get());
    for (const std::string& name : names) {
        bridge.register_module(name, *model, loom::GraphTopology::parse(model->topology_json(name)));
    }
    bridge.load_script(model->kv_str("model.driver_script"));

    const std::vector<double> waveform_d(waveform.begin(), waveform.end());
    const std::vector<double> length_d{static_cast<double>(waveform.size())};
    const auto got = std::get<std::vector<double>>(bridge.call(
        "infer", {{"waveform", waveform_d}, {"length", length_d}}));

    std::fprintf(stderr, "%s: driver -> %zu token(s), NeMo -> %zu\n", label, got.size(),
                  expected.size());
    LOOM_CHECK(got.size() == expected.size());
    for (size_t i = 0; i < got.size() && i < expected.size(); ++i) {
        if (static_cast<int32_t>(got[i]) != expected[i]) {
            std::fprintf(stderr, "  token %zu: driver %d, NeMo %d\n", i,
                          static_cast<int32_t>(got[i]), expected[i]);
        }
        LOOM_CHECK(static_cast<int32_t>(got[i]) == expected[i]);
    }
    return true;
}

} // namespace

int main() {
    const char* samples_env = std::getenv("LOOM_SAMPLES_DIR");
    const std::string jfk = std::string(samples_env != nullptr ? samples_env : "samples") + "/jfk.wav";
    const std::vector<float> waveform = read_wav_pcm16_mono(jfk);

    const char* tdt = loom_test::fixture_env("LOOM_PARAKEET_TDT_MIL_GGUF");
    const char* rnnt = loom_test::fixture_env("LOOM_PARAKEET_RNNT_MIL_GGUF");
    const bool have_tdt = tdt != nullptr && path_exists(tdt);
    const bool have_rnnt = rnnt != nullptr && path_exists(rnnt);

    if (waveform.empty() || (!have_tdt && !have_rnnt)) {
        std::fprintf(stderr,
                      "skipping: need '%s' plus LOOM_PARAKEET_TDT_MIL_GGUF and/or "
                      "LOOM_PARAKEET_RNNT_MIL_GGUF (loom-export <checkpoint> --task "
                      "automatic-speech-recognition --model parakeet-tdt|parakeet-rnnt)\n",
                      jfk.c_str());
        return kSkipReturnCode;
    }

    if (have_tdt) run_case("parakeet-tdt", tdt, waveform, kTdtExpected);
    if (have_rnnt) run_case("parakeet-rnnt", rnnt, waveform, kRnntExpected);

    LOOM_TEST_REPORT_AND_RETURN();
}
