// Nemotron 3.5 ASR decodes through its own embedded Lua driver, and agrees with `transformers`'
// `Nemotron3_5AsrForRNNT.generate()` token for token (P5, requested for the zoo 2026-09-25).
//
// The decode loop is `transducer_export.py`'s, shared with Parakeet and GigaAM, whose own tests sit
// beside this one. What is Nemotron's, and what this test is for:
//
//   * **the language prompt**, a `[1]` int input of the encoder that the driver fills from
//     `inputs.language` (else the checkpoint's `auto` prompt, 101). Its contribution is a 2048-wide
//     embedding row added to every frame, so a prompt that never arrived would still transcribe -- it
//     would just transcribe as `auto`. So the test asks for a WRONG language too: told English speech
//     is German, the checkpoint writes nothing at all (`transformers` returns an empty sequence for
//     de-DE, ja-JP and fr-FR alike), and so must the file. That row fails if the prompt is dropped.
//   * **the traced front end and mask.** The export rewrites the feature extractor (a padding that
//     yields exactly the valid frames) and builds the chunked-limited attention mask as a BOOLEAN,
//     because `transformers`' eager path inverts its own additive one (Retro-079). Either wrong
//     moves the tokens immediately.
//   * **the language tags as control ids.** The model writes `<en-US>` after each sentence;
//     `transcribe` must drop them, as `transformers`' `skip_special_tokens` does, and resolve the
//     name "de-DE" through the contract's language table.
//
// Expectations: `transformers` 5.14.1, `attn_implementation="sdpa"`, on `samples/jfk.wav`.
//
// Set LOOM_NEMOTRON_ASR_GGUF (`loom-export <nemotron-3.5-asr-streaming-0.6b dir>`, in the ovos venv);
// the test skips cleanly if it is absent.

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
#include <unordered_map>
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

// `generate()`'s tokens for JFK with the prompt left to `auto` (101) and with `en-US` (0): identical.
// 2947 is `<en-US>`, written after each sentence.
const std::vector<int32_t> kExpected = {
    2860, 2, 1290, 2, 1731, 1179, 290, 133, 309, 271, 581, 1257, 113, 40, 2900, 2847, 2831, 2815, 995,
    313, 113, 34, 1752, 2813, 2, 1288, 398, 2810, 4, 2, 2947, 309, 2127, 2831, 2810, 2813, 2, 1288, 398,
    2815, 995, 313, 113, 34, 1752, 4, 2, 2947,
};
// `processor.decode(..., skip_special_tokens=True)`, verbatim -- the double space is where a tag was.
const char* kExpectedText =
    "And so my fellow Americans ask not what your country can do for you.  Ask what you can do for your "
    "country. ";

std::vector<int32_t> decode(loom::LoomLuaBridge& bridge, const std::vector<double>& waveform, int language) {
    std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs{{"waveform", waveform}};
    if (language >= 0) inputs["language"] = static_cast<double>(language);
    const auto got = std::get<std::vector<double>>(bridge.call("infer", inputs));
    return std::vector<int32_t>(got.begin(), got.end());
}

bool same(const char* what, const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
    std::fprintf(stderr, "nemotron-asr %s: driver -> %zu token(s), transformers -> %zu\n", what, got.size(),
                  want.size());
    bool ok = got.size() == want.size();
    for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
        if (got[i] != want[i]) {
            std::fprintf(stderr, "  token %zu: driver %d, transformers %d\n", i, got[i], want[i]);
            ok = false;
        }
    }
    return ok;
}

} // namespace

int main() {
    const char* samples_env = std::getenv("LOOM_SAMPLES_DIR");
    const std::string jfk = std::string(samples_env != nullptr ? samples_env : "samples") + "/jfk.wav";
    const std::vector<float> waveform = read_wav_pcm16_mono(jfk);

    const char* gguf = loom_test::fixture_env("LOOM_NEMOTRON_ASR_GGUF");
    if (waveform.empty() || gguf == nullptr || !path_exists(gguf)) {
        std::fprintf(stderr, "skipping: need '%s' plus LOOM_NEMOTRON_ASR_GGUF (loom-export "
                              "<nemotron-3.5-asr-streaming-0.6b dir>)\n", jfk.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf, backend.get());
    loom::LoomLuaBridge bridge(backend.get());
    for (const std::string& name : model->topology_names()) {
        bridge.register_module(name, *model, loom::GraphTopology::parse(model->topology_json(name)));
    }
    bridge.load_script(model->kv_str("model.driver_script"));

    const std::vector<double> waveform_d(waveform.begin(), waveform.end());
    LOOM_CHECK(same("auto (no language)", decode(bridge, waveform_d, -1), kExpected));
    LOOM_CHECK(same("en-US (prompt 0)", decode(bridge, waveform_d, 0), kExpected));
    LOOM_CHECK(same("de-DE (prompt 9), the wrong language", decode(bridge, waveform_d, 9), {}));

    // The text door: tags dropped, the name resolved through the contract's language table.
    const loom::audio::Transcription text = loom::audio::transcribe(bridge, *model, waveform);
    std::fprintf(stderr, "nemotron-asr text: \"%s\"\n", text.text.c_str());
    LOOM_CHECK(text.text == kExpectedText);
    loom::audio::TranscribeOptions german;
    german.language = "de-DE";
    LOOM_CHECK(loom::audio::transcribe(bridge, *model, waveform, german).text.empty());

    LOOM_TEST_REPORT_AND_RETURN();
}
