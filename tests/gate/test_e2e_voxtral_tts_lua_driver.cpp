// Validates the Voxtral-4B-TTS export (loom-exporter's `voxtral_tts_export.py`) end to end against the
// reference: text in through the file's own Tekken vocabulary, the default voice written over the
// prompt's [AUDIO] slots, the cached Mistral LM, the flow head's seven guided Euler steps per frame, each
// frame's 37 codes summed back into the LM, [END_AUDIO], and the codec to 24 kHz -- one GGUF, five
// topologies, one driver.
//
// **The oracle is vllm-omni's own flow head and codec** (`scripts/voxtral_tts_reference.py` imports the
// two module files as they are), with the one `[36]` draw per frame pinned and handed in (`noise`). The
// loop's state is CODES -- integers -- so unlike VoxCPM2's latents it does not drift: the reference's own
// f32 and f64 runs emit the same 142 frames, and so does the engine. Arms:
//
//   * **Free-running codes** (`return_codes`): every frame's 37 codes, [END_AUDIO] frame included,
//     IDENTICAL. The sharpest arm and the one that checks the loop.
//   * **Teacher-forced codes**: the same, fed the reference's frames -- the arm to trust on an ISA
//     where a code sitting on a rounding boundary flips (the fixture's `min_rounding_margin` is 3e-5
//     of a level) and the free-running arm then diverges for no fault of anyone's.
//   * **The waveform**, decoded in one codec call and in 40-frame chunks with the driver's left
//     context: both within the reference's own f32-vs-f64 spread.
//   * **Sabotage**: CFG 1.3 for 1.2 must move the codes, or the arms above prove nothing.
//
// Fixtures:
//   LOOM_VOXTRAL_TTS_GGUF     voxtral_tts.gguf  -- `loom-export ~/Dev/models/voxtral-4b-tts-2603 -o ...`
//   LOOM_VOXTRAL_TTS_REF_DIR  voxtral_tts_ref/  -- prompt_ids, noise, codes, wave (.npy) and meta.json
//                                                  from scripts/voxtral_tts_reference.py

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"
#include "loom/core/bpe_vocab.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Measured on the reference's 142-frame sentence (11.3 s), x86-64 (the workstation), 2026-09-26:
//   free-running and teacher-forced codes   142/142 frames identical
//   waveform, one codec call                max 5.6e-07  rmse 2.0e-08
//   waveform, 40-frame chunks + context     max 5.6e-07  rmse 2.4e-08
//   the reference's own f32 vs f64 wave     max 7.5e-07
// The bounds are ~20x the measurements; the codes have no bound, they are equal or they are not.
constexpr double VOXTRAL_WAVE_MAX_ABS = 1e-5;
constexpr double VOXTRAL_WAVE_RMSE = 5e-7;
constexpr int CODEBOOKS = 37;
// The prompt's markers (tekken.json), for reading the text ids back out of `prompt_ids`.
constexpr float AUDIO = 24.0f;

std::string meta_text(const std::string& json) {
    const std::string needle = "\"text\": \"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) throw std::runtime_error("voxtral_tts_ref/meta.json has no \"text\"");
    const size_t start = at + needle.size();
    return json.substr(start, json.find('"', start) - start);
}

std::vector<double> as_doubles(const std::vector<float>& v) { return {v.begin(), v.end()}; }

struct Diff {
    double max_abs = 0.0, rmse = 0.0, peak = 0.0;
};

Diff compare(const std::vector<double>& got, const std::vector<float>& want) {
    Diff d;
    double sum_sq = 0.0;
    const size_t n = std::min(got.size(), want.size());
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(got[i] - want[i]);
        d.max_abs = std::max(d.max_abs, e);
        sum_sq += e * e;
        d.peak = std::max(d.peak, std::fabs(got[i]));
    }
    d.rmse = std::sqrt(sum_sq / static_cast<double>(std::max<size_t>(n, 1)));
    return d;
}

// How many whole frames agree, and the first that does not (or -1).
int first_differing_frame(const std::vector<double>& got, const std::vector<float>& want) {
    const size_t frames = std::min(got.size(), want.size()) / CODEBOOKS;
    for (size_t f = 0; f < frames; ++f) {
        for (int c = 0; c < CODEBOOKS; ++c) {
            if (got[f * CODEBOOKS + c] != static_cast<double>(want[f * CODEBOOKS + c])) return static_cast<int>(f);
        }
    }
    return got.size() == want.size() ? -1 : static_cast<int>(frames);
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_VOXTRAL_TTS_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_VOXTRAL_TTS_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_VOXTRAL_TTS_GGUF and LOOM_VOXTRAL_TTS_REF_DIR "
                              "(see scripts/voxtral_tts_reference.py)\n");
        return 77;
    }
    const std::string ref_dir = ref_env;
    std::ifstream meta_file(ref_dir + "/meta.json");
    LOOM_CHECK(meta_file.good());
    std::stringstream meta_buf;
    meta_buf << meta_file.rdbuf();
    const std::string text = meta_text(meta_buf.str());

    std::vector<int64_t> shape;
    const auto prompt = loom_test::read_npy_f32(ref_dir + "/prompt_ids.npy", shape);
    const auto noise = loom_test::read_npy_f32(ref_dir + "/noise.npy", shape);
    const auto codes = loom_test::read_npy_f32(ref_dir + "/codes.npy", shape);
    const auto ref_wave = loom_test::read_npy_f32(ref_dir + "/wave.npy", shape);
    LOOM_CHECK(!prompt.empty() && !noise.empty() && !codes.empty() && !ref_wave.empty());

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);

    // **The text door first.** The prompt is [BOS] [BEGIN_AUDIO] [AUDIO] x n [NEXT_AUDIO_TEXT] text
    // [REPEAT_AUDIO_TEXT] [BEGIN_AUDIO]; the file's own vocabulary must produce the text's ids, or the
    // comparisons below would be grading two different sentences.
    const size_t n_voice = static_cast<size_t>(std::count(prompt.begin(), prompt.end(), AUDIO));
    const std::vector<float> ref_ids(prompt.begin() + 3 + static_cast<long>(n_voice), prompt.end() - 2);
    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode(text);
    LOOM_CHECK(ids.size() == ref_ids.size());
    for (size_t i = 0; i < std::min(ids.size(), ref_ids.size()); ++i) {
        LOOM_CHECK(static_cast<float>(ids[i]) == ref_ids[i]);
    }
    const std::vector<double> tokens(ids.begin(), ids.end());

    const auto run = [&](std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs) {
        inputs["tokens"] = tokens;
        inputs["noise"] = as_doubles(noise);
        loom::Session session(*model, device.backends());
        return std::get<std::vector<double>>(session.bridge().call("infer", inputs));
    };

    // --- 0. Free-running codes: the loop itself. ---
    {
        const auto got = run({{"return_codes", 1.0}});
        const int bad = first_differing_frame(got, codes);
        std::fprintf(stderr, "free-running codes: loom=%zu reference=%zu frames, first differing %d\n",
                     got.size() / CODEBOOKS, codes.size() / CODEBOOKS, bad);
        LOOM_CHECK(bad == -1);
    }

    // --- 1. Teacher-forced codes: every frame on its own. ---
    {
        const auto got = run({{"return_codes", 1.0}, {"teacher_codes", as_doubles(codes)}});
        const int bad = first_differing_frame(got, codes);
        std::fprintf(stderr, "teacher-forced codes: loom=%zu reference=%zu frames, first differing %d\n",
                     got.size() / CODEBOOKS, codes.size() / CODEBOOKS, bad);
        LOOM_CHECK(bad == -1);
    }

    // --- 2. The waveform, in one codec call and in chunks. ---
    for (const double chunk : {0.0, 40.0}) {
        std::unordered_map<std::string, loom::LoomLuaBridge::Value> inputs;
        if (chunk > 0) inputs["codec_chunk"] = chunk;
        const auto got = run(inputs);
        std::fprintf(stderr, "waveform (codec_chunk %s): loom=%zu reference=%zu samples\n",
                     chunk > 0 ? "40" : "default", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "  max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        // A waveform collapsed toward silence can have a small difference for the wrong reason
        // (Retro-006).
        LOOM_CHECK(d.peak > 0.05);
        LOOM_CHECK(d.max_abs < VOXTRAL_WAVE_MAX_ABS);
        LOOM_CHECK(d.rmse < VOXTRAL_WAVE_RMSE);
    }

    // --- 3. Sabotage: a different guidance must move the codes. ---
    {
        const auto got = run({{"return_codes", 1.0}, {"teacher_codes", as_doubles(codes)}, {"cfg", 1.3}});
        const int bad = first_differing_frame(got, codes);
        std::fprintf(stderr, "sabotage (cfg 1.3): first differing frame %d\n", bad);
        LOOM_CHECK(bad != -1);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
