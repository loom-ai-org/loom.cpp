// Validates LFM2.5-Audio's text-to-speech export (loom-exporter's `lfm25_audio_export.py`,
// `Lfm25AudioTtsExportConfig`) end to end against liquid-audio's own `generate_sequential`
// (`scripts/lfm25_audio_tts_reference.py`): the voice and text prompt, the LM's text and audio modes,
// the depthformer drawing each frame's 8 codes, and the LFM2-based detokenizer with its ISTFT.
//
// The reference is run GREEDILY: the README samples codes at temperature 0.8, and two random streams
// cannot be compared draw for draw. Three arms:
//   1. CODES: every frame's 8 codes (`return_codes`), exactly. Measured 65/65 frames.
//   2. WAVE: the 24 kHz waveform. Measured max|d| 4.1e-06 over 124,800 samples.
//   3. SABOTAGE: another voice's prompt must change the codes.
//
// Fixtures:
//   LOOM_LFM25_AUDIO_TTS_GGUF     lfm25_audio_tts.gguf  -- `loom-export ~/Dev/models/lfm2.5-audio-1.5b
//                                                          --task text-to-speech --model lfm2.5-audio-tts`
//   LOOM_LFM25_AUDIO_TTS_REF_DIR  lfm25_audio_tts_ref/  -- from scripts/lfm25_audio_tts_reference.py

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_LFM25_AUDIO_TTS_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_LFM25_AUDIO_TTS_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_LFM25_AUDIO_TTS_GGUF and LOOM_LFM25_AUDIO_TTS_REF_DIR "
                              "(see scripts/lfm25_audio_tts_reference.py)\n");
        return 77;
    }
    const std::string ref_dir = ref_env;
    std::vector<int64_t> shape;
    const auto text_ids = loom_test::read_npy_f32(ref_dir + "/text_ids.npy", shape);
    const auto voice = loom_test::read_npy_f32(ref_dir + "/voice_prompt.npy", shape);
    const auto codes = loom_test::read_npy_f32(ref_dir + "/codes.npy", shape);    // [8, T]
    const size_t n_frames = static_cast<size_t>(shape[1]);
    const auto ref_wave = loom_test::read_npy_f32(ref_dir + "/wave.npy", shape);
    LOOM_CHECK(!text_ids.empty() && !voice.empty() && n_frames > 0);

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, device.backends());
    const std::vector<double> tokens(text_ids.begin(), text_ids.end());
    const std::vector<double> voice_prompt(voice.begin(), voice.end());

    const auto frames_identical = [&](const std::vector<double>& got) {
        size_t same = 0;
        for (size_t f = 0; f < n_frames && (f + 1) * 8 <= got.size(); ++f) {
            bool all = true;
            for (size_t j = 0; j < 8; ++j) all = all && got[f * 8 + j] == codes[j * n_frames + f];
            same += all;
        }
        return same;
    };

    // --- 1. Codes. ---
    {
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"voice_prompt", voice_prompt}, {"temperature", 0.0}, {"return_codes", 1.0}}));
        const size_t same = frames_identical(got);
        std::fprintf(stderr, "codes: %zu frames (reference %zu), %zu identical\n", got.size() / 8, n_frames, same);
        LOOM_CHECK(got.size() == n_frames * 8);
        LOOM_CHECK(same == n_frames);
    }

    // --- 2. Wave. ---
    {
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"voice_prompt", voice_prompt}, {"temperature", 0.0}}));
        LOOM_CHECK(got.size() == ref_wave.size());
        double max_abs = 0.0, peak = 0.0;
        for (size_t i = 0; i < std::min(got.size(), ref_wave.size()); ++i) {
            max_abs = std::max(max_abs, std::fabs(got[i] - ref_wave[i]));
            peak = std::max(peak, std::fabs(got[i]));
        }
        std::fprintf(stderr, "wave: %zu samples, max|d| %g, peak %g\n", got.size(), max_abs, peak);
        LOOM_CHECK(peak > 0.05);
        // ~25x the measured 4.1e-06, for other ISAs' accumulation order.
        LOOM_CHECK(max_abs < 1e-4);
    }

    // --- 3. Sabotage: the file's own (another) voice. ---
    {
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"temperature", 0.0}, {"return_codes", 1.0}}));
        const size_t same = frames_identical(got);
        std::fprintf(stderr, "sabotage (default voice): %zu of %zu frames identical\n", same, n_frames);
        LOOM_CHECK(same < n_frames / 2);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
