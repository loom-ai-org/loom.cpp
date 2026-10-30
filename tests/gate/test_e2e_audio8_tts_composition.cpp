// The Audio8 pair chained: text -> Audio8-TTS-Realistic-0.6b -> 10 codebooks per frame -> its codec ->
// 44.1 kHz mono, against the reference implementation running the identical pipeline
// (`scripts/audio8_tts_reference.py`).
//
// **Codes before the waveform, for `test_e2e_dia_dac_composition.cpp`'s diagnostic reason.** Greedy
// run to the model's own EOS, so the frame it stops on is checked too; the sampled default cannot be
// replayed (the reference draws by an exponential race, not an inverse-CDF walk -- ADR-095), and the
// sampler's options are pinned by `tests/ci/test_sample_row.cpp` instead.
//
// **Voice cloning through a voice FILE** (ADR-045) written by the exporter's tool, not by the script
// that made the oracle: its inputs must equal the reference's encode of the clip and the processor's
// formatting of its transcript, and with them the LM must reproduce the reference's clone generation.
// A code outside the codebook must be refused: the absent code's zero row would read it as silence.
//
// **Two codec lengths** so the waveform's length has to follow its input, by exactly 2048 samples a
// frame.
//
// Fixtures (all three, or this skips):
//   LOOM_AUDIO8_TTS_GGUF        audio8_tts.gguf        -- `loom-export <audio8-tts-realistic-0.6b>`
//   LOOM_AUDIO8_TTS_CODEC_GGUF  audio8_tts_codec.gguf  -- `loom-export <audio8-tts-realistic-0.6b>/codec.pth`
//   LOOM_AUDIO8_TTS_REF_DIR     audio8_tts_ref/        -- `scripts/audio8_tts_reference.py`, plus
//                               `voice_jfk.gguf` from `python -m loom_exporter.audio8_tts_voices`
//                               (the script's docstring)

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
constexpr int kCodebooks = 10;
constexpr int kCloneFrames = 24;
constexpr int kShortFrames = 24;

std::vector<double> read_npy(const std::string& path, std::vector<int64_t>& shape) {
    const std::vector<float> values = loom_test::read_npy_f32(path, shape);
    return std::vector<double>(values.begin(), values.end());
}

size_t count_mismatches(const std::vector<double>& got, const std::vector<double>& want, const char* what) {
    size_t mismatches = 0;
    for (size_t j = 0; j < std::min(got.size(), want.size()); ++j) {
        if (static_cast<int32_t>(got[j]) != static_cast<int32_t>(want[j])) {
            if (mismatches < 8) {
                std::fprintf(stderr, "  %s frame %zu codebook %zu: expected %d, got %d\n", what,
                             j / kCodebooks, j % kCodebooks, static_cast<int32_t>(want[j]),
                             static_cast<int32_t>(got[j]));
            }
            ++mismatches;
        }
    }
    return mismatches;
}

} // namespace

int main() {
    const char* lm_env = loom_test::fixture_env("LOOM_AUDIO8_TTS_GGUF");
    const char* codec_env = loom_test::fixture_env("LOOM_AUDIO8_TTS_CODEC_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_AUDIO8_TTS_REF_DIR");
    if (lm_env == nullptr || codec_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_AUDIO8_TTS_GGUF, LOOM_AUDIO8_TTS_CODEC_GGUF and "
                              "LOOM_AUDIO8_TTS_REF_DIR (see scripts/audio8_tts_reference.py)\n");
        return kSkipReturnCode;
    }
    const std::string ref_dir = ref_env;
    if (!loom_test::path_exists(ref_dir + "/text_ids.npy")) {
        std::fprintf(stderr, "skipping: %s/text_ids.npy not found\n", ref_dir.c_str());
        return kSkipReturnCode;
    }

    std::vector<int64_t> shape;
    const std::vector<double> text_ids = read_npy(ref_dir + "/text_ids.npy", shape);
    const std::vector<double> ref_greedy = read_npy(ref_dir + "/codes_greedy.npy", shape);
    LOOM_CHECK(shape.size() == 2 && shape[1] == kCodebooks);
    const int greedy_frames = static_cast<int>(shape[0]);

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    std::vector<double> greedy;
    {
        auto lm = loom::GgufModel::load(lm_env, backend.get());
        LOOM_CHECK(static_cast<int>(lm->hparam_u32("codec.n_codebooks")) == kCodebooks);
        loom::Session session(*lm, backend.get());
        loom::LoomLuaBridge& bridge = session.bridge();

        // Greedy, to the model's own EOS: the same codes AND the same stop.
        greedy = std::get<std::vector<double>>(
            bridge.call("infer", {{"tokens", text_ids}, {"temperature", 0.0}}));
        std::fprintf(stderr, "greedy: %zu codes, reference %zu (%d frames)\n", greedy.size(),
                     ref_greedy.size(), greedy_frames);
        LOOM_CHECK(greedy.size() == ref_greedy.size());
        LOOM_CHECK(count_mismatches(greedy, ref_greedy, "greedy") == 0);

        // Voice cloning, through the file the exporter's tool wrote.
        const loom::VoiceFile voice = loom::load_voice(*lm, ref_dir + "/voice_jfk.gguf");
        const auto codes = voice.inputs.find("reference_codes");
        const auto tokens = voice.inputs.find("reference_tokens");
        LOOM_CHECK(codes != voice.inputs.end() && tokens != voice.inputs.end());
        LOOM_CHECK(codes->second == read_npy(ref_dir + "/clone_ref_codes.npy", shape));
        LOOM_CHECK(tokens->second == read_npy(ref_dir + "/clone_ref_tokens.npy", shape));
        std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {
            {"tokens", text_ids}, {"temperature", 0.0}, {"max_new_tokens", static_cast<double>(kCloneFrames)}};
        for (const auto& [name, values] : voice.inputs) args[name] = values;
        const auto clone = std::get<std::vector<double>>(bridge.call("infer", args));
        const std::vector<double> ref_clone =
            read_npy(ref_dir + "/clone_codes_" + std::to_string(kCloneFrames) + "f.npy", shape);
        std::fprintf(stderr, "clone: %zu codes, reference %zu\n", clone.size(), ref_clone.size());
        LOOM_CHECK(clone.size() == ref_clone.size());
        LOOM_CHECK(count_mismatches(clone, ref_clone, "clone") == 0);

        // A code the codebook does not have, and codes without their transcript, are refused.
        std::vector<double> bad = codes->second;
        bad[3] = 4096.0;
        args["reference_codes"] = bad;
        args["max_new_tokens"] = 1.0;
        LOOM_CHECK_THROWS(bridge.call("infer", args), loom::Error);
        LOOM_CHECK_THROWS(bridge.call("infer", {{"tokens", text_ids}, {"max_new_tokens", 1.0},
                                                {"reference_codes", codes->second}}),
                          loom::Error);
    }

    auto codec = loom::GgufModel::load(codec_env, backend.get());
    LOOM_CHECK(static_cast<int>(codec->hparam_u32("codec.n_codebooks")) == kCodebooks);
    const int sample_rate = static_cast<int>(codec->hparam_u32("sample_rate"));
    const int hop = static_cast<int>(std::lround(sample_rate / codec->hparam_f32("codec.frame_rate")));
    LOOM_CHECK(sample_rate == 44100 && hop == 2048);
    loom::Session codec_session(*codec, backend.get());
    loom::LoomLuaBridge& bridge = codec_session.bridge();
    std::vector<size_t> sample_counts;
    for (const int frames : {kShortFrames, greedy_frames}) {
        const std::vector<double> rows(greedy.begin(), greedy.begin() + frames * kCodebooks);
        const std::vector<double> ref_wav = read_npy(ref_dir + "/wav_" + std::to_string(frames) + "f.npy", shape);
        const auto wav = std::get<std::vector<double>>(bridge.call("infer", {{"codes", rows}}));
        sample_counts.push_back(wav.size());
        std::fprintf(stderr, "codec %d frames: %zu samples, reference %zu\n", frames, wav.size(), ref_wav.size());
        LOOM_CHECK(wav.size() == ref_wav.size());
        LOOM_CHECK(wav.size() == static_cast<size_t>(frames) * hop);
        double max_abs_diff = 0.0, peak = 0.0, sum_sq = 0.0;
        for (size_t j = 0; j < std::min(wav.size(), ref_wav.size()); ++j) {
            max_abs_diff = std::max(max_abs_diff, std::abs(wav[j] - ref_wav[j]));
            peak = std::max(peak, std::abs(wav[j]));
            sum_sq += wav[j] * wav[j];
        }
        const double rms = std::sqrt(sum_sq / static_cast<double>(std::max<size_t>(wav.size(), 1)));
        std::fprintf(stderr, "  max |diff| %.3e, peak %.4f, rms %.4f\n", max_abs_diff, peak, rms);
        // Relative to the signal: measured 4.6e-06 on a ~0.9 peak, so 1e-3 of peak leaves two orders of
        // magnitude while a dropped frame or a wrong codebook is on the order of the peak itself.
        LOOM_CHECK(max_abs_diff < std::max(1e-3 * peak, 1e-6));
        LOOM_CHECK(rms > 1e-5);
        LOOM_CHECK(peak <= 1.0);
    }
    LOOM_CHECK(sample_counts.size() == 2 &&
               sample_counts[1] - sample_counts[0] == static_cast<size_t>(greedy_frames - kShortFrames) * hop);

    LOOM_TEST_REPORT_AND_RETURN();
}
