// The MOSS pair chained: text -> MOSS-TTS-Local-Transformer-v1.5 -> 12 codebooks per frame -> padded
// to the codec's 32 with its absent id -> MOSS-Audio-Tokenizer-v2 -> 48 kHz interleaved stereo, against
// the reference implementation running the identical pipeline (`scripts/moss_tts_reference.py`).
//
// **What this pair adds over Dia+DAC is the width gap.** The LM emits 12 codebooks and the codec has
// 32; the reference decodes the 12-codebook PREFIX (`num_quantizers=12`). The codec declares
// `codec.absent_code`, whose row contributes nothing (ADR-050), and a host fills each row with it --
// which is what this test does, the way loom-py's `codes2speech` does, so the join it checks is the
// one a caller makes.
//
// **Codes before the waveform, for the diagnostic reason `test_e2e_dia_dac_composition.cpp` gives**,
// and the codes three ways: greedy at two lengths, and once SAMPLED at the README's settings with every
// uniform pinned (ADR-047) -- the only way a sampled decode can be compared exactly, and what checks the
// sampler's options, the merged head's windows and the draw order rather than only the graphs.
//
// **Two clip lengths** so the waveform's length has to follow its input: a constant length is
// consistent with itself (family 11's first bug), and the sample counts must differ by exactly
// `hop * channels * (24 - 12)`.
//
// Fixtures (all three, or this skips):
//   LOOM_MOSS_TTS_GGUF              moss_tts.gguf              -- `loom-export <moss-tts-local-transformer-v1.5>`
//   LOOM_MOSS_AUDIO_TOKENIZER_GGUF  moss_audio_tokenizer.gguf  -- `loom-export <moss-audio-tokenizer-v2>`
//   LOOM_MOSS_TTS_REF_DIR           moss_tts_ref/              -- `scripts/moss_tts_reference.py`
//
// Memory: the LM is 16.8 GB at F32 and the codec 4.3 GB, loaded one after the other.

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;
// `scripts/moss_tts_reference.py`'s defaults, encoded in the fixture names.
constexpr int kFrames[] = {12, 24};
constexpr int kPinnedFrames = 24;
constexpr int kLmCodebooks = 12;

std::vector<double> read_npy(const std::string& path, std::vector<int64_t>& shape) {
    const std::vector<float> values = loom_test::read_npy_f32(path, shape);
    return std::vector<double>(values.begin(), values.end());
}

size_t count_mismatches(const std::vector<double>& got, const std::vector<double>& want,
                        const char* what) {
    size_t mismatches = 0;
    for (size_t j = 0; j < std::min(got.size(), want.size()); ++j) {
        if (static_cast<int32_t>(got[j]) != static_cast<int32_t>(want[j])) {
            if (mismatches < 8) {
                std::fprintf(stderr, "  %s frame %zu codebook %zu: expected %d, got %d\n", what,
                             j / kLmCodebooks, j % kLmCodebooks, static_cast<int32_t>(want[j]),
                             static_cast<int32_t>(got[j]));
            }
            ++mismatches;
        }
    }
    return mismatches;
}

} // namespace

int main() {
    const char* lm_env = loom_test::fixture_env("LOOM_MOSS_TTS_GGUF");
    const char* codec_env = loom_test::fixture_env("LOOM_MOSS_AUDIO_TOKENIZER_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_MOSS_TTS_REF_DIR");
    if (lm_env == nullptr || codec_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_MOSS_TTS_GGUF, LOOM_MOSS_AUDIO_TOKENIZER_GGUF and "
                              "LOOM_MOSS_TTS_REF_DIR (see scripts/moss_tts_reference.py)\n");
        return kSkipReturnCode;
    }
    const std::string ref_dir = ref_env;
    if (!loom_test::path_exists(ref_dir + "/text_ids.npy")) {
        std::fprintf(stderr, "skipping: %s/text_ids.npy not found\n", ref_dir.c_str());
        return kSkipReturnCode;
    }

    std::vector<int64_t> shape;
    const std::vector<double> text_ids = read_npy(ref_dir + "/text_ids.npy", shape);
    const std::vector<double> language = read_npy(ref_dir + "/language.npy", shape);
    const std::vector<double> draws = read_npy(ref_dir + "/draws.npy", shape);
    LOOM_CHECK(language.size() == 1);

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    // The LM, in its own scope: 16.8 GB of F32 weights that the codec half has no use for.
    std::vector<std::vector<double>> greedy;
    {
        auto lm = loom::GgufModel::load(lm_env, backend.get());
        LOOM_CHECK(static_cast<int>(lm->hparam_u32("codec.n_codebooks")) == kLmCodebooks);
        loom::Session session(*lm, backend.get());
        loom::LoomLuaBridge& bridge = session.bridge();
        for (const int frames : kFrames) {
            // Greedy for BOTH draws -- continue/stop and the codebooks -- which is the reference's
            // `do_sample=False`.
            auto result = bridge.call("infer", {{"tokens", text_ids}, {"language", language[0]},
                                                {"max_new_tokens", static_cast<double>(frames)},
                                                {"temperature", 0.0}, {"text_temperature", 0.0}});
            greedy.push_back(std::get<std::vector<double>>(result));
        }
        // Sampled at the file's own defaults (the README's), every draw pinned.
        auto result = bridge.call("infer", {{"tokens", text_ids}, {"language", language[0]},
                                            {"max_new_tokens", static_cast<double>(kPinnedFrames)},
                                            {"draws", draws}});
        const auto& pinned = std::get<std::vector<double>>(result);
        const std::vector<double> ref_pinned = read_npy(ref_dir + "/codes_pinned.npy", shape);
        std::fprintf(stderr, "pinned: %zu codes, reference %zu\n", pinned.size(), ref_pinned.size());
        LOOM_CHECK(pinned.size() == ref_pinned.size());
        LOOM_CHECK(count_mismatches(pinned, ref_pinned, "pinned") == 0);
    }

    for (size_t i = 0; i < std::size(kFrames); ++i) {
        const int frames = kFrames[i];
        const std::vector<double> ref_codes =
            read_npy(ref_dir + "/codes_" + std::to_string(frames) + "f.npy", shape);
        LOOM_CHECK(shape.size() == 2 && shape[0] == frames && shape[1] == kLmCodebooks);
        std::fprintf(stderr, "greedy %d frames: %zu codes, reference %zu\n", frames,
                     greedy[i].size(), ref_codes.size());
        LOOM_CHECK(greedy[i].size() == ref_codes.size());
        LOOM_CHECK(count_mismatches(greedy[i], ref_codes, "greedy") == 0);
    }

    auto codec = loom::GgufModel::load(codec_env, backend.get());
    // The join's contract, read off the two files as a host would: the codec is WIDER and says which
    // id means "absent", and its audio is interleaved stereo.
    const int width = static_cast<int>(codec->hparam_u32("codec.n_codebooks"));
    LOOM_CHECK(width >= kLmCodebooks);
    LOOM_CHECK(codec->has_kv("loom.codec.absent_code"));
    const double absent = static_cast<double>(codec->hparam_u32("codec.absent_code"));
    const int channels = static_cast<int>(codec->hparam_u32("channels"));
    LOOM_CHECK(channels == 2);
    const int sample_rate = static_cast<int>(codec->hparam_u32("sample_rate"));
    const int hop = static_cast<int>(std::lround(sample_rate / codec->hparam_f32("codec.frame_rate")));

    loom::Session codec_session(*codec, backend.get());
    loom::LoomLuaBridge& bridge = codec_session.bridge();
    std::vector<size_t> sample_counts;
    for (size_t i = 0; i < std::size(kFrames); ++i) {
        const int frames = kFrames[i];
        std::vector<double> rows;
        for (int f = 0; f < frames; ++f) {
            for (int g = 0; g < width; ++g) {
                rows.push_back(g < kLmCodebooks ? greedy[i][f * kLmCodebooks + g] : absent);
            }
        }
        const std::vector<double> ref_wav =
            read_npy(ref_dir + "/wav_" + std::to_string(frames) + "f.npy", shape);
        auto result = bridge.call("infer", {{"codes", rows}});
        const auto& wav = std::get<std::vector<double>>(result);
        sample_counts.push_back(wav.size());
        std::fprintf(stderr, "codec %d frames: %zu floats, reference %zu (hop %d x %d channels)\n",
                     frames, wav.size(), ref_wav.size(), hop, channels);
        LOOM_CHECK(wav.size() == ref_wav.size());
        LOOM_CHECK(wav.size() == static_cast<size_t>(frames) * hop * channels);

        double max_abs_diff = 0.0, peak = 0.0, sum_sq = 0.0;
        for (size_t j = 0; j < wav.size(); ++j) {
            max_abs_diff = std::max(max_abs_diff, std::abs(wav[j] - ref_wav[j]));
            peak = std::max(peak, std::abs(wav[j]));
            sum_sq += wav[j] * wav[j];
        }
        const double rms = std::sqrt(sum_sq / static_cast<double>(wav.size()));
        std::fprintf(stderr, "  max |diff| %.3e, peak %.4f, rms %.4f\n", max_abs_diff, peak, rms);
        // Relative to the signal. The codec matched its reference at 2.6e-06 on a 0.79 peak (ADR-049),
        // so 1e-3 of peak leaves two orders of magnitude of margin while a dropped frame or a wrong
        // absent row is on the order of the peak itself.
        LOOM_CHECK(max_abs_diff < std::max(1e-3 * peak, 1e-6));
        LOOM_CHECK(rms > 1e-5);
        LOOM_CHECK(peak <= 1.0);
    }
    LOOM_CHECK(sample_counts.size() == 2);
    LOOM_CHECK(sample_counts[1] - sample_counts[0] ==
               static_cast<size_t>(kFrames[1] - kFrames[0]) * hop * channels);

    LOOM_TEST_REPORT_AND_RETURN();
}
