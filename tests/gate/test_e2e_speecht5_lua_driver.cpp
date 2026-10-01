// Validates the SpeechT5 export (loom-exporter's `speecht5_export.py`) end to end against the reference
// implementation: text in through the file's own character vocabulary, the encoder with its relative
// position bias, the autoregressive mel loop (prenet dropout, speaker x-vector, KV-cached decoder, stop
// head), the postnet and HiFi-GAN, and a waveform out -- one GGUF, five topologies, one driver.
//
// **The oracle is the real SpeechT5** (`scripts/speecht5_reference.py`, transformers'
// `generate_speech`), with its prenet dropout masks pinned and handed in (`masks`): the reference
// applies that dropout at inference too, so the masks are the run's only randomness. Two arms:
//
//   * **Teacher-forced**, the exact one. Every frame is the next step's input, so f32 rounding
//     COMPOUNDS along the loop; fed the reference's frames (`teacher_frames`), each step is compared
//     on its own. Measured: max |d| 1.3e-04, rmse 5.9e-06 over 68,608 samples -- where a CORRECT torch
//     implementation of the same five graphs, at f32 or at f64, lands 1.7e-04 / 6.8e-06 from the
//     reference's own waveform. Inverting the masks (the sabotage arm) gives rmse 4.9e-02.
//   * **Free-running**, the one that checks the LOOP: the same step count (the stop head fired at the
//     same step: its summed probability is 0.99999976 there and never above 0.33 before) and a
//     waveform within a drift bound. rmse was 5.0e-04; the reference's own f32-vs-f64 spread on this
//     clip is 5.4e-02 max.
//
// Fixtures:
//   LOOM_SPEECHT5_GGUF     speecht5.gguf  -- `loom-export ~/Dev/models/speecht5-tts -o speecht5.gguf`
//                                            (the directory holds `hifigan/` and `xvectors/`)
//   LOOM_SPEECHT5_REF_DIR  speecht5_ref/  -- tokens, speaker, masks, spectra, wave (.npy) and meta.json
//                                            from scripts/speecht5_reference.py

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string meta_text(const std::string& json) {
    const std::string needle = "\"text\": \"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) throw std::runtime_error("speecht5_ref/meta.json has no \"text\"");
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

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_SPEECHT5_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_SPEECHT5_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_SPEECHT5_GGUF and LOOM_SPEECHT5_REF_DIR "
                              "(see scripts/speecht5_reference.py)\n");
        return 77;
    }
    const std::string ref_dir = ref_env;
    std::ifstream meta_file(ref_dir + "/meta.json");
    LOOM_CHECK(meta_file.good());
    std::stringstream meta_buf;
    meta_buf << meta_file.rdbuf();
    const std::string text = meta_text(meta_buf.str());

    std::vector<int64_t> shape;
    const auto ref_ids = loom_test::read_npy_f32(ref_dir + "/tokens.npy", shape);
    const auto speaker = loom_test::read_npy_f32(ref_dir + "/speaker.npy", shape);
    const auto masks = loom_test::read_npy_f32(ref_dir + "/masks.npy", shape);
    const auto spectra = loom_test::read_npy_f32(ref_dir + "/spectra.npy", shape);
    const auto ref_wave = loom_test::read_npy_f32(ref_dir + "/wave.npy", shape);
    LOOM_CHECK(!ref_ids.empty() && speaker.size() == 512 && !masks.empty() && !spectra.empty());

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);

    // **The text door first.** The CHAR SentencePiece model ships as Unigram (ADR-057), and its ids,
    // `</s>` appended, must be `SpeechT5Tokenizer`'s, or the waveform below grades another sentence.
    auto vocab = loom::Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode(text);
    LOOM_CHECK(ids.size() == ref_ids.size());
    for (size_t i = 0; i < std::min(ids.size(), ref_ids.size()); ++i) {
        LOOM_CHECK(static_cast<float>(ids[i]) == ref_ids[i]);
    }
    const std::vector<double> tokens(ids.begin(), ids.end());

    // --- 1. Teacher-forced: every step on its own. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"speaker", as_doubles(speaker)}, {"masks", as_doubles(masks)},
            {"teacher_frames", as_doubles(spectra)}}));
        // 256 samples per frame, 2 frames per step: a length mismatch is a stop decision that went the
        // other way, which makes every per-sample number meaningless -- checked first.
        std::fprintf(stderr, "teacher-forced samples: loom=%zu reference=%zu\n", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "teacher-forced max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        // A waveform collapsed toward silence can differ little for the wrong reason (Retro-006).
        LOOM_CHECK(d.peak > 0.05);
        // 15x the measured max and 8x the measured rmse, for other ISAs' accumulation order; the
        // sabotage arm's rmse is 1000x over the rmse bound.
        LOOM_CHECK(d.max_abs < 2e-3);
        LOOM_CHECK(d.rmse < 5e-5);
    }

    // --- 2. Free-running: the loop itself, from the same masks. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"speaker", as_doubles(speaker)}, {"masks", as_doubles(masks)}}));
        std::fprintf(stderr, "free-running samples: loom=%zu reference=%zu\n", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "free-running max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        // A drift bound, not an equality: 10x the measured rmse.
        LOOM_CHECK(d.rmse < 5e-3);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
