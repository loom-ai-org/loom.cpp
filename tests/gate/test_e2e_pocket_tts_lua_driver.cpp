// Validates the Pocket-TTS export (loom-exporter's `pocket_tts_export.py`) end to end against the
// reference implementation: text in through the file's own vocabulary, the built-in voice seeded into
// the flow LM's KV cache, the latent loop with its one-step flow head and EOS head, Mimi, and a
// waveform out -- one GGUF, five topologies, one driver.
//
// **The oracle is the real Pocket-TTS** (`scripts/pocket_tts_reference.py`), with its one random draw
// per step pinned and handed in (`noise`): the same seed is not the same noise. Two arms:
//
//   * **Teacher-forced**, the exact one. The loop feeds every latent back as the next step's input,
//     so f32 rounding COMPOUNDS along it, and two correct implementations drift apart -- the
//     reference's own f32-vs-f64 spread is 5.3e-04 on this clip, and loom free-running is 3.8e-03 by
//     the end. Fed the reference's latents (`teacher_latents`), every step is compared on its own:
//     max |d| 1.5e-04, rmse 1.8e-06 over 155,520 samples. The sabotage arm (temperature 0.7 for the
//     checkpoint's 0.3) gives 0.82.
//   * **Free-running**, the one that checks the LOOP: the same frame count (the EOS head fired at the
//     same step; its margin over the threshold was 1.4 there) and a waveform still within a drift
//     bound. rmse was 9.9e-05.
//
// Fixtures:
//   LOOM_POCKET_TTS_GGUF     pocket_tts.gguf  -- `loom-export ~/Dev/models/pocket-tts/languages/
//                                                english_2026-09 -o pocket_tts.gguf`
//   LOOM_POCKET_TTS_REF_DIR  pocket_tts_ref/  -- tokens.npy, noise.npy, latents.npy, wave.npy and
//                                                meta.json from scripts/pocket_tts_reference.py

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

// The one string this test needs out of `meta.json`: the text as the reference was given it.
std::string meta_text(const std::string& json) {
    const std::string needle = "\"text\": \"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) throw std::runtime_error("pocket_tts_ref/meta.json has no \"text\"");
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
    const char* gguf_env = loom_test::fixture_env("LOOM_POCKET_TTS_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_POCKET_TTS_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_POCKET_TTS_GGUF and LOOM_POCKET_TTS_REF_DIR "
                              "(see scripts/pocket_tts_reference.py)\n");
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
    const auto noise = loom_test::read_npy_f32(ref_dir + "/noise.npy", shape);
    const auto latents = loom_test::read_npy_f32(ref_dir + "/latents.npy", shape);
    const auto ref_wave = loom_test::read_npy_f32(ref_dir + "/wave.npy", shape);
    LOOM_CHECK(!ref_ids.empty() && !noise.empty() && !latents.empty() && !ref_wave.empty());

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);

    // **The text door first.** The reference's ids are its whole text path's (prepare, chunk,
    // prepare, tokenize) for a one-chunk text; the file's own vocabulary must produce them, or the
    // waveform comparison below would be grading two different sentences.
    auto vocab = loom::PocketTtsVocab::load(*model);
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
            {"tokens", tokens}, {"noise", as_doubles(noise)}, {"teacher_latents", as_doubles(latents)}}));
        // The LENGTH is the frame count times 1920, so a disagreement is an EOS decision that went the
        // other way -- checked first, because it makes every per-sample number meaningless.
        std::fprintf(stderr, "teacher-forced samples: loom=%zu reference=%zu\n", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "teacher-forced max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        // A waveform collapsed toward silence can have a small difference for the wrong reason
        // (Retro-006). This clip peaks at 0.986.
        LOOM_CHECK(d.peak > 0.05);
        // 2e-3: 13x what was measured and 400x below the sabotage arm's 0.82. The margin is for other
        // ISAs' accumulation order; the rmse bound is the sharper of the two.
        LOOM_CHECK(d.max_abs < 2e-3);
        LOOM_CHECK(d.rmse < 2e-5);
    }

    // --- 2. Free-running: the loop itself, from the same draws. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"noise", as_doubles(noise)}}));
        std::fprintf(stderr, "free-running samples: loom=%zu reference=%zu\n", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "free-running max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        // A drift bound, not an equality: 10x the measured rmse, and the sabotage arm is 650x over it.
        LOOM_CHECK(d.rmse < 1e-3);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
