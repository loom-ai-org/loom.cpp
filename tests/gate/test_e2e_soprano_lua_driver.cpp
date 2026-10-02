// Validates the Soprano TTS export (loom-exporter's `soprano_export.py`) end to end against the
// reference implementation: text in through the file's own front end (tortoise-tts's normaliser, the
// sentence split, the prompt), the Qwen3 LM's KV-cached loop with its repetition penalty, the final-norm
// hidden rows gathered per step, the Vocos decoder per sentence, and a waveform out.
//
// **The oracle is the reference's own `SopranoTTS.infer`** (`scripts/soprano_reference.py`), which
// generates its sentences as ONE left-padded batch where the driver runs them one at a time.
//
// **It samples, at temperature 0.001, and that is why this compares GREEDILY.** At that temperature a
// draw is the argmax unless two logits are within a few 1e-3: on this text the closest pair is 2.4e-3
// apart (sentence 2, step 31), where the second id has a 9.4% chance. Torch's draw and the engine's
// are different streams, so the default free run may resolve such a tie the other way and grade a
// different, equally valid utterance. The reference's sampled ids equal its greedy ones here, so
// `temperature = 0` is the comparison that can be exact. Measured: the ids identical and the waveform
// within 4.7e-05 (rmse 1.2e-06) over 436,224 samples; teacher-forced, every step's logits within 1.6e-05
// and its hidden row within 6.4e-06 of torch. Dropping the repetition penalty (the sabotage arm) changes
// the length.
//
// Fixtures:
//   LOOM_SOPRANO_GGUF     soprano.gguf  -- `loom-export ~/Dev/models/soprano-1.1-80m -o soprano.gguf`
//   LOOM_SOPRANO_REF_DIR  soprano_ref/  -- tokens, ids_<i>, hidden_<i>, wave (.npy) and meta.json from
//                                          scripts/soprano_reference.py

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
    if (at == std::string::npos) throw std::runtime_error("soprano_ref/meta.json has no \"text\"");
    const size_t start = at + needle.size();
    return json.substr(start, json.find('"', start) - start);
}

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
    const char* gguf_env = loom_test::fixture_env("LOOM_SOPRANO_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_SOPRANO_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_SOPRANO_GGUF and LOOM_SOPRANO_REF_DIR "
                              "(see scripts/soprano_reference.py)\n");
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
    const auto ref_wave = loom_test::read_npy_f32(ref_dir + "/wave.npy", shape);
    LOOM_CHECK(!ref_ids.empty() && !ref_wave.empty());

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);

    // **The text door first**: every sentence's prompt, as the reference tokenized it, or the waveform
    // below grades another text.
    auto vocab = loom::SopranoVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode(text);
    LOOM_CHECK(ids.size() == ref_ids.size());
    for (size_t i = 0; i < std::min(ids.size(), ref_ids.size()); ++i) {
        LOOM_CHECK(static_cast<float>(ids[i]) == ref_ids[i]);
    }
    const std::vector<double> tokens(ids.begin(), ids.end());

    // --- 1. Greedy: the reference's ids, and its waveform. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"temperature", 0.0}}));
        // 2048 samples per kept row: a length mismatch is a token that went the other way, which makes
        // every per-sample number meaningless -- checked first.
        std::fprintf(stderr, "greedy samples: loom=%zu reference=%zu\n", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "greedy max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        // A waveform collapsed toward silence can differ little for the wrong reason (Retro-006).
        LOOM_CHECK(d.peak > 0.1);
        // ~20x the measured max and ~40x the measured rmse, for other ISAs' accumulation order.
        LOOM_CHECK(d.max_abs < 1e-3);
        LOOM_CHECK(d.rmse < 5e-5);
    }

    // --- 2. Sabotage: no repetition penalty. The comparison above must be able to fail. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"temperature", 0.0}, {"repetition_penalty", 1.0}}));
        std::fprintf(stderr, "sabotage (penalty 1.0) samples: loom=%zu reference=%zu\n", got.size(),
                     ref_wave.size());
        LOOM_CHECK(got.size() != ref_wave.size() || compare(got, ref_wave).rmse > 1e-2);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
