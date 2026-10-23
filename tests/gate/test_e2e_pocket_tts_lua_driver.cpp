// Validates the Pocket-TTS export (loom-exporter's `pocket_tts_export.py`) end to end against the
// reference implementation: text in through the file's own vocabulary, the built-in voice seeded into
// the flow LM's KV cache, the latent loop with its one-step flow head and EOS head, Mimi, and a
// waveform out -- one GGUF, five topologies, one driver.
//
// **The oracle is the real Pocket-TTS** (`scripts/pocket_tts_reference.py`), with its one random draw
// per step pinned and handed in (`noise`): the same seed is not the same noise. Three arms:
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
//   * **A voice FILE** (ADR-045), when its fixture is present: `marius.gguf` read by `loom::load_voice`,
//     teacher-forced against the reference run with that voice. max |d| 8.0e-06, rmse 1.9e-07.
//
// Fixtures:
//   LOOM_POCKET_TTS_GGUF     pocket_tts.gguf  -- `loom-export ~/Dev/models/pocket-tts/languages/
//                                                english_2026-09 -o pocket_tts.gguf`
//   LOOM_POCKET_TTS_REF_DIR  pocket_tts_ref/  -- tokens.npy, noise.npy, latents.npy, wave.npy and
//                                                meta.json from scripts/pocket_tts_reference.py
//   LOOM_POCKET_TTS_MARIUS_REF_DIR  pocket_tts_marius_ref/  -- the same with `--voice marius`, plus
//                                                voice.gguf from `loom_exporter.pocket_tts_voices`
//                                                (optional: the arm is skipped without it)

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
    const std::string text = json.substr(start, json.find('"', start) - start);
    // A literal reader: an escape would reach the vocabulary as its six characters and the test would
    // grade a different sentence (an em dash written as `—` did: ids off, then the pinned noise ran
    // out). pocket_tts_reference.py writes UTF-8, so a backslash here is a stale or foreign fixture.
    if (text.find('\\') != std::string::npos) {
        throw std::runtime_error("meta.json's text holds an escape; regenerate it with pocket_tts_reference.py");
    }
    return text;
}

// `"wave_rmse": <number>` from meta.json's `f64` block -- the reference's OWN free-running drift
// between float32 and float64 from the same draws -- or a negative number when the fixture has none.
double meta_f64_wave_rmse(const std::string& json) {
    const std::string needle = "\"wave_rmse\": ";
    const size_t at = json.find(needle);
    return at == std::string::npos ? -1.0 : std::strtod(json.c_str() + at + needle.size(), nullptr);
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
    const double f64_wave_rmse = meta_f64_wave_rmse(meta_buf.str());

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
    // prepare, tokenize) for a one-chunk text; the file's own vocabulary must produce them behind the
    // chunk's header, or the waveform comparison below would be grading two different sentences.
    auto vocab = loom::PocketTtsVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode(text);
    LOOM_CHECK(ids.size() == ref_ids.size() + 1);
    LOOM_CHECK(!ids.empty() && vocab->is_chunk_header(ids[0]));
    for (size_t i = 1; i < std::min(ids.size(), ref_ids.size() + 1); ++i) {
        LOOM_CHECK(static_cast<float>(ids[i]) == ref_ids[i - 1]);
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
        // A drift bound, not an equality. The loop feeds every latent back, so f32 rounding compounds
        // along it, and HOW much is the model's: the full-size model drifts 9.9e-5 here (1e-3 is 10x
        // that, and the sabotage arm is 650x over it), while the small 2026-10 models' own
        // reference drifts up to 2.5e-2 between f32 and f64 from the same draws -- chaos that teacher
        // forcing (arm 1) shows is not error. So the bound is 3x the fixture's own float floor when its
        // meta records one (`--f64`), never tighter than 1e-3; the sample count above stays exact.
        const double drift_bound = std::max(1e-3, 3.0 * f64_wave_rmse);
        std::fprintf(stderr, "free-running drift bound %g (reference f32-vs-f64 rmse %g)\n", drift_bound,
                     f64_wave_rmse);
        LOOM_CHECK(d.rmse < drift_bound);
    }

    // --- 3. A voice file, through the loader both hosts use. ---
    if (const char* voice_env = loom_test::fixture_env("LOOM_POCKET_TTS_MARIUS_REF_DIR")) {
        const std::string dir = voice_env;
        const loom::VoiceFile voice = loom::load_voice(*model, dir + "/voice.gguf");
        LOOM_CHECK(voice.name == "marius" && voice.inputs.count("voice_kv") == 1);
        const auto v_ids = loom_test::read_npy_f32(dir + "/tokens.npy", shape);
        const auto v_noise = loom_test::read_npy_f32(dir + "/noise.npy", shape);
        const auto v_latents = loom_test::read_npy_f32(dir + "/latents.npy", shape);
        const auto v_wave = loom_test::read_npy_f32(dir + "/wave.npy", shape);
        loom::Session session(*model, device.backends());
        std::unordered_map<std::string, loom::LoomLuaBridge::Value> in{
            {"tokens", as_doubles(v_ids)}, {"noise", as_doubles(v_noise)},
            {"teacher_latents", as_doubles(v_latents)}};
        for (const auto& [name, values] : voice.inputs) in[name] = values;
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", in));
        std::fprintf(stderr, "voice file samples: loom=%zu reference=%zu\n", got.size(), v_wave.size());
        LOOM_CHECK(got.size() == v_wave.size());
        const Diff d = compare(got, v_wave);
        std::fprintf(stderr, "voice file max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        LOOM_CHECK(d.peak > 0.05);
        LOOM_CHECK(d.max_abs < 2e-3);
        LOOM_CHECK(d.rmse < 2e-5);
    } else {
        std::fprintf(stderr, "voice-file arm skipped: no LOOM_POCKET_TTS_MARIUS_REF_DIR\n");
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
