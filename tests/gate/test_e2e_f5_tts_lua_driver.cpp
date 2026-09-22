// Validates the F5-TTS export (loom-exporter's `f5_tts_export.py`) end to end against the reference
// implementation: one GGUF, four traced topologies, the engine's own guided ODE between two of them,
// and a waveform out.
//
// **The oracle is the real F5-TTS running the same pipeline**, not a frozen output of an earlier loom
// build -- `scripts/f5_tts_reference.py` produces it, and the numbers it is graded at were established
// by decomposing the reference: the mel front end and both text-embedding branches are BIT-IDENTICAL
// to the modules they took over, one estimator evaluation is 7.7e-06 (batch-1 here against the
// reference's packed batch-2 forward), and the integrated mel over 32 Euler steps is 1.42e-05 with
// cosine 0.99999994. What survives into the waveform is that residual through Vocos.
//
// **Three things have to be handed in rather than computed, and each is a real property of this
// family.** `duration` -- the reference measures its transcripts in UTF-8 bytes and the driver in ids,
// equal for ASCII and not in general, so pinning it makes this a comparison about the model.
// `n_ref_text` -- the ids are one concatenated sequence and nothing in them marks where the transcript
// ends. And `noise` -- **the initial state itself, not a seed.** torch's RNG and the engine's are
// different algorithms, so the same seed is a different draw, and flow matching from a different draw
// is a different valid sample: this test measured max |d| 1.25 on audio that was intelligible,
// correctly voiced and simply not the same realization. The driver forwards `noise` to
// `loom.run_ode`'s own `state`, and draws only when a caller supplies none.
//
// Fixtures:
//   LOOM_F5_TTS_GGUF     f5_tts.gguf   -- `loom-export <F5TTS_v1_Base> -o f5_tts.gguf`
//   LOOM_F5_TTS_REF_DIR  f5_tts_ref/   -- ref_audio.npy, text_ids.npy, noise.npy, wave.npy and
//                                         meta.json from scripts/f5_tts_reference.py

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The handful of numbers this test needs out of `meta.json`. A real parser is not worth a dependency
// for six integers, and a missing key is a hard failure rather than a default -- every one of them
// changes the waveform.
double meta_number(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) {
        throw std::runtime_error("f5_tts_ref/meta.json has no \"" + key + "\"");
    }
    const size_t colon = json.find(':', at + needle.size());
    return std::stod(json.substr(colon + 1));
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_F5_TTS_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_F5_TTS_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_F5_TTS_GGUF and LOOM_F5_TTS_REF_DIR "
                              "(see scripts/f5_tts_reference.py)\n");
        return 77;
    }
    const std::string ref_dir = ref_env;

    std::ifstream meta_file(ref_dir + "/meta.json");
    LOOM_CHECK(meta_file.good());
    std::stringstream meta_buf;
    meta_buf << meta_file.rdbuf();
    const std::string meta = meta_buf.str();

    std::vector<int64_t> shape;
    const std::vector<float> ref_audio = loom_test::read_npy_f32(ref_dir + "/ref_audio.npy", shape);
    const std::vector<float> text_ids = loom_test::read_npy_f32(ref_dir + "/text_ids.npy", shape);
    const std::vector<float> noise = loom_test::read_npy_f32(ref_dir + "/noise.npy", shape);
    const std::vector<float> ref_wave = loom_test::read_npy_f32(ref_dir + "/wave.npy", shape);
    LOOM_CHECK(!noise.empty());
    LOOM_CHECK(!ref_audio.empty());
    LOOM_CHECK(!text_ids.empty());
    LOOM_CHECK(!ref_wave.empty());

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);

    // The character table ships in the file, and the ids the reference produced must be the ids this
    // table produces -- otherwise the waveform comparison below would be grading two sentences.
    auto vocab = loom::F5Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    LOOM_CHECK(vocab->filler_offset() == 1);

    const std::string driver_script = model->kv_str("model.driver_script");
    LOOM_CHECK(!driver_script.empty());

    loom::LoomLuaBridge bridge(backend.get());
    for (const char* name : {"mel", "text_embed", "text_embed_uncond", "estimator", "vocoder"}) {
        bridge.register_module(name, *model, loom::GraphTopology::parse(model->topology_json(name)));
    }
    bridge.load_script(driver_script);

    const loom::LoomLuaBridge::Value result = bridge.call("infer", {
        {"waveform", std::vector<double>(ref_audio.begin(), ref_audio.end())},
        {"text_ids", std::vector<double>(text_ids.begin(), text_ids.end())},
        {"noise", std::vector<double>(noise.begin(), noise.end())},
        {"n_ref_text", meta_number(meta, "n_ref_text")},
        {"duration", meta_number(meta, "duration")},
        {"n_steps", meta_number(meta, "n_steps")},
        {"cfg_scale", meta_number(meta, "cfg_scale")},
        {"sway_coef", meta_number(meta, "sway_coef")},
        {"seed", meta_number(meta, "seed")},
    });
    const auto& wave_d = std::get<std::vector<double>>(result);
    std::vector<float> got(wave_d.begin(), wave_d.end());

    std::fprintf(stderr, "samples: loom=%zu reference=%zu\n", got.size(), ref_wave.size());
    LOOM_CHECK(got.size() == ref_wave.size());

    double max_abs_diff = 0.0, sum_sq_diff = 0.0, peak = 0.0;
    for (size_t i = 0; i < ref_wave.size(); ++i) {
        const double d = std::fabs(static_cast<double>(got[i]) - ref_wave[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        sum_sq_diff += d * d;
        peak = std::max(peak, std::fabs(static_cast<double>(got[i])));
    }
    const double rmse = std::sqrt(sum_sq_diff / static_cast<double>(ref_wave.size()));
    std::fprintf(stderr, "max_abs_diff=%g rmse=%g peak=%g\n", max_abs_diff, rmse, peak);

    // **The peak is checked as well as the difference, and it is not redundant.** Kokoro matched
    // PyTorch at cosine 0.996 and shipped unintelligible (Retro-006); a waveform that has collapsed
    // toward silence can have a small absolute difference for exactly the wrong reason. Real speech
    // from this checkpoint lands near 0.85 on this clip.
    LOOM_CHECK(peak > 0.05);

    // 0.02 on a (-1, 1) waveform, the same bound and the same reasoning as Matcha's own MIL-vs-oracle
    // comparison: a per-step residual of ~1e-5 in mel space, accumulated over 32 sequential Euler
    // updates and then amplified through a nonlinear vocoder. It has real margin over what was
    // measured and still catches a driver that is wrong rather than imprecise.
    //
    // **This bound is only meaningful because the noise is pinned.** Drawn independently on each side
    // it was 1.25 -- not a looser tolerance but a different quantity, two valid samples of one
    // distribution. A gate that had been "relaxed" to accommodate that would have measured nothing.
    //
    // **And it can fail**, which is the other half of that claim. Re-running this against a reference
    // generated with `cfg_scale = 1.0` -- guidance off, the one knob that is the model rather than a
    // tolerance -- gives max |d| 1.278 against the 4.14e-03 the passing arm measures. 300x the bound.
    LOOM_CHECK(max_abs_diff < 0.02);

    LOOM_TEST_REPORT_AND_RETURN();
}
