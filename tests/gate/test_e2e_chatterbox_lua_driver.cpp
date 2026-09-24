// Validates the Chatterbox export (loom-exporter's `chatterbox_export.py`) end to end against the
// reference implementation: text in through the file's own vocabulary, T3's guided decode, S3Gen's
// guided ODE, HiFT, and a waveform out -- one GGUF, seven topologies, one driver.
//
// **The oracle is the real Chatterbox running the same pipeline** (`scripts/chatterbox_reference.py`),
// with the watermarker stubbed out because loom ships none. Two things make it an EXACT comparison
// rather than a distributional one, and both are handed in:
//
//   * **T3 decodes GUIDED GREEDY.** The checkpoint samples (temperature 0.8, min_p 0.05), and two
//     samplers on different RNG streams agree on nothing. Classifier-free guidance and the
//     repetition penalty both move an argmax, so greedy still exercises both; `temperature = 0` is the
//     driver's spelling of it. A wrong token here changes the waveform's LENGTH, which is checked
//     first.
//   * **Every random draw is the reference's own**: the ODE's initial state (`noise`) and the NSF
//     source's harmonic phases and per-sample Gaussian (`nsf_phase`, `nsf_noise`). The same seed is
//     not the same noise ([[feedback-same-seed-is-not-same-noise]], F5-TTS's gate).
//
// Measured when this was written: max |d| 2.5e-05, rmse 1.5e-06 over 40,320 samples, against a
// float32-vs-float64 spread of 2.2e-05 in the reference itself -- the export is at the reference's own
// rounding floor. The sabotage arm (the flow's guidance off, `flow_cfg_rate = 0`) gives 0.897.
//
// Fixtures:
//   LOOM_CHATTERBOX_GGUF     chatterbox.gguf  -- `loom-export ~/Dev/models/chatterbox -o chatterbox.gguf`
//   LOOM_CHATTERBOX_REF_DIR  chatterbox_ref/  -- text_ids.npy, noise.npy, nsf_phase.npy, nsf_noise.npy,
//                                                wave.npy and meta.json from
//                                                scripts/chatterbox_reference.py

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

// The one string this test needs out of `meta.json`: the sentence, as the reference was given it.
std::string meta_text(const std::string& json) {
    const std::string needle = "\"text\": \"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) throw std::runtime_error("chatterbox_ref/meta.json has no \"text\"");
    const size_t start = at + needle.size();
    return json.substr(start, json.find('"', start) - start);
}

std::vector<double> as_doubles(const std::vector<float>& v) { return {v.begin(), v.end()}; }

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_CHATTERBOX_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_CHATTERBOX_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_CHATTERBOX_GGUF and LOOM_CHATTERBOX_REF_DIR "
                              "(see scripts/chatterbox_reference.py)\n");
        return 77;
    }
    const std::string ref_dir = ref_env;
    std::ifstream meta_file(ref_dir + "/meta.json");
    LOOM_CHECK(meta_file.good());
    std::stringstream meta_buf;
    meta_buf << meta_file.rdbuf();
    const std::string text = meta_text(meta_buf.str());

    std::vector<int64_t> shape;
    const auto ref_ids = loom_test::read_npy_f32(ref_dir + "/text_ids.npy", shape);
    const auto noise = loom_test::read_npy_f32(ref_dir + "/noise.npy", shape);
    const auto nsf_phase = loom_test::read_npy_f32(ref_dir + "/nsf_phase.npy", shape);
    const auto nsf_noise = loom_test::read_npy_f32(ref_dir + "/nsf_noise.npy", shape);
    const auto ref_wave = loom_test::read_npy_f32(ref_dir + "/wave.npy", shape);
    LOOM_CHECK(ref_ids.size() > 2 && !noise.empty() && nsf_phase.size() == 9 && !ref_wave.empty());

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);

    // **The text door first.** The reference's ids are `[START] + text_to_tokens(punc_norm(text)) +
    // [STOP]`; the file's own vocabulary must produce the middle, or the waveform comparison below
    // would be grading two different sentences.
    auto vocab = loom::ChatterboxVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode(text);
    const std::vector<float> want(ref_ids.begin() + 1, ref_ids.end() - 1);
    LOOM_CHECK(ids.size() == want.size());
    for (size_t i = 0; i < std::min(ids.size(), want.size()); ++i) {
        LOOM_CHECK(static_cast<float>(ids[i]) == want[i]);
    }

    loom::Session session(*model, device.backends());
    const loom::LoomLuaBridge::Value result = session.bridge().call("infer", {
        {"tokens", std::vector<double>(ids.begin(), ids.end())},
        {"temperature", 0.0},
        {"noise", as_doubles(noise)},
        {"nsf_phase", as_doubles(nsf_phase)},
        {"nsf_noise", as_doubles(nsf_noise)},
    });
    const auto& got = std::get<std::vector<double>>(result);

    // The LENGTH is T3's token count times 480 samples, so a disagreement here is a decode that took
    // a different path -- checked before any sample is, because a length mismatch makes every
    // per-sample number below meaningless.
    std::fprintf(stderr, "samples: loom=%zu reference=%zu\n", got.size(), ref_wave.size());
    LOOM_CHECK(got.size() == ref_wave.size());

    double max_abs_diff = 0.0, sum_sq_diff = 0.0, peak = 0.0;
    const size_t n = std::min(got.size(), ref_wave.size());
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(got[i] - ref_wave[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        sum_sq_diff += d * d;
        peak = std::max(peak, std::fabs(got[i]));
    }
    const double rmse = std::sqrt(sum_sq_diff / static_cast<double>(std::max<size_t>(n, 1)));
    std::fprintf(stderr, "max_abs_diff=%g rmse=%g peak=%g\n", max_abs_diff, rmse, peak);

    // The peak as well as the difference: a waveform collapsed toward silence can have a small
    // difference for the wrong reason (Retro-006). This clip peaks at 0.739.
    LOOM_CHECK(peak > 0.05);

    // 2e-3: 80x what was measured, and 450x below the sabotage arm's 0.897. Tighter than F5-TTS's 0.02
    // because there is less between the draws and the waveform here -- 10 Euler steps where F5 takes
    // 32 -- and because the measured residual is already at the reference's own f32-vs-f64 floor, so
    // the margin is for other ISAs' accumulation order and not for anything this export does.
    LOOM_CHECK(max_abs_diff < 2e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
