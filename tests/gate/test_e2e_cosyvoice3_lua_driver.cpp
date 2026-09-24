// Validates the CosyVoice3 export (loom-exporter's `cosyvoice3_export.py`) end to end against the
// reference implementation: text in through the file's own vocabulary, the Qwen2 LM's repetition-aware
// sampled decode, the flow's guided ODE over the DiT, CausalHiFT, and a waveform out -- one GGUF, four
// topologies, one driver, and the default voice the export computed.
//
// **The oracle is the real CosyVoice3** (`scripts/cosyvoice3_reference.py`) with every draw pinned and
// handed in. The LM SAMPLES, and greedy is not a usable oracle (an argmax decode stops after five tokens),
// so the reference's two `multinomial` calls per step were replaced by an inverse-CDF walk over recorded
// uniforms -- `draws`, which the driver passes to `loom.sample_row` as `uniform` (ADR-047). The flow's
// initial state and the sine source's uniform noise are the reference's own slices (`noise`,
// `nsf_noise`). Four arms, each on the reference's own input so one phase's rounding is not graded as
// the next phase's error:
//
//   * **LM, exact**: `return_tokens` hands back the raw draws, silence tokens and all. The ids must be
//     identical -- this is the arm that exercises `top_p_mass = "row"`, `banned` and the driver's RAS
//     redraw, which fired 6 times on the default sentence.
//   * **Flow**: the reference's (silence-filtered) tokens in as `speech_tokens`, `return_mel`.
//   * **Vocoder**: the reference's mel through `vocoder` alone, from a probe beside the driver.
//   * **Free-running**: everything from the text, the draws and the two noises.
//
// Fixtures:
//   LOOM_COSYVOICE3_GGUF     cosyvoice3.gguf  -- `loom-export ~/Dev/models/fun-cosyvoice3-0.5b-2512 -o ...`
//   LOOM_COSYVOICE3_REF_DIR  cosyvoice3_ref/  -- scripts/cosyvoice3_reference.py --f64

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

// Measured on the reference's default sentence (76 tokens, 3.04 s), x86-64, 2026-09-24, against the
// reference's own float32-vs-float64 spread on the same draws:
//
//                    loom vs reference            reference f32 vs f64
//   LM tokens        76 / 76 identical            --
//   flow mel         max 1.6e-03  rmse 1.4e-04    max 1.5e-03
//   vocoder          max 8.9e-03  rmse 3.2e-04    max 5.2e-03  rmse 1.6e-04  (the same mel)
//   free-running     max 5.1e-02  rmse 2.7e-03    max 1.4e-01  rmse 9.0e-03
//
// So the flow and the free-running waveform are AT the reference's own floor: a 22-layer DiT integrated
// over 10 guided steps spreads f32 rounding to ~1e-3 on the mel, and the vocoder turns that into the
// waveform numbers. The vocoder arm's excess over its floor is the F0 predictor, which the reference
// runs at float64 ("f0_predictor precision is crucial") and ggml cannot: its f0 sits ~2e-3 Hz from the
// f64 one, and the sine source integrates that into the phase. In torch, the same wrapper with an f32
// F0 lands at max 3.4e-02 / rmse 9.1e-04 from the reference, and with an f64 F0 at its floor --
// which is how the excess was attributed. Bounds are ~6-10x the measurement.
constexpr double COSYVOICE3_MEL_MAX_ABS = 1e-2;
constexpr double COSYVOICE3_MEL_RMSE = 1e-3;
constexpr double COSYVOICE3_VOCODER_MAX_ABS = 5e-2;
constexpr double COSYVOICE3_VOCODER_RMSE = 3e-3;
constexpr double COSYVOICE3_FREE_RMSE = 2e-2;

std::string meta_string(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\": \"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) throw std::runtime_error("cosyvoice3_ref/meta.json has no \"" + key + "\"");
    const size_t start = at + needle.size();
    return json.substr(start, json.find('"', start) - start);
}

std::vector<double> as_doubles(const std::vector<float>& v) { return {v.begin(), v.end()}; }

struct Diff { double max_abs = 0.0, rmse = 0.0, peak = 0.0; };

Diff compare(const std::vector<double>& got, const std::vector<float>& want) {
    Diff d;
    const size_t n = std::min(got.size(), want.size());
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(got[i] - want[i]);
        d.max_abs = std::max(d.max_abs, e);
        sum_sq += e * e;
        d.peak = std::max(d.peak, std::fabs(got[i]));
    }
    d.rmse = std::sqrt(sum_sq / static_cast<double>(std::max<size_t>(n, 1)));
    return d;
}

// The vocoder alone on a caller's mel, beside the driver (the phase probe).
const char* kProbe = R"lua(
    function probe_vocoder(inputs)
        return loom.run_subgraph('vocoder', {n_enc_frames = #inputs.mel / 80, n_past = 0},
                                 {mel = inputs.mel, nsf_noise = inputs.nsf_noise})
    end
)lua";

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_COSYVOICE3_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_COSYVOICE3_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_COSYVOICE3_GGUF and LOOM_COSYVOICE3_REF_DIR "
                              "(see scripts/cosyvoice3_reference.py)\n");
        return 77;
    }
    const std::string ref_dir = ref_env;
    std::ifstream meta_file(ref_dir + "/meta.json");
    LOOM_CHECK(meta_file.good());
    std::stringstream meta_buf;
    meta_buf << meta_file.rdbuf();
    const std::string text = meta_string(meta_buf.str(), "text");
    const std::string prompt_text = meta_string(meta_buf.str(), "prompt_text");

    std::vector<int64_t> shape;
    const auto ref_ids = loom_test::read_npy_f32(ref_dir + "/text_ids.npy", shape);
    const auto ref_prompt_ids = loom_test::read_npy_f32(ref_dir + "/voice_prompt_text.npy", shape);
    const auto draws = loom_test::read_npy_f32(ref_dir + "/draws.npy", shape);
    const auto raw_tokens = loom_test::read_npy_f32(ref_dir + "/speech_tokens_raw.npy", shape);
    const auto flow_tokens = loom_test::read_npy_f32(ref_dir + "/speech_tokens.npy", shape);
    const auto noise = loom_test::read_npy_f32(ref_dir + "/noise.npy", shape);
    const auto nsf_noise = loom_test::read_npy_f32(ref_dir + "/nsf_noise.npy", shape);
    const auto ref_mel_cm = loom_test::read_npy_f32(ref_dir + "/mel.npy", shape);   // (80, m)
    const int64_t n_mel_frames = shape.size() == 2 ? shape[1] : 0;
    const auto ref_wave = loom_test::read_npy_f32(ref_dir + "/wave.npy", shape);
    LOOM_CHECK(!ref_ids.empty() && !draws.empty() && !raw_tokens.empty() && n_mel_frames > 0);
    // The mel FRAME-major, the layout the driver returns and the vocoder takes.
    std::vector<float> ref_mel(ref_mel_cm.size());
    for (int64_t f = 0; f < n_mel_frames; ++f) {
        for (int64_t c = 0; c < 80; ++c) ref_mel[f * 80 + c] = ref_mel_cm[c * n_mel_frames + f];
    }

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);

    // **The text door first**, on both halves: the sentence, and the voice's prompt text, whose
    // `<|endofprompt|>` and CJK are the added-token and byte-level paths the sentence does not reach.
    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode(text);
    LOOM_CHECK(ids.size() == ref_ids.size());
    for (size_t i = 0; i < std::min(ids.size(), ref_ids.size()); ++i) {
        LOOM_CHECK(static_cast<float>(ids[i]) == ref_ids[i]);
    }
    const auto prompt_ids = vocab->encode(prompt_text);
    LOOM_CHECK(prompt_ids.size() == ref_prompt_ids.size());
    for (size_t i = 0; i < std::min(prompt_ids.size(), ref_prompt_ids.size()); ++i) {
        LOOM_CHECK(static_cast<float>(prompt_ids[i]) == ref_prompt_ids[i]);
    }
    const std::vector<double> tokens(ids.begin(), ids.end());

    // --- 1. LM, exact. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"draws", as_doubles(draws)}, {"return_tokens", 1.0}}));
        std::fprintf(stderr, "LM: loom=%zu reference=%zu tokens\n", got.size(), raw_tokens.size());
        LOOM_CHECK(got.size() == raw_tokens.size());
        size_t first_diff = got.size();
        for (size_t i = 0; i < std::min(got.size(), raw_tokens.size()); ++i) {
            if (got[i] != raw_tokens[i]) { first_diff = i; break; }
        }
        if (first_diff < got.size()) {
            std::fprintf(stderr, "LM: first differing token at step %zu: loom %g, reference %g\n",
                         first_diff, got[first_diff], raw_tokens[first_diff]);
        }
        LOOM_CHECK(first_diff == got.size());
    }

    // --- 2. Flow, on the reference's tokens and noise. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"speech_tokens", as_doubles(flow_tokens)}, {"noise", as_doubles(noise)},
            {"return_mel", 1.0}}));
        std::fprintf(stderr, "flow: loom=%zu reference=%zu values\n", got.size(), ref_mel.size());
        LOOM_CHECK(got.size() == ref_mel.size());
        const Diff d = compare(got, ref_mel);
        std::fprintf(stderr, "flow mel max_abs_diff=%g rmse=%g\n", d.max_abs, d.rmse);
        LOOM_CHECK(d.max_abs < COSYVOICE3_MEL_MAX_ABS);
        LOOM_CHECK(d.rmse < COSYVOICE3_MEL_RMSE);
    }

    // --- 3. Vocoder, on the reference's mel and noise. ---
    {
        loom::Session session(*model, device.backends());
        session.bridge().load_script(kProbe);
        const auto got = std::get<std::vector<double>>(session.bridge().call("probe_vocoder", {
            {"mel", as_doubles(ref_mel)}, {"nsf_noise", as_doubles(nsf_noise)}}));
        std::fprintf(stderr, "vocoder: loom=%zu reference=%zu samples\n", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "vocoder max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        // A waveform collapsed toward silence can have a small difference for the wrong reason
        // (Retro-006).
        LOOM_CHECK(d.peak > 0.05);
        LOOM_CHECK(d.max_abs < COSYVOICE3_VOCODER_MAX_ABS);
        LOOM_CHECK(d.rmse < COSYVOICE3_VOCODER_RMSE);
    }

    // --- 4. Free-running: text, draws and noises in, a waveform out. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"draws", as_doubles(draws)}, {"noise", as_doubles(noise)},
            {"nsf_noise", as_doubles(nsf_noise)}}));
        std::fprintf(stderr, "free-running: loom=%zu reference=%zu samples\n", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "free-running max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        LOOM_CHECK(d.peak > 0.05);
        LOOM_CHECK(d.rmse < COSYVOICE3_FREE_RMSE);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
