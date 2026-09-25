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
// The text arrives through `loom::CosyVoice3Vocab`, so each run above goes through the driver's chunk
// loop (one chunk, `<|endoftext|>` first). Two more arms cover what the text path and voice files add:
//
//   * **Chunks**: a two-chunk text run whole must equal its two chunks run one after the other on the
//     same stream -- the split, the per-chunk generation and the join -- and a pinned input is refused
//     for more than one chunk.
//   * **A cloned voice**: a voice FILE written by `cosyvoice3_voices` from a clip the default voice is
//     not (JFK, whisper.cpp's sample), loaded through `loom::load_voice`: its arrays must be the
//     reference front end's for that clip exactly, and with them the LM must draw the reference's
//     tokens exactly and the free-running waveform land inside the default voice's bound -- against
//     the reference's float64 waveform, which on this voice its own f32 run is far from.
//
// Fixtures:
//   LOOM_COSYVOICE3_GGUF           cosyvoice3.gguf        -- `loom-export ~/Dev/models/fun-cosyvoice3-0.5b-2512 -o ...`
//   LOOM_COSYVOICE3_REF_DIR        cosyvoice3_ref/        -- scripts/cosyvoice3_reference.py --f64
//   LOOM_COSYVOICE3_CLONE_REF_DIR  cosyvoice3_clone_ref/  -- the same script with `--prompt-wav jfk.wav
//                                  --prompt-text <its transcript> --seed 12 --f64`, plus `voice.gguf` from
//                                  `python -m loom_exporter.cosyvoice3_voices ... --name voice` (optional)

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
#include <unordered_map>
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

    // **The text door first**, on both halves: the sentence through the reference's text path (one
    // chunk, opened by the header), and the voice's prompt text through the BPE alone -- the reference
    // skips normalisation for it, since it holds `<|endofprompt|>` -- whose added token and CJK are
    // paths the sentence does not reach.
    auto vocab = loom::CosyVoice3Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode(text);
    LOOM_CHECK(ids.size() == ref_ids.size() + 1 && ids[0] == vocab->chunk_header());
    for (size_t i = 0; i + 1 < ids.size() && i < ref_ids.size(); ++i) {
        LOOM_CHECK(static_cast<float>(ids[i + 1]) == ref_ids[i]);
    }
    const auto prompt_ids = vocab->bpe().encode(prompt_text);
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

    // --- 5. Chunks: whole == its chunks in turn, on one stream. ---
    {
        const int32_t header = vocab->chunk_header();
        const auto first = vocab->bpe().encode("Hello there.");
        const auto second = vocab->bpe().encode("See you soon.");
        std::vector<double> both{static_cast<double>(header)};
        both.insert(both.end(), first.begin(), first.end());
        both.push_back(header);
        both.insert(both.end(), second.begin(), second.end());
        std::vector<double> whole, parts;
        {
            loom::Session session(*model, device.backends());
            whole = std::get<std::vector<double>>(session.bridge().call("infer", {{"tokens", both}, {"seed", 5.0}}));
        }
        {
            loom::Session session(*model, device.backends());
            parts = std::get<std::vector<double>>(session.bridge().call("infer", {
                {"tokens", std::vector<double>(first.begin(), first.end())}, {"seed", 5.0}}));
            const size_t n_first = parts.size();
            const auto rest = std::get<std::vector<double>>(session.bridge().call("infer", {
                {"tokens", std::vector<double>(second.begin(), second.end())}}));
            parts.insert(parts.end(), rest.begin(), rest.end());
            std::fprintf(stderr, "chunks: whole=%zu samples, parts=%zu + %zu\n", whole.size(), n_first, rest.size());
            LOOM_CHECK(n_first > 0 && !rest.empty());
        }
        LOOM_CHECK(whole == parts);
        bool refused = false;
        try {
            loom::Session session(*model, device.backends());
            session.bridge().call("infer", {{"tokens", both}, {"draws", as_doubles(draws)}});
        } catch (const std::exception& e) {
            refused = std::string(e.what()).find("pins one generation") != std::string::npos;
        }
        LOOM_CHECK(refused);
    }

    // --- 6. A cloned voice, from a voice file. ---
    const char* clone_env = loom_test::fixture_env("LOOM_COSYVOICE3_CLONE_REF_DIR");
    if (clone_env == nullptr) {
        std::fprintf(stderr, "cloned voice: skipped (no LOOM_COSYVOICE3_CLONE_REF_DIR)\n");
    } else {
        const std::string dir = clone_env;
        std::ifstream clone_meta_file(dir + "/meta.json");
        std::stringstream clone_meta;
        clone_meta << clone_meta_file.rdbuf();
        const auto clone_ids = vocab->encode(meta_string(clone_meta.str(), "text"));
        const auto clone_ref_ids = loom_test::read_npy_f32(dir + "/text_ids.npy", shape);
        LOOM_CHECK(clone_ids.size() == clone_ref_ids.size() + 1);
        const loom::VoiceFile voice = loom::load_voice(*model, dir + "/voice.gguf");
        // The file's arrays ARE the reference front end's, input by input.
        for (const auto& [input, npy] : {std::pair<std::string, std::string>{"prompt_text", "voice_prompt_text"},
                                         {"prompt_speech_tokens", "voice_prompt_speech_tokens"},
                                         {"prompt_feat", "voice_prompt_feat"},
                                         {"embedding", "voice_embedding"}}) {
            const auto want = loom_test::read_npy_f32(dir + "/" + npy + ".npy", shape);
            const auto it = voice.inputs.find(input);
            LOOM_CHECK(it != voice.inputs.end());
            if (it == voice.inputs.end()) continue;
            LOOM_CHECK(it->second.size() == want.size());
            bool same = it->second.size() == want.size();
            for (size_t i = 0; same && i < want.size(); ++i) same = static_cast<float>(it->second[i]) == want[i];
            std::fprintf(stderr, "cloned voice: %s %zu values %s\n", input.c_str(), want.size(),
                         same ? "identical" : "DIFFER");
            LOOM_CHECK(same);
        }
        const auto clone_draws = loom_test::read_npy_f32(dir + "/draws.npy", shape);
        const auto clone_raw = loom_test::read_npy_f32(dir + "/speech_tokens_raw.npy", shape);
        const auto clone_noise = loom_test::read_npy_f32(dir + "/noise.npy", shape);
        const auto clone_nsf = loom_test::read_npy_f32(dir + "/nsf_noise.npy", shape);
        // The FLOAT64 reference, not its f32 run: on this voice (a loud 11 s clip, 254 generated frames)
        // the reference's own f32 waveform is rmse 0.249 from its f64 one -- uncorrelated in four of eight
        // segments -- while loom lands at rmse 6.9e-03 from the f64 one, inside the default voice's own
        // f32-vs-f64 spread (9.0e-03). Measured 2026-09-25; the flow's mel is at its floor either way
        // (max 2.3e-03, rmse 1.6e-04 against the f32 reference).
        const auto clone_wave = loom_test::read_npy_f32(dir + "/wave_f64.npy", shape);
        // The voice's arrays as driver inputs by name, as `loom_cli --voice` and loom-py pass them.
        using Inputs = std::unordered_map<std::string, loom::LoomLuaBridge::Value>;
        auto with_voice = [&](Inputs in) {
            for (const auto& [name, values] : voice.inputs) in.emplace(name, values);
            return in;
        };
        const std::vector<double> clone_tokens(clone_ids.begin(), clone_ids.end());
        // Seed 12, not the script's default: the pinned draws must sit further from a CDF boundary than
        // f32 can blur. At seed 11 step 24's redraw sat 8.0e-07 of the mass from one (over all 6761 ids),
        // and loom took the neighbouring id after 24 identical tokens while the reference at f64 did not
        // move. This run's smallest margin is meta.json's `min_draw_margin`, 1.9e-05.
        bool lm_exact = false;
        {
            loom::Session session(*model, device.backends());
            const auto got = std::get<std::vector<double>>(session.bridge().call("infer", with_voice({
                {"tokens", clone_tokens}, {"draws", as_doubles(clone_draws)}, {"return_tokens", 1.0}})));
            size_t same = 0;
            while (same < std::min(got.size(), clone_raw.size()) && got[same] == clone_raw[same]) ++same;
            std::fprintf(stderr, "cloned voice LM: loom=%zu reference=%zu tokens, %zu identical from the start\n",
                         got.size(), clone_raw.size(), same);
            lm_exact = got.size() == clone_raw.size() && same == got.size();
            LOOM_CHECK(lm_exact);
        }
        // The reference's noise is sized for ITS token count, so a diverged decode cannot use it.
        if (lm_exact) {
            loom::Session session(*model, device.backends());
            const auto got = std::get<std::vector<double>>(session.bridge().call("infer", with_voice({
                {"tokens", clone_tokens}, {"draws", as_doubles(clone_draws)}, {"noise", as_doubles(clone_noise)},
                {"nsf_noise", as_doubles(clone_nsf)}})));
            LOOM_CHECK(got.size() == clone_wave.size());
            const Diff d = compare(got, clone_wave);
            std::fprintf(stderr, "cloned voice free-running: max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs,
                         d.rmse, d.peak);
            LOOM_CHECK(d.peak > 0.05);
            LOOM_CHECK(d.rmse < COSYVOICE3_FREE_RMSE);
        }
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
