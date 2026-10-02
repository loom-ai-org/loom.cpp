// Validates LFM2.5-Audio's speech-to-text export (loom-exporter's `lfm25_audio_export.py`) end to end
// against liquid-audio's own `generate_sequential` (`scripts/lfm25_audio_reference.py`): NeMo's log-mel
// and FastConformer, the adapter, and the LFM2 hybrid LM with BOTH caches -- KV for its 6 attention
// blocks, conv state for its 10 short-convolution blocks.
//
// Four arms, each a tensor or an id sequence:
//   1. ROWS: the encoder's audio rows against the reference's prefill embeddings at the audio
//      positions. Measured max|d| 9.1e-06 (absmax 3.4).
//   2. LOGITS, teacher-forced: the reference's whole prompt (157 rows, audio included) as ONE prefill,
//      then every drawn id fed back; every step's logits. Measured 1.9e-05, argmax 27/27.
//   3. IDS: the driver's own run -- the prompt in three cached segments -- against the reference's ids.
//   4. SABOTAGE: the same clip with its last 160 samples dropped must change the rows' count or values.
//
// Fixtures:
//   LOOM_LFM25_AUDIO_ASR_GGUF  lfm25_audio_asr.gguf  -- `loom-export ~/Dev/models/lfm2.5-audio-1.5b`
//   LOOM_LFM25_AUDIO_REF_DIR   lfm25_audio_ref/       -- from scripts/lfm25_audio_reference.py

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

const char* kTeacherForced = R"(
    local n = #inputs.emb / inputs.width
    local out = {}
    local function head(rows)
        loom.run_subgraph_and_retain('lm_head', {n_tokens = 1, n_past = 0},
            {hidden = {from = 'decoder', row = rows - 1, rows = 1}})
        local l = loom.get_output('lm_head', 1) for i = 1, #l do out[#out + 1] = l[i] end
    end
    loom.run_subgraph_and_retain('decoder', {n_tokens = n, n_past = 0},
        {inputs_embeds = inputs.emb, position_ids = loom.range(0, n), attention_mask = loom.causal_mask(n, 0)})
    head(n)
    local p = n
    for k = 1, #inputs.ids do
        loom.run_subgraph_and_retain('embed', {n_tokens = 1, n_past = 0}, {tokens = {inputs.ids[k]}})
        loom.run_subgraph_and_retain('decoder', {n_tokens = 1, n_past = p},
            {inputs_embeds = {from = 'embed'}, position_ids = {p}, attention_mask = loom.causal_mask(1, p)})
        p = p + 1
        head(1)
    end
    return out
)";

const char* kEncoder = R"(
    loom.run_subgraph_and_retain('encoder', {n_samples = #inputs.waveform, n_past = 0},
        {waveform = inputs.waveform, length = {#inputs.waveform}})
    return loom.get_output('encoder', 1)
)";

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_LFM25_AUDIO_ASR_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_LFM25_AUDIO_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_LFM25_AUDIO_ASR_GGUF and LOOM_LFM25_AUDIO_REF_DIR "
                              "(see scripts/lfm25_audio_reference.py)\n");
        return 77;
    }
    const std::string ref_dir = ref_env;
    std::vector<int64_t> shape;
    const auto wave = loom_test::read_npy_f32(ref_dir + "/wave16k.npy", shape);
    const auto in_emb = loom_test::read_npy_f32(ref_dir + "/in_emb.npy", shape);   // [rows, width]
    const size_t width = static_cast<size_t>(shape[1]), prompt_rows = static_cast<size_t>(shape[0]);
    const auto flags = loom_test::read_npy_f32(ref_dir + "/modality_flag.npy", shape);
    const auto ids = loom_test::read_npy_f32(ref_dir + "/ids.npy", shape);
    const auto logits = loom_test::read_npy_f32(ref_dir + "/step_logits.npy", shape);  // [ids, vocab]
    const size_t vocab = static_cast<size_t>(shape[1]);
    LOOM_CHECK(!wave.empty() && flags.size() == prompt_rows && logits.size() == ids.size() * vocab);

    // The reference's audio rows: the prefill rows whose modality is AUDIO_IN (the one flag that is
    // neither text, the first, nor audio-out). The prompt's text is far shorter than the audio, so the
    // most frequent flag is it.
    std::vector<size_t> audio_rows;
    {
        float audio_flag = flags[0];
        size_t best = 0;
        for (float f : flags) {
            const size_t c = static_cast<size_t>(std::count(flags.begin(), flags.end(), f));
            if (c > best) { best = c; audio_flag = f; }
        }
        for (size_t i = 0; i < flags.size(); ++i) {
            if (flags[i] == audio_flag) audio_rows.push_back(i);
        }
    }

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);
    loom::Session session(*model, device.backends());
    session.bridge().load_script(std::string("function rows(inputs)\n") + kEncoder + "\nend\n" +
                                 "function teacher(inputs)\n" + kTeacherForced + "\nend\n");
    const std::vector<double> wav(wave.begin(), wave.end());

    // --- 1. Rows. ---
    {
        const auto got = std::get<std::vector<double>>(session.bridge().call("rows", {{"waveform", wav}}));
        LOOM_CHECK(got.size() == audio_rows.size() * width);
        double max_abs = 0.0;
        for (size_t r = 0; r < audio_rows.size() && (r + 1) * width <= got.size(); ++r) {
            for (size_t c = 0; c < width; ++c) {
                max_abs = std::max(max_abs, std::fabs(got[r * width + c] - in_emb[audio_rows[r] * width + c]));
            }
        }
        std::fprintf(stderr, "rows: %zu, max|d| %g\n", got.size() / width, max_abs);
        LOOM_CHECK(max_abs < 2e-4);
    }

    // --- 2. Logits, teacher-forced. ---
    {
        std::vector<double> fed;
        for (size_t i = 0; i + 1 < ids.size(); ++i) fed.push_back(ids[i]);
        const auto got = std::get<std::vector<double>>(session.bridge().call("teacher", {
            {"emb", std::vector<double>(in_emb.begin(), in_emb.end())}, {"ids", fed},
            {"width", static_cast<double>(width)}}));
        LOOM_CHECK(got.size() == logits.size());
        double max_abs = 0.0;
        for (size_t i = 0; i < std::min(got.size(), logits.size()); ++i) {
            max_abs = std::max(max_abs, std::fabs(got[i] - logits[i]));
        }
        std::fprintf(stderr, "teacher-forced logits: max|d| %g over %zu steps\n", max_abs, ids.size());
        // ~10x the measured 1.9e-05.
        LOOM_CHECK(max_abs < 2e-4);
    }

    // --- 3. Ids, the driver's own segmented prompt. ---
    {
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"waveform", wav}, {"length", std::vector<double>{static_cast<double>(wav.size())}}}));
        // The reference's last id is `<|im_end|>`, which the driver stops on rather than returns.
        const std::vector<double> want(ids.begin(), ids.end() - 1);
        std::fprintf(stderr, "ids: loom=%zu reference=%zu\n", got.size(), want.size());
        LOOM_CHECK(got == want);
    }

    // --- 4. Sabotage: a different clip must give different rows. ---
    {
        const std::vector<double> short_wav(wav.begin(), wav.end() - 160);
        const auto got = std::get<std::vector<double>>(session.bridge().call("rows", {{"waveform", short_wav}}));
        double max_abs = 0.0;
        for (size_t i = 0; i < std::min(got.size(), audio_rows.size() * width); ++i) {
            max_abs = std::max(max_abs, std::fabs(got[i] - in_emb[audio_rows[i / width] * width + i % width]));
        }
        std::fprintf(stderr, "sabotage (160 samples short): max|d| %g\n", max_abs);
        LOOM_CHECK(max_abs > 1e-3);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
