// Validates the MIL-compiler-exported Whisper GGUF (tools/loom_mil_compiler/whisper_export.py,
// BACKLOG.md P4.1) against HF's own `WhisperForConditionalGeneration` -- the library, not this repo's
// bespoke converter.
//
// **This is the whole Whisper gate.** It replaced four tests built on the bespoke converter
// (`test_e2e_whisper_{encoder,decoder}_reference`, `test_e2e_whisper_driver`,
// `test_e2e_whisper_lua_driver`), which went with `tools/convert_whisper/` and
// `src/core/whisper_driver.cpp` under R6. Their coverage is carried here deliberately, check for check,
// which is why the halves are tested separately rather than only end to end -- and against a stronger
// oracle: those tests compared this engine's two implementations with each other, while every check
// below compares it with HuggingFace.
//
//   1.  the `encoder` topology's hidden states, against the same tensor HF's encoder produced. This half
//       contains the mel frontend, which the exported graph computes from a raw waveform and HF computes
//       in `WhisperFeatureExtractor` -- so it is the half where an inaccuracy would be arithmetic rather
//       than orchestration.
//   1b. the `decoder` topology teacher-forced over the whole prompt in one pass, logits and per-row
//       argmax -- what `test_e2e_whisper_decoder_reference` checked.
//   2.  the whole embedded Lua driver -- encoder once, the prompt it builds for itself, then the
//       KV-cached cross-attention decode loop -- against HF's greedy token sequence, with the language
//       both pinned and auto-detected. Integer equality, no tolerance: both are deterministic argmax.
//   3.  the same, with a `<|startofprev|>` context in front of the prompt -- long-form conditioning,
//       checked against an HF run that saw the same context.
//
// **Nothing here names a per-model C++ struct**, unlike its bespoke sibling: which topology needs a
// cache is `GraphTopology::uses_kv_cache()`'s answer, how big to make it is `make_kv_cache(*model)`'s,
// and how long a waveform to hand over is the GGUF's own `loom.n_samples`. That is the "self-contained
// artifact" claim being true rather than asserted (P4.0.8's first follow-up).
//
// Not generated at ctest time (needs the real checkpoint + coremltools) -- skips cleanly if the fixture
// isn't present. To (re)generate:
//   ./loom-export /home/flavio/Dev/models/whisper-small -o whisper_mil.gguf
//   python3 tools/fixture_gen/reference_forward_whisper_mil.py /home/flavio/Dev/models/whisper-small <ref>

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

// Minimal .npy reader for the two dtypes this fixture writes, matching the one
// test_e2e_whisper_lua_driver.cpp already carries.
std::string npy_header(std::ifstream& f, std::vector<int64_t>& shape_out) {
    char magic[6];
    f.read(magic, 6);
    f.ignore(2);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    const size_t shape_pos = header.find("'shape':");
    const size_t paren_open = header.find('(', shape_pos);
    const size_t paren_close = header.find(')', paren_open);
    std::stringstream ss(header.substr(paren_open + 1, paren_close - paren_open - 1));
    shape_out.clear();
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::string trimmed;
        for (char c : tok) if (c != ' ') trimmed += c;
        if (!trimmed.empty()) shape_out.push_back(std::stoll(trimmed));
    }
    return header;
}

template <typename T>
std::vector<T> read_npy(const std::string& path, std::vector<int64_t>& shape_out) {
    std::ifstream f(path, std::ios::binary);
    LOOM_CHECK(static_cast<bool>(f));
    npy_header(f, shape_out);
    int64_t total = 1;
    for (int64_t d : shape_out) total *= d;
    std::vector<T> data(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(data.data()), total * static_cast<int64_t>(sizeof(T)));
    return data;
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_WHISPER_MIL_GGUF");
    const std::string gguf_path = gguf_env != nullptr ? gguf_env : "whisper_mil.gguf";
    const char* ref_env = std::getenv("LOOM_WHISPER_MIL_REF_DIR");
    const std::string ref_dir = ref_env != nullptr ? ref_env : "";

    if (!path_exists(gguf_path) || ref_dir.empty() ||
        !path_exists(ref_dir + "/ref_mil_generated.npy")) {
        std::fprintf(stderr,
                      "skipping: MIL-exported Whisper GGUF ('%s') or LOOM_WHISPER_MIL_REF_DIR fixture "
                      "not found (run loom-export and tools/fixture_gen/reference_forward_whisper_mil.py)\n",
                      gguf_path.c_str());
        return kSkipReturnCode;
    }

    std::vector<int64_t> wav_shape, prompt_shape, gen_shape, enc_shape, lang_shape;
    const std::vector<float> waveform = read_npy<float>(ref_dir + "/ref_mil_waveform.npy", wav_shape);
    const std::vector<int32_t> prompt = read_npy<int32_t>(ref_dir + "/ref_mil_prompt.npy", prompt_shape);
    const std::vector<int32_t> expected = read_npy<int32_t>(ref_dir + "/ref_mil_generated.npy", gen_shape);
    const std::vector<float> expected_encoder = read_npy<float>(ref_dir + "/ref_mil_encoder.npy", enc_shape);
    const std::vector<int32_t> language_ref = read_npy<int32_t>(ref_dir + "/ref_mil_language.npy", lang_shape);
    std::vector<int64_t> dec_shape, cond_shape, prev_shape;
    const std::vector<float> expected_decoder = read_npy<float>(ref_dir + "/ref_mil_decoder.npy", dec_shape);
    // Added with the long-form conditioning check; absent on a fixture generated before it, which is a
    // skip of that one check rather than a failure.
    const std::vector<int32_t> conditioned_ref =
        path_exists(ref_dir + "/ref_mil_conditioned.npy")
            ? read_npy<int32_t>(ref_dir + "/ref_mil_conditioned.npy", cond_shape) : std::vector<int32_t>{};
    const std::vector<int32_t> prev_raw =
        path_exists(ref_dir + "/ref_mil_prev_raw.npy")
            ? read_npy<int32_t>(ref_dir + "/ref_mil_prev_raw.npy", prev_shape) : std::vector<int32_t>{};

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);

    // The clip length the encoder graph was built at, read from the artifact rather than hardcoded.
    const uint32_t n_samples = model->hparam_u32("n_samples");
    std::fprintf(stderr, "loom.n_samples = %u, fixture waveform = %zu\n", n_samples, waveform.size());
    LOOM_CHECK(waveform.size() == n_samples);

    // --- 1. the encoder half, on its own ---
    {
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("encoder"));
        LOOM_CHECK(!topo.uses_kv_cache());
        loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
        const loom::GraphBuilder::BuildResult& result =
            builder.build({{"n_samples", n_samples}, {"n_past", 0}});

        ggml_tensor* waveform_t = result.input_tensors.at("waveform");
        ggml_backend_tensor_set(waveform_t, waveform.data(), 0, waveform.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), result.graph);

        // HF returns (1, n_audio_ctx, d_model); the exported topology's ne is [d_model, n_audio_ctx],
        // which is the same memory order.
        LOOM_CHECK(expected_encoder.size() ==
                   static_cast<size_t>(result.output->ne[0]) * static_cast<size_t>(result.output->ne[1]));
        std::vector<float> actual(expected_encoder.size());
        ggml_backend_tensor_get(result.output, actual.data(), 0, actual.size() * sizeof(float));

        float max_abs_diff = 0.0f;
        double sum_abs_diff = 0.0;
        float ref_absmax = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float d = std::fabs(actual[i] - expected_encoder[i]);
            max_abs_diff = std::max(max_abs_diff, d);
            sum_abs_diff += d;
            ref_absmax = std::max(ref_absmax, std::fabs(expected_encoder[i]));
        }
        const double mean_abs_diff = sum_abs_diff / static_cast<double>(actual.size());
        std::fprintf(stderr, "encoder vs HF: mean_abs_diff=%g max_abs_diff=%g ref_absmax=%g "
                              "(max/absmax=%g)\n",
                     mean_abs_diff, static_cast<double>(max_abs_diff), static_cast<double>(ref_absmax),
                     static_cast<double>(max_abs_diff / ref_absmax));
        // Relative to the reference's own scale rather than an absolute epsilon: a Whisper encoder's
        // post-`ln_post` activations reach ~30, so a fixed bound would be a statement about this
        // checkpoint's dynamic range instead of about agreement. The two implementations differ only in
        // f32 accumulation order across 12 layers at 1500 positions -- the frontend and the wrapper are
        // bit-identical to HF in torch (see reference_forward_whisper_mil.py) -- and this is the gate
        // that localizes a real divergence to this half. The whole-driver check below is exact.
        LOOM_CHECK(max_abs_diff < 1e-3f * ref_absmax);
        LOOM_CHECK(mean_abs_diff < 1e-3);
    }

    // --- 1b. the decoder half, teacher-forced on the prompt in one pass ---
    //
    // Inherited from the retired `test_e2e_whisper_decoder_reference` (BACKLOG.md P4.1): n_past=0 with
    // n_tokens=T covers the whole causal triangle in a single call, so the decoder's own numbers are
    // checked rather than only the tokens the loop picks from them. Run through a hand-written script
    // rather than a bare GraphBuilder so the mask and positions come from `loom.causal_mask`/`loom.range`
    // -- the same host math a driver uses, instead of a second implementation living in a test.
    {
        loom::GraphTopology encoder_topo = loom::GraphTopology::parse(model->topology_json("encoder"));
        loom::GraphTopology decoder_topo = loom::GraphTopology::parse(model->topology_json("decoder"));
        std::unique_ptr<loom::KvCache> kv_cache = loom::make_kv_cache(*model, backend.get());
        loom::LoomLuaBridge bridge(backend.get());
        bridge.register_module("encoder", *model, std::move(encoder_topo), /*kv_cache=*/nullptr);
        bridge.register_module("decoder", *model, std::move(decoder_topo), kv_cache.get());
        bridge.load_script(R"lua(
            function decoder_logits(inputs)
                loom.run_subgraph_and_retain('encoder', {n_samples = inputs.n_samples, n_past = 0},
                                              {waveform = inputs.waveform})
                local n = #inputs.tokens
                return loom.run_subgraph('decoder', {n_tokens = n, n_past = 0}, {
                    tokens = inputs.tokens, position_ids = loom.range(0, n),
                    attention_mask = loom.causal_mask(n, 0), xa = {from = 'encoder'},
                })
            end
        )lua");

        const std::vector<double> waveform_d(waveform.begin(), waveform.end());
        const std::vector<double> prompt_d(prompt.begin(), prompt.end());
        const loom::LoomLuaBridge::Value value = bridge.call("decoder_logits", {
            {"waveform", waveform_d}, {"tokens", prompt_d},
            {"n_samples", static_cast<double>(n_samples)},
        });
        const auto& logits = std::get<std::vector<double>>(value);
        LOOM_CHECK(logits.size() == expected_decoder.size());

        const size_t n_tokens = prompt.size();
        const size_t n_vocab = logits.size() / n_tokens;
        double max_abs_diff = 0.0, sum_abs_diff = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            const double d = std::fabs(logits[i] - static_cast<double>(expected_decoder[i]));
            max_abs_diff = std::max(max_abs_diff, d);
            sum_abs_diff += d;
        }
        std::fprintf(stderr, "decoder vs HF: n_tokens=%zu n_vocab=%zu mean_abs_diff=%g max_abs_diff=%g\n",
                     n_tokens, n_vocab, sum_abs_diff / static_cast<double>(logits.size()), max_abs_diff);
        LOOM_CHECK(sum_abs_diff / static_cast<double>(logits.size()) < 1e-2);
        LOOM_CHECK(max_abs_diff < 5.0);

        // And the per-row argmax, which is what the loop actually consumes: a logits tensor can be
        // within tolerance everywhere and still pick a different token at a near-tie.
        for (size_t t = 0; t < n_tokens; ++t) {
            size_t best = 0, ref_best = 0;
            for (size_t v = 1; v < n_vocab; ++v) {
                if (logits[t * n_vocab + v] > logits[t * n_vocab + best]) best = v;
                if (expected_decoder[t * n_vocab + v] > expected_decoder[t * n_vocab + ref_best]) ref_best = v;
            }
            LOOM_CHECK(best == ref_best);
        }
    }

    // --- 2. the whole driver, both ways round the language: pinned, and auto-detected ---
    {
        loom::GraphTopology encoder_topo = loom::GraphTopology::parse(model->topology_json("encoder"));
        loom::GraphTopology decoder_topo = loom::GraphTopology::parse(model->topology_json("decoder"));
        LOOM_CHECK(decoder_topo.uses_kv_cache());
        std::unique_ptr<loom::KvCache> kv_cache = loom::make_kv_cache(*model, backend.get());

        loom::LoomLuaBridge bridge(backend.get());
        bridge.register_module("encoder", *model, std::move(encoder_topo), /*kv_cache=*/nullptr);
        bridge.register_module("decoder", *model, std::move(decoder_topo), kv_cache.get());
        bridge.load_script(model->kv_str("model.driver_script"));

        const std::vector<double> waveform_d(waveform.begin(), waveform.end());
        // The artifact's own eos, from the vocab KVs the export embedded -- not a constant this test
        // carries, which is the same "read it from the file" point as `n_samples` above.
        const double eos = static_cast<double>(model->kv_i32("tokenizer.ggml.eos_token_id", -1));
        const int32_t ref_language = language_ref.empty() ? -1 : language_ref[0];

        // **The test passes no token prefix at all.** The driver builds it: start-of-transcript, the
        // language, the task, no-timestamps -- from ids `whisper_export` read off the checkpoint. That
        // is the whole point of the constants living in the driver, so a host needs to know nothing
        // about Whisper's prompt convention.
        auto run = [&](bool pin_language) {
            std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {
                {"waveform", waveform_d},
                {"max_new_tokens", static_cast<double>(expected.size())},
                {"eos_token", eos},
            };
            // Omitting `language` entirely is what asks the driver to detect it -- the resolution order
            // the whole feature is about. Pinning it is the override.
            if (pin_language && ref_language >= 0) {
                args["language"] = static_cast<double>(ref_language);
            }
            // The Value is held in a named local before it is unpacked, deliberately: `call` returns by
            // value, so binding a reference straight into `std::get<...>(call(...))` reads the vector
            // after the temporary variant holding it has been destroyed. It fails in exactly the way
            // that is hardest to read -- the first elements come back as freed-heap garbage while the
            // later ones still look right, so the driver appears to compute a wrong prefix.
            const loom::LoomLuaBridge::Value result = bridge.call("infer", args);
            const auto& out = std::get<std::vector<double>>(result);
            std::vector<int32_t> ids;
            ids.reserve(out.size());
            for (double v : out) ids.push_back(static_cast<int32_t>(v));
            return ids;
        };

        std::fprintf(stderr, "HF language %d, HF generated %zu tokens: ", ref_language, expected.size());
        for (int32_t t : expected) std::fprintf(stderr, "%d ", t);
        std::fprintf(stderr, "\n");

        for (bool pin : {true, false}) {
            const std::vector<int32_t> generated = run(pin);
            std::fprintf(stderr, "MIL (%s) generated %zu tokens: ",
                         pin ? "language pinned" : "language auto-detected", generated.size());
            for (int32_t t : generated) std::fprintf(stderr, "%d ", t);
            std::fprintf(stderr, "\n");
            LOOM_CHECK(generated.size() == expected.size());
            for (size_t i = 0; i < generated.size(); ++i) {
                LOOM_CHECK(generated[i] == expected[i]);
            }
        }

        // Auto-detection agreeing with HF is implied by the token sequences above matching -- a
        // different language would change the prompt and therefore the very first generated token --
        // but the prompt the fixture recorded is checked directly too, so a failure says which half
        // went wrong rather than only that something did.
        if (ref_language >= 0) {
            LOOM_CHECK(prompt.size() >= 2);
            LOOM_CHECK(prompt[1] == ref_language);
        }

        // --- 3. the same audio decoded with a `<|startofprev|>` context in front of the prompt ---
        //
        // Two properties in one comparison, and the fixture is built so that failing either one shows:
        //
        //   * the context REACHES the model. HF's conditioned output differs completely from its
        //     unconditioned one (the fixture uses a real sentence for exactly this reason -- carrying
        //     this run's own output forward changes nothing, so it would pass even if `prev_tokens`
        //     were ignored). A driver that dropped the argument would produce check 2's tokens here.
        //   * the context is FILTERED. What the driver is handed includes a timestamp,
        //     `<|notimestamps|>` and eos appended to the sentence; HF's oracle saw the sentence alone.
        //     A driver that fed those through would condition on three tokens HF never saw.
        if (!prev_raw.empty()) {
            LOOM_CHECK(conditioned_ref != expected); // the fixture itself must be able to fail
            std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {
                {"waveform", waveform_d},
                {"max_new_tokens", static_cast<double>(conditioned_ref.size())},
                {"eos_token", eos},
                {"prev_tokens", std::vector<double>(prev_raw.begin(), prev_raw.end())},
            };
            if (ref_language >= 0) args["language"] = static_cast<double>(ref_language);

            const loom::LoomLuaBridge::Value result = bridge.call("infer", args);
            const auto& out = std::get<std::vector<double>>(result);
            std::vector<int32_t> generated;
            generated.reserve(out.size());
            for (double v : out) generated.push_back(static_cast<int32_t>(v));

            std::fprintf(stderr, "HF  conditioned %zu tokens: ", conditioned_ref.size());
            for (int32_t t : conditioned_ref) std::fprintf(stderr, "%d ", t);
            std::fprintf(stderr, "\nMIL conditioned %zu tokens: ", generated.size());
            for (int32_t t : generated) std::fprintf(stderr, "%d ", t);
            std::fprintf(stderr, "\n");

            LOOM_CHECK(generated.size() == conditioned_ref.size());
            for (size_t i = 0; i < generated.size(); ++i) {
                LOOM_CHECK(generated[i] == conditioned_ref[i]);
            }
        }
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
