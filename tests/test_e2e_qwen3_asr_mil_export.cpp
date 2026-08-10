// Validates the MIL-compiler-exported Qwen3-ASR GGUF (tools/loom_mil_compiler/speech_lm_export.py +
// qwen3_asr_export.py, BACKLOG.md P4.3) against HF's own `Qwen3ASRForConditionalGeneration`.
//
// **This is family 3's gate** -- the audio-encoder + projector + causal-LM composition, the largest
// group on EXPORT-ROADMAP.md's R5 table (~19 converters, ~36 models). Three checks, in the order that
// makes a failure readable:
//
//   1. the `encoder` topology's output -- the mel frontend, the chunked conv stem, the window-attention
//      stack and the projector, as one graph -- against the same tensor HF's `get_audio_features`
//      produced. A TENSOR oracle, and it is first deliberately: P4.2 established that a wrong encoder
//      can still decode a plausible transcript (71 of 80 tokens right), so token agreement alone is not
//      evidence that this half is correct.
//   2. the whole embedded Lua driver -- encoder once, the prompt fed to the KV cache as text/audio/text
//      segments, then the decode loop -- against HF's greedy token sequence. Integer equality, no
//      tolerance: both are deterministic argmax.
//   3. `audio_samples`: that omitting it means "all of it", and that supplying a different one changes
//      the answer.
//
// **Check 2 is what proves the composition mechanism**, and specifically that feeding a prompt as N
// successive cached calls is identical to feeding it concatenated: nearly every one of this prompt's
// positions is an audio embedding the LM never produced, written into the cache by its own call at
// `n_past = 9`. If that equivalence did not hold, the first generated token would already differ.
//
// **The fixture's audio deliberately does not fill its last chunk** (BACKLOG.md P4.3e). HF is run on
// the real audio and this test hands the driver the padded waveform plus the real length, so what is
// being compared is that the exported encoder produces HF's rows for a chunk it only partly fills --
// which is the case the family's contract creates and the one nothing checked before. An encoder
// without that handling still exports, still runs and still transcribes plausibly; it is simply out by
// 1.0e-01 on rows whose absmax is 0.128, which is the sort of wrongness only a tensor oracle on a
// partial chunk can see.
//
// **Nothing here names a per-model C++ struct.** How long a waveform to hand over comes from the GGUF's
// own `loom.samples_per_chunk`, whether a topology needs a cache is `GraphTopology::uses_kv_cache()`'s
// answer, and how big to make it is `make_kv_cache(*model)`'s. The driver supplies its own prompt and
// its own stop tokens, so this test passes a waveform and a length.
//
// Not generated at ctest time (needs the real checkpoint + coremltools + transformers >= 5.13) -- skips
// cleanly if the fixture isn't present. To (re)generate:
//   ./loom-export /home/flavio/Dev/models/qwen3-asr-0.6b-hf -o qwen3_asr_mil.gguf
//   python3 tools/fixture_gen/reference_forward_qwen3_asr_mil.py \
//       /home/flavio/Dev/models/qwen3-asr-0.6b-hf <ref>

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
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
// test_e2e_whisper_mil_export.cpp already carries.
void npy_header(std::ifstream& f, std::vector<int64_t>& shape_out) {
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
    const char* gguf_env = std::getenv("LOOM_QWEN3_ASR_MIL_GGUF");
    const std::string gguf_path = gguf_env != nullptr ? gguf_env : "qwen3_asr_mil.gguf";
    const char* ref_env = std::getenv("LOOM_QWEN3_ASR_MIL_REF_DIR");
    const std::string ref_dir = ref_env != nullptr ? ref_env : "";

    if (!path_exists(gguf_path) || ref_dir.empty() ||
        !path_exists(ref_dir + "/ref_asr_generated.npy")) {
        std::fprintf(stderr,
                     "skipping: MIL-exported Qwen3-ASR GGUF ('%s') or LOOM_QWEN3_ASR_MIL_REF_DIR "
                     "fixture not found (run loom-export and "
                     "tools/fixture_gen/reference_forward_qwen3_asr_mil.py)\n",
                     gguf_path.c_str());
        return kSkipReturnCode;
    }

    std::vector<int64_t> wav_shape, enc_shape, audio_shape, gen_shape, prompt_shape, valid_shape;
    const std::vector<float> waveform = read_npy<float>(ref_dir + "/ref_asr_waveform.npy", wav_shape);
    // What the DRIVER hands the encoder: the same waveform with the frontend's own STFT reflection
    // written over the head of the caller's zero padding (BACKLOG.md P4.3e). Check 1 drives that
    // topology directly, so it needs the driver's input rather than the host's.
    const std::vector<float> encoder_in = read_npy<float>(ref_dir + "/ref_asr_encoder_in.npy", enc_shape);
    const std::vector<float> expected_audio = read_npy<float>(ref_dir + "/ref_asr_audio.npy", audio_shape);
    const std::vector<int32_t> expected = read_npy<int32_t>(ref_dir + "/ref_asr_generated.npy", gen_shape);
    const std::vector<int32_t> prompt_len =
        read_npy<int32_t>(ref_dir + "/ref_asr_prompt_len.npy", prompt_shape);
    const std::vector<int32_t> valid = read_npy<int32_t>(ref_dir + "/ref_asr_valid.npy", valid_shape);
    LOOM_CHECK(!valid.empty());
    const auto valid_samples = static_cast<size_t>(valid[0]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);

    // The encoder's contract, read from the artifact rather than hardcoded: the waveform is a whole
    // number of chunks, and each one becomes `frames_per_chunk` prompt positions.
    const uint32_t samples_per_chunk = model->hparam_u32("samples_per_chunk");
    const uint32_t frames_per_chunk = model->hparam_u32("frames_per_chunk");
    LOOM_CHECK(samples_per_chunk > 0 && frames_per_chunk > 0);
    LOOM_CHECK(waveform.size() % samples_per_chunk == 0);
    LOOM_CHECK(encoder_in.size() == waveform.size());
    const size_t n_chunks = waveform.size() / samples_per_chunk;
    const size_t hf_rows = static_cast<size_t>(audio_shape[0]);
    std::fprintf(stderr,
                 "loom.samples_per_chunk = %u, loom.frames_per_chunk = %u, fixture waveform = %zu "
                 "(%zu real samples in %zu chunks -> %zu padded rows, %zu of them real)\n",
                 samples_per_chunk, frames_per_chunk, waveform.size(), valid_samples, n_chunks,
                 n_chunks * frames_per_chunk, hf_rows);
    // **The fixture must land mid-chunk or this file checks nothing it is here to check** (BACKLOG.md
    // P4.3e). HF is run on the real audio and produces fewer rows than the padded waveform's own count;
    // an equality here would mean the generator was handed a whole number of chunks, and every
    // comparison below would pass without ever exercising the padding.
    LOOM_CHECK(valid_samples > 0 && valid_samples < waveform.size());
    LOOM_CHECK(hf_rows < n_chunks * frames_per_chunk);

    // --- 1. the encoder half, on its own: a tensor oracle, not a token oracle ---
    {
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("encoder"));
        LOOM_CHECK(!topo.uses_kv_cache());
        loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
        const loom::GraphBuilder::BuildResult& result = builder.build(
            {{"n_samples", static_cast<uint32_t>(encoder_in.size())}, {"n_past", 0}});

        ggml_tensor* waveform_t = result.input_tensors.at("waveform");
        ggml_backend_tensor_set(waveform_t, encoder_in.data(), 0, encoder_in.size() * sizeof(float));
        // Where the real audio stops. Without it the encoder would read its own padding as speech --
        // 1.0e-01 out on rows whose absmax is 0.128, which is what this input exists to remove.
        ggml_tensor* valid_t = result.input_tensors.at("valid_samples");
        const float valid_f = static_cast<float>(valid_samples);
        ggml_backend_tensor_set(valid_t, &valid_f, 0, sizeof(float));
        ggml_backend_graph_compute(backend.get(), result.graph);

        // HF returns (n_rows, hidden) for the REAL audio; the exported topology's ne is
        // [hidden, padded_rows] -- same memory and the same rows, so HF's are its leading prefix.
        const auto hidden = static_cast<size_t>(result.output->ne[0]);
        LOOM_CHECK(static_cast<size_t>(result.output->ne[1]) == n_chunks * frames_per_chunk);
        LOOM_CHECK(expected_audio.size() == hidden * hf_rows);
        // A byte prefix, because the rows are contiguous and HF's are the leading ones -- the padded
        // rows past `hf_rows` are read by nothing, here or in the driver.
        std::vector<float> actual(expected_audio.size());
        ggml_backend_tensor_get(result.output, actual.data(), 0, actual.size() * sizeof(float));

        float max_abs_diff = 0.0f;
        double sum_abs_diff = 0.0;
        float ref_absmax = 0.0f;
        for (size_t i = 0; i < actual.size(); ++i) {
            const float d = std::fabs(actual[i] - expected_audio[i]);
            max_abs_diff = std::max(max_abs_diff, d);
            sum_abs_diff += d;
            ref_absmax = std::max(ref_absmax, std::fabs(expected_audio[i]));
        }
        const double mean_abs_diff = sum_abs_diff / static_cast<double>(actual.size());
        std::fprintf(stderr,
                     "encoder+projector vs HF: mean_abs_diff=%g max_abs_diff=%g ref_absmax=%g "
                     "(max/absmax=%g)\n",
                     mean_abs_diff, static_cast<double>(max_abs_diff),
                     static_cast<double>(ref_absmax),
                     static_cast<double>(max_abs_diff / ref_absmax));
        // Relative to the reference's own scale, as the Whisper gate is and for the same reason: a
        // fixed epsilon would be a statement about this checkpoint's dynamic range. The rewrite is
        // bit-identical to HF's eager path in torch -- on a PARTIALLY FILLED chunk too, since P4.3e
        // (see qwen3_asr_export.WindowedAudioEncoder) -- so everything left here is f32 accumulation
        // order across 18 layers.
        LOOM_CHECK(max_abs_diff < 2e-3f * ref_absmax);
        LOOM_CHECK(mean_abs_diff < 1e-3);
    }

    // --- 2. the whole driver, against HF's greedy token sequence ---
    {
        loom::GraphTopology encoder_topo = loom::GraphTopology::parse(model->topology_json("encoder"));
        loom::GraphTopology embed_topo = loom::GraphTopology::parse(model->topology_json("embed"));
        loom::GraphTopology decoder_topo = loom::GraphTopology::parse(model->topology_json("decoder"));
        loom::GraphTopology head_topo = loom::GraphTopology::parse(model->topology_json("lm_head"));
        // The decoder is the only cached phase: the encoder is one full-sequence pass, and `embed` and
        // `lm_head` have no attention at all. Asserted rather than assumed, because a decoder that
        // silently exported WITHOUT fused ATTENTION nodes is precisely the failure that produced a
        // 143x143 mask where the cache wanted 143x152 (BACKLOG.md P4.3).
        LOOM_CHECK(decoder_topo.uses_kv_cache());
        LOOM_CHECK(!encoder_topo.uses_kv_cache());
        LOOM_CHECK(!embed_topo.uses_kv_cache());
        LOOM_CHECK(!head_topo.uses_kv_cache());
        std::unique_ptr<loom::KvCache> kv_cache = loom::make_kv_cache(*model, backend.get());

        loom::LoomLuaBridge bridge(backend.get());
        bridge.register_module("encoder", *model, std::move(encoder_topo), /*kv_cache=*/nullptr);
        bridge.register_module("embed", *model, std::move(embed_topo), /*kv_cache=*/nullptr);
        bridge.register_module("decoder", *model, std::move(decoder_topo), kv_cache.get());
        bridge.register_module("lm_head", *model, std::move(head_topo), /*kv_cache=*/nullptr);
        bridge.load_script(model->kv_str("model.driver_script"));

        const std::vector<double> waveform_d(waveform.begin(), waveform.end());
        // **The test passes a waveform, how much of it is real, and a length cap.** No prompt, no eos:
        // the driver carries the checkpoint's own chat-template prefix/suffix and its own two stop
        // tokens, which is the whole point of `prompt_segment_constants`. `waveform` is the HOST's
        // array -- zero-padded, no mirror -- so this exercises the driver's own edge repair rather
        // than check 1's precomputed copy of it.
        std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {
            {"waveform", waveform_d},
            {"audio_samples", static_cast<double>(valid_samples)},
            {"max_new_tokens", static_cast<double>(expected.size())},
        };
        // Held in a named local before unpacking: `call` returns by value, so binding a reference
        // straight into `std::get<...>(call(...))` reads the vector after the returned variant has
        // been destroyed -- the bug that cost a full bisect in P4.1, because the first few elements
        // come back as freed-heap garbage while the rest look correct.
        const loom::LoomLuaBridge::Value result = bridge.call("infer", args);
        const auto& out = std::get<std::vector<double>>(result);
        std::vector<int32_t> generated;
        generated.reserve(out.size());
        for (double v : out) generated.push_back(static_cast<int32_t>(v));

        std::fprintf(stderr, "HF prompt was %d positions; HF generated %zu tokens: ",
                     prompt_len.empty() ? -1 : prompt_len[0], expected.size());
        for (int32_t t : expected) std::fprintf(stderr, "%d ", t);
        std::fprintf(stderr, "\nMIL driver generated %zu tokens: ", generated.size());
        for (int32_t t : generated) std::fprintf(stderr, "%d ", t);
        std::fprintf(stderr, "\n");

        LOOM_CHECK(generated.size() == expected.size());
        for (size_t i = 0; i < generated.size(); ++i) {
            LOOM_CHECK(generated[i] == expected[i]);
        }
    }

    // --- 3. `audio_samples` is optional, and it is not ignored ---
    //
    // Two properties, and neither is checked by check 2 above. The first is that omitting it means
    // "all of it" -- the driver's `inputs.audio_samples or #inputs.waveform`, which is what a host
    // whose audio already filled its last chunk relies on. The second is that a different value
    // produces a different answer: a driver that accepted the argument and dropped it would pass
    // check 2 as long as the fixture's own length happened to be the default.
    {
        loom::GraphTopology encoder_topo = loom::GraphTopology::parse(model->topology_json("encoder"));
        loom::GraphTopology embed_topo = loom::GraphTopology::parse(model->topology_json("embed"));
        loom::GraphTopology decoder_topo = loom::GraphTopology::parse(model->topology_json("decoder"));
        loom::GraphTopology head_topo = loom::GraphTopology::parse(model->topology_json("lm_head"));
        std::unique_ptr<loom::KvCache> kv_cache = loom::make_kv_cache(*model, backend.get());

        loom::LoomLuaBridge bridge(backend.get());
        bridge.register_module("encoder", *model, std::move(encoder_topo), /*kv_cache=*/nullptr);
        bridge.register_module("embed", *model, std::move(embed_topo), /*kv_cache=*/nullptr);
        bridge.register_module("decoder", *model, std::move(decoder_topo), kv_cache.get());
        bridge.register_module("lm_head", *model, std::move(head_topo), /*kv_cache=*/nullptr);
        bridge.load_script(model->kv_str("model.driver_script"));

        const std::vector<double> waveform_d(waveform.begin(), waveform.end());
        auto run_with = [&](double declared) {
            std::unordered_map<std::string, loom::LoomLuaBridge::Value> args = {
                {"waveform", waveform_d},
                {"max_new_tokens", static_cast<double>(expected.size())},
            };
            if (declared > 0.0) args["audio_samples"] = declared;
            const loom::LoomLuaBridge::Value result = bridge.call("infer", args);
            const auto& out = std::get<std::vector<double>>(result);
            std::vector<int32_t> ids;
            ids.reserve(out.size());
            for (double v : out) ids.push_back(static_cast<int32_t>(v));
            return ids;
        };

        const std::vector<int32_t> omitted = run_with(-1.0);
        const std::vector<int32_t> whole = run_with(static_cast<double>(waveform.size()));
        std::fprintf(stderr, "audio_samples omitted: %zu tokens; = %zu (the whole waveform): %zu\n",
                     omitted.size(), waveform.size(), whole.size());
        LOOM_CHECK(omitted.size() == whole.size());
        for (size_t i = 0; i < omitted.size(); ++i) LOOM_CHECK(omitted[i] == whole[i]);

        // Two seconds of an eleven-second clip. The transcript cannot still be the whole utterance,
        // and a driver that ignored `audio_samples` would return check 2's sequence here.
        const std::vector<int32_t> clipped = run_with(2.0 * samples_per_chunk);
        std::fprintf(stderr, "audio_samples = %u (2 chunks of the %zu): %zu tokens: ",
                     2 * samples_per_chunk, n_chunks, clipped.size());
        for (int32_t t : clipped) std::fprintf(stderr, "%d ", t);
        std::fprintf(stderr, "\n");
        bool differs = clipped.size() != expected.size();
        for (size_t i = 0; i < clipped.size() && i < expected.size() && !differs; ++i) {
            differs = clipped[i] != expected[i];
        }
        LOOM_CHECK(differs);
    }

    std::fprintf(stderr, "test_e2e_qwen3_asr_mil_export: OK\n");
    return 0;
}
