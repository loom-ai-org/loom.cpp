// Validates the MIL-compiler-exported Whisper GGUF (tools/loom_mil_compiler/whisper_export.py,
// BACKLOG.md P4.1) against HF's own `WhisperForConditionalGeneration` -- the library, not this repo's
// bespoke converter. Two checks, in the order that localizes a failure:
//
//   1. the `encoder` topology's hidden states, against the same tensor HF's encoder produced. This half
//      contains the mel frontend, which the exported graph computes from a raw waveform and HF computes
//      in `WhisperFeatureExtractor` -- so it is the half where an inaccuracy would be arithmetic rather
//      than orchestration.
//   2. the whole embedded Lua driver -- encoder once, then the KV-cached cross-attention decode loop --
//      against HF's greedy token sequence. Integer equality, no tolerance: both are deterministic
//      argmax, which is the same gate `test_e2e_whisper_lua_driver` uses against the C++ driver.
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

    std::vector<int64_t> wav_shape, prompt_shape, gen_shape, enc_shape;
    const std::vector<float> waveform = read_npy<float>(ref_dir + "/ref_mil_waveform.npy", wav_shape);
    const std::vector<int32_t> prompt = read_npy<int32_t>(ref_dir + "/ref_mil_prompt.npy", prompt_shape);
    const std::vector<int32_t> expected = read_npy<int32_t>(ref_dir + "/ref_mil_generated.npy", gen_shape);
    const std::vector<float> expected_encoder = read_npy<float>(ref_dir + "/ref_mil_encoder.npy", enc_shape);

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

    // --- 2. the whole driver ---
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
        const std::vector<double> prompt_d(prompt.begin(), prompt.end());
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"waveform", waveform_d},
            {"tokens", prompt_d},
            {"max_new_tokens", static_cast<double>(expected.size())},
            // The artifact's own eos, from the vocab KVs the export embedded -- not a constant this
            // test carries, which is the same "read it from the file" point as `n_samples` above.
            {"eos_token", static_cast<double>(model->kv_i32("tokenizer.ggml.eos_token_id", -1))},
        });

        const auto& generated_d = std::get<std::vector<double>>(result);
        std::vector<int32_t> generated;
        generated.reserve(generated_d.size());
        for (double v : generated_d) generated.push_back(static_cast<int32_t>(v));

        std::fprintf(stderr, "HF  generated %zu tokens: ", expected.size());
        for (int32_t t : expected) std::fprintf(stderr, "%d ", t);
        std::fprintf(stderr, "\nMIL generated %zu tokens: ", generated.size());
        for (int32_t t : generated) std::fprintf(stderr, "%d ", t);
        std::fprintf(stderr, "\n");

        LOOM_CHECK(generated.size() == expected.size());
        for (size_t i = 0; i < generated.size(); ++i) {
            LOOM_CHECK(generated[i] == expected[i]);
        }
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
