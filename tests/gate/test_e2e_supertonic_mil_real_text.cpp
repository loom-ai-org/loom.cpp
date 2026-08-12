// What P4.6 actually delivered: SupertonicTTS synthesizing from REAL text.
//
// The other supertonic tests all run ten ids, because ten was the whole text axis before P4.6 --
// and ten ids is the EMPTY STRING after the `<lang>` wrap, so none of them says anything about text
// a person would type. This one runs a real sentence (161 ids against a 256-wide axis), which is also
// the only shape where the real/padding boundary sits in the MIDDLE of the axis rather than exactly
// at its end. That matters: what makes padding inert is `supertonic_export.py`'s `_edge_fill`, whose
// job is precisely to make each ConvNext block's depthwise conv see, past the last real position,
// what it would see if the axis stopped there.
//
// The oracle is the reference implementation, not this project's own history:
// `reference_forward_supertonic_mil_extra.py` runs the real `TextVectorizer` and then the real
// `DurationPredictor`/`TTLTextEncoder` at T = the sentence's own length, UNPADDED -- which is what
// the reference implementation genuinely does for a single utterance, since
// `TextVectorizer.tokenize` pads only to the longest string in a batch and synthesis is a batch of
// one. So this asks: does the padded export reproduce, for real text, exactly the answer the real
// Python produces for that text?
//
// Four checks, in order of what they would catch:
//   1. The ids. `loom::SupertonicTextVectorizer` reading the GGUF's own vocabulary must produce the
//      same ids the real Python `TextVectorizer` produced into the fixture. This runs FIRST because
//      every number below is meaningless if the two implementations tokenized different text -- and
//      it is a real cross-check of two independent implementations at a string neither was tuned on.
//   2. `dp`'s duration. A wrong one is audio of the wrong LENGTH, which is how a padding bug in the
//      DPTextEncoder's attention-weighted pooling would present: not as a numeric near-miss.
//   3. `ttl_text`'s embedding over the real columns, plus an exact zero over the padded ones.
//   4. The DRIVER, end to end, at this length -- because 1-3 build the graphs directly and pad by
//      hand, which is a re-implementation of `supertonic_driver/01_text_inputs.lua` rather than a
//      test of it. Every other end-to-end test hands `infer` exactly ten ids, so nothing else
//      exercises the branch where the driver has real padding to do. There is no waveform oracle
//      here (the CFM noise is the driver's own), so what is checked is the sample COUNT, which the
//      reference duration determines exactly through the driver's own `get_latent_mask` arithmetic:
//      a driver that mis-padded would predict a different duration and produce a different length.
//
// Skips cleanly if the GGUF/reference files aren't present.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Must stay byte-for-byte `REAL_TEXT` in reference_forward_supertonic_mil_extra.py. Check 1 is what
// makes a divergence fail loudly instead of quietly comparing two different sentences.
const char* kRealText =
    "Supertonic is a text to speech model, and this sentence is deliberately long enough "
    "to be a realistic test of what a single synthesis call has to carry.";

std::vector<float> read_f32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

std::vector<int32_t> read_i32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<int32_t> data(static_cast<size_t>(bytes) / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_SUPERTONIC_MIL_GGUF");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_MIL_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_MIL_GGUF (the supertonic GGUF produced by "
                              "`loom-export`) and LOOM_SUPERTONIC_MIL_REF_DIR (real_*.bin, produced by "
                              "reference_forward_supertonic_mil_extra.py) to run this check\n");
        return 77;
    }
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/real_txt_ids.bin");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/real_txt_ids.bin not found -- regenerate the fixture with "
                              "reference_forward_supertonic_mil_extra.py (it grew a third case in "
                              "BACKLOG.md P4.6)\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    constexpr uint32_t kTxtDim = 256;
    const std::vector<int32_t> expected_ids = read_i32_binary(ref_dir + "/real_txt_ids.bin");
    const std::vector<float> dp_stl = read_f32_binary(ref_dir + "/real_dp_stl_emb.bin");
    const std::vector<float> ttl_stl = read_f32_binary(ref_dir + "/real_ttl_stl_emb.bin");
    const std::vector<float> expected_duration = read_f32_binary(ref_dir + "/real_expected_duration.bin");
    const std::vector<float> expected_txt_emb = read_f32_binary(ref_dir + "/real_expected_txt_emb.bin");
    const uint32_t n_real = static_cast<uint32_t>(expected_ids.size());
    LOOM_CHECK(dp_stl.size() == 8 * 16);
    LOOM_CHECK(ttl_stl.size() == 50 * kTxtDim);
    LOOM_CHECK(expected_duration.size() == 1);
    LOOM_CHECK(expected_txt_emb.size() == static_cast<size_t>(n_real) * kTxtDim);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);

    const uint32_t t_text = model->hparam_u32("txt_len");
    std::fprintf(stderr, "txt_len=%u, real sentence is %u ids\n", t_text, n_real);
    // Not an assertion about the model so much as about this test: a `txt_len` this sentence does not
    // fit in means the fixture cannot be fed to the export at all, which is a different failure from
    // a numeric one and deserves to say so.
    LOOM_CHECK(t_text >= n_real);
    // ...and the sentence has to actually leave padding behind, or this test is the ten-id test again.
    LOOM_CHECK(n_real < t_text);

    // --- 1. the ids, from the engine's own vectorizer reading the GGUF's own vocabulary ---
    auto vec = loom::SupertonicTextVectorizer::load(*model);
    LOOM_CHECK(vec != nullptr);
    const std::vector<int32_t> ids = vec->tokenize(kRealText, "en");
    LOOM_CHECK(ids == expected_ids);

    // The driver's padding, done by hand -- see supertonic_driver/01_text_inputs.lua.
    std::vector<int32_t> txt_ids(t_text, 162);
    std::copy(ids.begin(), ids.end(), txt_ids.begin());
    std::vector<float> txt_msk(t_text, 0.0f);
    std::fill(txt_msk.begin(), txt_msk.begin() + n_real, 1.0f);

    // --- 2. duration ---
    {
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("dp"));
        loom::GraphBuilder builder(topo, *model, backend.get());
        const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", t_text}, {"n_past", 0}});
        std::vector<float> stl = dp_stl;
        ggml_backend_tensor_set(r.input_tensors.at("txt_ids"), txt_ids.data(), 0, txt_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl.data(), 0, stl.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("txt_msk"), txt_msk.data(), 0, txt_msk.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), r.graph);

        LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == 1);
        float duration = 0.0f;
        ggml_backend_tensor_get(r.output, &duration, 0, sizeof(float));
        const double diff = std::fabs(static_cast<double>(duration) - static_cast<double>(expected_duration[0]));
        std::fprintf(stderr, "duration=%f expected=%f diff=%g\n", duration, expected_duration[0], diff);
        // Relative, not absolute: this sentence's duration is ~10.6 s where the ten-id fixtures'
        // is ~1.6 s, and what a listener would notice is a proportional error.
        LOOM_CHECK(diff / expected_duration[0] < 1e-3);
    }

    // --- 3. txt_emb over the real columns, and exact zero over the padded ones ---
    {
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("ttl_text"));
        loom::GraphBuilder builder(topo, *model, backend.get());
        const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", t_text}, {"n_past", 0}});
        std::vector<float> stl = ttl_stl;
        ggml_backend_tensor_set(r.input_tensors.at("txt_ids"), txt_ids.data(), 0, txt_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl.data(), 0, stl.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("txt_msk"), txt_msk.data(), 0, txt_msk.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), r.graph);

        std::vector<float> txt_emb(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, txt_emb.data(), 0, txt_emb.size() * sizeof(float));
        LOOM_CHECK(txt_emb.size() == static_cast<size_t>(t_text) * kTxtDim);

        // ne=[t_text, 256] against a (256, n_real) row-major reference: different channel strides.
        double max_abs_diff = 0.0;
        double max_abs_pad = 0.0;
        for (uint32_t c = 0; c < kTxtDim; ++c) {
            for (uint32_t t = 0; t < n_real; ++t) {
                const double got = txt_emb[static_cast<size_t>(c) * t_text + t];
                const double want = expected_txt_emb[static_cast<size_t>(c) * n_real + t];
                max_abs_diff = std::max(max_abs_diff, std::fabs(got - want));
            }
            for (uint32_t t = n_real; t < t_text; ++t) {
                max_abs_pad = std::max(max_abs_pad, std::fabs(static_cast<double>(txt_emb[static_cast<size_t>(c) * t_text + t])));
            }
        }
        std::fprintf(stderr, "txt_emb_max_abs_diff=%g over %u real columns, pad_tail_max_abs=%g\n",
                     max_abs_diff, n_real, max_abs_pad);
        LOOM_CHECK(max_abs_diff < 1e-2);
        LOOM_CHECK(max_abs_pad == 0.0);
    }

    // --- 4. the driver, at this length, through its own padding ---
    {
        const std::string driver_script = model->kv_str("model.driver_script");
        LOOM_CHECK(!driver_script.empty());

        loom::LoomLuaBridge bridge(backend.get());
        for (const char* name : {"dp", "ttl_text", "vfe", "decoder"}) {
            bridge.register_module(name, *model, loom::GraphTopology::parse(model->topology_json(name)));
        }
        bridge.load_script(driver_script);

        // The ids as the tokenizer produced them -- NOT padded. Padding is what the driver is being
        // tested on, so doing it here would test nothing.
        const std::vector<double> ids_d(ids.begin(), ids.end());
        const std::vector<double> ttl_d(ttl_stl.begin(), ttl_stl.end());
        const std::vector<double> dp_d(dp_stl.begin(), dp_stl.end());
        constexpr uint32_t kNSteps = 2;  // enough to run the loop; this check is about length, not audio
        loom::LoomLuaBridge::Value result = bridge.call("infer", {
            {"txt_ids", ids_d},
            {"style_ttl", ttl_d},
            {"style_dp", dp_d},
            {"n_steps", static_cast<double>(kNSteps)},
            {"seed", 42.0},
        });
        const auto& wav = std::get<std::vector<double>>(result);

        // supertonic_driver/02_latent_length.lua's own arithmetic, from the REFERENCE duration --
        // which is the point: the driver's `dp` call has to have predicted the same one.
        constexpr uint32_t kSampleRate = 44100;
        constexpr uint32_t kBaseChunk = 512;
        constexpr uint32_t kCompression = 6;
        const uint32_t latent_size = kBaseChunk * kCompression;
        const uint32_t wav_length = static_cast<uint32_t>(expected_duration[0] * kSampleRate);
        const uint32_t t_lat = (wav_length + latent_size - 1) / latent_size;
        const size_t expected_samples = static_cast<size_t>(t_lat) * latent_size;
        std::fprintf(stderr, "driver: %zu ids -> %zu samples (%.2f s), expected %zu\n",
                     ids.size(), wav.size(), static_cast<double>(wav.size()) / kSampleRate,
                     expected_samples);
        LOOM_CHECK(wav.size() == expected_samples);
        // ...and it has to be audio, not silence: a driver that fed the graphs all-zero ids would
        // still get the length right if the duration happened to survive.
        double peak = 0.0;
        for (double s : wav) peak = std::max(peak, std::fabs(s));
        std::fprintf(stderr, "driver: waveform peak=%g\n", peak);
        LOOM_CHECK(peak > 1e-3);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
