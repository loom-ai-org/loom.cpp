// Validates the MIL-compiler-exported NeMo Conformer-CTC-small GGUF (export_conformer_ctc_mil.py)
// against the SAME reference_forward_conformer.py fixture the bespoke-conversion test
// (test_e2e_conformer_ctc.cpp) uses. Unlike that test, this GGUF's topology traces the REAL
// EncDecCTCModelBPE (preprocessor + ConformerEncoder + ConvASRDecoder) directly via torch.jit.trace +
// coremltools. Loaded directly via GraphBuilder (bypassing the generic MIL-exporter's auto-generated Lua
// driver script, which assumes a causal-LM "argmax the last row" convention -- wrong for a CTC model's
// full (num_classes, n_subsampled) logits tensor).
//
// Not generated at ctest time (needs the real ~49MB checkpoint + coremltools) -- skips cleanly if the
// fixture isn't present. To (re)generate: `~/.venvs/piper/bin/python3 export_conformer_ctc_mil.py` from
// the repo root, and reuse the SAME ref/ dir test_e2e_conformer_ctc.cpp's own LOOM_CONFORMER_CTC_DIR
// produces (tools/convert_nemo/reference_forward_conformer.py).

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>

namespace {

constexpr int kSkipReturnCode = 77;

bool path_exists(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

std::vector<float> read_f32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_CONFORMER_CTC_MIL_GGUF");
    const std::string gguf_path = gguf_env != nullptr ? gguf_env : "conformer_ctc_small_mil_monolithic.gguf";

    const char* dir_env = std::getenv("LOOM_CONFORMER_CTC_DIR");
    const std::string dir = dir_env != nullptr ? dir_env : "/tmp/nemo_model";
    const std::string ref_dir = dir + "/ref";

    if (!path_exists(gguf_path) || !path_exists(ref_dir)) {
        std::fprintf(stderr,
                      "skipping: MIL-exported Conformer-CTC GGUF ('%s') or ref fixture ('%s') not found "
                      "(run export_conformer_ctc_mil.py and tools/convert_nemo/reference_forward_conformer.py)\n",
                      gguf_path.c_str(), ref_dir.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("main_topo"));

    // Must match reference_forward_conformer.py's own default trace length (see test_e2e_conformer_ctc.cpp).
    constexpr uint32_t kNSamples = 10240;
    constexpr uint32_t kNSubsampled = 17;
    constexpr uint32_t kNumClasses = 1025;

    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
    loom::GraphBuilder::BuildResult result = builder.build({{"n_samples", kNSamples}, {"n_past", 0}});

    ggml_tensor* waveform_t = result.input_tensors.at("waveform");
    ggml_tensor* length_t = result.input_tensors.at("length");

    const std::vector<float> waveform = read_f32_binary(ref_dir + "/waveform.bin");
    LOOM_CHECK(waveform.size() == kNSamples);

    ggml_backend_tensor_set(waveform_t, waveform.data(), 0, waveform.size() * sizeof(float));
    const int32_t length_val = static_cast<int32_t>(kNSamples);
    ggml_backend_tensor_set(length_t, &length_val, 0, sizeof(int32_t));

    ggml_backend_graph_compute(backend.get(), result.graph);

    std::fprintf(stderr, "MIL-exported logits shape: [%ld, %ld]\n",
                 static_cast<long>(result.output->ne[0]), static_cast<long>(result.output->ne[1]));
    LOOM_CHECK(static_cast<uint32_t>(result.output->ne[0]) == kNumClasses);
    LOOM_CHECK(static_cast<uint32_t>(result.output->ne[1]) == kNSubsampled);

    std::vector<float> actual(static_cast<size_t>(kNumClasses) * kNSubsampled);
    ggml_backend_tensor_get(result.output, actual.data(), 0, actual.size() * sizeof(float));

    // Unlike convert_conformer_ctc.py's hand-derived topology (whose declared "output" is the CTC
    // decoder's raw linear output, matching expected_logits.bin's own raw `F.linear(...)` directly), the
    // MIL trace here wraps the REAL EncDecCTCModelBPE.forward(), which returns nemo's own
    // ConvASRDecoder.forward() output -- `log_softmax(linear(...))`, not raw logits (see
    // nemo/collections/asr/modules/conv_asr.py's own ConvASRDecoder.forward). Apply the same log_softmax
    // to the reference's raw logits before comparing, rather than comparing two different quantities.
    std::vector<float> expected = read_f32_binary(ref_dir + "/expected_logits.bin");
    LOOM_CHECK(expected.size() == actual.size());
    for (uint32_t t = 0; t < kNSubsampled; ++t) {
        float* row = expected.data() + static_cast<size_t>(t) * kNumClasses;
        float max_logit = *std::max_element(row, row + kNumClasses);
        double sum_exp = 0.0;
        for (uint32_t c = 0; c < kNumClasses; ++c) sum_exp += std::exp(static_cast<double>(row[c] - max_logit));
        const float log_sum_exp = max_logit + static_cast<float>(std::log(sum_exp));
        for (uint32_t c = 0; c < kNumClasses; ++c) row[c] -= log_sum_exp;
    }

    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(expected[i] - actual[i]));
    }
    std::fprintf(stderr, "MIL-exported logits max abs diff vs. reference_forward_conformer.py = %f\n",
                 static_cast<double>(max_abs_diff));
    // Same tolerance as test_e2e_conformer_ctc.cpp's own bespoke-conversion check: with
    // compute_precision=ct.precision.FLOAT32 (export_conformer_ctc_mil.py) and the exporter's "conv" bias
    // fix in place (see BACKLOG.md), this path matches the reference to within ordinary fp32 rounding,
    // same as the bespoke conversion -- the earlier, much looser tolerance here was masking two real bugs
    // (a silently fp16-rounded constant weight, and a completely dropped conv bias), not a genuine,
    // unavoidable precision ceiling.
    LOOM_CHECK(max_abs_diff <= 1e-3f);

    // Plumbing smoke check: CTC-decode the real logits above end to end. Unlike the bespoke-conversion
    // path (convert_conformer_ctc.py, via tokenizer_common.write_sentencepiece_vocab), the generic MIL
    // exporter doesn't embed a tokenizer.ggml.* vocab at all yet -- a separate, known scope gap, not
    // something this fix touches -- so detokenization is skipped if none is present.
    const auto token_ids = loom::ctc_greedy_decode(actual.data(), kNSubsampled, kNumClasses,
                                                    /*blank_id=*/static_cast<int32_t>(kNumClasses) - 1);
    auto vocab = loom::Vocab::load(*model);
    if (vocab != nullptr) {
        const std::string text = vocab->decode(token_ids);
        std::fprintf(stderr, "ctc decode (%zu tokens) -> '%s'\n", token_ids.size(), text.c_str());
        for (int32_t id : token_ids) {
            LOOM_CHECK(id >= 0 && static_cast<size_t>(id) < vocab->size());
        }
    } else {
        std::fprintf(stderr, "no tokenizer embedded in this GGUF (expected -- MIL exporter doesn't emit "
                              "one yet), skipping detokenization\n");
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
