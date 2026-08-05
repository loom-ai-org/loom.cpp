// Real-model end-to-end test: loads the actual NVIDIA stt_en_conformer_ctc_small checkpoint (converted
// by tools/convert_nemo/convert_conformer_ctc.py into a GGUF file) and runs one non-autoregressive
// forward pass -- from a RAW WAVEFORM, through the in-graph mel-spectrogram frontend (preemphasis,
// STFT-via-CONV_1D, power, mel filterbank, log, per-feature CMVN normalize), the Conformer encoder, and
// the CTC decoder -- through GraphBuilder directly, comparing the final logits against
// tools/convert_nemo/reference_forward_conformer.py's independent plain-PyTorch computation of the
// identical weights/topology/features (that reference calls real torch.stft, not the graph's own
// conv-based DFT trick, so this is a genuine end-to-end check of that trick's correctness too).
//
// Unlike every other e2e test in this suite, the fixture here is NOT procedurally generated at ctest
// time: it requires the real ~49MB .nemo checkpoint (downloaded once from HuggingFace) and a PyTorch
// environment, neither of which belong in a fresh checkout or a plain offline CI run. Instead this test
// looks for an already-prepared directory (see tools/convert_nemo/README or BACKLOG.md for how to
// produce one) containing `conformer_ctc.gguf` and a `ref/` subdirectory with the reference .bin dumps,
// and skips cleanly (exit code 77, wired to ctest's SKIP_RETURN_CODE) if that directory isn't present --
// this keeps `ctest` fully green on a machine that hasn't set up the real model, without silently
// pretending the check ran.

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
    // LOOM_CONFORMER_CTC_DIR must contain conformer_ctc.gguf + ref/{waveform,pos_emb_raw,
    // expected_logits}.bin, e.g. produced via:
    //   python3 tools/convert_nemo/convert_conformer_ctc.py stt_en_conformer_ctc_small.nemo \
    //       $DIR/conformer_ctc.gguf
    //   python3 tools/convert_nemo/reference_forward_conformer.py stt_en_conformer_ctc_small.nemo \
    //       $DIR/ref
    const char* dir_env = std::getenv("LOOM_CONFORMER_CTC_DIR");
    const std::string dir = dir_env != nullptr ? dir_env : "/tmp/nemo_model";

    const std::string gguf_path = dir + "/conformer_ctc.gguf";
    const std::string ref_dir = dir + "/ref";
    if (!path_exists(gguf_path) || !path_exists(ref_dir)) {
        std::fprintf(stderr,
                      "skipping: real Conformer-CTC fixture not found at '%s' (set LOOM_CONFORMER_CTC_DIR "
                      "or see tools/convert_nemo/ to produce one)\n",
                      dir.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    // Must match convert_conformer_ctc.py's/reference_forward_conformer.py's defaults exactly:
    // pos_emb_raw/kq_mask shapes are baked into the topology as literals derived from these, not
    // symbolically tied to whatever n_tokens build() is called with.
    constexpr uint32_t kNSamples = 10240;  // 0.64s @ 16kHz raw waveform
    constexpr uint32_t kNSubsampled = 17;  // t_mel=65 -> (((65+2-3)/2+1)+2-3)/2+1
    constexpr uint32_t kNPos = 2 * kNSubsampled - 1;
    constexpr uint32_t kNEmbd = 176;
    constexpr uint32_t kNumClasses = 1025;

    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& result = builder.build({{"n_tokens", kNSamples}, {"n_past", /*n_past=*/0}});

    ggml_tensor* waveform_t = result.input_tensors.at("waveform");
    ggml_tensor* pos_emb_raw = result.input_tensors.at("pos_emb_raw");
    ggml_tensor* kq_mask = result.input_tensors.at("kq_mask");

    const std::vector<float> waveform = read_f32_binary(ref_dir + "/waveform.bin");
    const std::vector<float> pos_emb = read_f32_binary(ref_dir + "/pos_emb_raw.bin");
    LOOM_CHECK(waveform.size() == kNSamples);
    LOOM_CHECK(pos_emb.size() == static_cast<size_t>(kNEmbd) * kNPos);

    ggml_backend_tensor_set(waveform_t, waveform.data(), 0, waveform.size() * sizeof(float));
    ggml_backend_tensor_set(pos_emb_raw, pos_emb.data(), 0, pos_emb.size() * sizeof(float));

    const std::vector<float> zero_mask(static_cast<size_t>(kNSubsampled) * kNSubsampled, 0.0f);
    ggml_backend_tensor_set(kq_mask, zero_mask.data(), 0, zero_mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), result.graph);

    LOOM_CHECK(result.output->ne[0] == kNumClasses);
    LOOM_CHECK(result.output->ne[1] == kNSubsampled);

    std::vector<float> actual(static_cast<size_t>(kNumClasses) * kNSubsampled);
    ggml_backend_tensor_get(result.output, actual.data(), 0, actual.size() * sizeof(float));

    const std::vector<float> expected = read_f32_binary(ref_dir + "/expected_logits.bin");
    LOOM_CHECK(expected.size() == actual.size());

    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < expected.size() && i < actual.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, std::fabs(expected[i] - actual[i]));
    }
    LOOM_CHECK(max_abs_diff <= 1e-3f);
    if (max_abs_diff > 1e-3f) {
        std::fprintf(stderr, "max abs diff = %f\n", static_cast<double>(max_abs_diff));
    }

    // Plumbing smoke check: CTC-decode + detokenize the real logits above end to end. This is NOT a
    // word-accuracy check (the input waveform is synthetic noise, not real speech) -- it just confirms
    // ctc_greedy_decode + Vocab::decode run against the real model's real vocab without crashing and
    // produce well-formed text.
    const auto token_ids = loom::ctc_greedy_decode(actual.data(), kNSubsampled, kNumClasses,
                                                    /*blank_id=*/static_cast<int32_t>(kNumClasses) - 1);
    auto vocab = loom::Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const std::string text = vocab->decode(token_ids);
    std::fprintf(stderr, "ctc decode (%zu tokens) -> '%s'\n", token_ids.size(), text.c_str());
    for (int32_t id : token_ids) {
        LOOM_CHECK(id >= 0 && static_cast<size_t>(id) < vocab->size());
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
