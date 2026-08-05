// Gating-criterion proof of concept (BACKLOG.md): points the *same, unmodified* generic ATen-graph ->
// loom-topology converter (tools/convert_generic/aten_to_loom.py) the toy-LLM/Qwen3 POCs used at a
// genuinely non-decoder-transformer-shaped model -- the real Conformer-CTC encoder (subsampling Conv2d,
// LayerNorm, depthwise Conv1d, GLU, relative-position self-attention with no ATen equivalent). Unlike
// those POCs, this one DID need new op-mapping entries (LAYER_NORM, CONV_1D/CONV_1D_DW/CONV_2D, GLU, RELU,
// PERMUTE, a new loom::rel_pos_attention custom op) -- see BACKLOG.md for the real result.
//
// Declared inputs are "mel_input"/"pos_emb_raw"/"kq_mask" (skips the mel frontend -- see
// tools/convert_generic/conformer_ctc_module.py's module docstring). Reuses the *existing*
// test_e2e_conformer_ctc.cpp reference fixture's expected_logits.bin (same weights, same waveform seed =>
// byte-identical expected logits) for the comparison target, but reads its own mel_input.bin/
// pos_emb_raw.bin (written by make_conformer_ctc_gguf_generic.py) since this GGUF's declared inputs start
// one step later in the pipeline than the hand-written topology's "waveform" input does.
//
// Same "not generated at ctest time, skip cleanly if the real checkpoint isn't prepared" pattern as every
// other real-model test in this suite.

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
    // LOOM_CONFORMER_CTC_GENERIC_DIR must contain conformer_ctc_generic.gguf + mel_input.bin +
    // pos_emb_raw.bin, e.g. produced via:
    //   python3 tools/convert_generic/make_conformer_ctc_gguf_generic.py \
    //       stt_en_conformer_ctc_small.nemo $DIR
    const char* generic_dir_env = std::getenv("LOOM_CONFORMER_CTC_GENERIC_DIR");
    const std::string generic_dir = generic_dir_env != nullptr ? generic_dir_env : "/tmp/conformer_ctc_generic";
    const std::string gguf_path = generic_dir + "/conformer_ctc_generic.gguf";

    // Reference fixture is shared with test_e2e_conformer_ctc.cpp -- same weights/waveform seed, so
    // byte-identical expected logits regardless of which conversion pipeline built the GGUF.
    const char* ref_dir_env = std::getenv("LOOM_CONFORMER_CTC_DIR");
    const std::string ref_dir = (ref_dir_env != nullptr ? std::string(ref_dir_env) : "/tmp/nemo_model") + "/ref";

    if (!path_exists(gguf_path) || !path_exists(ref_dir) || !path_exists(generic_dir + "/mel_input.bin")) {
        std::fprintf(stderr,
                      "skipping: auto-converted Conformer-CTC fixture not found at '%s' or reference not "
                      "found at '%s' (set LOOM_CONFORMER_CTC_GENERIC_DIR / LOOM_CONFORMER_CTC_DIR or see "
                      "tools/convert_generic/ and tools/convert_nemo/ to produce them)\n",
                      gguf_path.c_str(), ref_dir.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    constexpr uint32_t kNSubsampled = 17;
    constexpr uint32_t kNPos = 2 * kNSubsampled - 1;
    constexpr uint32_t kNEmbd = 176;
    constexpr uint32_t kTMel = 65;
    constexpr uint32_t kNMels = 80;
    constexpr uint32_t kNumClasses = 1025;

    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& result = builder.build({{"n_tokens", /*n_tokens=*/0}, {"n_past", /*n_past=*/0}});

    ggml_tensor* mel_input_t = result.input_tensors.at("mel_input");
    ggml_tensor* pos_emb_raw_t = result.input_tensors.at("pos_emb_raw");
    ggml_tensor* kq_mask_t = result.input_tensors.at("kq_mask");

    const std::vector<float> mel_input = read_f32_binary(generic_dir + "/mel_input.bin");
    const std::vector<float> pos_emb = read_f32_binary(generic_dir + "/pos_emb_raw.bin");
    LOOM_CHECK(mel_input.size() == static_cast<size_t>(kTMel) * kNMels);
    LOOM_CHECK(pos_emb.size() == static_cast<size_t>(kNEmbd) * kNPos);

    ggml_backend_tensor_set(mel_input_t, mel_input.data(), 0, mel_input.size() * sizeof(float));
    ggml_backend_tensor_set(pos_emb_raw_t, pos_emb.data(), 0, pos_emb.size() * sizeof(float));

    const std::vector<float> zero_mask(static_cast<size_t>(kNSubsampled) * kNSubsampled, 0.0f);
    ggml_backend_tensor_set(kq_mask_t, zero_mask.data(), 0, zero_mask.size() * sizeof(float));

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

    LOOM_TEST_REPORT_AND_RETURN();
}
