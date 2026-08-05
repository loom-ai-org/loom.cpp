// Proves sequence length is genuinely dynamic (SPECIFICATION.md §4): loads the SAME converted
// conformer_ctc.gguf test_e2e_conformer_ctc.cpp uses (NOT a separately-regenerated file) and builds
// the graph at a DIFFERENT waveform length (1 second / 16000 samples, vs. the other test's 0.64s /
// 10240 samples), comparing against a second independent PyTorch reference generated at that length.
// Unlike test_e2e_conformer_ctc.cpp, this test does NOT hardcode n_subsampled/n_pos as compile-time
// constants -- it reads them back from the tensors GraphBuilder actually allocated for this call,
// since that's exactly the property being verified: one converted GGUF, multiple lengths, each
// producing correctly-shaped inputs from a "$n_tokens" symbol expression rather than a baked-in
// literal.

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
    // Reuses the same directory/env-var convention as test_e2e_conformer_ctc.cpp, but a second `ref_1s/`
    // subdirectory (a different waveform length), e.g. produced via:
    //   python3 tools/convert_nemo/reference_forward_conformer.py stt_en_conformer_ctc_small.nemo \
    //       $DIR/ref_1s --n-samples 16000
    const char* dir_env = std::getenv("LOOM_CONFORMER_CTC_DIR");
    const std::string dir = dir_env != nullptr ? dir_env : "/tmp/nemo_model";

    const std::string gguf_path = dir + "/conformer_ctc.gguf";
    const std::string ref_dir = dir + "/ref_1s";
    if (!path_exists(gguf_path) || !path_exists(ref_dir)) {
        std::fprintf(stderr,
                      "skipping: real Conformer-CTC dynamic-length fixture not found at '%s/ref_1s' (see "
                      "tools/convert_nemo/ to produce one)\n",
                      dir.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    auto model = loom::GgufModel::load(gguf_path, backend.get());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    constexpr uint32_t kNSamples = 16000; // 1.0s @ 16kHz -- deliberately NOT the conversion-time default
    constexpr uint32_t kNEmbd = 176;
    constexpr uint32_t kNumClasses = 1025;

    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& result = builder.build({{"n_tokens", kNSamples}, {"n_past", /*n_past=*/0}});

    ggml_tensor* waveform_t = result.input_tensors.at("waveform");
    ggml_tensor* pos_emb_raw = result.input_tensors.at("pos_emb_raw");
    ggml_tensor* kq_mask = result.input_tensors.at("kq_mask");

    // n_subsampled/n_pos are read from the tensors themselves, not hardcoded -- this length was never
    // baked into the GGUF at conversion time, only derived just now via the $n_tokens shape expression.
    const int64_t n_subsampled = kq_mask->ne[0];
    const int64_t n_pos = pos_emb_raw->ne[1];
    LOOM_CHECK(kq_mask->ne[1] == n_subsampled);
    LOOM_CHECK(pos_emb_raw->ne[0] == kNEmbd);
    LOOM_CHECK(n_subsampled > 0);
    LOOM_CHECK(n_pos == 2 * n_subsampled - 1);

    const std::vector<float> waveform = read_f32_binary(ref_dir + "/waveform.bin");
    const std::vector<float> pos_emb = read_f32_binary(ref_dir + "/pos_emb_raw.bin");
    LOOM_CHECK(waveform.size() == kNSamples);
    LOOM_CHECK(static_cast<int64_t>(pos_emb.size()) == kNEmbd * n_pos);

    ggml_backend_tensor_set(waveform_t, waveform.data(), 0, waveform.size() * sizeof(float));
    ggml_backend_tensor_set(pos_emb_raw, pos_emb.data(), 0, pos_emb.size() * sizeof(float));

    const std::vector<float> zero_mask(static_cast<size_t>(n_subsampled) * static_cast<size_t>(n_subsampled), 0.0f);
    ggml_backend_tensor_set(kq_mask, zero_mask.data(), 0, zero_mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), result.graph);

    LOOM_CHECK(result.output->ne[0] == kNumClasses);
    LOOM_CHECK(result.output->ne[1] == n_subsampled);

    std::vector<float> actual(static_cast<size_t>(kNumClasses) * static_cast<size_t>(n_subsampled));
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
