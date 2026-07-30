// Numerical-correctness check for SupertonicTTS v2's full DurationPredictor sub-model (DPTextEncoder:
// ConvNeXt stack + Shaw-et-al. relative-position attention (reusing VITS's own primitive) + sentence-
// token pooling; MLP head w/ PReLU) against the real `duration_predictor.pt` module, driven by a
// PRECOMPUTED style embedding (matching how `SpeechGenerator.predict()` itself calls
// `dur_predictor.predict(txt_ids, stl_emb=..., txt_msk)` with an already-computed style, real
// `dp-style-encoder.pt` output dumped by the reference script). This is the FIRST full coherent
// sub-model verified in this project's SupertonicTTS effort. Skips cleanly if the GGUF/reference files
// aren't present.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

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
    const char* dir_env = std::getenv("LOOM_SUPERTONIC_DIR");
    const char* ref_dir_env = std::getenv("LOOM_SUPERTONIC_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_DIR (supertonic_dp.gguf) and "
                              "LOOM_SUPERTONIC_REF_DIR (dp_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kT = 12;
    constexpr uint32_t kStlDim = 128;

    // Reference dumps via `txt_ids.numpy().astype(np.int32)` -- already i32 on disk, read directly.
    const std::vector<int32_t> txt_ids = read_i32_binary(ref_dir + "/dp_txt_ids.bin");
    const std::vector<float> stl_emb = read_f32_binary(ref_dir + "/dp_stl_emb.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/dp_expected_duration.bin");
    LOOM_CHECK(txt_ids.size() == kT);
    LOOM_CHECK(stl_emb.size() == kStlDim);
    LOOM_CHECK(expected.size() == 1);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/supertonic_dp.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", kT}, {"n_past", 0}});

    ggml_backend_tensor_set(r.input_tensors.at("txt_ids"), txt_ids.data(), 0, txt_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl_emb.data(), 0, stl_emb.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == 1);
    float duration = 0.0f;
    ggml_backend_tensor_get(r.output, &duration, 0, sizeof(float));

    const double diff = std::fabs(static_cast<double>(duration) - static_cast<double>(expected[0]));
    std::fprintf(stderr, "duration=%f expected=%f diff=%g\n", duration, expected[0], diff);
    LOOM_CHECK(diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
