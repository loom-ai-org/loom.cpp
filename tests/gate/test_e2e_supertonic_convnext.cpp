// Numerical-correctness check for SupertonicTTS v2's `ConvNextBlock` (real source:
// supertonic_tts/models/modules/components.py, converted via
// tools/convert_supertonic/convert_supertonic_convnext.py's `add_convnext_block`) against the REAL
// `nn.Module`'s own forward pass (the `supertonic-tts` package is importable in this environment, so the
// reference calls the real module directly rather than a hand-copied formula -- see
// reference_forward_supertonic_convnext.py's own docstring). Two causal instances checked (dilation=1
// and dilation=2, both from the real `vocoder.pt`'s own `SpeechDecoder.convnext` stack) -- exercises the
// replicate-pad composition (VIEW+REPEAT+CONCAT, no native ggml op) at two different pad widths. Skips
// cleanly if the GGUF/reference files aren't present.

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

std::vector<float> read_f32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

bool run_one(const std::string& dir, const std::string& ref_dir, const std::string& name, uint32_t T,
             uint32_t dim, ggml_backend_t backend) {
    const std::vector<float> x = read_f32_binary(ref_dir + "/convnext_" + name + "_x.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/convnext_" + name + "_y.bin");
    LOOM_CHECK(x.size() == static_cast<size_t>(T) * dim);
    LOOM_CHECK(expected.size() == x.size());

    auto model = loom::GgufModel::load(dir + "/supertonic_convnext_" + name + ".gguf", backend);
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend, nullptr);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", T}, {"n_past", 0}});
    ggml_backend_tensor_set(r.input_tensors.at("x"), x.data(), 0, x.size() * sizeof(float));
    ggml_backend_graph_compute(backend, r.graph);

    LOOM_CHECK(static_cast<size_t>(ggml_nelements(r.output)) == x.size());
    std::vector<float> out(x.size());
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(static_cast<double>(out[i]) - static_cast<double>(expected[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / out.size();
    std::fprintf(stderr, "%s: mean_diff=%g, max_diff=%g\n", name.c_str(), mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-4);
    LOOM_CHECK(max_diff < 1e-2);
    return true;
}

} // namespace

int main() {
    const char* dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_DIR");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_DIR (supertonic_convnext_*.gguf) and "
                              "LOOM_SUPERTONIC_REF_DIR (convnext_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    run_one(dir, ref_dir, "block0_d1_causal", /*T=*/12, /*dim=*/512, backend.get());
    run_one(dir, ref_dir, "block1_d2_causal", /*T=*/12, /*dim=*/512, backend.get());

    LOOM_TEST_REPORT_AND_RETURN();
}
