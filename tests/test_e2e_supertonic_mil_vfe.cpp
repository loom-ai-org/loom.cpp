// Numerical-correctness check for the MIL-traced SupertonicTTS "vfe" topology (export_supertonic_mil.py,
// part of supertonic_mil.gguf) against a fresh real-module reference fixture at T=10
// (reference_forward_supertonic_mil_extra.py -- the EXISTING reference_forward_supertonic_vfe.py fixture
// uses T=6, which doesn't match this topology's own fixed T_TEXT_FIXED=10; L=9 unchanged, dynamic here).
// Skips cleanly if the GGUF/reference files aren't present.

#include "test_util.h"

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

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_SUPERTONIC_MIL_GGUF");
    const char* ref_dir_env = std::getenv("LOOM_SUPERTONIC_MIL_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_MIL_GGUF (supertonic_mil.gguf, produced by "
                              "export_supertonic_mil.py) and LOOM_SUPERTONIC_MIL_REF_DIR (vfe_mil_*.bin, "
                              "produced by reference_forward_supertonic_mil_extra.py) to run this check\n");
        return 77;
    }
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/vfe_mil_z_t.bin");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/vfe_mil_z_t.bin not found\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    constexpr uint32_t kL = 9;
    constexpr uint32_t kLatDim = 144;
    constexpr uint32_t kTextLen = 10;
    constexpr uint32_t kTxtDim = 256;

    const std::vector<float> z_t = read_f32_binary(ref_dir + "/vfe_mil_z_t.bin");
    const std::vector<float> txt_emb = read_f32_binary(ref_dir + "/vfe_mil_txt_emb.bin");
    const std::vector<float> stl_emb = read_f32_binary(ref_dir + "/vfe_mil_stl_emb.bin");
    const std::vector<float> expected_v = read_f32_binary(ref_dir + "/vfe_mil_expected_v.bin");
    LOOM_CHECK(z_t.size() == kL * kLatDim);
    LOOM_CHECK(txt_emb.size() == kTextLen * kTxtDim);
    LOOM_CHECK(stl_emb.size() == 50 * 256);
    LOOM_CHECK(expected_v.size() == kL * kLatDim);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("vfe"));
    loom::GraphBuilder builder(topo, *model, backend.get());
    loom::GraphBuilder::BuildResult r = builder.build(kL, 0);

    std::vector<float> z_t_copy = z_t, txt_emb_copy = txt_emb, stl_emb_copy = stl_emb;
    std::vector<float> t_copy = {0.3f};
    ggml_backend_tensor_set(r.input_tensors.at("z_t"), z_t_copy.data(), 0, z_t_copy.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("txt_emb"), txt_emb_copy.data(), 0, txt_emb_copy.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl_emb_copy.data(), 0, stl_emb_copy.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("t"), t_copy.data(), 0, t_copy.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    std::vector<float> v(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, v.data(), 0, v.size() * sizeof(float));

    LOOM_CHECK(v.size() == expected_v.size());
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(v[i] - expected_v[i])));
    }
    std::fprintf(stderr, "v_max_abs_diff=%g (n=%zu)\n", max_abs_diff, v.size());
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
