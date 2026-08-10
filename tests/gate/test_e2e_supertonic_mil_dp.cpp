// Numerical-correctness check for the MIL-traced SupertonicTTS "dp" topology (export_supertonic_mil.py,
// part of supertonic_mil.gguf) against a fresh real-module reference fixture at T=10
// (reference_forward_supertonic_mil_extra.py -- the EXISTING reference_forward_supertonic_dp.py fixture
// uses T=12, which doesn't apply here since this "dp" topology is traced dynamically over T; T=10 is just
// what that fixture happened to use, chosen to match the rest of the MIL export's own fixed T_TEXT_FIXED
// for consistency, not because "dp" itself requires it -- see export_supertonic_mil.py's own docstring).
// Skips cleanly if the GGUF/reference files aren't present.

#include "test_util.h"
#include "fixtures.h"

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
    const char* gguf_env = loom_test::fixture_env("LOOM_SUPERTONIC_MIL_GGUF");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_MIL_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_MIL_GGUF (supertonic_mil.gguf, produced by "
                              "export_supertonic_mil.py) and LOOM_SUPERTONIC_MIL_REF_DIR (dp_mil_*.bin, "
                              "produced by reference_forward_supertonic_mil_extra.py) to run this check\n");
        return 77;
    }
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/dp_mil_txt_ids.bin");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/dp_mil_txt_ids.bin not found\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    const std::vector<int32_t> txt_ids = read_i32_binary(ref_dir + "/dp_mil_txt_ids.bin");
    const std::vector<float> stl_emb = read_f32_binary(ref_dir + "/dp_mil_stl_emb.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/dp_mil_expected_duration.bin");
    LOOM_CHECK(txt_ids.size() == 10);
    LOOM_CHECK(stl_emb.size() == 8 * 16);
    LOOM_CHECK(expected.size() == 1);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("dp"));
    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", static_cast<uint32_t>(txt_ids.size())}, {"n_past", 0}});

    std::vector<int32_t> txt_ids_copy = txt_ids;
    std::vector<float> stl_emb_copy = stl_emb;
    ggml_backend_tensor_set(r.input_tensors.at("txt_ids"), txt_ids_copy.data(), 0, txt_ids_copy.size() * sizeof(int32_t));
    ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl_emb_copy.data(), 0, stl_emb_copy.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == 1);
    float duration = 0.0f;
    ggml_backend_tensor_get(r.output, &duration, 0, sizeof(float));

    const double diff = std::fabs(static_cast<double>(duration) - static_cast<double>(expected[0]));
    std::fprintf(stderr, "duration=%f expected=%f diff=%g\n", duration, expected[0], diff);
    LOOM_CHECK(diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
