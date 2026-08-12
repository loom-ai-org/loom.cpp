// Numerical-correctness check for the MIL-traced SupertonicTTS "dp" topology (supertonic_export.py,
// part of supertonic_mil.gguf) against a fresh real-module reference fixture at T=10
// (reference_forward_supertonic_mil_extra.py -- the EXISTING reference_forward_supertonic_dp.py fixture
// uses T=12, which doesn't apply here since this "dp" topology is traced dynamically over T; T=10 is just
// what that fixture happened to use, chosen to match the rest of the MIL export's own fixed T_TEXT_FIXED
// for consistency, not because "dp" itself requires it -- see supertonic_export.py's own docstring).
//
// The reference is 10 REAL ids; the topology's text axis is `txt_len` wide (BACKLOG.md P4.6). So the
// ids are padded and `txt_msk` says how many are real, and what this asks is exactly P4.6's question:
// does a padded run reproduce the unpadded ground truth? For `dp` a failure shows up as a wrong
// predicted duration -- audio of the wrong LENGTH rather than a numeric near-miss -- which is why the
// bound here is the same 1e-2 it was when the axis was exactly 10 wide.
//
// Skips cleanly if the GGUF/reference files aren't present.

#include "test_util.h"
#include "fixtures.h"
#include "supertonic_buckets.h"

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
    // The bucket the driver would pick for these ten ids -- the smallest exported width that fits,
    // which is a much sharper test than always running the widest graph: it is the one production
    // actually runs for a short utterance (BACKLOG.md P4.6a).
    uint32_t t_text = 0;
    const std::string topo_name = loom_test::supertonic_bucket_topology(
        *model, "dp", static_cast<uint32_t>(txt_ids.size()), &t_text);
    LOOM_CHECK(!topo_name.empty());
    std::fprintf(stderr, "bucket: %s (%zu real ids)\n", topo_name.c_str(), txt_ids.size());
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json(topo_name));
    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", t_text}, {"n_past", 0}});

    // The driver's own padding, done by hand: PAD_ID into the tail, ones-then-zeros for the mask.
    // Which id pads is measured not to matter (`x = x * txt_msk` zeroes it first, BACKLOG.md P4.6);
    // 162 is the vocabulary's one unused row, so a dump of these ids is unambiguous.
    std::vector<int32_t> txt_ids_copy(t_text, 162);
    std::copy(txt_ids.begin(), txt_ids.end(), txt_ids_copy.begin());
    std::vector<float> txt_msk(t_text, 0.0f);
    std::fill(txt_msk.begin(), txt_msk.begin() + txt_ids.size(), 1.0f);
    std::vector<float> stl_emb_copy = stl_emb;
    ggml_backend_tensor_set(r.input_tensors.at("txt_ids"), txt_ids_copy.data(), 0, txt_ids_copy.size() * sizeof(int32_t));
    ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl_emb_copy.data(), 0, stl_emb_copy.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("txt_msk"), txt_msk.data(), 0, txt_msk.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == 1);
    float duration = 0.0f;
    ggml_backend_tensor_get(r.output, &duration, 0, sizeof(float));

    const double diff = std::fabs(static_cast<double>(duration) - static_cast<double>(expected[0]));
    std::fprintf(stderr, "duration=%f expected=%f diff=%g\n", duration, expected[0], diff);
    LOOM_CHECK(diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
