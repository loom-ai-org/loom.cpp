// Numerical-correctness check for the MIL-traced SupertonicTTS "vfe" topology (supertonic_export.py,
// part of supertonic_mil.gguf) against a fresh real-module reference fixture at T=10
// (reference_forward_supertonic_mil_extra.py -- the EXISTING reference_forward_supertonic_vfe.py fixture
// uses T=6, which doesn't match the fixture the rest of this export is checked at; L=9 unchanged,
// dynamic here).
//
// The reference's `txt_emb` is 10 columns wide and the topology's is `txt_len`, so this feeds the ten
// real columns and zeroes for the rest, with `txt_msk` saying which is which (BACKLOG.md P4.6). Unlike
// the two text ENCODERS, padding was expected to be inert here from a reading of the source and the
// measurement agreed: `VFTextCrossAttention` `masked_fill`s the padded key columns to -inf before its
// softmax and takes its fractional-RoPE text length from `txt_msk.sum()` rather than from the axis, so
// nothing it computes depends on how wide the padded axis is. This test is what keeps that true.
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
    const char* gguf_env = loom_test::fixture_env("LOOM_SUPERTONIC_MIL_GGUF");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_MIL_REF_DIR");
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
    const uint32_t t_text = model->hparam_u32("txt_len");
    LOOM_CHECK(t_text >= kTextLen);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("vfe"));
    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", kL}, {"n_past", 0}});

    // `txt_emb` widened from kTextLen to t_text, zeroes in the tail -- which is what "ttl_text"
    // really produces there, since its own last op multiplies by the mask.
    std::vector<float> txt_emb_copy(static_cast<size_t>(t_text) * kTxtDim, 0.0f);
    for (uint32_t c = 0; c < kTxtDim; ++c) {
        for (uint32_t t = 0; t < kTextLen; ++t) {
            txt_emb_copy[static_cast<size_t>(c) * t_text + t] = txt_emb[static_cast<size_t>(c) * kTextLen + t];
        }
    }
    std::vector<float> txt_msk(t_text, 0.0f);
    std::fill(txt_msk.begin(), txt_msk.begin() + kTextLen, 1.0f);
    std::vector<float> z_t_copy = z_t, stl_emb_copy = stl_emb;
    std::vector<float> t_copy = {0.3f};
    ggml_backend_tensor_set(r.input_tensors.at("z_t"), z_t_copy.data(), 0, z_t_copy.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("txt_emb"), txt_emb_copy.data(), 0, txt_emb_copy.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl_emb_copy.data(), 0, stl_emb_copy.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("t"), t_copy.data(), 0, t_copy.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("txt_msk"), txt_msk.data(), 0, txt_msk.size() * sizeof(float));
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
