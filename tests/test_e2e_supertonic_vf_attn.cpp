// Numerical-correctness check for SupertonicTTS v2's VFTextCrossAttention (FRACTIONAL RoPE -- the
// hardest single piece in this whole effort, `position = index/actual_length`, not integer positions)
// and VFStyleCrossAttention, against the real `vector_estimator.pt`'s own `text_attn[0]`/`style_attn[0]`
// modules. Skips cleanly if the GGUF/reference files aren't present.

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

double compare(const std::vector<float>& a, const std::vector<float>& b, const char* name) {
    LOOM_CHECK(a.size() == b.size());
    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / a.size();
    std::fprintf(stderr, "%s: mean_diff=%g, max_diff=%g\n", name, mean_diff, max_diff);
    return max_diff;
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_SUPERTONIC_DIR");
    const char* ref_dir_env = std::getenv("LOOM_SUPERTONIC_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_DIR (supertonic_vf_{text,style}_attn.gguf) "
                              "and LOOM_SUPERTONIC_REF_DIR (vf_attn_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kL = 7;
    constexpr uint32_t kT = 11;
    constexpr uint32_t kLatDim = 512;
    constexpr uint32_t kTxtDim = 256;
    constexpr uint32_t kStlDim = 256;
    constexpr uint32_t kNStyle = 50;

    const std::vector<float> latent = read_f32_binary(ref_dir + "/vf_attn_latent.bin");
    const std::vector<float> txt_emb = read_f32_binary(ref_dir + "/vf_attn_txt_emb.bin");
    const std::vector<float> stl_emb = read_f32_binary(ref_dir + "/vf_attn_stl_emb.bin");
    const std::vector<float> lat_frac = read_f32_binary(ref_dir + "/vf_attn_lat_frac.bin");
    const std::vector<float> txt_frac = read_f32_binary(ref_dir + "/vf_attn_txt_frac.bin");
    const std::vector<float> expected_text = read_f32_binary(ref_dir + "/vf_attn_expected_text_out.bin");
    const std::vector<float> expected_style = read_f32_binary(ref_dir + "/vf_attn_expected_style_out.bin");
    LOOM_CHECK(latent.size() == static_cast<size_t>(kLatDim) * kL);
    LOOM_CHECK(txt_emb.size() == static_cast<size_t>(kTxtDim) * kT);
    LOOM_CHECK(stl_emb.size() == static_cast<size_t>(kStlDim) * kNStyle);
    LOOM_CHECK(lat_frac.size() == kL);
    LOOM_CHECK(txt_frac.size() == kT);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // --- VFTextCrossAttention (fractional RoPE) ---
    {
        auto model = loom::GgufModel::load(dir + "/supertonic_vf_text_attn.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
        loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
        const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", kL}, {"n_past", 0}});
        ggml_backend_tensor_set(r.input_tensors.at("latent"), latent.data(), 0, latent.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("txt_emb"), txt_emb.data(), 0, txt_emb.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("lat_frac"), lat_frac.data(), 0, lat_frac.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("txt_frac"), txt_frac.data(), 0, txt_frac.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), r.graph);
        LOOM_CHECK(static_cast<size_t>(ggml_nelements(r.output)) == expected_text.size());
        std::vector<float> out(expected_text.size());
        ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
        LOOM_CHECK(compare(out, expected_text, "vf_text_attn") < 1e-2);
    }

    // --- VFStyleCrossAttention ---
    {
        auto model = loom::GgufModel::load(dir + "/supertonic_vf_style_attn.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
        loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
        const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", kL}, {"n_past", 0}});
        ggml_backend_tensor_set(r.input_tensors.at("latent"), latent.data(), 0, latent.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), stl_emb.data(), 0, stl_emb.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), r.graph);
        LOOM_CHECK(static_cast<size_t>(ggml_nelements(r.output)) == expected_style.size());
        std::vector<float> out(expected_style.size());
        ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
        LOOM_CHECK(compare(out, expected_style, "vf_style_attn") < 1e-2);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
