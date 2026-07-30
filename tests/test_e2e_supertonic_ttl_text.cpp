// Numerical-correctness check for SupertonicTTS v2's TTLStyleEncoder + TTLTextEncoder (real source:
// text_to_latent_encoding/encoders.py) against the real `ttl-style-encoder.pt`/`text_encoder.pt`
// modules. Exercises `SpeechPromptedCrossAttention`/`SpeechPromptedTextEncoder` (text queries attend
// over a LEARNABLE-key style value, distinct from `StyleCrossAttention`'s own role-reversed mechanism)
// for the first time. Skips cleanly if the GGUF/reference files aren't present.

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

std::vector<int32_t> read_i32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<int32_t> data(static_cast<size_t>(bytes) / sizeof(int32_t));
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
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_DIR (supertonic_ttl_{text,style}.gguf) and "
                              "LOOM_SUPERTONIC_REF_DIR (ttl_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kT = 10;
    constexpr uint32_t kCropLen = 50;
    constexpr uint32_t kLatDim = 144;
    constexpr uint32_t kStlDim = 256;
    constexpr uint32_t kNStyle = 50;
    constexpr uint32_t kTxtDim = 256;

    const std::vector<int32_t> txt_ids = read_i32_binary(ref_dir + "/ttl_txt_ids.bin");
    const std::vector<float> lat_crop = read_f32_binary(ref_dir + "/ttl_lat_crop.bin");
    const std::vector<float> expected_stl = read_f32_binary(ref_dir + "/ttl_expected_stl_emb.bin");
    const std::vector<float> expected_txt = read_f32_binary(ref_dir + "/ttl_expected_txt_emb.bin");
    LOOM_CHECK(txt_ids.size() == kT);
    LOOM_CHECK(lat_crop.size() == static_cast<size_t>(kCropLen) * kLatDim);
    LOOM_CHECK(expected_stl.size() == static_cast<size_t>(kStlDim) * kNStyle);
    LOOM_CHECK(expected_txt.size() == static_cast<size_t>(kT) * kTxtDim);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // --- TTLStyleEncoder alone ---
    {
        auto model = loom::GgufModel::load(dir + "/supertonic_ttl_style.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
        loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
        loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", kCropLen}, {"n_past", 0}});
        ggml_backend_tensor_set(r.input_tensors.at("lat_crop"), lat_crop.data(), 0, lat_crop.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), r.graph);
        LOOM_CHECK(static_cast<size_t>(ggml_nelements(r.output)) == expected_stl.size());
        std::vector<float> out(expected_stl.size());
        ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
        LOOM_CHECK(compare(out, expected_stl, "ttl_style") < 1e-2);
    }

    // --- Full TTLTextEncoder (TTLStyleEncoder -> TTLTextEncoder in one graph) ---
    {
        auto model = loom::GgufModel::load(dir + "/supertonic_ttl_text.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
        loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
        loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", kT}, {"n_past", 0}});
        ggml_backend_tensor_set(r.input_tensors.at("lat_crop"), lat_crop.data(), 0, lat_crop.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("txt_ids"), txt_ids.data(), 0, txt_ids.size() * sizeof(int32_t));
        ggml_backend_graph_compute(backend.get(), r.graph);
        LOOM_CHECK(static_cast<size_t>(ggml_nelements(r.output)) == expected_txt.size());
        std::vector<float> out(expected_txt.size());
        ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
        LOOM_CHECK(compare(out, expected_txt, "ttl_text") < 1e-1);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
