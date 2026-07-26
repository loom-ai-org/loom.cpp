// Numerical-correctness check for the MIL-traced StyleTTS2 "diffusion" topology (export_styletts2_mil.py's
// DiffusionNetWrapper, part of styletts2_mil.gguf -- a real trace of Modules/diffusion/modules.py's
// Transformer1d.run(), superseding convert_styletts2_diffusion.py's own hand-derived topology) against
// the SAME hand-rolled PyTorch reference test_e2e_styletts2_diffusion_net.cpp already uses
// (reference_forward_styletts2_diffusion.py) -- byte-identical inputs/outputs, so the exact same diff_*
// .bin fixtures are reused as-is. UNLIKE that test, no `attn_mask` input is set here: the real
// Transformer1d has no masking at all (see DiffusionNetWrapper's own docstring in export_styletts2_mil.py
// for why the old bespoke topology declared one anyway -- a loom ATTENTION-op API formality, not a real
// model input). Skips cleanly if the GGUF/reference files aren't present.

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
    const char* gguf_env = std::getenv("LOOM_STYLETTS2_MIL_GGUF");
    const char* ref_dir_env = std::getenv("LOOM_STYLETTS2_DIFFUSION_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STYLETTS2_MIL_GGUF (styletts2_mil.gguf, produced by "
                              "export_styletts2_mil.py) and LOOM_STYLETTS2_DIFFUSION_REF_DIR (diff_*.bin, "
                              "produced by reference_forward_styletts2_diffusion.py) to run this check\n");
        return 77;
    }
    const std::string gguf_path = gguf_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kChannels = 256;
    constexpr uint32_t kContextFeatures = 768;

    const std::vector<float> x_in = read_f32_binary(ref_dir + "/diff_x_in.bin");
    const std::vector<float> time = read_f32_binary(ref_dir + "/diff_time.bin");
    const std::vector<float> embedding = read_f32_binary(ref_dir + "/diff_embedding.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/diff_expected_model_out.bin");
    LOOM_CHECK(x_in.size() == kChannels);
    LOOM_CHECK(time.size() == 1);
    LOOM_CHECK(expected.size() == kChannels);
    LOOM_CHECK(embedding.size() % kContextFeatures == 0);
    const uint32_t T = static_cast<uint32_t>(embedding.size() / kContextFeatures);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("diffusion"));
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    loom::GraphBuilder::BuildResult r = builder.build(T, 0);

    ggml_backend_tensor_set(r.input_tensors.at("x_in"), x_in.data(), 0, x_in.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("time"), time.data(), 0, time.size() * sizeof(float));
    // embedding (ref: (T,768) row-major, time-major) is byte-identical to ggml ne=[768,T] -- no reordering.
    ggml_backend_tensor_set(r.input_tensors.at("embedding"), embedding.data(), 0, embedding.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);

    LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == kChannels);
    std::vector<float> out(kChannels);
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(static_cast<double>(out[i]) - static_cast<double>(expected[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / static_cast<double>(out.size());
    std::fprintf(stderr, "T=%u, mean_diff=%g, max_diff=%g\n", T, mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-4);
    LOOM_CHECK(max_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
