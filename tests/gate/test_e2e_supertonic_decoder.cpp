// Numerical-correctness check for SupertonicTTS v2's SpeechDecoder (real source:
// speech_autoencoding/speech_autoencoder.py) -- the FINAL stage of the whole pipeline, direct
// waveform emission from a causal ConvNeXt stack (no ISTFT/vocoder-GAN stack at all), against the real
// `vocoder.pt` module. Exercises folded eval-mode BatchNorm + the codebook-decompress
// reshape/permute/reshape composition for the first time. Skips cleanly if the GGUF/reference files
// aren't present.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

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
    const char* dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_DIR");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_DIR (supertonic_decoder.gguf) and "
                              "LOOM_SUPERTONIC_REF_DIR (decoder_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kT = 4;
    constexpr uint32_t kLatDim = 144;

    const std::vector<float> latent = read_f32_binary(ref_dir + "/decoder_latent.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/decoder_expected_wav.bin");
    LOOM_CHECK(latent.size() == static_cast<size_t>(kT) * kLatDim);
    LOOM_CHECK(expected.size() == static_cast<size_t>(kT) * 6 * 512);

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/supertonic_decoder.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", kT}, {"n_past", 0}});

    ggml_backend_tensor_set(r.input_tensors.at("latent"), latent.data(), 0, latent.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    LOOM_CHECK(static_cast<size_t>(ggml_nelements(r.output)) == expected.size());
    std::vector<float> out(expected.size());
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(static_cast<double>(out[i]) - static_cast<double>(expected[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / out.size();
    std::fprintf(stderr, "mean_diff=%g, max_diff=%g\n", mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-2);
    LOOM_CHECK(max_diff < 1e-1);

    LOOM_TEST_REPORT_AND_RETURN();
}
