// Numerical-correctness check for the MIL-traced SupertonicTTS "decoder" topology
// (export_supertonic_mil.py, part of supertonic_mil.gguf) against the SAME real-module reference fixture
// the bespoke conversion's own test_e2e_supertonic_decoder.cpp already uses
// (reference_forward_supertonic_decoder.py) -- valid ground truth directly (SpeechDecoder never touches
// the text axis at all, so the MIL export's own T_TEXT_FIXED scope limitation is irrelevant here).
// Skips cleanly if the GGUF/reference files aren't present.

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
    const char* gguf_env = loom_test::fixture_env("LOOM_SUPERTONIC_MIL_GGUF");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_SUPERTONIC_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_MIL_GGUF (supertonic_mil.gguf, produced by "
                              "export_supertonic_mil.py) and LOOM_SUPERTONIC_REF_DIR (decoder_*.bin, "
                              "produced by reference_forward_supertonic_decoder.py) to run this check\n");
        return 77;
    }
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/decoder_latent.bin");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/decoder_latent.bin not found\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    constexpr uint32_t kLatDim = 144;
    const std::vector<float> latent = read_f32_binary(ref_dir + "/decoder_latent.bin");
    const std::vector<float> expected_wav = read_f32_binary(ref_dir + "/decoder_expected_wav.bin");
    LOOM_CHECK(latent.size() % kLatDim == 0);
    const uint32_t T = static_cast<uint32_t>(latent.size() / kLatDim);

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_env, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("decoder"));
    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", T}, {"n_past", 0}});

    std::vector<float> latent_copy = latent;
    ggml_backend_tensor_set(r.input_tensors.at("latent"), latent_copy.data(), 0, latent_copy.size() * sizeof(float));
    ggml_backend_graph_compute(backend.get(), r.graph);

    std::vector<float> wav(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, wav.data(), 0, wav.size() * sizeof(float));

    LOOM_CHECK(wav.size() == expected_wav.size());
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < wav.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(wav[i] - expected_wav[i])));
    }
    std::fprintf(stderr, "wav_max_abs_diff=%g (n=%zu)\n", max_abs_diff, wav.size());
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
