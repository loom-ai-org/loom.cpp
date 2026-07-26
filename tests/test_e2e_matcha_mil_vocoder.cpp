// Numerical-correctness check for the MIL-traced Matcha-TTS "vocoder" topology (export_matcha_mil.py,
// part of matcha_mil.gguf, the real HiFi-GAN v1 `generator_v1` Generator with `remove_weight_norm()`
// applied before tracing -- see export_matcha_mil.py's own `VocoderWrapper` docstring) against the real-
// module reference fixture (reference_forward_matcha_vocoder.py). Layout: `mel` input and the flattened
// waveform output match the reference fixture's own numpy layout directly (see
// test_e2e_matcha_mil_text_encoder.cpp's comment on why numpy row-major and ggml T-fast ne=[T,C] are
// byte-identical). Skips cleanly if the GGUF/reference files aren't present.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<float> read_npy_f32(const std::string& path, std::vector<int64_t>& shape_out) {
    std::ifstream f(path, std::ios::binary);
    LOOM_CHECK(static_cast<bool>(f));
    char magic[6];
    f.read(magic, 6);
    f.ignore(2);
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    const size_t shape_pos = header.find("'shape':");
    const size_t paren_open = header.find('(', shape_pos);
    const size_t paren_close = header.find(')', paren_open);
    std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);
    shape_out.clear();
    std::stringstream ss(shape_str);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::string trimmed;
        for (char c : tok) if (c != ' ') trimmed += c;
        if (!trimmed.empty()) shape_out.push_back(std::stoll(trimmed));
    }
    int64_t total = 1;
    for (int64_t d : shape_out) total *= d;
    std::vector<float> data(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(data.data()), total * static_cast<int64_t>(sizeof(float)));
    return data;
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_MATCHA_MIL_GGUF");
    const char* ref_dir_env = std::getenv("LOOM_MATCHA_VOCODER_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_MIL_GGUF (matcha_mil.gguf, produced by "
                              "export_matcha_mil.py) and LOOM_MATCHA_VOCODER_REF_DIR "
                              "(ref_vocoder_*.npy, produced by reference_forward_matcha_vocoder.py) to "
                              "run this check\n");
        return 77;
    }
    const std::string gguf_path = gguf_env;
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/ref_vocoder_mel.npy");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/ref_vocoder_mel.npy not found\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    std::vector<int64_t> mel_shape, ref_shape;
    std::vector<float> mel = read_npy_f32(ref_dir + "/ref_vocoder_mel.npy", mel_shape);
    std::vector<float> ref_wav = read_npy_f32(ref_dir + "/ref_vocoder_wav.npy", ref_shape);

    LOOM_CHECK(mel_shape.size() == 2); // (n_feats, T)
    const auto T = static_cast<uint32_t>(mel_shape[1]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("vocoder"));

    loom::GraphBuilder builder(topo, *model, backend.get());
    loom::GraphBuilder::BuildResult r = builder.build(T, 0);

    ggml_tensor* mel_t = r.input_tensors.at("mel");
    LOOM_CHECK(static_cast<size_t>(ggml_nelements(mel_t)) == mel.size());
    ggml_backend_tensor_set(mel_t, mel.data(), 0, mel.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    LOOM_CHECK(out.size() == ref_wav.size());

    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(out[i] - ref_wav[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        sum_abs_diff += d;
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(out.size());
    std::fprintf(stderr, "T=%u, n_samples=%zu, mean_abs_diff=%g, max_abs_diff=%g\n",
                 T, out.size(), mean_abs_diff, max_abs_diff);
    LOOM_CHECK(mean_abs_diff < 1e-3);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
