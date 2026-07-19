// Numerical-correctness check for the real HiFi-GAN v1 vocoder conversion (`matcha_vocoder.gguf`)
// against the REAL `matcha.hifigan.models.Generator` module run directly (with real weight_norm
// removed, matching real inference usage) on a small mel input (T=4). Skips cleanly if the GGUF/
// reference files aren't present.

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
    const char* dir_env = std::getenv("LOOM_MATCHA_DIR");
    const char* ref_dir_env = std::getenv("LOOM_MATCHA_VOCODER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_DIR (matcha_vocoder.gguf) and "
                              "LOOM_MATCHA_VOCODER_REF_DIR (ref_vocoder_*.npy, produced by "
                              "reference_forward_matcha_vocoder.py) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/ref_vocoder_mel.npy");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/ref_vocoder_mel.npy not found\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    constexpr int64_t kNFeats = 80;
    std::vector<int64_t> mel_shape, wav_shape;
    std::vector<float> ref_mel = read_npy_f32(ref_dir + "/ref_vocoder_mel.npy", mel_shape);   // (80,T)
    std::vector<float> ref_wav = read_npy_f32(ref_dir + "/ref_vocoder_wav.npy", wav_shape);   // (T*256,)
    LOOM_CHECK(mel_shape.size() == 2 && mel_shape[0] == kNFeats);
    const auto T = static_cast<uint32_t>(mel_shape[1]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/matcha_vocoder.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    loom::GraphBuilder builder(topo, *model, backend.get());
    loom::GraphBuilder::BuildResult r = builder.build(T, 0);

    // ref_mel is (80,T) row-major -- same "[T,C]-convention flat layout equals (C,T)-numpy row-major
    // flat layout" identity established for the Decoder (addition commutes: t+c*T == c*T+t) -- direct
    // copy, no reindexing.
    ggml_backend_tensor_set(r.input_tensors.at("mel"), ref_mel.data(), 0, ref_mel.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> wav(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, wav.data(), 0, wav.size() * sizeof(float));
    LOOM_CHECK(wav.size() == ref_wav.size());

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < wav.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(wav[i] - ref_wav[i])));
    }
    std::fprintf(stderr, "T=%u, wav_len=%zu, max_abs_diff=%g\n", T, wav.size(), max_abs_diff);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
