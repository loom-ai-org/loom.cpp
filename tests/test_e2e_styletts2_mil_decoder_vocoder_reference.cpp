// Numerical-correctness check for the MIL-traced StyleTTS2 "decoder_vocoder" topology
// (export_styletts2_mil.py, part of styletts2_mil.gguf -- DIRECT reuse of export_kokoro_mil.py's own
// DecoderVocoderWrapper against StyleTTS2's own checkpoint weights) against a pure-PyTorch reference
// (reference_forward_styletts2_decoder_vocoder_mil.py). Mirrors
// test_e2e_kokoro_mil_decoder_vocoder_reference.cpp's own shape/tolerances exactly -- same wrapper code,
// same architecture, only the weights differ. Skips cleanly if the GGUF/reference files aren't present.

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

void set_input(loom::GraphBuilder::BuildResult& r, const std::string& name, const std::vector<float>& data) {
    ggml_tensor* t = r.input_tensors.at(name);
    LOOM_CHECK(static_cast<size_t>(ggml_nelements(t)) == data.size());
    std::vector<float> copy = data;
    ggml_backend_tensor_set(t, copy.data(), 0, copy.size() * sizeof(float));
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_STYLETTS2_MIL_GGUF");
    const char* ref_dir_env = std::getenv("LOOM_STYLETTS2_MIL_DECODER_VOCODER_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STYLETTS2_MIL_GGUF (styletts2_mil.gguf, produced by "
                              "export_styletts2_mil.py) and LOOM_STYLETTS2_MIL_DECODER_VOCODER_REF_DIR "
                              "(ref_styletts2_decoder_vocoder_*.npy, produced by "
                              "reference_forward_styletts2_decoder_vocoder_mil.py) to run this check\n");
        return 77;
    }
    const std::string gguf_path = gguf_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> asr_shape, f0_shape, n_shape, s_shape, rand_ini_shape, noise_shape, wsum_shape, out_shape;
    std::vector<float> asr = read_npy_f32(ref_dir + "/ref_styletts2_decoder_vocoder_asr.npy", asr_shape);
    std::vector<float> f0_curve = read_npy_f32(ref_dir + "/ref_styletts2_decoder_vocoder_f0_curve.npy", f0_shape);
    std::vector<float> n_curve = read_npy_f32(ref_dir + "/ref_styletts2_decoder_vocoder_n_curve.npy", n_shape);
    std::vector<float> s = read_npy_f32(ref_dir + "/ref_styletts2_decoder_vocoder_s.npy", s_shape);
    std::vector<float> rand_ini = read_npy_f32(ref_dir + "/ref_styletts2_decoder_vocoder_rand_ini.npy", rand_ini_shape);
    std::vector<float> noise_in = read_npy_f32(ref_dir + "/ref_styletts2_decoder_vocoder_noise_in.npy", noise_shape);
    std::vector<float> wsum = read_npy_f32(ref_dir + "/ref_styletts2_decoder_vocoder_wsum.npy", wsum_shape);
    std::vector<float> ref_out = read_npy_f32(ref_dir + "/ref_styletts2_decoder_vocoder_out.npy", out_shape);

    LOOM_CHECK(asr_shape.size() == 3); // (1, dim_in, T_frames)
    const auto t_frames = static_cast<uint32_t>(asr_shape[2]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("decoder_vocoder"));

    loom::GraphBuilder builder(topo, *model, backend.get());
    loom::GraphBuilder::BuildResult r = builder.build({{"n_enc_frames", t_frames}, {"n_past", 0}});

    set_input(r, "asr", asr);
    set_input(r, "f0_curve", f0_curve);
    set_input(r, "n_curve", n_curve);
    set_input(r, "s", s);
    set_input(r, "rand_ini", rand_ini);
    set_input(r, "noise_in", noise_in);
    set_input(r, "wsum", wsum);

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    LOOM_CHECK(out.size() == ref_out.size());

    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(out[i] - ref_out[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        sum_abs_diff += d;
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(out.size());
    std::fprintf(stderr, "t_frames=%u, n_samples=%zu, mean_abs_diff=%g, max_abs_diff=%g\n",
                 t_frames, out.size(), mean_abs_diff, max_abs_diff);
    // Same real, bounded HiFi-GAN-vocoder amplification ceiling as Kokoro's own decoder_vocoder test --
    // see that file's own comment for the full per-phase-bisection rationale (identical architecture,
    // only the checkpoint weights differ, so the same tolerances apply).
    LOOM_CHECK(mean_abs_diff < 2e-3);
    LOOM_CHECK(max_abs_diff < 0.05);

    LOOM_TEST_REPORT_AND_RETURN();
}
