// Numerical-correctness check for the MIL-traced Kokoro "decoder_vocoder" topology (export_kokoro_mil.py,
// part of kokoro_mil.gguf) against a pure-PyTorch reference (reference_forward_kokoro_decoder_vocoder_mil
// .py, which runs export_kokoro_mil.py's own DecoderVocoderWrapper eagerly on real checkpoint weights and
// concrete non-zero inputs -- see that script's own docstring for why the WRAPPER, not the original
// untraced Decoder.forward, is the correct ground truth for this topology specifically). Supersedes
// test_e2e_kokoro_mil_decoder_vocoder_smoke.cpp's purely-structural (zero-filled-input) check with a real
// numerical one. Skips cleanly if the GGUF/reference files aren't present.

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

void set_input(const loom::GraphBuilder::BuildResult& r, const std::string& name, const std::vector<float>& data) {
    ggml_tensor* t = r.input_tensors.at(name);
    LOOM_CHECK(static_cast<size_t>(ggml_nelements(t)) == data.size());
    std::vector<float> copy = data;
    ggml_backend_tensor_set(t, copy.data(), 0, copy.size() * sizeof(float));
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_KOKORO_MIL_GGUF");
    const char* ref_dir_env = std::getenv("LOOM_KOKORO_MIL_DECODER_VOCODER_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_MIL_GGUF (kokoro_mil.gguf, produced by "
                              "export_kokoro_mil.py) and LOOM_KOKORO_MIL_DECODER_VOCODER_REF_DIR "
                              "(ref_decoder_vocoder_*.npy, produced by "
                              "reference_forward_kokoro_decoder_vocoder_mil.py) to run this check\n");
        return 77;
    }
    const std::string gguf_path = gguf_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> asr_shape, f0_shape, n_shape, s_shape, rand_ini_shape, noise_shape, wsum_shape, out_shape;
    std::vector<float> asr = read_npy_f32(ref_dir + "/ref_decoder_vocoder_asr.npy", asr_shape);
    std::vector<float> f0_curve = read_npy_f32(ref_dir + "/ref_decoder_vocoder_f0_curve.npy", f0_shape);
    std::vector<float> n_curve = read_npy_f32(ref_dir + "/ref_decoder_vocoder_n_curve.npy", n_shape);
    std::vector<float> s = read_npy_f32(ref_dir + "/ref_decoder_vocoder_s.npy", s_shape);
    std::vector<float> rand_ini = read_npy_f32(ref_dir + "/ref_decoder_vocoder_rand_ini.npy", rand_ini_shape);
    std::vector<float> noise_in = read_npy_f32(ref_dir + "/ref_decoder_vocoder_noise_in.npy", noise_shape);
    std::vector<float> wsum = read_npy_f32(ref_dir + "/ref_decoder_vocoder_wsum.npy", wsum_shape);
    std::vector<float> ref_out = read_npy_f32(ref_dir + "/ref_decoder_vocoder_out.npy", out_shape);

    LOOM_CHECK(asr_shape.size() == 3); // (1, dim_in, T_frames)
    const auto t_frames = static_cast<uint32_t>(asr_shape[2]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("decoder_vocoder"));

    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_enc_frames", t_frames}, {"n_past", 0}});

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
    // Per-phase bisection (see BACKLOG.md's decoder_vocoder numerical-verification entry) independently
    // confirmed decoder_core exact (~1e-6), the SineGen chain to ~2e-3, forward STFT to ~2e-7 (excluding
    // 3/105622 elements landing exactly on a real atan2 sign-crossing boundary -- see VerifiedSTFT's own
    // `boundary_eps` comment for the general coremltools atan2-decomposition bug found and fixed there),
    // and the Generator core (given an exact `har`) to ~3e-3 -- so `mean_abs_diff` (averaging over the
    // whole waveform) has real, tight margin at 2e-3. `max_abs_diff` is looser: a HiFi-GAN-style vocoder
    // is a long (~20+ conv/resblock stage), non-linear, occasionally near-resonant network, and the small
    // per-phase residuals above compound through it the same way StyleTTS2's own ~1e-6 per-diffusion-step
    // residual was found to compound into a real ~3e-3 full-pipeline ceiling (see BACKLOG.md) -- not
    // further fixable by chasing individual primitives. 0.05 has real margin above the observed ~0.025
    // peak (a single-sample spike, consistent with a resonance point) while still catching a genuinely
    // broken (not just precision-limited) topology.
    LOOM_CHECK(mean_abs_diff < 2e-3);
    LOOM_CHECK(max_abs_diff < 0.05);

    LOOM_TEST_REPORT_AND_RETURN();
}
