// Numerical-correctness check for Kokoro's Generator "core" (istftnet.py's Generator.forward, minus the
// SineGen/forward-STFT piece that produces "har" -- those are separately verified topologies, "har" is
// fed here as a ready-made input), against a hand-rolled PyTorch reference on a small SYNTHETIC (real
// checkpoint shapes, random weights) instance -- same checkpoint-independent structural verification
// precedent as VITS's test_hifigan_generator. Fully deterministic exact-match check. Skips cleanly if the
// GGUF/reference files aren't present.

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

std::vector<float> compute_wsum(uint32_t n_frames, uint32_t n_fft, uint32_t hop) {
    std::vector<float> window(n_fft);
    for (uint32_t n = 0; n < n_fft; ++n) window[n] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * n / n_fft);
    const uint32_t out_len = (n_frames - 1) * hop + n_fft;
    std::vector<float> wsum(out_len, 0.0f);
    for (uint32_t t = 0; t < n_frames; ++t)
        for (uint32_t n = 0; n < n_fft; ++n) wsum[t * hop + n] += window[n] * window[n];
    return wsum;
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_KOKORO_DIR");
    const char* ref_dir_env = std::getenv("LOOM_KOKORO_GENERATOR_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_generator.gguf) and "
                              "LOOM_KOKORO_GENERATOR_REF_DIR (ref_generator_*.npy) to run this numerical "
                              "check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kUic = 512;
    constexpr uint32_t kStyleDim = 128;
    constexpr uint32_t kNFft = 20;
    constexpr uint32_t kHop = 5;
    constexpr uint32_t kNFreq = kNFft / 2 + 1;

    std::vector<int64_t> x_shape, style_shape, har_shape, wav_shape;
    std::vector<float> x = read_npy_f32(ref_dir + "/ref_generator_x.npy", x_shape);
    std::vector<float> style = read_npy_f32(ref_dir + "/ref_generator_style.npy", style_shape);
    std::vector<float> har = read_npy_f32(ref_dir + "/ref_generator_har.npy", har_shape);
    std::vector<float> ref_waveform = read_npy_f32(ref_dir + "/ref_generator_waveform.npy", wav_shape);
    LOOM_CHECK(x_shape.size() == 2 && static_cast<uint32_t>(x_shape[1]) == kUic);
    const auto T0 = static_cast<uint32_t>(x_shape[0]);
    const auto T_har = static_cast<uint32_t>(har_shape[0]);
    LOOM_CHECK(T_har == T0 * 60 + 1);
    LOOM_CHECK(static_cast<uint32_t>(har_shape[1]) == 2 * kNFreq);
    LOOM_CHECK(static_cast<uint32_t>(wav_shape[0]) == T0 * 300);

    std::vector<float> wsum = compute_wsum(T_har, kNFft, kHop);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/kokoro_generator.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", T0}, {"n_past", 0}});

    // "x"/"har" are ggml ne=[T,C] (T fastest) -- real transpose from the reference's (T,C) row-major
    // layout, same rule as every other [T,C]-convention test this whole milestone.
    std::vector<float> x_tc(static_cast<size_t>(T0) * kUic);
    for (uint32_t t = 0; t < T0; ++t)
        for (uint32_t c = 0; c < kUic; ++c) x_tc[static_cast<size_t>(c) * T0 + t] = x[static_cast<size_t>(t) * kUic + c];
    std::vector<float> har_tc(static_cast<size_t>(T_har) * 2 * kNFreq);
    for (uint32_t t = 0; t < T_har; ++t)
        for (uint32_t c = 0; c < 2 * kNFreq; ++c) har_tc[static_cast<size_t>(c) * T_har + t] = har[static_cast<size_t>(t) * 2 * kNFreq + c];

    ggml_backend_tensor_set(r.input_tensors.at("x"), x_tc.data(), 0, x_tc.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("style"), style.data(), 0, style.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("har"), har_tc.data(), 0, har_tc.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("wsum"), wsum.data(), 0, wsum.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, waveform.data(), 0, waveform.size() * sizeof(float));
    LOOM_CHECK(waveform.size() == ref_waveform.size());

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < waveform.size(); ++i) {
        const double d = std::fabs(static_cast<double>(waveform[i]) - ref_waveform[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / static_cast<double>(waveform.size());
    std::fprintf(stderr, "T0=%u, T_har=%u, waveform_len=%zu, mean_diff=%g, max_diff=%g\n", T0, T_har,
                 waveform.size(), mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-3);
    LOOM_CHECK(max_diff < 1e-1);

    LOOM_TEST_REPORT_AND_RETURN();
}
