// Numerical-correctness check for Kokoro Generator's forward/inverse STFT (istftnet.py's TorchSTFT),
// against real torch.stft/torch.istft (reference_forward_kokoro_stft.py) -- no real checkpoint weights
// involved (every tensor is a conversion-time-baked constant DFT/window kernel), but still gated on the
// GGUF/reference files existing (produced by convert_kokoro_stft.py + the reference script) rather than
// generated at ctest time, since the reference needs real torch. Magnitude is compared with a plain
// difference; phase is compared with a CIRCULAR difference (the true discontinuity at the +-pi branch
// cut means two independently-computed float32 pipelines can differ by ~2*pi at a handful of elements
// even when both are "correct" -- see kokoro_stft_common.py's module docstring).

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

constexpr double kPi = 3.14159265358979323846;

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

double circular_diff(double a, double b) {
    double d = std::fmod(a - b + kPi, 2.0 * kPi);
    if (d < 0) d += 2.0 * kPi;
    return std::fabs(d - kPi);
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_KOKORO_DIR");
    const char* ref_dir_env = std::getenv("LOOM_KOKORO_STFT_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_stft_forward/inverse.gguf) and "
                              "LOOM_KOKORO_STFT_REF_DIR (ref_stft_*.npy) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kNFft = 20;
    constexpr uint32_t kHop = 5;
    constexpr uint32_t kNFreq = kNFft / 2 + 1;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // --- forward ---
    {
        std::vector<int64_t> wpad_shape, mag_shape, phase_shape;
        std::vector<float> waveform_padded = read_npy_f32(ref_dir + "/ref_stft_waveform_padded.npy", wpad_shape);
        std::vector<float> ref_mag = read_npy_f32(ref_dir + "/ref_stft_mag_fwd.npy", mag_shape);
        std::vector<float> ref_phase = read_npy_f32(ref_dir + "/ref_stft_phase_fwd.npy", phase_shape);
        LOOM_CHECK(mag_shape.size() == 2 && static_cast<uint32_t>(mag_shape[0]) == kNFreq);
        const auto n_frames = static_cast<uint32_t>(mag_shape[1]);
        const auto n_samples_padded = static_cast<uint32_t>(wpad_shape[0]);
        LOOM_CHECK((n_samples_padded - kNFft) / kHop + 1 == n_frames);

        auto model = loom::GgufModel::load(dir + "/kokoro_stft_forward.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
        loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
        loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", n_samples_padded}, {"n_past", 0}});
        ggml_backend_tensor_set(r.input_tensors.at("waveform_padded"), waveform_padded.data(), 0,
                                 waveform_padded.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), r.graph);

        LOOM_CHECK(static_cast<uint32_t>(r.output->ne[0]) == n_frames);
        LOOM_CHECK(static_cast<uint32_t>(r.output->ne[1]) == 2 * kNFreq);
        std::vector<float> har(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, har.data(), 0, har.size() * sizeof(float));

        // ggml ne=[n_frames, 2*n_freq] (n_frames fastest) is byte-identical to numpy (2*n_freq,n_frames)
        // row-major -- coincidentally the SAME flat layout as the (n_freq,n_frames) reference arrays, no
        // reordering needed (same rule as everywhere else in this project).
        double max_mag_diff = 0.0, max_phase_cdiff = 0.0;
        for (uint32_t f = 0; f < kNFreq; ++f) {
            for (uint32_t t = 0; t < n_frames; ++t) {
                const size_t idx = static_cast<size_t>(f) * n_frames + t;
                const float actual_mag = har[idx];
                const float actual_phase = har[static_cast<size_t>(kNFreq) * n_frames + idx];
                max_mag_diff = std::max(max_mag_diff, std::fabs(static_cast<double>(actual_mag) - ref_mag[idx]));
                max_phase_cdiff = std::max(max_phase_cdiff, circular_diff(actual_phase, ref_phase[idx]));
            }
        }
        std::fprintf(stderr, "forward: n_frames=%u, max_mag_diff=%g, max_phase_circular_diff=%g\n",
                     n_frames, max_mag_diff, max_phase_cdiff);
        LOOM_CHECK(max_mag_diff < 1e-4);
        LOOM_CHECK(max_phase_cdiff < 1e-3);
    }

    // --- inverse ---
    {
        std::vector<int64_t> mag_shape, phase_shape, wsum_shape, wav_shape;
        std::vector<float> mag = read_npy_f32(ref_dir + "/ref_stft_mag_inv.npy", mag_shape);
        std::vector<float> phase = read_npy_f32(ref_dir + "/ref_stft_phase_inv.npy", phase_shape);
        std::vector<float> wsum = read_npy_f32(ref_dir + "/ref_stft_wsum.npy", wsum_shape);
        std::vector<float> ref_waveform = read_npy_f32(ref_dir + "/ref_stft_waveform_inv.npy", wav_shape);
        LOOM_CHECK(mag_shape.size() == 2 && static_cast<uint32_t>(mag_shape[0]) == kNFreq);
        const auto n_frames = static_cast<uint32_t>(mag_shape[1]);
        const auto out_len_full = static_cast<uint32_t>(wsum_shape[0]);
        LOOM_CHECK(out_len_full == (n_frames - 1) * kHop + kNFft);

        auto model = loom::GgufModel::load(dir + "/kokoro_stft_inverse.gguf", backend.get());
        LOOM_CHECK(model != nullptr);
        loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
        loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
        loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", n_frames}, {"n_past", 0}});
        ggml_backend_tensor_set(r.input_tensors.at("magnitude"), mag.data(), 0, mag.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("phase"), phase.data(), 0, phase.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("wsum"), wsum.data(), 0, wsum.size() * sizeof(float));
        ggml_backend_graph_compute(backend.get(), r.graph);

        std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
        LOOM_CHECK(out.size() == ref_waveform.size());

        double max_diff = 0.0;
        for (size_t i = 0; i < out.size(); ++i) max_diff = std::max(max_diff, std::fabs(static_cast<double>(out[i]) - ref_waveform[i]));
        std::fprintf(stderr, "inverse: out_len=%zu, max_diff=%g\n", out.size(), max_diff);
        LOOM_CHECK(max_diff < 1e-4);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
