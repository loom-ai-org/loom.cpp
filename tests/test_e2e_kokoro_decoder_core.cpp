// Numerical-correctness check for Kokoro's Decoder "core" (istftnet.py's Decoder.forward, everything
// except the final self.generator(...) call -- the Generator is a separately-verified topology, run next
// by the (future) host driver), against a hand-rolled PyTorch reference on a small SYNTHETIC (real
// checkpoint shapes, random weights) instance -- same checkpoint-independent structural verification
// precedent as every other Generator/Decoder piece this milestone. Fully deterministic exact-match
// check. Skips cleanly if the GGUF/reference files aren't present.

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
    const char* dir_env = std::getenv("LOOM_KOKORO_DIR");
    const char* ref_dir_env = std::getenv("LOOM_KOKORO_DECODER_CORE_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_decoder_core.gguf) and "
                              "LOOM_KOKORO_DECODER_CORE_REF_DIR (ref_decoder_core_*.npy) to run this "
                              "numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kAsrCh = 512;
    constexpr uint32_t kStyleDim = 128;

    std::vector<int64_t> asr_shape, f0_shape, n_shape, style_shape, x_shape;
    std::vector<float> asr = read_npy_f32(ref_dir + "/ref_decoder_core_asr.npy", asr_shape);
    std::vector<float> f0_curve = read_npy_f32(ref_dir + "/ref_decoder_core_f0_curve.npy", f0_shape);
    std::vector<float> n_curve = read_npy_f32(ref_dir + "/ref_decoder_core_n_curve.npy", n_shape);
    std::vector<float> style = read_npy_f32(ref_dir + "/ref_decoder_core_style.npy", style_shape);
    std::vector<float> ref_x = read_npy_f32(ref_dir + "/ref_decoder_core_x.npy", x_shape);
    LOOM_CHECK(asr_shape.size() == 2 && static_cast<uint32_t>(asr_shape[1]) == kAsrCh);
    const auto T = static_cast<uint32_t>(asr_shape[0]);
    LOOM_CHECK(static_cast<uint32_t>(f0_shape[0]) == 2 * T);
    LOOM_CHECK(static_cast<uint32_t>(x_shape[0]) == 2 * T && static_cast<uint32_t>(x_shape[1]) == kAsrCh);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/kokoro_decoder_core.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    loom::GraphBuilder::BuildResult r = builder.build(T, 0);

    // ggml ne=[T,C] (T fastest) -- real transpose from the reference's (T,C) row-major layout, same rule
    // as every other [T,C]-convention test this whole milestone. f0_curve/n_curve are 1D -- no transpose.
    std::vector<float> asr_tc(static_cast<size_t>(T) * kAsrCh);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < kAsrCh; ++c) asr_tc[static_cast<size_t>(c) * T + t] = asr[static_cast<size_t>(t) * kAsrCh + c];

    ggml_backend_tensor_set(r.input_tensors.at("asr"), asr_tc.data(), 0, asr_tc.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("f0_curve"), f0_curve.data(), 0, f0_curve.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("n_curve"), n_curve.data(), 0, n_curve.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("style"), style.data(), 0, style.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[0]) == 2 * T);
    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[1]) == kAsrCh);
    std::vector<float> x(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, x.data(), 0, x.size() * sizeof(float));

    double max_abs_diff = 0.0, sum_abs_diff = 0.0;
    for (uint32_t t = 0; t < 2 * T; ++t) {
        for (uint32_t c = 0; c < kAsrCh; ++c) {
            const double d = std::fabs(x[static_cast<size_t>(c) * 2 * T + t] - ref_x[static_cast<size_t>(t) * kAsrCh + c]);
            max_abs_diff = std::max(max_abs_diff, d);
            sum_abs_diff += d;
        }
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(x.size());
    std::fprintf(stderr, "T=%u, mean_abs_diff=%g, max_abs_diff=%g\n", T, mean_abs_diff, max_abs_diff);
    LOOM_CHECK(mean_abs_diff < 1e-3);
    LOOM_CHECK(max_abs_diff < 1e-1);

    LOOM_TEST_REPORT_AND_RETURN();
}
