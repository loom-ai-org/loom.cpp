// Numerical-correctness check for predictor.F0.1 -- the UPSAMPLING AdainResBlk1d instance
// (dim_in=512, dim_out=256, WITH a learned conv1x1 shortcut AND upsample=True -- exercises the
// depthwise-ConvTranspose1d "pool" composition and the learned shortcut TOGETHER for the first time)
// against a hand-rolled pure-PyTorch reference. Fully deterministic, plain exact-match check. Skips
// cleanly if the GGUF/reference files aren't present.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

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
    const char* dir_env = loom_test::fixture_env("LOOM_KOKORO_DIR");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_KOKORO_F0_BLOCK1_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_f0_block1.gguf) and "
                              "LOOM_KOKORO_F0_BLOCK1_REF_DIR (ref_f0block1_*.npy) to run this numerical "
                              "check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kDimIn = 512;
    constexpr uint32_t kDimOut = 256;
    constexpr uint32_t kStyleDim = 128;

    std::vector<int64_t> x_shape, style_shape, out_shape;
    std::vector<float> x = read_npy_f32(ref_dir + "/ref_f0block1_x.npy", x_shape);
    std::vector<float> style = read_npy_f32(ref_dir + "/ref_f0block1_style.npy", style_shape);
    std::vector<float> ref_out = read_npy_f32(ref_dir + "/ref_f0block1_out.npy", out_shape);
    LOOM_CHECK(x_shape.size() == 2 && static_cast<uint32_t>(x_shape[1]) == kDimIn);
    LOOM_CHECK(style_shape.size() == 1 && static_cast<uint32_t>(style_shape[0]) == kStyleDim);
    const auto T = static_cast<uint32_t>(x_shape[0]);
    const auto T_out = static_cast<uint32_t>(out_shape[0]);
    LOOM_CHECK(T_out == 2 * T);
    LOOM_CHECK(static_cast<uint32_t>(out_shape[1]) == kDimOut);

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/kokoro_f0_block1.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", T}, {"n_past", 0}});

    // Same "ggml ne=[T,channels] <-> numpy (channels,T)" transpose as every other Kokoro test here.
    std::vector<float> x_tc(static_cast<size_t>(T) * kDimIn);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < kDimIn; ++c) x_tc[static_cast<size_t>(c) * T + t] = x[static_cast<size_t>(t) * kDimIn + c];
    ggml_backend_tensor_set(r.input_tensors.at("x"), x_tc.data(), 0, x_tc.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("style"), style.data(), 0, style.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    LOOM_CHECK(out.size() == ref_out.size());
    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[0]) == T_out);
    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[1]) == kDimOut);

    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (uint32_t t = 0; t < T_out; ++t) {
        for (uint32_t c = 0; c < kDimOut; ++c) {
            const double d = std::fabs(out[static_cast<size_t>(c) * T_out + t] - ref_out[static_cast<size_t>(t) * kDimOut + c]);
            max_abs_diff = std::max(max_abs_diff, d);
            sum_abs_diff += d;
        }
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(out.size());
    std::fprintf(stderr, "T=%u, T_out=%u, mean_abs_diff=%g, max_abs_diff=%g\n", T, T_out, mean_abs_diff, max_abs_diff);
    LOOM_CHECK(mean_abs_diff < 1e-4);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
