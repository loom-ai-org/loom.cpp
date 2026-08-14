// Numerical-correctness check for Kokoro Generator's NSF harmonic source (istftnet.py's `SineGen`+
// `SourceModuleHnNSF`, converted via convert_kokoro_sinegen.py), against a hand-rolled real-PyTorch
// reference (reference_forward_kokoro_sinegen.py). No real checkpoint weights involved (l_linear's
// weight/bias are a fixed synthetic draw shared between the conversion script and the reference via
// kokoro_sinegen_l_linear_{w,b}.npy). Fully deterministic given the same host-drawn rand_ini/noise
// inputs (fed from the reference's own saved arrays, not re-drawn here) -- plain exact-match check.

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
    const char* ref_dir_env = loom_test::fixture_env("LOOM_KOKORO_SINEGEN_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_sinegen.gguf) and "
                              "LOOM_KOKORO_SINEGEN_REF_DIR (ref_sinegen_*.npy) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kUpsampleScale = 300;

    std::vector<int64_t> f0_shape, rand_ini_shape, noise_shape, har_shape;
    std::vector<float> f0_curve = read_npy_f32(ref_dir + "/ref_sinegen_f0_curve.npy", f0_shape);
    std::vector<float> rand_ini = read_npy_f32(ref_dir + "/ref_sinegen_rand_ini.npy", rand_ini_shape);
    std::vector<float> noise = read_npy_f32(ref_dir + "/ref_sinegen_noise.npy", noise_shape);
    std::vector<float> ref_har = read_npy_f32(ref_dir + "/ref_sinegen_har_source.npy", har_shape);
    const auto T_frames = static_cast<uint32_t>(f0_shape[0]);
    const auto dim = static_cast<uint32_t>(rand_ini_shape[0]);
    const uint32_t L = T_frames * kUpsampleScale;
    LOOM_CHECK(noise_shape.size() == 2 && static_cast<uint32_t>(noise_shape[0]) == L
               && static_cast<uint32_t>(noise_shape[1]) == dim);
    LOOM_CHECK(static_cast<uint32_t>(har_shape[0]) == L);

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/kokoro_sinegen.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend.get(), nullptr);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", T_frames}, {"n_past", 0}});

    ggml_backend_tensor_set(r.input_tensors.at("f0_curve"), f0_curve.data(), 0, f0_curve.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("rand_ini"), rand_ini.data(), 0, rand_ini.size() * sizeof(float));
    // ref_sinegen_noise.npy is (L,dim) row-major (dim fastest) -- the topology's "noise" input is ggml
    // ne=[L,dim] (L fastest, matching CONV_1D's own [T,C] convention), so this needs a real transpose,
    // NOT a no-op flatten, same rule as every other [T,C]-convention input in this project.
    std::vector<float> noise_tc(static_cast<size_t>(L) * dim);
    for (uint32_t t = 0; t < L; ++t)
        for (uint32_t c = 0; c < dim; ++c) noise_tc[static_cast<size_t>(c) * L + t] = noise[static_cast<size_t>(t) * dim + c];
    ggml_backend_tensor_set(r.input_tensors.at("noise"), noise_tc.data(), 0, noise_tc.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == L);
    std::vector<float> har(L);
    ggml_backend_tensor_get(r.output, har.data(), 0, har.size() * sizeof(float));

    double max_diff = 0.0, sum_diff = 0.0;
    for (uint32_t i = 0; i < L; ++i) {
        const double d = std::fabs(static_cast<double>(har[i]) - ref_har[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / static_cast<double>(L);
    std::fprintf(stderr, "T_frames=%u, L=%u, mean_diff=%g, max_diff=%g\n", T_frames, L, mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-4);
    LOOM_CHECK(max_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
