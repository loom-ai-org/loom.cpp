// Numerical-correctness check for SupertonicTTS v2's `VFTimeEncoder` (sinusoidal t*1000*freqs embedding
// + 2-layer Mish MLP, real source: vector_field_estimator.py) against the real
// `vector_estimator.pt`'s own `time_encoder` module, at 3 different `t` values. Skips cleanly if the
// GGUF/reference files aren't present.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

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

void run_one(loom::GgufModel& model, loom::GraphTopology& topo, ggml_backend_t backend,
             const std::string& ref_dir, const std::string& name) {
    const std::vector<float> t = read_f32_binary(ref_dir + "/vftime_" + name + "_t.bin");
    const std::vector<float> expected = read_f32_binary(ref_dir + "/vftime_" + name + "_out.bin");
    LOOM_CHECK(t.size() == 1);
    LOOM_CHECK(expected.size() == 64);

    loom::GraphBuilder builder(topo, model, backend, nullptr);
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", 0}, {"n_past", 0}});
    ggml_backend_tensor_set(r.input_tensors.at("t"), t.data(), 0, sizeof(float));
    ggml_backend_graph_compute(backend, r.graph);

    LOOM_CHECK(static_cast<size_t>(ggml_nelements(r.output)) == 64);
    std::vector<float> out(64);
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(static_cast<double>(out[i]) - static_cast<double>(expected[i]));
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / out.size();
    std::fprintf(stderr, "%s: mean_diff=%g, max_diff=%g\n", name.c_str(), mean_diff, max_diff);
    LOOM_CHECK(mean_diff < 1e-5);
    LOOM_CHECK(max_diff < 1e-3);
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_SUPERTONIC_DIR");
    const char* ref_dir_env = std::getenv("LOOM_SUPERTONIC_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_SUPERTONIC_DIR (supertonic_vftime.gguf) and "
                              "LOOM_SUPERTONIC_REF_DIR (vftime_*.bin) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/supertonic_vftime.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    run_one(*model, topo, backend.get(), ref_dir, "t0");
    run_one(*model, topo, backend.get(), ref_dir, "t037");
    run_one(*model, topo, backend.get(), ref_dir, "t09");

    LOOM_TEST_REPORT_AND_RETURN();
}
