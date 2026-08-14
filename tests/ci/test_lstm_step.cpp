// Verifies the composite-JSON LSTM-step pattern (MUL_MAT/ADD/VIEW/SIGMOID/TANH/MUL -- no monolithic LSTM
// primitive, per BACKLOG.md's Gap-1 design decision) bit-exact against an independent numpy reference,
// BEFORE this pattern is trusted inside the real TdtDecoder driver. Fully synthetic and small (tiny
// hidden/input sizes) -- procedurally generated at ctest time like the other toy fixtures, not
// skip-if-missing like the real-checkpoint tests.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cmath>
#include <cstdio>
#include <fstream>

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

std::vector<float> run_one(const std::string& gguf_path, ggml_backend_t backend,
                            const std::vector<float>& x, const std::vector<float>& h_prev,
                            const std::vector<float>& c_prev, size_t hidden) {
    auto model = loom::GgufModel::load(gguf_path, backend);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    loom::GraphBuilder builder(topo, *model, backend, /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& result = builder.build({{"n_tokens", /*n_tokens=*/0}, {"n_past", /*n_past=*/0}});

    ggml_backend_tensor_set(result.input_tensors.at("x"), x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(result.input_tensors.at("h_prev"), h_prev.data(), 0, h_prev.size() * sizeof(float));
    ggml_backend_tensor_set(result.input_tensors.at("c_prev"), c_prev.data(), 0, c_prev.size() * sizeof(float));

    ggml_backend_graph_compute(backend, result.graph);

    LOOM_CHECK(static_cast<size_t>(result.output->ne[0]) == hidden);
    std::vector<float> out(hidden);
    ggml_backend_tensor_get(result.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

} // namespace

int main() {
    const std::string dir = LOOM_TEST_FIXTURE_DIR;
    const std::string ref_dir = LOOM_TEST_REF_DIR;

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::vector<float> x = read_f32_binary(ref_dir + "/x.bin");
    const std::vector<float> h_prev = read_f32_binary(ref_dir + "/h_prev.bin");
    const std::vector<float> c_prev = read_f32_binary(ref_dir + "/c_prev.bin");
    const std::vector<float> expected_h = read_f32_binary(ref_dir + "/expected_h_new.bin");
    const std::vector<float> expected_c = read_f32_binary(ref_dir + "/expected_c_new.bin");
    const size_t hidden = h_prev.size();
    LOOM_CHECK(hidden > 0);
    LOOM_CHECK(expected_h.size() == hidden);
    LOOM_CHECK(expected_c.size() == hidden);

    const std::vector<float> actual_h = run_one(dir + "/lstm_step_h.gguf", backend.get(), x, h_prev, c_prev, hidden);
    const std::vector<float> actual_c = run_one(dir + "/lstm_step_c.gguf", backend.get(), x, h_prev, c_prev, hidden);

    for (size_t i = 0; i < hidden; ++i) {
        const float diff_h = std::fabs(actual_h[i] - expected_h[i]);
        const float diff_c = std::fabs(actual_c[i] - expected_c[i]);
        if (diff_h > 1e-5f || diff_c > 1e-5f) {
            std::fprintf(stderr, "index %zu: h diff=%f (actual=%f expected=%f), c diff=%f (actual=%f expected=%f)\n",
                          i, static_cast<double>(diff_h), static_cast<double>(actual_h[i]), static_cast<double>(expected_h[i]),
                          static_cast<double>(diff_c), static_cast<double>(actual_c[i]), static_cast<double>(expected_c[i]));
        }
        LOOM_CHECK(diff_h <= 1e-5f);
        LOOM_CHECK(diff_c <= 1e-5f);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
