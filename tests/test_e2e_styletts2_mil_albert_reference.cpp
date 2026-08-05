// Numerical-correctness check for the MIL-traced StyleTTS2 "albert" topology (export_styletts2_mil.py's
// AlbertWrapper, part of styletts2_mil.gguf) against a hand-rolled pure-PyTorch reference
// (reference_forward_styletts2_albert_mil.py, which reuses tools/convert_kokoro's own already-verified
// `albert_forward` unmodified against StyleTTS2's own checkpoint weights -- see that script's own
// docstring for why this is an independent ground truth, not trusting AlbertWrapper's own code at all).
// Mirrors test_e2e_kokoro_mil_albert_bert_encoder_reference.cpp's own npy-reading/comparison shape, but
// against the StyleTTS2 "albert"-only topology (no bert_encoder fused in -- see AlbertWrapper's own
// docstring for why) and its (T,768) time-major output convention. Skips cleanly if the GGUF/reference
// files aren't present.

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

std::vector<int32_t> read_npy_i32(const std::string& path, std::vector<int64_t>& shape_out) {
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
    std::vector<int32_t> data(static_cast<size_t>(total));
    f.read(reinterpret_cast<char*>(data.data()), total * static_cast<int64_t>(sizeof(int32_t)));
    return data;
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_STYLETTS2_MIL_GGUF");
    const char* ref_dir_env = std::getenv("LOOM_STYLETTS2_MIL_ALBERT_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_STYLETTS2_MIL_GGUF (styletts2_mil.gguf, produced by "
                              "export_styletts2_mil.py) and LOOM_STYLETTS2_MIL_ALBERT_REF_DIR "
                              "(ref_styletts2_albert_*.npy, produced by "
                              "reference_forward_styletts2_albert_mil.py) to run this check\n");
        return 77;
    }
    const std::string gguf_path = gguf_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> tok_shape, out_shape;
    std::vector<int32_t> tokens = read_npy_i32(ref_dir + "/ref_styletts2_albert_tokens.npy", tok_shape);
    std::vector<float> ref_out = read_npy_f32(ref_dir + "/ref_styletts2_albert_out.npy", out_shape);
    LOOM_CHECK(tok_shape.size() == 1);
    LOOM_CHECK(out_shape.size() == 2); // (T,768), time-major -- see AlbertWrapper's own docstring

    const auto n_tokens = static_cast<uint32_t>(tok_shape[0]);
    const auto hidden_dim = static_cast<uint32_t>(out_shape[1]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("albert"));

    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", n_tokens}, {"n_past", /*n_past=*/0}});

    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens.data(), 0, tokens.size() * sizeof(int32_t));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    LOOM_CHECK(out.size() == ref_out.size());
    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[0]) == hidden_dim);
    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[1]) == n_tokens);

    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(out[i] - ref_out[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        sum_abs_diff += d;
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(out.size());
    std::fprintf(stderr, "n_tokens=%u, hidden_dim=%u, mean_abs_diff=%g, max_abs_diff=%g\n",
                 n_tokens, hidden_dim, mean_abs_diff, max_abs_diff);
    LOOM_CHECK(mean_abs_diff < 1e-4);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
