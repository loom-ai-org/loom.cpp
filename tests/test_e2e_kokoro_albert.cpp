// Numerical-correctness check for Kokoro's CustomAlbert (PL-BERT) topology (tools/convert_kokoro/
// convert_kokoro_albert.py) against a hand-rolled pure-PyTorch reference
// (reference_forward_kokoro_albert.py). Fully deterministic (no sampling anywhere in this piece), so
// this is a plain exact-match check -- the first, isolated verification step before assembling the rest
// of Kokoro (DurationEncoder/ProsodyPredictor/TextEncoder/Decoder) around this output. Skips cleanly if
// the GGUF/reference files aren't present.

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
    const char* dir_env = std::getenv("LOOM_KOKORO_DIR");
    const char* ref_dir_env = std::getenv("LOOM_KOKORO_ALBERT_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_albert.gguf) and "
                              "LOOM_KOKORO_ALBERT_REF_DIR (ref_albert_*.npy, produced by "
                              "reference_forward_kokoro_albert.py) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> tok_shape, out_shape;
    std::vector<int32_t> tokens = read_npy_i32(ref_dir + "/ref_albert_tokens.npy", tok_shape);
    std::vector<float> ref_out = read_npy_f32(ref_dir + "/ref_albert_out.npy", out_shape);
    LOOM_CHECK(tok_shape.size() == 1);
    LOOM_CHECK(out_shape.size() == 2); // (T, hidden_size), native PyTorch, byte-identical to ggml ne=[hidden_size,T]

    const auto n_tokens = static_cast<uint32_t>(tok_shape[0]);
    const auto hidden_size = static_cast<uint32_t>(out_shape[1]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/kokoro_albert.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    loom::GraphBuilder builder(topo, *model, backend.get(), /*kv_cache=*/nullptr);
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", n_tokens}, {"n_past", /*n_past=*/0}});

    std::vector<int32_t> tokens_copy = tokens;
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));
    std::vector<int32_t> positions(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) positions[i] = static_cast<int32_t>(i);
    ggml_backend_tensor_set(r.input_tensors.at("positions"), positions.data(), 0, positions.size() * sizeof(int32_t));
    std::vector<float> mask(static_cast<size_t>(n_tokens) * n_tokens, 0.0f); // no padding, single utterance
    ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    LOOM_CHECK(out.size() == ref_out.size());
    LOOM_CHECK(static_cast<uint32_t>(r.output->ne[0]) == hidden_size);

    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs(out[i] - ref_out[i]);
        max_abs_diff = std::max(max_abs_diff, d);
        sum_abs_diff += d;
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(out.size());
    std::fprintf(stderr, "n_tokens=%u, hidden_size=%u, mean_abs_diff=%g, max_abs_diff=%g\n",
                 n_tokens, hidden_size, mean_abs_diff, max_abs_diff);
    LOOM_CHECK(mean_abs_diff < 1e-4);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
