// Numerical-correctness check for Kokoro's TextEncoder: the CNN topology (tools/convert_kokoro/
// convert_kokoro_text_encoder.py) plus the new loom::BiLstmStepper host driver for the trailing
// bidirectional LSTM, against a hand-rolled pure-PyTorch reference
// (reference_forward_kokoro_text_encoder.py). Fully deterministic, plain exact-match check. Skips
// cleanly if the GGUF/reference files aren't present.

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
    const char* ref_dir_env = std::getenv("LOOM_KOKORO_TEXT_ENCODER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_text_encoder_*.gguf) and "
                              "LOOM_KOKORO_TEXT_ENCODER_REF_DIR (ref_text_encoder_*.npy) to run this "
                              "numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> tok_shape, out_shape;
    std::vector<int32_t> tokens = read_npy_i32(ref_dir + "/ref_text_encoder_tokens.npy", tok_shape);
    std::vector<float> ref_out = read_npy_f32(ref_dir + "/ref_text_encoder_out.npy", out_shape);
    LOOM_CHECK(tok_shape.size() == 1);
    LOOM_CHECK(out_shape.size() == 2); // (T, 2*hidden_per_dir)

    const auto n_tokens = static_cast<uint32_t>(tok_shape[0]);
    const auto out_dim = static_cast<uint32_t>(out_shape[1]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // --- CNN portion: embedding + 3x[conv+LN+leakyrelu], producing [T,C] ---
    auto cnn_model = loom::GgufModel::load(dir + "/kokoro_text_encoder_cnn.gguf", backend.get());
    LOOM_CHECK(cnn_model != nullptr);
    loom::GraphTopology cnn_topo = loom::GraphTopology::parse(cnn_model->topology_json());
    loom::GraphBuilder cnn_builder(cnn_topo, *cnn_model, backend.get(), /*kv_cache=*/nullptr);
    const loom::GraphBuilder::BuildResult& cnn_r = cnn_builder.build({{"n_tokens", n_tokens}, {"n_past", /*n_past=*/0}});
    std::vector<int32_t> tokens_copy = tokens;
    ggml_backend_tensor_set(cnn_r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));
    ggml_backend_graph_compute(backend.get(), cnn_r.graph);

    const auto channels = static_cast<uint32_t>(cnn_r.output->ne[1]);
    LOOM_CHECK(static_cast<uint32_t>(cnn_r.output->ne[0]) == n_tokens);
    std::vector<float> cnn_out_flat(static_cast<size_t>(n_tokens) * channels);
    ggml_backend_tensor_get(cnn_r.output, cnn_out_flat.data(), 0, cnn_out_flat.size() * sizeof(float));

    // cnn_r.output has ggml ne=[n_tokens, channels] -- n_tokens is the FASTEST axis, so the flat buffer
    // is channel-major (all n_tokens values for channel 0, then all for channel 1, ...), NOT
    // token-major. Extracting a per-token vector needs a strided read, not a contiguous slice.
    std::vector<std::vector<float>> cnn_out(n_tokens, std::vector<float>(channels));
    for (uint32_t c = 0; c < channels; ++c) {
        for (uint32_t t = 0; t < n_tokens; ++t) {
            cnn_out[t][c] = cnn_out_flat[static_cast<size_t>(c) * n_tokens + t];
        }
    }

    // --- bidirectional LSTM via loom::BiLstmStepper -- a single shared GgufModel resolves all four
    //     topologies' weight references, since the conversion script writes the FULL weight set (both
    //     directions) into every one of the 4 small GGUFs (matching TdtDecoder's own convention). ---
    auto lstm_model = loom::GgufModel::load(dir + "/kokoro_text_encoder_lstm_h_fwd.gguf", backend.get());
    LOOM_CHECK(lstm_model != nullptr);
    auto fwd_h_model = loom::GgufModel::load(dir + "/kokoro_text_encoder_lstm_h_fwd.gguf", backend.get());
    auto fwd_c_model = loom::GgufModel::load(dir + "/kokoro_text_encoder_lstm_c_fwd.gguf", backend.get());
    auto bwd_h_model = loom::GgufModel::load(dir + "/kokoro_text_encoder_lstm_h_bwd.gguf", backend.get());
    auto bwd_c_model = loom::GgufModel::load(dir + "/kokoro_text_encoder_lstm_c_bwd.gguf", backend.get());
    loom::GraphTopology fwd_h_topo = loom::GraphTopology::parse(fwd_h_model->topology_json());
    loom::GraphTopology fwd_c_topo = loom::GraphTopology::parse(fwd_c_model->topology_json());
    loom::GraphTopology bwd_h_topo = loom::GraphTopology::parse(bwd_h_model->topology_json());
    loom::GraphTopology bwd_c_topo = loom::GraphTopology::parse(bwd_c_model->topology_json());

    constexpr uint32_t kHiddenPerDir = 256;
    loom::BiLstmStepper stepper(*lstm_model, std::move(fwd_h_topo), std::move(fwd_c_topo),
                                 std::move(bwd_h_topo), std::move(bwd_c_topo), backend.get(), kHiddenPerDir);
    std::vector<std::vector<float>> out = stepper.run(cnn_out);

    LOOM_CHECK(out_dim == 2 * kHiddenPerDir);
    double max_abs_diff = 0.0;
    double sum_abs_diff = 0.0;
    size_t n = 0;
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t c = 0; c < out_dim; ++c) {
            const double d = std::fabs(out[t][c] - ref_out[static_cast<size_t>(t) * out_dim + c]);
            max_abs_diff = std::max(max_abs_diff, d);
            sum_abs_diff += d;
            ++n;
        }
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(n);
    std::fprintf(stderr, "n_tokens=%u, out_dim=%u, mean_abs_diff=%g, max_abs_diff=%g\n",
                 n_tokens, out_dim, mean_abs_diff, max_abs_diff);
    LOOM_CHECK(mean_abs_diff < 1e-4);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
