// Numerical-correctness check for Matcha-TTS's TextEncoder conversion (`matcha_encoder_mu.gguf` +
// `matcha_encoder_logw.gguf`, from tools/convert_matcha/convert_matcha_text_encoder.py) against the
// REAL `matcha.models.components.text_encoder.TextEncoder` module run directly
// (reference_forward_matcha_text_encoder.py) -- exercises the ConvReluNorm prenet, the partial-rotary
// (NeoX mode=2, n_dims=48 of 96 k_channels) integer-position RoPE self-attention stack (6 layers), and
// the per-token DurationPredictor, all against real checkpoint weights. Skips cleanly if the GGUF/
// reference files aren't present.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Same minimal .npy reader used throughout this codebase's other reference tests (duplicated per
// translation unit, matching the established convention).
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

// Runs one topology (mu or logw) against the given tokens; returns the flattened output.
std::vector<float> run_topology(const std::string& gguf_path, const std::vector<int32_t>& tokens) {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    const auto T = static_cast<uint32_t>(tokens.size());
    loom::GraphBuilder builder(topo, *model, backend.get());
    loom::GraphBuilder::BuildResult r = builder.build(T, 0);

    std::vector<int32_t> tokens_copy = tokens;
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));

    std::vector<int32_t> positions(T);
    std::iota(positions.begin(), positions.end(), 0);
    ggml_backend_tensor_set(r.input_tensors.at("positions"), positions.data(), 0, positions.size() * sizeof(int32_t));

    std::vector<float> mask(static_cast<size_t>(T) * T, 0.0f);
    ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_MATCHA_DIR");
    const char* ref_dir_env = std::getenv("LOOM_MATCHA_TEXT_ENCODER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_DIR (matcha_encoder_{mu,logw}.gguf) and "
                              "LOOM_MATCHA_TEXT_ENCODER_REF_DIR (ref_text_encoder_*.npy, produced by "
                              "reference_forward_matcha_text_encoder.py) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/ref_text_encoder_tokens.npy");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/ref_text_encoder_tokens.npy not found\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    std::vector<int64_t> tok_shape, mu_shape, logw_shape;
    std::vector<int32_t> tokens = read_npy_i32(ref_dir + "/ref_text_encoder_tokens.npy", tok_shape);
    std::vector<float> ref_mu = read_npy_f32(ref_dir + "/ref_text_encoder_mu.npy", mu_shape);
    std::vector<float> ref_logw = read_npy_f32(ref_dir + "/ref_text_encoder_logw.npy", logw_shape);
    const auto T = static_cast<uint32_t>(tokens.size());
    LOOM_CHECK(mu_shape.size() == 2 && mu_shape[0] == 80 && mu_shape[1] == static_cast<int64_t>(T));
    LOOM_CHECK(logw_shape.size() == 1 && logw_shape[0] == static_cast<int64_t>(T));

    std::vector<float> mu = run_topology(dir + "/matcha_encoder_mu.gguf", tokens);
    std::vector<float> logw = run_topology(dir + "/matcha_encoder_logw.gguf", tokens);
    LOOM_CHECK(mu.size() == ref_mu.size());
    LOOM_CHECK(logw.size() == ref_logw.size());

    // mu: ggml ne=[80,T] (channel-first, C=ne[0] fastest) -- flat index = t*80+c. Reference .npy is
    // (80,T) row-major (C-slow, T-fast) -- flat index = c*T+t. Same axis-order mismatch as VITS's own
    // "stats" comparison in test_e2e_vits_stats_reference.cpp.
    constexpr int64_t kNFeats = 80;
    double max_abs_diff_mu = 0.0;
    for (uint32_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < kNFeats; ++c) {
            const float got = mu[static_cast<size_t>(t) * kNFeats + c];
            const float exp = ref_mu[static_cast<size_t>(c) * T + t];
            max_abs_diff_mu = std::max(max_abs_diff_mu, static_cast<double>(std::fabs(got - exp)));
        }
    }
    // logw: ggml ne=[1,T] (single channel) -- flat index = t, matching the reference's (T,) layout
    // directly, no reindexing needed.
    double max_abs_diff_logw = 0.0;
    for (size_t i = 0; i < logw.size(); ++i) {
        max_abs_diff_logw = std::max(max_abs_diff_logw, static_cast<double>(std::fabs(logw[i] - ref_logw[i])));
    }
    std::fprintf(stderr, "T=%u, max_abs_diff mu=%g, logw=%g\n", T, max_abs_diff_mu, max_abs_diff_logw);
    LOOM_CHECK(max_abs_diff_mu < 1e-3);
    LOOM_CHECK(max_abs_diff_logw < 1e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
