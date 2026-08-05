// Numerical-correctness check for the MIL-traced Matcha-TTS "encoder_mu"/"encoder_logw" topologies
// (export_matcha_mil.py, part of matcha_mil.gguf) against the SAME real-module reference fixture the
// bespoke conversion's own test_e2e_matcha_text_encoder.cpp already uses
// (reference_forward_matcha_text_encoder.py) -- valid ground truth for BOTH conversions: the MIL
// wrapper's own `sequence_mask -> ones_like` simplification (export_matcha_mil.py's own module
// docstring) is mathematically identical to the real forward pass whenever x_lengths == T, which the
// reference fixture's own tokens (no padding, single utterance) always satisfy.
//
// Layout note: unlike the bespoke topology's own `mu` output (C-fast, ne=[n_feats,T]), the MIL-traced
// "encoder_mu" topology's `mu` output is T-fast (ne=[T,n_feats], matching the real module's own native
// torch (1,n_feats,T) layout untouched -- see export_matcha_mil.py's module docstring). The reference
// fixture's `ref_text_encoder_mu.npy` was saved as `mu.squeeze(0)` (numpy shape (n_feats,T), C-order),
// i.e. flat[c*T+t] -- EXACTLY the T-fast ne=[T,n_feats] byte layout already (numpy's row-major (C,T)
// storage and ggml's T-fast ne=[T,C] storage are byte-for-byte identical, just axis-order-labeled
// differently), so no reindexing is needed comparing the two flat arrays directly.
//
// Skips cleanly if the GGUF/reference files aren't present.

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

std::vector<float> run_topology(loom::GgufModel& model, ggml_backend_t backend, const std::string& name,
                                 const std::vector<int32_t>& tokens) {
    loom::GraphTopology topo = loom::GraphTopology::parse(model.topology_json(name));
    const auto T = static_cast<uint32_t>(tokens.size());
    loom::GraphBuilder builder(topo, model, backend);
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", T}, {"n_past", 0}});

    std::vector<int32_t> tokens_copy = tokens;
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));

    ggml_backend_graph_compute(backend, r.graph);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    LOOM_CHECK(a.size() == b.size());
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_MATCHA_MIL_GGUF");
    const char* ref_dir_env = std::getenv("LOOM_MATCHA_TEXT_ENCODER_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_MIL_GGUF (matcha_mil.gguf, produced by "
                              "export_matcha_mil.py) and LOOM_MATCHA_TEXT_ENCODER_REF_DIR "
                              "(ref_text_encoder_*.npy, produced by "
                              "reference_forward_matcha_text_encoder.py) to run this check\n");
        return 77;
    }
    const std::string gguf_path = gguf_env;
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

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);

    std::vector<float> mu = run_topology(*model, backend.get(), "encoder_mu", tokens);
    std::vector<float> logw = run_topology(*model, backend.get(), "encoder_logw", tokens);

    LOOM_CHECK(mu.size() == ref_mu.size());
    LOOM_CHECK(logw.size() == ref_logw.size());

    const double mu_diff = max_abs_diff(mu, ref_mu);
    const double logw_diff = max_abs_diff(logw, ref_logw);
    std::fprintf(stderr, "mu_max_abs_diff=%g, logw_max_abs_diff=%g\n", mu_diff, logw_diff);
    LOOM_CHECK(mu_diff < 1e-3);
    LOOM_CHECK(logw_diff < 1e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
