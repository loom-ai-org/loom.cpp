// Numerical-correctness check for Matcha-TTS's Decoder U-Net (`matcha_decoder.gguf`, the CFM
// `estimator`) against the REAL `matcha.models.components.decoder.Decoder` module run directly
// (reference_forward_matcha_decoder.py) -- exercises GROUP_NORM, the one real downsample/upsample,
// skip connections, SnakeBeta-FeedForward BasicTransformerBlocks, and the sinusoidal time embedding,
// all against real checkpoint weights on a small T=8 hand-crafted input. Skips cleanly if the GGUF/
// reference files aren't present.

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
    const char* dir_env = std::getenv("LOOM_MATCHA_DIR");
    const char* ref_dir_env = std::getenv("LOOM_MATCHA_DECODER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_DIR (matcha_decoder.gguf) and "
                              "LOOM_MATCHA_DECODER_REF_DIR (ref_decoder_*.npy, produced by "
                              "reference_forward_matcha_decoder.py) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/ref_decoder_x.npy");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/ref_decoder_x.npy not found\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    constexpr int64_t kNFeats = 80;
    std::vector<int64_t> x_shape, mu_shape, t_shape, ref_shape;
    std::vector<float> ref_x = read_npy_f32(ref_dir + "/ref_decoder_x.npy", x_shape);
    std::vector<float> ref_mu = read_npy_f32(ref_dir + "/ref_decoder_mu.npy", mu_shape);
    std::vector<float> ref_t = read_npy_f32(ref_dir + "/ref_decoder_t.npy", t_shape);
    std::vector<float> ref_dphi = read_npy_f32(ref_dir + "/ref_decoder_dphi_dt.npy", ref_shape);
    LOOM_CHECK(x_shape.size() == 2 && x_shape[0] == kNFeats);
    const auto T = static_cast<uint32_t>(x_shape[1]);
    LOOM_CHECK(T % 4 == 0);

    // ref_x/ref_mu are (80,T) row-major numpy (C-slow, T-fast) -- T being the fastest-varying axis
    // there already matches our ne=[T,C] convention's own flat order (T=ne[0], fastest) BYTE FOR BYTE,
    // no reindexing needed (unlike the "mu" TextEncoder OUTPUT comparison in
    // test_e2e_matcha_text_encoder.cpp, which reindexes because it's comparing two DIFFERENT tensors'
    // layouts, not feeding one directly as another's input).
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/matcha_decoder.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    loom::GraphBuilder builder(topo, *model, backend.get());
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", T}, {"n_past", 0}});

    ggml_backend_tensor_set(r.input_tensors.at("z"), ref_x.data(), 0, ref_x.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("mu"), ref_mu.data(), 0, ref_mu.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("t"), ref_t.data(), 0, ref_t.size() * sizeof(float));

    std::vector<float> mask_full(static_cast<size_t>(T) * T, 0.0f);
    ggml_backend_tensor_set(r.input_tensors.at("attn_mask_full"), mask_full.data(), 0, mask_full.size() * sizeof(float));
    const uint32_t T_half = T / 2;
    std::vector<float> mask_half(static_cast<size_t>(T_half) * T_half, 0.0f);
    ggml_backend_tensor_set(r.input_tensors.at("attn_mask_half"), mask_half.data(), 0, mask_half.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> dphi(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, dphi.data(), 0, dphi.size() * sizeof(float));
    LOOM_CHECK(dphi.size() == static_cast<size_t>(kNFeats) * T);

    // dphi: ggml ne=[T,C] (T=ne[0], fastest) -- flat index = t + c*T. ref_dphi: numpy (C,T) row-major
    // -- flat index = c*T + t. Addition commutes: these are the SAME formula, so the two flat buffers
    // are already byte-identical in layout -- a direct element-by-element compare, no permutation.
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < dphi.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(dphi[i] - ref_dphi[i])));
    }
    std::fprintf(stderr, "T=%u, max_abs_diff dphi_dt=%g\n", T, max_abs_diff);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
