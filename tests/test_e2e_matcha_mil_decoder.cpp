// Numerical-correctness check for the MIL-traced Matcha-TTS "decoder" topology (export_matcha_mil.py,
// part of matcha_mil.gguf) against the SAME real-module reference fixture the bespoke conversion's own
// test_e2e_matcha_decoder.cpp already uses (reference_forward_matcha_decoder.py) -- valid ground truth
// for both: the MIL wrapper traces the real `Decoder.forward` unmodified except for an all-ones `mask`
// (built via `torch.ones_like`, not the real `sequence_mask`), and the reference fixture's own mask is
// already `torch.ones(1,1,T)` (T=8, no padding) -- mathematically identical.
//
// Layout: both z/mu (inputs) and dphi_dt (output) are T-fast (ne=[T,n_feats]), matching the reference
// fixture's own `x.squeeze(0)`/`dphi_dt.squeeze(0)` numpy (n_feats,T) C-order byte layout directly (see
// test_e2e_matcha_mil_text_encoder.cpp's own comment for why numpy (C,T) row-major and ggml T-fast
// ne=[T,C] are byte-for-byte identical). No MIL "encoder_mu" involved here -- this test only exercises
// the Decoder U-Net in isolation, same scope as the bespoke test it mirrors.
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

void set_input(loom::GraphBuilder::BuildResult& r, const std::string& name, const std::vector<float>& data) {
    ggml_tensor* t = r.input_tensors.at(name);
    LOOM_CHECK(static_cast<size_t>(ggml_nelements(t)) == data.size());
    std::vector<float> copy = data;
    ggml_backend_tensor_set(t, copy.data(), 0, copy.size() * sizeof(float));
}

} // namespace

int main() {
    const char* gguf_env = std::getenv("LOOM_MATCHA_MIL_GGUF");
    const char* ref_dir_env = std::getenv("LOOM_MATCHA_DECODER_REF_DIR");
    if (gguf_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_MATCHA_MIL_GGUF (matcha_mil.gguf, produced by "
                              "export_matcha_mil.py) and LOOM_MATCHA_DECODER_REF_DIR "
                              "(ref_decoder_*.npy, produced by reference_forward_matcha_decoder.py) to "
                              "run this check\n");
        return 77;
    }
    const std::string gguf_path = gguf_env;
    const std::string ref_dir = ref_dir_env;

    std::ifstream probe(ref_dir + "/ref_decoder_x.npy");
    if (!probe.good()) {
        std::fprintf(stderr, "skipping: %s/ref_decoder_x.npy not found\n", ref_dir.c_str());
        return 77;
    }
    probe.close();

    std::vector<int64_t> x_shape, mu_shape, t_shape, ref_shape;
    std::vector<float> x = read_npy_f32(ref_dir + "/ref_decoder_x.npy", x_shape);
    std::vector<float> mu = read_npy_f32(ref_dir + "/ref_decoder_mu.npy", mu_shape);
    std::vector<float> t_val = read_npy_f32(ref_dir + "/ref_decoder_t.npy", t_shape);
    std::vector<float> ref_dphi_dt = read_npy_f32(ref_dir + "/ref_decoder_dphi_dt.npy", ref_shape);

    LOOM_CHECK(x_shape.size() == 2); // (n_feats, T)
    const auto n_feats = static_cast<uint32_t>(x_shape[0]);
    const auto T = static_cast<uint32_t>(x_shape[1]);
    (void)n_feats;

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("decoder"));

    loom::GraphBuilder builder(topo, *model, backend.get());
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", T}, {"n_past", 0}});

    set_input(r, "z", x);
    set_input(r, "mu", mu);
    set_input(r, "t", t_val);

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    LOOM_CHECK(out.size() == ref_dphi_dt.size());

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < out.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(out[i] - ref_dphi_dt[i])));
    }
    std::fprintf(stderr, "T=%u, max_abs_diff=%g\n", T, max_abs_diff);
    LOOM_CHECK(max_abs_diff < 1e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
