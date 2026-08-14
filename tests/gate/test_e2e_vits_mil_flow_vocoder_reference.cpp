// Numerical-correctness check for the MIL-traced flow+vocoder topology (export_vits_mil.py) against a
// real PyTorch reference, at a REALISTIC z_p scale/length (T=194, values up to +-24) rather than the
// existing bespoke test's own small-scale Tp=8/std=0.5 one
// (tools/convert_piper_vits/reference_forward_vits.py / test_e2e_vits_flow_vocoder_reference.cpp).
//
// This wider-range fixture (reference_forward_vits_widerange.py) exists because it caught a REAL bug in
// the bespoke ggml topology: vits_flow_vocoder.gguf (convert_vits.py's hand-built
// RESIDUAL_COUPLING_LAYER_REVERSE/HiFi-GAN composition) diverges from the real PyTorch
// ResidualCouplingBlock+Generator by ~0.22 absolute (against a ~0.01-0.02 rms signal) on this exact z_p,
// while matching to ~1e-6 on the small-scale Tp=8 case its own test has ever exercised -- i.e. a latent
// bug never caught because the bespoke path's own numerical verification never used a realistic z_p
// range. THIS topology (traced from the real PyTorch ResidualCouplingBlock/Generator submodules directly,
// not hand-derived) matches the same real z_p to ~1e-6. See BACKLOG.md's VITS MIL-export entry.

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
    const char* dir_env = loom_test::fixture_env("LOOM_VITS_MIL_DIR");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_VITS_FLOW_VOCODER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_VITS_MIL_DIR (vits_mil.gguf) and "
                              "LOOM_VITS_FLOW_VOCODER_REF_DIR (ref_z_p_wide.npy/ref_wav_wide.npy, "
                              "produced by reference_forward_vits_widerange.py) to run this check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> z_p_shape, wav_shape;
    // Same [C, Tp] (Tp fastest) numpy-row-major-== ggml-ne[Tp,C] convention as
    // test_e2e_vits_flow_vocoder_reference.cpp's own z_p -- no transpose.
    std::vector<float> z_p = read_npy_f32(ref_dir + "/ref_z_p_wide.npy", z_p_shape);
    std::vector<float> ref_wav = read_npy_f32(ref_dir + "/ref_wav_wide.npy", wav_shape);
    if (z_p.empty()) {
        std::fprintf(stderr, "skipping: %s/ref_z_p_wide.npy not found\n", ref_dir.c_str());
        return 77;
    }
    LOOM_CHECK(z_p_shape.size() == 2 && z_p_shape[0] == 192);
    const auto Tp = static_cast<uint32_t>(z_p_shape[1]);

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/vits_mil.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json("flow_vocoder"));

    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", Tp}, {"n_past", 0}});
    ggml_backend_tensor_set(r.input_tensors.at("z_p"), z_p.data(), 0, z_p.size() * sizeof(float));
    LOOM_CHECK(r.output->ne[0] == static_cast<int64_t>(ref_wav.size()));
    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> wav(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, wav.data(), 0, wav.size() * sizeof(float));

    double max_abs_diff = 0.0;
    for (size_t i = 0; i < wav.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(wav[i] - ref_wav[i])));
    }
    std::fprintf(stderr, "Tp=%u, wav samples=%zu, max_abs_diff=%g\n", Tp, wav.size(), max_abs_diff);
    LOOM_CHECK(max_abs_diff < 1e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
