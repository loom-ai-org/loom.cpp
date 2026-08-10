// Numerical-correctness check for the flow+vocoder topology against a real PyTorch reference: unlike
// the full VitsDriver::synthesize() path (stochastic -- SDP's z_noise and z_p's own noise sampling),
// this test feeds a FIXED, externally-supplied z_p (produced once by
// /tmp/.../vits_flow_vocoder_ref.py, a real ResidualCouplingBlock+Generator forward pass with real
// checkpoint weights) directly into the flow_vocoder topology, making this a fully deterministic,
// bit-comparable check -- isolates the coupling-flow+vocoder WIRING (convert_vits.py's
// build_flow_vocoder_topology) from any RNG-related uncertainty. Skips cleanly if the reference files
// or LOOM_VITS_DIR aren't present.

#include "test_util.h"
#include "fixtures.h"

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

// Minimal .npy reader (float32, C-contiguous) -- same as test_e2e_vits_stats_reference.cpp's own
// (duplicated per this codebase's usual per-translation-unit small-helper convention).
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
    const char* dir_env = loom_test::fixture_env("LOOM_VITS_DIR");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_VITS_FLOW_VOCODER_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_VITS_DIR (vits_flow_vocoder.gguf) and "
                              "LOOM_VITS_FLOW_VOCODER_REF_DIR (ref_z_p.npy/ref_wav.npy) to run this check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    std::vector<int64_t> z_p_shape, wav_shape;
    // ref_z_p.npy is saved from PyTorch's own (1, 192, Tp) tensor WITHOUT transposing -- numpy shape
    // (192, Tp), row-major (Tp fastest), which is byte-for-byte the same layout as ggml's own ne=[Tp,
    // 192] (T=ne[0], fastest) convention. Do NOT transpose when dumping a reference tensor to compare
    // against this engine's [T,C] flow/vocoder convention -- caught once already in this same test (see
    // BACKLOG.md): transposing here silently produces a "same values, permuted order" mismatch that
    // looks like a real numerical bug at first glance.
    std::vector<float> z_p = read_npy_f32(ref_dir + "/ref_z_p.npy", z_p_shape); // [192, Tp]
    std::vector<float> ref_wav = read_npy_f32(ref_dir + "/ref_wav.npy", wav_shape); // [Tp*256]
    if (z_p.empty()) {
        std::fprintf(stderr, "skipping: %s/ref_z_p.npy not found\n", ref_dir.c_str());
        return 77;
    }
    LOOM_CHECK(z_p_shape.size() == 2 && z_p_shape[0] == 192);
    const auto Tp = static_cast<uint32_t>(z_p_shape[1]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/vits_flow_vocoder.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

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
    std::fprintf(stderr, "got[:10]:");
    for (int i = 0; i < 10; ++i) std::fprintf(stderr, " %f", static_cast<double>(wav[i]));
    std::fprintf(stderr, "\nexp[:10]:");
    for (int i = 0; i < 10; ++i) std::fprintf(stderr, " %f", static_cast<double>(ref_wav[i]));
    std::fprintf(stderr, "\n");
    LOOM_CHECK(max_abs_diff < 1e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
