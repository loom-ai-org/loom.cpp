// Numerical-correctness check for Kokoro's FULL `ProsodyPredictor.F0Ntrain` assembly: shared BiLSTM
// (640->512, loom::BiLstmStepper) -> two independent 3-block AdainResBlk1d stacks (F0: 512->512->256->256,
// N: same shape, plain per-block GraphBuilder::build() calls -- AdainResBlk1d has no recurrence at all) ->
// F0_proj/N_proj (plain Conv1d(256,1,kernel=1)), against a hand-rolled pure-PyTorch reference
// (reference_forward_kokoro_f0ntrain.py). Fully deterministic, plain exact-match check. Skips cleanly if
// the GGUF/reference files aren't present.

#include "test_util.h"

#include "loom/loom.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
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

// Runs one AdainResBlk1d block topology (already-built GGUF, no recurrence) over the whole [T,dim_in]
// sequence in one graph call. x_tc: T-major flat [T,dim_in] (matching CONV_1D's own [T,C] convention --
// same layout used directly as ggml tensor data, no reordering needed, unlike the channel-first
// AdaLayerNorm sites elsewhere in this project).
std::vector<float> run_block(const std::string& gguf_path, ggml_backend_t backend,
                              const std::vector<float>& x_tc, const std::vector<float>& style, uint32_t T,
                              uint32_t& T_out, uint32_t& dim_out) {
    auto model = loom::GgufModel::load(gguf_path, backend);
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend, nullptr);
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", T}, {"n_past", 0}});
    ggml_backend_tensor_set(r.input_tensors.at("x"), x_tc.data(), 0, x_tc.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("style"), style.data(), 0, style.size() * sizeof(float));
    ggml_backend_graph_compute(backend, r.graph);
    T_out = static_cast<uint32_t>(r.output->ne[0]);
    dim_out = static_cast<uint32_t>(r.output->ne[1]);
    std::vector<float> out(static_cast<size_t>(T_out) * dim_out);
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    return out; // ggml ne=[T_out,dim_out] (T_out fastest) -- byte-identical to numpy (dim_out,T_out)
}

// Runs a stack of 3 AdainResBlk1d blocks (F0.0/1/2 or N.0/1/2) given the shared BiLSTM output.
// shared_tc: T-major flat [T,512]. Returns [T_final,256] (T_final = 2*T from the middle upsampling block).
std::vector<float> run_stack(const std::string& dir, const std::string& name_prefix,
                              ggml_backend_t backend, const std::vector<float>& shared_tc, uint32_t T,
                              const std::vector<float>& style, uint32_t& T_final) {
    std::vector<float> x = shared_tc;
    uint32_t cur_T = T;
    uint32_t cur_dim = 0;
    for (int i = 0; i < 3; ++i) {
        const std::string path = dir + "/" + name_prefix + "_block" + std::to_string(i) + ".gguf";
        uint32_t out_T = 0, out_dim = 0;
        x = run_block(path, backend, x, style, cur_T, out_T, out_dim);
        cur_T = out_T;
        cur_dim = out_dim;
    }
    (void)cur_dim;
    T_final = cur_T;
    return x;
}

std::vector<float> run_proj(const std::string& gguf_path, ggml_backend_t backend,
                             const std::vector<float>& x_tc, uint32_t T) {
    auto model = loom::GgufModel::load(gguf_path, backend);
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());
    loom::GraphBuilder builder(topo, *model, backend, nullptr);
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", T}, {"n_past", 0}});
    ggml_backend_tensor_set(r.input_tensors.at("x"), x_tc.data(), 0, x_tc.size() * sizeof(float));
    ggml_backend_graph_compute(backend, r.graph);
    LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == T);
    std::vector<float> out(T);
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_KOKORO_DIR");
    const char* ref_dir_env = std::getenv("LOOM_KOKORO_F0N_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_f0n_*.gguf) and "
                              "LOOM_KOKORO_F0N_REF_DIR (ref_f0ntrain_*.npy) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kEnDim = 640;
    constexpr uint32_t kStyleDim = 128;
    constexpr uint32_t kHiddenPerDir = 256;

    std::vector<int64_t> en_shape, style_shape, f0_shape, n_shape;
    std::vector<float> en = read_npy_f32(ref_dir + "/ref_f0ntrain_en.npy", en_shape); // (640, T), row-major
    std::vector<float> style = read_npy_f32(ref_dir + "/ref_f0ntrain_style.npy", style_shape);
    std::vector<float> ref_F0 = read_npy_f32(ref_dir + "/ref_f0ntrain_F0.npy", f0_shape);
    std::vector<float> ref_N = read_npy_f32(ref_dir + "/ref_f0ntrain_N.npy", n_shape);
    LOOM_CHECK(en_shape.size() == 2 && static_cast<uint32_t>(en_shape[0]) == kEnDim);
    LOOM_CHECK(style_shape.size() == 1 && static_cast<uint32_t>(style_shape[0]) == kStyleDim);
    const auto T = static_cast<uint32_t>(en_shape[1]);
    const auto T_out = static_cast<uint32_t>(f0_shape[0]);
    LOOM_CHECK(T_out == 2 * T);
    LOOM_CHECK(static_cast<uint32_t>(n_shape[0]) == T_out);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // en is (640,T) row-major (channels outer, T inner) -- same "ggml ne=[a,b] <-> numpy (b,a)" rule as
    // everywhere else: BiLstmStepper::run's own convention wants T-major vector-of-vectors, so transpose.
    std::vector<std::vector<float>> x(T, std::vector<float>(kEnDim));
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < kEnDim; ++c) x[t][c] = en[static_cast<size_t>(c) * T + t];

    // --- shared BiLSTM (640 -> 512) ---
    std::vector<std::vector<float>> shared_out;
    {
        const std::string lp = dir + "/kokoro_f0n_shared_lstm";
        auto lstm_model = loom::GgufModel::load(lp + "_h_fwd.gguf", backend.get());
        LOOM_CHECK(lstm_model != nullptr);
        auto fwd_h = loom::GgufModel::load(lp + "_h_fwd.gguf", backend.get());
        auto fwd_c = loom::GgufModel::load(lp + "_c_fwd.gguf", backend.get());
        auto bwd_h = loom::GgufModel::load(lp + "_h_bwd.gguf", backend.get());
        auto bwd_c = loom::GgufModel::load(lp + "_c_bwd.gguf", backend.get());
        loom::GraphTopology fwd_h_topo = loom::GraphTopology::parse(fwd_h->topology_json());
        loom::GraphTopology fwd_c_topo = loom::GraphTopology::parse(fwd_c->topology_json());
        loom::GraphTopology bwd_h_topo = loom::GraphTopology::parse(bwd_h->topology_json());
        loom::GraphTopology bwd_c_topo = loom::GraphTopology::parse(bwd_c->topology_json());
        loom::BiLstmStepper stepper(*lstm_model, std::move(fwd_h_topo), std::move(fwd_c_topo),
                                     std::move(bwd_h_topo), std::move(bwd_c_topo), backend.get(), kHiddenPerDir);
        shared_out = stepper.run(x); // T x 512
    }

    // shared_out is a T-major vector-of-vectors (BiLstmStepper's own host-side convention, shared_out[t][c]).
    // The AdainResBlk1d blocks' "x" input is ggml ne=[T,512] (T FASTEST, i.e. flat index c*T+t) -- the
    // same convention test_e2e_kokoro_f0_block0.cpp/f0_block1.cpp feed their own "x" in (a real transpose,
    // not a no-op flatten, since T is fastest in ggml's layout but slowest in the host vector-of-vectors).
    std::vector<float> shared_tc(static_cast<size_t>(T) * 512);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < 512; ++c) shared_tc[static_cast<size_t>(c) * T + t] = shared_out[t][c];

    uint32_t f0_T_final = 0, n_T_final = 0;
    std::vector<float> f0_feat = run_stack(dir, "kokoro_f0n_f0", backend.get(), shared_tc, T, style, f0_T_final);
    std::vector<float> n_feat = run_stack(dir, "kokoro_f0n_n", backend.get(), shared_tc, T, style, n_T_final);
    LOOM_CHECK(f0_T_final == T_out);
    LOOM_CHECK(n_T_final == T_out);

    std::vector<float> F0 = run_proj(dir + "/kokoro_f0n_f0_proj.gguf", backend.get(), f0_feat, T_out);
    std::vector<float> N = run_proj(dir + "/kokoro_f0n_n_proj.gguf", backend.get(), n_feat, T_out);

    double max_abs_diff = 0.0, sum_abs_diff = 0.0;
    for (uint32_t t = 0; t < T_out; ++t) {
        const double d_f0 = std::fabs(static_cast<double>(F0[t]) - static_cast<double>(ref_F0[t]));
        const double d_n = std::fabs(static_cast<double>(N[t]) - static_cast<double>(ref_N[t]));
        max_abs_diff = std::max(max_abs_diff, d_f0);
        sum_abs_diff += d_f0;
        max_abs_diff = std::max(max_abs_diff, d_n);
        sum_abs_diff += d_n;
    }
    const double mean_abs_diff = sum_abs_diff / static_cast<double>(2 * T_out);
    std::fprintf(stderr, "T=%u, T_out=%u, mean_abs_diff=%g, max_abs_diff=%g\n", T, T_out, mean_abs_diff, max_abs_diff);
    LOOM_CHECK(mean_abs_diff < 1e-4);
    LOOM_CHECK(max_abs_diff < 1e-2);

    LOOM_TEST_REPORT_AND_RETURN();
}
