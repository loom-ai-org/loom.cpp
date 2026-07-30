// Numerical-correctness check for Kokoro's ProsodyPredictor duration-prediction half: DurationEncoder
// (3x interleaved BiLSTM + AdaLayerNorm, via loom::BiLstmStepper + a small per-instance AdaLayerNorm
// topology) -> ProsodyPredictor.lstm (another BiLstmStepper) -> duration_proj (a plain Linear topology),
// against a hand-rolled pure-PyTorch reference (reference_forward_kokoro_duration_predictor.py). Style/
// channel concatenation is done in plain host C++ vector code (no temporal recurrence in it at all --
// see convert_kokoro_duration_predictor.py's own module docstring). Fully deterministic, plain
// exact-match check. Skips cleanly if the GGUF/reference files aren't present.

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

// Runs a standalone AdaLayerNorm topology over a whole [channels,T] sequence in one graph call (no
// recurrence at all -- see the conversion script's own module docstring for why this doesn't need
// host-stepping the way the BiLSTM layers do). seq_ct: channels outer, T inner (row-major, matching the
// engine's own channel-first ggml ne=[channels,T] convention read back flat).
std::vector<float> run_adaln(loom::GgufModel& model, loom::GraphTopology& topo, ggml_backend_t backend,
                              const std::vector<float>& seq_ct, uint32_t channels, uint32_t T,
                              const std::vector<float>& style) {
    loom::GraphBuilder builder(topo, model, backend, nullptr);
    loom::GraphBuilder::BuildResult r = builder.build({{"n_tokens", T}, {"n_past", 0}});
    ggml_backend_tensor_set(r.input_tensors.at("x"), seq_ct.data(), 0, seq_ct.size() * sizeof(float));
    ggml_backend_tensor_set(r.input_tensors.at("style"), style.data(), 0, style.size() * sizeof(float));
    ggml_backend_graph_compute(backend, r.graph);
    std::vector<float> out(static_cast<size_t>(channels) * T);
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    (void)channels;
    return out; // ggml ne=[channels,T] (channels fastest) -- channel-major flat layout
}

} // namespace

int main() {
    const char* dir_env = std::getenv("LOOM_KOKORO_DIR");
    const char* ref_dir_env = std::getenv("LOOM_KOKORO_DURATION_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_KOKORO_DIR (kokoro_duration_*.gguf) and "
                              "LOOM_KOKORO_DURATION_REF_DIR (ref_duration_*.npy) to run this numerical "
                              "check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    constexpr uint32_t kDModel = 512;
    constexpr uint32_t kStyleDim = 128;
    constexpr uint32_t kHiddenPerDir = 256;
    constexpr uint32_t kMaxDur = 50;

    std::vector<int64_t> den_shape, style_shape, logits_shape;
    std::vector<float> d_en = read_npy_f32(ref_dir + "/ref_duration_d_en.npy", den_shape);   // (d_model, T), row-major
    std::vector<float> style = read_npy_f32(ref_dir + "/ref_duration_style.npy", style_shape);
    std::vector<float> ref_logits = read_npy_f32(ref_dir + "/ref_duration_logits.npy", logits_shape); // (T, max_dur)
    LOOM_CHECK(den_shape.size() == 2 && static_cast<uint32_t>(den_shape[0]) == kDModel);
    LOOM_CHECK(style_shape.size() == 1 && static_cast<uint32_t>(style_shape[0]) == kStyleDim);
    const auto T = static_cast<uint32_t>(den_shape[1]);

    ggml_backend_ptr backend(ggml_backend_cpu_init());
    LOOM_CHECK(backend != nullptr);

    // x[t]: (d_model+style_dim) per-position vector, T-major host vector-of-vectors (matches
    // BiLstmStepper::run's own convention) -- built by concatenating d_en's own per-position slice with
    // the (broadcast, per-position-identical) style vector, in plain C++.
    std::vector<std::vector<float>> x(T, std::vector<float>(kDModel + kStyleDim));
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t c = 0; c < kDModel; ++c) x[t][c] = d_en[static_cast<size_t>(c) * T + t]; // (d_model,T) row-major -> [c][t]
        for (uint32_t s = 0; s < kStyleDim; ++s) x[t][kDModel + s] = style[s];
    }

    for (int i = 0; i < 3; ++i) {
        const std::string lp = dir + "/kokoro_duration_lstm_" + std::to_string(i);
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
        std::vector<std::vector<float>> lstm_out = stepper.run(x); // T x d_model (512)

        // ggml ne=[channels,T] has channels as the FASTEST axis (flat index = t*channels + c) --
        // byte-identical to a numpy/host array of NATIVE shape (T,channels), i.e. a plain row-major
        // flatten of lstm_out (T outer, c inner) needs NO reordering at all (same "ggml ne=[a,b] <->
        // numpy (b,a)" rule as every other model in this project -- a real bug here previously used
        // `c*T+t`, which is backwards, caught via a scratch diagnostic isolating this exact stage).
        std::vector<float> seq_ct(static_cast<size_t>(kDModel) * T);
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t c = 0; c < kDModel; ++c) seq_ct[static_cast<size_t>(t) * kDModel + c] = lstm_out[t][c];

        auto ada_model = loom::GgufModel::load(dir + "/kokoro_duration_adaln_" + std::to_string(i) + ".gguf", backend.get());
        LOOM_CHECK(ada_model != nullptr);
        loom::GraphTopology ada_topo = loom::GraphTopology::parse(ada_model->topology_json());
        std::vector<float> ada_out = run_adaln(*ada_model, ada_topo, backend.get(), seq_ct, kDModel, T, style);

        x.assign(T, std::vector<float>(kDModel + kStyleDim));
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t c = 0; c < kDModel; ++c) x[t][c] = ada_out[static_cast<size_t>(t) * kDModel + c];
            for (uint32_t s = 0; s < kStyleDim; ++s) x[t][kDModel + s] = style[s];
        }
    }

    // --- ProsodyPredictor's own top `lstm` ---
    {
        const std::string lp = dir + "/kokoro_duration_top_lstm";
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
        std::vector<std::vector<float>> top_out = stepper.run(x); // T x d_model (512)

        // --- duration_proj: plain Linear(512,50), applied per position ---
        auto proj_model = loom::GgufModel::load(dir + "/kokoro_duration_proj.gguf", backend.get());
        LOOM_CHECK(proj_model != nullptr);
        loom::GraphTopology proj_topo = loom::GraphTopology::parse(proj_model->topology_json());
        loom::GraphBuilder proj_builder(proj_topo, *proj_model, backend.get(), nullptr);

        double max_abs_diff = 0.0;
        double sum_abs_diff = 0.0;
        for (uint32_t t = 0; t < T; ++t) {
            loom::GraphBuilder::BuildResult r = proj_builder.build({{"n_tokens", 0}, {"n_past", 0}});
            ggml_backend_tensor_set(r.input_tensors.at("x"), top_out[t].data(), 0, top_out[t].size() * sizeof(float));
            ggml_backend_graph_compute(backend.get(), r.graph);
            LOOM_CHECK(static_cast<uint32_t>(ggml_nelements(r.output)) == kMaxDur);
            std::vector<float> logits(kMaxDur);
            ggml_backend_tensor_get(r.output, logits.data(), 0, logits.size() * sizeof(float));
            for (uint32_t k = 0; k < kMaxDur; ++k) {
                const double d = std::fabs(logits[k] - ref_logits[static_cast<size_t>(t) * kMaxDur + k]);
                max_abs_diff = std::max(max_abs_diff, d);
                sum_abs_diff += d;
            }
        }
        const double mean_abs_diff = sum_abs_diff / static_cast<double>(T * kMaxDur);
        std::fprintf(stderr, "T=%u, mean_abs_diff=%g, max_abs_diff=%g\n", T, mean_abs_diff, max_abs_diff);
        LOOM_CHECK(mean_abs_diff < 1e-4);
        LOOM_CHECK(max_abs_diff < 1e-2);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
