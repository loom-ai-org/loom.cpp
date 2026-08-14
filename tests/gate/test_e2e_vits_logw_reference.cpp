// Numerical-correctness check for the `logw` (TextEncoder + StochasticDurationPredictor reverse)
// topology against a real PyTorch reference: SDP is genuinely stochastic (`z = torch.randn(...)` baked
// into the real model itself), so this test feeds a FIXED, externally-injected noise array (produced by
// tools/convert_piper_vits/reference_forward_vits.py, which monkeypatches `torch.randn` for the exact
// shape SDP's own forward calls it with) into both the real reference and this topology's own `z_noise`
// declared input, making this a fully deterministic, bit-comparable check -- isolates the SDP's real
// spline-flow assembly (ConvFlow/DDSConv/ElementwiseAffine/Flip wiring in convert_vits.py's
// build_sdp_reverse) from any RNG-related uncertainty in the full stochastic VitsDriver::synthesize()
// path. Uses the same real phonemized text (T=62) as test_e2e_vits_stats_reference.cpp, exercising the
// real emb_rel_k/v pad branch. Skips cleanly if the reference files or LOOM_VITS_DIR aren't present.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Same minimal .npy reader as the other VITS reference tests (duplicated per this codebase's usual
// per-translation-unit small-helper convention).
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

// Same port as vits_driver.cpp's own `pad_crop_relative_embeddings` (duplicated here -- test-only
// plumbing, matching this codebase's usual "small helper duplicated per translation unit" convention).
std::vector<float> pad_crop_relative_embeddings(const std::vector<float>& raw, int64_t window_size,
                                                 int64_t k_channels, int64_t length) {
    const int64_t table_len = 2 * window_size + 1;
    const int64_t pad_length = std::max<int64_t>(length - (window_size + 1), 0);
    const int64_t padded_len = table_len + 2 * pad_length;
    std::vector<float> padded(static_cast<size_t>(padded_len * k_channels), 0.0f);
    for (int64_t row = 0; row < table_len; ++row) {
        std::copy(raw.begin() + row * k_channels, raw.begin() + (row + 1) * k_channels,
                  padded.begin() + (row + pad_length) * k_channels);
    }
    const int64_t start = std::max<int64_t>((window_size + 1) - length, 0);
    const int64_t out_len = 2 * length - 1;
    std::vector<float> out(static_cast<size_t>(out_len * k_channels));
    std::copy(padded.begin() + start * k_channels, padded.begin() + (start + out_len) * k_channels, out.begin());
    return out;
}

} // namespace

int main() {
    const char* dir_env = loom_test::fixture_env("LOOM_VITS_DIR");
    const char* ref_dir_env = loom_test::fixture_env("LOOM_VITS_LOGW_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_VITS_DIR (vits_logw.gguf) and LOOM_VITS_LOGW_REF_DIR "
                              "(ref_token_ids.json/ref_sdp_z_noise.npy/ref_sdp_logw.npy, produced by "
                              "reference_forward_vits.py) to run this numerical check\n");
        return 77;
    }
    const std::string dir = dir_env;
    const std::string ref_dir = ref_dir_env;

    const std::string tokens_json_str = read_file(ref_dir + "/ref_token_ids.json");
    if (tokens_json_str.empty()) {
        std::fprintf(stderr, "skipping: %s/ref_token_ids.json not found\n", ref_dir.c_str());
        return 77;
    }
    const std::vector<int32_t> token_ids = nlohmann::json::parse(tokens_json_str).get<std::vector<int32_t>>();
    const auto T = static_cast<uint32_t>(token_ids.size());

    std::vector<int64_t> zn_shape, logw_shape;
    // ref_sdp_z_noise.npy is saved from PyTorch's own (2, T) tensor (batch dim dropped, no transpose)
    // -- numpy row-major (T-fastest) is byte-identical to ggml's ne=[T,2] convention.
    std::vector<float> z_noise = read_npy_f32(ref_dir + "/ref_sdp_z_noise.npy", zn_shape);
    std::vector<float> ref_logw = read_npy_f32(ref_dir + "/ref_sdp_logw.npy", logw_shape);
    LOOM_CHECK(zn_shape.size() == 2 && zn_shape[0] == 2 && zn_shape[1] == static_cast<int64_t>(T));

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(dir + "/vits_logw.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    loom::GraphTopology topo = loom::GraphTopology::parse(model->topology_json());

    constexpr int64_t kWindowSize = 4;
    constexpr int64_t kHiddenChannels = 192;
    constexpr int64_t kNHeads = 2;
    constexpr int64_t kKChannels = kHiddenChannels / kNHeads;
    constexpr uint32_t kNTextLayers = 6;

    loom::GraphBuilder builder(topo, *model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", T}, {"n_past", 0}});

    std::vector<int32_t> tokens_copy = token_ids;
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));
    std::vector<float> mask(static_cast<size_t>(T) * T, 0.0f);
    ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), mask.data(), 0, mask.size() * sizeof(float));
    for (uint32_t i = 0; i < kNTextLayers; ++i) {
        const std::string prefix = "enc_p.encoder.attn_layers." + std::to_string(i);
        for (const char* which : {"emb_rel_k", "emb_rel_v"}) {
            ggml_tensor* raw_t = model->weight(prefix + "." + which + "_raw");
            std::vector<float> raw(static_cast<size_t>(ggml_nelements(raw_t)));
            ggml_backend_tensor_get(raw_t, raw.data(), 0, raw.size() * sizeof(float));
            std::vector<float> table = pad_crop_relative_embeddings(raw, kWindowSize, kKChannels, T);
            ggml_backend_tensor_set(r.input_tensors.at(std::string(which) + "_" + std::to_string(i)), table.data(), 0,
                                     table.size() * sizeof(float));
        }
    }
    ggml_backend_tensor_set(r.input_tensors.at("z_noise"), z_noise.data(), 0, z_noise.size() * sizeof(float));

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> logw(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, logw.data(), 0, logw.size() * sizeof(float));
    LOOM_CHECK(logw.size() == T);

    double max_abs_diff = 0.0;
    for (uint32_t t = 0; t < T; ++t) {
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(logw[t] - ref_logw[t])));
    }
    std::fprintf(stderr, "T=%u, max_abs_diff logw=%g\n", T, max_abs_diff);
    LOOM_CHECK(max_abs_diff < 1e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
