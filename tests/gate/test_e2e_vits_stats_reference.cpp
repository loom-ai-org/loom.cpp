// Numerical-correctness check for the `stats` (TextEncoder) topology against a real PyTorch reference:
// unlike StochasticDurationPredictor/the coupling flow (both genuinely stochastic -- randn() sampling
// baked into the real model itself), TextEncoder is fully deterministic, so its output can be compared
// exactly against a real forward pass of piper's own `models.TextEncoder`, loaded directly from the real
// checkpoint's `enc_p.*` weights (see /tmp/.../vits_real_text_stats_ref.py, run once to produce the
// reference .npy files this test reads). Uses REAL phonemized text (via piper_phonemize + the real
// phoneme_id_map's BOS/blank-interleave/EOS convention -- not arbitrary token ids), giving T=62, which
// exercises the real emb_rel_k/v PAD branch (T=62 > window_size+1=5) that every earlier, smaller-scale
// test in this VITS effort sidestepped. Skips cleanly if the reference files or LOOM_VITS_DIR aren't
// present.

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

// Minimal .npy reader (float32, C-contiguous) -- enough to read back the two small reference arrays
// this test needs, without adding a new project-wide dependency for one-off numeric verification.
std::vector<float> read_npy_f32(const std::string& path, std::vector<int64_t>& shape_out) {
    std::ifstream f(path, std::ios::binary);
    LOOM_CHECK(static_cast<bool>(f));
    char magic[6];
    f.read(magic, 6);
    f.ignore(2); // version
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    // Parse "'shape': (a, b, c)," out of the header dict text.
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

// Same algorithm as vits_driver.cpp's own `pad_crop_relative_embeddings` (duplicated here rather than
// exposed from VitsDriver's public header -- this is test-only plumbing, matching this codebase's usual
// "small helper duplicated per translation unit" convention rather than growing the driver's public
// surface for a test's sake).
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
    const char* ref_dir_env = loom_test::fixture_env("LOOM_VITS_STATS_REF_DIR");
    if (dir_env == nullptr || ref_dir_env == nullptr) {
        std::fprintf(stderr, "skipping: set LOOM_VITS_DIR (vits_stats.gguf) and LOOM_VITS_STATS_REF_DIR "
                              "(ref_token_ids.json/ref_m_p.npy/ref_logs_p.npy, produced by the real-"
                              "TextEncoder reference script) to run this numerical check\n");
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

    std::vector<int64_t> m_p_shape, logs_p_shape;
    std::vector<float> ref_m_p = read_npy_f32(ref_dir + "/ref_m_p.npy", m_p_shape); // (1, 192, T), C-slow/T-fast
    std::vector<float> ref_logs_p = read_npy_f32(ref_dir + "/ref_logs_p.npy", logs_p_shape);
    LOOM_CHECK(m_p_shape.size() == 3 && m_p_shape[0] == 1 && m_p_shape[1] == 192 && m_p_shape[2] == static_cast<int64_t>(T));

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto stats_model = loom::GgufModel::load(dir + "/vits_stats.gguf", backend.get());
    LOOM_CHECK(stats_model != nullptr);
    loom::GraphTopology stats_topo = loom::GraphTopology::parse(stats_model->topology_json());

    constexpr int64_t kWindowSize = 4;
    constexpr int64_t kHiddenChannels = 192;
    constexpr int64_t kNHeads = 2;
    constexpr int64_t kKChannels = kHiddenChannels / kNHeads;
    constexpr uint32_t kNTextLayers = 6;

    loom::GraphBuilder builder(stats_topo, *stats_model, backend.get());
    const loom::GraphBuilder::BuildResult& r = builder.build({{"n_tokens", T}, {"n_past", 0}});

    std::vector<int32_t> tokens_copy = token_ids;
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));
    std::vector<float> mask(static_cast<size_t>(T) * T, 0.0f);
    ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), mask.data(), 0, mask.size() * sizeof(float));
    for (uint32_t i = 0; i < kNTextLayers; ++i) {
        const std::string prefix = "enc_p.encoder.attn_layers." + std::to_string(i);
        for (const char* which : {"emb_rel_k", "emb_rel_v"}) {
            ggml_tensor* raw_t = stats_model->weight(prefix + "." + which + "_raw");
            std::vector<float> raw(static_cast<size_t>(ggml_nelements(raw_t)));
            ggml_backend_tensor_get(raw_t, raw.data(), 0, raw.size() * sizeof(float));
            std::vector<float> table = pad_crop_relative_embeddings(raw, kWindowSize, kKChannels, T);
            ggml_backend_tensor_set(r.input_tensors.at(std::string(which) + "_" + std::to_string(i)), table.data(), 0,
                                     table.size() * sizeof(float));
        }
    }

    ggml_backend_graph_compute(backend.get(), r.graph);
    std::vector<float> stats(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, stats.data(), 0, stats.size() * sizeof(float));
    LOOM_CHECK(stats.size() == 2 * static_cast<size_t>(kHiddenChannels) * T);

    // stats: channel-first, [2*192, T] -- flat index = t*(2*192) + c. m_p = channels [0,192), logs_p =
    // channels [192,384). Reference .npy is (1,192,T) row-major (C-slow, T-fast): flat index = c*T + t.
    double max_abs_diff_m = 0.0, max_abs_diff_logs = 0.0;
    for (uint32_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < kHiddenChannels; ++c) {
            const float got_m = stats[static_cast<size_t>(t) * 2 * kHiddenChannels + c];
            const float got_logs = stats[static_cast<size_t>(t) * 2 * kHiddenChannels + kHiddenChannels + c];
            const float exp_m = ref_m_p[static_cast<size_t>(c) * T + t];
            const float exp_logs = ref_logs_p[static_cast<size_t>(c) * T + t];
            max_abs_diff_m = std::max(max_abs_diff_m, static_cast<double>(std::fabs(got_m - exp_m)));
            max_abs_diff_logs = std::max(max_abs_diff_logs, static_cast<double>(std::fabs(got_logs - exp_logs)));
        }
    }
    std::fprintf(stderr, "T=%u, max_abs_diff m_p=%g, logs_p=%g\n", T, max_abs_diff_m, max_abs_diff_logs);
    LOOM_CHECK(max_abs_diff_m < 1e-3);
    LOOM_CHECK(max_abs_diff_logs < 1e-3);

    LOOM_TEST_REPORT_AND_RETURN();
}
