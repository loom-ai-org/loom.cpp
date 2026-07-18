#include "loom/core/vits_driver.h"

#include "loom/loom_errors.h"

#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <random>

namespace loom {
namespace {

// Direct C++ port of piper's `attentions.py::MultiHeadAttention._get_relative_embeddings`, verified
// against the real function's Python translation (tools/convert_piper_vits/vits_common.py's own
// `get_relative_embeddings`, itself cross-checked against the real PyTorch method for lengths spanning
// both branches -- see BACKLOG.md). `raw` is the fixed `(2*window_size+1) * k_channels`-element learned
// table (row-major, k_channels fastest); returns the `(2*length-1) * k_channels`-element table
// `REL_POS_ATTENTION_SHAW` expects for this call's real token count.
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

VitsDriver::VitsDriver(GgufModel& stats_model, GraphTopology stats_topo, GgufModel& logw_model,
                        GraphTopology logw_topo, GgufModel& flow_vocoder_model, GraphTopology flow_vocoder_topo,
                        VitsConfig cfg, ggml_backend_t backend)
    : stats_model_(stats_model),
      logw_model_(logw_model),
      flow_vocoder_model_(flow_vocoder_model),
      cfg_(cfg),
      backend_(backend),
      stats_topo_(std::move(stats_topo)),
      logw_topo_(std::move(logw_topo)),
      flow_vocoder_topo_(std::move(flow_vocoder_topo)) {
    stats_builder_ = std::make_unique<GraphBuilder>(stats_topo_, stats_model_, backend_, /*kv_cache=*/nullptr);
    logw_builder_ = std::make_unique<GraphBuilder>(logw_topo_, logw_model_, backend_, /*kv_cache=*/nullptr);
    flow_vocoder_builder_ =
        std::make_unique<GraphBuilder>(flow_vocoder_topo_, flow_vocoder_model_, backend_, /*kv_cache=*/nullptr);
}

std::vector<float> VitsDriver::synthesize(const std::vector<int32_t>& token_ids, uint32_t seed) {
    const auto T = static_cast<uint32_t>(token_ids.size());
    if (T == 0) {
        throw Error("VitsDriver::synthesize: token_ids must be non-empty");
    }
    const int64_t k_channels = cfg_.hidden_channels / cfg_.n_heads;
    const int64_t inter_channels = cfg_.inter_channels;
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Shared per-call fill logic: tokens, an all-zero (no padding) additive attention mask, and every
    // TextEncoder layer's dynamic-T relative-position tables -- identical for both the `stats` and
    // `logw` builds (both recompute TextEncoder from scratch; see BACKLOG.md's "why two separate
    // topologies" note).
    auto fill_text_encoder_inputs = [&](GgufModel& model, GraphBuilder::BuildResult& r) {
        std::vector<int32_t> tokens_copy = token_ids;
        ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));
        std::vector<float> mask(static_cast<size_t>(T) * T, 0.0f);
        ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), mask.data(), 0, mask.size() * sizeof(float));

        for (uint32_t i = 0; i < cfg_.n_text_layers; ++i) {
            const std::string prefix = "enc_p.encoder.attn_layers." + std::to_string(i);
            for (const char* which : {"emb_rel_k", "emb_rel_v"}) {
                ggml_tensor* raw_t = model.weight(prefix + "." + which + "_raw");
                std::vector<float> raw(static_cast<size_t>(ggml_nelements(raw_t)));
                ggml_backend_tensor_get(raw_t, raw.data(), 0, raw.size() * sizeof(float));
                std::vector<float> table = pad_crop_relative_embeddings(raw, cfg_.window_size, k_channels, T);
                ggml_backend_tensor_set(r.input_tensors.at(std::string(which) + "_" + std::to_string(i)), table.data(),
                                         0, table.size() * sizeof(float));
            }
        }
    };

    // --- Phase 1a: stats = TextEncoder -> [m_p; logs_p] (channel-first, [2*inter_channels, T]) ---
    GraphBuilder::BuildResult stats_r = stats_builder_->build(T, 0);
    fill_text_encoder_inputs(stats_model_, stats_r);
    ggml_backend_graph_compute(backend_, stats_r.graph);
    std::vector<float> stats(static_cast<size_t>(ggml_nelements(stats_r.output)));
    ggml_backend_tensor_get(stats_r.output, stats.data(), 0, stats.size() * sizeof(float));

    // --- Phase 1b: logw = TextEncoder + StochasticDurationPredictor(reverse) -> [T] duration logits ---
    GraphBuilder::BuildResult logw_r = logw_builder_->build(T, 0);
    fill_text_encoder_inputs(logw_model_, logw_r);
    std::vector<float> z_noise(static_cast<size_t>(T) * 2);
    for (float& v : z_noise) v = normal(rng) * cfg_.noise_scale_w;
    ggml_backend_tensor_set(logw_r.input_tensors.at("z_noise"), z_noise.data(), 0, z_noise.size() * sizeof(float));
    ggml_backend_graph_compute(backend_, logw_r.graph);
    std::vector<float> logw(T);
    ggml_backend_tensor_get(logw_r.output, logw.data(), 0, logw.size() * sizeof(float));

    // --- Host-side generate_path: w_ceil[t] = ceil(exp(logw[t]) * length_scale); y_length =
    //     max(sum(w_ceil), 1) -- real models.py::infer(), with x_mask/y_mask dropped (both always
    //     all-ones: single unpadded utterance, no batching, matching every other simplification this
    //     whole VITS effort has made). The alignment matrix itself degenerates to a plain "replicate
    //     column t of m_p/logs_p for w_ceil[t] consecutive output frames" expansion once the mask is
    //     gone -- computed directly below, no explicit attn matrix ever materialized. ---
    std::vector<uint32_t> w_ceil(T);
    uint64_t y_length_u64 = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const float w = std::exp(logw[t]) * cfg_.length_scale;
        w_ceil[t] = static_cast<uint32_t>(std::ceil(w));
        y_length_u64 += w_ceil[t];
    }
    const uint32_t y_length = static_cast<uint32_t>(std::max<uint64_t>(y_length_u64, 1));

    // z_p: [y_length, inter_channels], T-major (matches the coupling-flow/vocoder's own [T,C]
    // convention established throughout this VITS effort). m_p[c,t] = stats[t*(2*inter_channels)+c],
    // logs_p[c,t] = stats[t*(2*inter_channels)+inter_channels+c] (stats is channel-first, [C,T]).
    std::vector<float> z_p(static_cast<size_t>(y_length) * inter_channels);
    uint32_t out_frame = 0;
    for (uint32_t t = 0; t < T && out_frame < y_length; ++t) {
        const float* m_p_col = stats.data() + static_cast<size_t>(t) * 2 * inter_channels;
        const float* logs_p_col = m_p_col + inter_channels;
        for (uint32_t rep = 0; rep < w_ceil[t] && out_frame < y_length; ++rep, ++out_frame) {
            float* dst = z_p.data() + static_cast<size_t>(out_frame) * inter_channels;
            for (int64_t c = 0; c < inter_channels; ++c) {
                dst[c] = m_p_col[c] + normal(rng) * std::exp(logs_p_col[c]) * cfg_.noise_scale;
            }
        }
    }

    // --- Phase 2: coupling flow (reverse) + HiFi-GAN vocoder -> waveform ---
    GraphBuilder::BuildResult wav_r = flow_vocoder_builder_->build(y_length, 0);
    ggml_backend_tensor_set(wav_r.input_tensors.at("z_p"), z_p.data(), 0, z_p.size() * sizeof(float));
    ggml_backend_graph_compute(backend_, wav_r.graph);
    std::vector<float> waveform(static_cast<size_t>(ggml_nelements(wav_r.output)));
    ggml_backend_tensor_get(wav_r.output, waveform.data(), 0, waveform.size() * sizeof(float));
    return waveform;
}

} // namespace loom
