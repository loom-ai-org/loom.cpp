#include "loom/core/matcha_driver.h"
#include "loom/core/duration_aligner.h"

#include <ggml-cpu.h>

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace loom {

namespace {

GraphTopology parse_topo(GgufModel& m) { return GraphTopology::parse(m.topology_json()); }

// Real `matcha.utils.model.generate_path`, confirmed (BACKLOG.md 2026-07-19) to degenerate to a plain
// row-repeat for single-utterance (unpadded) inference: `loom::expand_by_duration` already implements
// this. `mu_x_ct` is ggml channel-first [C,T] (C=ne[0] fastest) -- for a FIXED t, its C values are
// already contiguous (exactly the per-token "row" `expand_by_duration` wants), no transpose needed here.
std::vector<std::vector<float>> extract_rows_from_channel_first(const std::vector<float>& flat_ct,
                                                                  uint32_t T, uint32_t C) {
    std::vector<std::vector<float>> rows(T);
    for (uint32_t t = 0; t < T; ++t) {
        rows[t].assign(flat_ct.begin() + static_cast<size_t>(t) * C, flat_ct.begin() + static_cast<size_t>(t + 1) * C);
    }
    return rows;
}

// `rows`: T rows of C floats each (host, row-major). Returns a flat buffer in ggml's [T,C] convention
// (T=ne[0], fastest): idx = t + c*T.
std::vector<float> rows_to_tc_flat(const std::vector<std::vector<float>>& rows, uint32_t T, uint32_t C) {
    std::vector<float> out(static_cast<size_t>(T) * C);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t c = 0; c < C; ++c) out[static_cast<size_t>(t) + static_cast<size_t>(c) * T] = rows[t][c];
    }
    return out;
}

} // namespace

std::unique_ptr<GgufModel> MatchaDriver::load(const std::string& gguf_dir, const std::string& filename) {
    return GgufModel::load(gguf_dir + "/" + filename, backend_);
}

MatchaDriver::MatchaDriver(const std::string& gguf_dir, MatchaConfig cfg, ggml_backend_t backend)
    : cfg_(cfg), backend_(backend), rng_(std::random_device{}()) {
    encoder_mu_model_ = load(gguf_dir, "matcha_encoder_mu.gguf");
    encoder_mu_topo_ = parse_topo(*encoder_mu_model_);
    encoder_logw_model_ = load(gguf_dir, "matcha_encoder_logw.gguf");
    encoder_logw_topo_ = parse_topo(*encoder_logw_model_);
    decoder_model_ = load(gguf_dir, "matcha_decoder.gguf");
    decoder_topo_ = parse_topo(*decoder_model_);
    vocoder_model_ = load(gguf_dir, "matcha_vocoder.gguf");
    vocoder_topo_ = parse_topo(*vocoder_model_);
}

std::vector<float> MatchaDriver::synthesize(const std::vector<int32_t>& tokens, uint32_t n_steps, uint32_t seed) {
    rng_.seed(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    const auto T_text = static_cast<uint32_t>(tokens.size());
    if (T_text == 0) throw std::invalid_argument("MatchaDriver::synthesize: tokens must be non-empty");

    std::vector<int32_t> positions(T_text);
    std::iota(positions.begin(), positions.end(), 0);
    std::vector<float> attn_mask_text(static_cast<size_t>(T_text) * T_text, 0.0f);

    // --- TextEncoder: mu_x (channel-first [n_feats,T_text]) ---
    std::vector<float> mu_x_ct;
    {
        GraphBuilder builder(encoder_mu_topo_, *encoder_mu_model_, backend_);
        GraphBuilder::BuildResult r = builder.build(T_text, 0);
        std::vector<int32_t> tokens_copy = tokens;
        ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));
        ggml_backend_tensor_set(r.input_tensors.at("positions"), positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), attn_mask_text.data(), 0,
                                 attn_mask_text.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        mu_x_ct.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, mu_x_ct.data(), 0, mu_x_ct.size() * sizeof(float));
    }

    // --- TextEncoder: logw (per-token log duration, ne=[1,T_text]) ---
    std::vector<float> logw;
    {
        GraphBuilder builder(encoder_logw_topo_, *encoder_logw_model_, backend_);
        GraphBuilder::BuildResult r = builder.build(T_text, 0);
        std::vector<int32_t> tokens_copy = tokens;
        ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));
        ggml_backend_tensor_set(r.input_tensors.at("positions"), positions.data(), 0, positions.size() * sizeof(int32_t));
        ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), attn_mask_text.data(), 0,
                                 attn_mask_text.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        logw.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, logw.data(), 0, logw.size() * sizeof(float));
    }

    // --- Real `w_ceil = ceil(exp(logw))` (length_scale=1.0) -- per-token integer durations ---
    std::vector<uint32_t> durations(T_text);
    uint32_t T_mel = 0;
    for (uint32_t t = 0; t < T_text; ++t) {
        const auto d = static_cast<uint32_t>(std::ceil(std::exp(static_cast<double>(logw[t]))));
        durations[t] = std::max<uint32_t>(d, 1);
        T_mel += durations[t];
    }
    // Extend the LAST token's duration so T_mel is an exact multiple of 4 -- see MatchaConfig's own
    // docstring for why (the Decoder topology drops all padding-mask handling).
    const uint32_t remainder = T_mel % 4;
    if (remainder != 0) {
        const uint32_t extra = 4 - remainder;
        durations[T_text - 1] += extra;
        T_mel += extra;
    }

    // --- Row-repeat duration expansion (degenerate `generate_path`, see BACKLOG.md) -> mu_y [T_mel,80] ---
    const uint32_t n_feats = cfg_.n_feats;
    std::vector<std::vector<float>> mu_x_rows = extract_rows_from_channel_first(mu_x_ct, T_text, n_feats);
    std::vector<std::vector<float>> mu_y_rows = expand_by_duration(mu_x_rows, durations);
    std::vector<float> mu_y = rows_to_tc_flat(mu_y_rows, T_mel, n_feats);

    // --- Deterministic Euler CFM sampling loop over the Decoder U-Net estimator ---
    std::vector<float> z0(static_cast<size_t>(T_mel) * n_feats);
    for (float& v : z0) v = normal(rng_);

    std::vector<float> attn_mask_full(static_cast<size_t>(T_mel) * T_mel, 0.0f);
    const uint32_t T_half = T_mel / 2;
    std::vector<float> attn_mask_half(static_cast<size_t>(T_half) * T_half, 0.0f);

    VelocityFn velocity_fn = [&](const std::vector<float>& z, float t) {
        GraphBuilder builder(decoder_topo_, *decoder_model_, backend_);
        GraphBuilder::BuildResult r = builder.build(T_mel, 0);
        ggml_backend_tensor_set(r.input_tensors.at("z"), z.data(), 0, z.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("mu"), mu_y.data(), 0, mu_y.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("t"), &t, 0, sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("attn_mask_full"), attn_mask_full.data(), 0,
                                 attn_mask_full.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("attn_mask_half"), attn_mask_half.data(), 0,
                                 attn_mask_half.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);

        std::vector<float> v(z.size());
        ggml_backend_tensor_get(r.output, v.data(), 0, v.size() * sizeof(float));
        return v;
    };
    std::vector<float> mel = cfm_euler_sample(z0, velocity_fn, static_cast<int>(n_steps));

    // --- Denormalize (real `denormalize(decoder_outputs, mel_mean, mel_std)`) ---
    for (float& v : mel) v = v * cfg_.mel_std + cfg_.mel_mean;

    // --- HiFi-GAN v1 vocoder: mel [T_mel,80] -> waveform [T_mel*hop_size] ---
    std::vector<float> waveform;
    {
        GraphBuilder builder(vocoder_topo_, *vocoder_model_, backend_);
        GraphBuilder::BuildResult r = builder.build(T_mel, 0);
        ggml_backend_tensor_set(r.input_tensors.at("mel"), mel.data(), 0, mel.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        waveform.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, waveform.data(), 0, waveform.size() * sizeof(float));
    }
    return waveform;
}

} // namespace loom
