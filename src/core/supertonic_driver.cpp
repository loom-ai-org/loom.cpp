#include "loom/core/supertonic_driver.h"

#include <ggml-cpu.h>

#include <cmath>

namespace loom {

namespace {

GraphTopology parse_topo(GgufModel& m) { return GraphTopology::parse(m.topology_json()); }

// Real `get_latent_mask`: `wav_length = duration*sample_rate`; `latent_size = base_chunk_size *
// compression_factor`; `latent_length = ceil(wav_length / latent_size)`.
uint32_t compute_t_lat(float duration_seconds, float sample_rate, uint32_t base_chunk_size,
                        uint32_t compression_factor) {
    const uint32_t wav_length = static_cast<uint32_t>(duration_seconds * sample_rate);
    const uint32_t latent_size = base_chunk_size * compression_factor;
    return (wav_length + latent_size - 1) / latent_size;
}

// Layout A [T,C] (T=ne[0]) -> Layout B [C,T] (C=ne[0]), plain host transpose -- same crossing this
// whole model family does in-graph via PERMUTE+CONT, done here in C++ because it happens BETWEEN two
// separately-built topologies (TTLTextEncoder's own output, VectorFieldEstimator's own input), not
// within a single graph.
std::vector<float> layout_a_to_b(const std::vector<float>& flat_ta, uint32_t T, uint32_t C) {
    std::vector<float> out(static_cast<size_t>(T) * C);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < C; ++c) out[static_cast<size_t>(t) * C + c] = flat_ta[static_cast<size_t>(c) * T + t];
    return out;
}

} // namespace

std::unique_ptr<GgufModel> SupertonicDriver::load(const std::string& gguf_dir, const std::string& filename) {
    return GgufModel::load(gguf_dir + "/" + filename, backend_);
}

SupertonicDriver::SupertonicDriver(const std::string& gguf_dir, SupertonicConfig cfg, ggml_backend_t backend)
    : cfg_(cfg), backend_(backend), rng_(std::random_device{}()) {
    dp_model_ = load(gguf_dir, "supertonic_dp.gguf");
    dp_topo_ = parse_topo(*dp_model_);
    ttl_text_model_ = load(gguf_dir, "supertonic_ttl_text.gguf");
    ttl_text_topo_ = parse_topo(*ttl_text_model_);
    vfe_model_ = load(gguf_dir, "supertonic_vfe.gguf");
    vfe_topo_ = parse_topo(*vfe_model_);
    decoder_model_ = load(gguf_dir, "supertonic_decoder.gguf");
    decoder_topo_ = parse_topo(*decoder_model_);
}

std::vector<float> SupertonicDriver::synthesize(const std::vector<int32_t>& txt_ids,
                                                 const std::vector<float>& style_ttl,
                                                 const std::vector<float>& style_dp, uint32_t n_steps,
                                                 uint32_t seed) {
    rng_.seed(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    const uint32_t T_text = cfg_.txt_len_fixed;
    if (txt_ids.size() != T_text) {
        throw std::invalid_argument("SupertonicDriver::synthesize: txt_ids.size() must equal "
                                     "cfg.txt_len_fixed (a real, documented scope limitation -- see "
                                     "SupertonicConfig's own docstring)");
    }

    // --- DurationPredictor: DPTextEncoder + MLP head -> scalar duration (seconds) ---
    float duration = 0.0f;
    {
        GraphBuilder builder(dp_topo_, *dp_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build({{"n_tokens", T_text}, {"n_past", 0}});
        std::vector<int32_t> tokens_copy = txt_ids;
        ggml_backend_tensor_set(r.input_tensors.at("txt_ids"), tokens_copy.data(), 0,
                                 tokens_copy.size() * sizeof(int32_t));
        ggml_backend_tensor_set(r.input_tensors.at("stl_emb"), style_dp.data(), 0, style_dp.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        ggml_backend_tensor_get(r.output, &duration, 0, sizeof(float));
    }

    const uint32_t T_lat = compute_t_lat(duration, cfg_.sample_rate, cfg_.base_chunk_size,
                                          cfg_.compression_factor);

    // --- TTLTextEncoder -> txt_emb (Layout A [T_text, txt_dim]), crossed to Layout B for VFE ---
    std::vector<float> txt_emb_cb;
    {
        GraphBuilder builder(ttl_text_topo_, *ttl_text_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build({{"n_tokens", T_text}, {"n_past", 0}});
        std::vector<int32_t> tokens_copy = txt_ids;
        ggml_backend_tensor_set(r.input_tensors.at("txt_ids"), tokens_copy.data(), 0,
                                 tokens_copy.size() * sizeof(int32_t));
        ggml_backend_tensor_set(r.input_tensors.at("stl_emb_ttl_cb"), style_ttl.data(), 0,
                                 style_ttl.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        const auto txt_dim = static_cast<uint32_t>(r.output->ne[1]);
        std::vector<float> txt_emb_ta(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, txt_emb_ta.data(), 0, txt_emb_ta.size() * sizeof(float));
        txt_emb_cb = layout_a_to_b(txt_emb_ta, T_text, txt_dim);
    }

    // --- Deterministic Euler CFM sampling loop over VectorFieldEstimator ---
    const uint32_t lat_dim = cfg_.latent_dim;
    std::vector<float> z0(static_cast<size_t>(T_lat) * lat_dim);
    for (float& v : z0) v = normal(rng_);

    std::vector<float> lat_frac(T_lat);
    for (uint32_t i = 0; i < T_lat; ++i) lat_frac[i] = static_cast<float>(i) / static_cast<float>(T_lat);
    std::vector<float> txt_frac(T_text);
    for (uint32_t i = 0; i < T_text; ++i) txt_frac[i] = static_cast<float>(i) / static_cast<float>(T_text);

    VelocityFn velocity_fn = [&](const std::vector<float>& z, float t) {
        GraphBuilder builder(vfe_topo_, *vfe_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build({{"n_tokens", T_lat}, {"n_past", 0}});
        ggml_backend_tensor_set(r.input_tensors.at("z_t"), z.data(), 0, z.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("txt_emb_cb"), txt_emb_cb.data(), 0,
                                 txt_emb_cb.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("stl_emb_ttl_cb"), style_ttl.data(), 0,
                                 style_ttl.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("t"), &t, 0, sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("lat_frac"), lat_frac.data(), 0, lat_frac.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("txt_frac"), txt_frac.data(), 0, txt_frac.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);

        std::vector<float> v(z.size());
        ggml_backend_tensor_get(r.output, v.data(), 0, v.size() * sizeof(float));
        return v;
    };
    const std::vector<float> z_final = cfm_euler_sample(z0, velocity_fn, static_cast<int>(n_steps));

    // --- SpeechDecoder: z_final (Layout A [T_lat,144]) -> raw waveform ---
    std::vector<float> waveform;
    {
        GraphBuilder builder(decoder_topo_, *decoder_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build({{"n_tokens", T_lat}, {"n_past", 0}});
        ggml_backend_tensor_set(r.input_tensors.at("latent"), z_final.data(), 0, z_final.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        waveform.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, waveform.data(), 0, waveform.size() * sizeof(float));
    }
    return waveform;
}

} // namespace loom
