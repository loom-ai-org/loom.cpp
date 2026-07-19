#include "loom/core/styletts2_driver.h"

#include "loom/core/duration_aligner.h"

#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>

namespace loom {

namespace {

// --- Shared helpers, identical to kokoro_driver.cpp's own (duplicated per this project's usual
//     per-driver convention rather than factored into a shared library -- see BACKLOG.md). ---

std::vector<float> run_block_layout_a(GgufModel& model, GraphTopology& topo, ggml_backend_t backend,
                                       const std::vector<std::vector<float>>& x_tc, uint32_t T,
                                       const std::string& x_name, const std::vector<float>* style,
                                       uint32_t& out_T, uint32_t& out_C) {
    GraphBuilder builder(topo, model, backend, nullptr);
    GraphBuilder::BuildResult r = builder.build(T, 0);
    const uint32_t channels = static_cast<uint32_t>(x_tc.empty() ? 0 : x_tc[0].size());
    std::vector<float> x_flat(static_cast<size_t>(channels) * T);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < channels; ++c) x_flat[static_cast<size_t>(c) * T + t] = x_tc[t][c];
    ggml_backend_tensor_set(r.input_tensors.at(x_name), x_flat.data(), 0, x_flat.size() * sizeof(float));
    if (style != nullptr) {
        ggml_backend_tensor_set(r.input_tensors.at("style"), style->data(), 0, style->size() * sizeof(float));
    }
    ggml_backend_graph_compute(backend, r.graph);
    out_T = static_cast<uint32_t>(r.output->ne[0]);
    out_C = static_cast<uint32_t>(r.output->ne[1]);
    std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
    ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
    return out;
}

std::vector<std::vector<float>> layout_a_to_tc(const std::vector<float>& flat, uint32_t T, uint32_t channels) {
    std::vector<std::vector<float>> out(T, std::vector<float>(channels));
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t c = 0; c < channels; ++c) out[t][c] = flat[static_cast<size_t>(c) * T + t];
    return out;
}

std::unique_ptr<BiLstmStepper> make_stepper(GgufModel& model, GraphTopology h_fwd, GraphTopology c_fwd,
                                             GraphTopology h_bwd, GraphTopology c_bwd, ggml_backend_t backend,
                                             uint32_t hidden_per_dir) {
    return std::make_unique<BiLstmStepper>(model, std::move(h_fwd), std::move(c_fwd), std::move(h_bwd),
                                            std::move(c_bwd), backend, hidden_per_dir);
}

GraphTopology parse_topo(GgufModel& m) { return GraphTopology::parse(m.topology_json()); }

} // namespace

std::unique_ptr<GgufModel> StyleTTS2Driver::load(const std::string& gguf_dir, const std::string& filename) {
    return GgufModel::load(gguf_dir + "/" + filename, backend_);
}

StyleTTS2Driver::StyleTTS2Driver(const std::string& gguf_dir, StyleTTS2Config cfg, ggml_backend_t backend)
    : cfg_(cfg), backend_(backend), rng_(std::random_device{}()) {
    albert_model_ = load(gguf_dir, "kokoro_albert.gguf");
    albert_topo_ = parse_topo(*albert_model_);

    diffusion_model_ = load(gguf_dir, "styletts2_diffusion.gguf");
    diffusion_topo_ = parse_topo(*diffusion_model_);

    bert_encoder_model_ = load(gguf_dir, "kokoro_bert_encoder.gguf");
    bert_encoder_topo_ = parse_topo(*bert_encoder_model_);

    text_encoder_cnn_model_ = load(gguf_dir, "kokoro_text_encoder_cnn.gguf");
    text_encoder_cnn_topo_ = parse_topo(*text_encoder_cnn_model_);
    text_encoder_lstm_model_ = load(gguf_dir, "kokoro_text_encoder_lstm_h_fwd.gguf");
    text_encoder_lstm_h_fwd_topo_ = parse_topo(*text_encoder_lstm_model_);
    text_encoder_lstm_c_fwd_topo_ = parse_topo(*load(gguf_dir, "kokoro_text_encoder_lstm_c_fwd.gguf"));
    text_encoder_lstm_h_bwd_topo_ = parse_topo(*load(gguf_dir, "kokoro_text_encoder_lstm_h_bwd.gguf"));
    text_encoder_lstm_c_bwd_topo_ = parse_topo(*load(gguf_dir, "kokoro_text_encoder_lstm_c_bwd.gguf"));

    duration_blocks_.resize(3);
    for (int i = 0; i < 3; ++i) {
        const std::string lp = "kokoro_duration_lstm_" + std::to_string(i);
        duration_blocks_[i].lstm_model = load(gguf_dir, lp + "_h_fwd.gguf");
        duration_blocks_[i].h_fwd_topo = parse_topo(*duration_blocks_[i].lstm_model);
        duration_blocks_[i].c_fwd_topo = parse_topo(*load(gguf_dir, lp + "_c_fwd.gguf"));
        duration_blocks_[i].h_bwd_topo = parse_topo(*load(gguf_dir, lp + "_h_bwd.gguf"));
        duration_blocks_[i].c_bwd_topo = parse_topo(*load(gguf_dir, lp + "_c_bwd.gguf"));
        duration_blocks_[i].adaln_model = load(gguf_dir, "kokoro_duration_adaln_" + std::to_string(i) + ".gguf");
        duration_blocks_[i].adaln_topo = parse_topo(*duration_blocks_[i].adaln_model);
    }

    top_lstm_model_ = load(gguf_dir, "kokoro_duration_top_lstm_h_fwd.gguf");
    top_lstm_h_fwd_topo_ = parse_topo(*top_lstm_model_);
    top_lstm_c_fwd_topo_ = parse_topo(*load(gguf_dir, "kokoro_duration_top_lstm_c_fwd.gguf"));
    top_lstm_h_bwd_topo_ = parse_topo(*load(gguf_dir, "kokoro_duration_top_lstm_h_bwd.gguf"));
    top_lstm_c_bwd_topo_ = parse_topo(*load(gguf_dir, "kokoro_duration_top_lstm_c_bwd.gguf"));
    duration_proj_model_ = load(gguf_dir, "kokoro_duration_proj.gguf");
    duration_proj_topo_ = parse_topo(*duration_proj_model_);

    f0n_shared_lstm_model_ = load(gguf_dir, "kokoro_f0n_shared_lstm_h_fwd.gguf");
    f0n_shared_h_fwd_topo_ = parse_topo(*f0n_shared_lstm_model_);
    f0n_shared_c_fwd_topo_ = parse_topo(*load(gguf_dir, "kokoro_f0n_shared_lstm_c_fwd.gguf"));
    f0n_shared_h_bwd_topo_ = parse_topo(*load(gguf_dir, "kokoro_f0n_shared_lstm_h_bwd.gguf"));
    f0n_shared_c_bwd_topo_ = parse_topo(*load(gguf_dir, "kokoro_f0n_shared_lstm_c_bwd.gguf"));
    for (int i = 0; i < 3; ++i) {
        f0_block_models_[i] = load(gguf_dir, "kokoro_f0n_f0_block" + std::to_string(i) + ".gguf");
        f0_block_topos_[i] = parse_topo(*f0_block_models_[i]);
        n_block_models_[i] = load(gguf_dir, "kokoro_f0n_n_block" + std::to_string(i) + ".gguf");
        n_block_topos_[i] = parse_topo(*n_block_models_[i]);
    }
    f0_proj_model_ = load(gguf_dir, "kokoro_f0n_f0_proj.gguf");
    f0_proj_topo_ = parse_topo(*f0_proj_model_);
    n_proj_model_ = load(gguf_dir, "kokoro_f0n_n_proj.gguf");
    n_proj_topo_ = parse_topo(*n_proj_model_);

    decoder_core_model_ = load(gguf_dir, "styletts2_decoder_core.gguf");
    decoder_core_topo_ = parse_topo(*decoder_core_model_);

    sinegen_model_ = load(gguf_dir, "styletts2_sinegen.gguf");
    sinegen_topo_ = parse_topo(*sinegen_model_);
    stft_forward_model_ = load(gguf_dir, "kokoro_stft_forward.gguf");
    stft_forward_topo_ = parse_topo(*stft_forward_model_);
    stft_inverse_model_ = load(gguf_dir, "kokoro_stft_inverse.gguf");
    stft_inverse_topo_ = parse_topo(*stft_inverse_model_);

    generator_model_ = load(gguf_dir, "styletts2_generator.gguf");
    generator_topo_ = parse_topo(*generator_model_);
}

std::vector<float> StyleTTS2Driver::synthesize(const std::vector<int32_t>& input_ids, uint32_t diffusion_steps,
                                                uint32_t seed) {
    rng_.seed(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    const uint32_t T_text = static_cast<uint32_t>(input_ids.size());
    const uint32_t style_dim = cfg_.style_dim;

    // --- CustomAlbert: raw bert_dur (last_hidden_state), Layout B [768,T] ---
    std::vector<float> bert_out;
    {
        GraphBuilder builder(albert_topo_, *albert_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build(T_text, 0);
        std::vector<int32_t> tokens_copy = input_ids;
        ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));
        std::vector<int32_t> positions(T_text);
        for (uint32_t i = 0; i < T_text; ++i) positions[i] = static_cast<int32_t>(i);
        ggml_backend_tensor_set(r.input_tensors.at("positions"), positions.data(), 0, positions.size() * sizeof(int32_t));
        std::vector<float> mask(static_cast<size_t>(T_text) * T_text, 0.0f);
        ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), mask.data(), 0, mask.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        bert_out.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, bert_out.data(), 0, bert_out.size() * sizeof(float));
    }

    // --- Style-diffusion sampler: ADPM2 over the real Transformer1d, conditioned on RAW bert_dur (not
    //     bert_encoder's projection -- real source: `sampler(noise, embedding=bert_dur[0].unsqueeze(0),
    //     ...)`), embedding_scale=1.0 (the demo's own basic-synthesis default; the CFG branch is out of
    //     scope, see convert_styletts2_diffusion.py). ---
    const std::vector<float> attn_mask_zero(static_cast<size_t>(T_text) * T_text, 0.0f);
    DenoiseFn denoise_fn = [&](const std::vector<float>& x, float sigma) {
        const float sigma_data = cfg_.sigma_data;
        const float c_skip = (sigma_data * sigma_data) / (sigma * sigma + sigma_data * sigma_data);
        const float c_out = sigma * sigma_data / std::sqrt(sigma_data * sigma_data + sigma * sigma);
        const float c_in = 1.0f / std::sqrt(sigma * sigma + sigma_data * sigma_data);
        const float c_noise = std::log(sigma) * 0.25f;

        std::vector<float> x_scaled(x.size());
        for (size_t i = 0; i < x.size(); ++i) x_scaled[i] = x[i] * c_in;

        GraphBuilder builder(diffusion_topo_, *diffusion_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build(T_text, 0);
        ggml_backend_tensor_set(r.input_tensors.at("x_in"), x_scaled.data(), 0, x_scaled.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("time"), &c_noise, 0, sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("embedding"), bert_out.data(), 0, bert_out.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("attn_mask"), attn_mask_zero.data(), 0,
                                 attn_mask_zero.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);

        std::vector<float> model_out(x.size());
        ggml_backend_tensor_get(r.output, model_out.data(), 0, model_out.size() * sizeof(float));

        std::vector<float> x_denoised(x.size());
        for (size_t i = 0; i < x.size(); ++i) x_denoised[i] = c_skip * x[i] + c_out * model_out[i];
        return x_denoised;
    };
    GaussianSampleFn gaussian_sample = [&](std::vector<float>& out) {
        for (float& v : out) v = normal(rng_);
    };

    const uint32_t style_vec_dim = 2 * style_dim;
    std::vector<float> noise0(style_vec_dim);
    for (float& v : noise0) v = normal(rng_);
    const std::vector<float> sigmas = karras_schedule(static_cast<int>(diffusion_steps), cfg_.sigma_min,
                                                       cfg_.sigma_max, cfg_.rho);
    const std::vector<float> s_pred = adpm2_sample(noise0, denoise_fn, sigmas, static_cast<int>(diffusion_steps),
                                                    gaussian_sample);
    const std::vector<float> s_decoder(s_pred.begin(), s_pred.begin() + style_dim);       // "ref"
    const std::vector<float> s_predictor(s_pred.begin() + style_dim, s_pred.end());       // "s"

    // --- bert_encoder ---
    std::vector<float> d_en_flat;  // Layout A ([T,512], flat=c*T+t)
    {
        GraphBuilder builder(bert_encoder_topo_, *bert_encoder_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build(T_text, 0);
        ggml_backend_tensor_set(r.input_tensors.at("x"), bert_out.data(), 0, bert_out.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        d_en_flat.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, d_en_flat.data(), 0, d_en_flat.size() * sizeof(float));
    }

    // --- DurationEncoder: 3x (BiLSTM + AdaLayerNorm), each re-concatenating style ---
    constexpr uint32_t kDModel = 512;
    std::vector<std::vector<float>> x(T_text, std::vector<float>(kDModel + style_dim));
    for (uint32_t t = 0; t < T_text; ++t) {
        for (uint32_t c = 0; c < kDModel; ++c) x[t][c] = d_en_flat[static_cast<size_t>(c) * T_text + t];
        for (uint32_t s = 0; s < style_dim; ++s) x[t][kDModel + s] = s_predictor[s];
    }
    for (int i = 0; i < 3; ++i) {
        DurationBlock& blk = duration_blocks_[i];
        auto stepper = make_stepper(*blk.lstm_model, blk.h_fwd_topo, blk.c_fwd_topo, blk.h_bwd_topo,
                                     blk.c_bwd_topo, backend_, cfg_.hidden_per_dir);
        std::vector<std::vector<float>> lstm_out = stepper->run(x);  // T x 512

        std::vector<float> seq_ct(static_cast<size_t>(kDModel) * T_text);
        for (uint32_t t = 0; t < T_text; ++t)
            for (uint32_t c = 0; c < kDModel; ++c) seq_ct[static_cast<size_t>(t) * kDModel + c] = lstm_out[t][c];

        GraphBuilder ada_builder(blk.adaln_topo, *blk.adaln_model, backend_, nullptr);
        GraphBuilder::BuildResult ar = ada_builder.build(T_text, 0);
        ggml_backend_tensor_set(ar.input_tensors.at("x"), seq_ct.data(), 0, seq_ct.size() * sizeof(float));
        ggml_backend_tensor_set(ar.input_tensors.at("style"), s_predictor.data(), 0, s_predictor.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, ar.graph);
        std::vector<float> ada_out(static_cast<size_t>(ggml_nelements(ar.output)));
        ggml_backend_tensor_get(ar.output, ada_out.data(), 0, ada_out.size() * sizeof(float));

        x.assign(T_text, std::vector<float>(kDModel + style_dim));
        for (uint32_t t = 0; t < T_text; ++t) {
            for (uint32_t c = 0; c < kDModel; ++c) x[t][c] = ada_out[static_cast<size_t>(t) * kDModel + c];
            for (uint32_t s = 0; s < style_dim; ++s) x[t][kDModel + s] = s_predictor[s];
        }
    }
    const std::vector<std::vector<float>>& d = x;  // (T_text, 640)

    // --- predictor.lstm (top BiLSTM) -> duration_proj -> predict_durations ---
    std::vector<std::vector<float>> top_out;
    {
        auto stepper = make_stepper(*top_lstm_model_, top_lstm_h_fwd_topo_, top_lstm_c_fwd_topo_,
                                     top_lstm_h_bwd_topo_, top_lstm_c_bwd_topo_, backend_, cfg_.hidden_per_dir);
        top_out = stepper->run(d);
    }
    std::vector<std::vector<float>> duration_logits(T_text, std::vector<float>(cfg_.max_dur));
    {
        GraphBuilder proj_builder(duration_proj_topo_, *duration_proj_model_, backend_, nullptr);
        for (uint32_t t = 0; t < T_text; ++t) {
            GraphBuilder::BuildResult r = proj_builder.build(0, 0);
            ggml_backend_tensor_set(r.input_tensors.at("x"), top_out[t].data(), 0, top_out[t].size() * sizeof(float));
            ggml_backend_graph_compute(backend_, r.graph);
            ggml_backend_tensor_get(r.output, duration_logits[t].data(), 0, cfg_.max_dur * sizeof(float));
        }
    }
    std::vector<uint32_t> pred_dur = predict_durations(duration_logits, /*speed=*/1.0f);
    // Real quirk (Demo/Inference_LJSpeech.ipynb's own `inference()`): `pred_dur[-1] += 5` -- pads the
    // last token's duration. No division-by-speed either (unlike Kokoro's own forward, which accepts a
    // `speed` argument) -- the real demo's `inference()` has no such parameter at all.
    if (!pred_dur.empty()) pred_dur.back() += 5;

    // --- frame expansion: "en" (640ch, from d) and "asr" (512ch, from a SEPARATE plain TextEncoder) ---
    const std::vector<std::vector<float>> en = expand_by_duration(d, pred_dur);
    uint32_t T_frames = 0;
    for (uint32_t dcount : pred_dur) T_frames += dcount;

    std::vector<std::vector<float>> t_en;
    {
        GraphBuilder cnn_builder(text_encoder_cnn_topo_, *text_encoder_cnn_model_, backend_, nullptr);
        GraphBuilder::BuildResult cnn_r = cnn_builder.build(T_text, 0);
        std::vector<int32_t> tokens_copy = input_ids;
        ggml_backend_tensor_set(cnn_r.input_tensors.at("tokens"), tokens_copy.data(), 0, tokens_copy.size() * sizeof(int32_t));
        ggml_backend_graph_compute(backend_, cnn_r.graph);
        const auto channels = static_cast<uint32_t>(cnn_r.output->ne[1]);
        std::vector<float> cnn_out_flat(static_cast<size_t>(ggml_nelements(cnn_r.output)));
        ggml_backend_tensor_get(cnn_r.output, cnn_out_flat.data(), 0, cnn_out_flat.size() * sizeof(float));
        std::vector<std::vector<float>> cnn_out = layout_a_to_tc(cnn_out_flat, T_text, channels);

        auto stepper = make_stepper(*text_encoder_lstm_model_, text_encoder_lstm_h_fwd_topo_,
                                     text_encoder_lstm_c_fwd_topo_, text_encoder_lstm_h_bwd_topo_,
                                     text_encoder_lstm_c_bwd_topo_, backend_, cfg_.hidden_per_dir);
        t_en = stepper->run(cnn_out);
    }
    const std::vector<std::vector<float>> asr = expand_by_duration(t_en, pred_dur);

    // --- F0Ntrain: shared BiLSTM -> F0/N AdainResBlk1d stacks -> projections ---
    std::vector<std::vector<float>> shared_out;
    {
        auto stepper = make_stepper(*f0n_shared_lstm_model_, f0n_shared_h_fwd_topo_, f0n_shared_c_fwd_topo_,
                                     f0n_shared_h_bwd_topo_, f0n_shared_c_bwd_topo_, backend_, cfg_.hidden_per_dir);
        shared_out = stepper->run(en);
    }
    auto run_stack = [&](std::unique_ptr<GgufModel> (&models)[3], GraphTopology (&topos)[3]) {
        std::vector<std::vector<float>> cur = shared_out;
        for (int i = 0; i < 3; ++i) {
            const uint32_t cur_T = static_cast<uint32_t>(cur.size());
            uint32_t out_T = 0, out_c = 0;
            std::vector<float> flat = run_block_layout_a(*models[i], topos[i], backend_, cur, cur_T, "x",
                                                          &s_predictor, out_T, out_c);
            cur = layout_a_to_tc(flat, out_T, out_c);
        }
        return cur;
    };
    std::vector<std::vector<float>> f0_feat = run_stack(f0_block_models_, f0_block_topos_);
    std::vector<std::vector<float>> n_feat = run_stack(n_block_models_, n_block_topos_);
    const uint32_t T_f0 = static_cast<uint32_t>(f0_feat.size());

    auto run_proj = [&](GgufModel& model, GraphTopology& topo, const std::vector<std::vector<float>>& feat) {
        GraphBuilder builder(topo, model, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build(static_cast<uint32_t>(feat.size()), 0);
        const uint32_t channels = static_cast<uint32_t>(feat[0].size());
        std::vector<float> flat(static_cast<size_t>(channels) * feat.size());
        for (size_t t = 0; t < feat.size(); ++t)
            for (uint32_t c = 0; c < channels; ++c) flat[static_cast<size_t>(c) * feat.size() + t] = feat[t][c];
        ggml_backend_tensor_set(r.input_tensors.at("x"), flat.data(), 0, flat.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        std::vector<float> out(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, out.data(), 0, out.size() * sizeof(float));
        return out;
    };
    const std::vector<float> F0_curve = run_proj(*f0_proj_model_, f0_proj_topo_, f0_feat);
    const std::vector<float> N_curve = run_proj(*n_proj_model_, n_proj_topo_, n_feat);

    // --- Decoder core: F0_conv/N_conv + encode/decode AdainResBlk1d stack -> x (512ch, T_f0 long) ---
    std::vector<float> decoder_x_flat;
    {
        GraphBuilder builder(decoder_core_topo_, *decoder_core_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build(T_frames, 0);
        std::vector<float> asr_flat(static_cast<size_t>(512) * T_frames);
        for (uint32_t t = 0; t < T_frames; ++t)
            for (uint32_t c = 0; c < 512; ++c) asr_flat[static_cast<size_t>(c) * T_frames + t] = asr[t][c];
        ggml_backend_tensor_set(r.input_tensors.at("asr"), asr_flat.data(), 0, asr_flat.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("f0_curve"), F0_curve.data(), 0, F0_curve.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("n_curve"), N_curve.data(), 0, N_curve.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("style"), s_decoder.data(), 0, s_decoder.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        decoder_x_flat.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, decoder_x_flat.data(), 0, decoder_x_flat.size() * sizeof(float));
    }

    // --- SineGen: harmonic source from F0_curve (host rand_ini/noise draws) ---
    const uint32_t dim = cfg_.harmonic_num + 1;
    const uint32_t L = T_f0 * cfg_.upsample_scale;
    std::vector<float> rand_ini(dim, 0.0f);
    for (uint32_t h = 1; h < dim; ++h) rand_ini[h] = uniform01(rng_);
    std::vector<float> noise_tc(static_cast<size_t>(dim) * L);
    for (size_t i = 0; i < noise_tc.size(); ++i) noise_tc[i] = normal(rng_);
    std::vector<float> har_source;
    {
        GraphBuilder builder(sinegen_topo_, *sinegen_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build(T_f0, 0);
        ggml_backend_tensor_set(r.input_tensors.at("f0_curve"), F0_curve.data(), 0, F0_curve.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("rand_ini"), rand_ini.data(), 0, rand_ini.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("noise"), noise_tc.data(), 0, noise_tc.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        har_source.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, har_source.data(), 0, har_source.size() * sizeof(float));
    }

    // --- forward STFT: har_source (host reflect-padded) -> har (mag+phase concat) ---
    const uint32_t n_fft = cfg_.gen_istft_n_fft;
    const uint32_t hop = cfg_.gen_istft_hop;
    const uint32_t pad = n_fft / 2;
    std::vector<float> waveform_padded(L + 2 * pad);
    for (uint32_t i = 0; i < pad; ++i) waveform_padded[i] = har_source[pad - i];
    for (uint32_t i = 0; i < L; ++i) waveform_padded[pad + i] = har_source[i];
    for (uint32_t i = 0; i < pad; ++i) waveform_padded[pad + L + i] = har_source[L - 2 - i];
    std::vector<float> har_flat;
    {
        GraphBuilder builder(stft_forward_topo_, *stft_forward_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build(static_cast<uint32_t>(waveform_padded.size()), 0);
        ggml_backend_tensor_set(r.input_tensors.at("waveform_padded"), waveform_padded.data(), 0,
                                 waveform_padded.size() * sizeof(float));
        ggml_backend_graph_compute(backend_, r.graph);
        har_flat.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, har_flat.data(), 0, har_flat.size() * sizeof(float));
    }
    const uint32_t T_har = (static_cast<uint32_t>(waveform_padded.size()) - n_fft) / hop + 1;

    // --- Generator core: x + har + host-precomputed wsum -> waveform ---
    std::vector<float> waveform;
    {
        GraphBuilder builder(generator_topo_, *generator_model_, backend_, nullptr);
        GraphBuilder::BuildResult r = builder.build(T_f0, 0);
        ggml_backend_tensor_set(r.input_tensors.at("x"), decoder_x_flat.data(), 0, decoder_x_flat.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("style"), s_decoder.data(), 0, s_decoder.size() * sizeof(float));
        ggml_backend_tensor_set(r.input_tensors.at("har"), har_flat.data(), 0, har_flat.size() * sizeof(float));

        const uint32_t out_len_full = (T_har - 1) * hop + n_fft;
        std::vector<float> window(n_fft);
        for (uint32_t i = 0; i < n_fft; ++i) window[i] = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * i / n_fft);
        std::vector<float> wsum(out_len_full, 0.0f);
        for (uint32_t t = 0; t < T_har; ++t)
            for (uint32_t i = 0; i < n_fft; ++i) wsum[t * hop + i] += window[i] * window[i];
        ggml_backend_tensor_set(r.input_tensors.at("wsum"), wsum.data(), 0, wsum.size() * sizeof(float));

        ggml_backend_graph_compute(backend_, r.graph);
        waveform.resize(static_cast<size_t>(ggml_nelements(r.output)));
        ggml_backend_tensor_get(r.output, waveform.data(), 0, waveform.size() * sizeof(float));
    }
    return waveform;
}

} // namespace loom
