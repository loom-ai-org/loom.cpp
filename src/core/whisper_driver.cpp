#include "loom/core/whisper_driver.h"

#include "loom/loom_errors.h"

#include <ggml-backend.h>

#include <limits>

namespace loom {

WhisperDriver::WhisperDriver(GgufModel& encoder_model, GraphTopology encoder_topo, GgufModel& decoder_model,
                              GraphTopology decoder_topo, WhisperConfig cfg, ggml_backend_t backend)
    : encoder_model_(encoder_model),
      decoder_model_(decoder_model),
      cfg_(cfg),
      backend_(backend),
      encoder_topo_(std::move(encoder_topo)),
      decoder_topo_(std::move(decoder_topo)),
      kv_cache_(cfg_.n_text_layer, cfg_.n_text_state, cfg_.n_text_state, cfg_.n_text_ctx, backend_) {
    encoder_builder_ = std::make_unique<GraphBuilder>(encoder_topo_, encoder_model_, backend_, /*kv_cache=*/nullptr);
    decoder_builder_ = std::make_unique<GraphBuilder>(decoder_topo_, decoder_model_, backend_, &kv_cache_);
}

void WhisperDriver::fill_decoder_inputs(GraphBuilder::BuildResult& r, const std::vector<int32_t>& step_tokens,
                                        uint32_t n_past, const std::vector<float>& xa) {
    const auto n_tokens = static_cast<uint32_t>(step_tokens.size());
    const uint32_t n_kv = n_past + n_tokens;

    std::vector<int32_t> tokens_copy = step_tokens;
    ggml_backend_tensor_set(r.input_tensors.at("tokens"), tokens_copy.data(), 0, n_tokens * sizeof(int32_t));

    std::vector<int32_t> positions(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) positions[i] = static_cast<int32_t>(n_past + i);
    ggml_backend_tensor_set(r.input_tensors.at("positions"), positions.data(), 0, n_tokens * sizeof(int32_t));

    // Same causal-triangle construction as Generator::write_inputs: mask[i*n_kv+j] gates query token i
    // (absolute position n_past+i) attending to self-attention KV cell j.
    std::vector<float> mask(static_cast<size_t>(n_kv) * n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const uint32_t query_pos = n_past + i;
        for (uint32_t j = 0; j < n_kv; ++j) {
            mask[static_cast<size_t>(i) * n_kv + j] = (j <= query_pos) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    ggml_backend_tensor_set(r.input_tensors.at("kq_mask"), mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_tensor_set(r.input_tensors.at("xa"), xa.data(), 0, xa.size() * sizeof(float));
    // Cross-attention has no causal/padding structure at all -- every query attends to every encoder
    // position, unconditionally (matches the real model's own cross_attn(..., mask=None) call).
    std::vector<float> xa_mask(static_cast<size_t>(cfg_.n_audio_ctx) * n_tokens, 0.0f);
    ggml_backend_tensor_set(r.input_tensors.at("xa_mask"), xa_mask.data(), 0, xa_mask.size() * sizeof(float));
}

int32_t WhisperDriver::argmax(const float* row, uint32_t n) {
    int32_t best = 0;
    float best_val = row[0];
    for (uint32_t i = 1; i < n; ++i) {
        if (row[i] > best_val) {
            best_val = row[i];
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

std::vector<int32_t> WhisperDriver::transcribe(const std::vector<float>& waveform_padded,
                                                const std::vector<int32_t>& prompt_tokens,
                                                uint32_t max_new_tokens) {
    if (prompt_tokens.empty()) {
        throw Error("WhisperDriver::transcribe: prompt_tokens must be non-empty");
    }

    // --- Encoder: one fixed-shape pass, no dynamic symbol needed (every shape in this topology is a
    //     compile-time constant -- see convert_whisper_encoder.py's own header comment). ---
    GraphBuilder::BuildResult enc_r = encoder_builder_->build({{"n_tokens", 0}, {"n_past", 0}});
    std::vector<float> waveform_copy = waveform_padded;
    ggml_backend_tensor_set(enc_r.input_tensors.at("waveform"), waveform_copy.data(), 0,
                             waveform_copy.size() * sizeof(float));
    std::vector<float> enc_mask(static_cast<size_t>(cfg_.n_audio_ctx) * cfg_.n_audio_ctx, 0.0f);
    ggml_backend_tensor_set(enc_r.input_tensors.at("enc_attn_mask"), enc_mask.data(), 0,
                             enc_mask.size() * sizeof(float));
    ggml_backend_graph_compute(backend_, enc_r.graph);
    std::vector<float> xa(static_cast<size_t>(ggml_nelements(enc_r.output)));
    ggml_backend_tensor_get(enc_r.output, xa.data(), 0, xa.size() * sizeof(float));

    // --- Decoder: prefill the prompt in one shot, then greedily decode one token at a time. ---
    kv_cache_.reset();
    uint32_t n_past = 0;
    std::vector<int32_t> generated;

    const auto n_prompt_tokens = static_cast<uint32_t>(prompt_tokens.size());
    {
        GraphBuilder::BuildResult r = decoder_builder_->build({{"n_tokens", n_prompt_tokens}, {"n_past", n_past}});
        fill_decoder_inputs(r, prompt_tokens, n_past, xa);
        ggml_backend_graph_compute(backend_, r.graph);
        n_past += n_prompt_tokens;

        const auto n_vocab = static_cast<uint32_t>(r.output->ne[0]);
        std::vector<float> row(n_vocab);
        ggml_backend_tensor_get(r.output, row.data(), static_cast<size_t>(n_prompt_tokens - 1) * n_vocab * sizeof(float),
                                 n_vocab * sizeof(float));
        const int32_t next = argmax(row.data(), n_vocab);
        generated.push_back(next);
        if (cfg_.eot_token >= 0 && next == cfg_.eot_token) return generated;
    }

    while (generated.size() < max_new_tokens) {
        GraphBuilder::BuildResult r = decoder_builder_->build({{"n_tokens", 1}, {"n_past", n_past}});
        fill_decoder_inputs(r, {generated.back()}, n_past, xa);
        ggml_backend_graph_compute(backend_, r.graph);
        n_past += 1;

        const auto n_vocab = static_cast<uint32_t>(r.output->ne[0]);
        std::vector<float> row(n_vocab);
        ggml_backend_tensor_get(r.output, row.data(), 0, n_vocab * sizeof(float));
        const int32_t next = argmax(row.data(), n_vocab);
        generated.push_back(next);
        if (cfg_.eot_token >= 0 && next == cfg_.eot_token) break;
    }

    return generated;
}

} // namespace loom
