#include "loom/core/generation.h"

#include <ggml-backend.h>

#include <cmath>
#include <limits>

namespace loom {

Generator::Generator(GgufModel& model, GraphTopology topo, GenerationConfig cfg, ggml_backend_t backend)
    : model_(model),
      topo_(std::move(topo)),
      cfg_(cfg),
      backend_(backend),
      kv_cache_(model.hparam_u32("n_layer"),
                model.hparam_u32("n_head_kv") * model.hparam_u32("n_embd_head_k"),
                model.hparam_u32("n_head_kv") * model.hparam_u32("n_embd_head_v"),
                cfg.n_ctx_max, backend),
      builder_(topo_, model, backend, &kv_cache_) {
    builder_.reserve(cfg_.n_ctx_max);
}

void Generator::write_inputs(const GraphBuilder::BuildResult& result, const std::vector<int32_t>& step_tokens, uint32_t n_past) {
    const uint32_t n_tokens = static_cast<uint32_t>(step_tokens.size());
    const uint32_t n_kv = n_past + n_tokens;

    ggml_tensor* tokens_t = result.input_tensors.at("tokens");
    ggml_backend_tensor_set(tokens_t, step_tokens.data(), 0, n_tokens * sizeof(int32_t));

    std::vector<int32_t> positions(n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) positions[i] = static_cast<int32_t>(n_past + i);
    ggml_tensor* positions_t = result.input_tensors.at("positions");
    ggml_backend_tensor_set(positions_t, positions.data(), 0, n_tokens * sizeof(int32_t));

    // mask[i * n_kv + j] gates query token i (absolute position n_past+i) attending to KV cell j: 0.0 if
    // attendable (j <= n_past+i), -inf otherwise. Covers prefill's causal triangle and decode's
    // attend-to-everything-so-far single row uniformly -- no separate prefill/decode mask logic needed.
    std::vector<float> mask(static_cast<size_t>(n_kv) * n_tokens);
    for (uint32_t i = 0; i < n_tokens; ++i) {
        const uint32_t query_pos = n_past + i;
        for (uint32_t j = 0; j < n_kv; ++j) {
            mask[static_cast<size_t>(i) * n_kv + j] = (j <= query_pos) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    ggml_tensor* mask_t = result.input_tensors.at("kq_mask");
    ggml_backend_tensor_set(mask_t, mask.data(), 0, mask.size() * sizeof(float));
}

std::vector<float> Generator::read_row(ggml_tensor* logits, uint32_t row) const {
    const uint32_t n_vocab = static_cast<uint32_t>(logits->ne[0]);
    std::vector<float> row_data(n_vocab);
    ggml_backend_tensor_get(logits, row_data.data(), static_cast<size_t>(row) * n_vocab * sizeof(float),
                             n_vocab * sizeof(float));
    return row_data;
}

int32_t Generator::argmax(const std::vector<float>& row) {
    int32_t best = 0;
    float best_val = row[0];
    for (size_t i = 1; i < row.size(); ++i) {
        if (row[i] > best_val) {
            best_val = row[i];
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

std::vector<int32_t> Generator::generate(const std::vector<int32_t>& prompt_tokens) {
    kv_cache_.reset();
    n_past_ = 0;

    std::vector<int32_t> generated;
    if (prompt_tokens.empty() || cfg_.max_new_tokens == 0) {
        return generated;
    }

    // Prefill: one shot over the whole prompt.
    const auto n_prompt_tokens = static_cast<uint32_t>(prompt_tokens.size());
    {
        const GraphBuilder::BuildResult& result = builder_.build({{"n_tokens", n_prompt_tokens}, {"n_past", n_past_}});
        write_inputs(result, prompt_tokens, n_past_);
        ggml_backend_graph_compute(backend_, result.graph);
        n_past_ += n_prompt_tokens;

        const std::vector<float> row = read_row(result.output, n_prompt_tokens - 1);
        if (cfg_.on_token) cfg_.on_token(row);
        const int32_t next = argmax(row);
        generated.push_back(next);
        if (cfg_.eos_token >= 0 && next == cfg_.eos_token) return generated;
    }

    // Decode: one token at a time, feeding the previous step's sample back in.
    while (generated.size() < cfg_.max_new_tokens) {
        const GraphBuilder::BuildResult& result = builder_.build({{"n_tokens", 1}, {"n_past", n_past_}});
        write_inputs(result, {generated.back()}, n_past_);
        ggml_backend_graph_compute(backend_, result.graph);
        n_past_ += 1;

        const std::vector<float> row = read_row(result.output, 0);
        if (cfg_.on_token) cfg_.on_token(row);
        const int32_t next = argmax(row);
        generated.push_back(next);
        if (cfg_.eos_token >= 0 && next == cfg_.eos_token) break;
    }

    return generated;
}

} // namespace loom
