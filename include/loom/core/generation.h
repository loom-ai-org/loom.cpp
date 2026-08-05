#pragma once

#include "loom/core/graph_builder.h"
#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"
#include "loom/core/kv_cache.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace loom {

struct GenerationConfig {
    uint32_t max_new_tokens = 32;
    uint32_t n_ctx_max = 512; // KV cache capacity; also what GraphBuilder::reserve() sizes for
    int32_t eos_token = -1;   // stop early when sampled; negative disables the check

    // Optional: invoked once per generated token with that step's full logits row (size n_vocab), right
    // before the argmax that picks the token -- lets callers (tests, debugging tools) inspect/verify raw
    // logits without Generator needing to accumulate and return them all itself.
    std::function<void(const std::vector<float>&)> on_token;
};

// The autoregressive control-flow driver (SPECIFICATION.md §4: "the C++ engine handles the while loop").
// Assumes the topology follows milestone 1's fixed input-naming convention: declared graph inputs named
// "tokens" (i32, [n_tokens]), "positions" (i32, [n_tokens]), and "kq_mask" (f32, [n_kv, n_tokens]).
class Generator {
public:
    Generator(GgufModel& model, GraphTopology topo, GenerationConfig cfg, ggml_backend_t backend);

    // Prefills `prompt_tokens`, then greedily decodes one token at a time until either
    // cfg.max_new_tokens have been generated or eos_token is sampled. Returns just the generated tokens
    // (not the prompt). Resets the generator's KV cache/position state, so a Generator instance can be
    // reused across multiple independent calls.
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt_tokens);

private:
    void write_inputs(const GraphBuilder::BuildResult& result, const std::vector<int32_t>& step_tokens, uint32_t n_past);
    std::vector<float> read_row(ggml_tensor* logits, uint32_t row) const;
    static int32_t argmax(const std::vector<float>& row);

    GgufModel& model_;
    GraphTopology topo_;
    GenerationConfig cfg_;
    ggml_backend_t backend_;
    KvCache kv_cache_;
    GraphBuilder builder_;
    uint32_t n_past_ = 0;
};

} // namespace loom
