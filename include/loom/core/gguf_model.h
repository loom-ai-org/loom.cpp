#pragma once

#include "loom/core/symbol_table.h"

#include <ggml-cpp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

// The persistent "Model Context" half of the engine's two-context paradigm (see SPECIFICATION.md §1):
// loads a .gguf file's weights into a real ggml_backend_buffer (so the same loader works unmodified
// whichever backend is passed in), builds the Symbol Table mapping weight names to their tensors, and
// exposes the embedded JSON graph-topology string plus the model's scalar hyperparameters.
class GgufModel {
public:
    // Loads `path`. Weight tensor data is copied into a buffer allocated on `backend` (CPU-only for
    // now, but this is the same call site that would target a CUDA/Metal backend later). Throws
    // loom::LoadError if the file can't be opened/parsed, is missing the "model.graph_topology" KV, or
    // a tensor's data can't be read.
    static std::unique_ptr<GgufModel> load(const std::string& path, ggml_backend_t backend);

    // Looks up a weight tensor by its GGUF name (e.g. "blk.0.attn_q.weight"). Throws loom::LoadError if
    // not present -- callers that want a non-throwing check should use has_weight() first.
    ggml_tensor* weight(const std::string& name) const;
    bool has_weight(const std::string& name) const;

    // Read-only view of every loaded weight, for seeding a per-call SymbolTable in GraphBuilder.
    const SymbolTable& weights() const { return symbols_; }

    // The raw JSON text stored under the bare "model.graph_topology" GGUF string KV -- the common case
    // for single-topology files (every model before Whisper's own multi-topology GGUF, see
    // LOOM_PROCEDURAL_GENERALIZATION.md). Throws loom::LoadError if this file has no BARE topology (e.g.
    // it only has named "model.graph_topology.<name>" KVs -- use topology_json(name) instead).
    const std::string& topology_json() const;

    // A NAMED topology from a multi-topology file: reads "model.graph_topology.<name>" (e.g. "encoder",
    // "decoder" -- LoomLuaBridge::register_module registers each against the SAME GgufModel instance).
    // Throws loom::LoadError if absent.
    const std::string& topology_json(const std::string& name) const;
    bool has_topology(const std::string& name) const;

    // Scalar hyperparameters are stored as typed GGUF KVs under a "loom." namespace (e.g.
    // "loom.n_layer", "loom.rms_norm_eps") -- mirrors how llama.cpp stores its own "llama.*" hparams as
    // first-class KVs rather than burying them in the topology JSON. Throws loom::LoadError if `key`
    // (with "loom." prepended) is absent or has the wrong GGUF value type.
    uint32_t hparam_u32(const std::string& key) const;
    float hparam_f32(const std::string& key) const;
    std::string hparam_str(const std::string& key) const;

    // Convenience: "loom.architecture" string KV.
    std::string architecture() const { return hparam_str("architecture"); }

    // Builds a SymbolEnv pre-populated with every scalar "loom.*" hparam this model exposes (u32/i32/f32
    // KVs only), keyed by their bare name (e.g. "n_layer", not "loom.n_layer") -- used by GraphBuilder to
    // resolve "$"-prefixed attribute expressions in the graph topology.
    SymbolEnv hparam_env() const;

    // Generic KV accessors, unlike hparam_*() these take the FULL key verbatim (no "loom." prefix) --
    // used for namespaces outside "loom.*", e.g. llama.cpp's "tokenizer.ggml.*" vocab schema (see Vocab).
    bool has_kv(const std::string& full_key) const;
    std::string kv_str(const std::string& full_key) const;
    bool kv_bool(const std::string& full_key, bool default_value) const;
    int32_t kv_i32(const std::string& full_key, int32_t default_value) const;
    std::vector<std::string> kv_arr_str(const std::string& full_key) const;
    std::vector<float> kv_arr_f32(const std::string& full_key) const;
    std::vector<int32_t> kv_arr_i32(const std::string& full_key) const;
    std::vector<uint8_t> kv_arr_u8(const std::string& full_key) const;

    ggml_backend_t backend() const { return backend_; }

    GgufModel(const GgufModel&) = delete;
    GgufModel& operator=(const GgufModel&) = delete;

private:
    GgufModel() = default;

    gguf_context_ptr gguf_ctx_;
    ggml_context_ptr meta_ctx_;          // holds the ggml_tensor structs (no_alloc during gguf parse)
    ggml_backend_buffer_ptr weights_buf_; // real backing storage for all weight tensors
    ggml_backend_t backend_ = nullptr;    // not owned by GgufModel
    SymbolTable symbols_;
    // Bare "model.graph_topology" is stored under the key "" ; named "model.graph_topology.<name>" KVs
    // are stored under "<name>". A file always has at least one entry after a successful load() (empty
    // would have thrown LoadError instead).
    std::unordered_map<std::string, std::string> topologies_;
};

} // namespace loom
