#pragma once

#include "loom/core/symbol_table.h"

#include <ggml.h>
#include <nlohmann/json_fwd.hpp>

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class KvCache;
class ConvStateCache;

// Passed to every primitive function. `ctx` is the ephemeral no_alloc compute context nodes get built
// in; `symbols` resolves the JSON graph's "$"-prefixed attribute expressions (see symbol_table.h);
// `kv_cache` is non-null only when the ATTENTION primitive needs to read/write persistent KV storage
// and `conv_state` only when SHORT_CONV does (every other op ignores both).
//
// Two pointers rather than one aggregate because the two stores answer different questions and a
// topology can genuinely need either alone: a pure causal LM has K/V and no conv state, a Mamba-style
// model would have the reverse, and a hybrid like LFM2 has both.
struct PrimitiveContext {
    ggml_context* ctx;
    SymbolEnv& symbols;
    KvCache* kv_cache = nullptr;
    ConvStateCache* conv_state = nullptr;

    // Side-effecting nodes (ATTENTION's KV-cache writes, SHORT_CONV's state writes) that must be
    // included in the
    // compute graph even though nothing "downstream" of the topology's declared output references them
    // via ggml src[] pointers -- a plain memory write into the persistent KV cache has no such data
    // dependency to the eventual read of that same memory. GraphBuilder expands each of these via
    // ggml_build_forward_expand() *before* expanding the main output, in the order primitives push them,
    // guaranteeing writes execute before the reads that depend on them being done (ggml_cgraph nodes run
    // strictly in array order). Null for graphs with no such ops; primitives that don't need it just
    // leave it untouched.
    std::vector<ggml_tensor*>* side_effects = nullptr;
};

using PrimitiveFn = std::function<std::vector<ggml_tensor*>(
    PrimitiveContext& pc,
    const std::vector<ggml_tensor*>& inputs,
    const nlohmann::json& attrs)>;

// String op-name -> ggml call dispatch table (SPECIFICATION.md §2.B). Populated by static initializers
// in each primitives_*.cpp via LOOM_REGISTER_OP, so adding a new op is purely additive -- no central
// switch statement to edit.
class PrimitiveRegistry {
public:
    static PrimitiveRegistry& instance();

    void register_op(const std::string& name, PrimitiveFn fn);
    bool has(const std::string& name) const;

    // Throws loom::UnknownOpError if `name` has no registered primitive.
    const PrimitiveFn& get(const std::string& name) const;

private:
    std::unordered_map<std::string, PrimitiveFn> table_;
};

#define LOOM_REGISTER_OP(op_name, fn)                                                             \
    namespace {                                                                                    \
    struct op_name##_registrar_t {                                                                  \
        op_name##_registrar_t() { ::loom::PrimitiveRegistry::instance().register_op(#op_name, fn); } \
    };                                                                                               \
    static const op_name##_registrar_t op_name##_registrar_instance;                                  \
    }

// Reads a numeric attribute that may be given either as a literal JSON number or as a SymbolEnv
// expression string (e.g. attrs["eps"] may be 1e-5 or "$rms_norm_eps"). Throws loom::SchemaError if
// `key` is absent or neither a number nor a string.
double resolve_attr_number(const nlohmann::json& attrs, const std::string& key, const SymbolEnv& env);

// Same as resolve_attr_number, rounded to the nearest int64.
int64_t resolve_attr_int(const nlohmann::json& attrs, const std::string& key, const SymbolEnv& env);

// Resolves a JSON array attribute where each element may independently be a literal number or a
// SymbolEnv expression string (e.g. RESHAPE's "shape": ["n_embd_head_k", "n_head", "n_tokens"]).
std::vector<int64_t> resolve_attr_int_array(const nlohmann::json& attrs, const std::string& key, const SymbolEnv& env);

} // namespace loom
