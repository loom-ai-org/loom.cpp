#pragma once

#include <ggml.h>

#include <string>
#include <unordered_map>

namespace loom {

// Maps weight names and node-output names to live tensors during graph construction. Seeded from a
// GgufModel's loaded weights, then grows as the GraphBuilder walks the topology and registers each
// node's output(s) under their declared name(s).
using SymbolTable = std::unordered_map<std::string, ggml_tensor*>;

// Holds named numeric values -- GGUF hyperparameters (n_layer, rms_norm_eps, ...) plus per-call runtime
// values (n_tokens, n_past, n_kv) -- and evaluates the small "$"-prefixed expression language used in
// the JSON graph topology's node attributes, e.g. "$rms_norm_eps" or "1/sqrt($n_embd_head_k)".
//
// Grammar (deliberately minimal -- just enough for RoPE/attention scale attributes and derived
// declared-input shapes, e.g. a conv-stride output length, not a general expression language):
//   expr   := term (('+' | '-') term)*
//   term   := factor (('*' | '/') factor)*
//   factor := '-' factor | number | '$' ident | 'sqrt' '(' expr ')' | 'floor' '(' expr ')' | '(' expr ')'
//   ident  := [A-Za-z_][A-Za-z0-9_]*
//
// Note: shape dimensions are rounded via std::llround() after eval() (see GraphBuilder), which rounds
// halfway values AWAY FROM ZERO -- wrong for conv-style output-length formulas that need floor. Any
// expression needing floor-division semantics (e.g. "floor(($n_tokens - 1)/2) + 1") must call floor()
// explicitly rather than relying on the outer rounding.
class SymbolEnv {
public:
    void set(const std::string& name, double value);
    bool has(const std::string& name) const;

    // Throws loom::SchemaError if `name` isn't bound.
    double get(const std::string& name) const;

    // Evaluates a "$"-prefixed expression string (e.g. "$n_layer", "1/sqrt($n_embd_head_k)"). The
    // leading '$' is optional -- eval("n_layer") and eval("$n_layer") are equivalent when the whole
    // string is a single symbol reference; it matters for multi-symbol expressions where only
    // sub-references need the sigil (e.g. "1/sqrt($n_embd_head_k)" -- the "1" and "sqrt(...)" wrapper
    // are plain syntax, only "$n_embd_head_k" is a symbol lookup).
    // Throws loom::SchemaError on malformed expressions or unbound symbols.
    double eval(const std::string& expr) const;

private:
    std::unordered_map<std::string, double> values_;
};

} // namespace loom
