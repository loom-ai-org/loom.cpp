#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace loom {

// A declared graph input (SPECIFICATION.md §3): a named tensor the GraphBuilder must materialize before
// walking the node list. `shape` entries are symbol expressions resolved per-call (e.g. "n_tokens",
// "n_kv") -- see SymbolEnv.
struct TensorSpec {
    std::string name;
    std::string dtype; // "f32" | "i32" for milestone 1
    std::vector<std::string> shape;
};

// One computation node: an op name (a PrimitiveRegistry key), its named inputs/outputs (resolved
// against the running SymbolTable during a build), and an open `attrs` object each primitive
// interprets for itself.
struct TopologyNode {
    std::string op;
    std::string name; // optional, for debug/graph tensor naming
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    nlohmann::json attrs = nlohmann::json::object();
};

// Instantiates `nodes` `count_symbol` times, substituting every "{index_var}" occurrence in each
// instantiated node's inputs/outputs/attrs strings with the (0-based) loop index -- this is how a
// per-transformer-layer block avoids duplicating JSON n_layer times (SPECIFICATION.md §3). Only one
// level of repeat_for is supported (no nested repeat-within-repeat); sufficient for milestone 1's
// per-layer blocks.
struct RepeatBlock {
    std::string count_symbol; // e.g. "$n_layer" or "n_layer", evaluated once via SymbolEnv
    std::string index_var;    // e.g. "i"
    std::vector<TopologyNode> nodes;
};

// A top-level entry in the topology's "nodes" array: either a plain node or a repeat_for block.
struct TopologyItem {
    bool is_repeat = false;
    TopologyNode node;   // valid when !is_repeat
    RepeatBlock repeat;  // valid when is_repeat
};

// The parsed form of the JSON graph-topology document embedded in a GGUF's "model.graph_topology" KV.
//
// `output` is kept as the single primary output symbol (== outputs.front()) so every pre-P2 reader of
// this struct keeps compiling and behaving identically for the single-output topologies that make up
// every model on the roadmap today (EXPORT-ROADMAP.md P2 -- BACKLOG.md's implementation sequence).
// `outputs` is the full declared list; a topology with more than one co-equal output symbol (e.g. an
// encoder producing both a hidden-state tensor and a mask) declares JSON's plural "outputs" array
// instead of the singular "output" string -- see GraphTopology::parse.
struct GraphTopology {
    int version = 0;
    std::vector<TensorSpec> inputs;
    std::string output;
    std::vector<std::string> outputs;
    std::vector<TopologyItem> items;

    // Throws loom::SchemaError on malformed JSON, an unsupported "version", or a structurally invalid
    // node/repeat_for block.
    static GraphTopology parse(const std::string& json_text);

    // Whether running this topology needs a persistent KvCache -- true iff it contains an ATTENTION
    // node whose `kv_cache` attr is true (its default; see op_attention). This is DERIVED rather than
    // declared on purpose (KV-CACHE.md decision 5): the graph is the only authority on whether a cache
    // is reachable at all, since op_attention is the sole door to one, so a separate declaration could
    // only ever agree with it or be wrong. The cache's *geometry* is the opposite case -- n_embd_k and
    // the capacity are model facts no graph states -- and is read from the GGUF's own hparams instead
    // (see make_kv_cache in kv_cache.h).
    bool uses_kv_cache() const;

    // Whether running this topology needs a persistent ConvStateCache -- true iff it contains a
    // SHORT_CONV node whose `conv_state` attr is true (its default; see op_short_conv). Derived for
    // exactly the reasons uses_kv_cache() is, and kept as a SECOND predicate rather than folded into a
    // "needs state" one: a pure causal LM needs only the first, a Mamba-style model only the second,
    // and a hybrid both, so collapsing them would over-allocate for two of the three.
    bool uses_conv_state() const;

    // Does any part of this topology's own text name `symbol` -- a declared input's shape expression, a
    // repeat_for count, or any string inside a node's attrs? Conservative by construction: it is a
    // substring test, so "n_past_offset" counts as a mention of "n_past". That direction is the safe
    // one, because the single caller (GraphBuilder, deciding whether `n_past` may be dropped from the
    // key its retained graph is cached under -- BACKLOG.md P4.0.15) turns a "yes" into "rebuild every
    // step", which is merely the behaviour that predates the item.
    bool mentions_symbol(const std::string& symbol) const;
};

} // namespace loom
