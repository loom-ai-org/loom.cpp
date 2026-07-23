#pragma once

#include "loom/core/gguf_model.h"
#include "loom/core/graph_topology.h"

#include <ggml-cpp.h>

#include <random>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

struct lua_State;

namespace loom {

class KvCache;

// Embeds a LuaJIT VM into the engine (see LOOM_PROCEDURAL_GENERALIZATION.md /
// LOOM_MIL_CONVERSION.md): the data-driven replacement for bespoke per-model C++ drivers
// (WhisperDriver, VitsDriver, ...). A driver script (loaded from a GGUF's own "model.driver_script"
// string KV, or any other source) orchestrates one or more pre-registered subgraphs via
// `loom.run_subgraph(...)` plus a small set of host-math bindings, instead of hand-written C++ control
// flow.
//
// Scope: the `Value` variant and binding set below grow as each new model is ported (Whisper first --
// see BACKLOG.md -- then SupertonicTTS/Matcha-TTS added the RNG/duration bindings, then VITS added
// relative-position-table cropping and loom.get_weight for direct raw-weight introspection outside a
// graph computation) -- NOT a general-purpose Lua<->C++ serialization framework, just what real driver
// scripts have needed so far.
class LoomLuaBridge {
public:
    explicit LoomLuaBridge(ggml_backend_t backend);
    ~LoomLuaBridge();
    LoomLuaBridge(const LoomLuaBridge&) = delete;
    LoomLuaBridge& operator=(const LoomLuaBridge&) = delete;

    // Registers a subgraph under `name` so a driver script can invoke it via
    // `loom.run_subgraph(name, n_tokens, n_past, inputs_table)`. `model`/`kv_cache` are NOT owned by the
    // bridge (same non-owning-reference convention as GraphBuilder itself) -- the caller must keep them
    // alive for the bridge's own lifetime. `kv_cache` is null for non-autoregressive modules (e.g.
    // Whisper's encoder); pass one for modules using the ATTENTION primitive's persistent-cache path
    // (e.g. Whisper's decoder).
    void register_module(const std::string& name, GgufModel& model, GraphTopology topo, KvCache* kv_cache = nullptr);

    // Loads `lua_source` into the VM (defines whatever top-level functions/globals it declares) --
    // throws loom::Error on a syntax error.
    void load_script(const std::string& lua_source);

    // A small tagged variant covering exactly what driver scripts need to pass as named arguments to a
    // top-level Lua function and get back as its return value: a scalar, or a flat array of numbers
    // (Lua 5.1/LuaJIT has one numeric type, a double -- integer-valued data like token ids just rounds
    // cleanly through it; callers that need integers cast at the call site).
    using Value = std::variant<double, std::vector<double>>;

    // Calls the top-level Lua function `fn_name` with a single table argument built from `args` (each
    // entry becomes `inputs.<key>` inside the script, matching the design doc's own
    // `function transcribe(inputs)` convention). Throws loom::Error (wrapping the Lua error message) if
    // the function doesn't exist or raises an error.
    Value call(const std::string& fn_name, const std::unordered_map<std::string, Value>& args);

private:
    struct Module {
        GgufModel* model;
        GraphTopology topo;
        KvCache* kv_cache;
        ggml_backend_t backend;
    };

    lua_State* L_;
    ggml_backend_t backend_;
    std::unordered_map<std::string, Module> modules_;

    // Backs loom.seed_rng/loom.gaussian_array/loom.uniform_array -- the SAME engine/distribution shapes
    // (std::mt19937 + std::normal_distribution<float>(0,1) + std::uniform_real_distribution<float>(0,1))
    // every existing hand-written driver already uses, so an exact-match test against a C++ driver stays
    // possible (matches loom.argmax_row's own "mirror the real function verbatim" reasoning). Both
    // distributions share the ONE rng_ stream, same as e.g. KokoroDriver/StyleTTS2Driver's own single
    // rng_ feeding both a normal and a uniform01 distribution object -- draw ORDER against that shared
    // stream matters for an exact match. Persists across calls within one script invocation until
    // re-seeded, same "persistent state" shape as a registered KvCache.
    std::mt19937 rng_;
    std::normal_distribution<float> normal_dist_{0.0f, 1.0f};
    std::uniform_real_distribution<float> uniform_dist_{0.0f, 1.0f};

    // Trampolines registered into the Lua state; each retrieves `this` via its closure's upvalue. See
    // lua_bridge.cpp's own top comment for why every one of these MUST convert C++ exceptions to
    // `luaL_error` internally rather than let them unwind through the Lua C API.
    static int l_run_subgraph(lua_State* L);
    // `loom.run_recurrent(h_module, c_module, sequence_flat, seq_len, input_dim, hidden_dim, reverse)`:
    // steps an LSTM cell over `sequence_flat` (a flat, row-major (seq_len, input_dim) array) one
    // timestep at a time, threading hidden/cell state between GraphBuilder rebuilds exactly like
    // BiLstmStepper's own lstm_cell_step (src/core/bilstm_stepper.cpp) does -- generalized to run off
    // topology NAMES already registered via register_module, the same lookup l_run_subgraph itself uses,
    // instead of BiLstmStepper's own hardcoded 4-GGUF constructor. `h_module`/`c_module` must be the
    // per-timestep cell topologies tools/loom_mil_compiler/recurrent.py's build_lstm_cell_topologies()
    // produces (declaring "layer_input"/"h_prev"/"c_prev" inputs and an "h_new"/"c_new" output
    // respectively). `reverse=true` walks timesteps backward (seq_len-1 down to 0), writing each result
    // to its own real time index in the (still forward-ordered) output -- the same convention
    // BiLstmStepper::run uses for its own backward pass. Returns (output_flat, shape) where output_flat
    // is a flat (seq_len, hidden_dim) array and shape is [hidden_dim, seq_len, 1, 1] (ggml ne[] order,
    // matching loom.run_subgraph's own second return value).
    static int l_run_recurrent(lua_State* L);
    static int l_range(lua_State* L);
    static int l_causal_mask(lua_State* L);
    static int l_zero_mask(lua_State* L);
    static int l_argmax_row(lua_State* L);
    static int l_seed_rng(lua_State* L);
    static int l_gaussian_array(lua_State* L);
    static int l_uniform_array(lua_State* L);
    static int l_expand_by_duration(lua_State* L);
    static int l_pad_crop_relative_embeddings(lua_State* L);
    static int l_get_weight(lua_State* L);
};

} // namespace loom
