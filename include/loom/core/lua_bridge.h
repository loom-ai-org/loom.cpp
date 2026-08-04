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
class ConvStateCache;

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
//
// ---------------------------------------------------------------------------------------------------
// THE CRITERION A NEW BINDING MUST MEET (EXPORT-PREPARATION.md P4.0.8/decision 1, 2026-08-01)
// ---------------------------------------------------------------------------------------------------
// This engine is deliberately lean -- the target is edge devices, and the project's stated selling
// point is that adding a model family costs Python in the exporter, not C++ here. Every binding added
// below is a permanent tax on that leanness and on the "four headers are the whole contract" claim
// (EXPORT-PREPARATION.md 1.3), so the bar is:
//
//   A binding must be a GENERIC HOST-SIDE TENSOR OP, not model adaptation.
//
// Generic host-side tensor op:
//   * it reads no model config -- no hyperparameter, no architecture name, no layer count reaches it;
//     everything it needs arrives as a call argument from the driver script;
//   * it earns its place in C++ for a structural reason, not a convenience one. In practice that
//     reason has always been the same: the operation's OUTPUT LENGTH IS DATA-DEPENDENT, so it cannot
//     live in a static topology, whose shapes are fixed at export time by the declared axes;
//   * two unrelated families could use it unchanged.
//
// Model adaptation -- does NOT belong here, and belongs in the exporter instead:
//   * anything whose behaviour branches on which model is running;
//   * anything that encodes a family's orchestration (a sampler, a decode loop, a phase ordering).
//     These are Lua, and the hard cases already are: the CFM Euler loop, the ADPM2 diffusion sampler
//     and BiLSTM stepping all run as driver script today. If ADPM2 did not need C++, essentially no
//     orchestration shape does.
//
// The two bindings that look family-specific, labelled against the criterion (both were reviewed and
// KEPT on 2026-08-01; see their declarations below for the per-binding argument):
//
//   loom.expand_by_duration            -- PASSES. Repeat-rows-by-count. Reads no config; output length
//                                         is sum(durations), known only at run time.
//   loom.pad_crop_relative_embeddings  -- PASSES. Symmetric pad-or-crop of a relative-position table
//                                         about its centre. Reads no config; the crop width follows the
//                                         run-time sequence length.
//
// Both name a family in their comments because a family is where they were first needed, which is
// provenance, not a dependency.
class LoomLuaBridge {
public:
    explicit LoomLuaBridge(ggml_backend_t backend);
    ~LoomLuaBridge();
    LoomLuaBridge(const LoomLuaBridge&) = delete;
    LoomLuaBridge& operator=(const LoomLuaBridge&) = delete;

    // Registers a subgraph under `name` so a driver script can invoke it via
    // `loom.run_subgraph(name, axes_table, inputs_table)`, where `axes_table` is `{axis_name = value,
    // ...}` (EXPORT-ROADMAP.md R1 -- e.g. `{n_tokens=12, n_past=0}` or `{n_samples=16000}`, whatever
    // axis names this specific topology declares). `model`/`kv_cache`/`conv_state` are NOT owned by the
    // bridge (same non-owning-reference convention as GraphBuilder itself) -- the caller must keep them
    // alive for the bridge's own lifetime. `kv_cache` is null for non-autoregressive modules (e.g.
    // Whisper's encoder); pass one for modules using the ATTENTION primitive's persistent-cache path
    // (e.g. Whisper's decoder). `conv_state` is the same for SHORT_CONV -- null unless the topology has
    // stateful convolutions, which for the models on this roadmap means an LFM2-style hybrid.
    //
    // Binding both HERE rather than passing them per call is what gives a Lua driver persistence with no
    // address ever crossing the scripting boundary: the script names a module, and the C++ side knows
    // which stores that module owns.
    void register_module(const std::string& name, GgufModel& model, GraphTopology topo,
                         KvCache* kv_cache = nullptr, ConvStateCache* conv_state = nullptr);

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
        ConvStateCache* conv_state;
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
    // `loom.expand_by_duration(rows_flat, T, C, durations)`: repeats row `t` of a (T, C) row-major array
    // `durations[t]` times, returning a flat (sum(durations), C) array. MEETS the binding criterion
    // above: it reads no model config -- T, C and the durations are all call arguments -- and it is in
    // C++ because `sum(durations)` is only known at run time, so no static topology can declare the
    // output shape. VITS/Matcha/Kokoro all call it for duration expansion, and it would serve any
    // family with a length predictor unchanged.
    static int l_expand_by_duration(lua_State* L);
    // `loom.pad_crop_relative_embeddings(raw, window_size, k_channels, length)`: symmetrically pads or
    // crops a relative-position table about its centre to cover `length`. MEETS the binding criterion
    // above for the same two reasons -- every parameter is a call argument, and whether this pads or
    // crops (and by how much) follows the run-time sequence length, which a static topology cannot
    // express. First needed by VITS's relative-attention layers; nothing in it is VITS-specific.
    static int l_pad_crop_relative_embeddings(lua_State* L);
    static int l_get_weight(lua_State* L);
};

} // namespace loom
