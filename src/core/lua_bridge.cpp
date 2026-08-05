#include "loom/core/lua_bridge.h"

#include "loom/core/duration_aligner.h"
#include "loom/core/graph_builder.h"
#include "loom/core/conv_state_cache.h"

#include <functional>
#include "loom/core/kv_cache.h"
#include "loom/core/relative_position.h"
#include "loom/loom_errors.h"

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

#include <ggml-backend.h>

#include <cmath>
#include <limits>
#include <new>

// IMPORTANT: LuaJIT (like PUC Lua 5.1, whose C API it implements) reports its own internal errors via
// `longjmp`, which does NOT run C++ destructors -- throwing a C++ exception out of a function called BY
// Lua (a lua_CFunction reached via lua_pcall) is undefined behavior on this build (LuaJIT's default,
// non-"external unwind" configuration). Every lua_CFunction trampoline below therefore wraps its body in
// try/catch and converts any loom::Error/std::exception into `luaL_error(...)`, which itself uses Lua's
// OWN longjmp-based error path -- never lets a C++ exception cross into/through the Lua C stack. Do not
// remove this discipline when adding new bindings.

namespace loom {

namespace {

void push_number_array(lua_State* L, const double* data, size_t n) {
    lua_createtable(L, static_cast<int>(n), 0);
    for (size_t i = 0; i < n; ++i) {
        lua_pushnumber(L, data[i]);
        lua_rawseti(L, -2, static_cast<int>(i + 1)); // Lua arrays are 1-indexed
    }
}

void push_number_array(lua_State* L, const std::vector<double>& v) { push_number_array(L, v.data(), v.size()); }

void push_number_array(lua_State* L, const std::vector<float>& v) {
    std::vector<double> tmp(v.begin(), v.end());
    push_number_array(L, tmp);
}

// Reads a Lua array table (1-indexed, `#`-length) at `idx` into a flat double vector.
std::vector<double> read_number_array(lua_State* L, int idx) {
    luaL_checktype(L, idx, LUA_TTABLE);
    const auto n = static_cast<size_t>(lua_objlen(L, idx));
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) {
        lua_rawgeti(L, idx, static_cast<int>(i + 1));
        out[i] = luaL_checknumber(L, -1);
        lua_pop(L, 1);
    }
    return out;
}

// Sets `tensor`'s data from a Lua array table at `value_idx`, dispatching on the REAL declared ggml
// type (no per-input type annotation needed in the Lua script -- the topology itself is the source of
// truth, same principle the design doc's own binding sketch relies on).
void set_tensor_from_lua_array(lua_State* L, int value_idx, ggml_tensor* tensor) {
    const std::vector<double> vals = read_number_array(L, value_idx);
    const size_t expected_size = ggml_nelements(tensor);
    if (vals.size() != expected_size) {
        // Lua's own pushfstring (which luaL_error routes through) doesn't understand the C99 "%zu"
        // length modifier -- it silently mis-consumes the varargs, truncating the message before the
        // actual numbers ever print. "%d" is one of the directives Lua's formatter does support.
        luaL_error(L, "loom.run_subgraph: input tensor '%s' size mismatch. Expected %d elements, got %d elements from Lua",
                    ggml_get_name(tensor), static_cast<int>(expected_size), static_cast<int>(vals.size()));
    }
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> f(vals.begin(), vals.end());
        ggml_backend_tensor_set(tensor, f.data(), 0, f.size() * sizeof(float));
    } else if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> ivals(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) ivals[i] = static_cast<int32_t>(std::llround(vals[i]));
        ggml_backend_tensor_set(tensor, ivals.data(), 0, ivals.size() * sizeof(int32_t));
    } else {
        luaL_error(L, "loom.run_subgraph: input tensor has an unsupported ggml type (only f32/i32 are marshalled)");
    }
}

// Reads `tensor`'s data back out as a flat double vector -- the same f32/i32 dispatch
// l_run_subgraph's single-output path used inline before P2 generalized it to N outputs.
std::vector<double> read_tensor_as_doubles(lua_State* L, ggml_tensor* tensor) {
    const auto n = static_cast<size_t>(ggml_nelements(tensor));
    std::vector<double> out(n);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> f(n);
        ggml_backend_tensor_get(tensor, f.data(), 0, n * sizeof(float));
        out.assign(f.begin(), f.end());
    } else if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> iv(n);
        ggml_backend_tensor_get(tensor, iv.data(), 0, n * sizeof(int32_t));
        out.assign(iv.begin(), iv.end());
    } else {
        luaL_error(L, "loom.run_subgraph: output tensor has an unsupported ggml type");
    }
    return out;
}

LoomLuaBridge* bridge_from_upvalue(lua_State* L) {
    return static_cast<LoomLuaBridge*>(lua_touserdata(L, lua_upvalueindex(1)));
}

// Reads a Lua table of string->number pairs at `idx` into a `DynamicAxes` (EXPORT-ROADMAP.md R1: a
// topology declares its own axis names, e.g. {n_samples=16000} or {n_tokens=12, n_past=0}, rather than
// this binding assuming every topology has exactly the same two positional axes).
DynamicAxes read_axes_table(lua_State* L, int idx) {
    luaL_checktype(L, idx, LUA_TTABLE);
    DynamicAxes axes;
    lua_pushnil(L);
    while (lua_next(L, idx) != 0) {
        // key at -2, value at -1
        if (lua_type(L, -2) != LUA_TSTRING) {
            luaL_error(L, "loom.run_subgraph: axes table keys must be strings (axis names)");
        }
        axes[lua_tostring(L, -2)] = luaL_checknumber(L, -1);
        lua_pop(L, 1); // pop value, keep key for lua_next
    }
    return axes;
}

// Everything `run_subgraph` and `run_subgraph_argmax` share: build the graph for `axes`, fill every
// declared input from the Lua table at `inputs_idx`, compute, and hand the result to `emit` while the
// builder that owns its memory is still alive. Takes the module's pieces individually rather than the
// Module struct so it can live here, in the anonymous namespace, next to the code it serves.
//
// Extracted rather than duplicated because the two entry points differ ONLY in what they do with the
// outputs -- a copy would be free to drift on cache wiring or input validation, which is exactly the
// class of difference nothing would catch.
int compute_and_emit(lua_State* L, const char* fname, const char* module_name, GgufModel& model,
                      const GraphTopology& topo, ggml_backend_t backend, KvCache* kv_cache,
                      ConvStateCache* conv_state, const DynamicAxes& axes, int inputs_idx,
                      const std::function<int(GraphBuilder::BuildResult&)>& emit) {
    GraphBuilder builder(topo, model, backend, kv_cache, /*compute_meta_bytes=*/32 * 1024 * 1024,
                          conv_state);
    GraphBuilder::BuildResult r = builder.build(axes);

    lua_pushnil(L);
    while (lua_next(L, inputs_idx) != 0) {
        if (lua_type(L, -2) != LUA_TSTRING) {
            return luaL_error(L, "%s: inputs table keys must be strings", fname);
        }
        const std::string name = lua_tostring(L, -2);
        const auto input_it = r.input_tensors.find(name);
        if (input_it == r.input_tensors.end()) {
            return luaL_error(L, "%s: module '%s' has no declared input '%s'", fname, module_name,
                               name.c_str());
        }
        set_tensor_from_lua_array(L, -1, input_it->second);
        lua_pop(L, 1);
    }

    ggml_backend_graph_compute(backend, r.graph);
    return emit(r);
}

} // namespace

int LoomLuaBridge::l_run_subgraph(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const char* module_name = luaL_checkstring(L, 1);
        // EXPORT-ROADMAP.md R1: axes_table is {axis_name = value, ...} (e.g. {n_tokens=12, n_past=0}
        // or {n_samples=16000}) -- a topology declares its OWN axis names now, replacing the old
        // positional (n_tokens, n_past) pair every module was assumed to share.
        const DynamicAxes axes = read_axes_table(L, 2);
        luaL_checktype(L, 3, LUA_TTABLE);

        const auto it = self->modules_.find(module_name);
        if (it == self->modules_.end()) {
            return luaL_error(L, "loom.run_subgraph: unregistered module '%s'", module_name);
        }
        Module& mod = it->second;

        return compute_and_emit(L, "loom.run_subgraph", module_name, *mod.model, mod.topo, mod.backend,
                                 mod.kv_cache, mod.conv_state, axes, 3,
                                 [L](GraphBuilder::BuildResult& r) {
        // Returns every declared output's DATA first (in the topology's own declared order), THEN
        // every declared output's SHAPE in that same order -- e.g. for two outputs: (data1, data2,
        // shape1, shape2). For the single-output topology every model on the roadmap still uses as of
        // P2 (EXPORT-ROADMAP.md), that's exactly (data, shape), byte-for-byte the same two return
        // values this function always produced -- callers that only ever wrote
        // `local out, shape = loom.run_subgraph(...)` see no change. A caller only interested in DATA
        // for an N-output module can write `local out1, out2 = loom.run_subgraph(...)` (Lua discards
        // the trailing shape values it doesn't capture); one that also wants shapes captures all N data
        // locals first, then N shape locals (see driver_ir.py's check_subgraph_calls, which enforces
        // exactly this ordering at export time).
        for (ggml_tensor* out : r.outputs) {
            push_number_array(L, read_tensor_as_doubles(L, out));
        }
        for (ggml_tensor* out : r.outputs) {
            const std::vector<double> shape = {static_cast<double>(out->ne[0]), static_cast<double>(out->ne[1]),
                                                static_cast<double>(out->ne[2]), static_cast<double>(out->ne[3])};
            push_number_array(L, shape);
        }
        return static_cast<int>(r.outputs.size() * 2);
        });
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.run_subgraph: %s", e.what());
    }
}

// `loom.run_subgraph_argmax(module, axes, inputs, row)` -- run the module and return ONE number, the
// argmax of the requested row of its first output. `row` is 0-based; a negative row means the last.
//
// **Why this exists: `run_subgraph` cannot return a large logits tensor at all.** It marshals every
// output element into a Lua table, and LuaJIT's array part tops out near 2^27 entries -- so a
// 262144-wide vocab (Gemma 3) overflows at ~512 prompt tokens, and a driver whose only use for those
// logits is one argmax pays a 157M-element table to compute a single integer. Doing the reduction on
// the tensor removes both the ceiling and the copy: nothing crosses the boundary but the answer.
//
// This keeps the Lua boundary a *per-step* boundary rather than a per-logit one, the same reasoning
// KV-CACHE.md §1.1 gives for not driving attention from Lua.
int LoomLuaBridge::l_run_subgraph_argmax(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const char* module_name = luaL_checkstring(L, 1);
        const DynamicAxes axes = read_axes_table(L, 2);
        luaL_checktype(L, 3, LUA_TTABLE);
        const auto requested_row = static_cast<int64_t>(luaL_checknumber(L, 4));

        const auto it = self->modules_.find(module_name);
        if (it == self->modules_.end()) {
            return luaL_error(L, "loom.run_subgraph_argmax: unregistered module '%s'", module_name);
        }
        Module& mod = it->second;

        return compute_and_emit(L, "loom.run_subgraph_argmax", module_name, *mod.model, mod.topo,
                                 mod.backend, mod.kv_cache, mod.conv_state, axes, 3,
                                 [L, requested_row](GraphBuilder::BuildResult& r) {
            ggml_tensor* out = r.outputs.front();
            if (out->type != GGML_TYPE_F32) {
                return luaL_error(L, "loom.run_subgraph_argmax: output must be f32");
            }
            const int64_t n_vocab = out->ne[0];
            const int64_t n_rows = out->ne[1];
            const int64_t row = requested_row < 0 ? n_rows - 1 : requested_row;
            if (n_vocab <= 0 || row < 0 || row >= n_rows) {
                return luaL_error(L, "loom.run_subgraph_argmax: row %d out of range for an output with "
                                      "%d row(s) of width %d", static_cast<int>(requested_row),
                                  static_cast<int>(n_rows), static_cast<int>(n_vocab));
            }
            // One row only -- the whole point. `nb[1]` is the row stride, so this reads n_vocab floats
            // and never touches the other rows.
            std::vector<float> logits(static_cast<size_t>(n_vocab));
            ggml_backend_tensor_get(out, logits.data(), static_cast<size_t>(row) * out->nb[1],
                                     logits.size() * sizeof(float));
            int64_t best = 0;
            for (int64_t i = 1; i < n_vocab; ++i) {
                if (logits[static_cast<size_t>(i)] > logits[static_cast<size_t>(best)]) best = i;
            }
            lua_pushnumber(L, static_cast<lua_Number>(best));
            return 1;
        });
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.run_subgraph_argmax: %s", e.what());
    }
}

int LoomLuaBridge::l_run_recurrent(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const char* h_module_name = luaL_checkstring(L, 1);
        const char* c_module_name = luaL_checkstring(L, 2);
        const std::vector<double> sequence = read_number_array(L, 3);
        const auto seq_len = static_cast<uint32_t>(luaL_checknumber(L, 4));
        const auto input_dim = static_cast<uint32_t>(luaL_checknumber(L, 5));
        const auto hidden_dim = static_cast<uint32_t>(luaL_checknumber(L, 6));
        const bool reverse = lua_toboolean(L, 7) != 0;

        if (sequence.size() != static_cast<size_t>(seq_len) * input_dim) {
            return luaL_error(L, "loom.run_recurrent: sequence has %d elements, expected seq_len*input_dim=%d",
                               static_cast<int>(sequence.size()), static_cast<int>(seq_len * input_dim));
        }

        const auto h_it = self->modules_.find(h_module_name);
        if (h_it == self->modules_.end()) {
            return luaL_error(L, "loom.run_recurrent: unregistered module '%s'", h_module_name);
        }
        const auto c_it = self->modules_.find(c_module_name);
        if (c_it == self->modules_.end()) {
            return luaL_error(L, "loom.run_recurrent: unregistered module '%s'", c_module_name);
        }
        Module& h_mod = h_it->second;
        Module& c_mod = c_it->second;

        // A fresh GraphBuilder per direction (not per timestep) -- same "build once, rebuild per call"
        // shape BiLstmStepper's own constructor uses; GraphBuilder::build() itself is still called once
        // per timestep below (a step's h/c depend on the PREVIOUS step's real output values, so each
        // timestep genuinely needs its own compute, unlike loom.run_subgraph's single one-shot call).
        GraphBuilder h_builder(h_mod.topo, *h_mod.model, h_mod.backend, h_mod.kv_cache);
        GraphBuilder c_builder(c_mod.topo, *c_mod.model, c_mod.backend, c_mod.kv_cache);

        std::vector<float> h(hidden_dim, 0.0f);
        std::vector<float> c(hidden_dim, 0.0f);
        std::vector<double> out(static_cast<size_t>(seq_len) * hidden_dim);

        for (uint32_t step = 0; step < seq_len; ++step) {
            // `reverse` walks timesteps backward through the SAME (forward-ordered) `sequence` input,
            // writing each result to its own real time index `t` -- mirroring BiLstmStepper::run's own
            // backward pass (src/core/bilstm_stepper.cpp) exactly, so a reverse-direction MIL `lstm` op
            // (see recurrent.py's own "direction" handling) doesn't need its caller to pre-reverse
            // anything itself.
            const uint32_t t = reverse ? (seq_len - 1 - step) : step;
            std::vector<float> layer_input(input_dim);
            for (uint32_t k = 0; k < input_dim; ++k) {
                layer_input[k] = static_cast<float>(sequence[static_cast<size_t>(t) * input_dim + k]);
            }

            GraphBuilder::BuildResult hr = h_builder.build({{"n_tokens", 0}, {"n_past", 0}});
            ggml_backend_tensor_set(hr.input_tensors.at("layer_input"), layer_input.data(), 0,
                                     layer_input.size() * sizeof(float));
            ggml_backend_tensor_set(hr.input_tensors.at("h_prev"), h.data(), 0, h.size() * sizeof(float));
            ggml_backend_tensor_set(hr.input_tensors.at("c_prev"), c.data(), 0, c.size() * sizeof(float));
            ggml_backend_graph_compute(h_mod.backend, hr.graph);
            std::vector<float> h_new(hidden_dim);
            ggml_backend_tensor_get(hr.output, h_new.data(), 0, h_new.size() * sizeof(float));

            GraphBuilder::BuildResult cr = c_builder.build({{"n_tokens", 0}, {"n_past", 0}});
            ggml_backend_tensor_set(cr.input_tensors.at("layer_input"), layer_input.data(), 0,
                                     layer_input.size() * sizeof(float));
            ggml_backend_tensor_set(cr.input_tensors.at("h_prev"), h.data(), 0, h.size() * sizeof(float));
            ggml_backend_tensor_set(cr.input_tensors.at("c_prev"), c.data(), 0, c.size() * sizeof(float));
            ggml_backend_graph_compute(c_mod.backend, cr.graph);
            std::vector<float> c_new(hidden_dim);
            ggml_backend_tensor_get(cr.output, c_new.data(), 0, c_new.size() * sizeof(float));

            h = h_new;
            c = c_new;
            for (uint32_t k = 0; k < hidden_dim; ++k) {
                out[static_cast<size_t>(t) * hidden_dim + k] = h[k];
            }
        }

        push_number_array(L, out);
        const std::vector<double> shape = {static_cast<double>(hidden_dim), static_cast<double>(seq_len), 1.0, 1.0};
        push_number_array(L, shape);
        return 2;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.run_recurrent: %s", e.what());
    }
}

int LoomLuaBridge::l_range(lua_State* L) {
    try {
        const auto start = static_cast<int64_t>(luaL_checknumber(L, 1));
        const auto count = static_cast<int64_t>(luaL_checknumber(L, 2));
        if (count < 0) return luaL_error(L, "loom.range: count must be >= 0");
        std::vector<double> out(static_cast<size_t>(count));
        for (int64_t i = 0; i < count; ++i) out[static_cast<size_t>(i)] = static_cast<double>(start + i);
        push_number_array(L, out);
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.range: %s", e.what());
    }
}

// Matches WhisperDriver::fill_decoder_inputs's exact causal-triangle formula
// (src/core/whisper_driver.cpp): mask[i*n_kv+j] gates query token i (absolute position n_past+i)
// attending to self-attention KV cell j -- 0.0 if j <= n_past+i, else -inf. n_kv = n_past+n_tokens.
// `loom.causal_mask(n_tokens, n_past [, window])`.
//
// The optional third argument is sliding-window attention (BACKLOG.md P4.0.11a): a query at absolute
// position p attends to keys in `(p - window, p]` instead of `[0, p]`. Omitted, zero or negative means
// no window, which is the full-causal mask every caller before this got and still gets -- so this is
// additive, and no existing driver's output moves.
//
// **The window lives in the MASK, not in the cache.** ggml_soft_max_ext takes an arbitrary
// [n_kv, n_tokens] mask, so banding it is the whole of what a windowed model needs to be CORRECT; the
// cache still holds every key and merely spends n_ctx where it could spend `window`. Making it cheap is
// a separate and much larger item (a ring buffer breaks KvCache's "a plain view over [0, n_kv) suffices
// for reads" invariant, and per-layer capacity breaks its single kv_size) -- deliberately not attempted
// here, per P4.0.11's own split.
int LoomLuaBridge::l_causal_mask(lua_State* L) {
    try {
        const auto n_tokens = static_cast<uint32_t>(luaL_checknumber(L, 1));
        const auto n_past = static_cast<uint32_t>(luaL_checknumber(L, 2));
        // luaL_optnumber, not luaL_checknumber: every existing call site passes two arguments.
        const double window_arg = luaL_optnumber(L, 3, 0.0);
        const bool windowed = window_arg > 0.0;
        const auto window = static_cast<uint32_t>(window_arg > 0.0 ? window_arg : 0.0);
        const uint32_t n_kv = n_past + n_tokens;
        std::vector<double> mask(static_cast<size_t>(n_kv) * n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const uint32_t query_pos = n_past + i;
            for (uint32_t j = 0; j < n_kv; ++j) {
                // `query_pos - j < window` on unsigned values is only meaningful once j <= query_pos has
                // already been established, which the short-circuit guarantees.
                const bool visible = (j <= query_pos) && (!windowed || (query_pos - j) < window);
                mask[static_cast<size_t>(i) * n_kv + j] =
                    visible ? 0.0 : -std::numeric_limits<double>::infinity();
            }
        }
        push_number_array(L, mask);
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.causal_mask: %s", e.what());
    }
}

int LoomLuaBridge::l_zero_mask(lua_State* L) {
    try {
        const auto rows = static_cast<uint32_t>(luaL_checknumber(L, 1));
        const auto cols = static_cast<uint32_t>(luaL_checknumber(L, 2));
        std::vector<double> mask(static_cast<size_t>(rows) * cols, 0.0);
        push_number_array(L, mask);
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.zero_mask: %s", e.what());
    }
}

// Matches WhisperDriver::argmax, but takes the row index explicitly so the driver script can select
// "last prompt token" during prefill vs "the only row" during incremental decode (mirrors
// transcribe()'s own two call sites) -- kept as one host call rather than a Lua-side loop over
// potentially vocab-sized arrays.
int LoomLuaBridge::l_argmax_row(lua_State* L) {
    try {
        const std::vector<double> flat = read_number_array(L, 1);
        const auto n_vocab = static_cast<uint32_t>(luaL_checknumber(L, 2));
        const auto row_index = static_cast<uint32_t>(luaL_checknumber(L, 3));
        if (n_vocab == 0 || (static_cast<size_t>(row_index) + 1) * n_vocab > flat.size()) {
            return luaL_error(L, "loom.argmax_row: row_index/n_vocab out of bounds for the given array");
        }
        const double* row = flat.data() + static_cast<size_t>(row_index) * n_vocab;
        uint32_t best = 0;
        double best_val = row[0];
        for (uint32_t i = 1; i < n_vocab; ++i) {
            if (row[i] > best_val) {
                best_val = row[i];
                best = i;
            }
        }
        lua_pushnumber(L, static_cast<double>(best));
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.argmax_row: %s", e.what());
    }
}

// Resets the bridge's own std::mt19937 -- the SAME engine every hand-written driver's own RNG uses
// (VitsDriver/SupertonicDriver/MatchaDriver all construct `std::mt19937 rng(seed)` directly), so a
// script that calls loom.seed_rng(seed) then loom.gaussian_array(n) in the SAME order a C++ driver
// draws its own noise produces bit-identical values -- what keeps an exact-match test against that
// driver possible.
int LoomLuaBridge::l_seed_rng(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const auto seed = static_cast<uint32_t>(luaL_checknumber(L, 1));
        self->rng_.seed(seed);
        self->normal_dist_.reset();
        self->uniform_dist_.reset();
        return 0;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.seed_rng: %s", e.what());
    }
}

int LoomLuaBridge::l_gaussian_array(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const auto n = static_cast<int64_t>(luaL_checknumber(L, 1));
        if (n < 0) return luaL_error(L, "loom.gaussian_array: n must be >= 0");
        std::vector<double> out(static_cast<size_t>(n));
        for (double& v : out) v = static_cast<double>(self->normal_dist_(self->rng_));
        push_number_array(L, out);
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.gaussian_array: %s", e.what());
    }
}

// Uniform(0,1) counterpart to loom.gaussian_array, sharing the SAME rng_ stream (KokoroDriver's and
// StyleTTS2Driver's own synthesize() each construct ONE std::mt19937 rng_ and draw from both a
// std::normal_distribution AND a std::uniform_real_distribution against it -- e.g. SineGen's own
// rand_ini[1..dim-1] uses uniform01(rng_), drawn BEFORE its noise_tc gaussian draws -- so a script must
// call loom.uniform_array/loom.gaussian_array in the SAME order the C++ driver draws them to stay
// bit-exact).
int LoomLuaBridge::l_uniform_array(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const auto n = static_cast<int64_t>(luaL_checknumber(L, 1));
        if (n < 0) return luaL_error(L, "loom.uniform_array: n must be >= 0");
        std::vector<double> out(static_cast<size_t>(n));
        for (double& v : out) v = static_cast<double>(self->uniform_dist_(self->rng_));
        push_number_array(L, out);
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.uniform_array: %s", e.what());
    }
}

// Thin wrapper around loom::expand_by_duration (include/loom/core/duration_aligner.h), already proven
// by MatchaDriver's own C++ implementation. `rows_flat` is a plain row-major [T,C] flat array (T rows,
// C floats each, T slowest -- a HOST-only convention with no ggml axis-order subtlety, unlike
// loom.run_subgraph's own tensor marshalling). Returns a flat [sum(durations),C] array in the same
// convention.
int LoomLuaBridge::l_expand_by_duration(lua_State* L) {
    try {
        const std::vector<double> rows_flat = read_number_array(L, 1);
        const auto T = static_cast<uint32_t>(luaL_checknumber(L, 2));
        const auto C = static_cast<uint32_t>(luaL_checknumber(L, 3));
        const std::vector<double> durations_d = read_number_array(L, 4);
        if (rows_flat.size() != static_cast<size_t>(T) * C || durations_d.size() != T) {
            return luaL_error(L, "loom.expand_by_duration: rows_flat/durations size mismatch with T,C");
        }

        std::vector<std::vector<float>> seq(T);
        for (uint32_t t = 0; t < T; ++t) {
            seq[t].assign(rows_flat.begin() + static_cast<size_t>(t) * C, rows_flat.begin() + static_cast<size_t>(t + 1) * C);
        }
        std::vector<uint32_t> durations(T);
        for (uint32_t t = 0; t < T; ++t) durations[t] = static_cast<uint32_t>(std::llround(durations_d[t]));

        const std::vector<std::vector<float>> expanded = expand_by_duration(seq, durations);
        std::vector<double> out;
        out.reserve(expanded.size() * static_cast<size_t>(C));
        for (const auto& row : expanded) out.insert(out.end(), row.begin(), row.end());
        push_number_array(L, out);
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.expand_by_duration: %s", e.what());
    }
}

// Thin wrapper around loom::pad_crop_relative_embeddings (include/loom/core/relative_position.h),
// already proven by VitsDriver's own C++ implementation.
int LoomLuaBridge::l_pad_crop_relative_embeddings(lua_State* L) {
    try {
        const std::vector<double> raw_d = read_number_array(L, 1);
        const auto window_size = static_cast<int64_t>(luaL_checknumber(L, 2));
        const auto k_channels = static_cast<int64_t>(luaL_checknumber(L, 3));
        const auto length = static_cast<int64_t>(luaL_checknumber(L, 4));
        std::vector<float> raw(raw_d.begin(), raw_d.end());
        const std::vector<float> result = pad_crop_relative_embeddings(raw, window_size, k_channels, length);
        push_number_array(L, result);
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.pad_crop_relative_embeddings: %s", e.what());
    }
}

// Reads a registered module's own raw weight tensor DIRECTLY (via GgufModel::weight(), no graph
// computation involved) -- needed by VITS's own relative-position tables, which the C++ driver reads
// straight off the GGUF weight table (`model.weight("enc_p.encoder.attn_layers.{i}.emb_rel_k_raw")`)
// rather than through any topology's declared inputs/outputs.
int LoomLuaBridge::l_get_weight(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const char* module_name = luaL_checkstring(L, 1);
        const char* weight_name = luaL_checkstring(L, 2);
        const auto it = self->modules_.find(module_name);
        if (it == self->modules_.end()) {
            return luaL_error(L, "loom.get_weight: unregistered module '%s'", module_name);
        }
        ggml_tensor* t = it->second.model->weight(weight_name);
        const auto n = static_cast<size_t>(ggml_nelements(t));
        std::vector<double> out(n);
        if (t->type == GGML_TYPE_F32) {
            std::vector<float> f(n);
            ggml_backend_tensor_get(t, f.data(), 0, n * sizeof(float));
            out.assign(f.begin(), f.end());
        } else if (t->type == GGML_TYPE_I32) {
            std::vector<int32_t> iv(n);
            ggml_backend_tensor_get(t, iv.data(), 0, n * sizeof(int32_t));
            out.assign(iv.begin(), iv.end());
        } else {
            return luaL_error(L, "loom.get_weight: '%s' has an unsupported ggml type", weight_name);
        }
        push_number_array(L, out);
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.get_weight: %s", e.what());
    }
}

LoomLuaBridge::LoomLuaBridge(ggml_backend_t backend) : L_(luaL_newstate()), backend_(backend) {
    if (L_ == nullptr) throw Error("LoomLuaBridge: luaL_newstate() failed (out of memory)");
    luaL_openlibs(L_);

    lua_newtable(L_); // the "loom" table
    const struct {
        const char* name;
        lua_CFunction fn;
    } bindings[] = {
        {"run_subgraph", &LoomLuaBridge::l_run_subgraph}, {"run_recurrent", &LoomLuaBridge::l_run_recurrent},
        {"range", &LoomLuaBridge::l_range},
        {"run_subgraph_argmax", &LoomLuaBridge::l_run_subgraph_argmax},
        {"causal_mask", &LoomLuaBridge::l_causal_mask},   {"zero_mask", &LoomLuaBridge::l_zero_mask},
        {"argmax_row", &LoomLuaBridge::l_argmax_row},     {"seed_rng", &LoomLuaBridge::l_seed_rng},
        {"gaussian_array", &LoomLuaBridge::l_gaussian_array},
        {"uniform_array", &LoomLuaBridge::l_uniform_array},
        {"expand_by_duration", &LoomLuaBridge::l_expand_by_duration},
        {"pad_crop_relative_embeddings", &LoomLuaBridge::l_pad_crop_relative_embeddings},
        {"get_weight", &LoomLuaBridge::l_get_weight},
    };
    for (const auto& b : bindings) {
        lua_pushlightuserdata(L_, this);
        lua_pushcclosure(L_, b.fn, 1); // `this` becomes lua_upvalueindex(1) inside the trampoline
        lua_setfield(L_, -2, b.name);
    }
    lua_setglobal(L_, "loom");
}

LoomLuaBridge::~LoomLuaBridge() {
    if (L_ != nullptr) lua_close(L_);
}

void LoomLuaBridge::register_module(const std::string& name, GgufModel& model, GraphTopology topo,
                                     KvCache* kv_cache, ConvStateCache* conv_state) {
    modules_[name] = Module{&model, std::move(topo), kv_cache, conv_state, backend_};
}

void LoomLuaBridge::load_script(const std::string& lua_source) {
    if (luaL_loadstring(L_, lua_source.c_str()) != 0) {
        std::string msg = lua_tostring(L_, -1);
        lua_pop(L_, 1);
        throw Error("LoomLuaBridge::load_script: " + msg);
    }
    if (lua_pcall(L_, 0, 0, 0) != 0) {
        std::string msg = lua_tostring(L_, -1);
        lua_pop(L_, 1);
        throw Error("LoomLuaBridge::load_script: error executing script: " + msg);
    }
}

LoomLuaBridge::Value LoomLuaBridge::call(const std::string& fn_name, const std::unordered_map<std::string, Value>& args) {
    lua_getglobal(L_, fn_name.c_str());
    if (!lua_isfunction(L_, -1)) {
        lua_pop(L_, 1);
        throw Error("LoomLuaBridge::call: no such Lua function '" + fn_name + "'");
    }

    lua_newtable(L_);
    for (const auto& [key, value] : args) {
        if (std::holds_alternative<double>(value)) {
            lua_pushnumber(L_, std::get<double>(value));
        } else {
            push_number_array(L_, std::get<std::vector<double>>(value));
        }
        lua_setfield(L_, -2, key.c_str());
    }

    if (lua_pcall(L_, 1, 1, 0) != 0) {
        std::string msg = lua_tostring(L_, -1);
        lua_pop(L_, 1);
        throw Error("LoomLuaBridge::call: error in '" + fn_name + "': " + msg);
    }

    Value result;
    if (lua_istable(L_, -1)) {
        result = read_number_array(L_, -1);
    } else {
        result = luaL_checknumber(L_, -1);
    }
    lua_pop(L_, 1);
    return result;
}

} // namespace loom
