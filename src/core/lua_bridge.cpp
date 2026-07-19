#include "loom/core/lua_bridge.h"

#include "loom/core/duration_aligner.h"
#include "loom/core/graph_builder.h"
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

LoomLuaBridge* bridge_from_upvalue(lua_State* L) {
    return static_cast<LoomLuaBridge*>(lua_touserdata(L, lua_upvalueindex(1)));
}

} // namespace

int LoomLuaBridge::l_run_subgraph(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const char* module_name = luaL_checkstring(L, 1);
        const auto n_tokens = static_cast<uint32_t>(luaL_checknumber(L, 2));
        const auto n_past = static_cast<uint32_t>(luaL_checknumber(L, 3));
        luaL_checktype(L, 4, LUA_TTABLE);

        const auto it = self->modules_.find(module_name);
        if (it == self->modules_.end()) {
            return luaL_error(L, "loom.run_subgraph: unregistered module '%s'", module_name);
        }
        Module& mod = it->second;

        GraphBuilder builder(mod.topo, *mod.model, mod.backend, mod.kv_cache);
        GraphBuilder::BuildResult r = builder.build(n_tokens, n_past);

        // Iterate the `inputs_table` (string key -> flat number array) and set each declared input.
        lua_pushnil(L);
        while (lua_next(L, 4) != 0) {
            // key at -2, value at -1
            if (lua_type(L, -2) != LUA_TSTRING) {
                return luaL_error(L, "loom.run_subgraph: inputs table keys must be strings");
            }
            const std::string name = lua_tostring(L, -2);
            const auto input_it = r.input_tensors.find(name);
            if (input_it == r.input_tensors.end()) {
                return luaL_error(L, "loom.run_subgraph: module '%s' has no declared input '%s'", module_name,
                                   name.c_str());
            }
            set_tensor_from_lua_array(L, -1, input_it->second);
            lua_pop(L, 1); // pop value, keep key for lua_next
        }

        ggml_backend_graph_compute(mod.backend, r.graph);

        const auto n_out = static_cast<size_t>(ggml_nelements(r.output));
        std::vector<double> out(n_out);
        if (r.output->type == GGML_TYPE_F32) {
            std::vector<float> f(n_out);
            ggml_backend_tensor_get(r.output, f.data(), 0, n_out * sizeof(float));
            out.assign(f.begin(), f.end());
        } else if (r.output->type == GGML_TYPE_I32) {
            std::vector<int32_t> iv(n_out);
            ggml_backend_tensor_get(r.output, iv.data(), 0, n_out * sizeof(int32_t));
            out.assign(iv.begin(), iv.end());
        } else {
            return luaL_error(L, "loom.run_subgraph: output tensor has an unsupported ggml type");
        }
        push_number_array(L, out);
        // Second return value: the output's ggml shape [ne0,ne1,ne2,ne3] -- e.g. a decoder's logits
        // output has ne0=n_vocab, needed by the script to call loom.argmax_row correctly without
        // hardcoding the vocab size (the topology stays the single source of truth for shapes, same
        // principle as the input-type dispatch above).
        const std::vector<double> shape = {static_cast<double>(r.output->ne[0]), static_cast<double>(r.output->ne[1]),
                                            static_cast<double>(r.output->ne[2]), static_cast<double>(r.output->ne[3])};
        push_number_array(L, shape);
        return 2;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.run_subgraph: %s", e.what());
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
int LoomLuaBridge::l_causal_mask(lua_State* L) {
    try {
        const auto n_tokens = static_cast<uint32_t>(luaL_checknumber(L, 1));
        const auto n_past = static_cast<uint32_t>(luaL_checknumber(L, 2));
        const uint32_t n_kv = n_past + n_tokens;
        std::vector<double> mask(static_cast<size_t>(n_kv) * n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const uint32_t query_pos = n_past + i;
            for (uint32_t j = 0; j < n_kv; ++j) {
                mask[static_cast<size_t>(i) * n_kv + j] =
                    (j <= query_pos) ? 0.0 : -std::numeric_limits<double>::infinity();
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
        {"run_subgraph", &LoomLuaBridge::l_run_subgraph}, {"range", &LoomLuaBridge::l_range},
        {"causal_mask", &LoomLuaBridge::l_causal_mask},   {"zero_mask", &LoomLuaBridge::l_zero_mask},
        {"argmax_row", &LoomLuaBridge::l_argmax_row},     {"seed_rng", &LoomLuaBridge::l_seed_rng},
        {"gaussian_array", &LoomLuaBridge::l_gaussian_array},
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

void LoomLuaBridge::register_module(const std::string& name, GgufModel& model, GraphTopology topo, KvCache* kv_cache) {
    modules_[name] = Module{&model, std::move(topo), kv_cache, backend_};
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
