#include "loom/core/lua_bridge.h"

#include "loom/core/duration_aligner.h"
#include "loom/core/graph_builder.h"
#include "loom/core/conv_state_cache.h"

#include <functional>
#include "loom/core/kv_cache.h"
#include "loom/core/output_store.h"
#include "loom/core/relative_position.h"
#include "loom/loom_errors.h"

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

#include <ggml-backend.h>

#include <algorithm>
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

// Places a `[n_kv_real, rows]` mask, as a driver's `loom.causal_mask` builds it, into a tensor that was
// materialized at the BUCKET-PADDED n_kv (BACKLOG.md P4.0.15). Row i's real values go at the tensor's
// own ne[0] stride and the tail is `-inf`, which is what makes the padded cache cells contribute
// exactly zero -- they are not zero themselves, so leaving the tail at 0.0 would let a previous
// generation's K/V into this one's softmax.
//
// **This is why no driver script has to learn what a bucket is.** The bucket is the engine's choice
// about its own cache, made after the driver has already built the only mask it could describe: the one
// spanning the cells that actually hold this sequence. Padding it is a placement detail of writing that
// value into a tensor, and this is where the two widths are both known.
//
// Returns false when the array is not the real-width shape (a driver that already built the padded
// width, most obviously), leaving the ordinary exact-size path to handle it.
bool set_mask_tensor_padded(lua_State* L, int value_idx, ggml_tensor* tensor, int64_t n_kv_real) {
    if (tensor->type != GGML_TYPE_F32 || n_kv_real <= 0 || tensor->ne[0] <= n_kv_real) return false;
    const auto rows = static_cast<size_t>(ggml_nelements(tensor) / tensor->ne[0]);
    const auto n_kv_pad = static_cast<size_t>(tensor->ne[0]);
    const std::vector<double> vals = read_number_array(L, value_idx);
    if (vals.size() != static_cast<size_t>(n_kv_real) * rows) return false;

    std::vector<float> padded(static_cast<size_t>(ggml_nelements(tensor)),
                               -std::numeric_limits<float>::infinity());
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < static_cast<size_t>(n_kv_real); ++c) {
            padded[r * n_kv_pad + c] = static_cast<float>(vals[r * static_cast<size_t>(n_kv_real) + c]);
        }
    }
    ggml_backend_tensor_set(tensor, padded.data(), 0, padded.size() * sizeof(float));
    return true;
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

// Resolves a module name to its persistent OutputStore. A callable rather than a direct call because
// `modules_` is private and the run helper below deliberately lives in this anonymous namespace, taking
// the module's pieces individually rather than the Module struct.
using StoreLookup = std::function<OutputStore&(const std::string&)>;

// Copies the first `rows` rows of `src` into `dst`, backend-side and without a host round trip on any
// backend where a full copy would avoid one.
//
// **Why a prefix is a capability and not a caller's slicing problem** (BACKLOG.md P4.3d). A family-3
// encoder consumes audio in indivisible chunks -- one second for Qwen3-ASR, twelve for Granite Speech
// -- so a host zero-pads up to the next boundary and the encoder emits real embedding rows for that
// padding. Those rows are not audio and the language model should not read them. The driver knows how
// many rows are genuine (it is the same arithmetic the host padded with), but the encoder's output
// deliberately never becomes a Lua value, so until now there was no way to say "these rows of it".
//
// A `ggml_view_2d` over the source rather than three hand-written copy branches: the prefix of a
// contiguous tensor is contiguous, and a 2-D view of it has exactly the `ne`/`nb` a contiguous
// destination of that shape has -- which is what `ggml_backend_tensor_copy` asserts. So this reuses
// ggml's own copy strategy (host memcpy, device-to-device, or its own fallback) instead of restating
// it here.
void copy_row_prefix(ggml_tensor* src, ggml_tensor* dst, int64_t rows, const std::string& src_module,
                      const char* fname) {
    if (src->ne[2] != 1 || src->ne[3] != 1 || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        throw Error(std::string(fname) + ": module '" + src_module + "' output is not a contiguous 2-D "
                     "tensor, so a row prefix of it is not a contiguous view -- `rows` is only defined "
                     "for the [width, rows] outputs a driver splices into another module's input");
    }
    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead();
    params.no_alloc = true;
    ggml_context_ptr ctx(ggml_init(params));
    ggml_tensor* view = ggml_view_2d(ctx.get(), src, src->ne[0], rows, src->nb[1], 0);
    // A freshly created view carries no buffer -- ggml-alloc is what normally attaches one, and this
    // view never reaches a graph. `ggml_backend_view_init` is the supported way to point it at its
    // source's buffer, and without it `ggml_backend_tensor_copy` dereferences a null buffer.
    if (ggml_backend_view_init(view) != GGML_STATUS_SUCCESS) {
        throw Error(std::string(fname) + ": could not view the first " + std::to_string(rows) +
                     " row(s) of module '" + src_module + "' output");
    }
    ggml_backend_tensor_copy(view, dst);
}

// Is the Lua value at `value_idx` a retained-output reference -- `{from = "module"}`, optionally with
// `index`, `gen` and `rows` -- rather than a plain data array?
//
// Unambiguous by construction: an input table's ordinary value is a 1-indexed array of numbers, which
// has no `from` field at all, and a table that HAS one is not something any existing driver passes.
// Deliberately a table with named fields rather than a bare string or an `{module, index}` pair: it is
// self-describing where a driver script is read (out of a GGUF, most often), and it cannot collide with
// a two-element data array.
bool is_output_ref(lua_State* L, int value_idx) {
    if (lua_type(L, value_idx) != LUA_TTABLE) return false;
    lua_getfield(L, value_idx, "from");
    const bool is_ref = lua_type(L, -1) == LUA_TSTRING;
    lua_pop(L, 1);
    return is_ref;
}

// Fills `dst` from another module's retained output, backend-side: `ggml_backend_tensor_copy` moves the
// bytes directly, so on a multi-backend build this is a device->device copy and the values never become
// Lua doubles. This is the whole reason retained outputs exist (BACKLOG.md P4.0.12).
//
// Raises loom::Error rather than calling luaL_error: these run inside compute_and_emit, whose caller
// already converts exceptions at the trampoline, and luaL_error's longjmp would skip the GraphBuilder's
// destructor on the way out.
void set_tensor_from_output_ref(lua_State* L, int value_idx, ggml_tensor* dst, const StoreLookup& lookup,
                                 const char* fname) {
    lua_getfield(L, value_idx, "from");
    const std::string src_module = lua_tostring(L, -1);
    lua_pop(L, 1);

    // 1-based, like the declared-output list it indexes (and like `loom.get_output`'s own `index`).
    lua_getfield(L, value_idx, "index");
    const int64_t index1 = lua_isnil(L, -1) ? 1 : static_cast<int64_t>(std::llround(lua_tonumber(L, -1)));
    lua_pop(L, 1);

    // The generation the producing `loom.run_subgraph_and_retain` returned. Optional -- a synthesized
    // driver whose adjacency the exporter already checked need not carry it
    // (driver_ir.check_subgraph_calls) -- but it is what makes a stale read an error rather than
    // silently newer data.
    lua_getfield(L, value_idx, "gen");
    const bool pinned = !lua_isnil(L, -1);
    const auto gen = pinned ? static_cast<uint64_t>(std::llround(lua_tonumber(L, -1))) : 0;
    lua_pop(L, 1);

    OutputStore& store = lookup(src_module);
    if (pinned) store.check_generation(gen, src_module);
    if (index1 < 1) {
        throw Error(std::string(fname) + ": {from='" + src_module + "', index=" + std::to_string(index1) +
                     "} -- an output index is 1-based, like the declared-output list it indexes");
    }
    ggml_tensor* src = store.get(static_cast<size_t>(index1 - 1));

    // How many of the source's rows to copy. Absent means all of them, which is every caller but the
    // one that trims an audio encoder's chunk padding -- see `copy_row_prefix`.
    lua_getfield(L, value_idx, "rows");
    const bool trimmed = !lua_isnil(L, -1);
    const auto rows = trimmed ? static_cast<int64_t>(std::llround(lua_tonumber(L, -1))) : int64_t{0};
    lua_pop(L, 1);

    if (trimmed) {
        if (rows < 1 || rows > src->ne[1]) {
            throw Error(std::string(fname) + ": {from='" + src_module + "', rows=" +
                         std::to_string(rows) + "} -- module '" + src_module + "' retained " +
                         std::to_string(src->ne[1]) + " row(s), so that is not a prefix of it");
        }
        if (src->type != dst->type || src->ne[0] != dst->ne[0] || dst->ne[1] != rows ||
            dst->ne[2] != 1 || dst->ne[3] != 1) {
            throw Error(std::string(fname) + ": module '" + src_module + "' output " +
                         std::to_string(index1) + " is " + ggml_type_name(src->type) + " [" +
                         std::to_string(src->ne[0]) + "," + std::to_string(src->ne[1]) +
                         "], and its first " + std::to_string(rows) + " row(s) are " +
                         ggml_type_name(src->type) + " [" + std::to_string(src->ne[0]) + "," +
                         std::to_string(rows) + "], but input '" + ggml_get_name(dst) + "' is " +
                         ggml_type_name(dst->type) + " [" + std::to_string(dst->ne[0]) + "," +
                         std::to_string(dst->ne[1]) + "," + std::to_string(dst->ne[2]) + "," +
                         std::to_string(dst->ne[3]) + "] -- a row prefix is copied as-is, so the two "
                         "must agree exactly");
        }
        copy_row_prefix(src, dst, rows, src_module, fname);
        return;
    }

    if (!ggml_are_same_shape(src, dst) || src->type != dst->type) {
        throw Error(std::string(fname) + ": module '" + src_module + "' output " + std::to_string(index1) +
                     " is " + ggml_type_name(src->type) + " [" + std::to_string(src->ne[0]) + "," +
                     std::to_string(src->ne[1]) + "," + std::to_string(src->ne[2]) + "," +
                     std::to_string(src->ne[3]) + "], but input '" + ggml_get_name(dst) + "' is " +
                     ggml_type_name(dst->type) + " [" + std::to_string(dst->ne[0]) + "," +
                     std::to_string(dst->ne[1]) + "," + std::to_string(dst->ne[2]) + "," +
                     std::to_string(dst->ne[3]) + "] -- a retained output is copied as-is, so the two "
                     "must agree exactly (marshalling through Lua only ever compared element counts)");
    }
    ggml_backend_tensor_copy(src, dst);
}

// Argmax over ONE row of a 2D f32 tensor. `requested_row` is 0-based; negative means the last row.
// `nb[1]` is the row stride, so this reads ne0 floats and never touches the other rows.
//
// **This is what removes the marshalling ceiling.** `run_subgraph` marshals every output element into a
// Lua table, and LuaJIT's array part tops out near 2^27 entries -- so a 262144-wide vocab (Gemma 3)
// overflows at ~512 prompt tokens, and a driver whose only use for those logits is one argmax pays a
// 157M-element table to compute a single integer. Reducing on the tensor removes both the ceiling and
// the copy: nothing crosses the boundary but the answer, which keeps the Lua boundary a *per-step*
// boundary rather than a per-logit one -- the same reasoning KV-CACHE.md §1.1 gives for not driving
// attention from Lua.
// `lo`/`hi` restrict the reduction to the half-open id window `[lo, hi)`, and `hi < 0` means "to the end
// of the row" -- i.e. the whole row, which is every caller but one. The returned id is ABSOLUTE, not
// relative to `lo`: a restricted argmax exists to answer "which of THESE classes", and a caller that had
// to add `lo` back would be one addition away from a wrong token id.
//
// The window is what a classification over a *subset* of a shared vocabulary needs. Whisper's language
// detection is the case that asked for it (BACKLOG.md P4.1 follow-up): its 98 language tokens occupy one
// contiguous id block inside the same 51865-wide vocabulary the transcript is decoded from, so detecting
// a language is one decode step whose argmax must ignore every ordinary text token -- otherwise the
// answer is whatever word the model would have emitted. Only the window is copied off the backend, so
// asking about 99 ids does not read 51865 floats.
std::vector<float> read_row_window(ggml_tensor* out, int64_t requested_row, const char* fname,
                                   int64_t lo, int64_t& hi) {
    if (out->type != GGML_TYPE_F32) {
        throw Error(std::string(fname) + ": output must be f32");
    }
    const int64_t n_vocab = out->ne[0];
    const int64_t n_rows = out->ne[1];
    const int64_t row = requested_row < 0 ? n_rows - 1 : requested_row;
    if (n_vocab <= 0 || row < 0 || row >= n_rows) {
        throw Error(std::string(fname) + ": row " + std::to_string(requested_row) + " out of range for an "
                     "output with " + std::to_string(n_rows) + " row(s) of width " + std::to_string(n_vocab));
    }
    if (hi < 0) hi = n_vocab;
    if (lo < 0 || lo >= hi || hi > n_vocab) {
        throw Error(std::string(fname) + ": id window [" + std::to_string(lo) + ", " + std::to_string(hi) +
                     ") is not a non-empty sub-range of this output's " + std::to_string(n_vocab) +
                     " class(es)");
    }
    std::vector<float> logits(static_cast<size_t>(hi - lo));
    ggml_backend_tensor_get(out, logits.data(),
                             static_cast<size_t>(row) * out->nb[1] + static_cast<size_t>(lo) * sizeof(float),
                             logits.size() * sizeof(float));
    return logits;
}

// The maximum of an already-read window, as an offset into it. Split out from `argmax_tensor_row` so
// that the sampler's greedy path can reduce the SAME array it would otherwise have drawn from --
// under classifier-free guidance the logits being maximized are not the ones in any tensor, so
// "greedy sampling is argmax" can no longer be preserved by delegating to a tensor reduction.
int64_t argmax_of_window(const std::vector<float>& window) {
    size_t best = 0;
    for (size_t i = 1; i < window.size(); ++i) {
        if (window[i] > window[best]) best = i;
    }
    return static_cast<int64_t>(best);
}

int64_t argmax_tensor_row(ggml_tensor* out, int64_t requested_row, const char* fname,
                           int64_t lo = 0, int64_t hi = -1) {
    const std::vector<float> window = read_row_window(out, requested_row, fname, lo, hi);
    return lo + argmax_of_window(window);
}

// One token DRAWN from a row of a 2D f32 tensor, instead of the maximum of it (P4.24).
//
// **`argmax_tensor_row`'s own reduction when the settings are greedy**, and that is the invariant
// rather than an optimization: `temperature <= 0` and `top_k == 1` both mean "the highest-scoring
// id", so this runs `argmax_of_window` over the same window that function would have read. Two ways
// to get a token out of one forward pass that can disagree is the failure this project keeps
// removing, and P4.0.14 already retired one of them. It shares the reduction rather than calling the
// tensor form because guidance produces logits that are in no tensor -- see below.
//
// **`lo`/`hi` restrict it to the half-open id window `[lo, hi)`, ids returned ABSOLUTE**, which is
// `argmax_tensor_row`'s window with the same meaning and the same reason for returning absolute ids.
// Family 10 is what asked for it: Dia's four highest ids are control tokens and
// `DiaEOSChannelFilterLogitsProcessor` bans them PER CHANNEL rather than globally, so channel 8
// sampling over the whole row can emit PAD or BOS -- which under an argmax it never did, because the
// argmax already had the window. A sampler without one is not a smaller version of the same thing.
//
// **`uncond`/`guidance_scale` are classifier-free guidance**, off when `uncond` is null. See the body.
//
// The order is `transformers`' own processor order -- temperature, then top-k, then top-p, then a
// multinomial draw -- because the reference this is defined against is `generate` under the
// checkpoint's own `generation_config.json`. Any other order gives a different distribution from the
// same three numbers.
//
// It reduces ON the tensor for the same reason its greedy sibling does: `run_subgraph` marshals every
// output element into a Lua table and LuaJIT's array part tops out near 2^27 entries, so a 262144-wide
// vocab overflows at ~512 prompt tokens (Retro-004). Sampling in Lua would reinstate that ceiling at
// exactly the vocabulary size that found it.
int64_t sample_tensor_row(ggml_tensor* out, int64_t requested_row, const char* fname, std::mt19937& rng,
                           std::uniform_real_distribution<float>& uniform, float temperature,
                           int64_t top_k, float top_p, int64_t lo, int64_t hi,
                           ggml_tensor* uncond, float guidance_scale, int64_t guidance_top_k) {
    if (guidance_top_k < 0) {
        throw Error(std::string(fname) + ": guidance top_k is " + std::to_string(guidance_top_k) +
                     "; it is a count of candidates, and 0 means 'do not shortlist'");
    }
    if (top_k < 0) {
        throw Error(std::string(fname) + ": top_k is " + std::to_string(top_k) + "; it is a count of "
                     "candidates, and 0 means 'do not truncate'");
    }
    if (!(top_p > 0.0f) || top_p > 1.0f) {
        throw Error(std::string(fname) + ": top_p is " + std::to_string(top_p) + "; it is a cumulative "
                     "probability in (0, 1], and 1 means 'do not truncate'");
    }

    // **Classifier-free guidance, and it happens HERE rather than in a graph**: over two retained
    // outputs the caller names, in one of the two forms below. It is in this function because it is
    // part of turning logits into a token, and this project has one place that does that on purpose --
    // two ways to get a token out of a forward pass that can disagree is a failure it keeps removing
    // (P4.0.14, and the greedy/argmax invariant below).
    //
    // The alternative a driver has is marshalling both rows into Lua and combining them there, which
    // for Dia is 9252 floats twice per step and reinstates exactly the boundary cost every retained
    // reduction in this tree exists to avoid (Retro-004).
    std::vector<float> logits;
    if (uncond == nullptr) {
        logits = read_row_window(out, requested_row, fname, lo, hi);
    } else {
        // The two rows must be the same head over the same vocabulary, checked on the FULL widths
        // rather than on the window: with `hi` resolved against the conditional output, a wider
        // unconditional one would be silently guided against its first `hi` classes and a narrower one
        // would fail with a window message that blames the caller's `lo`/`hi`. Neither says the true
        // thing, which is that these are two runs of ONE model and something handed over two models.
        if (uncond->ne[0] != out->ne[0]) {
            throw Error(std::string(fname) + ": the guidance output's rows are " +
                         std::to_string(uncond->ne[0]) + " wide and this one's are " +
                         std::to_string(out->ne[0]) + " -- guidance combines two runs of ONE model, "
                         "over one vocabulary");
        }
        // **The guidance runs over the WHOLE row, and the window is applied after it.** That is the
        // order `transformers` composes these in -- the CFG processor is inserted at index 0, ahead of
        // the per-channel filter that bans the control ids -- and with `guidance_top_k` the two orders
        // give different candidate sets: a control token inside the guided top-k occupies one of the k
        // slots and is then banned, leaving k-1 real candidates. Restricting first would silently give
        // back the full k. Without a top-k the two orders agree, and only the window is read.
        const bool whole_row = guidance_top_k > 0;
        int64_t cond_hi = whole_row ? -1 : hi;
        int64_t uncond_hi = cond_hi;
        const int64_t base = whole_row ? 0 : lo;
        logits = read_row_window(out, requested_row, fname, base, cond_hi);
        const std::vector<float> other = read_row_window(uncond, requested_row, fname, base, uncond_hi);

        std::vector<float> guided(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            guided[i] = other[i] + guidance_scale * (logits[i] - other[i]);
        }

        if (!whole_row) {
            logits = std::move(guided);
        } else {
            // **`guidance_top_k` selects with the GUIDED logits and scores with the CONDITIONAL ones**,
            // which is not a variation on the formula above but a different operation -- and it is what
            // `DiaClassifierFreeGuidanceLogitsProcessor` does. Guidance sharpens the ranking, and the
            // model's own (unsharpened) distribution over that shortlist is what gets drawn from; the
            // guided values themselves are discarded. Anything outside the shortlist is -inf, spelled
            // here as "removed from the candidate set", which is the same thing one representation up.
            const int64_t k = std::min<int64_t>(guidance_top_k, static_cast<int64_t>(guided.size()));
            std::vector<int64_t> order(guided.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int64_t>(i);
            std::partial_sort(order.begin(), order.begin() + k, order.end(),
                              [&](int64_t a, int64_t b) {
                                  return guided[static_cast<size_t>(a)] > guided[static_cast<size_t>(b)];
                              });
            std::vector<bool> kept(guided.size(), false);
            for (int64_t i = 0; i < k; ++i) kept[static_cast<size_t>(order[static_cast<size_t>(i)])] = true;
            for (size_t i = 0; i < logits.size(); ++i) {
                if (!kept[i]) logits[i] = -std::numeric_limits<float>::infinity();
            }
            // Now apply the window that was deliberately not applied above -- validated here, since
            // the read above was of the whole row and so validated nothing about `lo`/`hi`.
            const auto width = static_cast<int64_t>(logits.size());
            if (hi < 0) hi = width;
            if (lo < 0 || lo >= hi || hi > width) {
                throw Error(std::string(fname) + ": id window [" + std::to_string(lo) + ", " +
                             std::to_string(hi) + ") is not a non-empty sub-range of this output's " +
                             std::to_string(width) + " class(es)");
            }
            logits = std::vector<float>(logits.begin() + lo, logits.begin() + hi);
        }
        // Every candidate banned. Reachable only when a guidance shortlist and an id window do not
        // intersect at all, which `transformers` answers with a row of -inf and a NaN softmax. Saying
        // so is the difference between a fixable export and a token drawn out of nothing.
        if (std::none_of(logits.begin(), logits.end(),
                         [](float v) { return v > -std::numeric_limits<float>::infinity(); })) {
            throw Error(std::string(fname) + ": the guidance shortlist and the id window [" +
                         std::to_string(lo) + ", " + std::to_string(hi) + ") have no id in common, so "
                         "there is nothing to draw from");
        }
    }

    // **The greedy invariant, preserved through guidance.** `temperature <= 0` and `top_k == 1` both
    // mean "the highest-scoring id", and that must be the maximum of the GUIDED logits -- which are
    // not in any tensor, so this can no longer delegate to `argmax_tensor_row` and instead shares its
    // reduction. Unguided, the two paths read the same window and run the same loop, which is the
    // invariant stated more strongly than before rather than less.
    if (temperature <= 0.0f || top_k == 1) {
        return lo + argmax_of_window(logits);
    }

    const auto n_vocab = static_cast<int64_t>(logits.size());

    // Top-k first, on the LOGITS, because ordering by logit and ordering by probability are the same
    // ordering -- softmax is monotone -- so the k candidates can be chosen before anything is
    // exponentiated. That also bounds every step after this one by k rather than by the vocabulary,
    // which for Gemma 3 is 64 against 262144.
    //
    // `candidates` holds offsets INTO THE WINDOW; `lo` is added back at every return, because a
    // restricted sampler exists to answer "which of THESE ids" and a caller that had to add it back
    // would be one addition away from a wrong token -- the same argument `argmax_tensor_row` makes.
    std::vector<int64_t> candidates(static_cast<size_t>(n_vocab));
    for (int64_t i = 0; i < n_vocab; ++i) candidates[static_cast<size_t>(i)] = i;
    const auto by_logit_desc = [&](int64_t a, int64_t b) {
        return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
    };
    if (top_k > 0 && top_k < n_vocab) {
        std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(), by_logit_desc);
        candidates.resize(static_cast<size_t>(top_k));
    } else {
        std::sort(candidates.begin(), candidates.end(), by_logit_desc);
    }

    // Softmax over the survivors, shifted by the maximum -- which is `candidates[0]`, since they are
    // sorted. Without the shift a logit around 30 (ordinary for a 262144-wide head at temperature 0.1)
    // overflows expf.
    const float max_logit = logits[static_cast<size_t>(candidates[0])] / temperature;
    std::vector<float> probs(candidates.size());
    float sum = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp(logits[static_cast<size_t>(candidates[i])] / temperature - max_logit);
        sum += probs[i];
    }

    // Top-p: the shortest prefix of the descending order whose probability mass reaches `top_p`. At
    // least one candidate always survives -- a threshold below the single most likely token's own
    // probability would otherwise select nothing to draw from.
    size_t keep = candidates.size();
    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i] / sum;
            if (cumulative >= top_p) {
                keep = i + 1;
                break;
            }
        }
    }

    // One draw from the SHARED stream, through the same distribution object `loom.uniform_array` uses
    // -- so a script that seeds with `loom.seed_rng` gets a reproducible token, and so the draw-ORDER
    // caveat that stream already documents covers this too.
    float mass = 0.0f;
    for (size_t i = 0; i < keep; ++i) mass += probs[i];
    const float target = uniform(rng) * mass;
    float running = 0.0f;
    for (size_t i = 0; i < keep; ++i) {
        running += probs[i];
        if (running >= target) return lo + candidates[i];
    }
    // Only reachable when floating-point accumulation lands the target past the total; the last
    // candidate is the answer either way.
    return lo + candidates[keep - 1];
}

// The same reduction over EVERY row: one id per row, in row order. `argmax_tensor_row`'s plural.
//
// Exists because a frame-wise classifier reduces the whole tensor rather than one row of it, and the
// per-row call cannot express that without first asking how many rows there are -- a question Lua can
// only answer today by marshalling the tensor it is trying not to marshal (BACKLOG.md P4.0.17). One
// call returns n_rows numbers and the logits never cross the boundary, which is the same trade
// `argmax_row` makes, applied to the shape CTC actually has.
//
// What it deliberately does NOT do is collapse blanks or duplicates. That is the model family's own
// convention, it belongs in the driver where the rest of the family's orchestration lives, and putting
// it here would be `ctc_decode.cpp` behind a binding rather than retired.
std::vector<double> argmax_tensor_rows(ggml_tensor* out, const char* fname) {
    if (out->type != GGML_TYPE_F32) {
        throw Error(std::string(fname) + ": output must be f32");
    }
    const int64_t n_classes = out->ne[0];
    const int64_t n_rows = out->ne[1];
    if (n_classes <= 0 || n_rows <= 0) {
        throw Error(std::string(fname) + ": output is " + std::to_string(n_classes) + "x" +
                     std::to_string(n_rows) + ", which has no rows to reduce");
    }
    std::vector<float> row(static_cast<size_t>(n_classes));
    std::vector<double> ids(static_cast<size_t>(n_rows));
    for (int64_t r = 0; r < n_rows; ++r) {
        ggml_backend_tensor_get(out, row.data(), static_cast<size_t>(r) * out->nb[1],
                                 row.size() * sizeof(float));
        int64_t best = 0;
        for (int64_t i = 1; i < n_classes; ++i) {
            if (row[static_cast<size_t>(i)] > row[static_cast<size_t>(best)]) best = i;
        }
        ids[static_cast<size_t>(r)] = static_cast<double>(best);
    }
    return ids;
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

// Everything `run_subgraph` and `run_subgraph_and_retain` share: build the graph for `axes`, fill every
// declared input from the Lua table at `inputs_idx`, compute, and hand the result to `emit`.
//
// Extracted rather than duplicated because the entry points differ ONLY in what they do with the
// outputs -- a copy would be free to drift on cache wiring or input validation, which is exactly the
// class of difference nothing would catch. That is also why an input may be a retained-output
// reference on EVERY one of them and not just the retaining one: which module produced a value and
// which entry point consumes it are independent questions.
//
// `builder` is the MODULE's own, persistent for the bridge's lifetime (BACKLOG.md P4.0.13) rather than
// constructed and destroyed per call as it used to be -- so a driver that calls the same module at the
// same axes twice in a row pays one rebuild, not two, and one compute-buffer allocation, not two. The
// same call chain otherwise: the builder retains the graph, so `r` stays valid for exactly as long as
// this function needs it, and P4.0.12's OutputStore is what already removed the reason a retained VALUE
// had to outlive it.
//
// `out_store` is null for the marshalling entry points and the module's own store for
// `run_subgraph_and_retain` -- see GraphBuilder::build for what it does with it.
int compute_and_emit(lua_State* L, const char* fname, const char* module_name, GraphBuilder& builder,
                      const DynamicAxes& axes, int inputs_idx,
                      const StoreLookup& lookup, OutputStore* out_store,
                      const std::function<int(const GraphBuilder::BuildResult&)>& emit) {
    const GraphBuilder::BuildResult& r = builder.build(axes, out_store);

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
        // lua_next leaves the value at the top of the stack; take its absolute index so the field reads
        // an output reference performs (which push and pop) can't shift it underneath us.
        const int value_idx = lua_gettop(L);
        if (is_output_ref(L, value_idx)) {
            set_tensor_from_output_ref(L, value_idx, input_it->second, lookup, fname);
        } else {
            // A mask spanning the KV cache is the one input whose declared width is the engine's
            // business rather than the driver's, because the engine rounds n_kv up to a bucket so a
            // decode loop can reuse its graph (BACKLOG.md P4.0.15). Everything else goes through the
            // ordinary exact-size path, and so does this one when there is no padding to do.
            const bool kv_padded = std::find(r.kv_padded_inputs.begin(), r.kv_padded_inputs.end(), name)
                                    != r.kv_padded_inputs.end();
            if (!kv_padded || !set_mask_tensor_padded(L, value_idx, input_it->second, r.n_kv_real)) {
                set_tensor_from_lua_array(L, value_idx, input_it->second);
            }
        }
        lua_pop(L, 1);
    }

    // Through the BUILDER rather than through a backend handle: on a device build this graph is
    // scheduler-allocated and split across two backends, and only the builder holds the scheduler that
    // knows how to run it (BACKLOG.md P4.7 / graph_builder.h).
    builder.compute();
    // After the compute, so a generation number only ever names a run whose values are really there.
    if (out_store != nullptr) out_store->bump_generation();
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

        return compute_and_emit(L, "loom.run_subgraph", module_name, self->module_builder(mod),
                                 axes, 3, store_lookup(self),
                                 /*out_store=*/nullptr,
                                 [L](const GraphBuilder::BuildResult& r) {
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

// `loom.run_subgraph_and_retain(module, axes, inputs)` -- run the module and leave every declared
// output in the module's own persistent OutputStore instead of marshalling it. Returns one number:
// the store's generation counter for this run, which a later read may pin itself to.
//
// See lua_bridge.h's declaration for the three ways a retained output is read back, and
// include/loom/core/output_store.h for why the buffer is module-owned rather than handed out as a
// handle.
int LoomLuaBridge::l_run_subgraph_and_retain(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const char* module_name = luaL_checkstring(L, 1);
        const DynamicAxes axes = read_axes_table(L, 2);
        luaL_checktype(L, 3, LUA_TTABLE);

        const auto it = self->modules_.find(module_name);
        if (it == self->modules_.end()) {
            return luaL_error(L, "loom.run_subgraph_and_retain: unregistered module '%s'", module_name);
        }
        Module& mod = it->second;
        if (!mod.outputs) mod.outputs = std::make_unique<OutputStore>(mod.backends.primary);
        OutputStore* store = mod.outputs.get();

        return compute_and_emit(L, "loom.run_subgraph_and_retain", module_name, self->module_builder(mod),
                                 axes, 3, store_lookup(self), store,
                                 [L, store](const GraphBuilder::BuildResult&) {
            lua_pushnumber(L, static_cast<lua_Number>(store->generation()));
            return 1;
        });
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.run_subgraph_and_retain: %s", e.what());
    }
}

// `loom.get_output(module [, index [, generation]])` -> (data, shape). The marshalling half of
// retrieval-by-name, for a value that is genuinely host-side: same (flat data, 4-element ne[]) pair
// `loom.run_subgraph` returns for one output, read out of the module's store rather than out of a
// BuildResult that no longer exists.
int LoomLuaBridge::l_get_output(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const std::string module_name = luaL_checkstring(L, 1);
        // 1-based, like the declared-output list it indexes.
        const auto index1 = static_cast<int64_t>(std::llround(luaL_optnumber(L, 2, 1)));
        if (index1 < 1) {
            return luaL_error(L, "loom.get_output: index %d is 1-based, like the declared-output list "
                                  "it indexes", static_cast<int>(index1));
        }
        OutputStore& store = retained_store(self, module_name);
        if (!lua_isnoneornil(L, 3)) {
            store.check_generation(static_cast<uint64_t>(std::llround(luaL_checknumber(L, 3))), module_name);
        }
        ggml_tensor* out = store.get(static_cast<size_t>(index1 - 1));
        push_number_array(L, read_tensor_as_doubles(L, out));
        const std::vector<double> shape = {static_cast<double>(out->ne[0]), static_cast<double>(out->ne[1]),
                                            static_cast<double>(out->ne[2]), static_cast<double>(out->ne[3])};
        push_number_array(L, shape);
        return 2;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.get_output: %s", e.what());
    }
}

int LoomLuaBridge::l_run_recurrent(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const char* module_name = luaL_checkstring(L, 1);
        const std::vector<double> sequence = read_number_array(L, 2);
        const auto seq_len = static_cast<uint32_t>(luaL_checknumber(L, 3));
        const auto input_dim = static_cast<uint32_t>(luaL_checknumber(L, 4));
        const auto hidden_dim = static_cast<uint32_t>(luaL_checknumber(L, 5));
        const bool reverse = lua_toboolean(L, 6) != 0;

        if (sequence.size() != static_cast<size_t>(seq_len) * input_dim) {
            return luaL_error(L, "loom.run_recurrent: sequence has %d elements, expected seq_len*input_dim=%d",
                               static_cast<int>(sequence.size()), static_cast<int>(seq_len * input_dim));
        }

        const auto it = self->modules_.find(module_name);
        if (it == self->modules_.end()) {
            return luaL_error(L, "loom.run_recurrent: unregistered module '%s'", module_name);
        }
        Module& mod = it->second;

        // The cell module's own persistent builder (BACKLOG.md P4.0.13), the same one every other
        // binding uses. build() is still called once per timestep below -- a step's h/c depend on the
        // PREVIOUS step's real output values, so each timestep genuinely needs its own compute -- but
        // the axes never move across a sequence, so every call after the first is served from the
        // retained graph. This is the loop that gains the most from that: one rebuild per direction
        // instead of one per timestep.
        GraphBuilder& builder = self->module_builder(mod);

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

            // ONE build and ONE compute per timestep: the cell topology declares both `h_new` and
            // `c_new`, so the gate stack is evaluated once and both halves of the step are read off
            // the same result. It was two of each until the topology gained its second declared
            // output -- the same node list computed twice per timestep, per direction, per BiLSTM
            // (see recurrent.py::_lstm_cell_topology).
            const GraphBuilder::BuildResult& r = builder.build({{"n_tokens", 0}, {"n_past", 0}});
            ggml_backend_tensor_set(r.input_tensors.at("layer_input"), layer_input.data(), 0,
                                     layer_input.size() * sizeof(float));
            ggml_backend_tensor_set(r.input_tensors.at("h_prev"), h.data(), 0, h.size() * sizeof(float));
            ggml_backend_tensor_set(r.input_tensors.at("c_prev"), c.data(), 0, c.size() * sizeof(float));
            builder.compute();

            if (r.outputs.size() < 2) {
                return luaL_error(L, "loom.run_recurrent: module '%s' declares %d output(s); a cell "
                                      "topology must declare both 'h_new' and 'c_new', in that order",
                                   module_name, static_cast<int>(r.outputs.size()));
            }
            std::vector<float> h_new(hidden_dim);
            std::vector<float> c_new(hidden_dim);
            ggml_backend_tensor_get(r.outputs[0], h_new.data(), 0, h_new.size() * sizeof(float));
            ggml_backend_tensor_get(r.outputs[1], c_new.data(), 0, c_new.size() * sizeof(float));

            h = h_new;
            c = c_new;
            for (uint32_t k = 0; k < hidden_dim; ++k) {
                out[static_cast<size_t>(t) * hidden_dim + k] = static_cast<double>(h[k]);
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

// Two spellings of one operation, distinguished by the first argument's type:
//
//   loom.argmax_row(flat, n_vocab, row)          -- a Lua array the driver already holds
//   loom.argmax_row(module, row [, generation])  -- module `module`'s retained first output
//
// Matches WhisperDriver::argmax, but takes the row index explicitly so the driver script can select
// "last prompt token" during prefill vs "the only row" during incremental decode (mirrors
// transcribe()'s own two call sites) -- kept as one host call rather than a Lua-side loop over
// potentially vocab-sized arrays.
//
// The module form is the retrieval-by-name half of BACKLOG.md P4.0.12, and is the same operation on the
// same values -- it just never marshals them, so it has no ceiling and no copy. An OVERLOAD rather than
// a second binding precisely because it is not a second operation: `n_vocab` is only a parameter of the
// array form because a flat Lua array has lost the shape the tensor still carries.
//
// Since P4.0.14 it is also the ONLY reducing spelling: every synthesized causal-LM driver retains and
// then calls this, and the fused `loom.run_subgraph_argmax` that used to do both at once is gone. Two
// ways to get a token out of a forward pass that can disagree is the failure this project keeps
// removing -- and the fused one composed with nothing, while retention composes with `get_output` and
// with `{from = ...}` on the same run.
int LoomLuaBridge::l_argmax_row(lua_State* L) {
    try {
        if (lua_type(L, 1) == LUA_TSTRING) {
            auto* self = bridge_from_upvalue(L);
            const std::string module_name = lua_tostring(L, 1);
            const auto requested_row = static_cast<int64_t>(luaL_checknumber(L, 2));
            OutputStore& store = retained_store(self, module_name);
            if (!lua_isnoneornil(L, 3)) {
                store.check_generation(static_cast<uint64_t>(std::llround(luaL_checknumber(L, 3))),
                                        module_name);
            }
            lua_pushnumber(L, static_cast<lua_Number>(
                argmax_tensor_row(store.get(0), requested_row, "loom.argmax_row")));
            return 1;
        }
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

// `loom.argmax_row_range(module, row, lo, hi [, generation])` -> the highest-scoring id in the half-open
// window `[lo, hi)` of `module`'s retained first output, as an ABSOLUTE id.
//
// The same reduction as `argmax_row` -- literally the same function, with a window -- so the two cannot
// disagree about how a maximum is found. A separate BINDING rather than an overload, which is the
// opposite of the choice `argmax_row` itself documents, for a mechanical reason: its module form already
// takes an optional trailing `generation`, so `(module, row, lo, hi)` and `(module, row, generation)`
// cannot be told apart by arity or type. A name is clearer than a rule about which trailing numbers mean
// what.
//
// Module-form only. The array form of `argmax_row` exists because a flat Lua array has lost its shape,
// and a driver already holding the row in Lua can slice it itself -- while the whole point here is not to
// marshal it.
int LoomLuaBridge::l_argmax_row_range(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const std::string module_name = luaL_checkstring(L, 1);
        const auto requested_row = static_cast<int64_t>(luaL_checknumber(L, 2));
        const auto lo = static_cast<int64_t>(luaL_checknumber(L, 3));
        const auto hi = static_cast<int64_t>(luaL_checknumber(L, 4));
        OutputStore& store = retained_store(self, module_name);
        if (!lua_isnoneornil(L, 5)) {
            store.check_generation(static_cast<uint64_t>(std::llround(luaL_checknumber(L, 5))),
                                    module_name);
        }
        lua_pushnumber(L, static_cast<lua_Number>(argmax_tensor_row(
            store.get(0), requested_row, "loom.argmax_row_range", lo, hi)));
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.argmax_row_range: %s", e.what());
    }
}

// `loom.sample_row(module, row, options)` -> one token id drawn from `module`'s retained first output.
//
// `options` is `{temperature =, top_k =, top_p =, generation =}`, every entry optional. The defaults
// are the greedy ones (`temperature = 0`, no truncation), so `loom.sample_row(m, r, {})` IS
// `loom.argmax_row(m, r)` -- see `sample_tensor_row`, which returns that function's own answer rather
// than reproducing it.
//
// **A table rather than positional arguments** because the knobs are a set that grows, and it has now
// grown twice: min-p and the repetition penalties are still not implemented (nothing in the fixture
// set asks for them), while `lo`/`hi` and `guidance` were added for family 10. Adding one must not
// renumber what a shipped GGUF's driver already passes, which is the whole reason for the table.
//
// **`lo`/`hi` here rather than a `sample_row_range` binding**, which is where this deliberately
// departs from `argmax_row`/`argmax_row_range`. That pair is two bindings for a stated MECHANICAL
// reason: `argmax_row`'s module form already ends in an optional `generation`, so `(module, row, lo,
// hi)` and `(module, row, generation)` cannot be told apart by arity or type. This one has a table,
// so it has no such ambiguity, and a second binding would be a second door onto one reduction --
// which is what `sample_tensor_row` exists to avoid one level down.
//
// **`guidance = {module =, scale =, top_k =}` is classifier-free guidance**: `uncond + scale * (cond
// - uncond)`, where `cond` is this call's own module and `uncond` is a SECOND module the driver ran
// over the unconditional input. Two modules rather than two calls because the combination happens on
// the logits, and logits are what never cross this boundary. `scale` defaults to 1.0, which is the
// identity -- guidance that changes nothing, so a mis-specified table degenerates to plain sampling
// rather than to noise.
//
// `guidance.top_k` selects a shortlist with the GUIDED logits and then draws from the CONDITIONAL
// ones restricted to it, which is a different operation rather than a variation -- see
// `sample_tensor_row`. It is what `DiaClassifierFreeGuidanceLogitsProcessor` means by
// `guidance_top_k`, and it is separate from the plain `top_k` above, which truncates whatever
// distribution is being drawn from.
//
// **A model whose own formula is centred on the conditional logits passes `scale + 1`.** Dia's is:
// `cond + g * (cond - uncond)` is `uncond + (g + 1) * (cond - uncond)`. The general form is the one
// here, because it is `ClassifierFreeGuidanceLogitsProcessor`'s and because the centring is the
// model's convention -- so the conversion belongs in that model's driver, next to its own constant.
//
// Module form only, like `argmax_row_range` and for the same reason: the array form of `argmax_row`
// exists for a driver that already holds the row in Lua, and a causal LM's row is exactly what must
// never get there.
int LoomLuaBridge::l_sample_row(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const std::string module_name = luaL_checkstring(L, 1);
        const auto requested_row = static_cast<int64_t>(luaL_checknumber(L, 2));
        float temperature = 0.0f;
        int64_t top_k = 0;
        float top_p = 1.0f;
        int64_t lo = 0;
        int64_t hi = -1;
        bool check_generation = false;
        uint64_t generation = 0;
        std::string uncond_module;
        float guidance_scale = 1.0f;
        int64_t guidance_top_k = 0;
        bool check_uncond_generation = false;
        uint64_t uncond_generation = 0;
        if (!lua_isnoneornil(L, 3)) {
            luaL_checktype(L, 3, LUA_TTABLE);
            const auto number_field = [&](int table_idx, const char* name, double fallback) {
                lua_getfield(L, table_idx, name);
                const double value = lua_isnil(L, -1) ? fallback : luaL_checknumber(L, -1);
                lua_pop(L, 1);
                return value;
            };
            temperature = static_cast<float>(number_field(3, "temperature", 0.0));
            top_k = static_cast<int64_t>(number_field(3, "top_k", 0.0));
            top_p = static_cast<float>(number_field(3, "top_p", 1.0));
            // `hi` defaults to -1, which `read_row_window` reads as "to the end of the row" -- so a
            // caller who names neither gets the whole row, which is what every pre-family-10 driver
            // passes and must keep meaning.
            lo = static_cast<int64_t>(number_field(3, "lo", 0.0));
            hi = static_cast<int64_t>(number_field(3, "hi", -1.0));
            lua_getfield(L, 3, "generation");
            check_generation = !lua_isnil(L, -1);
            if (check_generation) generation = static_cast<uint64_t>(std::llround(luaL_checknumber(L, -1)));
            lua_pop(L, 1);

            lua_getfield(L, 3, "guidance");
            if (!lua_isnil(L, -1)) {
                luaL_checktype(L, -1, LUA_TTABLE);
                const int guidance_idx = lua_gettop(L);
                lua_getfield(L, guidance_idx, "module");
                if (!lua_isstring(L, -1)) {
                    lua_pop(L, 2);
                    return luaL_error(L, "loom.sample_row: guidance needs a `module` naming the "
                                          "unconditional run's retained output");
                }
                uncond_module = lua_tostring(L, -1);
                lua_pop(L, 1);
                guidance_scale = static_cast<float>(number_field(guidance_idx, "scale", 1.0));
                guidance_top_k = static_cast<int64_t>(number_field(guidance_idx, "top_k", 0.0));
                lua_getfield(L, guidance_idx, "generation");
                check_uncond_generation = !lua_isnil(L, -1);
                if (check_uncond_generation) {
                    uncond_generation = static_cast<uint64_t>(std::llround(luaL_checknumber(L, -1)));
                }
                lua_pop(L, 1);
            }
            lua_pop(L, 1);
        }
        OutputStore& store = retained_store(self, module_name);
        if (check_generation) store.check_generation(generation, module_name);
        ggml_tensor* uncond = nullptr;
        if (!uncond_module.empty()) {
            if (uncond_module == module_name) {
                return luaL_error(L, "loom.sample_row: guidance names module '%s', which is the one "
                                      "being sampled -- a module has ONE retained output, so the "
                                      "conditional and unconditional runs cannot share it",
                                  uncond_module.c_str());
            }
            OutputStore& uncond_store = retained_store(self, uncond_module);
            if (check_uncond_generation) uncond_store.check_generation(uncond_generation, uncond_module);
            uncond = uncond_store.get(0);
        }
        lua_pushnumber(L, static_cast<lua_Number>(sample_tensor_row(
            store.get(0), requested_row, "loom.sample_row", self->rng_, self->uniform_dist_,
            temperature, top_k, top_p, lo, hi, uncond, guidance_scale, guidance_top_k)));
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.sample_row: %s", e.what());
    }
}

// `loom.argmax_rows(module [, generation])` -> a flat array of one class id per ROW of `module`'s
// retained first output, in row order. `loom.argmax_row`'s plural, and module-form only: the array form
// of the singular exists because a flat Lua array has lost its shape, and a caller who already holds
// every row in Lua can loop over them itself.
//
// This is what lets a frame-wise classifier keep its logits engine-side (BACKLOG.md P4.0.17). A CTC
// encoder's whole output is the reduction's input -- there is no single interesting row -- so the
// singular cannot express it without the driver first learning n_frames, which it can only do by
// marshalling the very tensor it is avoiding.
int LoomLuaBridge::l_argmax_rows(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        const std::string module_name = luaL_checkstring(L, 1);
        OutputStore& store = retained_store(self, module_name);
        if (!lua_isnoneornil(L, 2)) {
            store.check_generation(static_cast<uint64_t>(std::llround(luaL_checknumber(L, 2))),
                                    module_name);
        }
        push_number_array(L, argmax_tensor_rows(store.get(0), "loom.argmax_rows"));
        return 1;
    } catch (const std::exception& e) {
        return luaL_error(L, "loom.argmax_rows: %s", e.what());
    }
}

void LoomLuaBridge::seed_rng(uint32_t seed) {
    rng_.seed(seed);
    normal_dist_.reset();
    uniform_dist_.reset();
}

// Resets the bridge's own std::mt19937 -- the SAME engine every hand-written driver's own RNG uses
// (VitsDriver/SupertonicDriver/MatchaDriver all construct `std::mt19937 rng(seed)` directly), so a
// script that calls loom.seed_rng(seed) then loom.gaussian_array(n) in the SAME order a C++ driver
// draws its own noise produces bit-identical values -- what keeps an exact-match test against that
// driver possible.
int LoomLuaBridge::l_seed_rng(lua_State* L) {
    try {
        auto* self = bridge_from_upvalue(L);
        self->seed_rng(static_cast<uint32_t>(luaL_checknumber(L, 1)));
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

OutputStore& LoomLuaBridge::retained_store(LoomLuaBridge* self, const std::string& module) {
    const auto it = self->modules_.find(module);
    if (it == self->modules_.end()) {
        throw Error("unregistered module '" + module + "'");
    }
    if (!it->second.outputs || !it->second.outputs->filled()) {
        throw Error("module '" + module + "' has no retained outputs -- nothing has run it via "
                     "loom.run_subgraph_and_retain, so there is nothing to read by name");
    }
    return *it->second.outputs;
}

std::function<OutputStore&(const std::string&)> LoomLuaBridge::store_lookup(LoomLuaBridge* self) {
    return [self](const std::string& module) -> OutputStore& { return retained_store(self, module); };
}

LoomLuaBridge::LoomLuaBridge(Backends backends) : L_(luaL_newstate()), backends_(backends) {
    if (L_ == nullptr) throw Error("LoomLuaBridge: luaL_newstate() failed (out of memory)");
    luaL_openlibs(L_);

    lua_newtable(L_); // the "loom" table
    const struct {
        const char* name;
        lua_CFunction fn;
    } bindings[] = {
        {"run_subgraph", &LoomLuaBridge::l_run_subgraph}, {"run_recurrent", &LoomLuaBridge::l_run_recurrent},
        {"range", &LoomLuaBridge::l_range},
        {"run_subgraph_and_retain", &LoomLuaBridge::l_run_subgraph_and_retain},
        {"get_output", &LoomLuaBridge::l_get_output},
        {"causal_mask", &LoomLuaBridge::l_causal_mask},   {"zero_mask", &LoomLuaBridge::l_zero_mask},
        {"argmax_row", &LoomLuaBridge::l_argmax_row},
        {"argmax_row_range", &LoomLuaBridge::l_argmax_row_range},
        {"argmax_rows", &LoomLuaBridge::l_argmax_rows},
        {"sample_row", &LoomLuaBridge::l_sample_row},     {"seed_rng", &LoomLuaBridge::l_seed_rng},
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
    modules_[name] = Module{&model, std::move(topo), kv_cache, conv_state, backends_, nullptr, nullptr};
}

std::vector<LoomLuaBridge::ModuleDeviceReport> LoomLuaBridge::device_report() const {
    std::vector<ModuleDeviceReport> out;
    if (!backends_.hybrid()) return out;
    const std::string device_name = ggml_backend_name(backends_.primary);
    for (const auto& [name, mod] : modules_) {
        if (!mod.builder) continue;
        ModuleDeviceReport report;
        report.module = name;
        report.splits = mod.builder->splits();
        for (const std::string& node_backend : mod.builder->node_backends()) {
            (node_backend == device_name ? report.device_nodes : report.fallback_nodes)++;
        }
        if (report.device_nodes + report.fallback_nodes > 0) out.push_back(std::move(report));
    }
    std::sort(out.begin(), out.end(),
               [](const ModuleDeviceReport& a, const ModuleDeviceReport& b) { return a.module < b.module; });
    return out;
}

GraphBuilder& LoomLuaBridge::module_builder(Module& mod) {
    // `mod.topo` is held BY REFERENCE by the builder, so this is only safe because a Module lives in a
    // node-based unordered_map and is never moved once registered. Re-registering a name replaces the
    // Module wholesale, which destroys the old builder along with the topology it pointed at -- the two
    // cannot outlive each other by construction.
    if (!mod.builder) {
        mod.builder = std::make_unique<GraphBuilder>(mod.topo, *mod.model, mod.backends, mod.kv_cache,
                                                      /*compute_meta_bytes=*/32 * 1024 * 1024,
                                                      mod.conv_state);
    }
    return *mod.builder;
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

bool LoomLuaBridge::has_function(const std::string& fn_name) const {
    lua_getglobal(L_, fn_name.c_str());
    const bool ok = lua_isfunction(L_, -1);
    lua_pop(L_, 1);
    return ok;
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
