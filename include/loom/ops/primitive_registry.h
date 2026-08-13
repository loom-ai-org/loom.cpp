#pragma once

#include "loom/core/backend.h"
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

    // The [n_tokens] I64 cell-index tensor a cached ATTENTION writes its K/V through (BACKLOG.md
    // P4.0.15). Non-null exactly when `kv_cache` is, and synthesized by GraphBuilder rather than
    // declared by the topology for the same reason `n_kv` is derived rather than passed: its value is a
    // function of the (n_tokens, n_past) axes the caller already binds, so a driver that had to supply
    // it could only ever restate what the engine already knows -- and every already-exported cached
    // model would have to be re-exported to say it. It lives in the builder's persistent input buffer,
    // outside the gallocr pool, so a REUSED graph's cells can be rewritten between steps.
    ggml_tensor* kv_cells = nullptr;

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

    // The backend(s) this graph will run on, so a primitive can ask what they can actually execute --
    // see `backend_can_run` below. Defaulted, so a hand-built PrimitiveContext (several tests have one)
    // keeps compiling and simply gets the "assume it is supported" answer, which is what a CPU-only
    // arrangement would have answered anyway.
    Backends backends;
};

// ---------------------------------------------------------------------------------------------------
// PRIMITIVES THAT CHOOSE THEIR OWN LOWERING (BACKLOG.md P4.7e)
// ---------------------------------------------------------------------------------------------------
// Some ops that ggml defines are not implemented by every backend, and the gaps do not line up: CUDA
// has `PAD_REFLECT_1D` but no `POOL_1D`; Vulkan has `POOL_2D` but neither; OpenCL, OpenVINO and Hexagon
// have none of the three. A primitive whose op falls in one of those holes becomes a node the scheduler
// must hand back to the CPU, splitting the graph around it.
//
// **This is the engine's problem and not the exporter's**, and the reason is that the exporter cannot
// answer the question. It emits ONE GGUF that any backend may later run, so composing around a gap
// there compiles every artifact for the least capable backend anyone might use. The engine sees the
// actual backend, and ggml will answer directly.
//
// So a primitive in that position builds the native op, ASKS, and keeps it or emits an equivalent
// composition instead. The topology still says what the model does; only the lowering moves.
//
//     ggml_tensor* native = ggml_pad_reflect_1d(pc.ctx, a, lp0, rp0);
//     if (backend_can_run(pc, native)) return {native};
//     return {compose_it_from_views_and_concats(...)};
//
// Two rules for anything added this way, both learned the hard way (P4.7d):
//
//   * **The fallback must be EXACTLY equivalent, and shown to be.** Not "close enough" -- a composition
//     that differs at the edges is a wrong answer no shape check catches. `POOL_1D`'s own lowering found
//     a case where two spellings of the same ggml op divide by different numbers.
//   * **A composition has a size past which it stops being worth it.** A fallback that emits hundreds of
//     nodes to avoid one CPU node is a worse trade than the split it prevents; prefer the native op and
//     let it fall back.

// Whether the backend this graph will run on can execute `node` as built. False only when a real device
// backend lacks the op -- a CPU backend implements everything, and a PrimitiveContext with no backend
// (a hand-built one) answers true, which is the pre-P4.7e behaviour.
bool backend_can_run(const PrimitiveContext& pc, const ggml_tensor* node);

// PAD_1D_REFLECT's fallback lowering, built from views and concatenations -- what `op_pad_1d_reflect`
// emits when `backend_can_run` says the backend has no `ggml_pad_reflect_1d`. Declared here rather than
// kept private to the primitive so tests/ci/test_pad_reflect_lowering.cpp can hold it against ggml's own
// op directly: the branch that selects it cannot be reached on a CPU backend, which implements
// everything, so the composition has to be reachable on its own to be testable at all.
ggml_tensor* compose_pad_reflect_1d(ggml_context* ctx, ggml_tensor* a, int lp0, int rp0);

// ATAN's fallback lowering: range reduction, a minimax polynomial and a branchless reconstruction, all
// in ops every backend has. Unlike the two above this is an APPROXIMATION (~1.84 ULP, measured) and not
// an exact composition -- ggml has no inverse trigonometry in any backend, so there is nothing exact to
// compose from. `op_atan` reaches for it only where the backend cannot dispatch the `ggml_map_custom`
// callback at all; a CPU keeps libm. See compose_atan in primitives_basic.cpp for the method and for why
// the polynomial is degree 8.
ggml_tensor* compose_atan(ggml_context* ctx, ggml_tensor* x);

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

// ---------------------------------------------------------------------------------------------------
// READING A TENSOR'S VALUES WHILE THE GRAPH IS STILL BEING BUILT
// ---------------------------------------------------------------------------------------------------
// A few primitives need a tensor's CONTENTS at build time, not at compute time -- RANGE_1D's bounds and
// FILL's shape are values that decide what nodes get built at all. Those reads must go through here.
//
// `t->data` is NOT a host pointer in general: on a Vulkan/CUDA/Metal backend it addresses device memory,
// so the `*(float*)t->data` these primitives used to do reads a device address on the host and either
// faults or returns garbage (BACKLOG.md P4.7). `ggml_backend_tensor_get` is the portable form and costs
// nothing extra on a CPU buffer, where it is a memcpy.

// Whether `t` already holds real data -- i.e. it is a weight (or any other tensor allocated before this
// graph was), rather than an as-yet-unallocated node of the graph under construction. Only a
// materialized tensor can be read at build time, on any backend.
bool is_materialized(const ggml_tensor* t);

// Copies the first `nbytes` bytes of `t` into `dst`, from whichever backend holds it. `t` must be
// materialized; a caller that is not sure asks is_materialized() first.
void read_tensor_prefix(const ggml_tensor* t, void* dst, size_t nbytes);

// Reads element 0 of `t` as a float, converting from whichever of ggml's f32/i32/i16/i8 types it holds.
// Returns `fallback` for a tensor that is not materialized or is of some other type -- the callers here
// are all "use this value if it is knowable, otherwise fall back to an axis", and none of them has
// anything better to do with an unknowable one.
float scalar_value_or(const ggml_tensor* t, float fallback);

// POOL_1D's fallback lowering: whether a one-tall `ggml_pool_2d` is the SAME operation for this
// `(op, p0)` combination. `op_pool_1d` reaches for it only when `backend_can_run` says the backend has
// no `ggml_pool_1d` -- CUDA and Vulkan do not, Metal and SYCL do. False for an AVERAGE pool with
// padding, the one combination where the two spellings genuinely disagree (they divide by different
// counts; see op_pool_1d in primitives_conv.cpp). Exposed so a test can hold the predicate and the
// behaviour against each other.
bool pool_2d_fallback_is_equivalent(ggml_op_pool op, int p0);

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
