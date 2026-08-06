#pragma once

#include "loom/core/graph_topology.h"
#include "loom/core/gguf_model.h"
#include "loom/core/symbol_table.h"

#include <ggml-cpp.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace loom {

class KvCache;
class ConvStateCache;
class OutputStore;

// A topology's own declared axis values for one build() call -- EXPORT-ROADMAP.md R1's named-axes
// design: every topology declares which axis(es) its dynamic dims are actually named (axes.py's
// n_samples/n_enc_frames/n_tokens/...), replacing the old assumption that every model has exactly one
// axis and it's always called "n_tokens". A plain name->value map, mirroring SymbolEnv itself (which
// is exactly that already) -- there is no extra structure to add beyond naming what goes in it.
using DynamicAxes = std::unordered_map<std::string, double>;

// Turns a parsed GraphTopology into a real ggml_cgraph for a specific (n_tokens, n_past) shape,
// implementing SPECIFICATION.md §5's two-pass allocation strategy: build a no_alloc "ghost graph" from
// the topology, then hand it to ggml_gallocr to assign real memory.
//
// GraphBuilder only builds and allocates -- it deliberately does NOT copy input tensor data in or run
// ggml_backend_graph_compute. The caller (Generator, a test, ...) owns that: it writes whatever data it
// has into BuildResult::input_tensors, then calls ggml_backend_graph_compute itself. This keeps
// "construct the graph" and "execute it" independently testable.
//
// ---------------------------------------------------------------------------------------------------
// THE BUILT GRAPH IS PERSISTENT AND REUSED (BACKLOG.md P4.0.13)
// ---------------------------------------------------------------------------------------------------
// A builder keeps the LAST graph it built -- its ggml_context, its ggml_cgraph, its gallocr-assigned
// compute buffer and its declared-input tensors -- and `build()` hands that same graph straight back
// when it is called again with the same axes. So a builder is now the unit of "one live graph", not a
// factory that produces a new one per call, and every loop that re-runs one module at a fixed shape
// (an ODE/diffusion sampler's steps, an LSTM's timesteps, a chained module in a decode loop) stops
// paying a rebuild and a throwaway compute-buffer allocation per iteration.
//
// Two consequences follow, and both are deliberate:
//
//   * `build()` returns a REFERENCE into the builder, so a result is valid only until the next build()
//     on that builder and only while the builder lives. That was already the documented rule (see
//     BuildResult below); returning a reference is what stops it from being merely documented.
//   * The cache holds exactly ONE graph, the most recent. Not an LRU keyed by shape -- a retained
//     OutputStore is reshaped by the build that fills it, so "the last build" is the only entry whose
//     ggml_cpy destinations are guaranteed to still be the store's current tensors.
//
// The declared INPUT tensors get their own persistent ggml_context and backend buffer, outside the
// gallocr pool entirely -- the same seam KvCache/ConvStateCache/OutputStore use. That is what makes
// reuse safe rather than merely fast: ggml_gallocr may alias a computed tensor's buffer onto one of the
// graph's own declared inputs (tests/test_graph_reuse_safety.cpp pins that down as real ggml behaviour),
// which is why reusing a graph used to require rewriting EVERY declared input before EVERY compute. An
// input that gallocr never allocated cannot be aliased by anything gallocr placed, so that discipline
// is no longer load-bearing here; drivers that follow it anyway (OdeStepper) are simply unaffected.
class GraphBuilder {
public:
    // `kv_cache` may be null for topologies that don't use the ATTENTION primitive (e.g. no
    // autoregressive state); passing one wires it into every primitive call via PrimitiveContext.
    // `conv_state` is the same arrangement for SHORT_CONV, and is last so that every existing
    // positional call site keeps compiling unchanged -- a topology needs neither, either or both
    // (GraphTopology::uses_kv_cache / uses_conv_state answer which).
    GraphBuilder(const GraphTopology& topo, GgufModel& model, ggml_backend_t backend,
                 KvCache* kv_cache = nullptr, size_t compute_meta_bytes = 32 * 1024 * 1024,
                 ConvStateCache* conv_state = nullptr);

    struct BuildResult {
        // Owns every tensor/graph STRUCT produced by this build. The builder OWNS the BuildResult (see
        // the class comment): `build()` hands out a reference to it, valid until the next build() on
        // that same builder and only while the builder itself lives. The tensors' DATA lives in the
        // GraphBuilder's own gallocr and declared-input buffer, and the builder also holds `topo_` by
        // reference, so a copy of anything in here outlives its builder only as a dangling pointer --
        // reads may still appear to work, since whether the arena has been reused yet is
        // allocator-dependent (this cost a CI-only failure once; see
        // tests/test_graph_builder_shapes.cpp's multi-output test). Copy what you need into your own
        // storage before the builder goes out of scope, as every src/core/*_driver.cpp does.
        ggml_context_ptr ctx;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* output = nullptr; // the topology's primary declared output (== outputs.front())
        // Every declared output tensor, in the topology's own declared order -- outputs.front() ==
        // output. Single-output topologies (every model on the roadmap as of P2) get a one-element
        // vector; existing callers that only ever read `output` are unaffected. See EXPORT-ROADMAP.md
        // P2 / BACKLOG.md's implementation sequence.
        std::vector<ggml_tensor*> outputs;
        std::unordered_map<std::string, ggml_tensor*> input_tensors; // topology's declared inputs, by name
    };

    // Builds and allocates a graph for exactly the given axis values, or returns the previously built
    // one unchanged if `axes` and `out_store` are identical to the last call's. Always correct
    // regardless of what reserve() has (or hasn't) been called with -- ggml_gallocr_alloc_graph
    // reallocates automatically if the requested graph exceeds whatever was previously reserved (see
    // ggml-alloc.h).
    //
    // A REUSED graph keeps whatever its declared inputs last held: nothing is cleared between calls,
    // because the inputs live in the builder's own persistent buffer rather than the gallocr pool (see
    // the class comment). Rewriting every input every call therefore remains the clearest way to drive
    // one -- it is just no longer the thing standing between reuse and silent corruption.
    //
    // `axes` need only bind whatever names this specific topology's own declared shapes reference
    // (EXPORT-ROADMAP.md R1) -- SymbolEnv::get throws loom::SchemaError naming the missing symbol if
    // one is referenced but not bound, rather than silently defaulting it. If `axes` contains both
    // "n_tokens" and "n_past" and doesn't already declare "n_kv", it is set automatically to their sum
    // (n_past + n_tokens) -- the one derived axis a primitive itself reads directly from SymbolEnv
    // (`primitives_attention.cpp`'s ATTENTION op), rather than only ever appearing in a JSON shape
    // string, so every caller of an attention-bearing topology gets it without having to compute it.
    //
    // `out_store`, when given, is reshaped to this build's declared-output geometry and the graph ends
    // in a cpy of every declared output into it, routed through the same `side_effects` mechanism the
    // KV-cache writes use (BACKLOG.md P4.0.12). Those values then outlive this BuildResult -- and the
    // GraphBuilder itself -- because the store owns its own context and backend buffer, which is what
    // makes retrieval-by-module-name possible at all. A per-CALL argument rather than a constructor
    // one: whether a run retains its outputs is the caller's decision (`loom.run_subgraph_and_retain` vs
    // `loom.run_subgraph`), not a property of the module the way its caches are.
    const BuildResult& build(const DynamicAxes& axes, OutputStore* out_store = nullptr);

    // Builds worst-case prefill (n_tokens=n_ctx_max, n_past=0) and decode (n_tokens=1,
    // n_past=n_ctx_max-1) shapes and reserves the allocator for the larger of the two, so that ordinary
    // build() calls within those bounds don't trigger a gallocr reallocation. Purely a performance
    // optimization -- build() alone is already correct without ever calling this.
    //
    // **Calling this also disables the shrink below, permanently and deliberately.** The two are
    // opposite policies over the same buffer -- "hold the worst case so nothing ever reallocates" and
    // "give back what this shape does not need" -- and a builder cannot honour both. reserve() is the
    // caller stating a worst case, so it wins.
    void reserve(uint32_t n_ctx_max);

    // Current size (in bytes) of the gallocr-managed compute buffer, or 0 if build()/reserve() haven't
    // been called yet. Exposed mainly so tests can confirm reserve() sizes the allocator once and
    // ordinary build() calls within those bounds don't grow it further.
    size_t buffer_size() const;

    // How many build() calls actually constructed a graph, and how many were served from the retained
    // one. Exposed for the same reason buffer_size() is: a test can assert that a fixed-shape loop
    // rebuilds exactly once, which is the only externally visible difference reuse makes.
    uint64_t builds() const { return builds_; }
    uint64_t reuses() const { return reuses_; }
    // How many builds gave the oversized compute buffer back. A prefill-then-decode generation should
    // report exactly one, at the transition; a fixed-shape loop, zero.
    uint64_t shrinks() const { return shrinks_; }

private:
    const GraphTopology& topo_;
    GgufModel& model_;
    ggml_backend_t backend_;
    KvCache* kv_cache_;
    ConvStateCache* conv_state_;
    size_t compute_meta_bytes_;
    ggml_gallocr_ptr galloc_;

    // The declared inputs' own storage, outside the gallocr pool -- see the class comment for why they
    // are not allowed to share it. Replaced wholesale on every real rebuild, since a rebuild is exactly
    // the case where their shapes may have moved.
    ggml_context_ptr inputs_ctx_;
    ggml_backend_buffer_ptr inputs_buf_;

    // The one retained graph and the call that produced it. `cached_store_` is part of the key because
    // whether a build ends in a copy into an OutputStore is a property of the call, not of the module.
    BuildResult cached_;
    DynamicAxes cached_axes_;
    const OutputStore* cached_store_ = nullptr;
    bool has_cached_ = false;
    uint64_t builds_ = 0;
    uint64_t reuses_ = 0;
    // Set by reserve(); suppresses the shrink in build() -- see reserve()'s own comment.
    bool reserved_ = false;
    // Set by a build that grew the compute buffer, and the only thing that arms the shrink check on the
    // next one. gallocr only ever grows, so nothing else can make the buffer oversized -- and the check
    // is a second planning pass, which is far too expensive to run per build (measured in
    // graph_builder.cpp).
    bool may_shrink_ = false;
    uint64_t shrinks_ = 0;

    // Drops the gallocr when this graph needs materially less than the buffer currently holds, so the
    // next alloc sizes a fresh one. See build()'s own comment for why the allocator will not do this
    // by itself.
    void shrink_allocator_if_oversized(ggml_cgraph* gf);
};

} // namespace loom
