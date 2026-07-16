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

// Turns a parsed GraphTopology into a real ggml_cgraph for a specific (n_tokens, n_past) shape,
// implementing SPECIFICATION.md §5's two-pass allocation strategy: build a no_alloc "ghost graph" from
// the topology, then hand it to ggml_gallocr to assign real memory.
//
// GraphBuilder only builds and allocates -- it deliberately does NOT copy input tensor data in or run
// ggml_backend_graph_compute. The caller (Generator, a test, ...) owns that: it writes whatever data it
// has into BuildResult::input_tensors, then calls ggml_backend_graph_compute itself. This keeps
// "construct the graph" and "execute it" independently testable.
class GraphBuilder {
public:
    // `kv_cache` may be null for topologies that don't use the ATTENTION primitive (e.g. no
    // autoregressive state); passing one wires it into every primitive call via PrimitiveContext.
    GraphBuilder(const GraphTopology& topo, GgufModel& model, ggml_backend_t backend,
                 KvCache* kv_cache = nullptr, size_t compute_meta_bytes = 32 * 1024 * 1024);

    struct BuildResult {
        ggml_context_ptr ctx;   // owns every tensor/graph struct produced by this build; keep alive
                                // until you're done reading outputs, then let it drop.
        ggml_cgraph* graph = nullptr;
        ggml_tensor* output = nullptr; // the topology's declared "output" symbol
        std::unordered_map<std::string, ggml_tensor*> input_tensors; // topology's declared inputs, by name
    };

    // Builds and allocates a graph for exactly (n_tokens, n_past). Always correct regardless of what
    // reserve() has (or hasn't) been called with -- ggml_gallocr_alloc_graph reallocates automatically
    // if the requested graph exceeds whatever was previously reserved (see ggml-alloc.h).
    BuildResult build(uint32_t n_tokens, uint32_t n_past);

    // Builds worst-case prefill (n_tokens=n_ctx_max, n_past=0) and decode (n_tokens=1,
    // n_past=n_ctx_max-1) shapes and reserves the allocator for the larger of the two, so that ordinary
    // build() calls within those bounds don't trigger a gallocr reallocation. Purely a performance
    // optimization -- build() alone is already correct without ever calling this.
    void reserve(uint32_t n_ctx_max);

    // Current size (in bytes) of the gallocr-managed compute buffer, or 0 if build()/reserve() haven't
    // been called yet. Exposed mainly so tests can confirm reserve() sizes the allocator once and
    // ordinary build() calls within those bounds don't grow it further.
    size_t buffer_size() const;

private:
    const GraphTopology& topo_;
    GgufModel& model_;
    ggml_backend_t backend_;
    KvCache* kv_cache_;
    size_t compute_meta_bytes_;
    ggml_gallocr_ptr galloc_;
};

} // namespace loom
