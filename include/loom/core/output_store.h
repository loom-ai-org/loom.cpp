#pragma once

#include <ggml-cpp.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {

// Persistent, module-owned storage for a topology's declared OUTPUTS -- the third member of the
// engine's persistent-state family, after `KvCache` (attention) and `ConvStateCache` (conv/SSM), and
// deliberately built to the same seam (BACKLOG.md P4.0.12).
//
// Why it exists. Splitting a forward pass from the reduction that follows it appeared to require
// handing Lua an opaque tensor handle, and a `GraphBuilder::BuildResult` is only readable while the
// builder that produced it is alive -- so such a handle would dangle the moment the call returned. The
// KV cache dissolves that: it is persistent state addressed *by module name*, with no address ever
// crossing the scripting boundary (KV-CACHE.md 1.1). An output buffer works exactly the same way, and
// `loom.get_output('main_topology', 1)` names a module, which is what every Lua call already does.
//
// **The motivating case is inter-module data flow, not the large vocab.** A Lua driver that chains
// module A into module B otherwise reads A's output into a Lua table and writes it straight back as B's
// input: on CPU two copies of an intermediate nobody looks at, and on a GPU backend a
// device->host->device round trip per edge per step. With a retained output, B's input is filled by a
// backend-to-backend copy and the values never become Lua doubles at all. The rule this makes possible
// is **marshal only when a value is genuinely host-side** -- a final result, a control decision, or
// host math the driver actually performs.
//
// Same seam as the two caches: storage in its own ggml_context outside the compute graph so ggml-alloc
// never sees it; the write returned as a cpy node the caller routes through
// `PrimitiveContext::side_effects`; reads served as plain data, never as an address.
//
// **Where it differs from them, and why.** A cache's geometry is fixed at construction from declared
// hparams. An output's is not: the same topology produces `[n_vocab, n_tokens]` at prefill and
// `[n_vocab, 1]` at decode, and no hparam says which. So the store is shaped by the build that fills
// it -- `reshape()` reallocates only when the geometry actually moves, which for a decode loop is once,
// at the prefill->decode transition. That is still "the address is stable regardless of whether the
// graph was rebuilt" in the sense that matters: retrieval looks the buffer up by module name at read
// time, so it can never hold a pointer the store has since replaced.
class OutputStore {
public:
    explicit OutputStore(ggml_backend_t backend);

    // Makes this store hold exactly `outs`' geometry (count, type and ne[] of each), reallocating its
    // context and backend buffer if and only if that geometry differs from what it holds now, and
    // returns the destination tensors in the same order. Any previously stored data is discarded on a
    // reshape -- which is safe because a build fills *every* declared output, so a run either rewrites
    // all of them or reshapes and then rewrites all of them.
    const std::vector<ggml_tensor*>& reshape(const std::vector<ggml_tensor*>& outs);

    // The retained tensor for declared output `index` (0-based). Throws loom::Error if the store has
    // never been filled, or if the index is past what the last run produced.
    ggml_tensor* get(size_t index) const;

    size_t size() const { return slots_.size(); }
    bool filled() const { return generation_ > 0; }

    // Raised once per completed run, so a driver can pin a read to the run it meant. Starts at 0
    // ("never run"); `check_generation` rejects a read against any other value than the current one,
    // which is what turns "a second run on this module silently returned newer data" into an error.
    uint64_t generation() const { return generation_; }
    void bump_generation() { ++generation_; }
    // `module` only names the module in the error message; the store itself knows nothing about names.
    void check_generation(uint64_t expected, const std::string& module) const;

    OutputStore(const OutputStore&) = delete;
    OutputStore& operator=(const OutputStore&) = delete;

private:
    struct Geometry {
        ggml_type type;
        int64_t ne[GGML_MAX_DIMS];
        bool operator==(const Geometry& other) const;
    };

    ggml_backend_t backend_;
    ggml_context_ptr store_ctx_;
    ggml_backend_buffer_ptr store_buf_;
    std::vector<ggml_tensor*> slots_;
    std::vector<Geometry> geometry_;
    uint64_t generation_ = 0;
};

} // namespace loom
