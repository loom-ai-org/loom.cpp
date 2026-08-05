#include "loom/core/output_store.h"

#include "loom/loom_errors.h"

#include <ggml-backend.h>

#include <string>

namespace loom {

bool OutputStore::Geometry::operator==(const Geometry& other) const {
    if (type != other.type) return false;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (ne[d] != other.ne[d]) return false;
    }
    return true;
}

OutputStore::OutputStore(ggml_backend_t backend) : backend_(backend) {}

const std::vector<ggml_tensor*>& OutputStore::reshape(const std::vector<ggml_tensor*>& outs) {
    std::vector<Geometry> wanted(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
        wanted[i].type = outs[i]->type;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) wanted[i].ne[d] = outs[i]->ne[d];
    }
    if (store_buf_ && wanted == geometry_) {
        return slots_;
    }

    // Drop the old buffer BEFORE its context: the tensors in `store_ctx_` are what the buffer's
    // allocation was computed from, and freeing them first would leave the buffer describing memory
    // whose owners no longer exist.
    store_buf_.reset();
    store_ctx_.reset();
    slots_.clear();

    const size_t mem_size = outs.size() * ggml_tensor_overhead() + 4096;
    store_ctx_.reset(ggml_init(ggml_init_params{mem_size, nullptr, /*no_alloc=*/true}));
    if (!store_ctx_) {
        throw Error("OutputStore: ggml_init failed for the persistent output context");
    }

    slots_.reserve(outs.size());
    for (size_t i = 0; i < outs.size(); ++i) {
        ggml_tensor* slot = ggml_new_tensor(store_ctx_.get(), wanted[i].type, GGML_MAX_DIMS, wanted[i].ne);
        ggml_format_name(slot, "retained_out_%d", static_cast<int>(i));
        slots_.push_back(slot);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(store_ctx_.get(), backend_);
    if (!buf) {
        throw Error("OutputStore: failed to allocate the persistent output backend buffer");
    }
    store_buf_.reset(buf);
    geometry_ = std::move(wanted);
    return slots_;
}

ggml_tensor* OutputStore::get(size_t index) const {
    if (!filled()) {
        throw Error("OutputStore: nothing retained yet -- the module has not been run with its outputs "
                     "retained (loom.run_subgraph_and_retain)");
    }
    if (index >= slots_.size()) {
        throw Error("OutputStore: output index " + std::to_string(index) + " is out of range; the last "
                     "run retained " + std::to_string(slots_.size()) + " output(s)");
    }
    return slots_[index];
}

void OutputStore::check_generation(uint64_t expected, const std::string& module) const {
    if (expected == generation_) return;
    throw Error("OutputStore: stale read of module '" + module + "': the caller pinned generation " +
                 std::to_string(expected) + ", but this module's retained outputs are now at generation " +
                 std::to_string(generation_) +
                 " -- the module was re-run between the run that produced the value and this read");
}

} // namespace loom
