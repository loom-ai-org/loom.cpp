#include "loom/core/stream_state.h"

#include "loom/core/output_store.h"

namespace loom {

StreamState::StreamState() = default;
StreamState::~StreamState() = default;
StreamState::StreamState(StreamState&&) noexcept = default;
StreamState& StreamState::operator=(StreamState&&) noexcept = default;

const StreamState::Value* StreamState::value(const std::string& name) const {
    const auto it = values_.find(name);
    return it == values_.end() ? nullptr : &it->second;
}

void StreamState::set_value(const std::string& name, Value value) { values_[name] = std::move(value); }

void StreamState::erase_value(const std::string& name) { values_.erase(name); }

OutputStore* StreamState::tensors(const std::string& name) const {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second.get();
}

OutputStore& StreamState::tensors_for_write(const std::string& name, ggml_backend_t backend) {
    auto& slot = tensors_[name];
    if (!slot) slot = std::make_unique<OutputStore>(backend);
    return *slot;
}

void StreamState::clear() {
    values_.clear();
    tensors_.clear();
}

size_t StreamState::tensor_bytes() const {
    size_t total = 0;
    for (const auto& [name, store] : tensors_) {
        if (!store->filled()) continue;
        for (size_t i = 0; i < store->size(); ++i) total += ggml_nbytes(store->get(i));
    }
    return total;
}

size_t StreamState::value_bytes() const {
    size_t total = 0;
    for (const auto& [name, v] : values_) total += v.data.size() * sizeof(double);
    return total;
}

} // namespace loom
