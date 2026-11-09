#include "loom/core/stream_state.h"

#include "loom/core/conv_state_cache.h"
#include "loom/core/kv_cache.h"
#include "loom/core/output_store.h"

namespace loom {

bool stream_owns_cache(StreamCaches mode, ggml_backend_t backend) {
    switch (mode) {
        case StreamCaches::Park: return false;
        case StreamCaches::PerStream: return true;
        case StreamCaches::Auto: break;
    }
    // Only a discrete GPU: an integrated one (and the CPU, and a host-memory accelerator such as the
    // Pi's QPU) parks into the very RAM its cache lives in, and RAM is what an edge device runs out of.
    // Metal reports GPU on unified memory too, and gets a cache per stream, as the user asked.
    ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr;
    return dev != nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
}

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
    // Emptied rather than erased: a cache this stream holds has it as its weak owner, and a fresh slot
    // would make the next lease park the reopened stream's rows nowhere.
    for (auto& [serial, slot] : parked_) std::vector<float>().swap(slot->data);
}

const std::shared_ptr<ParkedRows>& StreamState::parked(uint64_t cache_serial) {
    auto& slot = parked_[cache_serial];
    if (!slot) slot = std::make_shared<ParkedRows>();
    return slot;
}

KvCache& StreamState::own_cache(const KvCache& shared) {
    auto& slot = own_kv_[shared.serial()];
    if (!slot) slot = shared.make_like();
    return *slot;
}

ConvStateCache& StreamState::own_cache(const ConvStateCache& shared) {
    auto& slot = own_conv_[shared.serial()];
    if (!slot) slot = shared.make_like();
    return *slot;
}

void StreamState::seed_rng(uint32_t seed) {
    rng_ = std::make_unique<Rng>();
    rng_->engine.seed(seed);
}

size_t StreamState::tensor_bytes() const {
    size_t total = 0;
    for (const auto& [name, store] : tensors_) {
        if (!store->filled()) continue;
        for (size_t i = 0; i < store->size(); ++i) total += ggml_nbytes(store->get(i));
    }
    return total;
}

size_t StreamState::parked_bytes() const {
    size_t total = 0;
    for (const auto& [serial, slot] : parked_) total += slot->data.size() * sizeof(float);
    return total;
}

size_t StreamState::cache_bytes() const {
    size_t total = 0;
    for (const auto& [serial, kv] : own_kv_) {
        total += static_cast<size_t>(kv->n_layer()) * (kv->n_embd_k() + kv->n_embd_v()) * kv->kv_size() * sizeof(float);
    }
    for (const auto& [serial, conv] : own_conv_) {
        total += static_cast<size_t>(conv->n_layer()) * conv->n_state() * conv->n_embd_conv() * sizeof(float);
    }
    return total;
}

size_t StreamState::value_bytes() const {
    size_t total = 0;
    for (const auto& [name, v] : values_) total += v.data.size() * sizeof(double);
    return total;
}

} // namespace loom
