#include "loom/core/cache_lease.h"

#include <atomic>

namespace loom {

namespace {
std::atomic<uint64_t> g_next_serial{1};
} // namespace

LeasedCache::LeasedCache() : serial_(g_next_serial.fetch_add(1)) {}

LeasedCache::~LeasedCache() = default;

void LeasedCache::lease(const std::shared_ptr<ParkedRows>& to) {
    const std::shared_ptr<ParkedRows> owner = owner_.lock();
    if (owner == to) return;
    if (owner) {
        park(owner->data);
        ++parks_;
    }
    if (to->data.empty()) {
        forget();
    } else {
        restore(to->data);
        // The rows live in the cache again; a second copy on the host would double the stream's
        // footprint for as long as it keeps the lease.
        std::vector<float>().swap(to->data);
    }
    owner_ = to;
}

} // namespace loom
