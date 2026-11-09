#pragma once
// One persistent cache, many live streams (ADR-099).
//
// A `KvCache` or `ConvStateCache` belongs to the loaded model: the session allocates one from the
// file's hparams and lends it to every module that declares one. ADR-098 put a stream's own state in
// a `StreamState` the host owns, so that streams on one model are independent -- but a decode loop
// that runs across calls (a token stream, a frame-synchronous recogniser) keeps its history in the
// cache, which is shared. Two ways to keep the streams independent, chosen per cache by
// `StreamCaches`:
//
//   * PARK -- this class. The cache is LEASED to one stream at a time. Whoever ran last owns it; when a
//     different stream (or a one-pass call) runs a module on it, the owner's rows are copied into the
//     owner's `ParkedRows`, and the newcomer's parked rows, if it has any, are copied back in. One
//     stream alone never copies anything. Measured on the dev box (8 GB/s for a park plus a restore):
//     Qwen3-0.6B at 1000 positions is ~57 ms per switch against a 155 ms decode step.
//   * PER STREAM -- the stream allocates a cache of its own (`StreamState::own_cache`) and no lease is
//     ever taken. No copy, a full cache per stream. Where parking would have to keep the parked rows
//     in VRAM to avoid a PCIe round trip -- a discrete GPU -- it costs the memory a cache of its own
//     costs and still copies, so `Auto` picks this there.
//
// The owner is held WEAKLY, by the parking slot rather than by a pointer to the stream: a stream that
// is dropped leaves nothing to park into, and the next lease simply finds the slot expired. A raw
// pointer could not tell that apart from a new stream allocated at the same address.
#include <cstdint>
#include <memory>
#include <vector>

namespace loom {

// One stream's rows of one cache, while another stream holds the cache. Empty means "nothing parked":
// either the stream holds the cache right now, or it never wrote to it.
struct ParkedRows {
    std::vector<float> data;
};

class LeasedCache {
public:
    LeasedCache();
    virtual ~LeasedCache();

    // Never reused within a process, unlike the object's address. What a builder compares to tell
    // whether the cache it last built against is the one it is being handed now.
    uint64_t serial() const { return serial_; }

    // Hands the cache to `to`'s stream. A no-op when it holds the cache already. Otherwise the current
    // owner's rows (if it still exists) go into its slot, and `to`'s parked rows come back -- or, for a
    // stream that has none, the cache is marked empty for it.
    void lease(const std::shared_ptr<ParkedRows>& to);

    // How many times a lease has copied rows out. Tests use it to prove one stream alone costs nothing.
    uint64_t parks() const { return parks_; }

    LeasedCache(const LeasedCache&) = delete;
    LeasedCache& operator=(const LeasedCache&) = delete;

protected:
    virtual void park(std::vector<float>& out) const = 0;
    virtual void restore(const std::vector<float>& in) = 0;
    // A stream that never wrote takes the cache: forget the previous owner's extent.
    virtual void forget() = 0;

private:
    std::weak_ptr<ParkedRows> owner_;
    uint64_t serial_;
    uint64_t parks_ = 0;
};

} // namespace loom
