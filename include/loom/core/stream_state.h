#pragma once
// The state of one live stream, owned by the engine and addressed by name -- the fourth member of the
// persistent-state family after `KvCache`, `ConvStateCache` and `OutputStore` (ADR-098).
//
// WHY THE ENGINE HOLDS IT, AND NOT THE DRIVER. A streaming driver exports `stream_open`, `stream_feed`
// and `stream_finish`, and something has to survive between those calls: a transducer's encoder caches,
// its decoder output, the audio not yet consumed, a feature cursor. Lua locals captured by the three
// functions would hold it, but then it lives in the bridge's one VM, so a model serves one stream at a
// time, a one-pass `infer` in between can clobber it, and the host can neither see nor reset it. Here
// the driver functions hold nothing; a host keeps one `StreamState` per stream and hands it to every
// call (`LoomLuaBridge::call(fn, args, state)`), so two streams share one loaded model and its weights.
//
// TWO KINDS OF ENTRY, by where the data lives:
//
//   * tensors -- named `OutputStore`s on the device. `loom.keep_state(name, module)` copies every
//     retained output of `module` into one, and a `run_subgraph` input reads it back as
//     `{state = name, index = k}` with the same `row`/`rows` a `{from = module}` reference takes. The
//     values never become Lua doubles, which is `output_store.h`'s rule for anything nothing host-side
//     reads.
//   * values -- small host arrays (`loom.state(name)`, `loom.set_state(name, v)`): a cursor, a token
//     context, audio waiting for the next chunk. A scalar stays a scalar on the way back.
//
// A one-pass `call(fn, args)` gets a fresh state for that call alone, so `infer` can be `stream_open`,
// one `stream_feed` with the whole input and `stream_finish`, and the one-pass gate tests the stream's
// code.
//
// THE MODEL'S CACHES (ADR-099). A decode loop that runs across calls keeps its history in the model's
// `KvCache` / `ConvStateCache`, which every stream shares. A state either PARKS -- leases the shared
// cache, its rows copied out while another stream runs (`cache_lease.h`) -- or owns a cache of its own,
// as `StreamCaches` says per backend. A one-pass call's state always leases: it is gone when the call
// returns, and a cache allocated for it would be too.
//
// ITS OWN RANDOM STREAM, once seeded. A sampled stream drawing from the bridge's one generator would
// draw a different sequence whenever another stream ran in between; a seeded state carries its own.
#include "loom/core/cache_lease.h"

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class ConvStateCache;
class KvCache;
class OutputStore;

// How a stream keeps its rows of the model's caches apart from other streams' (ADR-099).
enum class StreamCaches {
    Auto,      // PerStream on a discrete GPU, Park on the CPU, an integrated GPU or an accelerator
    Park,      // lease the model's cache; copy the rows out while another stream runs
    PerStream, // a cache of the stream's own; nothing copied, a full cache per stream
};
// What `mode` means for a cache on `backend`.
bool stream_owns_cache(StreamCaches mode, ggml_backend_t backend);

class StreamState {
public:
    StreamState();
    ~StreamState();
    StreamState(StreamState&&) noexcept;
    StreamState& operator=(StreamState&&) noexcept;
    StreamState(const StreamState&) = delete;
    StreamState& operator=(const StreamState&) = delete;

    struct Value {
        std::vector<double> data;
        // Set by a driver that stored a number rather than a table, so `loom.state` hands back a number.
        bool scalar = false;
    };

    // Null when nothing stored `name`.
    const Value* value(const std::string& name) const;
    void set_value(const std::string& name, Value value);
    void erase_value(const std::string& name);

    // Null when nothing kept `name`.
    OutputStore* tensors(const std::string& name) const;
    // The store `name` writes into, created on `backend` when absent. A store keeps its first backend:
    // `keep_state` reshapes it to the source's geometry and copies backend to backend.
    OutputStore& tensors_for_write(const std::string& name, ggml_backend_t backend);

    // Everything dropped: a stream reopened from scratch. Its own caches are kept for the next run,
    // which overwrites them from position 0; parked rows are not.
    void clear();

    // A one-pass call's state: always leases the model's caches, whatever the mode.
    bool one_pass() const { return one_pass_; }
    void set_one_pass(bool one_pass) { one_pass_ = one_pass; }

    // This stream's slot for the shared cache whose serial is `cache_serial`.
    const std::shared_ptr<ParkedRows>& parked(uint64_t cache_serial);
    // This stream's own cache standing in for `shared`, allocated like it on first use.
    KvCache& own_cache(const KvCache& shared);
    ConvStateCache& own_cache(const ConvStateCache& shared);

    struct Rng {
        std::mt19937 engine;
        std::normal_distribution<float> normal{0.0f, 1.0f};
        std::uniform_real_distribution<float> uniform{0.0f, 1.0f};
    };
    // Gives the stream a generator of its own, seeded; null until then, which means the bridge's.
    void seed_rng(uint32_t seed);
    Rng* rng() const { return rng_.get(); }

    // Bytes the tensor entries hold on their backends, and the values on the host.
    size_t tensor_bytes() const;
    size_t value_bytes() const;
    // Rows parked on the host while another stream holds a cache, and the caches this stream owns.
    size_t parked_bytes() const;
    size_t cache_bytes() const;

private:
    std::unordered_map<std::string, Value> values_;
    std::unordered_map<std::string, std::unique_ptr<OutputStore>> tensors_;
    std::unordered_map<uint64_t, std::shared_ptr<ParkedRows>> parked_;
    std::unordered_map<uint64_t, std::unique_ptr<KvCache>> own_kv_;
    std::unordered_map<uint64_t, std::unique_ptr<ConvStateCache>> own_conv_;
    std::unique_ptr<Rng> rng_;
    bool one_pass_ = false;
};

} // namespace loom
