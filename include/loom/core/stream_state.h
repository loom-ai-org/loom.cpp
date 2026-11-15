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
#include <ggml-backend.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class OutputStore;

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

    // Everything dropped: a stream reopened from scratch.
    void clear();

    // Bytes the tensor entries hold on their backends, and the values on the host.
    size_t tensor_bytes() const;
    size_t value_bytes() const;

private:
    std::unordered_map<std::string, Value> values_;
    std::unordered_map<std::string, std::unique_ptr<OutputStore>> tensors_;
};

} // namespace loom
