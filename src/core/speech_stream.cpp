#include "loom/core/speech_stream.h"

#include "loom/loom_errors.h"

#include <cstdint>
#include <utility>

namespace loom {
namespace audio {

namespace {
const char* const kEntryPoints[] = {"stream_open", "stream_feed", "stream_finish"};
} // namespace

bool SpeechStream::supported(const LoomLuaBridge& bridge) {
    for (const char* fn : kEntryPoints) {
        if (!bridge.has_function(fn)) return false;
    }
    return true;
}

SpeechStream::SpeechStream(LoomLuaBridge& bridge, Inputs inputs) : bridge_(bridge), inputs_(std::move(inputs)) {
    if (!supported(bridge)) {
        throw LoadError("this model's driver has no live path (stream_open / stream_feed / stream_finish): "
                        "it was exported before streaming, or its family does not stream yet. Its one-pass "
                        "synthesis is unaffected.");
    }
    reset();
}

void SpeechStream::reset() {
    state_.clear();
    // Seeded here as well as by the driver's own `loom.seed_rng`: the driver's call reseeds whichever
    // generator is swapped in, and without one of the stream's own that is the bridge's, which another
    // stream's draws would move between two of this one's pieces.
    const auto seed = inputs_.find("seed");
    if (seed != inputs_.end() && std::holds_alternative<double>(seed->second)) {
        state_.seed_rng(static_cast<uint32_t>(std::get<double>(seed->second)));
    }
    samples_ = 0;
    opened_ = false;
    finished_ = false;
}

std::vector<float> SpeechStream::next() {
    std::vector<float> out;
    if (finished_) return out;
    if (!opened_) {
        opened_ = true;
        out = as_samples(bridge_.call("stream_open", inputs_, state_));
    }
    while (out.empty() && state_number("done") == 0.0) {
        const double before = state_number("piece");
        out = as_samples(bridge_.call("stream_feed", {}, state_));
        // An empty piece is allowed (a Soprano sentence too short to decode), so the cursor is what says
        // the driver moved on; a feed that neither advanced nor finished would be called forever.
        if (out.empty() && state_number("done") == 0.0 && state_number("piece") == before) {
            throw Error("SpeechStream: the driver's stream_feed made no audio, did not advance its `piece` "
                        "and did not say it was done, so another call would not either.");
        }
    }
    if (state_number("done") != 0.0) {
        finished_ = true;
        const std::vector<float> tail = as_samples(bridge_.call("stream_finish", {}, state_));
        out.insert(out.end(), tail.begin(), tail.end());
    }
    samples_ += out.size();
    return out;
}

size_t SpeechStream::pieces_done() const { return static_cast<size_t>(state_number("piece")); }

size_t SpeechStream::pieces_total() const { return static_cast<size_t>(state_number("n_pieces")); }

double SpeechStream::state_number(const char* name) const {
    const StreamState::Value* v = state_.value(name);
    return (v == nullptr || v->data.empty()) ? 0.0 : v->data.front();
}

std::vector<float> SpeechStream::as_samples(const LoomLuaBridge::Value& v) {
    if (!std::holds_alternative<std::vector<double>>(v)) return {};
    const auto& d = std::get<std::vector<double>>(v);
    return std::vector<float>(d.begin(), d.end());
}

CodecStream::CodecStream(LoomLuaBridge& bridge, Inputs inputs) : bridge_(bridge), inputs_(std::move(inputs)) {
    if (!supported(bridge)) {
        throw LoadError("this codec's driver has no live path (stream_open / stream_feed / stream_finish): "
                        "it was exported before streaming, or its decoder does not stream yet. Its one-pass "
                        "decode is unaffected.");
    }
    reset();
}

void CodecStream::reset() {
    state_.clear();
    // Always a generator of its own (ADR-104): a stochastic codec (SNAC) draws noise as frames arrive, and
    // drawing from the bridge's would let another stream's -- or a one-pass decode's -- draws land between
    // two of this one's pushes. The driver's `stream_open` seeds it (from `inputs.seed`, else the file's
    // default), so the 0 here is never what a draw sees.
    const auto seed = inputs_.find("seed");
    state_.seed_rng(seed != inputs_.end() && std::holds_alternative<double>(seed->second)
                        ? static_cast<uint32_t>(std::get<double>(seed->second)) : 0u);
    samples_ = 0;
    finished_ = false;
    bridge_.call("stream_open", inputs_, state_);
}

std::vector<float> CodecStream::push(const std::vector<int32_t>& codes) {
    if (finished_) throw Error("CodecStream::push: the stream has finished; reset() reopens it");
    return absorb(bridge_.call("stream_feed", {{"codes", std::vector<double>(codes.begin(), codes.end())}}, state_));
}

std::vector<float> CodecStream::finish() {
    if (finished_) throw Error("CodecStream::finish: the stream has already finished");
    finished_ = true;
    return absorb(bridge_.call("stream_finish", {}, state_));
}

std::vector<float> CodecStream::absorb(const LoomLuaBridge::Value& v) {
    std::vector<float> out = SpeechStream::as_samples(v);
    samples_ += out.size();
    return out;
}

} // namespace audio
} // namespace loom
