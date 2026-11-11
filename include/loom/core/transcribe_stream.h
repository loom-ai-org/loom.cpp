#pragma once
// Live transcription: audio pushed as it arrives, the transcript read back as it grows (ADR-098).
//
// THE DRIVER CONTRACT. A file that can stream exports three Lua functions beside its one-pass `infer`:
//
//   stream_open(inputs)               starts a stream; nothing is decoded yet. `inputs` are the host's
//                                     knobs for the live path (Moonshine's `update_seconds`), empty by
//                                     default; a driver ignores what it does not read
//   stream_feed({waveform = pcm})     takes the next samples, returns the token ids they completed
//   stream_finish(inputs)             the input has ended: decodes what is left, returns those ids
//
// The functions keep nothing between calls: every cache, cursor and pending sample is in the
// `StreamState` this object owns and hands to each call, so streams on one loaded model are independent
// and a one-pass `transcribe` between two pushes disturbs neither. A file without the three functions
// is refused at construction -- it was exported before streaming, or its family has no live path yet --
// and its one-pass door is unaffected, which is the backward-compatibility rule: streaming is a second
// door beside `transcribe`, never a change to it.
//
// What the ids mean is the transcript, so the same `AsrText` turns them into text as `transcribe` does.
// A transducer's emissions are final, so its ids are appended and `text` only grows.
//
// A RE-DECODING DRIVER (ADR-106) keeps its whole transcript so far in the state value `hypothesis`
// instead: an encoder-decoder whose decoder cross-attends to every encoder row (Moonshine streaming) has
// no prefix that a later row cannot change, so each decode replaces the last. When the driver has set
// `hypothesis`, it is the stream's ids after every call and the returned array is ignored; `revised`
// says the new text does not extend the old, and `delta` is then the whole of `text`.
#include "loom/core/asr_text.h"
#include "loom/core/gguf_model.h"
#include "loom/core/lua_bridge.h"
#include "loom/core/stream_state.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {
namespace audio {

struct TranscriptUpdate {
    // The ids this call committed, control ids included -- for a re-decoding driver, the whole
    // hypothesis.
    std::vector<int32_t> tokens;
    // The whole transcript so far.
    std::string text;
    // What this call appended to `text`. When `revised` is set the earlier text changed and `delta` is
    // the whole of `text`.
    std::string delta;
    bool revised = false;
    // Samples pushed so far, over every call.
    size_t samples = 0;
};

class TranscriptionStream {
public:
    using Inputs = std::unordered_map<std::string, LoomLuaBridge::Value>;

    // Opens a stream on a loaded model (`stream_open`, handed `inputs`). Throws loom::LoadError when the
    // file's driver has no streaming entry points. `bridge` and `model` are not owned and must outlive
    // this object.
    TranscriptionStream(LoomLuaBridge& bridge, const GgufModel& model, Inputs inputs = {});

    // Whether the loaded driver exports `stream_open`, `stream_feed` and `stream_finish`.
    static bool supported(const LoomLuaBridge& bridge);

    // Samples at the model's own rate (`loom.sample_rate`), mono, any count including zero.
    TranscriptUpdate push(const float* samples, size_t n);
    TranscriptUpdate push(const std::vector<float>& samples) { return push(samples.data(), samples.size()); }
    // Ends the input and decodes what is left. A stream takes no push after this; `reset` reopens it.
    TranscriptUpdate finish();
    // Drops everything and opens the stream again, on the same model.
    void reset();

    bool finished() const { return finished_; }
    const std::vector<int32_t>& tokens() const { return tokens_; }
    const std::string& text() const { return text_; }
    // The engine-side state, for a host that reports its size.
    const StreamState& state() const { return state_; }

private:
    TranscriptUpdate absorb(const LoomLuaBridge::Value& result);

    LoomLuaBridge& bridge_;
    Inputs inputs_;
    AsrText asr_text_;
    StreamState state_;
    std::vector<int32_t> tokens_;
    std::string text_;
    size_t samples_ = 0;
    bool finished_ = false;
};

} // namespace audio
} // namespace loom
