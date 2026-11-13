#pragma once
// Audio in, rows out: a frame classifier or frame embedder answers each frame once its audio has arrived
// (ADR-106). The fifth live door, after `TranscriptionStream`, `GenerationStream`, `SpeechStream` and
// `CodecStream`, on the same `StreamState`; `classify` and `embed` (audio_classify.h) are its one-pass
// halves.
//
// THE DRIVER CONTRACT. ADR-098's three functions, as `TranscriptionStream` calls them:
//
//   stream_open(inputs)               starts a stream; nothing is run yet. `inputs` are the caller's
//                                     options (MarbleNet's `lookahead_frames`, ADR-120)
//   stream_feed({waveform = pcm})     takes the next samples, returns the rows of the frames they completed
//   stream_finish(inputs)             the input has ended: returns the rows of what is left
//
// A row is what one frame of `infer`'s answer is -- the declared labels' probabilities, or
// `loom.output.embedding_dim` values -- and the rows come back in frame order, each exactly once, so the
// rows of every call joined are one call's answer. Which frame a row is, and when it starts, is the count
// of rows before it and the file's `frame_rate` / `frame_offset`: the same cut `classify` makes, made
// here once so two hosts cannot disagree about it, and checked the same way -- an answer that is not whole
// rows is an error naming both numbers.
//
// Only a `frame` answer streams. A clip classifier's one row needs the whole clip; it has nothing to
// hand back before the end, and its door is `classify`.
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

struct FrameUpdate {
    // `n_rows` rows of `FrameStream::width()` values, row-major.
    std::vector<float> values;
    // The frame index of the first row, counted over the whole stream.
    uint32_t first_row = 0;
    uint32_t n_rows = 0;
    // Samples pushed so far, over every call.
    size_t samples = 0;
};

class FrameStream {
public:
    using Inputs = std::unordered_map<std::string, LoomLuaBridge::Value>;

    // Opens a stream on a loaded model (`stream_open`, handed `inputs`; `reset` hands them again). Throws
    // loom::LoadError when the driver has no streaming entry points, and loom::Error when the contract is
    // not audio in and `class` or `embeddings` out per `frame`, or does not say how wide a row is. `bridge`
    // is not owned and must outlive this object.
    FrameStream(LoomLuaBridge& bridge, const GgufModel& model, Inputs inputs = {});

    // Whether the loaded driver exports `stream_open`, `stream_feed` and `stream_finish`.
    static bool supported(const LoomLuaBridge& bridge);

    // Samples at the model's own rate (`loom.sample_rate`), mono, any count including zero.
    FrameUpdate push(const float* samples, size_t n);
    FrameUpdate push(const std::vector<float>& samples) { return push(samples.data(), samples.size()); }
    // Ends the input and returns the rows of what is left. A stream takes no push after this; `reset`
    // reopens it.
    FrameUpdate finish();
    // Drops everything and opens the stream again, on the same model.
    void reset();

    bool finished() const { return finished_; }
    // `modality::CLASS` or `modality::EMBEDDINGS`.
    const std::string& output_kind() const { return output_kind_; }
    // The declared labels, for a class answer; empty for embeddings.
    const std::vector<std::string>& labels() const { return labels_; }
    uint32_t width() const { return width_; }
    double frame_rate() const { return frame_rate_; }
    double frame_offset() const { return frame_offset_; }
    // Rows handed back so far.
    uint32_t rows() const { return rows_; }
    // Where frame `row` starts, in seconds from the start of the stream.
    double row_start(uint32_t row) const { return frame_offset_ + static_cast<double>(row) / frame_rate_; }
    const StreamState& state() const { return state_; }

private:
    FrameUpdate absorb(const LoomLuaBridge::Value& result, const char* fn);

    LoomLuaBridge& bridge_;
    Inputs inputs_;
    std::string output_kind_;
    std::vector<std::string> labels_;
    uint32_t width_ = 0;
    double frame_rate_ = 0.0;
    double frame_offset_ = 0.0;
    StreamState state_;
    uint32_t rows_ = 0;
    size_t samples_ = 0;
    bool finished_ = false;
};

} // namespace audio
} // namespace loom
