#pragma once
// Audio in, lines of text out, for a recogniser that has no streaming mode of its own (ADR-127): Moonshine's
// streaming loop for its non-streaming models (moonshine-ai/moonshine `core/transcriber.cpp` and
// `core/voice-activity-detector.cpp`), with Silero VAD cutting the speech into lines and the recogniser's
// one-pass `transcribe` re-run over the line still being spoken.
//
// Two models, two sessions: a frame VAD whose `FrameStream` answers a `speech` probability per 512-sample
// hop (the zoo's silero-vad, re-exported with its live door, ADR-106), and any recogniser `transcribe`
// takes. Composing them is per task, not per model, which is why it is C++ here and not a driver.
//
// The loop, per push (Moonshine's `add_audio_to_stream` then `transcribe_stream`):
//   * pushed audio waits until `update_seconds` of it has gathered (compared in float, as upstream does);
//   * the VAD takes it in hops of `hop` samples, a remainder waiting for the next push (and dropped at the
//     end). Each hop's probability is averaged over a ring of the last `ceil(window_seconds * rate / hop)`,
//     scaled down over the last third of a `max_line_seconds` line so a line always ends, and compared
//     `> threshold`;
//   * voice starting opens a line with the last `look_behind` samples, the hop included; voice continuing
//     appends the hop; voice ending appends the hop and closes the line; `finish` closes an open one;
//   * every line a push touched is transcribed again from its first sample, the open one included
//     (`decode_incomplete`): the open line's text is revised until it closes, a closed line's is final.
// The numbers default to Moonshine's `TranscriberOptions`.
#include "loom/core/frame_stream.h"
#include "loom/core/gguf_model.h"
#include "loom/core/lua_bridge.h"
#include "loom/core/transcribe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {
namespace audio {

struct LineOptions {
    double update_seconds = 0.5;     // `transcription_interval`
    float threshold = 0.5f;          // `vad_threshold`
    float window_seconds = 0.5f;     // `vad_window_duration`
    uint32_t hop = 512;              // `vad_hop_size`
    size_t look_behind = 8192;       // `vad_look_behind_sample_count`
    float max_line_seconds = 15.0f;  // `vad_max_segment_duration`; 0: no limit
    bool decode_incomplete = true;   // `decode_incomplete_lines`
    // Moonshine's own token budget per line, `ceil(seconds * max_tokens_per_second)` in float, handed to the
    // recogniser as `max_new_tokens`. 0 (the default): the recogniser's own budget.
    float max_tokens_per_second = 0.0f;
    // The recogniser's options (language, task, ...), for every line.
    TranscribeOptions transcribe;
};

struct Line {
    // Where the line starts, and how long it is, in samples of the stream.
    size_t start = 0;
    size_t samples = 0;
    std::string text;
    bool complete = false;
};

struct LineUpdate {
    // Indices into `lines()` of the lines this call transcribed, in order; empty when it ran nothing.
    std::vector<size_t> touched;
    // Whether this call ran the VAD and the recogniser (enough audio had gathered, or the stream ended).
    bool ran = false;
    // Samples pushed so far, over every call.
    size_t samples = 0;
};

class LineStream {
public:
    // `asr` is the recogniser, `vad` the frame VAD; each on its own bridge, neither owned, both outliving
    // this object. Throws loom::Error when the VAD has no `speech` label, the two rates differ, or an
    // option is out of range, and loom::LoadError when the VAD has no live door.
    LineStream(LoomLuaBridge& asr_bridge, const GgufModel& asr, LoomLuaBridge& vad_bridge, const GgufModel& vad,
               LineOptions options = {});

    LineUpdate push(const float* samples, size_t n);
    LineUpdate push(const std::vector<float>& samples) { return push(samples.data(), samples.size()); }
    // Ends the input: what has gathered runs (whatever its length), the open line closes. A stream takes no
    // push after this; `reset` reopens it.
    LineUpdate finish();
    void reset();

    bool finished() const { return finished_; }
    const std::vector<Line>& lines() const { return lines_; }
    // The lines' texts joined by single spaces (empty texts skipped).
    std::string text() const;
    uint32_t sample_rate() const { return rate_; }
    // Per hop the VAD has taken: the speech probability, and what was compared to the threshold.
    const std::vector<float>& probabilities() const { return probs_; }
    const std::vector<float>& smoothed() const { return smoothed_; }

private:
    LineUpdate run(bool stopping);
    void hop(const float* samples);
    void transcribe_touched(LineUpdate& update);

    LoomLuaBridge& asr_bridge_;
    const GgufModel& asr_;
    LineOptions options_;
    std::unique_ptr<FrameStream> vad_;
    uint32_t rate_ = 0;
    uint32_t speech_column_ = 0;
    uint32_t window_ = 0;
    size_t max_segment_ = 0;

    std::vector<float> pending_;       // pushed, not yet given to the VAD
    std::vector<float> remainder_;     // short of a hop
    std::vector<float> look_behind_;
    std::vector<float> segment_;       // the open line's audio
    std::vector<float> window_probs_;
    uint32_t window_index_ = 0;
    bool previous_voice_ = false;
    size_t processed_ = 0;
    std::vector<Line> lines_;
    std::vector<std::vector<float>> line_audio_;  // per line, its audio as last touched
    std::vector<bool> touched_;
    std::vector<float> probs_, smoothed_;
    size_t samples_ = 0;
    bool finished_ = false;
};

} // namespace audio
} // namespace loom
