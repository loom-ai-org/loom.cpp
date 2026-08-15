#pragma once

// Long-form speech transcription: the whole loop, in the one place both front ends can reach it.
//
// WHY THIS IS IN THE ENGINE RATHER THAN IN A HOST. An earlier version of this lived in
// `tools/loom_cli/main.cpp`, and when loom-py needed the same thing the reflex was to reimplement the
// easy half -- windowing -- and tell Python users that the CLI transcribed long audio better. That is
// not a defensible thing to ship. The parts that make the CLI better are not properties of the CLI:
// the timestamp ids come from the vocabulary the GGUF embeds, the frame duration is arithmetic on
// three declared hparams, and the seek policy is the same for any caller. Everything needed is here,
// so it belongs here, and a host is left with formatting.
//
// WHAT THE LOOP ACTUALLY DOES, since "windowing" undersells it:
//
//   * a model whose graph is built at one clip length (Whisper: `loom.n_samples`) gets its audio in
//     zero-padded windows of exactly that length -- see audio_window.h for the file-format facts;
//   * each window is decoded with the driver's early stop ARMED, because a driver without it emits
//     the sentence and then two hundred end-of-text tokens;
//   * the output is split into timestamped segments, because Whisper's `<|t|>` tokens both close the
//     span before them and open the one after;
//   * **the next window starts where the model closed its last segment**, not a fixed stride ahead,
//     so an utterance straddling a window edge is re-decoded whole instead of becoming two fragments.
//     This is the part that makes long-form transcription good, and the part a naive windower lacks;
//   * each window's tokens are carried forward as `prev_tokens` unless the caller turns that off.
//
// A model with a dynamic clip length -- the NeMo families, which is every ASR export but Whisper --
// takes one pass with the whole waveform and none of the above applies.

#include "loom/core/backend.h"
#include "loom/core/gguf_model.h"
#include "loom/core/lua_bridge.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace audio {

// One span the model closed with a timestamp, or the tail it had not finished when a window ran out.
struct Segment {
    double start = 0.0;
    double end = 0.0;
    std::string text;
    // False for text after the final timestamp of a window: real transcript, but its end time is the
    // window edge rather than a boundary the model chose, so it must not be used as a seek target.
    bool closed = false;
};

struct TranscribeOptions {
    // Negative means OMIT the argument rather than pass a default, which is how a driver is told to
    // decide for itself -- one that can detect the language does so, one that cannot falls back to its
    // own default, and only the driver knows which it is.
    int32_t language = -1;
    int32_t task = -1;
    // Ask the model for timestamps in the OUTPUT. They are requested internally whenever the seek
    // needs them regardless, so this only controls what the caller gets back.
    bool timestamps = false;
    // Carry each window's tokens into the next as `prev_tokens`.
    bool condition_on_previous = true;
};

struct Transcription {
    std::vector<Segment> segments;
    // The segments joined, which is what a caller who did not ask for timestamps wants.
    std::string text;
    // How many windows were decoded: 1 for a dynamic-length model or short audio.
    size_t windows = 0;
    // Whether the model exposed timestamp tokens at all. False means the segments below are window
    // slices rather than boundaries the model chose, which is worth a caller knowing.
    bool timestamped = false;
};

// Transcribes `waveform` with the model's own driver. Throws loom::LoadError when the file carries no
// vocabulary or no driver script, which are the two things this cannot substitute for.
Transcription transcribe(LoomLuaBridge& bridge, const GgufModel& model,
                         const std::vector<float>& waveform, const TranscribeOptions& options = {});

} // namespace audio
} // namespace loom
