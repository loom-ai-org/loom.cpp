#pragma once
// Audio out: a TTS hands back its waveform a piece at a time, while the rest is still being made
// (ADR-100). The third live door, after `TranscriptionStream` (audio in) and `GenerationStream` (tokens
// out), on the same `StreamState`.
//
// THE DRIVER CONTRACT. A file that can stream exports, beside its one-pass `infer`:
//
//   stream_open(inputs)    takes `infer`'s inputs whole, keeps what the pieces need in the stream's state
//                          and returns any audio it already made (usually none)
//   stream_feed({})        makes the next piece and returns its samples
//   stream_finish({})      returns whatever audio is left (usually none)
//
// and keeps two state values: `piece`, the index of the next piece, which every `stream_feed` advances,
// and `done`, set once the last piece is made. `n_pieces`, when the driver knows it up front, is how many
// there are. A piece is the driver's own unit -- a sentence chunk for Pocket-TTS and Soprano, whose
// references decode each chunk from a fresh state, so the pieces joined are the one-pass waveform
// exactly. `infer` is the stream opened, fed to the end and finished inside one call.
//
// The input is the whole text: nothing is pushed after `stream_open`. Text arriving while audio is made
// (an LM feeding a TTS) is a different contract.
//
// A seeded stream (`inputs.seed`) draws from a generator of its own, so its audio is the one-pass audio
// for that seed however other streams on the model interleave with it.
#include "loom/core/lua_bridge.h"
#include "loom/core/stream_state.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {
namespace audio {

class SpeechStream {
public:
    using Inputs = std::unordered_map<std::string, LoomLuaBridge::Value>;

    // `inputs` are what `infer` takes (`tokens`, a voice's inputs, `seed`, the sampler's knobs). Throws
    // loom::LoadError when the driver has no streaming entry points. `bridge` is not owned and must
    // outlive this object. Nothing runs until the first `next`.
    SpeechStream(LoomLuaBridge& bridge, Inputs inputs);

    // Whether the loaded driver exports `stream_open`, `stream_feed` and `stream_finish`.
    static bool supported(const LoomLuaBridge& bridge);

    // The next piece's samples, at the model's own rate. Empty only once the stream has finished: a piece
    // that made no audio is skipped over, not handed back.
    std::vector<float> next();
    // Drops everything; the next `next` opens the stream again with the same inputs.
    void reset();

    bool finished() const { return finished_; }
    // Pieces made so far, and how many there are (0 when the driver does not say up front).
    size_t pieces_done() const;
    size_t pieces_total() const;
    // Samples handed back so far.
    size_t samples() const { return samples_; }
    const StreamState& state() const { return state_; }

private:
    double state_number(const char* name) const;
    static std::vector<float> as_samples(const LoomLuaBridge::Value& v);

    LoomLuaBridge& bridge_;
    Inputs inputs_;
    StreamState state_;
    size_t samples_ = 0;
    bool opened_ = false;
    bool finished_ = false;
};

} // namespace audio
} // namespace loom
