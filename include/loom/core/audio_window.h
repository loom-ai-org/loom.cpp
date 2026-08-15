#pragma once

// Windowing a waveform for a model whose graph is built at ONE fixed clip length.
//
// WHY THIS IS IN THE ENGINE. Whisper's encoder is compiled at a single audio length -- its export
// declares `waveform` as a literal 480000 where every other ASR export declares a symbolic length --
// so a driver handed anything else gets a shape its graph cannot take. Every host therefore has to do
// the same three things before calling `infer`: read the clip length off the file, zero-pad or split
// the caller's audio to match, and supply the two decode arguments a fixed-clip driver needs.
//
// That was implemented twice -- once in `tools/loom_cli/main.cpp` and once in loom-py's binding -- and
// the second copy is what prompted this file. The facts below are properties of the FILE FORMAT, not
// of either front end: which key holds the clip length, that padding is zeros, that a driver whose
// early stop is unarmed will happily emit two hundred `<|endoftext|>` tokens. A third host should
// inherit them rather than rediscover them.
//
// WHAT IS DELIBERATELY NOT HERE: the timestamp-aware seek, which advances to where the model closed
// its last segment rather than by a fixed stride, so an utterance straddling a window edge is
// re-decoded whole. That needs a segment splitter and the model's declared timestamp ids, and it is
// long-form transcription POLICY rather than a property of the file -- so it lives one layer up, in
// `transcribe.h`, which owns the loop that calls these. An earlier version of this comment said it
// "stays in the CLI", which was true for about a day: the same reasoning that brought windowing here
// took the seek the rest of the way, since a policy every host needs is not a property of any one of
// them.

#include "loom/core/gguf_model.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace loom {
namespace audio {

// The clip length this model's graph is built at, in samples, or 0 when the length is dynamic.
//
// Zero is the common case and the right default: the NeMo families (Conformer-CTC, Parakeet, GigaAM)
// take any length, and a caller that gets 0 should pass its waveform through untouched.
inline uint32_t fixed_clip_samples(const GgufModel& model) {
    return model.has_kv("loom.n_samples") ? model.hparam_u32("n_samples") : 0;
}

// One window of exactly `clip` samples starting at `seek`, zero-padded when the audio runs out.
//
// Zeros rather than edge-repeat or reflection, because that is what Whisper's own `pad_or_trim` does
// and what the checkpoint was trained against -- a padded clip transcribes normally, where a clip
// padded some other way is a distribution the model has not seen.
inline std::vector<double> window_at(const std::vector<float>& waveform, size_t seek, uint32_t clip) {
    std::vector<double> window(clip, 0.0);
    if (seek >= waveform.size()) return window;
    const size_t avail = std::min(static_cast<size_t>(clip), waveform.size() - seek);
    for (size_t i = 0; i < avail; ++i) window[i] = static_cast<double>(waveform[seek + i]);
    return window;
}

// The end-of-sequence id that ARMS a driver's early stop, or -1 when the file names none.
//
// Load-bearing rather than an optimisation: the generated drivers treat a negative `eos_token` as
// "no early stop" (their own header says so), so a fixed-clip decode without this runs to the token
// ceiling and pads the transcript with end-of-text tokens. Measured on an 11 s clip: the sentence,
// then roughly two hundred of them.
inline int32_t default_eos_token(const GgufModel& model) {
    return model.kv_i32("tokenizer.ggml.eos_token_id", -1);
}

// How many tokens one window may generate.
//
// Half the declared text context, which is Whisper's own long-form convention (224 of its 448) --
// derived from the file rather than hardcoded, so a model with a different context gets a
// proportional ceiling instead of Whisper's. The fallback matches Whisper's context for a file that
// declares none, since a fixed-clip model that omits it is not a case that exists today.
//
// Without it the drivers fall back to their own `max_new_tokens or 16`, which truncates mid-sentence.
inline uint32_t default_max_new_tokens(const GgufModel& model) {
    const uint32_t text_ctx = model.has_kv("loom.n_text_ctx") ? model.hparam_u32("n_text_ctx") : 448;
    return text_ctx / 2;
}

} // namespace audio
} // namespace loom
