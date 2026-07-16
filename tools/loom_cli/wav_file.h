#pragma once

#include <string>
#include <vector>

namespace loom_cli {

// Minimal 16-bit PCM WAV reader (CLI-only, not part of the engine library): reads mono, or the first
// (left) channel of a multi-channel file, normalized to [-1, 1] float32. Throws std::runtime_error if
// the file isn't a valid 16-bit-PCM WAV, or if its sample rate isn't 16000Hz -- resampling is out of
// scope (see BACKLOG.md), so a clear error beats silently-wrong results.
std::vector<float> load_wav_pcm16_mono_16k(const std::string& path);

} // namespace loom_cli
