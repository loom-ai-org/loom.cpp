#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace loom_cli {

// Minimal 16-bit PCM WAV reader (CLI-only, not part of the engine library): reads mono, or the first
// (left) channel of a multi-channel file, normalized to [-1, 1] float32. Throws std::runtime_error if
// the file isn't a valid 16-bit-PCM WAV, or if its sample rate isn't 16000Hz -- resampling is out of
// scope (see BACKLOG.md), so a clear error beats silently-wrong results.
std::vector<float> load_wav_pcm16_mono_16k(const std::string& path);

// The other direction, for the models whose answer is audio. 16-bit PCM mono at `rate`, samples
// clipped to [-1, 1] -- a synthesised waveform that leaves that range is a conditioning bug rather
// than something to wrap around, and clipping keeps the file playable while the peak this prints
// says so. Throws std::runtime_error if the file cannot be written.
void write_wav_pcm16_mono(const std::string& path, const std::vector<float>& samples, uint32_t rate);

} // namespace loom_cli
