#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace loom_cli {

// A run of codec tokens on disk: what `--codes-out` writes and what a codec's `--prompt @file` reads.
//
// **A GGUF rather than text, because the width is the one thing the reader must not guess.** A codes
// file of 36 frames x 12 codebooks and one of 27 x 16 are both 432 numbers; flattened, the second
// decodes as the first with nothing raised, and a codec wider than the LM (MOSS-Audio-Tokenizer's 32
// against MOSS-TTS's 12) cannot tell a narrow row from a mis-split one at all. So the file states its
// shape -- a 2-D `codes` tensor, [n_frames, n_codebooks] row-major, and `loom.codes.n_codebooks` beside
// it -- and says what it is (`general.architecture = "loom-codes"`), the way a voice file does (ADR-045).
struct CodesFile {
    uint32_t n_codebooks = 0;
    std::vector<int32_t> codes;  // frame-major: all n_codebooks of frame 0, then frame 1, ...
    std::string source;          // the architecture of the model that produced them, when known
    size_t n_frames() const { return n_codebooks ? codes.size() / n_codebooks : 0; }
};

// Throws std::runtime_error if `codes.size()` is not a whole number of frames or the file cannot be
// written.
void write_codes_gguf(const std::string& path, const std::vector<int32_t>& codes, uint32_t n_codebooks,
                      const std::string& source);

// Whether `path` opens and starts with the GGUF magic -- how a codec's `--prompt @file` tells a codes
// file from a list of numbers.
bool is_gguf(const std::string& path);

// Throws std::runtime_error if `path` is not a codes file, or its tensor disagrees with its own width.
CodesFile read_codes_gguf(const std::string& path);

} // namespace loom_cli
