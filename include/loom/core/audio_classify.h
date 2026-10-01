#pragma once

// The audio-classifier and audio-embedder doors: a waveform in, and out of it a class distribution
// per clip or per frame, or one embedding vector (P5, family 13; ADR-062).
//
// WHY THIS IS IN THE ENGINE, when the call itself is one `infer`. `text_classify.h` gives the argument
// for policy over loops, and it applies here as SHAPE over loops: the driver returns one flat array,
// and which rows it holds, what each row's columns are named and what time a frame row covers are
// read off the file's contract (`loom.output.granularity`, `loom.labels`, `loom.output.frame_rate`,
// `loom.output.frame_offset`). Two hosts reshaping it independently is how they would come to disagree
// about which frame is which -- so it is done once, here, and checked: an answer whose size is not a
// whole number of label rows is an error naming both numbers, not a silently skewed table.
//
// WHAT IS NOT HERE. A decision. The answer is the model's own distribution, and thresholding a VAD,
// smoothing it, picking a language's top-k or comparing two embeddings are the caller's rules.

#include "loom/core/gguf_model.h"
#include "loom/core/lua_bridge.h"
#include "loom/core/model_contract.h"

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace audio {

// What `classify` returns: `n_rows` rows of `labels.size()` probabilities, row-major.
struct ClassProbabilities {
    // `granularity::FRAME` or `granularity::CLIP`. A clip answer is one row.
    std::string granularity;
    std::vector<std::string> labels;
    uint32_t n_rows = 0;
    // Frames per second and the start of frame 0, in seconds, for a frame answer; 0 for a clip one.
    double frame_rate = 0.0;
    double frame_offset = 0.0;
    std::vector<float> probabilities;

    // Row `row`'s probability of class `label`.
    float at(uint32_t row, uint32_t label) const { return probabilities[row * labels.size() + label]; }
    // Where frame `row` starts, in seconds from the start of the clip.
    double row_start(uint32_t row) const {
        return frame_rate > 0.0 ? frame_offset + static_cast<double>(row) / frame_rate : 0.0;
    }
};

// Runs `model`'s driver once over `samples` (mono, at the file's own `loom.sample_rate` -- resampling
// is the host's) and shapes the answer by its contract. Throws for a file whose contract is not an
// audio `class` output at `frame` or `clip` granularity, and for an answer that does not divide into
// rows of its declared labels.
ClassProbabilities classify(LoomLuaBridge& bridge, const GgufModel& model, const std::vector<float>& samples);

// Runs `model`'s driver once over `samples` and returns its embedding. Throws for a file whose
// contract is not an audio `embeddings` output at `clip` granularity.
std::vector<float> embed(LoomLuaBridge& bridge, const GgufModel& model, const std::vector<float>& samples);

} // namespace audio
} // namespace loom
