#include "loom/core/ctc_decode.h"

#include <cstddef>

namespace loom {

std::vector<int32_t> ctc_greedy_decode(const float* logits, int64_t n_frames, int64_t n_classes, int32_t blank_id) {
    std::vector<int32_t> raw(static_cast<size_t>(n_frames));
    for (int64_t f = 0; f < n_frames; ++f) {
        const float* frame = logits + f * n_classes;
        int32_t best_id = 0;
        float best_val = frame[0];
        for (int64_t c = 1; c < n_classes; ++c) {
            if (frame[c] > best_val) {
                best_val = frame[c];
                best_id = static_cast<int32_t>(c);
            }
        }
        raw[static_cast<size_t>(f)] = best_id;
    }

    std::vector<int32_t> collapsed;
    int32_t prev = blank_id; // seed with blank so a leading real token isn't accidentally deduped
    for (int32_t id : raw) {
        if (id != prev && id != blank_id) {
            collapsed.push_back(id);
        }
        prev = id;
    }
    return collapsed;
}

} // namespace loom
