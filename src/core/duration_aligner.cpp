#include "loom/core/duration_aligner.h"

#include <algorithm>
#include <cmath>

namespace loom {

std::vector<uint32_t> predict_durations(const std::vector<std::vector<float>>& duration_logits,
                                         float speed) {
    std::vector<uint32_t> out(duration_logits.size());
    for (size_t t = 0; t < duration_logits.size(); ++t) {
        double sum = 0.0;
        for (float v : duration_logits[t]) sum += 1.0 / (1.0 + std::exp(-static_cast<double>(v)));
        const double duration = sum / static_cast<double>(speed);
        // std::nearbyint honors the ambient FP rounding mode, which defaults to round-half-to-even --
        // the same convention torch.round/np.round use (unlike std::lround's round-half-away-from-zero).
        const double rounded = std::nearbyint(duration);
        out[t] = static_cast<uint32_t>(std::max(rounded, 1.0));
    }
    return out;
}

std::vector<std::vector<float>> expand_by_duration(const std::vector<std::vector<float>>& seq,
                                                    const std::vector<uint32_t>& durations) {
    size_t total = 0;
    for (uint32_t d : durations) total += d;
    std::vector<std::vector<float>> out;
    out.reserve(total);
    for (size_t t = 0; t < seq.size(); ++t) {
        for (uint32_t r = 0; r < durations[t]; ++r) out.push_back(seq[t]);
    }
    return out;
}

} // namespace loom
