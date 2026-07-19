#include "loom/core/relative_position.h"

#include <algorithm>

namespace loom {

std::vector<float> pad_crop_relative_embeddings(const std::vector<float>& raw, int64_t window_size,
                                                 int64_t k_channels, int64_t length) {
    const int64_t table_len = 2 * window_size + 1;
    const int64_t pad_length = std::max<int64_t>(length - (window_size + 1), 0);
    const int64_t padded_len = table_len + 2 * pad_length;

    std::vector<float> padded(static_cast<size_t>(padded_len * k_channels), 0.0f);
    for (int64_t row = 0; row < table_len; ++row) {
        std::copy(raw.begin() + row * k_channels, raw.begin() + (row + 1) * k_channels,
                  padded.begin() + (row + pad_length) * k_channels);
    }

    const int64_t start = std::max<int64_t>((window_size + 1) - length, 0);
    const int64_t out_len = 2 * length - 1;
    std::vector<float> out(static_cast<size_t>(out_len * k_channels));
    std::copy(padded.begin() + start * k_channels, padded.begin() + (start + out_len) * k_channels, out.begin());
    return out;
}

} // namespace loom
