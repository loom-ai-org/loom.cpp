#pragma once

#include <cstdint>
#include <vector>

namespace loom {

// Direct C++ port of piper's `attentions.py::MultiHeadAttention._get_relative_embeddings` (real source:
// see tools/convert_piper_vits/vits_common.py's own `get_relative_embeddings`, cross-checked against the
// real PyTorch method for lengths spanning both branches -- see BACKLOG.md). Converts a FIXED
// `(2*window_size+1) * k_channels`-element learned relative-position table (row-major, k_channels
// fastest) into the `(2*length-1) * k_channels`-element table `REL_POS_ATTENTION_SHAW` expects for a
// given per-call sequence length -- real phoneme counts routinely exceed `window_size+1`, so this needs
// padding OR cropping depending on `length` vs `window_size`. Shared between VITS's own TextEncoder
// (originally `vits_driver.cpp`-local) and any other model using the same Shaw et al. lookup-table
// mechanism (e.g. SupertonicTTS's `MultiHeadRelativeAttention`).
std::vector<float> pad_crop_relative_embeddings(const std::vector<float>& raw, int64_t window_size,
                                                 int64_t k_channels, int64_t length);

} // namespace loom
