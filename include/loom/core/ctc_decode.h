#pragma once

#include <cstdint>
#include <vector>

namespace loom {

// Greedy CTC decode: per-frame argmax over `logits` (row-major, n_classes fastest -- i.e. frame f's
// class scores are logits[f*n_classes .. f*n_classes+n_classes-1], matching a ggml ne=[n_classes,
// n_frames] tensor's flat layout), then collapse consecutive duplicate ids and drop every `blank_id`.
// Pure host-side logic, no ggml graph involvement, same "host logic, not a graph primitive" precedent
// as Generator::argmax.
std::vector<int32_t> ctc_greedy_decode(const float* logits, int64_t n_frames, int64_t n_classes, int32_t blank_id);

} // namespace loom
