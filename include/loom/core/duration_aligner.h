#pragma once

#include <cstdint>
#include <vector>

namespace loom {

// Host-side implementation of the duration-based frame-expansion mechanism shared by Kokoro/StyleTTS2
// (`KModel.forward_with_tokens`: `duration = sigmoid(duration_proj(x)).sum(-1)/speed`, `pred_dur =
// round(duration).clamp(min=1)`, `indices = repeat_interleave(arange(T), pred_dur)`, `en =
// d.transpose(-1,-2) @ pred_aln_trg`) and, in degenerate form, VITS's own `generate_path` (already
// inlined directly in vits_driver.cpp -- both real formulas collapse to "replicate token/frame t's own
// feature row `durations[t]` consecutive times" once the per-utterance mask is dropped, since
// `pred_aln_trg`/the attention path are always exact one-hot alignment matrices, never soft alignments,
// at inference time). Kept as free functions (not a class) since there is no state to carry between
// calls, unlike BiLstmStepper/TdtDecoder's own per-step recurrence.
//
// `torch.round` uses round-half-to-even (banker's rounding, the IEEE754/`std::nearbyint` default
// rounding mode) -- NOT round-half-away-from-zero (`std::lround`) -- matched deliberately here even
// though real duration-logit sums are float32 and essentially never land exactly on a tie in practice.

// duration_logits: T rows, each max_dur floats (the real `duration_proj`'s raw pre-sigmoid output).
// Returns T per-token frame counts (each >= 1), matching `torch.round(sigmoid(x).sum(-1)/speed
// ).clamp(min=1)`.
std::vector<uint32_t> predict_durations(const std::vector<std::vector<float>>& duration_logits,
                                         float speed = 1.0f);

// seq: T rows, each `channels` floats. durations: T per-row repeat counts (from predict_durations).
// Returns sum(durations) rows: row t repeated durations[t] consecutive times, in order -- exactly what
// `seq^T @ pred_aln_trg` computes when `pred_aln_trg` is the one-hot alignment matrix built from
// `durations` via `repeat_interleave`, without ever materializing that matrix.
std::vector<std::vector<float>> expand_by_duration(const std::vector<std::vector<float>>& seq,
                                                    const std::vector<uint32_t>& durations);

} // namespace loom
