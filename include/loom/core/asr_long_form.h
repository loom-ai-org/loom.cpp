#pragma once

// Long-form decoding for a DYNAMIC-LENGTH ASR model that was trained on clips up to some ceiling: the
// audio is cut into overlapping windows, each window is decoded on its own, and the token sequences are
// stitched back together where the windows overlap.
//
// WHY THIS IS NOT THE WHISPER LOOP IN `transcribe.cpp`. Whisper's graph is built at one clip length, so
// it has to be windowed, and it seeks on the timestamps it emits -- the next window starts where the
// model closed its last segment. A model like Canary takes any length and emits no timestamps, so there
// is nothing to seek on; what it has instead is a training ceiling (40 s for canary-1b-v2), past which
// one decode degrades. Measured on 79 s of LibriSpeech: one pass returned 116 of 181 words, WER 0.43.
//
// WHAT IS PORTED, AND FROM WHERE. This is NeMo's own long-form scheme for its multitask (AED) models,
// reproduced decision for decision so that a loom transcript of a long file is NeMo's transcript of it:
//
//   * `plan_windows` -- `PromptedAudioToTextLhotseDataset._find_optimal_chunk_size` and `_chunk_waveform`
//     (nemo/collections/asr/data/audio_to_text_lhotse_prompted.py): a window length between a minimum
//     and the ceiling, chosen so the LAST window is as long as possible, stepped by length - overlap;
//   * `lcs_merge` -- `longest_common_subsequence_merge` plus the `parallel_chunking=True` branch of
//     `lcs_alignment_merge_buffer` (nemo/collections/asr/parts/utils/streaming_utils.py);
//   * `merge_windows` -- `merge_parallel_chunks` (nemo/collections/asr/parts/utils/chunking_utils.py):
//     only the head of each new window takes part in the alignment, and the rest is appended.
//
// None of the NUMBERS live here. The window bounds, the overlap and the two merge widths are declared
// by the file (`loom.asr.window_*`, `loom.asr.merge_*`), because each is derived from the checkpoint's
// own config at export (the merge widths come from the encoder's frame rate) and a different model
// trained with a different ceiling must not inherit Canary's.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace loom {
namespace audio {

// What a file declares about its long-form decode. All zero means "one pass, whatever the length",
// which is every dynamic-length ASR export before this existed.
struct LongFormPolicy {
    // The training ceiling: audio no longer than this is decoded in one pass, exactly as before.
    uint32_t max_window_samples = 0;
    // The shortest window the length search may choose, and the granularity it searches in.
    uint32_t min_window_samples = 0;
    uint32_t window_search_step_samples = 0;
    // How much consecutive windows share.
    uint32_t overlap_samples = 0;
    // How many tokens at the end of what has been merged so far are searched for the alignment, and
    // how many at the head of each new window take part in it.
    uint32_t merge_search_tokens = 0;
    uint32_t merge_head_tokens = 0;

    bool declared() const {
        return max_window_samples > 0 && min_window_samples > 0 && window_search_step_samples > 0 &&
               overlap_samples > 0 && overlap_samples < min_window_samples;
    }
};

// One window: where it starts in the caller's audio, how many of its samples are real, and the length
// it is handed to the model at (`padded`, zero-filled past `real`). Every window but possibly the last
// has `real == padded`.
struct Window {
    size_t start = 0;
    size_t real = 0;
    size_t padded = 0;
};

// The windows NeMo would cut `total` samples into. One window covering everything when the audio is
// shorter than the ceiling, or when the policy is not declared.
std::vector<Window> plan_windows(size_t total, const LongFormPolicy& policy);

// Merges `data` (the head of a new window) onto `buffer` (everything merged so far) at their longest
// overlap, as NeMo's parallel-chunking merge does: the buffer is kept up to the end of the aligned run
// and the data continues after it. With no alignment at all, the two are simply concatenated.
std::vector<int32_t> lcs_merge(const std::vector<int32_t>& buffer, const std::vector<int32_t>& data,
                               uint32_t search_tokens);

// The windows' token sequences, in order, stitched into one.
std::vector<int32_t> merge_windows(const std::vector<std::vector<int32_t>>& windows,
                                   const LongFormPolicy& policy);

} // namespace audio
} // namespace loom
