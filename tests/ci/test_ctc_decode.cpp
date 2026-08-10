// Hand-computed isolated unit test for ctc_greedy_decode (per-frame argmax + collapse-repeats +
// drop-blank), pure host-side logic with no ggml graph involved.

#include "test_util.h"

#include "loom/loom.h"

#include <cstdint>
#include <vector>

int main() {
    // n_classes=4, blank_id=3. Frame-by-frame argmax (by construction): 1,1,3,2,2,0,0,3.
    // Collapsing consecutive duplicates then dropping blank (id 3) gives: 1, 2, 0.
    const std::vector<float> logits = {
        0, 5, 0, 0,  // frame 0: argmax=1
        0, 4, 1, 0,  // frame 1: argmax=1 (dup of prev)
        0, 0, 0, 9,  // frame 2: argmax=3 (blank)
        0, 0, 7, 1,  // frame 3: argmax=2
        1, 0, 6, 0,  // frame 4: argmax=2 (dup of prev)
        8, 0, 0, 1,  // frame 5: argmax=0
        7, 0, 0, 2,  // frame 6: argmax=0 (dup of prev)
        0, 0, 0, 9,  // frame 7: argmax=3 (blank)
    };
    const auto result = loom::ctc_greedy_decode(logits.data(), /*n_frames=*/8, /*n_classes=*/4, /*blank_id=*/3);
    LOOM_CHECK((result == std::vector<int32_t>{1, 2, 0}));

    // Leading blank shouldn't suppress a real token immediately after it.
    const std::vector<float> logits2 = {
        0, 0, 0, 9,  // blank
        0, 5, 0, 0,  // 1
    };
    const auto result2 = loom::ctc_greedy_decode(logits2.data(), 2, 4, 3);
    LOOM_CHECK((result2 == std::vector<int32_t>{1}));

    // All-blank input collapses to empty.
    const std::vector<float> logits3 = {0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 9};
    const auto result3 = loom::ctc_greedy_decode(logits3.data(), 3, 4, 3);
    LOOM_CHECK(result3.empty());

    LOOM_TEST_REPORT_AND_RETURN();
}
