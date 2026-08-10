// Verifies loom::predict_durations/loom::expand_by_duration (the host-side duration-based
// frame-expansion shared by Kokoro/StyleTTS2's KModel.forward_with_tokens, and in degenerate form
// VITS's own generate_path) against an independent numpy reference of the real formula (sigmoid+sum,
// round-half-to-even, clamp, repeat_interleave-based one-hot expand). Fully synthetic, procedurally
// generated at ctest time like the other toy fixtures -- not skip-if-missing.

#include "test_util.h"

#include "loom/loom.h"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>

namespace {

std::vector<float> read_f32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(static_cast<size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

std::vector<int32_t> read_i32_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    const std::streamsize bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<int32_t> data(static_cast<size_t>(bytes) / sizeof(int32_t));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

} // namespace

int main() {
    const std::string ref_dir = LOOM_TEST_REF_DIR;

    constexpr size_t kT = 6;
    constexpr size_t kMaxDur = 50;
    constexpr size_t kChannels = 5;

    const std::vector<float> duration_logits_flat = read_f32_binary(ref_dir + "/duration_logits.bin");
    const std::vector<float> seq_flat = read_f32_binary(ref_dir + "/seq.bin");
    const std::vector<int32_t> expected_pred_dur_i32 = read_i32_binary(ref_dir + "/expected_pred_dur.bin");
    const std::vector<float> expected_expanded_flat = read_f32_binary(ref_dir + "/expected_expanded.bin");
    LOOM_CHECK(duration_logits_flat.size() == kT * kMaxDur);
    LOOM_CHECK(seq_flat.size() == kT * kChannels);
    LOOM_CHECK(expected_pred_dur_i32.size() == kT);

    std::vector<std::vector<float>> duration_logits(kT, std::vector<float>(kMaxDur));
    for (size_t t = 0; t < kT; ++t)
        for (size_t k = 0; k < kMaxDur; ++k) duration_logits[t][k] = duration_logits_flat[t * kMaxDur + k];

    std::vector<std::vector<float>> seq(kT, std::vector<float>(kChannels));
    for (size_t t = 0; t < kT; ++t)
        for (size_t c = 0; c < kChannels; ++c) seq[t][c] = seq_flat[t * kChannels + c];

    const std::vector<uint32_t> pred_dur = loom::predict_durations(duration_logits, /*speed=*/1.0f);
    LOOM_CHECK(pred_dur.size() == kT);
    for (size_t t = 0; t < kT; ++t) {
        if (static_cast<int32_t>(pred_dur[t]) != expected_pred_dur_i32[t]) {
            std::fprintf(stderr, "pred_dur[%zu]: actual=%u expected=%d\n", t, pred_dur[t], expected_pred_dur_i32[t]);
        }
        LOOM_CHECK(static_cast<int32_t>(pred_dur[t]) == expected_pred_dur_i32[t]);
    }

    const std::vector<std::vector<float>> expanded = loom::expand_by_duration(seq, pred_dur);
    size_t expected_t_frames = 0;
    for (int32_t d : expected_pred_dur_i32) expected_t_frames += static_cast<size_t>(d);
    LOOM_CHECK(expanded.size() == expected_t_frames);
    LOOM_CHECK(expected_expanded_flat.size() == expected_t_frames * kChannels);

    for (size_t t = 0; t < expected_t_frames; ++t) {
        for (size_t c = 0; c < kChannels; ++c) {
            const float actual = expanded[t][c];
            const float expected = expected_expanded_flat[t * kChannels + c];
            LOOM_CHECK(actual == expected);
        }
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
