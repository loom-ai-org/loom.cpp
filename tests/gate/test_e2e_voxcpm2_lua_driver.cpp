// Validates the VoxCPM2 export (loom-exporter's `voxcpm2_export.py`) end to end against the reference
// implementation: text in through the file's own vocabulary, the two cached LMs, the local DiT's
// guided Euler solve per patch, the local encoder feeding each patch back, the stop head, the AudioVAE,
// and a 48 kHz waveform out -- one GGUF, five topologies, one driver.
//
// **The oracle is the real VoxCPM2** (`scripts/voxcpm2_reference.py`), with its one `[4, 64]` draw per
// patch pinned and handed in (`noise`): the same seed is not the same noise. Three arms:
//
//   * **Teacher-forced, patches and waveform**, the exact ones. Each patch is re-encoded into the next step's input and is the
//     next patch's DiT condition, so f32 rounding COMPOUNDS along the loop and two correct
//     implementations drift apart (Retro-055). Fed the reference's patches (`teacher_patches`), every
//     step is compared on its own.
//   * **Free-running**, the one that checks the LOOP: the same patch count (the stop head fired at the
//     same step) and a waveform within a drift bound.
//
// The measured numbers and the reference's own f32-vs-f64 spread are in the bounds' comments.
//
// Fixtures:
//   LOOM_VOXCPM2_GGUF     voxcpm2.gguf  -- `loom-export ~/Dev/models/voxcpm2 -o voxcpm2.gguf`
//   LOOM_VOXCPM2_REF_DIR  voxcpm2_ref/  -- tokens.npy, noise.npy, patches.npy, wave.npy and meta.json
//                                          from scripts/voxcpm2_reference.py

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Measured on the reference's default sentence (35 patches, 5.6 s), x86-64, 2026-09-24:
//   teacher-forced patches   max 1.0e-05  rmse 1.2e-06   (sabotage, guidance 2.2 for 2.0: 4.57 / 0.225)
//   teacher-forced waveform  max 9.0e-06  rmse 4.3e-07
//   free-running waveform    max 2.5e-04  rmse 7.1e-06   (same patch count: the stop head's margin
//                                                          there was 4.4)
// Each bound is ~20x its measurement, and the sabotage arm is four orders of magnitude over the patch
// bounds. The margin is for other ISAs' accumulation order.
//
// **The reference's own f32-vs-f64 spread is NOT a floor here**, unlike Pocket-TTS's: it is 1.7e-02 on
// the FIRST patch and O(1) on the waveform by the end. The prefill alone moves the DiT's conditioning
// by ~1e-4 relative at f64, and from the second patch the FSQ bottleneck (`round(tanh(x) * 9)`) turns
// any difference that crosses a rounding boundary into a whole level. Engine and reference agree far
// better than that because both are f32 with similar summation. It also means the FREE-RUNNING arm
// can fail on another ISA by a large number (one flipped level diverges the rest of the clip) without
// anything being wrong; the teacher-forced arms are the ones to trust there.
constexpr double VOXCPM2_PATCH_MAX_ABS = 2e-4;
constexpr double VOXCPM2_PATCH_RMSE = 2e-5;
constexpr double VOXCPM2_TEACHER_MAX_ABS = 2e-4;
constexpr double VOXCPM2_TEACHER_RMSE = 1e-5;
constexpr double VOXCPM2_FREE_RMSE = 1e-4;

// The one string this test needs out of `meta.json`: the text as the reference was given it.
std::string meta_text(const std::string& json) {
    const std::string needle = "\"text\": \"";
    const size_t at = json.find(needle);
    if (at == std::string::npos) throw std::runtime_error("voxcpm2_ref/meta.json has no \"text\"");
    const size_t start = at + needle.size();
    return json.substr(start, json.find('"', start) - start);
}

std::vector<double> as_doubles(const std::vector<float>& v) { return {v.begin(), v.end()}; }

struct Diff {
    double max_abs = 0.0, rmse = 0.0, peak = 0.0;
};

Diff compare(const std::vector<double>& got, const std::vector<float>& want) {
    Diff d;
    double sum_sq = 0.0;
    const size_t n = std::min(got.size(), want.size());
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(got[i] - want[i]);
        d.max_abs = std::max(d.max_abs, e);
        sum_sq += e * e;
        d.peak = std::max(d.peak, std::fabs(got[i]));
    }
    d.rmse = std::sqrt(sum_sq / static_cast<double>(std::max<size_t>(n, 1)));
    return d;
}

} // namespace

int main() {
    const char* gguf_env = loom_test::fixture_env("LOOM_VOXCPM2_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_VOXCPM2_REF_DIR");
    if (gguf_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_VOXCPM2_GGUF and LOOM_VOXCPM2_REF_DIR "
                              "(see scripts/voxcpm2_reference.py)\n");
        return 77;
    }
    const std::string ref_dir = ref_env;
    std::ifstream meta_file(ref_dir + "/meta.json");
    LOOM_CHECK(meta_file.good());
    std::stringstream meta_buf;
    meta_buf << meta_file.rdbuf();
    const std::string text = meta_text(meta_buf.str());

    std::vector<int64_t> shape;
    const auto ref_ids = loom_test::read_npy_f32(ref_dir + "/tokens.npy", shape);
    const auto noise = loom_test::read_npy_f32(ref_dir + "/noise.npy", shape);
    const auto patches = loom_test::read_npy_f32(ref_dir + "/patches.npy", shape);
    const auto ref_wave = loom_test::read_npy_f32(ref_dir + "/wave.npy", shape);
    LOOM_CHECK(!ref_ids.empty() && !noise.empty() && !patches.empty() && !ref_wave.empty());

    loom::Device device = loom::Device::open("cpu");
    auto model = loom::GgufModel::load(gguf_env, device.backends().primary);
    LOOM_CHECK(model != nullptr);

    // **The text door first.** The file's own vocabulary must produce the reference's ids, or the
    // waveform comparison below would be grading two different sentences.
    auto vocab = loom::VoxCpmVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode(text);
    LOOM_CHECK(ids.size() == ref_ids.size());
    for (size_t i = 0; i < std::min(ids.size(), ref_ids.size()); ++i) {
        LOOM_CHECK(static_cast<float>(ids[i]) == ref_ids[i]);
    }
    const std::vector<double> tokens(ids.begin(), ids.end());

    // --- 0. Teacher-forced PATCHES: the loop's own state, step by step. ---
    // The sharpest arm: `return_patches` hands back the latents instead of decoding them, so one
    // patch's error is not smeared over the decoder's receptive field. It is also the one that found
    // Retro-057 (two cached stacks in one cache) -- a build whose speech transcribed exactly.
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"noise", as_doubles(noise)}, {"teacher_patches", as_doubles(patches)},
            {"return_patches", 1.0}}));
        std::fprintf(stderr, "teacher-forced patches: loom=%zu reference=%zu values\n", got.size(), patches.size());
        LOOM_CHECK(got.size() == patches.size());
        const Diff d = compare(got, patches);
        std::fprintf(stderr, "teacher-forced patches max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        LOOM_CHECK(d.max_abs < VOXCPM2_PATCH_MAX_ABS);
        LOOM_CHECK(d.rmse < VOXCPM2_PATCH_RMSE);
    }

    // --- 1. Teacher-forced: every step on its own. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"noise", as_doubles(noise)}, {"teacher_patches", as_doubles(patches)}}));
        // The LENGTH is the patch count times 4 x 1920, so a disagreement is a stop decision that went
        // the other way -- checked first, because it makes every per-sample number meaningless.
        std::fprintf(stderr, "teacher-forced samples: loom=%zu reference=%zu\n", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "teacher-forced max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        // A waveform collapsed toward silence can have a small difference for the wrong reason
        // (Retro-006).
        LOOM_CHECK(d.peak > 0.05);
        LOOM_CHECK(d.max_abs < VOXCPM2_TEACHER_MAX_ABS);
        LOOM_CHECK(d.rmse < VOXCPM2_TEACHER_RMSE);
    }

    // --- 2. Free-running: the loop itself, from the same draws. ---
    {
        loom::Session session(*model, device.backends());
        const auto got = std::get<std::vector<double>>(session.bridge().call("infer", {
            {"tokens", tokens}, {"noise", as_doubles(noise)}}));
        std::fprintf(stderr, "free-running samples: loom=%zu reference=%zu\n", got.size(), ref_wave.size());
        LOOM_CHECK(got.size() == ref_wave.size());
        const Diff d = compare(got, ref_wave);
        std::fprintf(stderr, "free-running max_abs_diff=%g rmse=%g peak=%g\n", d.max_abs, d.rmse, d.peak);
        LOOM_CHECK(d.rmse < VOXCPM2_FREE_RMSE);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
