// Family 10 and family 11 chained: text -> Dia -> nine delayed code streams -> realign -> DAC ->
// waveform, against `transformers` running the identical pipeline.
//
// **This is the point of family 10, and it is the first check here that crosses two GGUFs.** Dia is
// silent on its own -- its output kind is `audio_codes` (ADR-020) and what turns codes into audio is a
// second file. ADR-022 records why it stays a second file rather than being merged in, and the short
// version is that the codec is shared: Dia, Parler, CSM and Orpheus all decode through DAC, so merging
// would ship the same 217 MB inside every one of them and make the codes unreachable. What that
// decision costs is precisely the join this test covers -- with two files, nothing in either one
// asserts that they fit together.
//
// **The codes are checked before the waveform, and that ordering is the diagnostic.** The composition
// has three ways to fail and a waveform comparison alone cannot tell them apart: Dia emits the wrong
// codes, the realignment hands DAC the wrong layout, or DAC decodes the right codes into the wrong
// audio. Checking the codes first means a waveform mismatch can only be the last of the three.
//
// **Two clip lengths, and that is not thoroughness.** It is family 11's own lesson from one family
// over: DAC's first working export returned one frame's worth of audio for every input and nothing
// raised (`test_codec_output_length_follows_the_input`). A single clip length cannot see that,
// because any constant length is consistent with itself. Two lengths whose sample counts differ by
// exactly `hop * (frames_b - frames_a)` is what makes the length follow its input.
//
// Greedy, classifier-free guidance OFF, matching the driver -- see `test_e2e_dia_mil_export.cpp` and
// `scripts/dia_dac_reference.py` for why that is load-bearing and what has to change with it.
//
// Fixtures (all three, or this skips):
//   LOOM_DIA_MIL_GGUF     dia_mil.gguf     -- `loom-export <dia-1.6b> -o dia_mil.gguf`
//   LOOM_DAC_44KHZ_GGUF   dac_44khz.gguf   -- `loom-export <dac-44khz> -o dac_44khz.gguf`
//   LOOM_DIA_DAC_REF_DIR  dia_dac_ref/     -- prompt_ids.npy, codes_<N>f.npy, wav_<N>f.npy from
//                                             `scripts/dia_dac_reference.py`

#include "test_util.h"
#include "fixtures.h"
#include "npy_fixture.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

// The two clip lengths, in AUDIO frames. `scripts/dia_dac_reference.py`'s own default, and the file
// names encode it, so changing one means regenerating the fixtures.
constexpr int kFrames[] = {16, 32};
constexpr int kChannels = 9;

// A generation's worth of codes, frame-major and flat -- the array the Dia driver returns and the one
// the DAC driver takes, which is the whole of the calling convention between the two files.
std::vector<double> generate_codes(const std::string& gguf_path, ggml_backend_t backend,
                                   const std::vector<double>& prompt_ids, int frames) {
    auto model = loom::GgufModel::load(gguf_path, backend);
    // `Session` rather than a registration loop: this file declares five topologies, two of them
    // second streams, and one of those asks for its own KV cache (ADR-023). Getting that wrong is a
    // wrong answer rather than an error, and `Session` is where the rule lives.
    loom::Session session(*model, backend);
    loom::LoomLuaBridge& bridge = session.bridge();
    // `max_new_tokens` counts AUDIO FRAMES here, not decoder rows -- the driver's contract, and the
    // reference script converts to rows on its own side.
    //
    // **`temperature` and `guidance_scale` are passed, and passing them is the point.** This file
    // declares its own defaults -- the checkpoint's, which are `do_sample` at 1.8/50/0.9 with
    // guidance at 3.0 -- so `infer` with neither named draws a different generation every call. What
    // this test compares is the composition, and the cheapest real generation is the right input for
    // that; the guided decode has its own oracle in `test_e2e_dia_mil_export.cpp`, on the codes,
    // where a difference is attributable.
    loom::LoomLuaBridge::Value result = bridge.call(
        "infer", {{"tokens", prompt_ids}, {"max_new_tokens", static_cast<double>(frames)},
                  {"temperature", 0.0}, {"guidance_scale", 1.0}});
    return std::get<std::vector<double>>(result);
}

} // namespace

int main() {
    const char* dia_env = loom_test::fixture_env("LOOM_DIA_MIL_GGUF");
    const char* dac_env = loom_test::fixture_env("LOOM_DAC_44KHZ_GGUF");
    const char* ref_env = loom_test::fixture_env("LOOM_DIA_DAC_REF_DIR");
    if (dia_env == nullptr || dac_env == nullptr || ref_env == nullptr) {
        std::fprintf(stderr, "skipping: needs LOOM_DIA_MIL_GGUF, LOOM_DAC_44KHZ_GGUF and "
                              "LOOM_DIA_DAC_REF_DIR (see scripts/dia_dac_reference.py)\n");
        return kSkipReturnCode;
    }
    const std::string ref_dir = ref_env;
    if (!loom_test::path_exists(ref_dir + "/prompt_ids.npy")) {
        std::fprintf(stderr, "skipping: %s/prompt_ids.npy not found\n", ref_dir.c_str());
        return kSkipReturnCode;
    }

    std::vector<int64_t> shape;
    const std::vector<float> prompt_f32 = loom_test::read_npy_f32(ref_dir + "/prompt_ids.npy", shape);
    const std::vector<double> prompt_ids(prompt_f32.begin(), prompt_f32.end());

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    // Dia first and DAC second, in separate scopes rather than both resident: this file is 6.4 GB of
    // F32 weights and the codec is 217 MB, and there is no reason for a gate test to hold both. It is
    // also how a host would run it -- codes are a serialisable intermediate, which is the property
    // ADR-022 keeps.
    std::vector<std::vector<double>> generated;
    for (const int frames : kFrames) {
        generated.push_back(generate_codes(dia_env, backend.get(), prompt_ids, frames));
    }

    for (size_t i = 0; i < std::size(kFrames); ++i) {
        const int frames = kFrames[i];
        const std::string tag = std::to_string(frames) + "f";
        std::vector<int64_t> codes_shape;
        const std::vector<float> ref_codes =
            loom_test::read_npy_f32(ref_dir + "/codes_" + tag + ".npy", codes_shape);
        LOOM_CHECK(codes_shape.size() == 2 && codes_shape[0] == frames && codes_shape[1] == kChannels);

        const std::vector<double>& got = generated[i];
        std::fprintf(stderr, "dia %d frames: %zu codes, reference %zu\n",
                     frames, got.size(), ref_codes.size());
        LOOM_CHECK(got.size() == ref_codes.size());
        size_t mismatches = 0;
        for (size_t j = 0; j < got.size(); ++j) {
            if (static_cast<int32_t>(got[j]) != static_cast<int32_t>(ref_codes[j])) {
                if (mismatches < 8) {
                    std::fprintf(stderr, "  frame %zu channel %zu: expected %d, got %d\n",
                                 j / kChannels, j % kChannels,
                                 static_cast<int32_t>(ref_codes[j]), static_cast<int32_t>(got[j]));
                }
                ++mismatches;
            }
        }
        LOOM_CHECK(mismatches == 0);
    }

    // The codec half. One model, both clips -- which is also the reason a host holds the codec open
    // across utterances rather than reloading it per sentence.
    auto dac = loom::GgufModel::load(dac_env, backend.get());
    // The two files have to AGREE on the width of a frame, and this is the key that says so on both
    // sides: Dia writes `loom.codec.n_codebooks` from its own channel count and DAC from its
    // quantizer count. A host chaining them can check it, so this test checks that a host could.
    LOOM_CHECK(static_cast<int>(dac->hparam_u32("codec.n_codebooks")) == kChannels);
    const int sample_rate = static_cast<int>(dac->hparam_u32("sample_rate"));
    const float frame_rate = dac->hparam_f32("codec.frame_rate");
    const int hop = static_cast<int>(std::lround(sample_rate / frame_rate));

    loom::Session dac_session(*dac, backend.get());
    loom::LoomLuaBridge& bridge = dac_session.bridge();

    std::vector<size_t> sample_counts;
    for (size_t i = 0; i < std::size(kFrames); ++i) {
        const int frames = kFrames[i];
        const std::string tag = std::to_string(frames) + "f";
        std::vector<int64_t> wav_shape;
        const std::vector<float> ref_wav =
            loom_test::read_npy_f32(ref_dir + "/wav_" + tag + ".npy", wav_shape);

        loom::LoomLuaBridge::Value result = bridge.call("infer", {{"codes", generated[i]}});
        const auto& wav = std::get<std::vector<double>>(result);
        sample_counts.push_back(wav.size());

        std::fprintf(stderr, "dac %d frames: %zu samples, reference %zu (hop %d)\n",
                     frames, wav.size(), ref_wav.size(), hop);
        LOOM_CHECK(wav.size() == ref_wav.size());
        LOOM_CHECK(wav.size() == static_cast<size_t>(frames) * hop);

        double max_abs_diff = 0.0, peak = 0.0, sum_sq = 0.0;
        for (size_t j = 0; j < wav.size(); ++j) {
            max_abs_diff = std::max(max_abs_diff, std::abs(wav[j] - ref_wav[j]));
            peak = std::max(peak, std::abs(wav[j]));
            sum_sq += wav[j] * wav[j];
        }
        const double rms = std::sqrt(sum_sq / static_cast<double>(wav.size()));
        std::fprintf(stderr, "  max |diff| %.3e, peak %.4f, rms %.4f\n", max_abs_diff, peak, rms);
        // **Relative to the signal, not absolute, because this clip is quiet.** The captured sentence
        // peaks at 0.009 -- it is the leading 0.37 s of an utterance -- so an absolute 1e-3, which is
        // the tolerance the VITS vocoder reference uses against a signal two orders of magnitude
        // louder, would here be 11% of full scale and would pass a waveform that was visibly wrong.
        // Observed at HEAD: 5.3e-6 and 5.0e-6, i.e. 6e-4 of peak, so this bound has ~17x of margin
        // while any structural error -- a dropped frame, a transposed layout -- is on the order of
        // the peak itself. The floor keeps it from going below float noise if a quieter capture is
        // ever taken.
        LOOM_CHECK(max_abs_diff < std::max(0.01 * peak, 1e-6));
        // Not silence, and not clipped. Every code matching exactly already implies the signal is the
        // reference's, but a decoder that returned zeros would agree with a zero reference, and
        // Retro-006's lesson is that a number agreeing with a number is not audio being audio.
        LOOM_CHECK(rms > 1e-5);
        LOOM_CHECK(peak <= 1.0);
    }

    // The length follows the input: two clips, and the difference is exactly the hop times the frame
    // difference. This is the assertion the family-11 bug would have failed while every per-clip check
    // above passed.
    LOOM_CHECK(sample_counts.size() == 2);
    LOOM_CHECK(sample_counts[1] - sample_counts[0] ==
               static_cast<size_t>(kFrames[1] - kFrames[0]) * hop);

    LOOM_TEST_REPORT_AND_RETURN();
}
