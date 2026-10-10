// MusicGen (family 14): the exported GGUF's own driver, end to end, against `transformers`.
//
// The prompt goes through the file's own T5 SentencePiece vocabulary, then the driver runs the T5
// encoder, the cross-attention K/V and the delayed four-codebook decode loop. Both arms are greedy,
// so they are graded code for code against `MusicgenForConditionalGeneration.generate(do_sample=
// False)` at max_new_tokens = 35, which is 32 frames here (this driver counts FRAMES):
//
//   * guidance off -- the decoder alone, one stream;
//   * guidance 3.0, the checkpoint's own scale -- the unconditional stream with its private KV cache
//     and its ZERO cross-attention K/V (`generate()` zeroes the encoder output and its mask).
//
// The guided reference differs from the unguided one in most cells, so the second arm cannot pass by
// ignoring guidance. The codes were produced by loom-exporter's MusicGen reference script; the
// teacher-forced logits behind them match this engine to 7e-5 on values of ~23 (f32 rounding).
//
// Fixture: LOOM_MUSICGEN_GGUF, or `loom-export <musicgen-small> -o musicgen-small.gguf`.

#include "test_util.h"
#include "fixtures.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kSkipReturnCode = 77;

const char* kPrompt = "80s pop track with bassy drums and synth";
// `T5Tokenizer`'s ids for kPrompt, `</s>` (1) appended.
const std::vector<double> kPromptIds = {2775, 7, 2783, 1463, 28, 7981, 63, 5253, 7, 11, 13353, 1};

constexpr int kFrames = 32;
constexpr int kCodebooks = 4;

const int kUnguidedCodes[] = {
    1062, 2032, 1898, 1628, 993, 2032, 1898, 1628, 993, 2032, 1898, 1628, 993, 2032, 1898, 1628,
    993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537,
    993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537,
    993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537,
    993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537,
    993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537,
    993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537,
    993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537, 993, 1938, 2009, 1537,
};

const int kGuidedCodes[] = {
    1062, 574, 1898, 1759, 993, 2027, 1895, 346, 1210, 1210, 1895, 346, 1210, 1473, 1898, 346,
    1210, 1210, 1869, 1850, 237, 1670, 1898, 1394, 508, 826, 1898, 616, 508, 1931, 1895, 1394,
    778, 1938, 2019, 1650, 609, 2044, 1895, 1770, 609, 2044, 1895, 1770, 609, 1931, 2019, 1770,
    609, 1931, 2019, 1770, 83, 2044, 1895, 1770, 83, 2044, 1895, 1770, 83, 1931, 2019, 1770,
    83, 1931, 2019, 1770, 83, 1931, 2019, 1770, 83, 1931, 2019, 1770, 83, 1931, 2019, 1770,
    83, 1931, 2019, 1770, 83, 1931, 2019, 1770, 83, 1931, 2019, 1770, 83, 1931, 2019, 1770,
    83, 1931, 2019, 1770, 83, 1931, 2019, 1770, 83, 1931, 2019, 1770, 83, 1931, 2019, 1770,
    83, 1931, 2019, 1770, 83, 1931, 2019, 1770, 83, 1931, 2019, 1770, 83, 1931, 2019, 1758,
};

size_t count_mismatches(const std::vector<double>& got, const int* want, const char* arm) {
    size_t mismatches = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (static_cast<int>(got[i]) != want[i]) {
            if (mismatches < 8) {
                std::fprintf(stderr, "  %s frame %zu codebook %zu: expected %d, got %d\n", arm,
                             i / kCodebooks, i % kCodebooks, want[i], static_cast<int>(got[i]));
            }
            ++mismatches;
        }
    }
    return mismatches;
}

} // namespace

int main() {
    const char* env = loom_test::fixture_env("LOOM_MUSICGEN_GGUF");
    const std::string gguf_path = env != nullptr ? env : "musicgen-small.gguf";
    if (!loom_test::path_exists(gguf_path)) {
        std::fprintf(stderr, "skipping: '%s' not found (set LOOM_MUSICGEN_GGUF, or run "
                              "`loom-export <musicgen-small> -o musicgen-small.gguf`)\n",
                     gguf_path.c_str());
        return kSkipReturnCode;
    }

    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(gguf_path, backend.get());

    // The file's own vocabulary reproduces the reference's ids, `</s>` included.
    auto vocab = loom::Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    const auto ids = vocab->encode(kPrompt);
    LOOM_CHECK(ids.size() == kPromptIds.size());
    for (size_t i = 0; i < ids.size() && i < kPromptIds.size(); ++i) {
        LOOM_CHECK(ids[i] == static_cast<int32_t>(kPromptIds[i]));
    }

    loom::Session session(*model, backend.get());
    const size_t expected = static_cast<size_t>(kFrames) * kCodebooks;

    const auto unguided = std::get<std::vector<double>>(session.bridge().call(
        "infer", {{"tokens", kPromptIds}, {"max_new_tokens", static_cast<double>(kFrames)},
                  {"temperature", 0.0}, {"guidance_scale", 1.0}}));
    LOOM_CHECK(unguided.size() == expected);
    const size_t unguided_bad = count_mismatches(unguided, kUnguidedCodes, "unguided");
    std::fprintf(stderr, "musicgen unguided: %zu/%zu codes differ from transformers\n",
                 unguided_bad, expected);
    LOOM_CHECK(unguided_bad == 0);

    const auto guided = std::get<std::vector<double>>(session.bridge().call(
        "infer", {{"tokens", kPromptIds}, {"max_new_tokens", static_cast<double>(kFrames)},
                  {"temperature", 0.0}, {"guidance_scale", 3.0}}));
    LOOM_CHECK(guided.size() == expected);
    const size_t guided_bad = count_mismatches(guided, kGuidedCodes, "guided");
    size_t guidance_moved = 0;
    for (size_t i = 0; i < expected; ++i) guidance_moved += kGuidedCodes[i] != kUnguidedCodes[i];
    std::fprintf(stderr, "musicgen guided: %zu/%zu codes differ from transformers; the reference "
                          "itself differs from the unguided one in %zu\n",
                 guided_bad, expected, guidance_moved);
    LOOM_CHECK(guided_bad == 0);
    LOOM_CHECK(guidance_moved > 0);

    LOOM_TEST_REPORT_AND_RETURN();
}
