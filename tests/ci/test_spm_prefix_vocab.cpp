// Tests the SentencePiece-style BPE shape's dummy prefix (`tokenizer.ggml.add_space_prefix`, HF's
// `Prepend("▁")` normalizer + `Strip(" ", 1, 0)` decoder) against the hand-traced fixture from
// tests/fixtures/make_spm_prefix_vocab_gguf.py. The real vocabulary (Moonshine Streaming's) is diffed
// against `transformers`' own tokenizer in the exporter's Moonshine check.

#include "test_util.h"

#include "loom/loom.h"
#include "loom/core/bpe_vocab.h"

#include "cpu_backend.h"

namespace {
constexpr int32_t kSpace = 259, kA = 260, kAB = 263, kSpaceAB = 264;
int32_t byte(unsigned char b) { return 3 + b; }
}  // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    const std::string dir = std::string(LOOM_TEST_FIXTURE_DIR);
    auto model = loom::GgufModel::load(dir + "/spm_prefix_vocab_test.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);

    // The prefix makes the first word "▁ab", which merges to one piece; without it the merges
    // would stop at "ab".
    {
        const auto ids = vocab->encode("ab");
        const std::vector<int32_t> expected = {kSpaceAB};
        LOOM_CHECK(ids == expected);
        LOOM_CHECK(vocab->decode(ids) == "ab");
    }
    {
        const auto ids = vocab->encode("ab ab");
        const std::vector<int32_t> expected = {kSpaceAB, kSpaceAB};
        LOOM_CHECK(ids == expected);
        LOOM_CHECK(vocab->decode(ids) == "ab ab");
    }
    // A leading space is a second U+2581 before the prefixed word, and the decoder strips ONE space.
    {
        const auto ids = vocab->encode(" ab");
        const std::vector<int32_t> expected = {kSpace, kSpaceAB};
        LOOM_CHECK(ids == expected);
        LOOM_CHECK(vocab->decode(ids) == " ab");
    }
    // A character the vocabulary lacks falls back to its UTF-8 bytes; the prefix stays its own piece.
    {
        const auto ids = vocab->encode("\xc3\xa9");
        const std::vector<int32_t> expected = {kSpace, byte(0xc3), byte(0xa9)};
        LOOM_CHECK(ids == expected);
        LOOM_CHECK(vocab->decode(ids) == "\xc3\xa9");
    }
    // Decoding ids that do not start with the prefix strips nothing.
    {
        const std::vector<int32_t> ids = {kAB, kA};
        LOOM_CHECK(vocab->decode(ids) == "aba");
    }

    // The key on a byte-level shape is refused: that shape spells a space as a byte, and a prefix it
    // does not implement would otherwise be dropped silently.
    auto refused = loom::GgufModel::load(dir + "/bpe_prefix_refused_test.gguf", backend.get());
    LOOM_CHECK(refused != nullptr);
    bool threw = false;
    try {
        loom::BpeVocab::load(*refused);
    } catch (const loom::LoadError&) {
        threw = true;
    }
    LOOM_CHECK(threw);
    return 0;
}
