// Tests loom::WordPieceVocab (src/core/wordpiece_vocab.cpp) against the small synthetic fixture generated
// by tests/fixtures/make_wordpiece_vocab_gguf.py -- exact hand-traced token ids for a fully-covered word,
// a "##"-continuation split, punctuation isolation, [UNK] fallback, accent-stripping, and CLS/SEP
// auto-wrap (this fixture has both add_bos_token and add_sep_token set, so every case below is wrapped).

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/wordpiece_vocab_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);

    auto vocab = loom::WordPieceVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    LOOM_CHECK(vocab->size() == 11);
    LOOM_CHECK(vocab->unk_id() == 1);
    LOOM_CHECK(vocab->cls_id() == 2);
    LOOM_CHECK(vocab->sep_id() == 3);
    LOOM_CHECK(vocab->pad_id() == 0);
    LOOM_CHECK(vocab->mask_id() == 4);

    // "hello world": both whole-word pieces match directly -- CLS/SEP-wrapped.
    {
        const auto ids = vocab->encode("hello world");
        LOOM_CHECK((ids == std::vector<int32_t>{2, 5, 6, 3}));
    }

    // "unhappy": greedy longest-match-first splits into "▁un" (whole-word start) + "happy"
    // (bare continuation piece, was "##happy" before the exporter's phantom() transform).
    {
        const auto ids = vocab->encode("unhappy");
        LOOM_CHECK((ids == std::vector<int32_t>{2, 7, 8, 3}));
    }

    // "hello, world": ',' is punctuation -- isolated into its own single-character word between
    // "hello" and "world", never absorbed into either neighbor.
    {
        const auto ids = vocab->encode("hello, world");
        LOOM_CHECK((ids == std::vector<int32_t>{2, 5, 9, 6, 3}));
    }

    // "xyz": no vocab piece covers any prefix of "▁xyz" -- the whole word collapses to a single [UNK],
    // not partial pieces + UNK (matches llama.cpp's own llm_tokenizer_wpm_session::tokenize exactly).
    {
        const auto ids = vocab->encode("xyz");
        LOOM_CHECK((ids == std::vector<int32_t>{2, 1, 3}));
    }

    // "HELLO": lowercase_ is true in this fixture -- uppercase input still matches "▁hello".
    {
        const auto ids = vocab->encode("HELLO");
        LOOM_CHECK((ids == std::vector<int32_t>{2, 5, 3}));
    }

    // "café": strip_accents_ is true -- NFD-decomposes then drops the combining acute mark, matching
    // the accent-stripped "▁cafe" vocab entry (real BERT-uncased convention).
    {
        const auto ids = vocab->encode("caf\xC3\xA9"); // "café"
        LOOM_CHECK((ids == std::vector<int32_t>{2, 10, 3}));
    }

    // Empty input still gets CLS/SEP-wrapped (no words in between).
    {
        const auto ids = vocab->encode("");
        LOOM_CHECK((ids == std::vector<int32_t>{2, 3}));
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
