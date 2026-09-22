// Tests loom::F5Vocab (src/core/f5_vocab.cpp) against the synthetic character-table fixture generated
// by tests/fixtures/make_f5_vocab_gguf.py.
//
// **The measured claim this class makes is that a codepoint scan IS `convert_char_to_pinyin` for
// ordinary prose.** That comparison lives where the reference can be run -- the export side, against
// the real function -- and what belongs here is the half that is answerable from the table: that the
// scan is a scan, that the five substitutions happen, that the pinyin rows are unreachable from ASCII,
// and that CJK is refused by name rather than mapped to the unknown id.

#include "test_util.h"

#include "loom/loom.h"

#include "cpu_backend.h"

#include <string>
#include <vector>

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);

    const std::string path = std::string(LOOM_TEST_FIXTURE_DIR) + "/f5_vocab_test.gguf";
    auto model = loom::GgufModel::load(path, backend.get());
    LOOM_CHECK(model != nullptr);

    auto vocab = loom::F5Vocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    LOOM_CHECK(vocab->size() == 16);
    LOOM_CHECK(vocab->unk_id() == 0);
    // Every id the GRAPH sees is one more than an id from this table -- the reference reserves
    // embedding row 0 for "no character here". Declared in the file rather than baked into the driver,
    // because a host building ids itself needs the number.
    LOOM_CHECK(vocab->filler_offset() == 1);

    // One row per CHARACTER, in order. "note" -> n, o, t, e.
    LOOM_CHECK((vocab->encode("note") == std::vector<int32_t>{8, 9, 10, 7}));
    LOOM_CHECK((vocab->encode(" a ") == std::vector<int32_t>{0, 6, 0}));

    // **The pinyin rows are unreachable from ASCII, which is the reason `single_` exists.** A
    // longest-match scan -- which is what `PhonemeVocab` does over its symbol table -- would consume
    // "an1" as row 14 here. F5-TTS never produces that id from English text: it produces it from the
    // Chinese character whose pinyin it is, through a conversion this class does not do.
    LOOM_CHECK((vocab->encode("an1") == std::vector<int32_t>{6, 8, 12}));
    LOOM_CHECK(vocab->id_to_piece(14) == "an1");

    // The reference's own five-character `custom_trans`, applied before any lookup: `;` -> `,` and the
    // four curly quotes onto their straight forms. Each of the five has no row of its own, which is
    // why the reference substitutes rather than letting them fall to the unknown id.
    {
        size_t unknown = 0;
        const auto ids = vocab->encode("a;e", &unknown);
        LOOM_CHECK((ids == std::vector<int32_t>{6, 2, 7}));
        LOOM_CHECK(unknown == 0);
    }
    LOOM_CHECK((vocab->encode("\xE2\x80\x98""a\xE2\x80\x99") == std::vector<int32_t>{4, 6, 4}));
    LOOM_CHECK((vocab->encode("\xE2\x80\x9C""a\xE2\x80\x9D") == std::vector<int32_t>{5, 6, 5}));

    // A multi-byte row is matched by CODEPOINT, not by byte: "é" is one row and two bytes, and a
    // byte-length test would have indexed neither it nor the pinyin rows correctly.
    LOOM_CHECK((vocab->encode("\xC3\xA9") == std::vector<int32_t>{13}));

    // An unmapped character is the unknown id and is COUNTED -- the reference's own
    // `vocab_char_map.get(c, 0)`, which is a real fallback rather than an error. Counting it is what
    // lets a host say the sentence changed (Retro-006's lesson, one family over).
    {
        size_t unknown = 0;
        const auto ids = vocab->encode("axe", &unknown);
        LOOM_CHECK((ids == std::vector<int32_t>{6, 0, 7}));
        LOOM_CHECK(unknown == 1);
    }

    // **CJK is refused, not mapped.** A per-character lookup of Chinese finds no row and would return
    // the unknown id for a whole sentence -- audible as babble and reported as nothing.
    {
        std::string first;
        LOOM_CHECK(vocab->needs_pinyin("hello \xE4\xB8\xAD\xE6\x96\x87", &first));
        LOOM_CHECK(first == "\xE4\xB8\xAD");
        LOOM_CHECK(!vocab->needs_pinyin("hello"));
        bool threw = false;
        try {
            vocab->encode("\xE4\xB8\xAD");
        } catch (const loom::Error&) {
            threw = true;
        }
        LOOM_CHECK(threw);
    }

    // Decode is plain concatenation: no boundary marker exists in this scheme, so there is nothing
    // else it could be. It is the inverse of `encode` for everything `encode` produced.
    LOOM_CHECK(vocab->decode({8, 9, 10, 7}) == "note");
    LOOM_CHECK(vocab->decode({}).empty());

    // An id past the end raises rather than rendering as nothing, and the message names the one
    // mistake that produces such an id: forgetting the filler offset.
    {
        bool threw = false;
        try {
            vocab->decode({16});
        } catch (const loom::Error&) {
            threw = true;
        }
        LOOM_CHECK(threw);
    }

    // A file of another family is declined, not claimed: `load` returns null so a caller can try the
    // next scheme, which is what every vocabulary class here does.
    {
        auto other = loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/ctc_vocab_test.gguf",
                                           backend.get());
        LOOM_CHECK(loom::F5Vocab::load(*other) == nullptr);
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
