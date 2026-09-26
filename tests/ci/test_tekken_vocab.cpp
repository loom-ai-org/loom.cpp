// Tests loom::BpeVocab's `tekken` shape (Mistral's tiktoken vocabulary) against the hand-traced fixture
// from tests/fixtures/make_tekken_vocab_gguf.py. Every case names a rule the shape adds over the other
// byte-level shapes; the real vocabulary is diffed against `mistral_common` in the gate
// (test_e2e_voxtral_tts_lua_driver, and the tokenizer differential it cites).

#include "test_util.h"

#include "loom/loom.h"
#include "loom/core/bpe_vocab.h"

#include "cpu_backend.h"

namespace {
// Ids are rank + 4: four markers come first.
int32_t rank(int r) { return r + 4; }
int32_t byte(unsigned char b) { return rank(b); }
}  // namespace

int main() {
    ggml_backend_ptr backend(loom_test::cpu_backend());
    LOOM_CHECK(backend != nullptr);
    auto model = loom::GgufModel::load(std::string(LOOM_TEST_FIXTURE_DIR) + "/tekken_vocab_test.gguf", backend.get());
    LOOM_CHECK(model != nullptr);
    auto vocab = loom::BpeVocab::load(*model);
    LOOM_CHECK(vocab != nullptr);
    LOOM_CHECK(vocab->size() == 4 + 256 + 9);

    // The case change splits the letter run: "Hello" + "World", so the lowest-ranked pair "oW" (256),
    // which one chunk would merge first, never meets. Merged BY RANK with no merge list: ll, He, Hell,
    // Hello; then Wo.
    {
        const auto ids = vocab->encode("HelloWorld");
        const std::vector<int32_t> expected = {rank(260), rank(261), byte('r'), byte('l'), byte('d')};
        LOOM_CHECK(ids == expected);
        LOOM_CHECK(vocab->decode(ids) == "HelloWorld");
    }
    // An all-caps run fails the first letter alternative (`A* B+` needs a lowercase-ish letter) and is
    // one chunk of the second.
    {
        const auto ids = vocab->encode("HELLO");
        const std::vector<int32_t> expected = {byte('H'), byte('E'), byte('L'), byte('L'), byte('O')};
        LOOM_CHECK(ids == expected);
    }
    // A marker's spelling typed as text is text: tiktoken is built with `special_tokens={}`. Marker 3
    // is spelled "Wo", like rank 261, and "Wo" reaches the rank -- neither split out as the marker nor
    // shadowed by it in the piece table.
    {
        const auto ids = vocab->encode("Wo");
        LOOM_CHECK(ids.size() == 1 && ids[0] == rank(261));
        LOOM_CHECK(vocab->is_control(3));
        LOOM_CHECK(!vocab->is_control(rank(261)));
    }
    // A typed "<s>" is two chunks, "<s" and ">": the letter alternative's optional prefix is "<".
    {
        const auto ids = vocab->encode("<s>");
        const std::vector<int32_t> expected = {rank(262), byte('>')};
        LOOM_CHECK(ids == expected);
    }
    // One digit per chunk, so the "12" rank never applies.
    {
        const auto ids = vocab->encode("12");
        const std::vector<int32_t> expected = {byte('1'), byte('2')};
        LOOM_CHECK(ids == expected);
    }
    // No NFC: "e" + U+0301 stays three bytes, where NFC would compose it into U+00E9 and reach rank 264.
    {
        const auto ids = vocab->encode("e\xcc\x81");
        const std::vector<int32_t> expected = {byte('e'), byte(0xcc), byte(0x81)};
        LOOM_CHECK(ids == expected);
        const auto composed = vocab->encode("\xc3\xa9");
        LOOM_CHECK(composed.size() == 1 && composed[0] == rank(264));
    }
    // A caseless script is in both letter classes: "你好" is one chunk, the backtracking A* giving its
    // last character to B+. No rank joins its bytes, so six byte ids -- and no throw.
    {
        const auto ids = vocab->encode("\xe4\xbd\xa0\xe5\xa5\xbd");
        LOOM_CHECK(ids.size() == 6);
        LOOM_CHECK(vocab->decode(ids) == "\xe4\xbd\xa0\xe5\xa5\xbd");
    }
    LOOM_TEST_REPORT_AND_RETURN();
}
