// Unit tests for loom::nfc_normalize/is_letter_or_number (src/core/unicode.cpp), exercised in isolation
// before BpeVocab ever depends on them (see BACKLOG.md's "verify before trusting" discipline).

#include "test_util.h"

#include "loom/core/unicode.h"

int main() {
    using loom::is_letter_or_number;
    using loom::nfc_normalize;
    using loom::utf8_decode;

    // --- is_letter_or_number ---
    LOOM_CHECK(is_letter_or_number(U'A'));
    LOOM_CHECK(is_letter_or_number(U'z'));
    LOOM_CHECK(is_letter_or_number(U'5'));
    LOOM_CHECK(is_letter_or_number(0x00E9));  // e-acute (Latin-1 Supplement letter)
    LOOM_CHECK(is_letter_or_number(0x4E2D));  // CJK "中"
    LOOM_CHECK(!is_letter_or_number(U' '));
    LOOM_CHECK(!is_letter_or_number(U'.'));
    LOOM_CHECK(!is_letter_or_number(U'\n'));
    LOOM_CHECK(!is_letter_or_number(0x0301)); // combining acute accent -- a Mark, not a Letter/Number

    // --- nfc_normalize: already-composed text is a no-op ---
    LOOM_CHECK(nfc_normalize("hello world") == "hello world");
    LOOM_CHECK(nfc_normalize("caf\xC3\xA9") == "caf\xC3\xA9"); // "café" (precomposed e-acute), unchanged

    // --- nfc_normalize: decomposed "e" + combining acute (U+0065 U+0301) recomposes to U+00E9 ---
    {
        const std::string decomposed = "e\xCC\x81"; // U+0065, U+0301 (UTF-8: 0xCC 0x81)
        const std::string normalized = nfc_normalize(decomposed);
        const std::string expected_precomposed = "\xC3\xA9"; // U+00E9 in UTF-8
        LOOM_CHECK(normalized == expected_precomposed);
    }

    // --- nfc_normalize: Hangul algorithmic decomposition/recomposition round-trips (starts precomposed,
    //     must stay precomposed -- exercises the Hangul-specific arithmetic path, not the table path) ---
    {
        const std::string hangul = "\xEA\xB0\x80"; // U+AC00 "가" (a full LV syllable)
        LOOM_CHECK(nfc_normalize(hangul) == hangul);
    }

    // --- nfc_normalize: another real pairwise recomposition case (base letter + combining ring above) ---
    {
        const std::string a_ring = "A\xCC\x8A"; // U+0041, U+030A
        const std::string normalized = nfc_normalize(a_ring);
        const std::string expected = "\xC3\x85"; // U+00C5 "Å"
        LOOM_CHECK(normalized == expected);
    }

    // --- utf8_decode/encode round-trip sanity ---
    {
        const std::string s = "hello \xE4\xB8\xAD\xE6\x96\x87 caf\xC3\xA9"; // "hello 中文 café"
        const auto cps = utf8_decode(s);
        LOOM_CHECK(loom::utf8_encode(cps) == s);
    }

    // --- is_punctuation/is_mark (backing WordPieceVocab's word-splitting + qwen35's BPE shape) ---
    LOOM_CHECK(loom::is_punctuation(U'.'));
    LOOM_CHECK(loom::is_punctuation(U'!'));
    LOOM_CHECK(loom::is_punctuation(U'('));
    LOOM_CHECK(!loom::is_punctuation(U'A'));
    LOOM_CHECK(!loom::is_punctuation(U' '));
    LOOM_CHECK(loom::is_mark(0x0301)); // combining acute accent
    LOOM_CHECK(!loom::is_mark(U'A'));
    LOOM_CHECK(!loom::is_mark(U'5'));

    // --- to_lower ---
    LOOM_CHECK(loom::to_lower(U'A') == U'a');
    LOOM_CHECK(loom::to_lower(U'z') == U'z'); // already lowercase -- identity
    LOOM_CHECK(loom::to_lower(0x00C9) == 0x00E9); // É -> é
    LOOM_CHECK(loom::to_lower(U'5') == U'5'); // no mapping -- identity

    // --- nfd_normalize: precomposed "é" decomposes to "e" + combining acute (opposite of nfc_normalize) ---
    {
        const std::string precomposed = "\xC3\xA9"; // U+00E9 "é"
        const std::string decomposed = "e\xCC\x81";  // U+0065 U+0301
        LOOM_CHECK(loom::nfd_normalize(precomposed) == decomposed);
    }
    // --- nfd_normalize: already-decomposed input is a no-op ---
    {
        const std::string decomposed = "e\xCC\x81";
        LOOM_CHECK(loom::nfd_normalize(decomposed) == decomposed);
    }
    // --- nfd_normalize: Hangul decomposes algorithmically to jamo (unlike NFC, which stays precomposed) ---
    {
        const std::string hangul = "\xEA\xB0\x80"; // U+AC00 "가"
        const std::string jamo = loom::nfd_normalize(hangul);
        LOOM_CHECK(jamo != hangul);
        LOOM_CHECK(loom::utf8_decode(jamo).size() == 2); // L+V, no trailing consonant
    }

    LOOM_TEST_REPORT_AND_RETURN();
}
