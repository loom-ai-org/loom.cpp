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

    LOOM_TEST_REPORT_AND_RETURN();
}
