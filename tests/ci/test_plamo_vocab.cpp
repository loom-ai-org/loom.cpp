// loom::PlamoVocab's rules on a hand-made vocabulary, each against what `tokenization_plamo.AhoCorasick`
// does. The whole real vocabulary is the gate's (test_e2e_plamo against the reference's own encodings);
// this pins the rules one at a time, so a regression names the rule.

#include "test_util.h"

#include "loom/core/plamo_vocab.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr uint8_t N = 0, C = 1, B = 2, U = 3;

loom::PlamoVocab make() {
    std::vector<std::string> pieces = {"<|plamo:unk|>", "<|plamo:bos|>", "<|plamo:tag|>", "a", "b", "ab", "abc", "c",
                                       " ", "  ", "x", "xx", "<|plamo:begin_"};
    std::vector<int32_t> scores = {0, 0, 0, -10000, -10000, -15000, -40000, -10000, -5000, -6000, -10000, -10000, 0};
    std::vector<uint8_t> types = {U, C, C, N, N, N, N, N, N, N, N, N, C};
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02X>", b);
        pieces.emplace_back(name);
        scores.push_back(0);
        types.push_back(B);
    }
    return loom::PlamoVocab(pieces, scores, types, /*break_spaces=*/2, /*break_repeats=*/4);
}

std::vector<int32_t> ids(std::initializer_list<int32_t> v) { return v; }
int32_t byte(unsigned char b) { return 13 + b; }

} // namespace

int main() {
    const loom::PlamoVocab v = make();

    // The cheapest split: "ab" (1.5) beats "a"+"b" (2.0); "abc" (4.0) loses to "ab"+"c" (2.5).
    LOOM_CHECK(v.encode("ab") == ids({5}));
    LOOM_CHECK(v.encode("abc") == ids({5, 7}));

    // A tie goes to the longer first piece: "xx" alone (1.0) vs "x"+"x" (2.0) is not a tie, so make one --
    // "xxx" = "xx"+"x" or "x"+"xx" (both 2.0): the walk from the end keeps the longer piece FIRST.
    LOOM_CHECK(v.encode("xxx") == ids({11, 10}));

    // An uncovered character is its UTF-8 bytes; "é" is two.
    LOOM_CHECK(v.encode("aé") == ids({3, byte(0xC3), byte(0xA9)}));

    // A `<|plamo:...|>` tag is cut out on both sides, then matched as a piece (score 0).
    LOOM_CHECK(v.encode("a<|plamo:tag|>b") == ids({3, 2, 4}));

    // A partial tag piece is a piece like any other, where no full tag surrounds it.
    LOOM_CHECK(v.encode("<|plamo:begin_a") == ids({12, 3}));

    // Spaces: a run of 2 or more is cut out; inside it the cheapest split ("  " 0.6 < " "+" " 1.0).
    LOOM_CHECK(v.encode("a  b") == ids({3, 9, 4}));
    // A single space is not a run, so "b c" is one segment.
    LOOM_CHECK(v.encode("b c") == ids({4, 8, 7}));

    // A run of 4+ of one character is cut out: "aaaa" alone, then "b".
    LOOM_CHECK(v.encode("aaaab") == ids({3, 3, 3, 3, 4}));
    // ... which differs from no cut only when a piece would have crossed it: "xxxxab" keeps "ab" whole
    // either way, but "xxxx" is segmented on its own.
    LOOM_CHECK(v.encode("xxxxab") == ids({11, 11, 5}));

    // Decode: pieces concatenated, bytes reassembled, specials skippable.
    LOOM_CHECK(v.decode(v.encode("aé<|plamo:tag|>ab")) == "aé<|plamo:tag|>ab");
    LOOM_CHECK(v.decode(ids({1, 3, 2}), /*skip_special=*/true) == "a");
    // A broken byte sequence decodes to U+FFFD, as `errors="replace"`.
    LOOM_CHECK(v.decode(ids({byte(0xC3)})) == "\xEF\xBF\xBD");

    LOOM_CHECK(v.bos_id() == 1);
    LOOM_TEST_REPORT_AND_RETURN();
}
