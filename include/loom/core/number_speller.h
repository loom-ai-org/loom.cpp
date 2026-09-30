#pragma once

// Spells the numbers in a text as words before a vocabulary segments it (ADR-059).
//
// A file declares one with `tokenizer.ggml.numbers.scheme`. The only scheme is
// "english_number_normalizer": transformers' `EnglishNumberNormalizer`, which `SpeechT5Tokenizer`
// runs when `normalize=True`, and whose vocabulary has no digits to encode them with. The C++ holds
// the function's SHAPE; its words, currency names and the two Unicode classes its regexes use (`\d`
// and `\w`) are the file's data, read off the reference at export (ADR-041's split).
//
// Two intentional differences from the reference, both where it drops words:
//   * an integer part that is all zeros is "zero" (upstream spells it as nothing, so "0" disappears
//     and "0.5" is " point five");
//   * every thousands separator goes (upstream's comma pass never rescans, so "1,000,000" becomes
//     "1000,000" and is spoken "one thousand" and then nothing).
// Inputs the reference raises on (two currency symbols in one number, more than 36 integer digits)
// are left unchanged.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;

class NumberSpeller {
public:
    // nullptr when the file declares no speller. Throws LoadError for an unknown scheme or a
    // malformed table.
    static std::unique_ptr<NumberSpeller> load(const GgufModel& model);

    // `text` with its numbers spelled out, UTF-8 in and out.
    std::string apply(const std::string& text) const;

private:
    NumberSpeller() = default;

    bool is_word(char32_t c) const;
    int digit_value(char32_t c) const;          // 0..9, or -1 when `c` is not an Nd digit
    size_t match_at(const std::u32string& t, size_t i) const;   // length of the match at i, or 0
    bool convert(const std::u32string& number, std::string& out) const;
    std::string spell(int num) const;           // 0..999

    std::vector<std::string> ones_, teens_, tens_, scales_;
    std::vector<char32_t> chain_;
    std::vector<char32_t> currency_symbols_;
    std::vector<std::string> currency_names_;
    std::vector<char32_t> digit_zeros_;         // sorted
    std::vector<std::pair<char32_t, char32_t>> word_ranges_;   // sorted, inclusive
};

} // namespace loom
