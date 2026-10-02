#pragma once

#include "loom/core/py_regex.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class GgufModel;

// Soprano TTS's text front end, "tokenizer.ggml.model"=="soprano" (family 9's eighth leaf).
//
// `SopranoTTS._preprocess_text`, which runs before any id reaches the model:
//
//   1. `text.strip()`, then `clean_text`: tortoise-tts's English normaliser -- a pre-`unidecode` table,
//      `unidecode`, newlines to sentence ends, numbers (dates, phones, times, money, decimals,
//      arithmetic, fractions, ordinals, inflect's `number_to_words`), links/dashes/dots/brackets,
//      abbreviations, CamelCase split, symbols to words, lower case, unknown characters dropped,
//      whitespace and punctuation collapsed, tripled letters cut to two.
//   2. `split_and_recombine_text`: one piece per sentence (its quote-aware state machine, verbatim).
//   3. Pieces shorter than `min_length` merged into the one before (or, first, the one after).
//   4. Each piece wrapped `[STOP][TEXT]{piece}[START]` and tokenized by the character BPE.
//
// `encode` returns every piece's prompt, one after another. Each opens with `[STOP]`, the model's EOS,
// which is how the driver finds where one generation ends and the next begins (ADR-044's shape with
// the reference's own opener as the header): after step 1 no bracket is left to spell it with.
//
// **The rules are the reference's own pattern strings** (`tokenizer.ggml.soprano.*`, written by
// `soprano_tokenizer_export` and checked there against the reference's source), run by `PyRegex`.
// What is code here is each callback's SHAPE, with its few literal words, as the reference's function
// bodies spell them -- including `_expand_time`'s f-string slip, which renders a three-part time's
// minutes as a Python set (`{'15'}`); the brackets are dropped by step 1's own clean-up later.
//
// **Where the reference raises, this throws `Error`**: a money amount `int()` cannot parse (`$,5`), a
// number of more than 36 digits (inflect's `NumOutOfRangeError`), and a text with nothing left to say.
class SopranoVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it is present but not "soprano".
    // Throws `LoadError` if the tag is present and a required key is missing or malformed.
    static std::unique_ptr<SopranoVocab> load(const GgufModel& model);

    // Step 1.
    std::string clean_text(const std::string& text) const;
    // Steps 1-3: the pieces the model will be asked to say, one generation each.
    std::vector<std::string> sentences(const std::string& text) const;
    // Steps 1-4: every piece's prompt ids, concatenated.
    std::vector<int32_t> encode(const std::string& text) const;
    // The tokenizer alone (normalizer, pre-tokenizer, BPE) on one piece, no prompt around it.
    std::vector<int32_t> tokenize(const std::string& text) const;
    // inflect's `number_to_words(digits, andword='')`, with an ordinal suffix when `ordinal`
    // ("21" -> "twenty-first"), and `_expand_number`'s whole rule (years included) -- for the tests.
    std::string number_to_words(const std::string& digits, bool ordinal) const;
    std::string expand_number(const std::string& digits) const;

    std::string decode(const std::vector<int32_t>& ids) const;
    const std::string& id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t stop_id() const { return stop_id_; }

    SopranoVocab(const SopranoVocab&) = delete;
    SopranoVocab& operator=(const SopranoVocab&) = delete;
    ~SopranoVocab();

private:
    SopranoVocab() = default;

    struct Rule {
        PyRegex re;
        std::u32string repl;
    };

    const PyRegex& rule(const char* name) const;
    const std::u32string& templ(const char* name) const;
    std::u32string apply(const std::vector<Rule>& table, std::u32string text) const;

    std::u32string unidecode(const std::u32string& text) const;
    std::u32string normalize_numbers(std::u32string text) const;
    std::u32string normalize_special(std::u32string text) const;
    std::u32string normalize_mixedcase(const std::u32string& text) const;
    std::u32string clean(const std::u32string& text) const;
    std::vector<std::u32string> split_and_recombine(std::u32string text) const;

    // inflect.
    const std::string& mill(size_t index) const;
    std::string tenfn(int tens, int units, size_t mindex) const;
    std::string words(const std::string& digits, int group, const std::string& andword,
                      const std::string& zero, bool ordinal) const;
    std::string sub_ord(const std::string& value) const;

    void bpe(const std::u32string& word, std::vector<int32_t>& ids) const;

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    int32_t unk_id_ = 0, stop_id_ = 3, text_id_ = 1, start_id_ = 2;

    std::unordered_map<std::string, PyRegex> rules_;
    std::unordered_map<std::string, std::u32string> templates_;
    std::vector<Rule> preunicode_, abbreviations_, special_;

    std::vector<std::string> unit_, teen_, ten_, mill_, nth_suffixes_;
    std::vector<std::pair<std::string, std::string>> ordinal_;

    size_t desired_length_ = 1, max_length_ = 300, min_length_ = 30;

    std::vector<char32_t> unidecode_cps_;      // sorted
    std::vector<int32_t> unidecode_offsets_;
    std::u32string unidecode_text_;
};

} // namespace loom
