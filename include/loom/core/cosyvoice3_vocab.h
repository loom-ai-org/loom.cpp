#pragma once

#include "loom/core/bpe_vocab.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace loom {

class GgufModel;

// CosyVoice3's text front end, "tokenizer.ggml.model"=="cosyvoice3" (family 9's seventh leaf).
//
// **The vocabulary is an ordinary byte-level Qwen2 BPE**, held as a `BpeVocab`. What makes this its own
// tag (ADR-033: a tag answers "which scheme") is the reference's text path around it,
// `CosyVoiceFrontEnd.text_normalize(text, split=True)`, which `inference_zero_shot` runs before any id
// reaches the model. This is that function as the release runs it with neither `ttsfrd` nor `wetext`
// installed -- the rules path:
//
//   * A text holding both `<|` and `|>` (markup such as `<|endofprompt|>`) skips all of it and is one
//     chunk, unstripped.
//   * Otherwise `strip()`, then by `contains_chinese` (a CJK Unified Ideograph anywhere):
//     - **Chinese**: newlines dropped; `replace_blank` (a space survives only between two ASCII
//       non-spaces); an ordered replacement table (`²` -> 平方, `.` -> `。`, ` - ` -> `，`, brackets
//       removed...); a trailing run of `，,、` becomes `。`; `split_paragraph` counting CHARACTERS.
//     - **Anything else**: `spell_out_number` (every run of `str.isdigit()` characters through
//       inflect's `number_to_words` -- so "3.5" is "three.five" and 1998 is "one thousand, nine hundred
//       and ninety-eight"); `split_paragraph` counting BPE TOKENS.
//   * `split_paragraph`: terminate the text if it does not end in a sentence ender, cut after every
//     ender (a closing quote right after one stays with it), then regroup greedily: a piece is closed
//     when adding the next sentence would pass `token_max_n` and it already holds more than
//     `token_min_n`; a last piece shorter than `merge_len` joins the one before.
//   * Pieces made only of punctuation and symbols (`[\p{P}\p{S}]*`) are dropped.
//
// Each piece is synthesised SEPARATELY by the reference -- its own LM decode, flow and vocoder, from the
// same voice -- and the caller concatenates the audio. So `encode` returns every piece's ids, each OPENED
// by `chunk_header()` (`<|endoftext|>`), and the driver loops over them (ADR-044's shape). A marked-up
// text is one chunk; one that itself spells `<|endoftext|>` is refused, since that id would be read as a
// boundary.
//
// **Every constant is the file's** (`tokenizer.ggml.cosyvoice3.*`), written by the exporter from the
// reference and from inflect (ADR-041): the replacement table, the ender sets, the budgets, Python's
// `isdigit` set, the `\d` digits and their values, inflect's number words, and the `\p{P}\p{S}` ranges.
//
// **Where the reference crashes, this throws `Error`**: an empty or all-whitespace text, a run of more
// digits than inflect has scale words for (37+), and a closing quote after an ender with no sentence
// before it (`split_paragraph` pops an empty list). A text that normalises to nothing but punctuation
// yields no chunk in the reference and no audio; here it throws too, rather than return no ids.
class CosyVoice3Vocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it is present but not
    // "cosyvoice3". Throws `LoadError` if the tag is present and a required key is not.
    static std::unique_ptr<CosyVoice3Vocab> load(const GgufModel& model);

    // The text path above: the pieces the model will be asked to say, one generation each.
    std::vector<std::string> chunks(const std::string& text) const;

    // Every chunk's BPE ids, each opened by `chunk_header()`.
    std::vector<int32_t> encode(const std::string& text) const;

    // `spell_out_number` and inflect's `number_to_words` on one digit run, exposed for hosts and tests.
    std::string spell_out_number(const std::string& text) const;
    std::string number_to_words(const std::u32string& run) const;

    // The BPE's decode, headers dropped.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const { return bpe_->id_to_piece(id); }
    size_t size() const { return bpe_->size(); }
    int32_t chunk_header() const { return header_; }
    const BpeVocab& bpe() const { return *bpe_; }

    CosyVoice3Vocab(const CosyVoice3Vocab&) = delete;
    CosyVoice3Vocab& operator=(const CosyVoice3Vocab&) = delete;

private:
    CosyVoice3Vocab() = default;

    // `*markup` receives whether the text skipped the front end (and is returned whole).
    std::vector<std::string> normalize(const std::string& text, bool* markup) const;
    std::vector<std::u32string> split_paragraph(std::u32string text, bool zh) const;
    size_t length(const std::u32string& text, bool zh) const;
    bool contains_chinese(const std::u32string& text) const;
    bool is_only_punctuation(const std::u32string& text) const;
    const std::string& scale(size_t index) const;

    std::unique_ptr<BpeVocab> bpe_;
    int32_t header_ = -1;
    std::string markup_open_;
    std::string markup_close_;
    std::vector<int32_t> zh_ranges_;
    std::vector<int32_t> punct_ranges_;
    std::vector<std::string> zh_pre_from_, zh_pre_to_;
    std::vector<std::string> zh_replace_from_, zh_replace_to_;
    std::unordered_set<char32_t> zh_trailing_;
    std::string zh_trailing_to_;
    std::unordered_set<char32_t> zh_enders_, en_enders_, closers_;
    std::u32string zh_terminal_, en_terminal_;
    size_t token_max_n_ = 0, token_min_n_ = 0, merge_len_ = 0;
    std::unordered_set<char32_t> digits_;
    std::unordered_map<char32_t, int> decimal_;
    std::vector<std::string> units_, teens_, tens_, scales_;
    std::string hundred_, and_word_, zero_word_, one_word_;
};

} // namespace loom
