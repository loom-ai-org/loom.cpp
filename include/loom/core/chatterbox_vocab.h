#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace loom {

class GgufModel;

// Chatterbox's text front end, "tokenizer.ggml.model"=="chatterbox" (EXPORT-ROADMAP.md's family 9).
//
// **A character-level BPE, and NOT the byte-level one "gpt2" names.** `tokenizer.json` is a
// `tokenizers` BPE with no normalizer, a `Whitespace` pre-tokenizer (`\w+|[^\w\s]+`), no GPT-2 byte
// mapping and `[UNK]` per unknown character (`fuse_unk: false`). `BpeVocab` NFC-normalizes and
// byte-maps, and either would change ids, so this is its own tag (ADR-033: a tag answers "which
// scheme").
//
// `encode` is `ChatterboxTTS.generate`'s whole text path, in its order:
//   1. `punc_norm`: an empty text becomes the reference's own stand-in sentence; the first character is
//      upper-cased when it is lower-case (`str.upper()`, from the file's table); whitespace runs collapse to one space; a replacement table
//      is applied pair by pair, in order; trailing spaces are stripped and a full stop is appended
//      unless the text already ends in a sentence ender. **The table, the enders and the stand-in are
//      the FILE's** (`tokenizer.ggml.chatterbox.*`), written by the exporter from the reference, so
//      no model constant lives here.
//   2. `EnTokenizer.encode`: every space becomes the `[SPACE]` token's spelling.
//   3. `tokenizers`' encode: split on the ADDED tokens (leftmost-longest, their literal spelling --
//      which is how a caller's `[laughter]` reaches the model), pre-tokenize each remaining run, and
//      BPE-merge each pre-token by merge rank.
//
// **The pre-tokenizer's `\w` is the file's, and that is what makes it exact without a Unicode table.**
// An unknown character becomes its own `[UNK]` and can never take part in a merge, so where the
// pre-token boundaries fall matters only for characters that ARE in the vocabulary. The exporter asks
// the reference's own pre-tokenizer which of those are word characters and ships the set
// (`word_chars`). After step 1 there is no whitespace left for `\s` to decide.
//
// **Case is the file's too.** `str.upper()` is a FULL case mapping (`ß` -> `SS`, `ﬁ` -> `FI`) that
// no single-codepoint table reproduces, so the exporter ships Python's own `c.upper()` for every
// codepoint whose `c.islower()` holds (1494 pairs, 8.5 KB). Measured before this: 17 of 500
// out-of-vocabulary texts differed, every one of them opening with such a character.
//
// The start/stop ids are NOT added: the reference adds them after tokenizing, and so does the driver,
// so a host that tokenizes elsewhere hands over the same thing this returns.
class ChatterboxVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it is present but not
    // "chatterbox". Throws `LoadError` if the tag is present and a required key is not.
    static std::unique_ptr<ChatterboxVocab> load(const GgufModel& model);

    // Steps 1-3 above. `*unknown` (when non-null) receives how many `[UNK]` ids the text produced --
    // the reference's own fallback, and a silent substitution a host should be able to report.
    std::vector<int32_t> encode(const std::string& text, size_t* unknown = nullptr) const;

    // Step 1 alone, exposed so a host (and the tests) can see what the model will actually be asked
    // to say.
    std::string normalize(const std::string& text) const;

    // `EnTokenizer.decode`: pieces concatenated, `[SPACE]` back to a space, `[STOP]`/`[UNK]` dropped.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t unk_id() const { return unk_id_; }

    ChatterboxVocab(const ChatterboxVocab&) = delete;
    ChatterboxVocab& operator=(const ChatterboxVocab&) = delete;

private:
    ChatterboxVocab() = default;

    void encode_run(const std::u32string& run, std::vector<int32_t>& ids, size_t* unknown) const;
    void encode_pretoken(const std::u32string& word, std::vector<int32_t>& ids, size_t* unknown) const;
    int32_t added_token_at(const std::string& text, size_t pos, size_t* len) const;

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> piece_to_id_;
    // "left right" -> rank, the order `merges` lists them in (lower merges first).
    std::unordered_map<std::string, int32_t> merge_rank_;
    // The added tokens' spellings -> id, matched literally before any pre-tokenization.
    std::unordered_map<std::string, int32_t> added_;
    size_t longest_added_ = 0;
    std::unordered_set<char32_t> word_chars_;
    std::unordered_map<char32_t, std::string> upper_;
    int32_t unk_id_ = 1;
    std::string space_token_;
    std::vector<std::string> replace_from_;
    std::vector<std::string> replace_to_;
    std::vector<std::string> enders_;
    std::string empty_text_;
    std::string terminal_;
    std::vector<std::string> dropped_on_decode_;
};

} // namespace loom
