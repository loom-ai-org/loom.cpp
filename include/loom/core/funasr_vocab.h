#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;

// FunASR's character/subword vocabulary, "tokenizer.ggml.model"=="funasr" -- the scheme Paraformer and
// the other FunASR `CharTokenizer` checkpoints carry (EXPORT-ROADMAP.md's family 5).
//
// **IT IS DECODE-ONLY, for the same reason `CtcVocab` is** (ADR-033): the ids come out of a
// non-autoregressive decoder's per-row argmax and the model has no text input, so there is nothing for
// an `encode` to be the inverse of.
//
// **WHAT MAKES IT A SCHEME OF ITS OWN IS HOW PIECES COMPOSE, NOT THE TABLE.** The table is flat, one
// piece per row, exactly like family 4's. What differs is the marking convention and the assembly:
//
//   ["and","so","my","f@@","el@@","low"]  ->  "and so my fellow"   `@@` continues into the NEXT piece
//   ["hello","你","好"]                    ->  "hello你好"           the space is REMOVED before CJK
//   ["b","b","c","news"]                  ->  "BBC news"           letter runs collapse AND UPPERCASE
//   ["<s>","and","</s>"]                  ->  "and"                control pieces drop
//
// `@@` is a SUFFIX marking "I continue", where SentencePiece's U+2581 and WordPiece's "##" both mark
// word START. Those are duals and not interchangeable: whether a piece begins a word depends on its
// PREDECESSOR here, and the same piece string appears in both roles -- so no rewrite of the table into
// an existing tag can express it, which is why this is a fifth reader rather than a fourth spelling.
//
// This is `funasr.utils.postprocess_utils.sentence_postprocess` plus `abbr_dispose`, which is what the
// reference pipeline puts between its tokenizer and its transcript. Its own `tokenizer.decode` returns
// the pieces concatenated -- "andsomyf@@el@@low" -- which is not text in any sense a caller wants, so
// the composition belongs here rather than in a door above: `decode` is where every other vocabulary
// class answers "what do these ids spell", and both `transcribe` and `detokenize` reach it.
//
// The reference has three branches (all-CJK, all-Latin, mixed) and they differ only in the timestamps
// they build; for TEXT the mixed branch reproduces all three, verified over 20,000 random piece
// sequences drawn from the real 8,404-piece vocabulary. Only that branch is implemented.
//
// **THE PER-PIECE SCRIPT IS READ OFF THE FILE, NOT COMPUTED HERE.** The reference decides `isAllChinese`
// / `isAllAlpha` per CHARACTER with Python's Unicode `isalpha()`, and the vocabulary contains one
// character (U+2B5AF, a CJK extension ideograph) that is alphabetic and outside the U+4E00-U+9FFF block
// the Chinese test uses -- so a plausible ASCII-plus-CJK rule in C++ gets it wrong, silently, on one
// row in 8,404. The exporter knows the answer exactly and writes it; `tokenizer.ggml.piece_script` is
// one small int per piece. ADR-027's principle one family over: whoever actually knows a fact states it.
class FunasrVocab {
public:
    // How a piece behaves when assembled. Written by the exporter, one per row of `tokens`.
    enum Script : uint8_t {
        OTHER = 0,    // punctuation and anything else: emitted literally, spacing untouched
        CJK = 1,      // every character is CJK / an ASCII digit / '@' -- emitted bare, and EATS a
                      // pending space, which is what makes "hello你好" rather than "hello 你好"
        LATIN = 2,    // every character is alphabetic-and-not-CJK, or an apostrophe -- a word, spaced
    };
    // The `@@` continuation marker is ORTHOGONAL to this: `f@@` is OTHER (an '@' is neither CJK nor
    // alphabetic) and is still a continuation, while `9@@` is CJK and is NOT one. `decode` therefore
    // tests the marker between the two script cases, in the reference's own order.

    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it is present but not "funasr"
    // (i.e. this model uses one of the other schemas -- callers should try those too).
    static std::unique_ptr<FunasrVocab> load(const GgufModel& model);

    // Assembles the transcript these ids spell. Ids outside the vocabulary throw, matching every other
    // class here: an id the model cannot have produced is a caller error, not an empty string.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }

    FunasrVocab(const FunasrVocab&) = delete;
    FunasrVocab& operator=(const FunasrVocab&) = delete;

private:
    FunasrVocab() = default;

    // One array indexed by id, the convention every vocab class in this schema shares.
    std::vector<std::string> tokens_;
    std::vector<uint8_t> script_;
    // True for a piece the transcript never contains -- `<s>`, `</s>`, `<unk>`, `<OOV>` and the blank.
    // Read off `tokenizer.ggml.token_type`, so the file states which rows those are rather than this
    // matching spellings.
    std::vector<bool> control_;
};

} // namespace loom
