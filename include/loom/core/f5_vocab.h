#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class GgufModel;

// F5-TTS's character vocabulary, "tokenizer.ggml.model"=="f5" (EXPORT-ROADMAP.md's family 9).
//
// **The table is flat and decode-ordered like "ctc"'s and "funasr"'s; what is new is that its rows are
// CHARACTERS and that the reference's own text function is not a table.** `convert_char_to_pinyin`
// runs `rjieba` word segmentation and `pypinyin` before a single id is looked up -- Chinese characters
// become toned pinyin syllables (`zhong1`, which is why the table has multi-character rows), and a
// space is inserted before a multi-character SEGMENT whose predecessor did not end in one. That last
// decision comes from jieba's dictionary-and-HMM DAG and no table reproduces it.
//
// **So this implements the half that IS a table, and its boundary was measured rather than assumed.**
// `encode` applies the reference's own five-character substitution (`;`->`,` and the four curly
// quotes) and then maps codepoints. Against `convert_char_to_pinyin` over generated English prose:
// ordinary space-separated prose agrees on 2000/2000 inputs; text carrying multi-character punctuation
// runs (`--`, `...`) agrees on 1100/2000 and text carrying hyphen-joined digit groups (`2026-09-18`)
// on 1758/2000. Both divergent classes differ by ONE INSERTED SPACE, always in the same direction.
// A caller who needs reference-exact ids for text of those shapes runs the reference function and
// passes ids; the driver's `text_ids` input takes them directly. Same boundary the phoneme-input
// families draw around g2p.
//
// **CJK is refused rather than mapped.** A per-character lookup of Chinese finds no row and would
// return the unknown id for a whole sentence -- audible as silence or babble, and reported as nothing.
// `encode` reports the count and the first offending character so a host can say what it cannot do.
//
// **Every id the GRAPH sees is one more than an id from this table.** The reference reserves embedding
// row 0 for "no character here" (`text = text + 1`), and the offset is declared in the file rather
// than baked into the driver, because a host that builds ids itself needs it. `encode` returns TABLE
// ids; adding the offset is the driver's job and `filler_offset()` is how a host learns the number.
class F5Vocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it is present but not "f5".
    static std::unique_ptr<F5Vocab> load(const GgufModel& model);

    // Codepoints -> table ids, with the reference's `custom_trans` applied first. A character with no
    // row becomes `unk_id()` and increments `*unknown` when that is non-null -- the reference's own
    // `vocab_char_map.get(c, 0)`, which is a real fallback rather than an error path.
    //
    // Throws if the text contains a CJK codepoint: see the class comment. `*cjk` receives the first
    // such character's UTF-8 spelling when it is non-null and the call throws, so a caller can report
    // it without decoding the message.
    std::vector<int32_t> encode(const std::string& text, size_t* unknown = nullptr) const;

    // True when `text` contains a codepoint this table cannot represent without pinyin conversion.
    // Exposed so a host can ask before it encodes rather than catching.
    bool needs_pinyin(const std::string& text, std::string* first = nullptr) const;

    // Table ids -> text, by plain concatenation. The inverse of `encode` for everything `encode`
    // produced, and the only sensible reading of a row list for a vocabulary with no boundary marker.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t unk_id() const { return unk_id_; }
    uint32_t filler_offset() const { return filler_offset_; }

    F5Vocab(const F5Vocab&) = delete;
    F5Vocab& operator=(const F5Vocab&) = delete;

private:
    F5Vocab() = default;

    std::vector<std::string> tokens_;
    // Only the SINGLE-codepoint rows, which is what `encode` looks up. The multi-character rows are
    // pinyin syllables and are reachable only through the conversion this does not do, so indexing
    // them here would let a two-character English substring match one by accident.
    std::unordered_map<std::string, int32_t> single_;
    int32_t unk_id_ = 0;
    uint32_t filler_offset_ = 1;
};

} // namespace loom
