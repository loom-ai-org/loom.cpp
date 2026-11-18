#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class GgufModel;

// PLaMo's tokenizer (`Plamo3Tokenizer`, pfnet), "tokenizer.ggml.model" == "plamo".
//
// A unigram model decoded by Viterbi, which the reference runs as an Aho-Corasick walk from the END of
// the text (`tokenization_plamo.AhoCorasick._encode`). Its rules, each kept here because a different
// answer would follow from dropping it:
//
//   * every non-byte vocabulary entry is a piece (CONTROL ones included -- `<|plamo:begin_` and
//     `:plamo|>` are pieces too), scored `round(score * 1e4)` as an INTEGER, so ties are exact;
//   * the best split of `text[i:]` minimises the summed negated scores, and of two equal splits the one
//     whose first piece is LONGER wins (the reference lists a suffix's pieces longest first and keeps
//     the first strict improvement);
//   * a character no piece covers costs 10^7 (the reference's UNKNOWN_SCORE sentinel) and is emitted as
//     its UTF-8 bytes, one `<0xNN>` token each;
//   * hard boundaries, which no piece crosses: around every `<|plamo:...|>` tag (up to 64 characters
//     without `|` or whitespace), around runs of `break_spaces` or more spaces, and around runs of
//     `break_repeats` or more of one character (not newline: Python's `.`) -- the reference's
//     `_sp_token_matcher` and `_matcher`, applied in that order.
//
// No normalisation, no BOS: the reference adds `<|plamo:bos|>` in `build_inputs_with_special_tokens`,
// and a driver building a PFML3 prompt places it itself.
class PlamoVocab {
public:
    // nullptr when "tokenizer.ggml.model" is absent or not "plamo"; throws LoadError when it is "plamo"
    // and a required key is missing or malformed.
    static std::unique_ptr<PlamoVocab> load(const GgufModel& model);

    std::vector<int32_t> encode(const std::string& text) const;
    // Pieces concatenated, byte tokens as their bytes, the result read as UTF-8 with U+FFFD for a broken
    // sequence (the reference's `errors="replace"`). `skip_special` drops CONTROL and UNKNOWN pieces.
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const;

    size_t size() const { return pieces_.size(); }
    const std::string& id_to_piece(int32_t id) const { return pieces_.at(static_cast<size_t>(id)); }
    int32_t piece_to_id(const std::string& piece) const;   // -1 when absent
    bool is_special(int32_t id) const;
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }

    PlamoVocab(const PlamoVocab&) = delete;
    PlamoVocab& operator=(const PlamoVocab&) = delete;
    PlamoVocab(PlamoVocab&&) = default;

    // Built from parts, for tests: `types` 0 normal, 1 control, 2 byte, 3 unknown.
    PlamoVocab(std::vector<std::string> pieces, std::vector<int32_t> scores, std::vector<uint8_t> types,
               int32_t break_spaces, int32_t break_repeats);

private:
    struct Piece {
        int32_t id;
        int32_t score;
    };
    void encode_segment(const std::vector<char32_t>& text, size_t begin, size_t end, std::vector<int32_t>& out) const;

    std::vector<std::string> pieces_;
    std::vector<uint8_t> types_;
    std::unordered_map<std::string, Piece> lookup_;   // UTF-8 piece -> id, score (non-byte pieces)
    std::unordered_map<std::string, int32_t> by_piece_;
    int32_t byte_ids_[256] = {};
    size_t max_len_ = 1;                               // the longest piece, in codepoints
    int32_t break_spaces_ = 0;
    int32_t break_repeats_ = 0;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
};

} // namespace loom
