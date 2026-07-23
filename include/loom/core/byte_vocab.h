#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;

// ByT5-family byte-level vocabulary, "tokenizer.ggml.model"=="byt5". ByT5 tokenizes raw UTF-8 bytes
// directly with NO learned vocabulary, merges, or pretokenizer regex at all (unlike BpeVocab's GPT2-style
// byte-level BPE, which still learns merge rules over a "Ġ"-style byte-to-codepoint mapping): each raw
// byte `b` maps to a fixed id `b + 3` (3 leading special tokens precede the byte range: pad=0, eos=1,
// unk=2), and `extra_ids` T5-style span-corruption sentinels ("<extra_id_0>".."<extra_id_{n-1}>") are
// appended sequentially right after the byte range (ids 259, 260, ... by default) -- confirmed directly
// against a real `transformers.ByT5Tokenizer` instance's actual token ids, NOT its docstring (which
// describes a different, reversed sentinel-ordering scheme that doesn't match the real implementation).
//
// Byte-range piece text is deliberately NOT stored in `tokens_` (unlike every other id in this schema) --
// a raw byte >= 0x80 can't round-trip through a GGUF string array the way the other vocab types' pieces
// do (Python's `chr(b)` for b>=128 produces a Unicode codepoint whose OWN UTF-8 encoding is multiple
// bytes, not the single raw byte `b` -- storing it as a normal token string would silently corrupt every
// non-ASCII byte). `id_to_piece()` computes byte-range pieces arithmetically instead; only the 3 special
// tokens and the sentinels have real stored piece text.
class ByteVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it's present but not "byt5".
    static std::unique_ptr<ByteVocab> load(const GgufModel& model);

    // Maps `text`'s raw UTF-8 bytes 1:1 to ids (byte + 3 -- always succeeds, every byte value has a valid
    // id, so unk_id_ is never actually produced by real text), then appends eos_id_ (ByT5's own
    // `build_inputs_with_special_tokens` always does this unconditionally for a single sequence -- there
    // is no config flag to disable it, so this isn't gated by an "add_eos_token" KV like Vocab/BpeVocab).
    std::vector<int32_t> encode(const std::string& text) const;

    // Concatenates each id's piece bytes directly -- no marker-unescaping needed (unlike
    // Vocab/WordPieceVocab's SentencePiece-style "▁" convention; ByT5 has no such marker at all).
    std::string decode(const std::vector<int32_t>& ids) const;

    std::string id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t pad_id() const { return pad_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t unk_id() const { return unk_id_; }

    ByteVocab(const ByteVocab&) = delete;
    ByteVocab& operator=(const ByteVocab&) = delete;

private:
    ByteVocab() = default;

    // Full vocab_size-length array (3 special + 256 byte-range placeholders, never read + extra_ids
    // sentinels), matching every other vocab class's "one array indexed by id" convention -- byte-range
    // entries are unused empty placeholders (see class doc comment for why they can't hold real data).
    std::vector<std::string> tokens_;
    int32_t pad_id_ = 0;
    int32_t eos_id_ = 1;
    int32_t unk_id_ = 2;
    // Fixed by the "byt5" tag itself (ByT5Tokenizer hardcodes `_added_tokens_decoder = {0: pad, 1: eos,
    // 2: unk}`), not a per-checkpoint KV -- every real ByT5-family tokenizer has exactly this layout.
    static constexpr int32_t kByteOffset = 3;
    static constexpr int32_t kByteRangeSize = 256;
};

} // namespace loom
