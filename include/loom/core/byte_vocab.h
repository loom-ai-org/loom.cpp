#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class GgufModel;

// Raw-UTF-8-byte vocabulary, "tokenizer.ggml.model"=="byt5". A byte vocabulary has NO learned
// vocabulary, merges, or pretokenizer regex at all (unlike BpeVocab's GPT2-style byte-level BPE, which
// still learns merge rules over a "Ġ"-style byte-to-codepoint mapping): each raw byte `b` maps to a
// fixed id `b + byte_offset_`, and anything outside that range is an id with real stored piece text.
//
// **The tag still reads "byt5" although this now covers a second, differently-parameterised family
// (Dia's `DiaTokenizer`).** ByT5 is where the scheme was first met and every file written before this
// carries that string; a second spelling for one class would mean every future reader had to know the
// two were the same thing. What genuinely differs between checkpoints is carried by KVs instead, which
// is the honest place for it -- see `kByteOffsetKey`/`kAddEosKey` below, whose defaults are exactly
// ByT5's, so every pre-existing file loads and tokenizes byte-for-byte as it did.
//
// The two parameterisations this covers today, to make the axes concrete:
//
//                      | ByT5                        | Dia
//   byte_offset        | 3 (pad/eos/unk precede)     | 0 (bytes start at id 0)
//   append eos         | yes, unconditionally        | no
//   extra ids          | "<extra_id_N>" above 258    | "[S1]"/"[S2]" INSIDE the byte range, at 1/2
//   vocab size         | 259 + extra_ids             | 256
//
// Dia's row is the one that forced the added-token machinery below rather than merely a constant: its
// speaker tags occupy ids 1 and 2, which are byte values under its own offset, so an id's stored piece
// text has to be able to WIN over the byte-range arithmetic, and `"[S1] Hello"` has to encode that tag
// atomically instead of as five literal bytes.
//
// Byte-range piece text is deliberately NOT stored in `tokens_` (unlike every other id in this schema)
// -- a raw byte >= 0x80 can't round-trip through a GGUF string array the way the other vocab types'
// pieces do (Python's `chr(b)` for b>=128 produces a Unicode codepoint whose OWN UTF-8 encoding is
// multiple bytes, not the single raw byte `b` -- storing it as a normal token string would silently
// corrupt every non-ASCII byte). `id_to_piece()` computes byte-range pieces arithmetically instead;
// only the special tokens and the added ones have real stored piece text.
class ByteVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it's present but not "byt5".
    static std::unique_ptr<ByteVocab> load(const GgufModel& model);

    // ADDED TOKENS FIRST, on the raw bytes, then every remaining byte 1:1 to `byte + byte_offset_`.
    // The added-token scan is `BpeVocab::encode`'s, for the same reason and with the same longest-match
    // rule (P4.23): HF's `AddedVocabulary` splits the input on the added set before anything else sees
    // it, so `"[S1] Hello"` is one id and then five bytes, not ten bytes. A file declaring no added
    // tokens skips the scan entirely and encodes exactly as this class did before they existed.
    //
    // Byte mapping always succeeds -- every byte value has a valid id, so unk_id_ is never actually
    // produced by real text. eos_id_ is appended only when the file asks for it: ByT5's own
    // `build_inputs_with_special_tokens` always does this for a single sequence, and Dia's tokenizer
    // has no eos to append at all.
    std::vector<int32_t> encode(const std::string& text) const;

    // Concatenates each id's piece bytes directly -- no marker-unescaping needed (unlike
    // Vocab/WordPieceVocab's SentencePiece-style "▁" convention; a byte vocabulary has no such marker).
    std::string decode(const std::vector<int32_t>& ids) const;

    std::string id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t pad_id() const { return pad_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t unk_id() const { return unk_id_; }
    int32_t byte_offset() const { return byte_offset_; }
    bool adds_eos() const { return add_eos_; }

    ByteVocab(const ByteVocab&) = delete;
    ByteVocab& operator=(const ByteVocab&) = delete;

private:
    ByteVocab() = default;

    // The id of the longest added token whose spelling starts at `pos` in `text`, or -1 when none
    // does; `len` receives that spelling's byte length on a match. `BpeVocab::added_token_at`'s twin.
    int32_t added_token_at(const std::string& text, size_t pos, size_t* len) const;

    // Full vocab_size-length array (specials + byte-range placeholders, never read for the byte range
    // unless an added token overlaps it + any sentinels), matching every other vocab class's "one array
    // indexed by id" convention.
    std::vector<std::string> tokens_;
    int32_t pad_id_ = 0;
    int32_t eos_id_ = 1;
    int32_t unk_id_ = 2;
    // Where byte 0 lives. ByT5's 3 is the default, so a file that predates this KV is unchanged.
    int32_t byte_offset_ = kDefaultByteOffset;
    // Whether `encode` appends eos. True by default, which is ByT5's unconditional behaviour.
    bool add_eos_ = true;

    // The ADDED tokens, populated from "tokenizer.ggml.token_type" exactly as `BpeVocab` does: every
    // CONTROL and USER_DEFINED entry with non-empty piece text. Empty for a file without that KV, which
    // is every ByT5 export written before Dia and which therefore behaves identically.
    std::unordered_map<std::string, int32_t> added_to_id_;
    size_t max_added_len_ = 0;
    // Which byte values any added token can START with, so the scan skips most positions with one
    // array read instead of `max_added_len_` hash lookups.
    std::array<bool, 256> added_first_byte_{};
    // Ids whose stored piece text wins over the byte-range arithmetic in `id_to_piece`. Only ever
    // non-empty where an added token falls INSIDE the byte range, which is Dia's `[S1]`/`[S2]` and
    // nothing in ByT5.
    std::unordered_map<int32_t, std::string> added_piece_;

    static constexpr int32_t kDefaultByteOffset = 3;
    static constexpr int32_t kByteRangeSize = 256;
};

} // namespace loom
