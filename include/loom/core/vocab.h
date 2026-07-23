#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;

// A SentencePiece vocabulary loaded from a GGUF's "tokenizer.ggml.*" KVs (llama.cpp's own schema --
// see gguf-py's Keys.Tokenizer and include/llama.h's llama_vocab_type/llama_token_type). Mirrors
// llama.cpp's own tokenizer designs closely, per the project's decision to match llama.cpp's approach
// where applicable: same XCDA-based precompiled_charsmap normalizer, same naive_trie-style token
// matcher, for BOTH vocab types below (the normalizer/trie/decode side is identical between them --
// confirmed by inspecting a real SentencePiece BPE model's ModelProto directly, not assumed -- only the
// encode *algorithm* differs).
//
// Two SentencePiece vocab types are implemented, selected by "tokenizer.ggml.model":
//   - "t5" -- UGM (unigram): Viterbi best-path segmentation using each piece's score directly as a
//     log-probability (see encode()).
//   - "llama" -- real SentencePiece BPE (llama.cpp's own tag for this -- the original LLaMA/Mistral
//     tokenizers are themselves SentencePiece BPE models, confirmed via gguf-py's own
//     SentencePieceVocab class, which writes this exact tag): greedy merge-by-score (see encode_bpe()).
//
// Still not implemented: GPT2-style byte-level BPE's byte-to-"Ġ" convention (that's loom::BpeVocab,
// a wholly separate class/schema tag -- "gpt2", used by Qwen3) and WPM's "##"-continuation convention
// (not needed by any model this engine has converted so far, see BACKLOG.md).
class Vocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV (no vocab present). Throws
    // loom::LoadError if the KV is present but neither "t5" (UGM) nor "llama" (SentencePiece BPE), or if
    // required array KVs are missing/malformed.
    static std::unique_ptr<Vocab> load(const GgufModel& model);

    // Normalizes `text` (via the XCDA-based charsmap walk, matching SentencePiece's own normalizer),
    // then segments it into vocab piece ids -- via Viterbi best-path search (UGM) or greedy
    // merge-by-score (SentencePiece BPE), selected by which "tokenizer.ggml.model" this Vocab was
    // loaded from. Falls back to unk_id() for any codepoint no vocab piece covers. Prepends `bos_id_`
    // first when "tokenizer.ggml.add_bos_token" is true, appends `eos_id_` last when
    // "tokenizer.ggml.add_eos_token" is true (mirrors BpeVocab's identical convention; absent, both
    // default to false) -- closes the gap that otherwise blocks ALBERT/XLNet-style Unigram models, which
    // wrap sequences via SentencePiece's own BOS/EOS convention rather than a separate CLS/SEP concept.
    std::vector<int32_t> encode(const std::string& text) const;

    // Joins each id's piece text, unescapes the SentencePiece word-boundary marker (U+2581 "▁") back
    // to a literal space, and strips a single leading space if add_space_prefix is set.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    int32_t unk_id() const { return unk_id_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    size_t size() const { return tokens_.size(); }

    Vocab(const Vocab&) = delete;
    Vocab& operator=(const Vocab&) = delete;

private:
    Vocab() = default;

    // Mirrors llama.cpp's naive_trie exactly (std::map<char,...>, not unordered_map) per the project's
    // decision to match llama.cpp's design where applicable.
    struct TrieNode {
        std::map<char, TrieNode> children;
        bool has_value = false;
        int32_t value = 0;
    };
    void trie_insert(const std::string& key, int32_t value);

    // XCDA (XOR-compressed compact double array) walk over precompiled_charsmap_, mirroring llama.cpp's
    // xcda_array_view::get_base/get_lcheck/get_leaf/get_value bit-unpacking exactly.
    uint32_t xcda_node(size_t index) const;
    uint32_t xcda_base(size_t index) const;
    uint32_t xcda_lcheck(size_t index) const;
    bool xcda_leaf(size_t index) const;
    uint32_t xcda_value(size_t index) const;

    // Returns (replacement text, input bytes consumed) for the longest charsmap match starting at
    // input[offset], or a pass-through of the next UTF-8 codepoint if no match exists.
    struct PrefixMatch {
        std::string replacement;
        size_t consumed = 0;
    };
    PrefixMatch normalize_prefix(const std::string& input, size_t offset) const;
    std::string normalize(const std::string& text) const;

    // Exact-match lookup of `key` in token_trie_ (not a prefix walk) -- used by encode_bpe() to test
    // whether merging two adjacent symbols would form a real vocab piece. Returns false if absent.
    bool trie_find(const std::string& key, int32_t* id) const;

    // Real SentencePiece BPE: split `normalized` into per-codepoint initial symbols, then repeatedly
    // merge the single highest-scoring adjacent pair whose concatenation is a real vocab piece (ties
    // broken leftmost, matching SentencePiece's own convention) until no more merges apply. Naive
    // O(n^2)-per-merge scan rather than SentencePiece's real priority-queue implementation -- correctness
    // over performance, same tradeoff as everywhere else in this engine; produces the identical result
    // since both are "always merge the single current-highest-priority valid pair" (verified against the
    // real `sentencepiece` library's own output before trusting this, not assumed equivalent on paper
    // alone). No byte-fallback (this engine hasn't needed a model with byte_fallback=true yet, see
    // BACKLOG.md) -- a symbol with no matching piece and no further merge falls back to unk_id().
    std::vector<int32_t> encode_bpe(const std::string& normalized) const;

    // The Viterbi/BPE segmentation itself, before encode()'s bos/eos prepend/append.
    std::vector<int32_t> encode_impl(const std::string& text) const;

    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<int32_t> token_type_; // llama_token_type values, 1:1 with SentencePiece's own enum
    int32_t unk_id_ = 0;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    bool add_bos_token_ = false; // default false -- unchanged behavior for existing GGUFs without this KV
    bool add_eos_token_ = false; // default false -- ditto (llama.cpp's own UGM *implicit* default with no
                                  // KV at all is true; loom follows its own "explicit KV, safe default"
                                  // convention instead, same as BpeVocab::add_bos_token_'s precedent)
    bool add_space_prefix_ = true;
    bool remove_extra_whitespaces_ = true;
    bool is_bpe_ = false; // true for "llama" (SentencePiece BPE); false for "t5" (UGM)
    TrieNode token_trie_;

    // Raw precompiled_charsmap: first 4 bytes = byte length of the XCDA blob (array of uint32_t
    // entries), remainder = a NUL-terminated prefix-replacement string table indexed by byte offset.
    std::vector<uint8_t> charsmap_;
    const uint32_t* xcda_array_ = nullptr;
    size_t xcda_array_size_ = 0;
    const char* prefix_replacements_ = nullptr;
    size_t prefix_replacements_size_ = 0;
};

} // namespace loom
