#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;

// A SentencePiece-unigram vocabulary loaded from a GGUF's "tokenizer.ggml.*" KVs (llama.cpp's own
// schema -- see gguf-py's Keys.Tokenizer and include/llama.h's llama_vocab_type/llama_token_type).
// Mirrors llama.cpp's llm_tokenizer_ugm/llm_tokenizer_ugm_session design closely, per the project's
// decision to match llama.cpp's approach where applicable: same XCDA-based precompiled_charsmap
// normalizer, same naive_trie-style token matcher, same Viterbi best-path segmentation for encode.
//
// Only the UGM (unigram) vocab type is implemented -- SPM's byte-level-BPE-with-scores, BPE's
// byte-to-"Ġ" convention, and WPM's "##"-continuation convention all decode/encode differently and
// aren't needed by any model this engine has converted so far (see BACKLOG.md).
class Vocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV (no vocab present). Throws
    // loom::LoadError if the KV is present but not "t5" (UGM), or if required array KVs are missing/
    // malformed.
    static std::unique_ptr<Vocab> load(const GgufModel& model);

    // Unigram Viterbi tokenization: normalizes `text` (via the XCDA-based charsmap walk, matching
    // SentencePiece's own normalizer), then finds the highest-total-score segmentation into vocab
    // pieces. Falls back to unk_id() for any single codepoint no vocab piece covers.
    std::vector<int32_t> encode(const std::string& text) const;

    // Joins each id's piece text, unescapes the SentencePiece word-boundary marker (U+2581 "▁") back
    // to a literal space, and strips a single leading space if add_space_prefix is set.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    int32_t unk_id() const { return unk_id_; }
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

    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<int32_t> token_type_; // llama_token_type values, 1:1 with SentencePiece's own enum
    int32_t unk_id_ = 0;
    bool add_space_prefix_ = true;
    bool remove_extra_whitespaces_ = true;
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
