#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;

// BERT-style WordPiece vocabulary, "tokenizer.ggml.model"=="bert" (a new tag this project introduces --
// llama.cpp instead tags this vocab type via its own internal LLAMA_VOCAB_TYPE_WPM enum rather than a
// tokenizer.ggml.model string; "bert" follows the SAME per-family-tag convention every other vocab type in
// this schema already uses -- "t5"/"llama" (Vocab), "gpt2" (BpeVocab) -- rather than inventing a different
// dispatch mechanism for just this one).
//
// Algorithm ported natively from llama.cpp's llm_tokenizer_wpm_session (src/llama-vocab.cpp): preprocess()
// NFD-decomposes + strips combining marks (if strip_accents) + lowercases (if lowercase) the input, and
// splits it into "words" isolating punctuation/ASCII-symbols/CJK characters into single-character words of
// their own; each word gets prefixed with U+2581 ("▁") and greedily longest-match-first matched against
// the vocab trie from position 0, falling back to a single [UNK] for the whole word on total failure.
//
// CLS/SEP reuse the generic BOS/SEP KVs rather than dedicated CLS/SEP-only concepts -- matches llama.cpp's
// own WPM convention (its vocab type defaults add_bos=true, add_sep=true, with no separate CLS KV at all).
//
// Known first-pass limitation: only ASCII whitespace is treated as a word boundary (matches BpeVocab's
// existing ASCII-only whitespace-scanning convention elsewhere in this engine) -- genuine Unicode
// whitespace (NBSP, ideographic space) as a word-splitter is deferred until a real fixture needs it.
class WordPieceVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it's present but not "bert" (i.e.
    // this model uses BpeVocab's or Vocab's schema instead -- callers should try those too).
    static std::unique_ptr<WordPieceVocab> load(const GgufModel& model);

    // Prepends cls_id_ (read from the generic "tokenizer.ggml.bos_token_id" KV, reused as CLS) when
    // "tokenizer.ggml.add_bos_token" is true; appends sep_id_ ("tokenizer.ggml.seperator_token_id") when
    // "tokenizer.ggml.add_sep_token" is true.
    std::vector<int32_t> encode(const std::string& text) const;

    // Joins each id's piece text and unescapes the SentencePiece-style word-boundary marker (U+2581 "▁")
    // back to a literal space -- the exporter's own `phantom()` transform (mirroring llama.cpp's
    // conversion/bert.py) already rewrites "##continuation" pieces to have NO marker at all (so they glue
    // directly onto the preceding piece) and non-continuation pieces to be ▁-prefixed instead, so plain
    // concatenation + unescape reconstructs the original spacing with no extra "##" bookkeeping needed
    // here (identical to loom::Vocab::decode's own convention).
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t cls_id() const { return cls_id_; }
    int32_t sep_id() const { return sep_id_; }
    int32_t pad_id() const { return pad_id_; }
    int32_t mask_id() const { return mask_id_; }
    int32_t unk_id() const { return unk_id_; }

    WordPieceVocab(const WordPieceVocab&) = delete;
    WordPieceVocab& operator=(const WordPieceVocab&) = delete;

private:
    WordPieceVocab() = default;

    // Mirrors loom::Vocab's naive_trie exactly (std::map<char,...>, not unordered_map), same convention.
    struct TrieNode {
        std::map<char, TrieNode> children;
        bool has_value = false;
        int32_t value = 0;
    };
    void trie_insert(const std::string& key, int32_t value);
    bool trie_find(const std::string& key, int32_t* id) const;

    std::vector<std::string> preprocess(const std::string& text) const;
    std::vector<int32_t> tokenize_word(const std::string& word) const;
    static bool is_chinese_char(char32_t cp);

    std::vector<std::string> tokens_;
    TrieNode token_trie_;
    size_t max_token_len_ = 0;
    int32_t unk_id_ = 0;
    int32_t cls_id_ = -1;
    int32_t sep_id_ = -1;
    int32_t pad_id_ = -1;
    int32_t mask_id_ = -1;
    bool add_bos_token_ = false;
    bool add_sep_token_ = false;
    bool lowercase_ = true;
    bool strip_accents_ = true;
};

} // namespace loom
