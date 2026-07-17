#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class GgufModel;

// Byte-level BPE vocabulary loaded from a GGUF's "tokenizer.ggml.*" KVs, llama.cpp's own "gpt2" schema
// (distinct from Vocab's SentencePiece-unigram "t5" schema -- see vocab.h's doc comment, which already
// reserves this exact split: "BPE's byte-to-'Ġ' convention... decode/encode differently"). Targets the
// specific, fixed Qwen2/Qwen3 tokenizer.json pretokenizer convention (NFC normalize -> a fixed
// Unicode-category-aware regex split -> GPT2 byte-level mapping -> greedy BPE merge), confirmed directly
// against the real tokenizer.json's `pre_tokenizer`/`normalizer`/`model` fields -- not a general BPE
// framework for arbitrary tokenizer.json pretokenizer configurations.
class BpeVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or if it's present but not "gpt2"
    // (i.e. this model uses Vocab's SentencePiece-unigram schema instead -- callers should try both).
    static std::unique_ptr<BpeVocab> load(const GgufModel& model);

    // NFC-normalizes `text` (loom::nfc_normalize), splits it via the fixed Qwen2-style pretokenizer
    // regex (hand-scanned against loom::is_letter/is_number -- see bpe_vocab.cpp), GPT2 byte-level-maps
    // each chunk's raw UTF-8 bytes, and greedily BPE-merges each chunk independently (merges never cross
    // a pretokenizer chunk boundary, matching the reference tokenizer exactly).
    std::vector<int32_t> encode(const std::string& text) const;

    // Joins each id's piece text and reverses the GPT2 byte-level mapping back to raw UTF-8 bytes.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }

    BpeVocab(const BpeVocab&) = delete;
    BpeVocab& operator=(const BpeVocab&) = delete;

private:
    BpeVocab() = default;

    std::vector<std::string> pretokenize(const std::string& nfc_text) const;
    void bpe_merge(std::vector<std::string>& pieces) const;

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_; // key: piece_a + '\x01' + piece_b
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
};

} // namespace loom
