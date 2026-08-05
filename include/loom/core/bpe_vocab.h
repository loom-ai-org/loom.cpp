#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace loom {

class GgufModel;

// The pretokenizer regex "shape" a given `tokenizer.ggml.pre` name maps to -- see bpe_vocab.cpp's
// `pre_spec_table()` for the full name->shape mapping, verified directly against llama.cpp's
// `llm_tokenizer_bpe`'s `regex_exprs` switch (src/llama-vocab.cpp). Every real HF BPE tokenizer.json's
// pretokenizer regex collapses into a small number of *shapes* -- most of the ~74 pre-tokenizer names
// llama.cpp recognizes are aliases sharing one of a handful of literal regex-alternative sets, not 74
// bespoke patterns. Only the shapes below are implemented; anything else is a bounded, later addition
// (see pre_spec_table()'s own comment for the full "not yet" list and why each is harder: CJK-script
// splitters, case-transition/camelCase shapes, and the "byte_encode=false" SPM-style-BPE family all need
// more than a new regex, they need a different symbol-initialization step in encode() itself).
enum class BpeShape {
    // The original 7-alternative regex this project already implemented for Qwen2/Qwen3/LFM2:
    // (?:'[sS]|'[tT]|...)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,K}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|
    // \s+(?!\S)|\s+ -- parameterized by the digit-run bound K (`max_number_run_`) and, for qwen35, by
    // whether \p{M} (combining marks) attaches to the letter run / is excluded from the punct run
    // (`include_marks_`).
    kQwenLlama3,
    // The "classic" GPT-2 4-alternative regex (starcoder/gpt-2/mpt/olmo/... families): same alternative
    // ORDER but case-SENSITIVE contraction literals, no `[^\r\n\p{L}\p{N}]?` prefix on the letter run (just
    // an optional leading space), no `[\r\n]*` suffix-absorption on the punct run, and no `\s*[\r\n]+`
    // alternative. The declared regex text has no bare `\s+` fallback either (only `\s+(?!\S)`), but
    // llama.cpp's own dedicated custom scanner for this exact shape (unicode_regex_split_custom_gpt2,
    // confirmed directly against src/unicode.cpp) DOES fall back to unconditionally consuming any
    // remaining whitespace run when the `(?!\S)` lookahead can't be satisfied (a lone whitespace character
    // immediately followed by non-whitespace) -- so this shape still needs match_ws_fallback as a final
    // alternative, same as kQwenLlama3, or a lone space before a word would throw instead of tokenizing.
    // Also parameterized by `max_number_run_` (0 == unbounded `\p{N}+`, matching
    // gpt-2/mpt/olmo/jais/trillion/granite-docling; 1 == single-digit, matching
    // starcoder/refact/command-r/smollm/codeshell/exaone/minerva-7b/mellum2, whose regex_exprs splits a
    // leading bare `\p{N}` alternative ahead of the main pattern's `\p{N}+`, which in practice means a
    // digit never gets the chance to join a multi-digit run).
    kGpt2Classic,
    // The single-alternative " ?[^(\s|.,!?…。，、।۔،)]+" pattern (poro-chat/bloom/gpt3-finnish/viking --
    // note the parenthesized class contents are LITERAL characters in a `[...]` char class, not regex
    // metacharacters). `split_leading_digit_` additionally tries a single-digit alternative first (viking
    // only, whose regex_exprs has an extra bare `\p{N}` entry ahead of this pattern).
    kWhitespacePunctExclude,
    // SentencePiece-style BPE with byte fallback (Gemma 3, and llama.cpp's
    // `granite-embed-multi-311m` family, whose chkhsh it shares). Structurally unlike every shape
    // above, which is why it needed more than a new regex:
    //
    //   * **no regex pretokenization at all** -- the whole text is one chunk;
    //   * **no GPT-2 byte-level mapping** -- the vocabulary holds literal UTF-8, so the initial BPE
    //     symbols are CHARACTERS, not byte-mapped stand-ins (confirmed on the real vocab: its merges
    //     contain actual U+2581 and real spaces, never `Ġ`);
    //   * a normalizer that replaces every space with U+2581 (`▁`), and no dummy prefix -- HF gives
    //     `"Hello world"` -> `['Hello', '▁world']`, the first word bare;
    //   * **byte fallback**: a character with no vocab entry becomes its UTF-8 bytes as `<0xNN>`
    //     tokens, rather than being an error as it is for a byte-level vocab where every byte maps.
    //
    // `max_number_run_`/`include_marks_` are unused here; digits come out one per token because the
    // vocab simply has no multi-digit merges, not because anything splits them.
    kSpmByteFallback,
};

// Byte-level BPE vocabulary loaded from a GGUF's "tokenizer.ggml.*" KVs, llama.cpp's own "gpt2" schema
// (distinct from Vocab's SentencePiece-unigram "t5" schema -- see vocab.h's doc comment, which already
// reserves this exact split: "BPE's byte-to-'Ġ' convention... decode/encode differently"). Covers a
// bounded set of real HF tokenizer.json pretokenizer regex "shapes" (see BpeShape) -- NFC normalize -> a
// hand-scanned Unicode-category-aware regex split -> GPT2 byte-level mapping -> greedy BPE merge -- not a
// general framework for arbitrary tokenizer.json pretokenizer configurations. `tokenizer.ggml.pre` selects
// both the shape and its parameters via `pre_spec_table()` (bpe_vocab.cpp), llama.cpp's own convention for
// exactly this kind of per-model pretokenizer variant.
class BpeVocab {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or if it's present but not "gpt2"
    // (i.e. this model uses Vocab's SentencePiece-unigram schema instead -- callers should try both).
    // Throws loom::LoadError if "tokenizer.ggml.pre" is present but names a pretokenizer family
    // `pre_spec_table()` doesn't implement (fail loud rather than silently mis-tokenizing) -- absent
    // entirely, it still defaults to "qwen2" unchanged, same as before this family registry existed.
    static std::unique_ptr<BpeVocab> load(const GgufModel& model);

    // NFC-normalizes `text` (loom::nfc_normalize), splits it via this vocab's `BpeShape` pretokenizer
    // regex (hand-scanned against loom::is_letter/is_number/is_mark -- see bpe_vocab.cpp), GPT2
    // byte-level-maps each chunk's raw UTF-8 bytes, and greedily BPE-merges each chunk independently
    // (merges never cross a pretokenizer chunk boundary, matching the reference tokenizer exactly).
    // Prepends `bos_id_` first when "tokenizer.ggml.add_bos_token" is true, appends `sep_id_` last when
    // "tokenizer.ggml.add_sep_token" is true (both mirror llama.cpp's own convention; absent, both default
    // to false -- unchanged behavior for existing GGUFs that never wrote them).
    std::vector<int32_t> encode(const std::string& text) const;

    // Joins each id's piece text and reverses the GPT2 byte-level mapping back to raw UTF-8 bytes.
    std::string decode(const std::vector<int32_t>& ids) const;

    const std::string& id_to_piece(int32_t id) const;
    size_t size() const { return tokens_.size(); }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t sep_id() const { return sep_id_; }

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
    int32_t sep_id_ = -1;
    bool add_bos_token_ = false;
    bool add_sep_token_ = false;
    BpeShape shape_ = BpeShape::kQwenLlama3;
    size_t max_number_run_ = 1; // meaning is per-shape -- see BpeShape's own doc comment
    bool include_marks_ = false; // qwen35 only -- \p{M} attaches to the letter run / punct exclusion
};

} // namespace loom
