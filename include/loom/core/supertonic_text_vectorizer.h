#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace loom {

class GgufModel;

// Native port of SupertonicTTS v2's real `TextVectorizer` (models/modules/text_vectorizer.py) --
// NOT a general HF tokenizer family (unlike Vocab/BpeVocab/WordPieceVocab/ByteVocab), so it doesn't share
// their "tokenizer.ggml.*" schema or live in tools/loom_mil_compiler/'s generic exporter path; it's
// bespoke to this one model, alongside the rest of tools/convert_supertonic/'s own per-model conversion
// scripts. `tokenizer.ggml.model`=="supertonic" still follows the same per-family-tag convention those
// classes use.
//
// The real scheme is a direct BMP-codepoint -> small-vocab-id array lookup (real vocab_size=163, only 162
// codepoints actually mapped -- see supertonic_common.py's own `nn.Embedding(vocab_size=163, ...)`; the
// one spare row is unused headroom, not a distinct pad id -- the real Python `tokenize()`'s own batching
// helper pads with `nn.functional.pad`'s default value 0, which happens to coincide with id 0 (space), not
// a separate reserved pad token) -- confirmed directly against a real `unicode_indexer.json` asset (a
// flat 65536-entry array, -1 for unsupported codepoints) and the real preprocessing pipeline that produces
// the text fed into it (NFKD normalize, emoji stripping, a fixed character-replacement table, punctuation-
// spacing cleanup, quote deduplication, whitespace collapsing, trailing-punctuation insertion, and
// wrapping the result in `<lang>...</lang>` tags) -- every step ported natively here, verified against the
// real Python class's actual output on representative strings (see tests/test_supertonic_text_vectorizer.cpp),
// not assumed from reading the source alone.
//
// Known, deliberate simplification: uses `loom::nfd_normalize` (canonical decomposition only) rather than
// building a full NFKD (canonical + compatibility) decomposition table, mirroring this project's existing
// nfc_normalize/nfd_normalize precedent (which already excludes compatibility decompositions). This is a
// no-op difference for every character this vocabulary actually supports -- the accented Latin letters
// (es/pt/fr) and Hangul syllables (ko) this model needs all decompose CANONICALLY, and NFKD-only
// compatibility forms (ligatures, fullwidth forms, etc.) aren't in the 162-entry vocab either way, so they
// get dropped by the codepoint lookup regardless of which decomposition form is used.
class SupertonicTextVectorizer {
public:
    // Returns nullptr if `model` has no "tokenizer.ggml.model" KV, or it's present but not "supertonic".
    static std::unique_ptr<SupertonicTextVectorizer> load(const GgufModel& model);

    // Runs the full preprocessing pipeline (see class doc comment) on `text`, wraps it in
    // "<lang>...</lang>", then maps each resulting codepoint to its vocab id via the loaded lookup table
    // -- codepoints with no entry (id < 0, or codepoint >= the table's size) are silently dropped, exactly
    // matching the real `tokenize_str`'s own filter. Does NOT pad/truncate to a fixed length (matching
    // `tokenize_str`'s own unbatched, unpadded contract) -- `SupertonicDriver::synthesize()`'s own
    // `txt_len_fixed` requirement is the caller's responsibility, same as the real `tokenize()`'s own
    // separate padding/truncation step.
    std::vector<int32_t> tokenize(const std::string& text, const std::string& lang) const;

    size_t vocab_size() const { return table_.size(); }

    SupertonicTextVectorizer(const SupertonicTextVectorizer&) = delete;
    SupertonicTextVectorizer& operator=(const SupertonicTextVectorizer&) = delete;

private:
    SupertonicTextVectorizer() = default;

    std::string preprocess(const std::string& text, const std::string& lang) const;

    // table_[cp] = vocab id for codepoint `cp`, or -1 if unsupported. Indexed 0..0xFFFF (BMP only, same
    // as the real unicode_indexer.json).
    std::vector<int32_t> table_;
};

} // namespace loom
