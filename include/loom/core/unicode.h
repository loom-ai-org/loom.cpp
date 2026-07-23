#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace loom {

// Small Unicode subsystem backing BpeVocab's pretokenizer + normalizer -- NOT a general-purpose Unicode
// library, just the two primitives a real byte-level-BPE tokenizer.json needs: `\p{L}`/`\p{N}` character
// classification (for the fixed Qwen2/Qwen3 pretokenizer regex, hand-scanned rather than run through a
// regex engine -- see bpe_vocab.h) and NFC normalization (the tokenizer.json's declared `normalizer`).
// All lookup tables are generated (include/loom/core/unicode_data.h, see
// tools/codegen/gen_unicode_tables.py) from Python's stdlib `unicodedata` plus the Unicode Character
// Database's own CompositionExclusions.txt -- this header/its .cpp hand-implement the UAX #15 canonical
// decomposition/ordering/composition algorithm against those tables, not ported from any other
// engine's source.

// Decodes a UTF-8 string into Unicode scalar values (codepoints). Malformed sequences are lenient: an
// invalid/truncated byte is passed through as its own codepoint (never throws) -- real tokenizer prompts
// are valid UTF-8; this is a defensive fallback, not a validator.
std::vector<char32_t> utf8_decode(const std::string& text);

// Encodes codepoints back to UTF-8.
std::string utf8_encode(const std::vector<char32_t>& codepoints);

// True if `cp`'s Unicode general category is a Letter (L*) major category -- `\p{L}`.
bool is_letter(char32_t cp);

// True if `cp`'s Unicode general category is a Number (N*) major category -- `\p{N}`.
bool is_number(char32_t cp);

// is_letter(cp) || is_number(cp) -- the combined class the pretokenizer regex's `[^\r\n\p{L}\p{N}]`
// (and similar negated-class) alternatives need.
bool is_letter_or_number(char32_t cp);

// True if `cp`'s Unicode general category is a Punctuation (P*) major category -- `\p{P}`. Needed by
// WordPieceVocab's word-splitting (isolates punctuation into single-character words) and by BPE
// pretokenizer shapes that exclude punctuation explicitly.
bool is_punctuation(char32_t cp);

// True if `cp`'s Unicode general category is a Mark (M*) major category -- `\p{M}` (combining marks).
// Needed by WordPieceVocab's accent-stripping and by the qwen35-family BPE pretokenizer shape, whose
// regex attaches marks to the preceding letter run (`[\p{L}\p{M}]+`) rather than splitting on them.
bool is_mark(char32_t cp);

// Single-codepoint lowercase mapping (loom::unicode_data::kLowercaseMap), identity if `cp` has no
// mapping or its lowercase form isn't a single codepoint. Mirrors llama.cpp's own `unicode_tolower`
// scope -- a table lookup, not a general case-folding algorithm.
char32_t to_lower(char32_t cp);

// Unicode Normalization Form C (canonical decomposition followed by canonical composition, UAX #15).
// Operates on a full UTF-8 string end to end (decode -> decompose -> canonically order -> compose ->
// encode); a no-op for already-NFC input, which is the overwhelming majority of real-world UTF-8 text.
std::string nfc_normalize(const std::string& text);

// Unicode Normalization Form D (canonical decomposition + canonical ordering, no recomposition step) --
// the same first two passes nfc_normalize() already performs, exposed standalone since WordPieceVocab's
// accent-stripping needs the decomposed form (base letter + separate combining marks) so it can drop the
// marks, not the recomposed one.
std::string nfd_normalize(const std::string& text);

} // namespace loom
