#!/usr/bin/env python3
"""One-off generator for include/loom/core/unicode_data.h -- NOT part of the CMake build and not run at
model-conversion time. The C++ engine has no runtime Python dependency; this script's only job is to
turn Python's stdlib `unicodedata` module (backed by the Unicode Character Database, UCD) plus the UCD's
own published CompositionExclusions.txt into a checked-in, static C++ header, exactly once. Re-run only
if intentionally bumping the target Unicode version.

Emits four tables, all `loom::unicode` has no other way to obtain (Python's unicodedata doesn't expose
the exclusion set at all, and doesn't list algorithmic Hangul decompositions -- see below):

1. `kLetterRanges`/`kNumberRanges`: sorted, merged [lo, hi] codepoint ranges whose general category is
   L*/N* respectively (Unicode's Letter/Number major categories, kept as two separate tables since the
   pretokenizer regex uses `\\p{L}` and `\\p{N}` in distinct positions, not just combined) -- used by
   BpeVocab's hand-written pretokenizer scanner to evaluate the real Qwen2/Qwen3 tokenizer.json regex (no
   general-purpose Unicode-aware regex engine is linked in; these tables are what makes a hand-written
   scanner correct instead of an ASCII-only approximation).
2. `kCanonicalDecomp`: codepoint -> decomposition sequence, canonical decompositions ONLY (multi-codepoint
   <tag> compatibility decompositions from `unicodedata.decomposition()`, e.g. "<font> 0041", are
   filtered out -- NFC must never apply those). Hangul syllables (U+AC00..U+D7A3) are deliberately absent
   here: UnicodeData.txt (and thus Python's unicodedata) omits their ~11172 decompositions on purpose,
   since they're fully specified by a closed-form arithmetic formula (UAX #15 section 16) -- loom's
   nfc_normalize() computes those directly instead of consulting this table.
3. `kCombiningClass`: sparse codepoint -> canonical_combining_class, non-zero entries only (needed for
   NFC's canonical-ordering step before composition).
4. `kCompositionExclusions`: sorted codepoint set. NOT derivable from `unicodedata` at all -- fetched
   from the UCD's own normative CompositionExclusions.txt. Skipping this table would make composition
   silently WRONG for the characters it lists (e.g. it would wrongly recompose some sequences NFC must
   leave decomposed), not just incomplete.

Usage: python3 gen_unicode_tables.py > include/loom/core/unicode_data.h
"""
import sys
import unicodedata
import urllib.request

UNICODE_VERSION = unicodedata.unidata_version
EXCLUSIONS_URL = f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucd/CompositionExclusions.txt"
MAX_CODEPOINT = 0x110000  # exclusive


def fetch_composition_exclusions() -> list[int]:
    with urllib.request.urlopen(EXCLUSIONS_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    exclusions = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        exclusions.append(int(line, 16))
    return sorted(exclusions)


def compute_category_ranges(major_category: str) -> list[tuple[int, int]]:
    """Range-compresses every codepoint whose general category starts with `major_category` ("L" or
    "N") -- kept as two separate tables (not one combined L-or-N table) because the pretokenizer regex
    uses \\p{L} and \\p{N} in distinct positions (e.g. `\\p{N}` alone, without `+`, must NOT match a
    letter) as well as together (`[^\\r\\n\\p{L}\\p{N}]`, matched via is_letter(c) || is_number(c))."""
    ranges: list[tuple[int, int]] = []
    run_start = None
    for cp in range(MAX_CODEPOINT):
        matches = unicodedata.category(chr(cp))[0] == major_category
        if matches and run_start is None:
            run_start = cp
        elif not matches and run_start is not None:
            ranges.append((run_start, cp - 1))
            run_start = None
    if run_start is not None:
        ranges.append((run_start, MAX_CODEPOINT - 1))
    return ranges


HANGUL_SYLLABLE_START = 0xAC00
HANGUL_SYLLABLE_END = 0xD7A3


def compute_canonical_decomp() -> dict[int, list[int]]:
    decomp: dict[int, list[int]] = {}
    for cp in range(MAX_CODEPOINT):
        if HANGUL_SYLLABLE_START <= cp <= HANGUL_SYLLABLE_END:
            continue  # algorithmic, handled in C++, deliberately excluded (see module docstring)
        d = unicodedata.decomposition(chr(cp))
        if not d or d.startswith("<"):
            continue  # empty (no decomposition) or a compatibility decomposition -- skip both
        decomp[cp] = [int(tok, 16) for tok in d.split()]
    return decomp


def compute_combining_class() -> dict[int, int]:
    cls: dict[int, int] = {}
    for cp in range(MAX_CODEPOINT):
        c = unicodedata.combining(chr(cp))
        if c != 0:
            cls[cp] = c
    return cls


def emit_header(letter_ranges, number_ranges, canonical_decomp, combining_class, exclusions) -> str:
    lines = []
    lines.append("// GENERATED FILE -- do not hand-edit. Produced by tools/codegen/gen_unicode_tables.py")
    lines.append(f"// against Python's stdlib `unicodedata` (Unicode Character Database version {UNICODE_VERSION})")
    lines.append(f"// plus {EXCLUSIONS_URL}. Re-run the script to regenerate against a newer Unicode version.")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace loom {")
    lines.append("namespace unicode_data {")
    lines.append("")
    lines.append("struct CpRange { char32_t lo; char32_t hi; };")
    lines.append("struct CpEntry { char32_t cp; uint8_t value; };")
    lines.append("struct CpDecomp { char32_t cp; const char32_t* seq; uint8_t seq_len; };")
    lines.append("")

    # 1. Letter ranges and Number ranges (kept separate -- see compute_category_ranges's docstring)
    lines.append(f"// {len(letter_ranges)} ranges, general category L* (\\p{{L}}).")
    lines.append(f"inline constexpr CpRange kLetterRanges[] = {{")
    for lo, hi in letter_ranges:
        lines.append(f"    {{0x{lo:06X}, 0x{hi:06X}}},")
    lines.append("};")
    lines.append(f"inline constexpr size_t kLetterRangesCount = {len(letter_ranges)};")
    lines.append("")
    lines.append(f"// {len(number_ranges)} ranges, general category N* (\\p{{N}}).")
    lines.append(f"inline constexpr CpRange kNumberRanges[] = {{")
    for lo, hi in number_ranges:
        lines.append(f"    {{0x{lo:06X}, 0x{hi:06X}}},")
    lines.append("};")
    lines.append(f"inline constexpr size_t kNumberRangesCount = {len(number_ranges)};")
    lines.append("")

    # 2. Canonical decompositions -- flatten all sequences into one backing array, entries point into it.
    flat_seq: list[int] = []
    decomp_entries = []
    for cp in sorted(canonical_decomp.keys()):
        seq = canonical_decomp[cp]
        offset = len(flat_seq)
        flat_seq.extend(seq)
        decomp_entries.append((cp, offset, len(seq)))

    lines.append(f"// {len(flat_seq)} codepoints across {len(decomp_entries)} canonical decompositions"
                  " (Hangul syllables excluded -- computed algorithmically, see unicode.cpp).")
    lines.append("inline constexpr char32_t kCanonicalDecompSeq[] = {")
    for i in range(0, len(flat_seq), 12):
        chunk = flat_seq[i:i + 12]
        lines.append("    " + ", ".join(f"0x{c:06X}" for c in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append("struct CpDecompEntry { char32_t cp; uint32_t offset; uint8_t len; };")
    lines.append(f"inline constexpr CpDecompEntry kCanonicalDecomp[] = {{")
    for cp, offset, length in decomp_entries:
        lines.append(f"    {{0x{cp:06X}, {offset}, {length}}},")
    lines.append("};")
    lines.append(f"inline constexpr size_t kCanonicalDecompCount = {len(decomp_entries)};")
    lines.append("")

    # 3. Combining class
    lines.append(f"// {len(combining_class)} codepoints with non-zero canonical combining class.")
    lines.append("inline constexpr CpEntry kCombiningClass[] = {")
    for cp in sorted(combining_class.keys()):
        lines.append(f"    {{0x{cp:06X}, {combining_class[cp]}}},")
    lines.append("};")
    lines.append(f"inline constexpr size_t kCombiningClassCount = {len(combining_class)};")
    lines.append("")

    # 4. Composition exclusions
    lines.append(f"// {len(exclusions)} codepoints, from CompositionExclusions.txt (NOT derivable from"
                  " unicodedata alone).")
    lines.append("inline constexpr char32_t kCompositionExclusions[] = {")
    for i in range(0, len(exclusions), 12):
        chunk = exclusions[i:i + 12]
        lines.append("    " + ", ".join(f"0x{c:06X}" for c in chunk) + ",")
    lines.append("};")
    lines.append(f"inline constexpr size_t kCompositionExclusionsCount = {len(exclusions)};")
    lines.append("")
    lines.append("} // namespace unicode_data")
    lines.append("} // namespace loom")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    sys.stderr.write(f"Unicode version: {UNICODE_VERSION}\n")
    sys.stderr.write("Fetching composition exclusions...\n")
    exclusions = fetch_composition_exclusions()
    sys.stderr.write(f"  {len(exclusions)} exclusions\n")
    sys.stderr.write("Computing letter ranges...\n")
    letter_ranges = compute_category_ranges("L")
    sys.stderr.write(f"  {len(letter_ranges)} ranges\n")
    sys.stderr.write("Computing number ranges...\n")
    number_ranges = compute_category_ranges("N")
    sys.stderr.write(f"  {len(number_ranges)} ranges\n")
    sys.stderr.write("Computing canonical decompositions...\n")
    decomp = compute_canonical_decomp()
    sys.stderr.write(f"  {len(decomp)} entries\n")
    sys.stderr.write("Computing combining classes...\n")
    comb = compute_combining_class()
    sys.stderr.write(f"  {len(comb)} entries\n")
    sys.stdout.write(emit_header(letter_ranges, number_ranges, decomp, comb, exclusions))


if __name__ == "__main__":
    main()
