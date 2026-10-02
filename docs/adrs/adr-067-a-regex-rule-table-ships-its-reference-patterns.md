---
type: adr
status: accepted
date: 2026-10-02
tags: [engine, tokenizer, text-front-end, regex, soprano, family-9]
supersedes: []
---

# ADR-067: A Regex Rule Table Ships the Reference's Own Patterns

## Context

Soprano TTS runs tortoise-tts's English normaliser before any id reaches the model: `clean_text`, about
45 Python regexes applied in order -- dates, phones, times, money, decimals, arithmetic, ordinals,
inflect's `number_to_words`, abbreviations, CamelCase, symbols, punctuation clean-up -- then a
quote-aware sentence splitter. [ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md) says a front
end's rules ship as data and its shape as code. Every front end so far kept that split with
hand-written scanners, one per rule, because their rules were string tables (CosyVoice3's replacements,
Pocket-TTS's terminals). These rules are regexes with backtracking, alternation and back references.

## Options

1. **A hand-written scanner per pattern.** Forty-odd re-derivations of Python's semantics, each a place
   to be subtly wrong (leftmost-first alternation, greedy backtracking, `\b`, IGNORECASE), and a C++
   change for every rule the reference adds.
2. **`std::regex`.** ECMAScript semantics, not Python's (`\s` misses `\x1c`-`\x1f`, among others), slow,
   and not used anywhere in the engine.
3. **A small matcher for the subset of Python `re` these patterns use, and the pattern STRINGS in the
   file.** `loom::PyRegex`: literals, classes with ranges and escapes, `\d \w \s \b \A \Z ^ $ .`, groups,
   alternation, greedy quantifiers including `{n,m}` and 3.11's `{,m}`, back references, IGNORECASE --
   and a load error for anything else, so a newer reference fails at load instead of matching wrongly.

## Decision

Option 3. `tokenizer.ggml.soprano.*` carries the reference's compiled patterns verbatim: module-level
regexes read as attributes, patterns written INLINE in function bodies checked against
`inspect.getsource` at export (a changed reference fails the export), inflect's word tables and its
clean-up patterns, `unidecode`'s table (41,379 codepoints, packed). The C++ holds the pipeline's order and
each callback's shape -- including `_expand_time`'s f-string slip that renders minutes as a Python set.
A quantified single-character atom iterates rather than recurses, so a long whitespace run or a `.*`
over a paragraph costs no stack per character.

## Consequences

* Exact: `PyRegex` against Python's `re` on all 115 patterns, 92,000/92,000 identical; `SopranoVocab`
  against `_preprocess_text` + the HF tokenizer, 20,000/20,000 over ten input classes, the reference's
  crashes compared as refusals; four sabotaged references red (883, 891, 494 and 13 of 2000 differ).
* Classes are Python's on ASCII; outside it `\w` is `\p{L}\p{N}` and `\d` is ASCII only. Soprano's text is
  ASCII by its first class (`unidecode` runs first), so neither gap is reached; both are pinned in
  `tests/ci/test_py_regex.cpp`.
* The next front end written as Python regexes costs a writer and a pipeline, not a matcher.
* Needs an engine with this (1.0.0-rc14).

## Related

* [ADR-041](adr-041-a-text-front-ends-rules-ship-as-data.md), [ADR-044](adr-044-a-front-end-that-chunks-returns-its-chunks-in-the-ids.md)
* `loom-exporter/loom_exporter/soprano_tokenizer_export.py`: the writer
