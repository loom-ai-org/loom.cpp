---
type: adr
status: accepted
date: 2026-09-24
tags: [engine, tokenizer, text-front-end, family-9, chatterbox, model-coverage]
supersedes: []
---

# ADR-041: A Text Front End's Rules Ship as Data, and Only Its Shape Is Code

## Context

Chatterbox, family 9's fourth leaf, has a text path of three steps
(`ChatterboxTTS.generate`):

1. **`punc_norm`**: a stand-in sentence for empty input, the first character upper-cased with
   `str.upper()`, whitespace collapsed, twelve replacement pairs applied in order, trailing spaces
   stripped, and a full stop added unless the text already ends in one of five enders.
2. **`EnTokenizer.encode`**: every space becomes the `[SPACE]` token's spelling.
3. **`tokenizers`' encode** of a `tokenizer.json`: a BPE with no normalizer, a `Whitespace`
   pre-tokenizer (`\w+|[^\w\s]+`), no GPT-2 byte mapping, and one `[UNK]` per unknown character. The
   file has 58 added tokens, including event tags such as `[laughter]` that a user can type.

A host needs every step, in the engine, to offer `text2speech.infer(text)`. Three things about them
could have been written as C++ literals: the `punc_norm` table, which characters count as `\w`, and
Python's case mapping.

## Options

* **Reuse `BpeVocab` ("gpt2") with a new pre-tokenizer shape.** Rejected. That class NFC-normalizes
  and byte-maps, and the reference does neither, so ids for any non-ASCII text would change. A "gpt2"
  tag would also tell every host the vocabulary is byte-level when it is not.
* **A new class with the reference's constants written into it.** Rejected, on
  [ADR-003](adr-003-per-model-complexity-in-the-exporter.md)'s rule: the replacement table and the
  enders are properties of one model, not of a task.
* **A Unicode `\w` table and a full case mapping in the engine.** Rejected. `unicode.h` has letter,
  number and mark categories but cannot tell `Nd` from `No`, and it has no multi-codepoint uppercase.
  Getting either exactly right would mean a new Unicode table in the runtime for one model's
  normalizer.
* **A new tag whose rules are the file's.** Chosen.

## Decision

`tokenizer.ggml.model == "chatterbox"`, implemented by `loom::ChatterboxVocab`. The C++ holds the
**shape** of the function: capitalise, collapse whitespace, replace, strip, terminate, substitute the
space token, split on added tokens, pre-tokenize, merge by rank. Everything else is a
`tokenizer.ggml.chatterbox.*` key the exporter writes from the reference:

* the replacement pairs (in order), the sentence enders, the terminal and the empty-text stand-in;
* **`word_chars`**: the table's own single-character pieces that the reference's `Whitespace`
  pre-tokenizer joins to a letter, found by asking that pre-tokenizer;
* **`upper_from`/`upper_to`**: Python's `c.upper()` for every codepoint whose `c.islower()` holds.
  That is 1494 pairs and 8.5 KB, and it includes `ß` → `SS` and `ﬁ` → `FI`.

**Why `word_chars` can be restricted to the table and still be exact.** A character that is not in
the table becomes its own `[UNK]`, and no merge names `[UNK]`. So which side of a pre-token boundary an
unknown character falls on cannot change any id. Only the known characters' classes matter, and the
exporter has all of them. This also caught a case no reading of "whitespace" would predict: HF
classifies `U+00A0` as punctuation.

## Consequences

* **Measured exact.** Across 3000 generated texts in six classes (prose, lower-case starts, event
  tags, random table characters, out-of-vocabulary mixes, whitespace edges), the ids equal
  `EnTokenizer.text_to_tokens(punc_norm(text))` in **3000/3000** cases. The first version used a
  single-codepoint `to_upper` and differed in 17 cases, every one a text opening with a character
  whose uppercase is two codepoints. The shipped case table is what closed that gap.
* **The exporter refuses a `tokenizer.json` that is not this scheme**: a normalizer, a non-`Whitespace`
  pre-tokenizer, `fuse_unk`, `byte_fallback`, or a subword prefix or suffix. Such a file would load and
  tokenize wrongly, so it is refused by name.
* **The start and stop ids are the driver's**, as they are the reference's: `generate` adds them
  after tokenizing. A host that tokenizes elsewhere hands over the same ids `encode` returns.
* This is the pattern for the next model-specific normalizer: ship its constants, implement its shape.
  If a second family ends up needing the same shape, the class generalises by renaming its tag family,
  not by changing its data.

## Related

* [ADR-033](adr-033-a-decode-only-table-is-still-a-vocabulary-family.md): a tag names a scheme
* [ADR-003](adr-003-per-model-complexity-in-the-exporter.md): per-model constants belong in the
  exporter
* [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md): text front ends
* `loom-exporter/loom_exporter/chatterbox_tokenizer_export.py`: the writer
