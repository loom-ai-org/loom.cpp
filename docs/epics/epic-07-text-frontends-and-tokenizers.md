---
type: epic
status: active
domain: text-frontend
last_updated: 2026-08-22
---

# Epic-07: Text Front-Ends and Tokenizers

## 1. Context and Scope

Everything between a string and the ids a model consumes: BPE and SentencePiece tokenizers, grapheme
vectorizers, and the phonemizer that four TTS models are waiting on.

This is one of the few areas where C++ in the engine is correct, because a tokenizer is per-**task**,
not per-model ([ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md)) — with one
deliberate, temporary exception noted below.

## 2. Architectural Overview

### Tokenizers

`bpe_vocab.cpp`'s `pre_spec_table()` carries ~40 pretokenizer families. `BpeShape::kSpmByteFallback`
covers SentencePiece-style byte-fallback BPE, which has four structural differences from every other
shape: no regex pretokenization (one chunk), no GPT-2 byte-level mapping (initial symbols are
characters), and two more — each measured against the real tokenizer rather than inferred.

**A family the table does not cover raises a named error rather than mis-tokenizing.** That is what
turns "add a tokenizer" into a bounded job instead of a mystery, and it is the pattern to keep.

### Grapheme front ends

`src/core/supertonic_text_vectorizer.cpp` is per-**model** C++ in an engine whose rule forbids it. It
is kept because it is written and verified — and **a second grapheme TTS model must not add a second
class.**

The split that generalizes, when a real second data point exists:

* the **codepoint table is data** and already lives in the GGUF;
* the **normalization pipeline** (emoji ranges, the replacement table, the `<lang>` wrap) is per-model
  and belongs in exporter-emitted data or driver Lua.

Qwen3-TTS is **not** that second data point: it has a real HF tokenizer and needs only `tokenizer_dir`
set in its export config — no new C++ at all.

### Phonemization

Four TTS models (VITS, Kokoro, StyleTTS2, Matcha) take phoneme ids and have no text door. The licence
blocker is resolved: a C++ port of the Apache-2.0 `orthography2ipa`, vendored as a submodule, with
rules and data in the engine as one copy —
[ADR-012](../adrs/adr-012-permissive-phonemizer.md).

**The work splits in two, and only the second half needs the port:**

1. **Export the phoneme symbol table** as a vocabulary family. The data is already in each checkpoint
   and simply not exported. Gives four models a real `model.tokenizer` and a working
   `synthesize(phonemes=...)`, with no licence question at any point.
2. **The C++ port**, verified against the Python door as its oracle.

Target shape: `src/text/phonemize.cpp` + `include/loom/text/phonemize.h`. The Python half already ships
behind `loom-py-rt[phonemes]`.

**Two risks, both measurable before any C++ is written:** the fold-down into each checkpoint's fixed
symbol→id table (a symbol outside it has *no id*, which is a lookup with no answer rather than a
degradation), and pinning the beam search's tie-break so the CLI and `loom-py` cannot drift.

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-012](../adrs/adr-012-permissive-phonemizer.md), [ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md) |
| Design | [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md) §5 |
| Retros | [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md) |
| Active tasks | [Backlog → Text front-ends](../backlog/active-index.md#text-front-ends) |

## 4. The Record

### SentencePiece-style byte-fallback BPE — DONE


`BpeShape::kSpmByteFallback`, added so Gemma 3 tokenizes correctly. `pre_spec_table()`'s own comment had
already scoped it ("needs a different symbol-initialization step in `BpeVocab::encode()` itself") and
`tokenizer_detect.py` raised a named `NotImplementedError` rather than mis-tokenizing — which is what
made this a bounded job instead of a mystery.

Four structural differences from every other shape, each measured against the real tokenizer: no regex
pretokenization (one chunk); no GPT-2 byte-level mapping, so initial symbols are characters and the
vocabulary holds literal UTF-8; a space→U+2581 normalizer with no dummy prefix (`"Hello world"` →
`['Hello', '▁world']`), and no NFC, because the HF normalizer is that substitution and nothing else; and
`<0xNN>` byte fallback for characters with no entry. `decode` mirrors all of it.

Gated by `test_e2e_spm_byte_fallback_tokenizer` — nine cases, every expectation `AutoTokenizer.encode`
verbatim, all encoding exactly and round-tripping. Gemma now exports with no `--tokenizer-pre` override.
The remaining unimplemented families in `_LLAMA_PRE_TO_LOOM_PRE_TYPE` (CJK-script splitters,
case-transition shapes, cascading-whitespace shapes) are still `None` and still raise by name.


### Grapheme text front-ends: the shape to generalize to

`src/core/supertonic_text_vectorizer.cpp` is per-MODEL C++ in an engine whose rule is that per-model
complexity belongs in the exporter. It is kept because it is written and verified, but a SECOND grapheme
TTS model must not add a second class. The split that generalizes: the codepoint table is data and already
lives in the GGUF; the normalization pipeline (emoji ranges, the replacement table, the `<lang>` wrap) is
per-model and belongs in exporter-emitted data or driver Lua. Worth doing when there is a real second data
point to generalize against — Qwen3-TTS is not one, since it has a real HF tokenizer and needs only
`tokenizer_dir` set in its export config, no new C++ at all.

---

