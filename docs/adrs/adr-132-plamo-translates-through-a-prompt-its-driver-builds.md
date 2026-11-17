---
type: adr
status: accepted
date: 2026-11-17
tags: [model-coverage, tokenizer, translation, plamo, causal-lm, host-api, streaming]
supersedes: []
---

# ADR-132: PLaMo Translates Through a Prompt Its Driver Builds

Serves [Epic-03](../epics/epic-03-model-coverage.md), [Epic-06](../epics/epic-06-high-level-api-and-hosts.md)
and [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md).

## Context

PLaMo 3 translate 0.8B (pfnet) is a 16-layer causal LM with sliding-window attention. Seven layers in
eight see 2,048 positions; every eighth is full, with YaRN RoPE. It has one K/V head and offset RMS norms,
and ships as remote code. Two things are new here.

* **Its tokenizer.** `Plamo3Tokenizer` is a unigram model with integer-rounded scores, decoded by Viterbi
  (an Aho-Corasick walk from the end), with byte fallback. It cuts hard boundaries around
  `<|plamo:...|>` tags, runs of 2+ spaces and runs of 4+ of one character. It is neither SentencePiece
  nor BPE.
* **Its prompt.** PFML3 is a tag format built from the request's arguments: `target_lang` (required),
  `source_lang`, `context`, a glossary of term pairs, a Japanese `form`, and earlier source/translation
  pairs as history. The checkpoint's chat template renders it from those arguments, which the engine's
  role/prefix/suffix chat template ([ADR-018](adr-018-chat-template-as-role-tags.md)) cannot express.

## Options

1. **Plain text generation.** Ship it as a causal LM; callers write PFML3 by hand.
2. **The host renders PFML3.** loom-py would carry one model's prompt format, which its one rule forbids.
3. **The driver renders PFML3.** The export appends a `translation_prompt` function to the synthesized
   causal-LM driver. A host encodes the caller's strings, calls it, and generates from the ids it returns
   with the ordinary loop.

## Decision

Option **3** (the user, 2026-11-17).

* **`loom::PlamoVocab`** (`tokenizer.ggml.model = "plamo"`) implements the reference's rules: integer
  scores, the longest piece on a tie, the 10^7 unknown cost and its bytes, and the tag and run boundaries.
  It is held to the reference on real text. `loom.tokenize`/`detokenize` and loom-py dispatch to it.
* **`driver_appendix`** (exporter): a top-level Lua chunk appended to a synthesized driver, for a function
  a host calls by name. It is refused if it runs a topology. PLaMo's is `translation_prompt`, with the
  tag ids bound above it.
* **The prompt is the concatenation of the pieces' ids.** Every tag is a tokenizer boundary, so encoding
  the caller's strings one by one gives the ids of the whole rendered string. Lists travel as one id
  array plus offsets. The driver keeps the template's refusals: an empty text, a malformed language
  code, `form` without a Japanese target, a reserved tag inside a string.
* **loom-py**: `text2text.translate(text, target_lang, source_lang=, context=, glossary=, form=,
  history=)` and `translate_stream`, which is the token stream of
  [ADR-099](adr-099-a-stream-leases-the-models-caches-or-owns-its-own.md) over the built prompt.
  `Model.can_translate` checks for the function in the file.
* **The export** is the causal-LM family's leaf `plamo_export`. Its own trace wrapper computes RoPE from
  the positions the driver passes, folds the norm offsets into the weights, and spells MQA in HF's
  `repeat_kv` form. The window widths come from `window_size`/`sliding_window_pattern`.

## Consequences

* A translation is greedy by default, as the card recommends (`do_sample=False`); the checkpoint's stop
  set is `{bos, tag}`, read from `generation_config.json`.
* The PLaMo Community License restricts commercial use (registration under 1 billion yen of revenue, a
  separate agreement above it). The card says so.
