---
type: adr
status: accepted
date: 2026-10-01
tags: [tokenizer, sentencepiece, unigram, vocab, exporter, engine]
supersedes: []
---

# ADR-060: A Unigram Tie Is Broken by the Vocabulary's Own Reference

## Context

After unknown runs fused ([Retro-067](../retros/retro-067-unigram-did-not-fuse-unknown-runs.md)),
flan-t5's ids still differed from SentencePiece on 2 of 5,000 random strings, and on 88 of 20,000
tie-heavy ones. Each was an exact tie: `g`+`gg` and `gg`+`g` score the same in real arithmetic.

SentencePiece 0.2.1's `EncodeOptimized` stores each path score as a FLOAT. A piece's score is the
ternary `IsUserDefined ? (length * max_score_ - 0.1) : GetScore()`, and the `0.1` literal makes the
whole expression a DOUBLE. So a piece candidate is the exact double sum of two floats, compared in
double against the stored float and then rounded to float. The unknown candidate is float throughout.
A tie therefore comes out one rounding step apart, and which split wins depends on every score before
it. loom used doubles throughout, and so does `tokenizers`, which is why transformers' fast flan-t5
tokenizer agreed with loom on both strings and only the slow, SentencePiece one did not.

## Options

* **Always SentencePiece's arithmetic.** Rejected. A vocabulary that ships only `tokenizer.json`
  (XLM-R's `punctuate-all`) has `tokenizers` as its reference, which loom already matched 5,000/5,000.
* **Always doubles** (today). Rejected for `.model` vocabularies, whose reference is SentencePiece.
* **Let the file say which reference it has.** Chosen.

## Decision

`write_sentencepiece_vocab` writes `tokenizer.ggml.unigram_scoring = "sentencepiece"` for a Unigram
vocabulary it read off a SentencePiece protobuf (`.model`, including the CHAR model ADR-057 maps to
Unigram). It writes nothing for a `tokenizer.json`-only one. On the key, `loom::Vocab` reproduces
`EncodeOptimized` exactly:

* float-stored path scores, a double comparison for a piece, a float one for an unknown;
* the unknown score as `min_score_ - 10` over NORMAL pieces only, in float;
* UNUSED pieces skipped;
* a user-defined piece scored `bytes * max_score_ - 0.1`, where `max_score_` starts at `FLT_MIN`
  (the smallest POSITIVE float) and so stays ~1e-38 over all-negative scores. That is reproduced, not
  fixed.

Without the key, every file behaves as before, which includes every GGUF published today.

**Verified through the engine, against the real libraries:**

* flan-t5 re-exported with the key: 5,000/5,000 random strings and 20,000/20,000 tie-heavy ones
  identical to SentencePiece. The published file without it still gives 2 and 88.
* SpeechT5 with the key: 20,023/20,023 identical to `SpeechT5Tokenizer` (the number speller's
  differential).
* XLM-R (`tokenizer.json`, no key): 5,000/5,000 identical to `tokenizers`.
* `tests/ci/test_unigram_scoring.cpp`: a five-piece vocabulary on which the two arithmetics split
  `yyxyxggg` differently (found by searching scores against `sentencepiece` itself). It fails when the
  key is ignored.

## Consequences

* An older engine ignores the key and keeps doubles, so a re-exported file there differs from
  SentencePiece only on exact ties, as today. WHEELS FIRST still holds for anyone who needs the exact
  ids.
* The exporter's two SentencePiece paths are no longer byte-identical: they differ by this one key,
  on purpose, because their references differ. `test_no_protobuf_reproduces_the_protobuf_path` says
  so.
* A future SentencePiece release that changes this arithmetic needs a new value for the key, not an
  edit to this one.
