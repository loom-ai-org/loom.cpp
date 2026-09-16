---
type: adr
status: accepted
date: 2026-09-16
tags: [engine, exporter, tokenizers, family-5, model-coverage]
supersedes: []
---

# ADR-036: How Pieces Compose Is the Scheme, Not the Table — and Whoever Knows the Unicode Writes It Down

## Context

Family 5's Paraformer carries a FunASR `CharTokenizer`: `tokens.json`, a flat JSON array of 8,404
pieces, one per row of a non-autoregressive decoder's output. No merges, no scores, no normalizer —
the same shape as family 4's CTC table, which [ADR-033](adr-033-a-decode-only-table-is-still-a-vocabulary-family.md)
already gave its own tag.

So the table needs nothing new. What is new is that concatenating the pieces does not produce text:

    ["and","so","my","f@@","el@@","low"]  ->  "and so my fellow"   `@@` continues into the NEXT piece
    ["hello","你","好"]                    ->  "hello你好"           the space is REMOVED before CJK
    ["b","b","c","news"]                  ->  "BBC news"           letter runs collapse AND UPPERCASE
    ["<s>","and","</s>"]                  ->  "and"                control pieces drop

The first instinct — recorded in this ledger and wrong — was that this is a per-TASK text postprocess
belonging in the `transcribe` door, on the grounds that the CJK rule needs lookahead and the
abbreviation rule is a transform rather than a lookup.

## Decision

**It is a vocabulary scheme. It gets a tag, `tokenizer.ggml.model == "funasr"`, and a `FunasrVocab`
whose `decode` performs the whole assembly.**

Two things settled it. `decode` has never been a lookup for any family here — SentencePiece's rewrites
U+2581 and WordPiece's strips `##`, both of which are composition over a sequence — so "needs
lookahead" does not disqualify it. And `model.detokenize(ids)` has to answer *something*:
`"andsomyf@@el@@low"` is not text in any sense a caller wants, so if the assembly lives above the
vocabulary then the one route that reaches it is `transcribe` and the other route returns rubbish.

**It could not have been folded into an existing tag.** `@@` is a SUFFIX meaning "I continue"; U+2581
and `##` are PREFIXES meaning "a word starts here". They are duals — whether a piece begins a word
depends on its predecessor under this convention — and the same piece string occurs in both roles, so
no per-piece rewrite of the table converts one into the other. That is ADR-033's argument arriving at
the same answer from a different direction.

**And the per-piece SCRIPT is computed by the exporter, not the engine.** The reference decides
`isAllChinese` / `isAllAlpha` per CHARACTER using Python's Unicode `isalpha()`. This vocabulary contains
exactly one character (U+2B5AF, a CJK extension ideograph) that is alphabetic and outside the
U+4E00–U+9FFF block the Chinese test uses, so the obvious "ASCII letters plus the CJK block" range check
in C++ is wrong on one row in 8,404 — silently, with a still-readable transcript.
`tokenizer.ggml.piece_script` is one small int per piece, and it is
[ADR-027](adr-027-the-protobuf-owns-pieces-the-fast-tokenizer-owns-ids.md)'s principle one family over:
whoever actually knows a fact is the one who writes it down.

## Consequences

* `transcribe` gains one branch and `detokenize` gains a fifth reader; nothing else in the engine moves.
  Family 5's second leaf therefore costs the engine one class and one KV.
* **The `@@` test is orthogonal to the script, and the order is the reference's.** `f@@` is neither CJK
  (an `f` is not) nor Latin (an `@` is not alphabetic), so it classifies as OTHER and is *still* a
  continuation; `9@@` is CJK by the reference's own test (digits and `@` count) and is *not* one. A
  first implementation nested the marker test inside the Latin branch and emitted `f@@` literally.
* **Verified differentially rather than by example**: 20,000 random id sequences drawn from the real
  vocabulary, engine against `sentence_postprocess`. That is what found the `f@@` ordering, the
  per-character predicates, and a `<blank>` the exporter was dropping where the reference prints it —
  the last of which no realistic input could have exposed, because a non-autoregressive decoder cannot
  emit a blank. Reproducing a reference does not get to stop at the rows we expect to see.
* The three branches the reference splits into (all-CJK, all-Latin, mixed) differ only in the timestamps
  they build. For text the mixed branch reproduces all three, so only that one is implemented — checked,
  not assumed.
