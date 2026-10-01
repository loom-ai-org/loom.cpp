---
type: retro
date: 2026-10-01
domain: engine
tags: [tokenizer, sentencepiece, unigram, vocab, speecht5, t5, differential-testing]
---

# Retro-067: The Unigram Encoder Did Not Fuse Unknown Runs

## The Issue

SpeechT5's number-speller differential ([ADR-059](../adrs/adr-059-a-number-speller-ships-as-data-beside-the-vocabulary.md))
compared loom's ids with `SpeechT5Tokenizer`'s over 20,023 random strings. The spelled TEXT matched on
every one, but the ids differed on 14,543. On each of them loom emitted one `<unk>` per unknown
character, where SentencePiece emits one `<unk>` for the whole run.

## Root Cause

SentencePiece's Unigram model fuses consecutive unknown pieces. In flan-t5's own `spiece.model`,
`☃☃☃` and `ẞẞ` are each a single id 2. `loom::Vocab`'s Viterbi mirrors llama.cpp's UGM tokenizer
"almost line-for-line", and llama.cpp does not fuse either. So every Unigram file (T5, XLM-R,
SpeechT5) diverged from its reference on any text with two unknown characters in a row.

It had been measured, just not in the right place. ADR-057's 20,000-string check proved that a CHAR
model and a Unigram model are the same encoder **in SentencePiece**, where both fuse. The engine gate
compared one sentence with no unknown characters in it. Each check was right about what it measured,
and between them they never exercised the engine on an unknown run.

## The Fix

After the Viterbi backtrack, consecutive `unk_id` entries collapse into one, unless the vocabulary
has byte fallback (where an unknown becomes its bytes and there is nothing to fuse). Re-measured on
the engine:

* SpeechT5: 40,046/40,046 strings, identical ids.
* XLM-R (`punctuate-all`): 5,000/5,000.
* flan-t5: 4,998/5,000.

The two flan-t5 differences are an older and unrelated Viterbi TIE: `g`+`gg` and `gg`+`g` score
exactly the same, and the two implementations break the tie differently. **Resolved by
[ADR-060](../adrs/adr-060-a-unigram-tie-is-broken-by-the-vocabularys-own-reference.md)**: SentencePiece
stores path scores as float and compares in double, and a `.model` vocabulary now says so.
`tests/ci/test_number_speller.cpp` checks that `☃☃☃` encodes to one `<unk>`, and it fails with the
fusion removed.

## Takeaway

**An equivalence proved in the reference library is not proved in the engine.** "A is B in
SentencePiece" and "loom matches SentencePiece on this sentence" do not add up to "loom matches
SentencePiece". A tokenizer claim needs a differential run THROUGH `loom::Vocab`, over random text
that includes the edge the claim is about. Here that edge was unknown characters, which a
hand-picked English sentence never has.

**Mirroring a port is inheriting its gaps.** "Line-for-line with llama.cpp" was a statement about
provenance, and it read as a statement about correctness.

## Related

* [ADR-057](../adrs/adr-057-a-char-sentencepiece-model-ships-as-unigram.md), whose engine-side claim
  this corrects.
* [Retro-066](retro-066-a-matrix-index-and-a-batch-guess.md): SpeechT5's other finding in an engine
  path everything shared.
