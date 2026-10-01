---
type: adr
status: accepted
date: 2026-09-30
tags: [tokenizer, sentencepiece, exporter, speecht5, family-9b]
supersedes: []
---

# ADR-057: A CHAR SentencePiece Model Ships as Unigram

## Context

SpeechT5's text front end is `spm_char.model`, a SentencePiece protobuf with `model_type = CHAR`
(4). `loom::Vocab` implements UNIGRAM and BPE, and the exporter refused anything else:
"SentencePiece model_type 4 (WORD/CHAR) is not implemented on the C++ side."

A CHAR model splits the normalised text into characters and looks each one up. That is a simpler
segmenter than either of the two the engine has, and SpeechT5's vocabulary is 79 pieces: four
control/unknown tokens and 75 single characters, with no digits.

## Options

* **A CHAR segmenter in `loom::Vocab`.** Rejected. It is an engine change, so a loom-py pin bump,
  and the model could not be published before a release carries it (WHEELS FIRST). It adds a third
  segmenter for behaviour the second one already has.
* **A per-model vocabulary tag** (`tokenizer.ggml.model = "speecht5"`) with its own reader, as
  Pocket-TTS and Chatterbox have. Rejected. Those exist because their front ends do something
  SentencePiece does not (chunking, rules as data). This one does nothing SentencePiece does not.
* **Write it as Unigram when every matchable piece is one character.** Chosen.

## Decision

`write_sentencepiece_vocab` accepts `model_type == CHAR` when every NORMAL and USER_DEFINED piece
is a single code point, and writes it under the Unigram tag (`"t5"`) with its pieces, scores, types
and normaliser unchanged. A CHAR model with a multi-character piece is still refused.

**Why it is the same encoder:** with only single-character pieces, a Unigram lattice has exactly one
path, the character split. Unknown characters are the only place the two could differ, and they do
not: SentencePiece merges a run of unknowns into one `<unk>` in both models, so `2026` is one id (3),
not four.

**Measured, not argued.** The protobuf re-typed UNIGRAM encodes 20,000/20,000 random strings
(unknown runs, accents, CJK, repeated whitespace) identically to the CHAR original and to
`SpeechT5Tokenizer`. On the engine side, the SpeechT5 gate
(`tests/gate/test_e2e_speecht5_lua_driver.cpp`) checks `loom::Vocab::encode` against the
reference's ids before it compares any audio. The exporter's own test re-checks the equivalence in
SentencePiece and refuses the multi-character case.

`SpeechT5Tokenizer` appends `</s>` and the protobuf records no such flag, so the export states
`add_eos_token` (T5's precedent). `tokenizer_detect` gained `spm_char.model` as a protobuf name.

## Consequences

* SpeechT5 needed **no engine change**, so its GGUF runs on the released wheels.
* SpeechT5's vocabulary has no digits. `SpeechT5Tokenizer(normalize=True)` spells numbers out, but
  it defaults to False, and so does this export: `2026` reaches the model as one `<unk>`, as it does
  in the reference. A host that wants numbers spoken must spell them first.
