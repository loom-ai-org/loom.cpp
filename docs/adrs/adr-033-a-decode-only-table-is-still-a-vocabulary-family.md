---
type: adr
status: accepted
date: 2026-09-12
tags: [exporter, engine, tokenizers, family-4, model-coverage]
---

# ADR-033: A Decode-Only Table Is Still a Vocabulary Family of Its Own

## Context

Family 4 (wav2vec 2.0 / HuBERT / data2vec-audio, `EXPORT-ROADMAP.md`'s CNN + transformer + CTC group)
carries HF's `Wav2Vec2CTCTokenizer`: a flat `vocab.json` of one piece per CTC class, a `pad_token`
that is the blank, and a `word_delimiter_token` that decodes to a space. There is no merge table, no
score, no normalizer and no segmentation algorithm, because nothing ever encodes with it — the ids
come out of a CTC head's argmax and the only question ever asked is what they spell.

The engine had four vocabulary readers and none of them was this. It also had a cheap way to avoid
writing a fifth: `loom::Vocab`'s SentencePiece decode is *concatenate the pieces, then replace U+2581
with a space*, which is the same two steps if the exporter rewrites the delimiter piece to `▁` and
tags the file `"t5"`. That works. It ships family 4 with **zero engine changes**, which is the
roadmap's own stated acceptance criterion for a new family.

## Decision

**Write it as its own tag, `tokenizer.ggml.model == "ctc"`, with a small decode-only `CtcVocab` in the
engine.** Do not reuse `"t5"`.

The word delimiter is declared as an **id** (`tokenizer.ggml.word_delimiter_id`), not a spelling, and
the pieces are written exactly as `vocab.json` has them.

## Consequences

**What the rejected option would have cost.** A `"t5"` file would decode correctly and then lie about
itself in two ways a reader cannot detect: `id_to_piece(4)` would answer `▁` where the checkpoint says
`|`, and the tag would claim a Viterbi segmentation over unigram scores against a normalizer the file
does not carry. The per-family-tag convention — `"t5"`/`"llama"`, `"gpt2"`, `"bert"`, `"byt5"`,
`"phonemes"`, `"supertonic"` — exists precisely so that a tag answers "which scheme is this", and the
cheap option spends that property to save about a hundred lines.

**The acceptance criterion is a measurement, not a rule to satisfy.** "A new family should need no
engine work" is there to keep per-model C++ out of the engine
([EXPORT-PREPARATION.md](../../../loom-exporter/docs/EXPORT-PREPARATION.md) 1.3). A vocabulary reader
is the standing exception that document already names: it is per-TASK, not per-model — one `CtcVocab`
covers every CTC character checkpoint that will ever be exported, the same way one `ctc_greedy_decode`
covers every CTC head. Family 4 needed no new primitive, no new driver component and no new builder;
what it needed was a reader, which is the category the criterion was never about.

**The blank is read, not derived, and that is the part a second family would get wrong.** Family 1's
NeMo CTC convention puts the blank at `num_classes - 1`; HF's puts it at the tokenizer's `pad_token`,
which is row 0. The two disagree at both ends of the row. Worse, `pad_token` is not always spelled
`<pad>`: `omniASR-CTC-300M-v2` declares `pad_token: "<s>"` and carries a *separate* unused `<pad>`
piece at id 1. So the exporter resolves the NAME through the vocabulary and writes the resulting id,
and nothing anywhere assumes a position.

**`CtcVocab` has no `encode`, deliberately.** Every other class here carries an algorithm read off the
checkpoint; this one carries a table. There is nothing for an encode to be the inverse of, and adding
one would mean inventing a segmentation rule no file in this family states.

**Three loaders now run in order in `transcribe`.** `BpeVocab::load` and `CtcVocab::load` return
`nullptr` for a schema that is not theirs; `Vocab::load` **throws**. So `Vocab::load` is last, and any
future reader has to go ahead of it for the same reason.
