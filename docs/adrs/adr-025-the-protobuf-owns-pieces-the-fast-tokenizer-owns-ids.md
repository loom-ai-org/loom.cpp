---
type: adr
status: accepted
date: 2026-09-04
tags: [tokenizers, exporter, model-coverage, family-12]
---

# ADR-025: The SentencePiece Protobuf Owns Pieces; The Fast Tokenizer Owns Ids

## Context

Every SentencePiece caller in this repo before family 12's third checkpoint was a NeMo ASR model. Those
ship a bare `tokenizer.model` inside a `.nemo` archive, and their ids **are** the protobuf's piece
order, so `write_sentencepiece_vocab` transcribing `ModelProto.pieces` in order was not merely correct —
there was no other candidate.

`oliverguhr/fullstop-punctuation-multilang-large` is XLM-R, and it is not like that. `transformers`'
own `XLMRobertaConverter` builds the model's vocabulary from the protobuf by:

1. dropping the proto's leading `<unk>`, `<s>`, `</s>` (indices 0–2),
2. prepending `<s>`, `<pad>`, `</s>`, `<unk>` at ids 0–3,
3. appending every remaining proto piece, so each lands one id higher than its proto index,
4. appending `<mask>` at the end.

250,000 proto pieces become a 250,002-entry vocabulary in which `,` is id **4** and not id **3**. This
is fairseq's convention and it is shared by every fairseq-derived checkpoint, which is a large family:
XLM-R, and the RoBERTa-style Unigram models generally.

**Nothing in the protobuf records the remapping.** Writing it verbatim produces a GGUF that loads,
whose vocabulary is the right size, whose `decode` returns readable text, and whose every id is off by
one against the embedding table the model was trained with. There is no error path — the failure is a
plausible answer.

## Options

1. **Key the remapping on `model_type`.** A table mapping `xlm-roberta` (and roberta, camembert, xlm-r
   derivatives, …) to "apply the fairseq offset". Rejected: it is a list of names to keep in step with
   `transformers`' own converter list, and it is wrong the first time a checkpoint declares an
   architecture the table has not seen while shipping the same tokenizer.
2. **Derive the offset arithmetically** — detect that the proto opens with `<unk>/<s>/</s>` and
   synthesize fairseq's four specials. Rejected for the same reason in a subtler form: it reimplements
   one converter's recipe and silently produces a *different* wrong answer for any checkpoint whose
   converter did something else.
3. **Read the ids off the `tokenizer.json` the checkpoint ships.** Its Unigram `model.vocab` is that
   converter's OUTPUT — an id-ordered list of `[piece, score]`. Whatever remapping happened, this file
   is what happened.
4. **Use `tokenizer.json` for everything.** Rejected: it carries no `precompiled_charsmap`, no
   `add_dummy_prefix`, no `remove_extra_whitespaces` and no per-piece TYPE. `loom::Vocab` needs all
   four — without the charsmap it segments an un-normalized string, and without the types a `<mask>`
   or a `<0x41>` byte piece enters the match trie and starts matching ordinary text.

## Decision

**The protobuf is the authority on what a piece IS. A `tokenizer.json` beside it, where one exists, is
the authority on what its ID is.** `read_hf_id_layout` reads the second; `write_sentencepiece_vocab`
combines them — pieces, scores and ids from the fast tokenizer, per-piece types and the whole
normalizer from the protobuf. A piece the protobuf does not have at all is one the converter *added*,
so it is written `CONTROL`: unmatchable in text, which is what `<pad>` and `<mask>` have to be.

The framing (`add_bos_token`/`add_eos_token` and their ids) is read from the **post-processor**, not
from `special_tokens_map.json`. The question is not "does this tokenizer have a `<s>`" but "does its
encode put one there", and a `TemplateProcessing.single` of `<s> $A </s>` is that question written down.
T5's special-token map names an `eos_token` its encode does not add.

**The seam is `tokenizer.json`'s presence**, and that is the load-bearing part of the decision rather
than an implementation detail: every caller that predates this ships no `tokenizer.json`, reads `None`,
and writes byte-for-byte the file it wrote before. The sweep baseline for six NeMo models is unmoved.

## Consequences

* One new optional parameter (`hf_ids`) and one new reader. No engine change: `loom::Vocab`'s UGM path
  already did everything, and it now gets a vocabulary whose ids match what the graph was trained on.
  Engine encode matches `AutoTokenizer` exactly on 8/8 multilingual sentences.
* The explicit `bos_token_id`/`eos_token_id` kwargs (the ALBERT/XLNet door) still override, because a
  caller naming a framing is making a claim the files do not.
* **This generalizes past family 12.** Any future family whose checkpoint is fairseq-derived —
  translation encoder-decoders (family 6) most obviously — gets the right ids without knowing it needed
  them.
* `sentencepiece.bpe.model` joins `tokenizer.model` and `spiece.model` as a name the detector knows.
  It is a misnomer: those files are Unigram models, which `write_sentencepiece_vocab` reads off
  `trainer_spec.model_type` rather than off the filename.

## See Also

* [Epic-03 §2](../epics/epic-03-model-coverage.md) — what family 12's third checkpoint cost.
* [ADR-019](adr-019-family-12-needs-no-attention-mask.md) — what its first two cost.
* [Retro-033](../retros/retro-033-position-zero-was-not-row-zero.md) — the other half of the same
  checkpoint, and the one a token-level oracle would have shipped.
