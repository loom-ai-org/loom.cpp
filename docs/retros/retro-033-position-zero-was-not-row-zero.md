---
type: retro
date: 2026-09-04
domain: exporter
tags: [tracing, family-12, oracles, verification]
---

# Retro-033: Position Zero Was Not Row Zero, and 24 of 27 Argmaxes Agreed Anyway

## The Issue

Family 12's driver hands every member of the family `position_ids = loom.range(0, n_tokens)` — one
contract, 0-based, for BERT and DistilBERT alike ([ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md)).
The family's third checkpoint is XLM-R, and XLM-R does not number its positions from zero.

`create_position_ids_from_input_ids` — fairseq's convention, inherited by every RoBERTa-family encoder
in `transformers` — starts counting at `padding_idx + 1`. With `padding_idx = 1`, token 0 reads position
table **row 2**, and rows 0 and 1 are never addressed by an unpadded sequence at all. The wrapper passed
0-based ids straight through, so the exported graph read two rows of a learned table that had been
trained as padding.

Nothing raised. No shape changed. The export succeeded, the file loaded, the driver returned six
plausible logits per token.

## Root Cause Analysis

The family's whole design is written against *the traced length reaching the graph* — three separate
places where `transformers` computes a tensor from `.shape[1]`, each harmless at the traced length and
wrong past it. Passing `position_ids` explicitly is the fix for the third of them, and it worked: the
axis stayed symbolic.

What the design did not ask was **what the ids passed explicitly should CONTAIN**. Having taken
ownership of a tensor the model would otherwise have computed, the wrapper inherited the obligation to
compute it the way the model would have — and "the way the model would have" is a per-architecture fact
sitting on the position table itself. `nn.Embedding`'s `padding_idx` is the marker: the fairseq family
constructs `nn.Embedding(max_pos, hidden, padding_idx=self.padding_idx)`, BERT and DistilBERT construct
the same table without one.

The same offset has a second consequence that was also missed: it **shortens** the family's length cap.
`max_position_embeddings` is the SIZE OF THE TABLE, not the length it can serve. XLM-R declares 514 and
can answer 512 tokens; the export was accepting 514 and would have indexed two rows past the end.

## The Number That Matters

Measured on the real checkpoint, same tokens, `position_ids` 0-based versus offset:

| | |
|---|---|
| max \|Δ\| in the logits | **7.040** |
| argmax agreement | **24 / 27** |

**A token-level oracle would have shipped this.** 89% of the labels were still right, on a punctuation
model whose labels are 89% `O` to begin with. The tensor comparison against `transformers` is what made
it a two-line finding instead of a released artifact — the same lesson as
[Retro-006](retro-006-kokoro-shipped-noise.md) and the standing rule in `CLAUDE.md`, arriving this time
through a family whose output is *already* a discrete label rather than audio. **The rule is not about
audio.**

## The Second Finding, Which Came Out of Sabotaging the First

`tests/ci/test_text_classify.cpp` ended on `return 0` rather than on
`LOOM_TEST_REPORT_AND_RETURN()`. `LOOM_CHECK` counts a failure and keeps going, so **every check in the
token-classification door's own hermetic test printed to stderr and exited green**. It was found only
because the new BOS/EOS arm was sabotage-checked before being trusted — the file reported
`CHECK FAILED at :75` and `exit=0` in the same breath.

Two of 50 CI tests were in this state (the other was `test_lua_bridge_has_function.cpp`); all 84 gate
tests were already correct. Both are fixed.

## Takeaways

* **When a wrapper takes ownership of a tensor the model would have computed, it owes the model that
  tensor's CONTENTS, not just its shape.** Making an axis dynamic and making it correct are two
  questions, and passing the first does not test the second.
* **`max_position_embeddings` is a table size, not a length.** Read the usable cap off the offset.
* **A discrete output does not make a token-level oracle sufficient.** Compare the tensor.
* **Sabotage the gate before trusting it** — the standing rule in `CLAUDE.md`, which here caught not
  the finding it was aimed at but the fact that the harness could not report one.

## See Also

* [ADR-025](../adrs/adr-025-the-protobuf-owns-pieces-the-fast-tokenizer-owns-ids.md) — the tokenizer
  half of the same checkpoint.
* [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md) — the family's tracing decisions.
* [Retro-006](retro-006-kokoro-shipped-noise.md) — the first time a plausible output passed a weak oracle.
