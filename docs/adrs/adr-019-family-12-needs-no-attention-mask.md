---
type: adr
status: accepted
date: 2026-09-03
tags: [model-coverage, exporter, tracing, dynamic-shapes]
---

# ADR-019: A Single-Sequence Encoder Exports With No Attention Mask At All

## Context

Family 12 — BERT-family token classifiers — is the smallest template on the roadmap and the first
non-audio task ([Epic-03 §3](../epics/epic-03-model-coverage.md)). Its acceptance criterion is the
standing one: **a new family should need no engine work.** Everything it wants already existed —
`WordPieceVocab` reads its vocabulary, `loom.argmax_rows` performs its reduction (built for
Conformer-CTC's frame-wise head in P4.0.17), and `loom.task`/`loom.output.kind` carry its contract.

One thing stood in the way, and it is a tracing problem rather than a modelling one. **Every route
`transformers` takes to build a BERT attention mask bakes the traced sequence length into the graph.**
A 2-D mask reaches `_prepare_4d_attention_mask_for_sdpa`, which expands to a Python-level `tgt_len`;
its `torch.all(mask == 1)` early-out is explicitly skipped when `torch.jit.is_tracing()`, so even an
all-ones mask becomes a baked `[1, 1, 128, 128]` constant. The `token_type_ids` default is a buffer
slice `[:, :seq_length]` and `position_ids` is derived from `.shape[1]`, both at the traced length.

Every one of these is **silently harmless at the traced length and only diverges past it** — which is
the property that makes it worth a decision record rather than a commit message.

## Options Considered

1. **Let `transformers` build the mask.** Bakes the length, as above. Rejected on the source read and
   confirmed on the emitted MIL program.
2. **Declare `attention_mask` as a host-computed input filled by `loom.zero_mask`.** The binding
   already exists (an all-zeros additive mask, no engine work), and this is the shape the causal-LM
   family uses for `loom.causal_mask`. It does not fit: `BertModel` rejects a 4-D mask outright —
   `ModuleUtilsMixin.get_extended_attention_mask` handles dim 2 and dim 3 and raises otherwise — so
   an already-additive mask can only be delivered by reaching past `BertModel` into `.encoder`, which
   is a per-architecture attribute path (DistilBERT and DeBERTa spell it differently).
3. **Declare a 3-D ones mask**, which `get_extended_attention_mask` accepts and converts with pure
   arithmetic on the input tensor — no baked shapes. Rejected: `loom.zero_mask` produces zeros and
   this path needs ones (it computes `(1 - mask) * min`), so it costs a new engine binding — for a
   family whose acceptance criterion is that it needs none — and marshals an n² tensor across the Lua
   boundary on every call for a value that is a constant.
4. **Do not build a mask at all.**

## Decision

**Option 4: the export neutralises `get_extended_attention_mask` on the base model and the encoder
runs its no-mask path.** A mask exists to hide padding; this family's door hands the model exactly the
tokens the caller wrote, one unpadded sequence, so the correct mask is no mask. The emitted MIL program
contains no mask tensor and declares exactly two inputs, `tokens` and `position_ids`, both on the one
symbolic token axis.

The two smaller siblings follow the same principle — *derive it from an input, or do not have it*:
`token_type_ids` is `tokens * 0` rather than a buffer slice, and `position_ids` is an explicit input
the synthesized driver fills with `loom.range(0, n_tokens)` (already in
`driver_components.POSITION_INPUT_NAMES`, so it costs the caller nothing).

The patch is applied to `model.base_model` — HF's own accessor for the encoder under a task head — so
it holds for BERT, RoBERTa, XLM-R, ELECTRA and DeBERTa without a per-model table, and it is set on the
instance rather than the class.

## Consequences

* **Positive: the family cost zero engine C++ for its graph**, which is the acceptance criterion.
  What C++ it did add (`loom::text::classify`) is the task's door, not its architecture — the
  per-task/per-model line [ADR-013](adr-013-one-door-per-task.md) draws.
* **Positive: the driver is five lines** — read the input, fill the positions, one retained call, one
  `loom.argmax_rows`. A new family that is one forward pass costs one epilogue.
* **Negative: this export cannot serve a padded batch**, which is a real capability a masked export
  would have. It is not one this engine offers: the KV cache is single-sequence and every family here
  is called one utterance at a time ([Epic-01 §4](../epics/epic-01-inference-engine-core.md)). A
  family that genuinely needs padding is what would make option 3's binding worth building.
* **Negative: it depends on a `transformers` internal.** `get_extended_attention_mask` is a public
  method of `ModuleUtilsMixin`, but *that BertModel calls it at all* is internal, and a version that
  routed around it would produce a baked mask rather than an error. The guard is the second export in
  `test_token_classification_export.py`: the same checkpoint traced at two lengths must produce the
  identical topology, which no baked graph can do.

## See Also

* [Epic-03](../epics/epic-03-model-coverage.md) — the family roadmap this closes an item on
* [ADR-003](adr-003-per-model-complexity-in-the-exporter.md) — per-model complexity in the exporter
* [ADR-013](adr-013-one-door-per-task.md) — one door per task, declared by the file
