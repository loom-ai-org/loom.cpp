---
type: adr
status: accepted
date: 2026-09-03
tags: [model-coverage, exporter, tracing, dynamic-shapes]
---

# ADR-019: Every Tensor a Token Classifier Needs Is Derived From an Input

## Context

Family 12 — BERT-family token classifiers — is the smallest template on the roadmap and the first
non-audio task ([Epic-03 §3](../epics/epic-03-model-coverage.md)). Its acceptance criterion is the
standing one: **a new family should need no engine work.** Everything it wants already existed —
`WordPieceVocab` reads its vocabulary, `loom.argmax_rows` performs its reduction (built for
Conformer-CTC's frame-wise head in P4.0.17), and `loom.task`/`loom.output.kind` carry its contract.

One thing stood in the way, and it is a tracing problem rather than a modelling one. **`transformers`
computes three tensors from the Python-level sequence length**, and a traced graph bakes each of them
at whatever length the trace ran at:

* the **attention mask**. Absent one, BERT builds `torch.ones(input_shape)` and DistilBERT does the
  same inline. Supplying a 2-D one is not automatically enough either: on the sdpa path,
  `_prepare_4d_attention_mask_for_sdpa` expands to a Python-level `tgt_len`, and its
  `torch.all(mask == 1)` early-out is explicitly skipped when `torch.jit.is_tracing()`, so even an
  all-ones mask becomes a baked `[1, 1, 128, 128]` constant.
* **`token_type_ids`**, whose BERT default is the buffer slice `self.embeddings.token_type_ids[:, :seq_length]`.
* **`position_ids`**. BERT accepts them as an argument; DistilBERT does not, and its `Embeddings.forward`
  reads `self.position_ids[:, :seq_length]` — a buffer slice whose own source comment says it "helps
  when tracing", which is true of a fixed-shape export and exactly wrong here.

Every one of these is **silently harmless at the traced length and only diverges past it** — which is
the property that makes it worth a decision record rather than a commit message.

## Options Considered

1. **Let `transformers` compute them.** Bakes all three. Rejected on the source read and confirmed on
   the emitted MIL program.
2. **Declare `attention_mask` as a host-computed input filled by `loom.zero_mask`.** The binding
   already exists (an all-zeros *additive* mask, no engine work), and this is the shape the causal-LM
   family uses for `loom.causal_mask`. It does not fit: `BertModel` rejects a 4-D mask outright —
   `ModuleUtilsMixin.get_extended_attention_mask` handles dim 2 and dim 3 and raises otherwise — so an
   already-additive mask can only be delivered by reaching past `BertModel` into `.encoder`, which is a
   per-architecture attribute path.
3. **Neutralise the mask so the encoder runs a no-mask path.** What the first version of this ADR
   decided, by overriding `get_extended_attention_mask` to return `None`. It works for BERT and **does
   not generalise**: DistilBERT has no such method, and its `MultiHeadSelfAttention` has no
   `if mask is not None` guard at all — it unconditionally evaluates `(mask == 0).view(...)`, so a
   `None` is an `AttributeError` rather than a fast path. It also depended on a `transformers`
   internal in a way that would degrade to a baked mask rather than an error.
4. **Derive every one of them from an input.**

## Decision

**Option 4, as one rule: every tensor the model needs is derived from an input, never computed from
`.shape[1]`.**

* **The mask is `torch.ones_like(tokens)`** — a real, all-ones padding mask, 2-D, passed in. Passing
  one is what stops the model building its own, and the 2-D shape is what routes it through
  `get_extended_attention_mask`, which for an encoder is `mask[:, None, None, :]` and
  `(1 - x) * finfo.min` — pure arithmetic on the input tensor. `load_model` asks for
  `attn_implementation="eager"` so the sdpa expand in the Context above is not reached; that flag is
  load-bearing, not conservatism.
* **`token_type_ids` is `tokens * 0`**, where the model takes one at all.
* **`position_ids` is a graph input**, passed as a kwarg where the model accepts one and, where it does
  not, substituted into the position-embedding table by a forward pre-hook. **Both routes produce the
  same two graph inputs and the same driver.**

Which of these a given checkpoint needs is read from `base.forward`'s **signature**, not from its name:
`base_model` is HF's own accessor for the encoder under a task head, and `inspect.signature` answers
"does this take `position_ids`" for a model this template has never seen. There is no `model_type`
table.

The synthesized driver is unchanged by any of it — `tokens` plus `loom.range(0, n_tokens)`, which
`driver_components.POSITION_INPUT_NAMES` already knew how to fill.

## Consequences

* **Positive: the family cost zero engine C++ for its graph**, which is the acceptance criterion.
  What C++ it did add (`loom::text::classify`) is the task's door, not its architecture — the
  per-task/per-model line [ADR-013](adr-013-one-door-per-task.md) draws.
* **Positive: the driver is five lines** — read the input, fill the positions, one retained call, one
  `loom.argmax_rows`. A new family that is one forward pass costs one epilogue.
* **Positive: it survived the second architecture, which is what changed this ADR.** The first version
  decided option 3 and was written as though it generalised; DistilBERT falsified that within a day.
  Option 4 was then verified on both — BERT (token-type embeddings, a `position_ids` argument,
  `.encoder`) and DistilBERT (none of the three) — producing byte-identical graph inputs and drivers.
* **Negative: this export cannot serve a padded batch.** The mask is all ones by construction, so
  padding would be attended to. That is not a capability this engine offers anywhere — the KV cache is
  single-sequence and every family here is called one utterance at a time
  ([Epic-01 §4](../epics/epic-01-inference-engine-core.md)) — and a family that genuinely needs it is
  what would make the mask a real caller-supplied input rather than a derived constant.
* **Negative: `attn_implementation="eager"` is a correctness requirement spelled as a performance
  knob.** Nothing in the export fails if it is dropped; the graph simply bakes its length. The guard
  is `test_token_classification_export.py`, which exports the same checkpoint at two `seq_len` values
  and requires an identical topology — which no baked graph can produce — for both architectures.

## Verification

Both checkpoints, against `transformers`, on the **tensor** rather than the argmax (the standing
*tensor oracle, not token oracle* rule), over 138 tokens of 10 sentences:

| | max abs Δ | worst cosine | argmax | sabotage arm |
|---|---|---|---|---|
| `dslim/bert-base-NER` | 1.24e-05 | 0.99999988 | 138/138 | 11.94 |
| `dslim/distilbert-NER` | 5.72e-06 | 0.99999988 | 138/138 | 9.54 |

The sabotage arm is the same graph measured against a *different sentence's* reference; six orders of
magnitude between the two columns is what makes the first one mean something. The engine's own
WordPiece encode is checked against HF's ids on every sentence first, since otherwise the two sides
are not comparing the same input.

## See Also

* [Epic-03](../epics/epic-03-model-coverage.md) — the family roadmap this closes an item on
* [ADR-003](adr-003-per-model-complexity-in-the-exporter.md) — per-model complexity in the exporter
* [ADR-013](adr-013-one-door-per-task.md) — one door per task, declared by the file
