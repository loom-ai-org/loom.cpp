---
type: adr
status: accepted
date: 2026-09-10
tags: [engine, exporter, attention, model-coverage, family-6]
---

# ADR-028: T5's Relative Attention Bias Is a Mask, and the Driver Builds It

## Context

T5 has no positional embedding. Every attention score is offset by a learned value chosen by the
*bucketed* distance between the query and the key: `_relative_position_bucket` maps `key - query` into
32 log-spaced buckets, indexes a `[32, n_head]` table, and the result is added to the scores. The
table is on block 0 of each stack and its output is threaded through every later block, so one
`[1, n_head, q, k]` tensor serves the whole stack — recomputed per sequence length.

The engine has no primitive for this, and the backlog entry that scoped family 6 said so, with a
warning worth keeping: `src/core/relative_position.cpp` has the matching name and is a **different**
mechanism (VITS's windowed `pad_crop_relative_embeddings`), so reusing it is wrong.

The entry offered three ways out, none scoped further: a new engine primitive, computing the bias in
the Lua driver per call, or folding it per length at export.

## What Reading the Seam Changed

The three options all assume the bias is a thing that must be *added to the scores* somewhere. It is
not — not by the time it reaches attention. `T5Attention.forward` computes

```python
position_bias = self.compute_bias(...)          # [1, n_head, q, k]
position_bias = position_bias + causal_mask     # the SAME tensor carries the mask
...
scores += position_bias_masked
```

so bias and mask arrive as **one additive tensor**, in exactly the place `fuse_loom_attention` already
anchors on (`add(scores, mask)`) and in exactly the shape `ggml_soft_max_ext` already accepts: its mask
may carry a head axis, because it asks only that `a->ne[2] % mask->ne[2] == 0`.

T5's bias is therefore not a missing primitive. It is a **mask this engine could already take**, and
the only open question was who computes it.

## Decision

**The driver computes it, from a table the export writes into the driver as constants.**

* `relative_attention_bias.weight` — 32×6 = 192 floats per stack, two stacks — is read off the
  checkpoint in `phases()` and bound by `ExportConstants`, flattened `bucket * n_head + head`.
* `t5_position_bias` in `t5_driver/00_header.lua` is `_relative_position_bucket` in Lua, and folds the
  causal mask into the same array for the decoder.
* The traced graph takes that array as its `position_bias` input, and every stack is run by walking
  `T5Stack.block` directly so the bias is PASSED rather than computed — calling the stack would trace
  `compute_bias`'s `torch.arange(query_length)` at the dummy length and bake it.

Two consequences fall out for free. The decoder's mask input is a mask in the engine's own sense, so
`_retype_fused_mask_input` widens its key axis to `n_kv` with nothing new to check, and
`set_mask_tensor_padded` pads its bucket tail with `-inf` exactly as it does for
`loom.causal_mask` — a driver still does not have to learn what a bucket is.

## Options Rejected

1. **A new engine primitive.** A `REL_POS_BIAS` op computing buckets and gathering from a table. It is
   real work in the half of the tree that is meant to stay small ([Epic-01]'s lean-runtime principle),
   and it buys nothing the mask path does not already have.
2. **Gathering the bias in the GRAPH**, from an int32 bucket-index tensor the driver supplies. Keeps
   the table as a quantizable GGUF tensor and marshals one sixth as many numbers per call. Rejected
   for what it costs elsewhere: the tensor reaching the fused node is then an intermediate
   (`add(gathered, mask)`) rather than a declared input, so `_retype_fused_mask_input` has to walk a
   PATH back to two inputs and prove nothing else consumes anything on it — and an i32 input over
   `n_kv` needs a zero-padding write in the engine, since the existing one pads with `-inf` and an
   `-inf` cast to an index is not an index. Two changes to load-bearing code to avoid marshalling 192
   floats' worth of lookups.
3. **Folding the bias to a constant per length at export.** Forfeits dynamic length and re-invites the
   18.9 MB of zeros P4.28 removed from VITS.

## Consequences

* **Model WEIGHTS live in a driver script**, which is new. It is bounded and stated: 192 floats per
  stack, three orders of magnitude below any real tensor, and the same category of fact as Whisper's
  prompt table — a number only the checkpoint knows, which Lua cannot read from the GGUF
  ([ADR-006](adr-006-model-constants-belong-to-the-export.md)). A checkpoint with a materially larger
  table is what should reopen this.
* **The per-call cost is `n_head * n_tokens * n_kv` doubles across the Lua boundary**, against
  `n_tokens * n_kv` for an ordinary causal mask. On a decode step at `n_tokens = 1` that is
  `6 * n_kv`; on the encoder's single call it is `6 * n_src²`, which for a sentence is tens of
  thousands and for a 512-token source is 1.5M. If a longer-context leaf in this family makes that
  measurable, option 2 above is the written-down alternative.
* **A second family reusing this** gets the mask path for free. Any architecture whose positional
  information reaches attention as an additive term — ALiBi, T5-style buckets, Shaw-style
  windows — is a mask in this engine, and only needs a driver that can build one.
