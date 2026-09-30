---
type: retro
date: 2026-09-30
domain: exporter
tags: [exporter, lowering, get-rows, shape-walk, family-9b, speecht5]
---

# Retro-066: A Matrix Index, and a Batch Guess That Answered 1

## The Issue

SpeechT5's first export wrote, loaded and built all five topologies, then aborted the process at
its first call: `GGML_ASSERT(a->ne[2] == b->ne[1])` in `ggml_get_rows`. The first fix exported and
loaded too, and then failed at the same call with `RESHAPE: target shape [64,1,1,] has 64 elements
but input has 246016`. **Two defects in series, and neither raised during the export.**

## Root Cause

SpeechT5's encoder adds a Shaw-style relative position bias, `q . pe_k[clip(i - j) + 160]`. The
wrapper gathered `pe_k` rows by an `[n, n]` index matrix, which is the reference's own spelling.

1. **`ggml_get_rows` is a batched lookup.** It gathers one index ROW per batch of a 3-D table, so for
   a 2-D table every index axis except the innermost must be 1. The exporter's generic `gather ->
   GET_ROWS` lowering checked neither the index rank nor the table rank, so a rank-2 index lowered
   without complaint. The engine then asserted, and an assert cannot be caught.
2. **The shape walk answered 1 for `rel_index.shape[0]`.** The fix flattened the index, gathered, and
   reshaped back with `.view(n, n, 64)`, reading `n = rel_index.shape[0]`. `value_facts`' batch
   guess reads a torch axis 0 whose derivation comes out as the ROOT axis as a batch size of 1. That
   guess exists because the walk's own fallback for "I don't know" is also the root axis, so the two
   cannot be told apart. Here axis 0 genuinely was `n_tokens`, and the target shape became
   `[64, 1, 1]`.

## The Fix

* The wrapper reads `n` off `position_ids.shape[1]`, an axis 1, and reshapes with
  `pe_k(rel_index.reshape(n * n)).view(n, n, 64)`. The emitted target is `["64", "n_tokens",
  "n_tokens"]`, and `test_speecht5_export.py` asserts exactly that. With `rel_index.shape[0]` put
  back, that test fails with `['8', '1', '1']`.
* The `gather` lowering now **warns** at export time about a multi-row index into a 2-D table, and
  names the fix: "Flatten the index in the wrapper, gather, and reshape the rows back."
  `test_tile_lowering.py` covers both spellings. **A warning, not a refusal**, because the first
  version refused and broke Dia's export. Dia's multi-channel embedding gathers a `(1, T, C)` index
  from its one table, and it runs only because the decode loop never calls it with T > 1. That is
  the same latent abort, kept from firing by a length no export-time check can see. The real fix
  flattens inside the engine's `op_get_rows` (or in the lowering), and it is on the hub.

The batch guess is unchanged. It is still right far more often than it is wrong (see its own
docstring and the GigaAM counterexample it already carves out). What changed is that a new wrapper
does not ask it the ambiguous question.

## Takeaway

**A lowering that passes an op through must check the op's preconditions, not only its arity.**
`GET_ROWS` pruned the `axis` input and kept going; the rank was the part that mattered. When a ggml
kernel asserts where the engine's primitive layer could have thrown, the exporter is the last place a
readable error is possible.

**Read a dynamic extent off an axis that is not 0.** An axis-0 read that derives to the root axis
means "batch" to the walk. This is the third time a reshape target has been the only symptom
([Retro-051](retro-051-a-negative-begin-doubled-the-slice.md)'s slice, the `-1` in
Qwen3-TTS's mask, now this). After any export with a dynamic length, read the emitted RESHAPE and
VIEW targets before running anything.

## Related

* [ADR-057](../adrs/adr-057-a-char-sentencepiece-model-ships-as-unigram.md), SpeechT5's other export
  cost.
* [Retro-063](retro-063-an-expand-as-was-lowered-as-an-identity.md): the same class, a lowering that
  read a live input as a constant, and an engine assert as the only symptom.
