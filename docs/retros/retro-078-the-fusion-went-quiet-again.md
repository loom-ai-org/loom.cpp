---
type: retro
date: 2026-10-09
domain: exporter
tags: [exporter, attention, kv-cache, fusion, musicgen, family-14, gates]
---

# Retro-078: The Fusion Went Quiet Again, Because the Prevention Was a Convention

## The Issue

MusicGen's first export (family 14, 2026-10-09) succeeded, loaded, and had **no cached attention at
all**. The decoder carried 48 bare `SOFTMAX` nodes and zero `ATTENTION` nodes. In the driver's decode
loop each step runs one row at `n_past > 0`, so without a cache every step attends to itself alone. The
result is plausible codes and no error. Nothing in the export said so, and it was caught only by reading
the op counts before the first run.

This is [Retro-041](retro-041-two-transposes-merged-and-the-fusion-went-quiet.md)'s failure exactly:
`fuse_loom_attention` matched nothing, and "matched nothing" and "nothing to match" look the same.

## Root Cause

Two things in transformers' MusicGen each break the pattern the pass anchors on,
`add(matmul(q_scaled, k^T), mask)`:

1. **The scale is on the scores, not on Q.** MusicGen's `eager_attention_forward` computes
   `matmul(q, k^T) * scaling` and then adds the mask, so a `mul` sits between the matmul and the add.
2. **Its mask builder inverts a 4-D mask.** Dia's builder passes an additive 4-D mask through untouched.
   MusicGen's eager path calls `_prepare_4d_causal_attention_mask`, which reads a 4-D mask as 0/1 and
   computes `(1 - mask).masked_fill(min)`. The mask that reaches attention is therefore computed rather
   than being a graph input. It is also inverted.

Fixing (1) alone turns the silent miss into a loud one: the exporter refuses a cached ATTENTION whose
mask is not a declared input. Together, the pass just declined every block, as it is documented to.

**Why it recurred:** Retro-041's prevention was per family. "A family that DEPENDS on [fusion] has to
count", with `t5_export._check_fused_attention` as the tool. T5, Canary, Moonshine and SpeechT5 adopted
it; Dia and Whisper never did; and MusicGen, written by copying Dia's module, did not either.

## What Changed

* **`phase_conversion.convert_phase` now refuses every phase that declares a `kv_cache_size` and fused
  no cached ATTENTION node.** `fuse_attention` stays a request. `kv_cache_size` is a promise that a
  decode loop will run the graph one step at a time against a cache, and with no cached block that
  promise is always false. All 21 phases that set it are attention decoders. A CI test drives the
  guard with a block scaled on the scores (refused) and the same block scaled on Q (fuses), and goes
  red with the guard disabled.
* **MusicGen** patches its own `eager_attention_forward` to scale Q first. That is bit-identical at
  head_dim 64, since 1/8 is a power of two, and the export refuses any head_dim where it would not be.
  It also bypasses `_update_causal_mask` so the additive mask reaches attention as passed. The shared
  pass was not loosened: matching `mul` after the matmul would also catch masked cross-attention in
  other fused families and hand it a cache slot.
* MusicGen's decoder also counts its fused blocks (`_check_fused_attention`, now with a family-neutral
  message), and its CI test goes red with either of its fixes removed.

## Takeaway

**A prevention that each new piece of code has to remember to opt into is not a prevention, and the
next new piece of code is the one that forgets.** Retro-041 had the right diagnosis and the right
check, and put it where only the families written after it, by someone who had read it, would get it.
When a failure is silent and the condition that makes it a failure can be stated without knowing the
family ("this phase promised a cache"), check it once in the shared path.

The second lesson is narrower: **a model's own mask builder is part of the model.** "Pass a 4-D additive
mask and the builder passes it through" was true for Dia and false for MusicGen, both transformers
models. Read the builder the decoder actually calls before assuming a mask trick carries over.
