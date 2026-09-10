---
type: retro
date: 2026-09-10
domain: exporter
tags: [tracing, coremltools, attention, family-6, gates]
---

# Retro-041: Two Transposes Merged, the Fusion Matched Nothing, and Nothing Said So

## The Issue

`fuse_loom_attention` matched **zero** blocks in T5's decoder. Not some — none. The export still
succeeded, the topology was still valid, and the model would still have produced correct output: an
unfused attention block runs as an expanded softmax, just without a KV cache, so a decode loop would
have recomputed the whole sequence at every step and answered correctly while doing it.

What made it visible was a check written for a different reason. `_check_fused_attention` exists to
assert that CROSS-attention did **not** fuse (its zero bias is folded away by a coremltools pass, not
by anything in this tree, and a release that stopped folding it would hand every cross block a KV
cache slot the self-attention blocks address). Counting is symmetric, so the same assertion caught the
opposite failure — 0 fused where 8 were expected.

## Root Cause Analysis

`fuse_loom_attention` matches `matmul(q, k, transpose_y=True)`. No HF attention writes that; they all
write `torch.matmul(q, k.transpose(-1, -2))`, and MIL's own `fuse_transpose_matmul` folds the trailing
transpose into the flag. That pass folds a permutation of the **last two axes only** — and
`merge_consecutive_transposes` runs first.

In T5 nothing sits between the head-split `.transpose(1, 2)` and the score `.transpose(3, 2)`, so the
two merge into a single `[0, 2, -1, -3]`, which is no longer last-two-dims and no longer foldable.
Qwen3 and Whisper are unaffected for a reason that has nothing to do with attention: RoPE sits between
their two transposes. T5 has no positional projection at all, which is the same fact that gave it a
relative bias in the first place.

Two smaller things fell out of the same fix and are worth recording:

* The merged perm comes back with **negative axes** (`[0, 2, -1, -3]`, not `[0, 2, 3, 1]`), because
  the second transpose was written against the end of the shape. The first version of the recovery
  compared against the positive spelling and matched nothing — the same total miss, one layer down.
  coremltools' own pass canonicalizes before comparing, which is what pointed at it.
* T5's cross-attention K/V are `num_heads * d_kv` wide (384) and its residual stream is `d_model`
  (512). Declaring them at `d_model` traced and exported cleanly and failed at the first decode step,
  on the engine's own retained-output-vs-input shape check. The unit fixture had `d_kv * num_heads ==
  d_model` by accident and could not have caught it; it now sets them deliberately unequal.

## What Changed

* `fuse_loom_attention._key_before_merged_transpose` recovers K's pre-transpose var and re-splits it
  into the `[b, heads, seq, dim]` layout `op_attention` reads, when and only when `transpose_y` is
  false and the perm is exactly the merged one. Every guard bails to "leave it alone", which is the
  rule the rest of that pass follows.
* `t5_export._check_fused_attention` asserts the count in both directions, and the family's CI test
  drives it with 4 and with 2-but-uncached.

## Prevention

**A pass that is documented to leave what it does not recognise exactly as it was cannot report a
total miss, so a family that DEPENDS on it has to count.** `fuse_loom_attention`'s "a partial match
must never half-rewrite" rule is right and is not the issue: bailing safely is what makes the pass
sound. The gap is that "matched nothing" and "this model has nothing to match" are the same output,
and only the family knows which it expected.

`ExportPhase.fuse_attention` is explicitly declared UNCHECKED against whether the pattern then
matched — deliberately, because it is a request rather than a claim. That is defensible for a family
whose export is merely slower without fusion. It is not enough for one whose decode loop is built on
the cache, and the cheap remedy is a `topology_rewrite` that rewrites nothing and counts instead.
