---
type: retro
date: 2026-10-01
domain: exporter
tags: [exporter, topology, reduce-mean, layout, concat, ecapa, family-13, bisect]
---

# Retro-069: A Mean That Kept Its Axis Interleaved a Concat

## The Issue

ECAPA-TDNN's language id (family 13, `speechbrain/lang-id-voxlingua107-ecapa`) exported, ran, and named
the right top-1 language on every clip. Its probabilities were up to **0.10** away from speechbrain's.

A stage-by-stage bisect (one truncated export per stage, each compared with torch) put every stage
through the last ECAPA block at ~1e-5 relative, and the attentive pool at **0.23**. Inside the pool,
both statistics matched torch to 1e-6, and the matmul that consumes them was **0.95** off.

## Root Cause

The pool takes `x.mean(dim=2)` and `std` (both `[B, C]`, `keep_dims=False`), concatenates them along
`dim=1`, and multiplies. The exporter lowers a dynamic-count mean over `ne[0]` to ggml's `MEAN`
(`loom_mean`). **ggml's MEAN always leaves the reduced axis in place at size 1**, so its output is
`[1, C, B]` in ne order where MIL's is `[C, B]`. The CONCAT was emitted for MIL's shape, `dim 0`, and
ran along the leftover unit axis. It **interleaved** mean and std (`m0, s0, m1, s1, ...`) instead of
appending them.

Nothing could see it from either end. The flattened bytes of each mean are identical with and without
the unit axis, so comparing the means alone passes, which is exactly what the bisect's first answer
was. The CONCAT's shapes were legal. Only the first consumer that reads the layout (the matmul) was
wrong, by its whole magnitude.

The engine's own `REDUCE_SUM` drops the reduced axis when `keep_dims` is false, for this reason, and
says so in a comment. `loom_mean` had been written for the `keep_dims=True` shape and never read the
flag.

## The Fix

`topology_ops._op_loom_mean` emits `MEAN` and then a `RESHAPE` to MIL's own output shape when
`keep_dims` is false (loom-exporter `feat/p5-family-13`). ECAPA then matched speechbrain to **1.9e-6**
at five lengths.

**Blast radius, measured:** four published GGUFs contain a `MEAN`. data2vec-audio, HuBERT-large and
Qwen3-TTS's speaker encoder all take `keep_dims=True` (their consumers are a SUB, an ADD and a CONV_1D,
which could not broadcast otherwise), so their topologies do not move. **StyleTTS2's** diffusion
transformer takes `x.mean(axis=1)` into a `MUL_MAT`. It was re-exported: the only difference in the
file is that one `RESHAPE([1024, 1])`, every weight is identical, and its synthesis is **bit-identical**
to the published file's (78,000 samples, max |Δ| 0). It had been running correctly through the
engine's layout healing in `op_mul_mat`. Nothing needs republishing.

Pinned by `tests/ci/test_audio_classification_export.py` (a mean feeding a concat is reshaped; a
`keep_dims=True` mean is untouched), which goes red with the fix reverted.

## Takeaway

**A comparison of flattened values cannot see a layout bug: check the first consumer that reads the
layout.** The bisect had the right answer one stage earlier than it looked. The means were "right" and
the concat of them was wrong, so the stage boundary to compare at is AFTER an op that depends on
which axis is which (a concat, a matmul, a reshape that changes the rank). **A lowering whose engine op
keeps a reduced axis must drop it explicitly when MIL does.** The engine's `REDUCE_SUM` already did;
any new reduction should be read for the same flag. And it is one more case where a healer made a
wrong layout harmless in one model (StyleTTS2) and therefore invisible, which is the argument the
layout-healing audit ([Retro-001](retro-001-layout-healing-heuristics.md)) already makes.
