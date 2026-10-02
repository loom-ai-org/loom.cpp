---
type: retro
date: 2026-10-02
domain: exporter
tags: [exporter, broadcast, rope, decoder, prefill, range, gather, moonshine, family-2, oracle]
---

# Retro-070: A One-Token Decode Step Hid a Two-Sided Broadcast

## The Issue

Moonshine Streaming's first full export transcribed jfk.wav with token ids identical to
transformers'. The tensor oracle (encoder output and teacher-forced logits, through the engine) then
**aborted** on the decoder: `MUL: incompatible shapes a=[32,1,1,1] b=[1,27,1,1]`.

Two earlier aborts on the same export had the same shape of cause:

* the encoder's sliding-window mask, `idx[:, None] - idx[None, :]`: `SUB: incompatible shapes
  [1,550,1,1] [550,1,1,1]`;
* the encoder-position table, `pos_emb(arange(n))`: `GET_ROWS: the index must be I32, got f32`.

## Root Cause

**The engine's elementwise ops broadcast one operand into the other, never both.** MIL and torch both
broadcast `[n, 1]` against `[1, m]` to `[n, m]`; ggml's binary ops repeat `b` into `a`'s shape and
refuse anything else. A product of a column and a row is therefore an abort, not a wrong answer -- but
only at a shape where both sides are wider than one.

The RoPE angle, `position_ids[..., None] * inv_freq`, is `[t, 1] * [32]`. **In the decode loop `t` is
always 1** -- the prompt is `<s>` alone and every step feeds one token -- so the "two-sided" broadcast
was one-sided at every call the driver makes, and every transcript was right. The teacher-forced pass
was the first call with `t > 1`.

The gather is a separate mismatch: MIL types an integer `range_1d` int32, so no `cast` exists for the
exporter to lower, while the engine's `RANGE_1D` always builds F32. Every earlier range fed float
arithmetic (`arange(T) < length` masks), which wants F32; this was the first to index a gather.

## The Fix

* The window mask's `q - k` and the RoPE angle are outer products, `torch.matmul` with an inner
  dimension of 1 or 2 -- one MUL_MAT each, exact (one product per output, integers below 2^24 for the
  mask). transformers itself spells the angle as a matmul.
* A gather whose indices are a `range_1d` reads an I32 copy (`_gather_by_range`); every other reader
  keeps the F32 range, so no other topology moves. None of the 34 GGUFs in the staging tree has that
  edge (scanned).

Pinned by loom-exporter `tests/ci/test_moonshine_export.py` (the gather test goes red with the rule
disabled).

## Takeaway

**An autoregressive decoder export is not verified until a multi-token prefill has run through the
engine.** A loop whose every step is one token never exercises the token axis at a width above one,
so any op that mis-broadcasts, mis-reshapes or mis-masks along it passes every transcript. The
teacher-forced logits pass -- the whole reference sequence as ONE decoder call, run through a probe
script loaded beside the driver -- is the cheapest call that widens it, and it is also the tensor
oracle that matching token ids never replaces. One probe answers both. And a broadcast in a wrapper
should be read for which side is wide: if both can be, write it as a matmul.

## Recurrence (2026-10-02, the same day): Kyutai STT's RMSNorm

The takeaway was followed and caught it again. Kyutai STT's LM steps one token per audio frame, so its
re-spelled RMSNorm, `x * (alpha * rsqrt(mean(x^2)))` -- `[dim]` against `[1, n, 1]` -- passed every
transcript; the 40-token teacher-forced prefill aborted on `MUL: a=[2048,1,1,1] b=[1,40,1,1]`. One new
detail for the rule above: **`expand_as` is not a fix**, because MIL folds a broadcast away and the
two-sided MUL comes back. `rsqrt(...) + x * 0` carries x's shape by arithmetic (exact for finite x) and
each multiply then broadcasts one operand. `kyutai_stt_export._RMSNorm`; the gate's prefill arm keeps it.
