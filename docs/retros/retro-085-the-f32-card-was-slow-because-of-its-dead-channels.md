---
type: retro
date: 2026-10-14
domain: performance
tags: [armv6, pi-zero, vfp, denormals, runfast, citrinet, batchnorm, benchmark-baseline]
---

# Retro-085: The F32 Card Was Slow Because of Its Dead Channels

## The Issue

On the Pi Zero W, Citrinet-256-ls ran jfk in 64 s from its F32 card and 37.5 s from its Q8_0 card.
That was read as "the quantized card is the fast one on this board", and ADR-077 took the Q8_0 card's
37.5 s as its baseline. Its gate-1 profile blamed 76% of the time on the 1x1 convolutions, running at
~0.22 GFLOP/s, "the board's GEMM ceiling".

Gate 2's bench then ran ggml's own F32 conv, the same `CONV_2D` op the F32 card runs, on the same shapes
in 14.4 s. The F32 card's profile spends 52.3 s on those 131 nodes. Per call the two agreed at T = 1101,
and the card was 3-5x slower at T <= 551, which no FLOP count explains.

## Root Cause Analysis

**Denormal arithmetic, which the ARM1176's VFP11 traps to software in its default IEEE mode.** The F32
card has 137,472 weights below 1e-20 and 17,989 below 1e-30. They sit in the 1x1 kernels: channels whose
BatchNorm scale trained to ~0, folded into the conv at export, leave weights at ~1e-33. Any product of
one with an ordinary activation underflows to a denormal, and each such product costs microseconds.
Zeroing those weights recovers only part of the time (below), so activations that reach the denormal
range carry the rest.

* The Q8_0 card escaped it by accident. Q8_0 rounds a ~1e-33 weight to exactly 0, and a product with an
  exact 0 never traps. "Quantization is faster here" was partly "quantization flushed the denormals".
* The bench missed it at first, because its Gaussian weights have no dead channels.

What settled it, jfk on the board, same transcript ids in every arm: zeroing every weight below 1e-20 in
a copy of the F32 card took it from ~61 s to ~44 s. Setting RunFast (FPSCR `FZ | DN`) on the calling
thread from Python took it to **20-22 s**, past the Q8_0 card, which went from 33-35 s to 29-31 s.

## Resolution & Lesson Learned

Nothing is patched yet. [ADR-078](../adrs/adr-078-runfast-on-armv6-compute-threads.md) carries
RunFast as a gated decision across the ARMv6 zoo, and
[ADR-077](../adrs/adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md#found-while-measuring-gate-2-the-f32-card-was-paying-for-denormal-traps)
moves its baseline to F32 in RunFast.

**Lesson: on a core that traps denormals, a precision comparison measures the weights' tails as well as
the arithmetic.** Before calling one card faster than another on ARMv6, or a kernel at "the board's
ceiling", count the weights below ~1e-20 and run the slower arm once in RunFast. A gap that survives
RunFast is about the kernels; one that doesn't is about denormals. A per-call time that does not scale
with the work is the cheap first sign. Bench inputs drawn from a Gaussian cannot reproduce this.
Related: the masked-softmax case in ggml-0029, where a `-inf` made XNNPACK's exp 18x slower on the same
core.
