---
type: retro
date: 2026-09-11
domain: exporter
tags: [tracing, coremltools, numerics, oracles, verification, family-11]
---

# Retro-042: `reciprocal` Meant `1/(x + 1e-4)`, and It Looked Exactly Like Float Noise

## The Issue

SNAC's first export matched the reference to **cosine 0.999998** on real speech, decoded to the exact
sample count, transcribed 22/22 words through the ASR oracle, and was wrong.

The tensor oracle read `max|Δ| 6.8e-3` against `rms 0.15` — 9.2e-3 in relative RMS. Large for this
project (family 12 shipped at 1.2e-5) but not obviously *broken*: a 29-layer convolutional decoder
with four transposed-convolution upsamplings is exactly the kind of stack where "it is f32, and ggml
sums in a different order" is the expected answer.

It was not the expected answer. `Snake1d`'s `(alpha + 1e-9).reciprocal()` had been folded into a
constant that was **1/(alpha + 1e-4)**.

## Root Cause Analysis

coremltools' torch frontend lowers `aten::reciprocal` to MIL's `inverse`, whose `epsilon` input
**defaults to 1e-4** and is added to the operand: `y = 1 / (x + epsilon)`. torch's own reciprocal adds
nothing. So every `.reciprocal()` that reaches this pipeline computes a different function than the
model does.

It is silent three times over:

* the operand is a parameter, so the whole expression **const-folds into a weight** — no op, no shape
  and no error survives to be inspected;
* 1e-4 is small enough that the output stays plausible — the error here was 2.3e-4 relative on the
  constant, and it grows without bound only as `alpha` approaches the epsilon;
* `compute_precision=ct.precision.FLOAT32` is already set, so the obvious suspect for a 1e-3-scale
  discrepancy — an fp16 cast — was already ruled out and pointed away from the real cause.

The other two `epsilon`-carrying unary ops are harmless: `rsqrt` defaults to 1e-12 and `log` to 1e-45.
`inverse` is the outlier, and 1e-4 is not a rounding-scale number.

This is the first `.reciprocal()` to reach MIL in this project. Kokoro's Snake has the same `1/alpha`
and never hit it — its converter is one of the hand-written pre-MIL ones, which folds the reciprocal
in numpy.

## The Arm That Turned "Noise" Into "Bug"

The decisive measurement was not on the engine at all. **Run the reference model against itself at
double precision**: same ops, same order, same weights, one more mantissa.

| | max abs diff | relative RMS |
|---|---|---|
| torch f32 vs torch f64 | 1.8e-6 | — |
| loom vs torch f32 (before) | 6.8e-3 | 9.2e-3 |
| loom vs torch f32 (after) | 2.5e-6 | 1.9e-7 |

f32-vs-f64 answers "how much can rounding move this computation" **for this graph**, which is the
question a relative-error number by itself cannot answer. 1.8e-6 said the computation is
well-conditioned, which made the 6.8e-3 a 3700x gap and therefore a defect. After the fix loom lands
at 2.5e-6 — the same order as the reference's own f32 spread, which is where a correct export belongs.

Localising it after that was a depth bisect: export the decoder truncated after *k* modules and
compare the tensor. The gap first appeared at the first `DecoderBlock`, then at its first module — a
single `Snake1d`, at 1.8e-4 relative for one activation. From there the folded constant was one
`np.abs(got - 1/(alpha + 1e-9)).max()` away.

## Takeaways

* **A converter's op DEFAULTS are part of the model.** An op-by-op mapping can be complete, named
  correctly, and still compute a different function, because the target op takes a parameter the source
  op does not have. Reading the target op's docstring is part of trusting the lowering.
* **A relative error is not a verdict until you know the conditioning.** "It is f32" explains an error
  only if f32 can produce it. One f64 run of the reference answers that, costs nothing, and is now the
  first thing to reach for when an oracle reads worse than expected.
* **The word-level oracle cannot see this class of bug.** 22/22 words, correct length, correct sample
  rate, no clipping — all true of a decoder computing a slightly different function. It stays the test
  for *is it audio*, and it is not the test for *is it this model*.
* The fix is in `torch_patches.py` (patch 3, `reciprocal` → `inverse(epsilon=0)`), so it protects every
  future export rather than SNAC's. `tests/ci/test_torch_patches.py` pins both halves: that the folded
  constant is `1/x`, and that the default being overridden is still 1e-4 — so if coremltools ever
  changes it, the patch does not quietly become wrong.
