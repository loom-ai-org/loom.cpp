---
type: retro
date: 2026-11-05
domain: exporter
tags: [mil, lowering, conv2d, log1p, zipformer, silent-failure]
---

# Retro-103: An Argument the Lowering Did Not Read

Serves [Epic-03](../epics/epic-03-model-coverage.md) (Zipformer) and
[Epic-02](../epics/epic-02-mil-exporter-and-compiler.md).

## Issue

The first Zipformer export failed in the engine on a RESHAPE that would not divide, just after its
subsampling. With that fixed it ran end to end and returned an empty transcript for every clip: the encoder
output was all NaN, and an argmax over NaNs lands on the blank.

## Root cause

Two lowerings each ignored part of what the MIL op said, and neither model before this one could tell.

* **A 2-D convolution's attrs were in torch's axis order.** `s0`, `p0` and `d0` came from torch's first spatial
  axis, while ggml's axis 0 is the width, torch's last. Every 2-D convolution in the zoo had the same values on
  both axes, so the swap was invisible. Zipformer's `padding=(0, 1)` and `stride=(1, 2)` made the frequency
  axis the wrong width.
* **`log1p` lost its `+ 1`.** coremltools lowers it to `log(x, epsilon=1.0)`, where the op means
  `log(x + epsilon)`. The engine's LOG takes one input and the generic path forwarded only `x`. So
  `log1p(exp(-d))` became `log(exp(-d))`, which is `-d`, off by tens per feed-forward, and `log(0)` where the
  exp underflowed.

The second was found by exporting truncated copies of the encoder and comparing each on the engine with torch:
attention weights agreed to 1.8e-6 and `feed_forward1` was off by 18.5, and the node list then read `EXP, LOG`.

## Takeaway

**A lowering that forwards some of an op's arguments is a silent wrong answer waiting for the first model that
sets the others.** The generic OP_MAP path passes inputs through, so an argument with a harmless default looks
handled. `generate_graph_topology` now refuses a `log` whose epsilon is not rounding-sized. Other ops with
defaulted arguments have not been audited.

**An empty transcript is a numeric symptom.** A blank-heavy decoder hides NaN as silence. Compare a tensor
before reading the tokens.

Both fixes are covered in `tests/ci/test_zipformer_export.py`, and those tests fail with the fixes reverted.
The convolution fix changes no published file: a scan of the 71 models on the Hub found 9 with a 2-D convolution
and none asymmetric. The `log1p` fix has no such scan, because a dropped epsilon leaves no mark in a GGUF. Reading
the reference code available locally found `log1p` only on paths these exports do not run, such as beam search;
that reading does not cover every tree.
