---
type: retro
date: 2026-11-17
domain: backends
tags: [ggml-vc4, fusion, views, padding, pi-zero, sanotts, export-shape-change]
---

# Retro-109: A Cropped View Passed the VC4 Fusion Matcher

Serves [Epic-05](../epics/epic-05-edge-performance.md); found re-benchmarking sanoTTS for
[ADR-134](../adrs/adr-134-tiny-voca-splits-its-pulse-train-at-the-frame-and-draws-torch-noise-in-the-engine.md).
The matcher is [ADR-092](../adrs/adr-092-the-qpus-take-the-epilogue-q4-0-and-the-pointwise-convolutions.md)'s; the padding is
[ADR-081](../adrs/adr-081-the-exporter-pads-weight-axes-to-what-the-kernels-need.md)'s.

## Issue

Re-exported for ADR-134, sanoTTS heart-nano aborted on every Pi Zero run with `device="auto"`:
`GGML_ASSERT(... ggml_nelements(target) == ggml_nelements(conv))` in `conv_f32_run`. The file published
on the Hub since rc19 aborts the same way, on the P4.47 build the chart was measured with and on today's
(a released wheel was not tried). The README chart's heart-nano row never saw it: it was measured on an
rc18-era file.

## Root cause

ADR-081's padding pass rounds a convolution's channels up to what the kernels want: heart-nano computes
some at 32, adds the bias at 32, and VIEWs 31 back out before the residual ADD. ggml-vc4's matcher
(`operand_of`, and `step_of` on the support side) accepted an ADD when the value it reads has the ADD's
own element count -- and a 31-channel crop of a 32-channel result has exactly that. So it fused the
residual onto the convolution, and the fused result was 31 channels' worth of a 32-channel
convolution. Where the result goes straight to the ADD's tensor the size check caught it. Where it goes
through the convolution's own memory, the check compares the convolution with itself and passes.

## Takeaway

**An element count is not an identity.** A view's consumer can match its count while the view crops
what it views; a fusion that writes the producer's whole result must check the value against the
producer (`whole_of`: the view covers all of what it views). And **a pass that changes a graph's
shapes ships with a run on every backend that pattern-matches graphs**: the padding pass was gated on
the CPU, and its published files met the VC4 matcher first on a user's board.

## Record

* Old (rc18-era, unpadded) heart-nano on the VC4: runs, on the P4.47 build and on today's. The published
  rc19/rc20 file and today's export: abort on both. Same topology node count; the differing tensors are
  the padded ones (`synth._104_weight_0` 5x31x31 -> 5x32x32, ...), with a `VIEW [n, 31, 1]` after each
  bias ADD.
* `GGML_VC4_DEBUG=1`: `fused CONV_2D +bias +residual -> node_25` right before the abort.
* Fix (`ggml-vc4.cpp`, `whole_of`), on the board (ARMv6 build): `test_vc4_backend`'s new case "padded
  conv+bias, 31 of 32 viewed out, +residual" aborts unfixed and passes fixed (7.3e-5 from the CPU,
  3.2x); every case passes plain and with the arena capped (37 OK). heart-nano runs on the VC4; heart's
  VC4 output is 59 dB from its CPU's, as before; tiny-voca's VC4 output keeps its hash.
