---
type: retro
date: 2026-10-29
domain: backends
tags: [vc4, qpu, tmu, cache, stride, aliasing, raspberry-pi-zero, measurement]
---

# Retro-099: A Slowdown Blamed on the Dimension That Was Large

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (P4.46); the decision is
[ADR-092](../adrs/adr-092-the-qpus-take-the-epilogue-q4-0-and-the-pointwise-convolutions.md).

## Issue

P4.44 measured one MUL_MAT, 128 x 3656 x 384, at 2.2 GFLOPS where its neighbours ran at 9-11, and
filed it as "the tile order for large N" (ADR-090, carried by ADR-091 and the P4.45 handover). P4.46
built the fix that diagnosis asked for -- panels of column blocks, so the packed X a pass streams
through the uniform caches stays bounded -- and measured it: m-major 2.21 GFLOPS, panels of 1, 2, 3, 4,
6 and 12 between 1.93 and 2.20. Nothing moved.

## Root cause

Not N. The weight's row stride: K = 384 is 1536-byte rows, and every k the 16 lanes gather one word
from each of 16 rows, whose cache lines then fall into two of the TMU cache's sets. The sweep that
showed it varied the dimension nobody suspected, one at a time:

| M x N x K | GFLOPS |
|---|---:|
| 128 x 3656 x 384 | 2.21 |
| 128 x 1024 x 384, 128 x 457 x 384, 192 x 3656 x 384 | 2.21, 2.18, 2.21 |
| 128 x 3656 x 368 / 400 / 416 | 9.42 / 9.33 / 9.38 |
| 128 x 3656 x 512, 128 x 1024 x 1024 | 1.53, 0.80 |

Every row stride that is a multiple of 512 bytes was slow (K = 384, 512, 768, 1024, 1152 in F32), and
64 bytes of padding fixed each one: K = 384 9.1, K = 768 (distilbert's 768 x 768) 2.05 -> 9.98,
K = 1024 0.80 -> 7.1. The N it was named after made no difference at all. The panel order the wrong
diagnosis produced did turn out to be worth keeping, but for another reason: once the strides are
clean, panels of two run 1.17-1.36x faster than m-major everywhere.

## Takeaway

**A slow case is not explained by what is unusual about it until the other dimensions have been
varied.** "Large N" was the conspicuous property of the one slow shape, so it became the cause; one
sweep that held N and moved K, then held K and moved N, would have shown the opposite in minutes,
before anything was built on it. When filing a performance anomaly, record the one-at-a-time sweep or
say it has not been done -- "128 x 3656 x 384 is slow" is a fact, "large N is slow" was a guess wearing
its clothes.

## Record

* Board, `test_p446_hw order M N K panel 12 [pad]` (gemm_f32_wt_kernel, 12 QPUs), full table in
  `~/.claude/tmp/p446` (hwb.log, hwc.log). Spot checks 0/64 wrong in every run.
* Panels, unpadded 128 x 3656 x 384: 0 -> 2.21, 1 -> 1.93, 2 -> 2.19, 3 -> 2.12, 4 -> 2.19, 6 -> 2.18,
  12 -> 2.20. Padded: panel 0 9.14, panel 2 12.30.
* Padding where the stride is not a multiple of 512 changes little (K = 192: 8.88 -> 8.09; K = 704:
  7.15 -> 8.45), so `gemm_row_stride` pads only the multiples.
