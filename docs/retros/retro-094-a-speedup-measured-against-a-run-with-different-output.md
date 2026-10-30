---
type: retro
date: 2026-10-28
domain: backends
tags: [vulkan, v3dv, raspberry-pi-4, benchmarking, measurement, a-b]
---

# Retro-094: A Speedup Measured Against a Run With Different Output

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md#7-p443-v3dv-on-the-raspberry-pi-4-2026-10-28)
(P4.43); the decision is [ADR-089](../adrs/adr-089-v3dv-is-correct-and-opt-in.md).

## Issue

A V3D-specific conv2d tile (256 invocations, `TS_K = 2`) was credited with taking VITS on the Pi 4's GPU
from 93.9 s to 65.9 s, 1.42x, and was written into the patch with that number in its comment. An
interleaved A/B of the tile alone, same build, same input, a 60 s cool-down between arms, put it at
**1.5% slower** (65.93 s against 64.92 s, both rounds).

## Root cause

The 93.9 s run was not the same work. It came from the build before the CONV_2D_DW fix, whose VITS
durations were wrong: it synthesised 116 992 samples, the fixed build 74 240 (1.58x the audio, and the
vocoder convolutions scale with it). Two things changed between those runs -- the tile and the fix that
changed the output length -- and the whole difference was credited to the one being looked at. The
spill counts halving looked like confirmation; they were real and did not matter.

## Takeaway

**A before/after timing is only a comparison if the two runs produced the same output.** Check the
output (length, then values) before reading the times, and attribute a change only through an A/B that
toggles that change alone on one build. Here the output check existed in the harness and was not read
before the number went into a comment.

## Record

* perf1 (wg256 build, CONV_2D_DW still 512 wide): `cpu (74240,) vk (116992,)`, VITS 93.9 s/run.
* perf2 (conv2d_dw fixed + V3D tile): `cpu (74240,) vk (74240,)`, max abs diff 1.02e-5, 65.9 s/run.
* perf3 A/B (`SCRATCH_V3DV_CONV_DEFAULT` switch, scratch build only): tiled `[65.937, 65.928]`,
  `[65.943, 65.916]`; default `[64.934, 64.922]`, `[64.908, 64.934]`; all four outputs within 1.02e-5 of
  the CPU. The tile was removed from `ggml-0055`.
