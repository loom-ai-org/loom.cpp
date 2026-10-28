---
type: retro
date: 2026-10-28
domain: backends
tags: [vc4, device-selection, hybrid, repack, lowering, raspberry-pi-zero, quantization, validation]
---

# Retro-096: An Accelerator Primary Rewrote the Graphs It Did Not Touch

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (P4.44); the decisions are 6 and 7 of
[ADR-088](../adrs/adr-088-the-pi-zero-qpus-are-an-accelerator-beside-the-cpu.md).

## Issue

ggml-vc4 was validated on what it accelerates: F32 MUL_MATs, bit-for-bit in an emulator and within
tolerance through `ggml_backend_sched`, and Pocket-TTS 7m F32 end to end under `"auto"`. Then the
Pi Zero's shipped models -- Q8_0_PC, which the QPUs decline entirely -- ran under `"auto"`: amy 13.8 s
on the CPU against 27.5 s, VITS 19.8 s against 117.4 s with a different output length, and
`GGML_VC4_STATS` said the QPUs took **zero** MUL_MATs in both.

## Root cause

Two decisions in the engine were keyed on the PRIMARY backend, and both assumed a primary is a GPU,
for which the CPU is an expensive fallback:

* `GgufModel::load` placed weights in the CPU's extra buffer types only on a plain CPU primary, so a
  hybrid lost `CPU_ARMV6_I8` -- ADR-079's int8 kernels -- for every per-channel weight.
* `backend_can_run` (P4.7e) chose between a native op and a GPU-friendly composition by asking the
  primary alone. VC4 declines everything but MUL_MAT, so every lowering took the composition, and the
  CPU ran it: the folded quantized convolutions became im2col + mul_mat (93 s of the 112 s, profiled),
  and ATAN got its polynomial.

Fixing the first alone moved VITS from 117 s to 112 s, which is how the second was found -- the
profile, not the guess (a cache-attribute suspicion was measured out first: the buffers are cached).

## Takeaway

**A new device that `"auto"` will pick must be validated on the models it does NOT accelerate, not
only on the ones it does.** Becoming the primary changes more than op placement: weight placement and
graph lowering read it too. For a host-memory accelerator the CPU is not a fallback but the main
engine, so the CPU's answers have to keep applying. Before trusting such a device, run the shipped
model set `cpu` against `"auto"` on the board and require identical output wherever the device takes
nothing -- identity is the oracle there, and it would have caught this in one run.

## Record

* Pre-fix (P4.41 engine + ggml-vc4 via `GGML_BACKEND_PATH`, `~/p444/base`): amy-q8pc cpu 13.806 s
  `d31634a5`, VC4 27.483 s `2f0dc339`; vits-q8pc cpu 19.775 s n=75008, VC4 117.364 s n=75264.
* Repack fix only: vits-q8pc VC4 112.253 s, n=75008, still not identical; 66 weights in CPU_ARMV6_I8.
* `LOOM_PROFILE`, vits-q8pc: CPU arm `CONV_2D` 41+7+6+6+57 calls; VC4 arm none, `IM2COL` + `MUL_MAT`
  instead (MUL_MAT 32x75008 30.2 s, 64x18752 25.5 s, 384x293 18.3 s, 128x2344 12.1 s).
* `test_mem_diag`: dma-heap block 0.107 ms per 4 x 16 KB read-modify-write, uncached 0.865 ms.
