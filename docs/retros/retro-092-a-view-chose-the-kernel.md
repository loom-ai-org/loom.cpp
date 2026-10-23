---
type: retro
date: 2026-10-23
domain: performance
tags: [attention, ggml, mul_mat, layout, armv6, pi-zero, profiling]
---

# Retro-092: A View Chose the Kernel

Serves [Epic-05](../epics/epic-05-edge-performance.md) (P4.41); the decision is
[ADR-085](../adrs/adr-085-a-windowed-attention-computes-only-its-band.md).

## The Issue

The windowed `ATTENTION` cut Mimi's attention arithmetic by ~8x, and the first Pi Zero run of the 2m model
came in at 11.2 s where the projection said ~9.5. The node profile put the attention at 2.1 s, not ~0.5:
each block's `Q @ K^T` (925k MACs) took 27.7 ms, 33 MMAC/s, while the same block's `A @ V` (same size) took
3.3 ms.

## Root Cause Analysis

ggml routes a batched F32 `mul_mat` to tinyBLAS only when `src1` is contiguous
(`ggml_compute_forward_mul_mat`'s `src1_cont`); otherwise it runs `ggml_vec_dot_f32` per output element.
The Q block was a view of a permuted tensor: rows contiguous, the whole not. `A @ V`'s `src1` was the
softmax output, a fresh contiguous tensor, so only one of the two products fell off. The ARMv6 GEMM tile
(`ggml-0034`) was never a candidate either way: it takes no batched operands. One `ggml_cont` per block
(8k floats) fixed it: 9.6 s.

## Resolution & Lesson Learned

**When a change adds `mul_mat` nodes, profile them by node and divide MACs by time before reading the end
to end number.** An operand's layout, not its shape, picks ggml's kernel, and the slow path is silent:
same output, 8x the time. The cheap check is that two products of the same size in the same graph run at
the same rate.

The same view is in every dense `ATTENTION` (its Q is permuted, never copied), which is why the window
switched off ran 26.9 s against the unfused graph's 18.3 on the Pi Zero. That one is left as a hub item:
copying it changes the summation order of every attention model.
