---
type: retro
date: 2026-10-29
domain: backends
tags: [vc4, cma, memory, ggml-backend-sched, supports-op, fallback, raspberry-pi-zero]
---

# Retro-102: A Room Check Made Before the Plan Took Its Room

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (P4.47); the decision is
[ADR-093](../adrs/adr-093-the-qpus-take-the-transposed-convolutions-and-the-weights-leave-the-pool.md).

## Issue

With the transposed convolutions on the QPUs (and the weights still in the GPU's pool), VITS Q8_0_PC
failed its synthesis on the Pi Zero after the first of its four upsampling stages:
`loom.run_subgraph: dma_heap alloc: Cannot allocate memory`. The other six HiFi-GAN models of the same
A/B ran, up to 1.4x faster than P4.46.

## Root cause

ggml-vc4 asks whether a convolution fits when the scheduler asks `supports_op`: its weights packed for
the call and its staging slots, against what the pool has free *then*. The scheduler asks before it
allocates its own buffers -- and taking the upsampling stages moved their results into the pool, 2.4 MB
and three of 9.8 MB for VITS, which the CPU split had held in ordinary memory. When the second stage
ran, the pool held 55.4 MB in 72 buffers, and the call's arena -- its staged input, and its weights packed
for the call when the pool is below the packing reserve -- could not be had: the one allocation of a
call that throws (a refused packed copy falls back to the arena). That allocation is the
first of the call, but it was at compute time, where nothing fell back: the exception ended the graph.

## Takeaway

**A capacity check made while a plan is being formed is made before the plan's own allocations.**
Answering "does this fit?" in `supports_op` measures a pool the answer itself is about to shrink --
every node it accepts brings its result into that pool. Such a check can only be a heuristic; what
makes the backend correct is a fallback where the memory is actually taken. ggml-vc4 now runs a node
whose memory it is refused at compute time (and the chain folded into it) on the CPU backend, warns once,
counts it, and has a knob (`GGML_VC4_ARENA_MAX_KB`) with which the board test forces that path. And the
pool got the room it was missing: the weights left it (ADR-093).

## Record

* `~/p447/outA/stderr-vits-q8pc-p447a.txt`: the traceback above; `ggml-vc4: 0 MUL_MATs, 95 CONV_2Ds and
  1 CONV_TRANSPOSE_1Ds in 96 launches ... 72 buffers, 55.4 MB`.
* `~/p447/tb/refuse.log` (`GGML_VC4_ARENA_MAX_KB=64 test_vc4_backend`): ALL OK, every case's result the
  CPU's; `192 run by the CPU backend (88 for want of GPU memory)`.
* `~/p447/out3`: VITS Q8_0_PC with both changes, 6.68 s (5.74 s the second inference), SNR 49.8 dB.
