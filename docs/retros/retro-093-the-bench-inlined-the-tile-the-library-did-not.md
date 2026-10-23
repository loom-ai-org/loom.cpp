---
type: retro
date: 2026-10-23
domain: performance
tags: [convolution, ggml, aarch64, cortex-a72, pi-4, vits, codegen, register-allocation, benchmarking]
---

# Retro-093: The Bench Inlined the Tile, the Library Did Not

Serves [Epic-05](../epics/epic-05-edge-performance.md#p440-vitss-vocoder-convolutions-on-the-raspberry-pi-4-2026-10-23)
(P4.40); the decision is [ADR-086](../adrs/adr-086-the-direct-convolution-stages-its-activation-per-thread.md).

## The Issue

`scripts/bench67.cpp` raced an 8 x 12 direct-convolution tile with per-thread staging against the shipped
4 x 16 kernel at the eighteen resblock shapes of a VITS vocoder on the Pi 4: 1.91x over all of them, every
output byte-identical. The same source, pasted into `ops.cpp` as `ggml-0053`, made the whole synthesis
**slower**: 1.035 -> 1.159 s (0.893x over 5 paired rounds), with the audio still byte-identical. The node
profile had every staged convolution slower than the kernel it replaced (32 x 32 kw3: 24.5 -> 29.7 ms).

## Root Cause Analysis

The disassembly of `libggml-cpu.so` showed the hot instantiation of the tile
(`ggml_conv_1d_staged_tile<true, false>`) storing all 24 accumulators to the stack on every tap: twelve
`stp q, q, [sp, #...]` after each group of 24 `fmla`. In the bench the tile is `static inline` and was
inlined into its caller, where GCC 14 kept `float32x4_t acc[8][3]` in registers. In `ops.cpp` it called
the same tile from four specialisations of a larger function, decided not to inline it, and in the
out-of-line copy the array stayed an aggregate in memory. The flags were the same (`-O3`, the same
`-mcpu`, only `-fPIC` added). The 4 x 16 kernel next to it records the same failure for a different
trigger (a branch in its store loop, 1.44 -> 1.73 s), and fixed it with template parameters; that was not
enough here.

Rewritten with 24 named `float32x4_t` variables, the inner loop is 24 `fmla` and five loads with no stack
access, and over 5 paired rounds the synthesis went 1.041 -> 0.799 s (1.301x).

## Resolution & Lesson Learned

**When a kernel moves from a standalone bench into the library, disassemble the library's copy before the
first end-to-end run.** Count the stack accesses in the inner loop (`grep '\[sp'` between the first
`fmla` and the loop's back branch). A bench measures the code its own compiler run produced, and whether
an accumulator array stays in registers depends on an inlining decision the bench does not reproduce. For
SIMD accumulators, name them; do not rely on scalar replacement of an array.

The bench's byte-identity check could not catch this: a spilled accumulator gives the same answer more slowly.
