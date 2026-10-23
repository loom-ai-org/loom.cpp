---
type: retro
date: 2026-10-22
domain: performance
tags: [benchmark, harness, baseline, tinyblas, llamafile, thread-pool, pi-4, p4.38]
---

# Retro-090: The Baseline Arm Was Not the Engine

## The Issue

P4.38 (an F32 GEMM tile for the Pi 4's A72, [ADR-082](../adrs/adr-082-an-outer-product-f32-gemm-under-an-l2-blocked-schedule-for-the-a72.md))
started from a lead in its own handover. P4.37's `bench65` F32 arm had put whisper's `fc2` (k = 3072) at
**6.05 GFLOP/s at four threads** against 12-12.8 for `fc1` and `qkv`: "large k collapses at four threads".
The first race of the new tile against ggml then reported **1.84x** at `qkv`. Both numbers were wrong, and
both for the same reason: **the arm that stood for the engine was not running what the engine runs.**

* `bench65` linked a standalone ggml tree (`~/p437/ggml`). ggml's own CMake defaults `GGML_LLAMAFILE` to
  **OFF** when ggml is built by itself (llama.cpp and loom turn it on), and that tree also had OpenMP on. Its
  "f32" arm was therefore ggml's generic `vec_dot` loop, not tinyBLAS: 6.05 GF at `fc2` and 12.1 at `fc1`,
  where the engine's own build measures 15.7 and 24.1.
* `bench66`'s first thread pool kept its idle workers **spinning** on an atomic. While the ggml arm ran, three
  of the board's four cores belonged to threads that were only waiting. ggml came out at 1.5-2x its in-model
  time, and the new tile "won" 1.84x. With the workers sleeping on a condition variable it was 1.09x for
  that schedule. (The schedule that finally shipped is 1.32x.)

## Root Cause Analysis

A bench's baseline arm is a claim about the engine, and neither of these benches checked that claim. Each
bench reported a ratio between two arms, and a ratio is equally clean when both arms are slow and when
only one is. Nothing inside either program could have told the difference. What did tell it was a number
from outside: the per-op profile of the real model. It says whisper's `fc1` node takes **275 ms per call**
at four threads, and ggml's arm took 427 (pool spinning), then 585 (bench65). The `fc2` lead failed the same
test: the profile's `768 x 1500` bucket cannot hold twelve 1167 ms calls.

## Resolution & Lesson Learned

**Actionable takeaway: before reading a ratio, put the baseline arm's time next to the in-model time of the
same shape, from a profile of the real engine.** If they disagree by more than the bench's noise, the bench
is measuring something else, and its ratios mean nothing yet.

* **A standalone ggml must be configured like the engine's**, `-DGGML_LLAMAFILE=ON -DGGML_OPENMP=OFF`, or
  the bench links the engine's own `_deps/ggml-build`. `bench65`'s header now says so. `bench66` links the
  engine build.
* **A harness's own threads must sleep while another arm runs.** Spin-waiting is right inside a timed
  region (the barrier), never between them. `bench66`'s pool sleeps on a condition variable when idle.
* The earlier records are unaffected: P4.37's numbers come from `bench65`'s q8 and i8 arms, and tinyBLAS
  has no Q8_0 GEMM without DOTPROD, so LLAMAFILE changes neither.

Related: [Retro-010](retro-010-an-unpinned-competitor-baseline.md) (a baseline that was one unrepeatable
wall time), [Retro-023](retro-023-a-bench-whose-graph-was-the-treatment.md) (a bench whose graph was the
treatment).
