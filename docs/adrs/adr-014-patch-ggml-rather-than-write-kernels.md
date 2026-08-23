---
type: adr
status: accepted
date: 2026-08-21
tags: [performance, ggml, upstream, patches, gemm, convolution]
---

# ADR-014: Improve `ggml`'s Kernels by Patching It, Not by Writing Our Own

## Context

On a Raspberry Pi 4, VITS ran ~2.2x slower than the same checkpoint under onnxruntime, and the gap was
**F32 micro-kernel quality**: `ggml`'s generic `mul_mat` reached 17% of the machine's fp32 peak where
MLAS reached 45%, on identical arithmetic. That is a kernel-quality problem in a dependency, not
something the exporter or a new backend can reach.

## Options Considered

1. **Write loom's own GEMM and convolution kernels.** Full control; a permanent maintenance burden in a
   repository whose entire premise is that it computes nothing itself, plus a second implementation to
   keep correct across architectures.
2. **Link an external BLAS** (OpenBLAS, MLAS). Adds a heavyweight dependency to an engine that exists to
   be lean, and does not cover the convolution path.
3. **Patch `ggml`'s own kernels**, carried in-tree as versioned patches applied at configure time.

## Decision

**Patch `ggml`.** Nothing in this repository computes a GEMM. The hand-written 4x4 kernel in
`scripts/bench6.cpp` **shipped nothing** — it existed as the measuring stick, and `ggml`'s tinyBLAS
ended up beating it.

What ships is a set of numbered patches under `cmake/patches/` plus `GGML_LLAMAFILE=ON`, each behind an
A/B switch so the before state is exactly reproducible:

| patch | what it does |
|---|---|
| tinyBLAS tile + follow-up | register blocking in the F32 `mul_mat` path |
| `ggml-0005` | fuse a per-channel bias `ADD` into `CONV_2D` |
| `ggml-0006` | direct 1-D convolution behind a cache-size heuristic |
| `ggml-0007` | fold a resblock's `LEAKY_RELU` and residual `ADD` into the convolution |
| `ggml-0008` / `ggml-0009` | `conv_transpose_1d`'s serial prologue, then its compute as a GEMM |

**Every performance claim is A/B-switchable inside one binary** (`LOOM_TINYBLAS=OFF` reproduces the
shipped-before state exactly), interleaved ABBA in both orders, medians of repeated runs.

## Consequences

* **Positive:** the fixes are where the arithmetic is, so they benefit every model and every consumer of
  the pinned `ggml`, and they are candidates for upstreaming.
* **Positive:** no second kernel implementation to maintain, and no heavyweight BLAS dependency.
* **Negative:** patches must be rebased on every `ggml` pin bump, and a bump is already a real piece of
  work — v0.16.0 → v0.19.0 was 154 upstream commits.
* **Negative:** loom now ships a `ggml` that is not upstream's, so a bug reproduced here is not
  automatically reproducible upstream. The A/B switches are what keep that diagnosable.

## Related

* Epic: [Epic-05: Edge Performance](../epics/epic-05-edge-performance.md)
* Retro: [Retro-011: Chasing the GEMM and Convolution Gap](../retros/retro-011-chasing-the-gemm-and-convolution-gap.md)
* Retro: [Retro-012: Optimizations That Were Measured Out](../retros/retro-012-optimizations-that-were-measured-out.md)
