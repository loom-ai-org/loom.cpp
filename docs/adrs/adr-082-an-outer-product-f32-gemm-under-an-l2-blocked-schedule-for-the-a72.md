---
type: adr
status: accepted
date: 2026-10-22
domain: performance
tags: [ggml, tinyblas, f32, gemm, aarch64, cortex-a72, pi-4, xnnpack, cache-blocking, whisper]
---

# ADR-082: An Outer-Product F32 GEMM Under an L2-Blocked Schedule, for the A72

P4.38 in [Epic-05](../epics/epic-05-edge-performance.md#p435-p438-whisper-small-on-the-raspberry-pi-4-per-op-2026-10-22).
Built as `ggml-0048` (`cmake/patches/UPSTREAM.md` PR 44). Kernel reference: XNNPACK, as for P4.35-P4.37
(user, 2026-10-22). Re-opens one line of
[Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md) (the outer-product tile, measured out
at `k = 64` in P4.21) on new evidence, as that register's own rule asks. Harness lessons:
[Retro-090](../retros/retro-090-the-baseline-arm-was-not-the-engine.md).

## Context

On the Pi 4 at four threads, whisper-small's dense encoder GEMMs ran at 1.20-1.46x onnxruntime's MLAS in the
same model (per-op profile, 2026-10-22). What the incumbent actually does, measured in one process at
whisper's shapes against the engine's own ggml build (`scripts/bench66.cpp`):

| | one thread | four threads | MLAS in model, four threads |
|---|---:|---:|---:|
| `qkv` 768 x 768 x 1500 | 8.3 GF (58% of 14.3) | 24.2 | 30.3 |
| `fc1` 768 x 3072 x 1500 | 8.3 | 24.1 | 34.8 |
| `fc2` 3072 x 768 x 1500 | 6.1 | 15.7 | 29.7 |

XNNPACK's F32 kernel for this core (`gemm-config.c` picks `f32-gemm 4x8 aarch64-neonfma-cortex-a75-prfm`
for `xnn_uarch_cortex_a72`, and drops the 6x8 it gives the A57/A75) ran at **10.6 GF on one core (74%) at all
three shapes**. On 2026-10-11 it had also reached 30-44 GF at four threads on VITS's GEMMs, yet only 7-10 GF
at whisper's. Its operator walks 4 rows of activations against all of the weights, which at 2.4-9.4 MB
re-reads every weight from DRAM for every 4 rows (~3.5 GB for `fc1`). So the kernel was not the question
on its own. The question was the kernel under a schedule that keeps its panels in the L2.

## Decision

A second F32 GEMM inside tinyBLAS's file, for **graph `MUL_MAT` nodes, F32 x F32, on AArch64 cores without
DOTPROD**, with at least **128 activation rows**:

* **The tile is XNNPACK's 4x8 outer product**, in NEON intrinsics: the weight packed into 8-row panels with k
  outermost, the activations read as q-vectors of four k, `fmla` by lane into 8 accumulators, no horizontal
  reduction. GCC 14 compiles it to the asm kernel's instruction mix (32 `fmla`, 8 loads per 4 k, no spills)
  and it runs at 98-99% of the asm on one core.
* **The schedule shares one block of packed weight between the threads**: `nb` rows by `kc = 256` of k, with
  `nb x kc x 4 = L2 / 4` (256 x 256 on the 1 MB L2), packed cooperatively into `params->wdata`
  (`ggml_graph_plan` reserves 256 KB + 64 for every F32 x F32 `MUL_MAT`). After a barrier each thread walks
  its own quarter of the activation rows against the whole block, four rows at a time, carrying the sum
  through C from one block of k to the next. Then a second barrier, before the next pack overwrites it.
* **No software prefetch.**
* **Off switch** `GGML_CPU_NEON_F32_4X8=0`, for A/B runs.

## Alternatives considered

| option | measured | verdict |
|---|---|---|
| XNNPACK's own schedule (4 rows x all of N) | 7-10 GF at four threads on whisper (Oct-11) | DRAM-bound at these weight sizes |
| a private block per thread, no shared pack | 6-8% slower than the shared block | rejected |
| a full-k block | fc2 **0.43x** of tinyBLAS (64 rows x k 3072 = 768 KB per thread) | rejected; k must be blocked |
| block size | 64-512 rows x 128-384 k plateau between 256 and 288 KB | `L2 / 4`, derived rather than fitted |
| one `prfm` per 4-k step, 512 B ahead (XNNPACK's spacing, half the lines here) | **20% slower than none** on one core | rejected |
| two `prfm` per step at 256 / 512 / 1024 B | tied with none on one core, 1-6% slower on four | none chosen |
| the asm kernel itself | 1-3% faster than the intrinsics at full k, but it cannot carry a sum through C (it initialises from a bias slot) | intrinsics: k-blocking is the four-thread win |
| packing at load (an extra buffer type, like `CPU_NEON_I8`) | would save the per-call pack: 1.3 ms (`qkv`) to 5.6 ms (`fc2`) per node, ~1.7% | not now: a buffer type plus a placement rule for ~2% |
| predicate `n / nth >= 32` | fits both one thread (n = 24 already wins) and four | **rejected**: it makes which sum a product gets depend on the thread count. tinyBLAS is byte-identical across thread counts, and `test_tinyblas_k_halves` would fail |
| taking the path inside `llamafile_sgemm` | the convolutions' im2col fallback calls mul_mat with its patches already in `wdata`, so the pack overwrote them: conv sweeps and STFT at relative errors of 0.9-1.5 on the Pi | graph nodes only (`dst->op == GGML_OP_MUL_MAT`), exactly as ARMv6's ggml-0034 had to |

## Consequences

**Measured**, Pi 4, four threads, against tinyBLAS (`bench66`, per-round ratio over three rounds, distinct
weights per node): `qkv` 1.32x (31.9 GF), `fc1` 1.31x (31.7), `fc2` 1.71x (27.0). That is at MLAS's
in-model rate or within 10% of it. Across 27 shapes the path wins from n = 200 up (1.11-1.58x at four
threads: VITS's convolution GEMMs, Moonshine, whisper's `QK^T` 1.15x and `A@V` 1.42x). It loses below
n = 64 at four threads, which the predicate keeps on tinyBLAS.

End to end, paired launches, `GGML_CPU_NEON_F32_4X8` off against on, rc18 harness and cards (F32 unless named):

| card | ratio | |
|---|---|---|
| whisper-small | **1.174x** (means, 7 launches; 24.6 -> 20.9 s) | same transcript |
| Moonshine tiny | 1.057x (p10 1.055, p90 1.060) | same transcript |
| whisper-small **Q8_0_PC** | 1.082x (means, 7 launches; 13.78 -> 12.74 s) | same transcript: its attention (`QK^T`, `A@V`) is F32 x F32 at 1500 rows, on top of P4.37's int8 linears |
| Citrinet-256-ls | 0.998x, straddles 1.0 | null control: its GEMMs are convolutions, which keep tinyBLAS |
| VITS (piper miro) | 1.011x, straddles 1.0 | null control: the audio is **byte-identical** on and off (same FNV-1a); every GEMM it has is a convolution |
| Qwen3-0.6B | 0.991x, straddles 1.0 | same 24 tokens: a short prompt (< 128 rows) and one-row decode steps never qualify |

* **Numerics change on these cores.** Each output is one sequential fused sum over k: deterministic, the
  same at every thread count, and bit-exact to a scalar `fmaf` loop (`test_mul_mat_neon_f32_4x8`). But it
  is not tinyBLAS's order (four lane partials, then a horizontal add). Against a double product its error
  is ~2x tinyBLAS's: 5.0e-7 against 2.5e-7 relative at k = 768, both far inside every F32 gate.
* **Only measured on the A72.** The guard is the missing ISA feature, as for `CPU_NEON_I8`, so an A53, A55,
  A57 or A73 takes it too. No such board is available (user, 2026-10-22).
* **The convolutions do not get it yet.** VITS's convolution GEMMs won 1.15-1.58x in the shape sweep. To
  take this path they would reserve the block beside their own im2col patches and pass it in: a follow-up
  in the hub.
* An M1 or any DOTPROD core is untouched: there tinyBLAS beat XNNPACK's F32 kernels (92 vs 74 GF on one
  core), and this was not raced.
