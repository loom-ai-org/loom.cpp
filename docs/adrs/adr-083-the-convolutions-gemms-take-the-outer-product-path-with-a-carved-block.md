---
type: adr
status: accepted
date: 2026-10-23
domain: performance
tags: [ggml, convolution, im2col, f32, gemm, aarch64, cortex-a72, pi-4, vits, citrinet]
---

# ADR-083: The Convolutions' GEMMs Take the Outer-Product Path, With a Block Carved Off Their Buffer

P4.39 in [Epic-05](../epics/epic-05-edge-performance.md#p435-p438-whisper-small-on-the-raspberry-pi-4-per-op-2026-10-22).
Built as `ggml-0049` (`cmake/patches/UPSTREAM.md` PR 45), on top of
[ADR-082](adr-082-an-outer-product-f32-gemm-under-an-l2-blocked-schedule-for-the-a72.md)'s `ggml-0048`.

## Context

ADR-082's GEMM runs only for graph `MUL_MAT` nodes. It packs a block of A into `params->wdata`, and the
convolutions' batched im2col (`ggml_compute_forward_conv_2d_impl`, and `conv_transpose_1d`) calls mul_mat
with its patches already in that buffer: taking the path from inside `llamafile_sgemm` overwrote them. So
on the Pi 4 every convolution GEMM stayed on tinyBLAS. That covers VITS's text encoder and flow, Citrinet's
pointwise and wide convolutions, and Moonshine's stem, in F32 and in Q8_0_PC too, whose kernels are
dequantized to F32 before the GEMM (P4.29).

A per-node profile of VITS on the Pi 4 at four threads says where its convolutions run:

| group | path | time share (profiled) |
|---|---|---|
| vocoder resblocks, 32-128 channels, 2 288-73 216 positions | ggml's **direct** kernel (`ggml_conv_1d_direct_ok`) | ~37% |
| text encoder and flow, 192-768 channels, 100-286 positions | batched im2col + GEMM | ~21% |
| three upsamplers (`CONV_TRANSPOSE_1D`) | GEMM + overlap-add | ~6% |

## Decision

* **The convolution carves the path's block off the END of its own work buffer**
  (`ggml_conv_f32_op_carve`, `llamafile_sgemm_f32_wsize()` bytes), sizes its patches from what is left,
  and hands the GEMM a copy of `params` whose `wdata` is the carved block. The patches are never under it.
* **All three of `conv_2d_impl`'s GEMM calls** try the path first: the one that writes straight into dst
  (patches packed, the kernel's rows walked four at a time), the deferred one that stages a batch, and the
  permute one (kernel packed, patches walked). Each keeps its tinyBLAS call as the fallback.
* **`conv_transpose_1d`** too, whose GEMM already has the path's operand order. `ggml_graph_plan` reserves the
  block after its other regions. `CONV_2D`'s 16 MB reservation already holds it.
* **The predicate is ADR-082's**, at least 128 rows of B, applied to whichever operand the call walks. It is
  independent of the thread count, so the bytes are too.
* **Switch** `GGML_CPU_NEON_F32_4X8_CONV=0`, for this decision alone. `GGML_CPU_NEON_F32_4X8=0` switches off
  both.

## Alternatives considered

| option | measured | verdict |
|---|---|---|
| the vocoder's resblocks through im2col + this GEMM instead of the direct kernel (`GGML_CPU_CONV1D_BUDGET=0`, the 128 floor lowered to 32) | VITS 1.178 s against 1.022 with the direct kernel; im2col + tinyBLAS 1.237 | rejected: the direct kernel already runs those at ~20-23 GF/s, and the gather plus a per-call pack of patches for 32-64 output channels costs more than the tile gains |
| re-tune the im2col patch budget (512 KB, half the L2) for the new GEMM | 256 KB ties; 1 MB +6%, 2 MB +9%, no cap +45% (slower) | unchanged |
| a larger `conv_transpose_1d` block, so the 1024- and 2048-row upsamplers reach 128 positions | 2x / 4x / 8x the tile: 1-2% slower over a synthesis | unchanged: the result block stops fitting where the overlap-add reads it |
| take the block from the front of the buffer and shift the patches | the same bytes, and every offset in the function moves | the end is the one place nothing else addresses |

P4.38's 27-shape sweep had listed VITS's resblock shapes as "convolution GEMMs" that won 1.15-1.58x, which
is what this item was opened on. The incumbent in that sweep was tinyBLAS, but in the model those shapes
run the direct kernel ([Retro-090](../retros/retro-090-the-baseline-arm-was-not-the-engine.md), second
instance). The gain that remained is the 21% of VITS that really is im2col.

## Consequences

**Measured**, Pi 4, four threads, `GGML_CPU_NEON_F32_4X8_CONV` off against on (P4.38 on in both arms),
5 paired launches, rc18 harness and cards:

| card | ratio (p10-p90) | |
|---|---|---|
| VITS (piper miro) F32 | **1.068x** (1.062-1.090), 1.097 -> 1.025 s | 101 dB SNR against off; see below |
| VITS Q8_0_PC | 1.054x (1.024-1.070) | |
| Citrinet-256-ls F32 | **1.062x** (1.059-1.064) | same transcript |
| Citrinet-256-ls Q8_0_PC | 1.060x (1.055-1.068) | same transcript |
| Moonshine tiny F32 | 1.034x (1.029-1.084) | same transcript: the stem |
| whisper-small F32 | 1.000x, straddles 1.0 | same transcript: two stem convolutions in a 21 s run |
| Pocket TTS F32 | 1.003x, straddles 1.0 | |

* **Numerics change for these convolutions on these cores**, the same way ADR-082's did for the linears:
  one sequential fused sum per output, in im2col order (input channel, then tap), bit-exact to an `fmaf` loop
  at 1/2/4 threads (`test_conv_neon_f32_4x8`, which reaches all four call sites; five sabotages red). VITS's
  audio is no longer byte-identical with the switch off: against it the largest difference is 3.3e-6 on a
  peak of 0.16 (101 dB SNR) at F32, and 5.3e-6 (97 dB) at Q8_0_PC.
* **The vocoder is still the direct kernel.** It is the largest share left in VITS, and the next step there is
  a better direct kernel rather than this GEMM.
* Only measured on the A72, like ADR-082. An M1, or any DOTPROD core, is untouched.
