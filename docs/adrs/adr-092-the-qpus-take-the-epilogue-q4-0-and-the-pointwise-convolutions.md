---
type: adr
status: accepted
date: 2026-10-29
domain: backends
tags: [vc4, qpu, raspberry-pi-zero, fusion, epilogue, q4_0, q8_0, int8, tile-order, cma, ggml-backend-sched]
---

# ADR-092: The QPUs Take the Epilogue, Q4_0 and the Pointwise Convolutions

P4.46 in [Epic-04](../epics/epic-04-backends-and-accelerators.md), the open items of
[ADR-091](adr-091-the-qpus-take-convolutions-and-per-channel-int8.md). Code: `backends/ggml-vc4/`
(`qpu_gemm_qb.h` new; `qpu_conv1d.h`, `qpu_gemm_q8.h`, `qpu_gemm.h`, `qpu_gemm_wt.h`, `vc4_mbox.h`,
`ggml-vc4.cpp`); `src/core/gguf_model.cpp`. What broke on the way:
[Retro-099](../retros/retro-099-a-slowdown-blamed-on-the-dimension-that-was-large.md) (the "large N"
slowdown was the row stride), [Retro-100](../retros/retro-100-a-free-memory-figure-that-is-a-lower-bound.md)
(`CmaFree` made "auto" nondeterministic), [Retro-101](../retros/retro-101-a-board-test-that-linked-what-production-loads.md)
(a module ggml's loader dropped while the board test passed).

## Context

Profiled under "auto" on the Pi Zero with P4.45 (`LOOM_PROFILE`, node by node): with the convolutions
on the QPUs, what the CPU did around them was most of what was left -- amy F32: ADD 1.50 s and
LEAKY_RELU 0.69 s of 5.79 s, against 1.93 s for the convolutions; VITS Q8_0_PC: 1.67 s of 8.15 s.
Citrinet-256 Q8_0_PC gained nothing (7.14 s either way): its pointwise convolutions, 4.8 s, are int8
in CPU_ARMV6_I8, which the QPUs could not read. Q4_0 MUL_MATs (conformer-ctc, distilbert) stayed on the
CPU. The F32 kernel ran one P4.44 shape at 2.2 GFLOPS, filed as "large N".

## Decision

1. **The epilogue goes into the convolution.** `conv1d_f32_kernel` takes ggml-0005/0007/0036's chain:
   the accumulators start at the bias (16 uniforms a tile in place of 16 zeroing ldi -- free), the
   residual is one TMU request per output channel, then RELU (`fmax`) and the time mask. A LEAKY_RELU
   before the convolution goes into the host's staging copy (`stage_rows`, the sign chosen on the bits).
   The backend claims those nodes only in their place in that chain (`step_of`: conv, bias ADD,
   residual ADD, RELU, mask MUL), and graph_compute folds what is private to the chain; what it claimed
   and cannot fold runs on the CPU backend, as a run of nodes so the CPU's own fusions still see them.
2. **A folded chain computes in place.** `graph_optimize`, before allocation, makes each chain node a
   view of the convolution's tensor: ggml's allocator would do it for the bias ADD but never does over a
   reshape view (its view-source reuse is unreachable). Without this every node had its own tensor in
   the 64 MB pool and VITS's vocoder needed 48 MB there. The residual ADD, RELU and mask, and a
   LEAKY_RELU (whose result cannot take its input's memory), are claimed only up to 1/32 of the pool
   (`GGML_VC4_FUSE_ACT_MB` / `_RES_MB`): claiming VITS's 9.6 MB LEAKY_RELUs did not fit beside its
   weights. When the scheduler's buffers do not fit, the packed-weight cache and the arena are released
   and the allocation retried; when that fails too, the backend stops claiming those nodes, undoes the
   in-place rewrite of the graph being planned, and loom re-splits once (`allocate_scheduled`), so a
   model whose weights leave too little pool (Citrinet F32, 39 MB) runs unfused instead of failing.
3. **Q4_0 and Q8_0 with a scale per block** (`qpu_gemm_qb.h`): qpu_gemm_q8.h's offset-binary bytes, a
   row repacked as [32 bytes | F32 d] per block so each block's scale arrives in r4 after its last word;
   each block's sums start at -128 * sum x (a uniform) and are flushed as T += p * d. 70 instructions per
   4 k against the per-channel kernel's 65 and the F32 kernel's 68. Weights where the model put them,
   repacked once and kept like a packed convolution weight.
4. **1x1 convolutions over a VC4_Q8 weight** run as `gemm_q8_kernel` with the output transposed
   (`GEMM_OUT_M`: accumulators written vertically into the VPM, the vertical writes qpu_gemm.h proved),
   X packed from [IC][T], with the same epilogue (bias by TMU per lane, mask from uniforms). loom gives a
   CONV_2D over a CPU-repacked weight the primary's copy the way it gives a MUL_MAT one.
5. **Rows a multiple of 512 bytes apart are padded** (`gemm_row_stride`, +64): the 16 rows a gather
   reads fell into one or two TMU cache sets. F32 weights read in place get a padded copy, kept or (when
   small) made per call; the int8 layouts are padded when packed.
6. **Tiles in panels of two column blocks** (`gemm_tiles`): 1.17-1.36x over m-major at every model shape.
7. **The chunks of a staged convolution overlap the QPUs**: the mailbox call that starts them runs on a
   launcher thread, and the host stages chunk k + 2 while chunk k runs -- two ahead, because the result
   may be written over x (the allocator gives a HiFi-GAN residual ADD its input's block).
8. **The pool's free memory is the pool less what this process holds** (`GpuBuffer::live_bytes`) less
   8 MB, or `CmaFree` when larger; a failed dma-heap allocation is retried once.

## Alternatives considered

* **The host computing the epilogue after the QPUs.** A full pass per node again, which is what cost the
  1.5-2.2 s.
* **The LEAKY_RELU on the QPUs, per tap.** 2 instructions per k (+12% of the kernel) against a staging
  copy the host makes anyway and the launcher thread now hides.
* **Patching ggml-alloc's view reuse** (`ggml_gallocr_is_own` for views). It changes every backend's
  memory plan, and the branch it would enable has never run (its address assignment reads the view's
  unset address); the in-place rewrite is local to this backend's splits.
* **Q4_0 as F16 through r4's float16 unpack**, or requantized to one scale per row. 2 bytes a weight in
  a 64 MB pool, or an error added to the CPU's.
* **A probe allocation to read the pool's real free memory.** A 60 MB probe was not returned to the pool
  until the process exited.
* **A different tile order for "large N"** -- what the P4.44 note asked for. Retro-099: it was the stride.

## Consequences

* Pi Zero W, `cpu` / P4.45 "auto" / P4.46 "auto" (run `out3`, one process per arm):

  | model | cpu s | P4.45 s | P4.46 s | vs cpu | vs P4.45 | output against the CPU |
  |---|---:|---:|---:|---:|---:|---|
  | VITS Q8_0_PC | 19.93 | 8.55 | 7.63 | 2.61x | 1.12x | SNR 50.0 dB, same length |
  | VITS Q4_0 | 18.88 | 7.38 | 7.15 | 2.64x | 1.03x | SNR 61.7 dB |
  | amy F32 / Q8_0_PC | 13.68 / 13.47 | 6.12 / 6.09 | 5.42 / 5.52 | 2.52x / 2.44x | 1.13x / 1.10x | SNR 76.2 / 76.7 dB |
  | heart F32, de051 F32 | 1.97, 3.20 | 0.90, 2.12 | 0.92, 1.90 | 2.13x, 1.69x | 0.97x, 1.12x | SNR 59.4, 84.3 dB |
  | Citrinet-256 Q8_0_PC | 7.29 | 7.18 | 3.65 | 2.00x | 1.97x | one CTC id, P4.45's near tie |
  | Citrinet-256 F32 | 9.89 | 3.48 | 3.85 | 2.57x | 0.90x | identical (pool refused the fused plan; unfused retry) |
  | Moonshine tiny Q8_0_PC | 20.21 | 13.32 | 13.00 | 1.55x | 1.02x | identical transcript |
  | conformer-ctc Q8_0_PC | 5.65 | 5.85 (CPU) | 3.97 | 1.42x | 1.47x | identical text |
  | conformer-ctc Q4_0 | 7.67 | 6.21 | 4.18 | 1.84x | 1.49x | identical text |
  | wake F32, VAD Q8_0_PC | 2.11, 1.98 | 1.39, 1.72 | 1.38, 1.76 | 1.52x, 1.13x | 1.01x, 0.98x | SNR 97.7 dB, max\|d\| 1.4e-4 |

  P4.45's conformer Q8_0_PC arm ran on the CPU: its `CmaFree` read 44.7 MB free (Retro-100). Citrinet
  F32's 3.85 s is from a separate run of the final build; in the A/B run its fused plan was refused before
  the fallback below existed. With `GGML_VC4_FUSE=0`: VITS Q8_0_PC 8.16 s, amy 5.88, Citrinet Q8_0_PC
  3.82 -- the pointwise route without its epilogue.

* Kernels on the board (`test_p446_hw`, 12 QPUs, against f64): Q4_0 / Q8_0 176 x 76 x 704 5.4 ms with
  the X pack; Citrinet's 256 x 551 x 256 1x1 int8 15.1 ms with bias, RELU and mask; HiFi-GAN 32-channel
  k = 7 with bias and residual 10.9 GFLOPS (11.35 without). F32 MUL_MATs: 128 x 3656 x 384 2.2 -> 12.3
  GFLOPS, 768 x 156 x 768 2.05 -> 11.9.
* The QPUs' Q4_0 / Q8_0 results are closer to an f64 product of the dequantized weights than the CPU's,
  which quantizes the activations to Q8_0 (176 x 76 x 704: 5.4e-4 against 0.088): for those the board
  test compares against f64, not the CPU.
* `test_vc4_module_symbols` (DL builds): every unversioned import of the module is libggml-base's.
* Open: weights only the QPUs' packed copies read still occupy the pool in ggml's layout; a LEAKY_RELU
  above the cap and the residual stream of the largest TTS layers stay on the CPU in a 64 MB pool;
  CONV_TRANSPOSE_1D (VITS, amy: ~1.5 s each) is the CPU's; "auto" has no fallback when another process
  holds the pool.
