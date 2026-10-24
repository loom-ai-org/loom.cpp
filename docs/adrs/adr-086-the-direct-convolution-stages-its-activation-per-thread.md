---
type: adr
status: accepted
date: 2026-10-23
domain: performance
tags: [ggml, convolution, direct-conv, f32, aarch64, cortex-a72, pi-4, vits, hifi-gan]
---

# ADR-086: The Direct Convolution Stages Its Activation Per Thread, Under an 8 x 12 Tile

P4.40 in [Epic-05](../epics/epic-05-edge-performance.md#p440-vitss-vocoder-convolutions-on-the-raspberry-pi-4-2026-10-23).
Built as `ggml-0053` (`cmake/patches/UPSTREAM.md` PR 49), on top of the direct kernel of `ggml-0006`/`0007`.

## Context

After P4.39 ([ADR-083](adr-083-the-convolutions-gemms-take-the-outer-product-path-with-a-carved-block.md)),
the largest share of a VITS synthesis on the Pi 4 was its vocoder's resblocks (~37% profiled): eighteen
convolutions of 32-128 channels over 2 288-73 216 positions, kernels 3/5/7 at dilations 1-12, each with
a LEAKY_RELU before it and a bias and a residual after it that `ggml-0007` fuses in. They run ggml's
direct 1-D kernel (`ggml_conv_1d_direct_ok`) at 14-24 GFLOP/s, against 57 for four A72s at one 4-lane FMA
per cycle. ADR-083 measured the im2col route out for them. That kernel:

* copies the whole activation, zero-padded and through the fused LEAKY_RELU, into the work buffer, then
  sweeps it: 9.4 MB written and read back from DRAM per convolution at 32 x 73 216;
* packs the weights `[ic][kx][oc]`, so a tile walks them at a stride of OC floats, one new line per
  (channel, tap), and from 64 channels up those lines come from L2;
* holds a 4 x 16 tile: five loads per 16 FMAs.

## Decision

On AArch64 cores without DOTPROD, a convolution the direct path takes, with OC a multiple of 8, runs
`ggml_conv_1d_staged_run` instead of the direct or the phase-major kernel:

* **Weights packed per tile**, `[OC/8][IC][KW][8]`: one contiguous run per tile.
* **An 8 x 12 tile** (24 accumulators, named variables, see
  [Retro-093](../retros/retro-093-the-bench-inlined-the-tile-the-library-did-not.md)).
* **No global padded copy.** Each thread owns a contiguous range of 12-position blocks. It stages its input
  positions in slices of about 16 KB of activation into its own buffer, applying the fused LEAKY_RELU on
  the way, and sweeps every output channel over each slice. After the first slice, each one takes its
  overlap with the one before from that slice (a `memmove` of the halo) and stages only its new positions.
* **Output over input** (the fused residual's result in its input's memory, the common case) is handled,
  not declined. Each thread saves its two halos (positions its neighbours' outputs will overwrite) before
  the barrier, and the carry above is the only copy of positions its own earlier slices have overwritten.
  Any other overlap, and an in-place case where output position p is not input position p, falls back to
  the kernels above.
* **Same arithmetic.** Every output is the bias, then each channel's taps in order as one fused
  multiply-add each, then the residual: the order of the 4 x 16 and phase-major kernels. VITS's audio is
  byte-identical with the path on or off.
* **Switch** `GGML_CPU_CONV1D_STAGED`: unset follows the DOTPROD rule, `0` is off everywhere, `1` on
  everywhere.

## Alternatives considered

Raced in `scripts/bench67.cpp` at the eighteen resblock shapes, one process, four threads, the same inputs,
every arm byte-identical to the shipped kernel (sum of the per-shape medians, Pi 4; four runs, each arm's
ratio against the shipped kernel in its own run, whose total was 570-590 ms):

| arm | total | vs shipped |
|---|---|---|
| shipped: global padded copy, `[ic][kx][oc]` pack, 4 x 16 | 580 ms | 1.00x |
| + weights packed per tile | 443 ms | 1.29x |
| + 8 x 12 tile | 408 ms | 1.40x |
| 8 x 8 tile / 4 x 24 tile (packed per tile) | 441 / 424 ms | 1.30x / 1.35x: rejected |
| 8 x 12, 4 position blocks per channel sweep (`g4`) | 405 ms | 1.41x: no better than 1 |
| **8 x 12 + per-thread staging, ~16 KB slices** | **309 ms** | **1.91x** |
| staging in slices of 504 / 252 / 96 / 36 / 12 positions | 395 / 343 / 322 / 311 / 324 ms | the byte rule is as good as the best fixed size |

* **Keep the phase-major path for the wide dilated 128-channel shapes** (`ggml_conv1d_phase_ok`): the staged
  kernel takes them too, at 13.5-15.4 ms against the shipped kernel's 29-39 ms in the bench and the phase
  path's 28 ms per call in the model's own profile. The phase path stays for what this one declines
  (OC not a multiple of 8, other ISAs).
* **Fuse a resblock's two convolutions** was measured out in P4.15b (bench11): what it recovered was what
  the two padded copies cost. Staging removes those copies without the fusion.

## Consequences

**Measured end to end**, Pi 4, four threads, `GGML_CPU_CONV1D_STAGED` 0 against 1, 5 paired launches
cooled to 60 C, rc18 harness and cards, engine main `159f4cb` + `ggml-0053`:

| card | ratio (p10-p90) | |
|---|---|---|
| VITS (piper miro) F32 | **1.301x** (1.269-1.312), 1.041 -> 0.799 s | audio byte-identical |
| VITS Q8_0_PC | **1.284x** (1.256-1.295), 1.051 -> 0.823 s | audio byte-identical (its kernels are dequantized to F32 first) |
| Citrinet-256-ls F32 | **1.070x** (1.065-1.074), 0.481 -> 0.448 s | same transcript |
| Pocket TTS F32 (`english_2026-09`) | **1.034x** (1.018-1.037), 9.23 -> 8.93 s | same output checksum |
| Pocket TTS 2m / 7m F32 (windowed exports) | **1.065x** (1.006-1.096) / **1.040x** (1.015-1.083) | same output checksums |
| Moonshine tiny F32, control | 1.002x (0.991-1.004) | no direct convolution; same transcript |

* **No numerics change**, so no card's gate moves. `test_conv_1d_staged` pins the kernel bit-exact against
  an `fmaf` loop at 1-4 threads with the path on and off, including outputs placed over their inputs, tails
  of 1-11 positions, halos wider than a slice and a thread with no positions. Four sabotages (halos not
  saved, no carry, the tail's store floor, the carry's offset) go red with it on and stay green off.
* **Only measured on the A72.** An M1, or any DOTPROD core, keeps the 4 x 16 kernel until it is measured
  there (`GGML_CPU_CONV1D_STAGED=1` forces it). x86 and ARMv6 do not compile it.
