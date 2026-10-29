---
type: adr
status: accepted
date: 2026-10-29
domain: backends
tags: [vc4, qpu, raspberry-pi-zero, conv-transpose, polyphase, hifi-gan, buffer-type, weights, cma]
---

# ADR-091: The QPUs Take the Transposed Convolutions, and the Weights Leave the Pool

P4.47 in [Epic-04](../epics/epic-04-backends-and-accelerators.md), two of
[ADR-090](adr-090-the-qpus-take-the-epilogue-q4-0-and-the-pointwise-convolutions.md)'s open items. Code:
`backends/ggml-vc4/` (`qpu_conv1d.h`, `ggml-vc4.cpp`); `src/core/backend.cpp`, `src/core/gguf_model.cpp`.
What broke on the way: [Retro-102](../retros/retro-102-a-room-check-made-before-the-plan-took-its-room.md).

## Context

After P4.46, HiFi-GAN's upsampling stages (CONV_TRANSPOSE_1D, LEAKY_RELU before and a bias after)
were the largest thing the CPU still did in the vocoders: ~1.5 s of amy's 5.4 s and of VITS's 7.6 s
under "auto". And every model's weights sat in the GPU's 64 MB pool in ggml's layout although the QPUs
read almost none of them there -- convolutions and Q4_0 / Q8_0 MUL_MATs read packed copies, also in the
pool -- so the copies, the fused chains and the scheduler's buffers competed for what the weights left.
VITS F32 (62.6 MB of weights) and Moonshine F32 could not use the QPUs at all.

## Decision

1. **A transposed convolution is s ordinary ones, run as one** (`CONV1D_OUT_PHASES`). With K = J*s
   taps, output phase p of y[oc][q*s + p] is a J-tap convolution of x padded by J - 1 zeros a side
   (ggml-0038's polyphase form). `conv1d_f32_kernel` runs it unchanged over OC*s virtual channels
   v = oc*s + p (`conv1d_pack_wt`), each tile 16/s real channels' s phases; only the output differs.
   The accumulators go into the VPM vertically, so a VPM row holds one position's 16 virtual channels
   and real channel oc's 16*s outputs, contiguous in dst, are VPM columns oc*s .. oc*s + s - 1: one DMA
   of mvalid rows x s words a channel, the VDW setup's X field stepped by s. Strides 1, 2, 4, 8, 16.
2. **Its chain folds like a convolution's**: the LEAKY_RELU before it into the staging copy, the bias
   after it into the accumulators (a bias per virtual channel). Not a residual or mask, which are read
   along one channel's row and a phase tile is not one -- HiFi-GAN puts neither there.
3. **Weights go in the device's host buffer type** (`weights_buffer_type`): for a device whose own
   buffers are host memory and which offers one, loom loads a model's weights there instead. ggml-vc4
   offers VC4_HOST, ordinary memory, whose buffers are its own so their free drops what was packed from
   them; an F32 MUL_MAT over a weight there reads a (padded) copy in the pool like the convolutions do,
   kept while the pool has room and made per call otherwise. The pool then holds what the QPUs read.
   "auto"'s room check skips such a device: its weights need none of the pool.
4. **The copies go in 2 MB slabs**, not an allocation each: a GPU allocation is a dma-heap buffer
   imported into vcsm, ~2 ms on the Pi Zero, and with its weights in RAM conformer-ctc's 195 F32
   matrices each got one on first use -- its first inference 0.45 s slower than P4.46's, which read
   them in place (its second, equal). A slab goes with the last copy in it.
5. **A node the pool refuses memory at compute time runs on the CPU** -- the arena for its staged input
   and per-call weights is grown on demand, after the scheduler's buffers took their share (Retro-102).
   The node and the chain folded into it go to the CPU backend. A VC4_Q8 weight, whose layout only the
   QPUs read, goes as its Q8_0 blocks rebuilt in RAM (`gemm_q8_unpack`: exact, since each row had one
   fp16 scale), made once and dropped with the VC4_Q8 buffer. `GGML_VC4_ARENA_MAX_KB` caps the arena,
   so the board test forces that path.
6. **VC4_Q8 buffers are RAM too**, the QPUs reading a slab copy of a weight (`q8_weight`) kept like a
   packed one, or made per call into the arena's first bytes. In the pool, Pocket TTS 18m Q8_0_PC's
   per-channel weights left no room for the scheduler's 15-29 MB buffers and its graph failed to
   allocate -- for matrices most of whose products are frame-by-frame N = 1 steps the QPUs decline.
7. **An F32 MUL_MAT over a VC4_HOST weight is taken from 16 columns**, as the int8 kernels are: Pocket
   18m's 512 x 1 x 512 decode products copied a 1 MB weight into the pool per call, 2700 times a run
   (QPU 6.9 s, host 18 s), where the CPU does the product in less than the copy.
8. **The CPU copy of a refused VC4_Q8 weight goes in CPU_ARMV6_I8** when the CPU device takes the node
   from there (MUL_MAT, pointwise CONV_2D), else in VC4_HOST as plain Q8_0: in VC4_HOST, ggml's generic
   Q8_0 kernel ran Pocket 18m's refused nodes and CPU+VC4 took 38.9 s to the CPU's 23.3. And a refused
   arena is remembered: until some GPU block is freed (`GpuBuffer::frees()`), a call needing at least
   what was refused goes to the CPU at once, instead of four failed dma-heap requests and every packed
   weight dropped, for each of the ~158 nodes the full pool refuses in two 18m inferences.
9. **The pool holds compute buffers only** (`state_buffer_type`): on a device whose weights
   `weights_buffer_type` moved to RAM, loom allocates the KV and conv-state caches, retained outputs and
   declared inputs in the CPU's buffer type. Measured with `GGML_VC4_POOL_LOG` and a buffer log in loom,
   Pocket 18m Q8_0_PC's pool at its peak held a 32.0 MB KV cache (allocated first), the Mimi decoder's
   28.1 MB compute buffer, under 1 MB of everything else, and squeezed out the arena: the decoder's
   convolutions, which 7m (an 8 MB cache, same codec) runs on the QPUs, went to the CPU. Not VC4_HOST: a
   copy of what the QPUs read there is cached by address, as for weights that never change, and a cache
   changes every frame. In a CPU buffer the scheduler runs what reads the cache on the CPU (VC4 declines
   an F32 MUL_MAT over a weight outside its buffers): attention, whose frame steps (N = 1) it declined
   anyway. And VC4 declines a MUL_MAT or convolution whose result is already placed outside the pool --
   the QPUs write it by bus address.

## Alternatives considered

* **Zero-stuffing** x and a K-tap convolution: s times the multiply-adds, all but 1/s against zeros.
* **The phases interleaved on the host** after s launches into a [s*OC][Q] scratch: a pass over the
  whole output, 9.8 MB at three of VITS's four stages, on the ARM1176 that writes without allocating.
* **One launch a phase with strided output**: the VDW writes rows of contiguous words, not a stride.
* **GEMM + overlap-add** (ggml-0009's CPU form): K/s partial sums into every output, on the host.
* **Weights placed per buffer by its first use** (CMA until a set_tensor under WEIGHTS usage, then RAM
  mapped over it): the scheduler's compute buffers must be CMA at allocation, where a refusal can still
  fall back (ADR-090), and the weight buffer would be CMA-sized at allocation -- VITS F32 never gets it.
* **Per-weight routing by loom to an extra buffer type**, asking the device op by op: a weight is read
  by nodes of both kinds, and the graph does not exist at load time. With every QPU read a copy, there
  is nothing to route.
* **Freeing the ggml-layout bytes once packed**: one CMA allocation cannot be partly freed.
* **A first-come pool budget for packed copies, reserved at supports_op time**: VITS's text encoder
  took the budget, its vocoder's convolutions were declined, and VITS CPU+VC4 went from 5.86 to 15.5 s.
  Which nodes deserve the pool is not known in the order the scheduler asks.
* **Weights at Q4_0** for Pocket 18m: the pool held no weights (decisions 3 and 6); F32 and Q8_0_PC
  filled it identically, 19 buffers and 63.5 MB.
* **A larger CMA pool** (`dtoverlay=cma,cma-128`): a board setting, not a default install's.

## Consequences

* Kernels on the board (`test_p447_hw convt`, 12 QPUs, f64 reference, rows padded by 3 untouched
  words): VITS's stages 512 -> 256 s8 T 300 121 ms (10.4 GFLOPS), 256 -> 128 s8 T 2408 237 ms (10.7),
  128 -> 64 s2 T 19264 128 ms (9.8), 64 -> 32 s2 T 38530 67 ms (9.4); amy-sized 192 -> 96, 96 -> 48 and
  48 -> 24 s4 17, 33, 31 ms (10.4-11.4 GFLOPS). Correct at one tile on one QPU first, then everywhere.
* Pi Zero W, run `out3`, one process per arm, two inferences each (the second is what a process that
  stays loaded pays); output against the CPU arm:

  | model | cpu s | P4.46 s | P4.47 s | vs cpu | vs P4.46 | 2nd: P4.46 / P4.47 s | output |
  |---|---:|---:|---:|---:|---:|---:|---|
  | VITS Q8_0_PC | 19.68 | 7.88 | 6.68 | 2.94x | 1.18x | 6.92 / 5.74 | SNR 49.8 dB |
  | VITS Q4_0 | 19.06 | 7.14 | 5.79 | 3.29x | 1.23x | 6.23 / 4.99 | SNR 61.8 dB |
  | VITS F32 | 20.16 | 19.68 (CPU) | 5.88 | 3.43x | 3.35x | 19.35 / 5.13 | SNR 63.9 dB |
  | amy F32 / Q8_0_PC | 13.49 / 13.61 | 5.43 / 5.62 | 4.16 / 4.07 | 3.24x / 3.34x | 1.30x / 1.38x | 5.07 / 3.75, 5.18 / 3.83 | SNR 76 dB |
  | heart F32, de051 F32 | 2.00, 3.21 | 0.93, 1.90 | 0.92, 1.54 | 2.17x, 2.08x | 1.01x, 1.23x | 0.79 / 0.77, 1.64 / 1.34 | SNR 59, 84 dB |
  | Citrinet-256 F32 / Q8_0_PC | 10.11 / 7.19 | 3.88 / 3.55 | 3.50 / 3.54 | 2.89x / 2.03x | 1.11x / 1.00x | 3.34 / 2.90, 3.05 / 3.01 | identical / P4.45's near-tie id |
  | Moonshine tiny Q8_0_PC | 20.00 | 13.35 | 13.03 | 1.53x | 1.02x | 12.38 / 12.36 | identical |
  | Moonshine tiny F32 | 25.46 | 25.60 (CPU) | 17.00 | 1.50x | 1.51x | 25.57 / 16.21 | identical |
  | conformer-ctc Q8_0_PC / Q4_0 | 5.89 / 7.43 | 3.96 / 4.20 | 4.19 / 4.40 | 1.41x / 1.69x | 0.95x / 0.95x | 3.58 / 3.55, 4.13 / 3.78 | identical |
  | wake F32, VAD Q8_0_PC | 2.08, 1.95 | 1.35, 1.76 | 1.32, 1.76 | 1.58x, 1.11x | 1.03x, 1.00x | 1.27 / 1.26, 1.71 / 1.66 | SNR 97.7 dB, max\|d\| 1.4e-4 |

  The transposed convolutions alone (run `outA`, weights still in the pool): VITS Q4_0 5.79 s, amy
  4.31 / 4.10, de051 1.62 -- and VITS Q8_0_PC failed (Retro-102). conformer-ctc's first inference pays
  the copies of its F32 matrices (40 MB: the "Q4_0" export keeps its linear layers F32), ~0.2 s with
  slabs (0.45 s with an allocation each), which P4.46 did not pay; its second is P4.46's or better.
  Moonshine Q8_0_PC under P4.46 failed in one run of three: a plain copy of a repacked weight the pool
  could not hold; under P4.47 plain copies are in RAM.

* `test_vc4_backend` runs every case with the weights in the pool and in VC4_HOST, and again with the
  arena capped (every QPU node then on the CPU, results unchanged). `test_host_primary_lowering` holds
  `weights_buffer_type` to its truth table over a fake device and checks GgufModel::load puts the
  weights in the fake host buffer type; it fails with the function returning the device's own.
* No other ggml device is affected: the CPU, BLAS and Metal offer no host buffer type, and the GPU
  backends that do have buffers of their own that are not host memory.
* Pocket TTS 18m (60 frames, 4.80 s of audio, one process an arm, two inferences, one run each):

  | | CPU s | decision 8 only s (`q18`) | + decision 9 s (`poolmap2`) | pool peak | QPU work |
  |---|---:|---:|---:|---:|---|
  | 18m Q8_0_PC | 23.28 / 23.25 | 22.72 / 22.06 | 18.80 / 18.29 | 61.2 -> 47.9 MB | 112 -> 240 MUL_MATs, 2 -> 14 CONV_2D, 0 -> 2 convT; 158 -> 0 refused |
  | 18m F32 | 31.08 / 30.97 | 30.09 / 31.04 | 25.86 / 24.85 | -> 41.9 MB | 56 -> 240 MUL_MATs, 2 -> 18 CONV_2D, 0 -> 2 convT; 162 -> 0 refused |
  | 7m Q8_0_PC | | 14.08 / 13.55 | 14.80 / 13.91 | ~38 -> 39.9 MB | 258 -> 242 MUL_MATs (attention to the CPU) |
  | 2m Q8_0_PC | | 7.08 / 6.74 | 7.24 / 6.92 | ~26 -> 28.8 MB | 160 -> 144 MUL_MATs |

  The F32 arm's two inferences are now identical (they were not when the pool decided placement). The
  latent loop samples, so waveforms are no oracle: the CPU's own Q8_0_PC output is at -1.4 dB against
  its F32 output, and VC4's against the CPU's is 9.3 dB (Q8_0_PC: the QPU kernel keeps activations in
  float, the CPU's quantizes them) and 30.8 dB (F32). Whisper base transcribes all nine outputs --
  CPU and CPU+VC4 for each size and precision -- to the same sentence (the audio reversed does not).
  Three rounds (run `out4`, cold, median and range; the CPU arms equal the run before within 0.3%):

  | | CPU s | CPU+VC4 before decision 9 s (`out3`) | CPU+VC4 s (`out4`) |
  |---|---:|---:|---:|
  | 18m Q8_0_PC / F32 | 23.30 / 31.27 | -- | 19.03 [18.92-19.42] / 25.38 [25.35-25.57] |
  | 7m Q8_0_PC / F32 | 19.00 / 20.65 | 14.13 [14.11-14.13] / 15.05 [14.96-15.59] | 14.64 [14.50-14.68] / 15.34 [15.33-15.64] |
  | 2m Q8_0_PC / F32 | 9.06 / 8.87 | 7.06 [7.06-7.07] / 6.70 [6.65-6.73] | 7.26 [7.22-7.48] / 6.80 [6.79-6.86] |

  So decision 9 costs 7m and 2m 2-4%: their prefill attention (16 MUL_MATs) reads the KV cache, now in
  RAM, so the CPU runs it. The other ten Pi Zero models are within 1-3% of the run before, either way.
* Open: 7m / 2m's prefill attention back on the QPUs (a per-call copy of the cache, N >= 16); the first inference of a model with many F32 matrices copies them into the pool (conformer-ctc
  0.2 s); CONV_TRANSPOSE_1D with a stride that is not a power of two, or with a residual or mask after it,
  stays on the CPU; "auto" has no fallback when another process holds the pool.
