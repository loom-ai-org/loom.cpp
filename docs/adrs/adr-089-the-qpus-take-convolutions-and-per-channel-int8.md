---
type: adr
status: accepted
date: 2026-10-28
domain: backends
tags: [vc4, qpu, raspberry-pi-zero, convolution, int8, q8_0_pc, ggml-backend-sched, weights]
---

# ADR-089: The QPUs Take Convolutions and Per-Channel Int8

P4.45 in [Epic-04](../epics/epic-04-backends-and-accelerators.md), the open items of
[ADR-088](adr-088-the-pi-zero-qpus-are-an-accelerator-beside-the-cpu.md). Code:
`backends/ggml-vc4/qpu_conv1d.h`, `qpu_gemm_q8.h`, `ggml-vc4.cpp`; `src/core/gguf_model.cpp`;
`cmake/patches/ggml-0056`. What broke on the way: [Retro-097](../retros/retro-097-an-emulator-that-runs-the-qpus-one-at-a-time.md)
(a QPU race the emulator could not see), [Retro-098](../retros/retro-098-a-scheduler-that-placed-weight-ops-by-accident.md)
(weight ops placed by accident).

## Context

P4.44 gave the Pi Zero's QPUs F32 MUL_MATs, and the shipped models barely used them. Profiled on the
board (CPU, `LOOM_PROFILE` with node sources): convolutions are most of every TTS and ASR model --
VITS Q8_0_PC 15.0 s of 19.4, amy 9.5 of 13.3 (F32 kernels), Citrinet 4.8 of 6.8, Moonshine tiny's
encoder 3.4 -- and per-channel int8 MUL_MATs most of the rest (Moonshine 6.7 s, NER, conformer).
A convolution as an F32 GEMM needs its im2col: 224 x 75 008 floats, 67 MB, for one VITS vocoder layer.
An int8 weight cannot be read as ggml stores it: a Q8_0 block is 34 bytes, so half of them start off a
word, and the TMU reads words.

## Decision

1. **1-D convolutions read in place** (`qpu_conv1d.h`): lane l is output position t0 + l, gathered
   from x through the TMU -- 16 consecutive floats per k at stride 1, so the cache serves the KW - 1
   shifted reads after the first -- and accumulator r is output channel oc0 + r, its weight from the
   uniform stream. The im2col is an address step per k: d*4 within an input channel, x_row - (KW-1)*d*4
   to the next. Any CONV_2D with H = 1, declared or folded (ggml-0013), any kernel type with a
   `to_float`, F32 included.
2. **The input is staged zero-padded, in chunks along T** of about 2 MB, so every position is the
   QPUs' (a host's scalar edges cost more than the QPUs on a short convolution) and the arena stays
   small inside a 64 MB pool. In place, with the host computing the edges, only when x already lives in
   GPU memory and the edges are small.
3. **Output**: a tile is one DMA when dst's rows are close enough for the 13-bit VDW stride
   (T_out up to ~2048); a VITS row is 300 KB apart from the next, so there each row is its own DMA.
4. **Weights are packed once**, F32 in the uniform stream's order, and kept while the pool keeps a
   24 MB reserve, else packed into the arena per call (as the CPU dequantizes per call); dropped when
   the vc4 buffer holding the weight is freed -- a pointer is not an identity.
5. **An op runs where its weights are**: loom marks its weight buffers `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`,
   and `ggml-0056` makes the scheduler's rule see a view of a weight as a weight.
6. **Per-channel int8 on the QPUs** (`qpu_gemm_q8.h`): offset-binary bytes, u = q + 128, four to a
   TMU word, each read by the multiply that uses it through r4's 8-bit unpack -- which yields u/255 as
   a float (measured: truncated, every u). So activations are packed x 255 and each column carries
   128 * sum x; the epilogue is (acc - 128 sum x) * s_row. 65 instructions per 4 k against the F32
   kernel's 68, a quarter of its TMU traffic. The layout lives in a `VC4_Q8` extra buffer type; loom
   gives a MUL_MAT over a CPU-repacked per-channel weight a copy there when the host-memory primary,
   asked with a stand-in, would take it -- from 16 columns, where it wins; decode steps keep the CPU's.
7. **The device reports its pool less 16 MB**, the room the QPUs work in, to loom's room check.

## Alternatives considered

* **im2col on the host, then the GEMM.** 67 MB of host writes per VITS layer, beside the conv itself.
* **The convolution's weights through the TMU, activations in uniforms.** The im2col again, packed per
  call into the uniform stream; or 16 TMU requests per k, one per output position.
* **Signed int8 through r4.** The unpack is a colour conversion, unsigned only; a sign fix per weight
  would cost the instructions the unpack saves. Offset binary moves the sign to one subtraction a tile.
* **Reading ggml's Q8_0 blocks in place.** 34-byte blocks put half the rows off a word.
* **Placing ops with `ggml_backend_sched_set_tensor_backend` from loom.** A second placement policy
  beside the scheduler's; marking the weights uses the one ggml and llama.cpp already have.
* **The host computing the padded edges always.** Measured 30 ms of scalar work against 20 ms on the
  QPUs for a 293-step VITS layer.

## Consequences

* Pi Zero W, `cpu` against `"auto"` (run 5, one process per arm), output against the CPU's:

  | model | cpu s | auto s | | output |
  |---|---:|---:|---:|---|
  | vits Q4_0 | 18.90 | 7.67 | 2.46x | SNR 60.0 dB, same length |
  | vits Q8_0_PC | 19.63 | 8.56 | 2.29x | SNR 52.8 dB, same length |
  | amy F32 / Q8_0_PC | 13.38 / 13.46 | 6.09 / 6.08 | 2.20x / 2.22x | SNR 75.6 / 76.2 dB |
  | Citrinet-256 F32 | 9.88 | 3.48 | 2.84x | identical ids |
  | heart F32 / Q8_0_PC | 1.97 / 1.70 | 0.89 / 0.98 | 2.21x / 1.73x | SNR 60.4 / 59.1 dB |
  | de051 F32 / Q8_0_PC | 3.17 / 3.21 | 1.97 / 2.11 | 1.61x / 1.52x | SNR 84 dB |
  | Moonshine tiny Q8_0_PC | 19.75 | 12.99 | 1.52x | identical transcript |
  | conformer F32 / Q8_0_PC / Q4_0 | 5.92 / 5.74 / 7.47 | 4.19 / 4.00 / 6.11 | 1.41x / 1.44x / 1.22x | identical text |
  | wake F32, VAD F32 / Q8_0_PC | 2.08, 2.35 / 2.02 | 1.40, 2.20 / 1.71 | 1.48x, 1.07x / 1.18x | SNR 84-110 dB |

  Citrinet Q8_0_PC differs from the CPU in one CTC id (41 -> 33 at frame 15): a near tie -- the CPU's
  own profiled run produced the same 33.
* Per convolution on the board: VITS's 32-channel k = 7 layer 782 ms on the CPU, 104 ms on the QPUs
  (10.3 GFLOPS); Moonshine's encoder stem 2689 -> 195 ms.
* Marking weights also moves F32 MUL_MATs with weights to the QPUs where only expansion reached them
  before (conformer: 145). On a GPU primary nothing moves: RADV's device-parity gates run the same
  2 069 nodes with no fallback.
* Open: Q4_0 weights in the int8 kernel (per-block scales); the tile order for large N; the bias and
  activation the CPU fused into a convolution, now separate CPU nodes; staging the next chunk while the
  QPUs run this one; Citrinet's pointwise weights, which CPU_ARMV6_I8 holds and the QPUs could take as
  MUL_MATs are taken.
