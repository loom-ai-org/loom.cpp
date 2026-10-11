---
type: adr
status: proposed
date: 2026-10-11
domain: performance
tags: [cpu, int8, qs8, static-quantization, calibration, armv6, pi-zero, citrinet, exporter, ggml-patch]
---

# ADR-077: Static Int8 for Citrinet-256 on the Pi Zero, Behind Gates That Can Stop It

Step 4 of the LiteRT/XNNPACK queue in the [hub](../backlog/active-index.md#engine--performance). Serves
[Epic-05](../epics/epic-05-edge-performance.md) and [Epic-03](../epics/epic-03-model-coverage.md)
(Citrinet-256-ls). Shares the weight format of
[ADR-076](adr-076-quantized-gemms-reach-xnnpacks-microkernels-through-a-ggml-buffer-type.md), and
puts its kernels in ggml patches per [ADR-014](adr-014-patch-ggml-rather-than-write-kernels.md).
**Proposed:** each gate below can stop it, in order.

## Context

Citrinet-256-ls is the zoo's best candidate for real-time ASR on a Pi Zero W, and it misses. jfk
(11.0 s) takes **37.5 s at Q8_0, RTF 3.4** (64.0 s at F32). **76% of that is the 1x1 convolutions**:
~6 GFLOP at ~0.22 GFLOP/s, run as F32 because a Q8_0 conv dequantizes its kernel (ggml-0013). The
other 24% (9.0 s: depthwise convs at k = 11-41, squeeze-excitation, residual adds, ReLU, the front
end) has not been split by op. The model is 21 blocks of 5 repeats of depthwise conv -> 1x1 conv
256 -> 256 -> ReLU, with strides that take jfk's 1100 frames to 550, 275 and 138.

**The hub's estimate was taken against the wrong baseline.** "RTF 3.4 -> ~1.5 from the GEMMs alone"
applied XNNPACK `qs8`'s 3.9x over ggml's Q8_0 `mul_mat` (0.15 GFLOP/s). Citrinet does not run that
path; it runs the F32 conv sweep at 0.22. Against what it does run, with the rates measured on the
board (`bench_xnnpack.cpp`, VITS and whisper shapes, flat at K <= 1344):

| GEMM engine on the 1x1 convs | rate (GOP/s) | 1x1 convs | whole jfk | RTF |
|---|---|---|---|---|
| today (F32 sweep over a dequantized Q8_0 kernel) | 0.22 | 28.5 s | 37.5 s | 3.4 |
| XNNPACK `qd8` (dynamic; scalar on ARMv6) | 0.23-0.32 | ~22 s | ~31 s | ~2.8 |
| XNNPACK `qs8` (static; ARMSIMD32, 2x2c4 tile) | 0.54-0.62 | 10.6 s | 19.6 s | **~1.8** |
| infinitely fast | - | 0 | 9.0 s | 0.82 |

So **int8 GEMMs alone cannot reach real time on this board.** Even an infinitely fast GEMM leaves
RTF 0.82. Real time (<= 11.0 s) needs the other 24% made faster as well: for instance 1x1 convs at
~1.2 GOP/s (5.2 s) and the rest at 1.55x (5.8 s).

What XNNPACK offers on ARMv6 (`src/configs/*-config.c` at the pinned commit):

* **int8 GEMM:** `qs8_qc8w` 1x2c4 / 2x2c4 ARMSIMD32 tiles.
* **No SIMD kernel** for `qd8` (scalar 1x2), the int8 depthwise conv (`3p1c/9p1c/25p1c__scalar_fmagic`)
  or the int8 add (`scalar_u1`).

The ARMv6 instruction bound is SMLAD's two 16x16 MACs per cycle, 4 GOP/s at 1 GHz. That is an upper
bound, not an estimate; XNNPACK reaches 15% of it. loom's own ARMv6 int8 kernel (`tinyBLAS_Q0_ARMV6`,
Q4_0 x Q8_0) measured 0.34 GOP/s.

**`qd8` vs `qs8` is not dynamic vs static here.** Both inner loops are int8 x int8 -> int32. On ARMv6
their 2x gap is which kernels XNNPACK happens to carry. Static int8 buys exactly two things over a
dynamic int8 GEMM: no per-op activation quantize, and int8 storage and arithmetic for the non-GEMM 24%.

## Options

1. **Dynamic int8 GEMM, F32 elsewhere** (ADR-076's shape, with an ARMv6 kernel). No calibration, no
   new ops, drop-in for any Q8_0_PC export. Capped at 9.0 s + GEMM time: RTF >= ~1.3 even at 1.2 GOP/s.
2. **Static int8 through the whole encoder** (LiteRT's full-int8). Calibrated activation scales, int8
   tensors between ops, quantized variants of five ops. The only option whose ceiling is below RTF 1.
3. **Fewer FLOPs** (a smaller checkpoint). Out of scope; the 10M model is already the family's leaf.

## Decision

**Option 2 is the target design, built in the order of option 1 first,** because its first
deliverable (a fast ARMv6 int8 GEMM) is option 1's whole content and is worth having on its own.

* **Weights:** per-channel int8 as row-uniform Q8_0 (`--quantize Q8_0_PC`, ADR-076), with BatchNorm
  folded into the conv weights and biases at export.
* **Activations:** int8, per-tensor asymmetric (scale, zero point). The scales come from calibration
  at export, over LibriSpeech clips, and are written as node attributes in the topology.
* **Graph:** the exporter fuses ReLU into each requantization clamp and the residual add into an int8
  add with rescale. It wraps F32 islands (mel front end, the SE gate's sigmoid, the CTC `log_softmax`)
  in explicit `QUANTIZE` / `DEQUANTIZE` nodes. It emits the encoder channels-last, so that a 1x1 conv
  is a plain `mul_mat` and no transposes are needed.
* **Kernels, in ggml patches:** on NEON and x86, ADR-076's bridge carries XNNPACK's `qs8` kernels. On
  ARMv6 they are loom's own, in the `tinyBLAS_Q0_ARMV6` lineage, because XNNPACK's ARMv6 coverage is a
  2x2 GEMM and scalar code elsewhere.

**Gates, in order. Each one can stop the work:**

1. **Split the 9.0 s by op on the board** (the profiler, one thread). If no op in it gets faster as
   int8 on ARM1176 than as VFP F32 in a microbenchmark, option 2 reduces to option 1 and stops there.
2. **An ARMv6 int8 GEMM at >= 1.2 GOP/s** on `[T x 256] x [256 x 256]`, T = 138-550, measured in
   `bench_xnnpack.cpp` beside XNNPACK's 0.54-0.62. Below that, real time is out of reach (above), and
   the gain is option 1's.
3. **WER** on the 73 LibriSpeech-dummy clips, against F32's 9.39%, with the threshold fixed before
   the first run, plus a tensor oracle on the log-probs (a WER alone hid an encoder defect before).
4. **jfk on the Pi Zero, cooled and paired**, against the Q8_0 card's 37.5 s.

## Consequences

* The work is front-loaded on two measurements (gates 1 and 2) that cost a day. Neither needs the
  exporter, the calibration or a new op.
* If it goes through: a calibration step and a `Q8_0_PC`+static export flavour in the exporter;
  quantized `mul_mat`, depthwise conv, add, quantize and dequantize in ggml patches (with
  ARMv6-specific kernels to maintain); and a model that is CPU-only in its int8 form. Its F32 and
  Q8_0 cards stay as they are.
* **The hub's RTF ~1.5 is withdrawn.** It is ~1.8 from int8 GEMMs at XNNPACK's ARMv6 rate, and real
  time needs the whole graph.
