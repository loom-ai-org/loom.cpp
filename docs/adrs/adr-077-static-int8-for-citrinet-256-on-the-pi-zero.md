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

## Gate 1, measured 2026-10-14: the split, with the verdict held for a cross-model analysis

**The split.** The profile of the Q8_0 card (rc16, jfk, one thread, 35.9 s of node time):

| op | s | what it is | does int8 remove it? |
|---|---|---|---|
| `CONV_2D` (1x1) | 27.4 | the GEMMs | gate 2 |
| `CONV_2D_DW` | 4.90 | depthwise, K = 5-41, 107 calls | measured below: no |
| `MUL` | 1.31 | NeMo's padding mask before every conv, the identity at batch 1 | no: skip or fuse it |
| `ADD` | 0.78 | the 1x1 conv's bias, plus the residuals | no: a GEMM epilogue in either precision |
| `UNARY` | 0.59 | ReLU, the SE gate's sigmoid | no: same |
| the rest | ~0.8 | front end (`SQR`, `CONCAT`, `PAD`, mel `MUL_MAT`), the SE gates' small GEMMs | no |

BatchNorm is already folded at export (conv, then a bias `ADD`, then `RELU`). The 148 full-size `ADD`s
are about one per 1x1 conv plus the residuals, so ggml-0005's bias fusion apparently does not fire on
this conv path. That is not confirmed yet.

**The depthwise convolutions** (`scripts/bench43.cpp`): the whole jfk schedule replayed on the board,
best of 5, in two rounds that agree within 2%. Every arm was verified before it was timed: `f32` is equal to
ggml bit for bit, and `i8` is exact against an int64 reference.

| arm | jfk's 107 dw convs | GMAC/s | |
|---|---|---|---|
| ggml today (ggml-0021: a scalar `vec_mad` sweep per tap) | 3.96 s | 0.044 | |
| F32, 8-output register block, unrolled window | 0.67 s | 0.262 | 5.9x |
| **F32, same, VFP in RunFast (FZ \| DN)** | **0.50 s** | 0.348 | **7.9x** |
| int8 SMLAD (SXTB16 unpack, SMMULR/SSAT requant) | 0.68 s | 0.258 | 0.74x of RunFast F32 |

**No op in the 9.0 s gets faster as int8 than as VFP F32 in this microbenchmark**, so by this gate's
rule as written option 2 would reduce to option 1. The ceilings (`scripts/bench44.c`) bound how wrong
that could be. SMLAD with the loads a kernel needs is ~1.07 GMAC/s, and `vmla` is 0.41 GMAC/s in IEEE
mode and 0.59-0.68 in RunFast. The int8 depthwise kernel sits at ~24% of its ceiling and F32 at ~55%,
so a tuned int8 depthwise conv might reach ~2x F32: at most ~0.25 s of jfk against 0.50 s.

**The verdict is held (user, 2026-10-14), because the rule asks too narrow a question.** It weighs
one model's non-GEMM ops, and the ~0.25 s undercounts static int8 three ways:

* **It leaves out the activation quantize**, which static int8 removes from every GEMM. Gate 2 below
  measures it at 0.78 s of jfk's 5.75 s, before a channels-last layout.
* **The int8 depthwise arm got C, not asm.** The GEMM went from 0.64 to 1.20 GOP/s between those two
  (gate 2), so the int8 depthwise number is not yet that arm's best.
* **A family of ARMv6 int8 kernels serves every model on the board, not just Citrinet.**
  [ADR-026](adr-026-armv6-is-the-floor-and-gets-its-own-kernels.md) already accepts kernels written
  for this tier alone. The question is what the family buys across the zoo, not across one model.

### The analysis that decides gate 1

1. **Models:** every zoo model that runs on the Pi Zero. That means the hub's Pi Zero set (Citrinet-256-ls,
   Moonshine tiny, sanoTTS heart-nano, the piperlite voices) and the P7 set (VITS, conformer-ctc,
   distilbert-ner), plus any small model whose op mix differs (a VAD, a wake word). The first step fixes
   that list.
2. **Per model, a one-thread profile by op on the board**, as above, so that every op class carries
   seconds rather than a guess.
3. **Per op class, the best F32 kernel (IEEE and RunFast) against the best int8 kernel**, each at the
   same level of effort (asm where either arm needs it) and each read in its disassembly before its
   ratio is quoted. The classes are the 1x1 conv / `mul_mat`, the conv with k > 1, the depthwise conv,
   `conv_transpose_1d`, attention (`QK^T`, softmax, `AV`), the norms, and the elementwise ops fused into
   a requantize.
4. **Static vs dynamic:** what the int8 tensors between ops save (no per-GEMM quantize, a quarter of the
   activation traffic), and what the islands cost (`QUANTIZE` / `DEQUANTIZE` around softmax, norms and
   the front end).
5. **The cost side:** exporter calibration, the quantized op variants and their ARMv6 kernels, and an
   accuracy gate per model (WER, or the ASR oracle for TTS).

**Output:** per model, the seconds left under (a) the F32 items below, (b) those plus a dynamic int8
GEMM (option 1), and (c) full static int8 (option 2). The decision between options 1 and 2 is taken on
that table, not on Citrinet's depthwise convs alone.

**The Context's premise is withdrawn as well.** "Even an infinitely fast GEMM leaves RTF 0.82" assumed
the 9.0 s was F32 arithmetic. It is F32 *inefficiency*: a depthwise kernel at 4% of the VFP's rate, and
2.7 s of elementwise nodes that fusion removes in any precision. With those fixed, the non-GEMM part
plausibly drops to ~1.5-2.5 s. That is unmeasured: an isolated op's number is a ceiling, not an
estimate, and it has to be confirmed end to end. Then the GEMM is the whole question. At gate 2's
1.2 GOP/s (~5 s) jfk would take ~7 s (RTF ~0.65). At XNNPACK's 0.6 (~10 s) it would take ~12 s
(RTF ~1.1).

**Three F32 items this surfaced.** None needs int8, and each helps every ARMv6 model, not just
Citrinet:

1. **An ARMv6 depthwise-conv kernel** in a ggml patch: 7.9x on this schedule, measured in isolation.
2. **RunFast on ARMv6 compute threads.** It gives `vmla` 1.65x throughput, and it should also stop the
   VFP11 trapping denormals to software support code. It is a numerics change: denormals flush to zero
   and NaNs become the default NaN. So it needs its own decision and a gate across the ARMv6 zoo
   ([ADR-078](adr-078-runfast-on-armv6-compute-threads.md)).
3. **Skip or fuse the padding-mask `MUL`, the bias `ADD` and the `RELU`** around Citrinet's 1x1 convs.

RunFast is scoped across the zoo in [ADR-078](adr-078-runfast-on-armv6-compute-threads.md). Gate 2,
the int8 GEMM, is needed by both options and goes ahead. Its comparison arm is now F32 in RunFast as
well as XNNPACK's `qs8`, since RunFast moved F32's ceiling.

## Gate 2, in progress 2026-10-14: 1.20 GOP/s at the GEMM, 1.04 with the activation quantize

These are dev-bench numbers, not yet the gate. The gate is measured in `bench_xnnpack.cpp` beside
XNNPACK on these exact shapes. Citrinet's 1x1 convs are `Y[256][T] = W[256][256] . X[256][T]`, channels
first. The bench weights them as jfk runs them: 7 calls at T = 1101, 36 at 551, 42 at 276 and 46 at
138. All results are on the Pi Zero, in two rounds that agree within ~2%.

| arm | jfk's 1x1 convs | GOP/s |
|---|---|---|
| ggml today: Q8_0 `conv_2d_direct_packed` + bias | 22.0-24.2 s | 0.25-0.27 |
| F32, 4x4 VFP tile, naive layout (RunFast) | 34.5 s | 0.17 |
| int8, C, SMLAD 2x4 tile, int16 operands | 9.2 s | 0.64 |
| **int8, asm 2x4 tile (K = 256 unrolled), fast quantize** | **5.72-5.78 s** | **1.04** |
| the same without the activation quantize | ~4.95 s | **~1.20** |

The kernel is dynamic int8 (option 1). X is quantized per time step (symmetric), widened to int16 and
transposed to `[T][K]`. W is per-channel symmetric int8, widened to int16 at load. A 2x4 tile does 8
SMLADs and 6 loads per K-pair, with every load consumed at least 3 instructions later. Its output is
exact against an int64 reference, and every variant is bit-identical to the first. The accuracy vs
F32 is ~1.1e-2 of max|y|, which is gate 3's concern, not this one's.

**Where the remaining time goes**, from diagnostic variants that pin one operand in cache:

| part of jfk's 5.75 s | s |
|---|---|
| the tile on L1-resident operands | ~3.1 (1.95 GOP/s alone) |
| epilogue (int32 to F32 with the two scales and the bias) and loop | ~0.5 |
| W from DRAM: 128 KB of int16 weights never stay in the 128 KB L2; ~80 ns per line, no prefetcher | ~0.9 |
| X lines evicted by the W stream (the L1 replaces at random) | ~0.4 |
| quantize: transposing channels-first X is latency-bound; t-blocking and RunFast do not help | 0.78 |

The ARM does see the L2: a streamed operand costs 25-38 ns per line at 32-96 KB and ~80 ns past
128 KB (`memprobe`). PLD did not help in C (2.5%) and hurt in asm. RunFast changes nothing here
(~1%).

**Levers not yet tried**, roughly by expected value:

1. A channels-last encoder, which the Decision already wants for the static design. Quantize becomes
   one sequential pass, and its 0.78 s mostly goes.
2. A K-split or int8 W, so the weights stay in the L2. That costs ~19% more instructions for the
   `SXTB16` unpack, so it is close to a wash.
3. An asm epilogue.

**Projection, unmeasured end to end:** 1x1 convs ~5.7 s, depthwise ~0.5 s, elementwise fused,
front end ~1 s. That puts jfk at ~7.5-8 s, RTF ~0.7, from today's 37.5 s.

## Consequences

* The work is front-loaded on two measurements (gates 1 and 2) that cost a day. Neither needs the
  exporter, the calibration or a new op.
* If it goes through: a calibration step and a `Q8_0_PC`+static export flavour in the exporter;
  quantized `mul_mat`, depthwise conv, add, quantize and dequantize in ggml patches (with
  ARMv6-specific kernels to maintain); and a model that is CPU-only in its int8 form. Its F32 and
  Q8_0 cards stay as they are.
* **The hub's RTF ~1.5 is withdrawn.** It is ~1.8 from int8 GEMMs at XNNPACK's ARMv6 rate, and real
  time needs the whole graph.
