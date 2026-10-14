---
type: adr
status: accepted
date: 2026-10-14
domain: performance
tags: [armv6, pi-zero, vfp, short-vectors, runfast, f32, int8, gemm, gemv, conv, ggml-patch]
---

# ADR-079: ARMv6 F32 Kernels on the VFP11's Short Vectors First, Then Dynamic Int8; Static Int8 Held

Decided on [ADR-077](adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md)'s gate 1, [the
cross-model analysis](adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md#gate-1-the-cross-model-analysis-measured-2026-10-14).
Serves [Epic-05](../epics/epic-05-edge-performance.md) on the tier
[ADR-026](adr-026-armv6-is-the-floor-and-gets-its-own-kernels.md) makes the engine's floor. The kernels
are ggml patches ([ADR-014](adr-014-patch-ggml-rather-than-write-kernels.md)), and they run in RunFast
([ADR-078](adr-078-runfast-on-armv6-compute-threads.md)). Option 1's weights are
[ADR-076](adr-076-quantized-gemms-reach-xnnpacks-microkernels-through-a-ggml-buffer-type.md)'s `Q8_0_PC`.
**Accepted 2026-10-14 (user).** The gates below are ahead, and each can stop its part.

## Context

ADR-077 asked whether static int8 was worth building for the Pi Zero, first for Citrinet-256-ls and
then for the zoo. Its gate 1 profiled 13 models on the board and compared each op class's best F32 kernel
with its best int8 kernel, both in asm. Three findings decide this ADR:

* **The VFP11's short vectors were never used.** With FPSCR `LEN = 4`, one `vmla` is four MACs, and the
  ceiling is 0.94 GMAC/s against 0.49-0.69 for scalar `vmla` in RunFast (`scripts/bench44.c vfpvec`).
  Tiles built on them run Citrinet's 1x1 convs at 0.72 GOP/s, against ggml's 0.50. They run the vocoders'
  direct convs at 1.10-1.11, against 0.25-0.65 in the models. A plain C F32 GEMV streams at 316 MB/s, where ggml's
  ARMv6 GEMV gets 91.
* **At equal effort, int8 still wins every GEMM-shaped class, but by less:** 1.46x (1x1 convs), 1.06x
  (vocoder convs), 2.14x (GEMV) with a dynamic quantize, and 1.66x / 1.31x without one (static).
  The int8 depthwise conv lost (0.74x, C in both arms).
* **The seconds** (`scripts/armv6_projection.py`, projected from those rates over each model's class
  profile): the F32 items alone give 1.3-3.0x on 11 of 13 models, with Citrinet at RTF 0.94 and de051 at 0.74. Dynamic int8 adds
  1.02-1.33x, and 3.25x for distilbert-ner, whose F32 weights do not fit the board. Static int8 adds a further
  1.05-1.39x over dynamic, and the vocoders gain most.

## Options

1. **F32 kernels only.** The largest step, with no new weights or numerics, but it leaves int8's 1.3-2.1x on
   the GEMV and GEMM classes unclaimed.
2. **F32 kernels, then dynamic int8 where it wins** (ADR-077's option 1). It needs no calibration and no
   new op, and it reads `Q8_0_PC` weights that every backend still sees as Q8_0.
3. **Option 2 on top: static int8 through the graph** (ADR-077's option 2).

## Decision

**Option 2 of this list.** First the ARMv6 F32 kernel family, then dynamic int8 kernels for the
classes where int8 measured ahead. **Static int8 is held** (below).

**The F32 family**, ordered by the zoo's seconds (the hub items carry the P-numbers):

| item | kernel (bench it comes from) | where the time is |
|---|---|---|
| P4.32a | the short-vector harness: `LEN` set only around integer-only loops, plus the gate-1 test below | all of them |
| P4.32b | direct conv, 4 channels x 4 time steps, taps unrolled per (k, d), input window packed per block (`bench49`, `gen_armv6_conv_tiles.py`) | VITS 46.8 s, amy 29.1, de051 5.4, heart 0.7, the vocoders' transposed convs |
| P4.32c | F32 GEMV, four rows a pass (`bench48`) | Moonshine 20.2 s, Silero 2.0 |
| P4.32d | GEMM, 4x4 tile over K chunks of 128, X packed per time block, TB 24-32 (`bench_xnnpack` `loom f32cf`) | Citrinet 10.9 s, Moonshine 16.1 + 10.2 (stem), conformer, heart, WakeHuBERT |
| P4.32e | depthwise conv, the blocked kernel (`bench43`), on short vectors if they help it | Citrinet 4.1 s |
| P4.32f | bias, mask and activation folded into the producer's epilogue | 1.3-3.7 s in seven models |
| P4.32g | an FFT for the DFT-basis convolutions (STFT, iSTFT). This is an op in the topology, not a kernel, and needs its own design | 1.2-2.1 s in five models |

**Dynamic int8 (P4.33)** applies to the two classes where it measured ahead of the F32 kernels: the GEMV (2.14x,
`bench48`'s asm) and the GEMMs (1.46x, ADR-077 gate 2's tile, generalized beyond K = 256 and to small T
for distilbert-ner and conformer-ctc). **Not the vocoder convs.** At 1.06x the dynamic quantize eats
int8's whole edge, which does not carry a second kernel. Activations are quantized per time step.
The tile holds them as int16, so the int16 range is free where SMLAD's int32 sum has the headroom
(K <= 512 at full range). Above that, the range is the accuracy gate's choice.

**Gates, in order. Each can stop its part:**

1. **Short vectors are safe in the engine** (P4.32a). The test, on the board: a signal handler doing
   float arithmetic fires during a `LEN = 4` kernel, and both the handler and the kernel give the scalar
   results. Linux is expected to clear `LEN` for a handler and to save FPSCR on a switch; this gate checks that
   instead of assuming it. The kernel must call nothing while `LEN != 0`.
2. **Each kernel reproduces its bench rate inside ggml**, within 5%, on the bench's shapes.
3. **Numerics across the ARMv6 zoo**, through ADR-078's harness (`scripts/armv6_runfast_gate.sh`, on the
   board, since qemu differs in this mode, [Retro-086](../retros/retro-086-qemu-matches-the-board-only-where-the-vfp-never-bounces.md)).
   The bar is today's tolerances: the tensor oracle for ASR and NER, and the ASR oracle for every TTS model.
   The kernels change summation order, so outputs move at rounding level and no further.
4. **End to end, cooled and paired, per model**, against the projection's column (a). A kernel that
   does not deliver end to end is not on by default. An isolated op's rate is a ceiling.
5. **Option 1, after the F32 family has shipped:** per class, the int8 kernel beats the shipped F32 kernel
   end to end. WER is checked on the 73 LibriSpeech-dummy clips (thresholds fixed before the first run),
   along with the tensor oracle on the log-probs and the NER labels.

## Why static int8 is held

* **It buys the least of the three steps:** 1.05-1.39x over dynamic int8, projected, after the F32
  items have taken 1.3-3.0x and dynamic int8 up to 1.33x.
* **It costs the most.** It needs a calibration step in the exporter and quantized variants of five ops, each
  with an ARMv6 kernel. That would be on top of the x86 and NEON ones, which ADR-076's bridge provides only for
  GEMMs. It also needs a CPU-only model flavour per model, and an accuracy gate per model.
* **Its gain sits where int8 activations are riskiest.** The vocoders gain 1.30-1.39x from it, because
  there the dynamic quantize is all of int8's edge. But a vocoder's activations become the waveform,
  and int8 error there is heard rather than read. That is the ASR oracle's territory, model by model.
* **It crosses no real-time line that the cheaper steps leave short, once the DFT-basis convs go to an FFT.**
  Citrinet and de051 cross on the F32 items. heart crosses on static int8 only without an FFT (RTF 1.15 /
  1.03 / 0.94), and with one it is ~0.6 on the F32 items alone. Moonshine tiny, amy and VITS stay above 1
  under all three.

**What would reopen it:** a model with a real-time target that the shipped F32 kernels and dynamic int8
miss by less than static int8's projected margin, measured end to end. And before that, a cheaper
middle path should fail first: a producer that emits int8 straight from its epilogue. Where the scale is
per time step, as in the 1x1 convs and the GEMMs, a row's amax is known when the row is written. That
removes the separate quantize pass with no calibration. A per-tensor scale, as in the vocoders' convs,
does not allow it.

## Consequences

* The ARMv6 build gains a kernel family that only it uses, as ADR-026 allows. It is generated asm
  (`scripts/gen_armv6_*`), with the generators kept beside the patches.
* The engine runs part of its ARMv6 F32 math in short-vector mode, which no other target has. Every
  kernel that sets `LEN` restores it before returning, and gate 1 is the evidence that this holds under
  signals.
* Every ARMv6 model gains without a re-export from the F32 family. Option 1 needs `Q8_0_PC` exports
  (ADR-076), and the Hub's Q8_0 cards stay valid as they are.
* ADR-077 is held at its option 2. Its gates 1-2 stand as the measurements this ADR was decided on.

## Gate 1, measured 2026-10-14: passed

**The harness** is `scripts/armv6_shortvec.py`, which the kernel generators import. `enter` sets `LEN = 4,
STRIDE = 1` and keeps the thread's RunFast bits. `leave` clears `LEN` and `STRIDE` only, because restoring a
saved FPSCR would also roll back the exception flags the kernel raised. `lint` runs on a generator's whole
output and rejects these inside a region: any way out of the function (`bl`, `blx`, `bx`, `svc`, a pc
write), a second `vmsr`, f64 data processing, and a vector operand that would wrap around its bank. It also
rejects a function that returns inside a region. So "calls nothing while `LEN != 0`" is checked when the
asm is generated. Each case was shown to raise.

**The test** is `scripts/armv6_shortvec_gate.c`. Two threads run a GEMM-tile kernel whose driver loops sit
inside the region. A third thread runs scalar float probes: a bank-1 `vmla` whose result depends on `LEN`,
plus compiled C math. A forked process does the same, all on the one core. A CLOCK_MONOTONIC timer signals
each thread every 0.1 ms. The handler reads its live FPSCR and the interrupted one (the sigframe's VFP
block), then runs the probe, the C math, and the short-vector kernel itself. Results are compared bit for
bit with a scalar reference made in the same mode. A pass needs zero mismatches, and it also needs evidence
that the case happened: signals that landed inside the kernel with `LEN = 4` saved, and involuntary switches.

Pi Zero W, kernel 6.6.28+rpt-rpi-v6, 20 s per mode (parent + child summed):

| mode | kernel calls (bad) | scalar checks (bad) | signals inside the kernel at `LEN = 4` | handler `LEN != 0` | involuntary switches |
|---|---|---|---|---|---|
| RunFast | 42,286 (0) | 1,321,516 (0) | 117,940 | 0 | 17,877 |
| IEEE, every `vmla` underflows | 1,160 (0) | 813,533 (0) | 117,796 | 0 | 18,080 |

In IEEE mode one call took 5.19 ms against 0.19 ms, so every vector `vmla` bounced. The kernel's VFP
support code emulated all four lanes exactly, under signals. This means the `GGML_CPU_RUNFAST=0` path is
slow but correct. **Sabotage turns the gate red.** With `SV_SABOTAGE=drop`, the handler clears `LEN` in the
saved FPSCR, and all 25,215 kernel calls come out wrong. With `SV_SABOTAGE=leak`, it sets `LEN` in a scalar
thread's saved FPSCR, and 309,398 of 309,464 scalar checks fail.

**qemu-user fails this gate, and the board passes it.** qemu (`loom-armv6-build:bookworm`) enters a handler
with the interrupted `LEN` still set, so every handler's own float math was vectorized. The interrupted
kernel's results stayed right. This adds to [Retro-086](../retros/retro-086-qemu-matches-the-board-only-where-the-vfp-never-bounces.md):
under qemu, ARMv6 tests of short-vector kernels must not depend on a signal handler that does float math.
