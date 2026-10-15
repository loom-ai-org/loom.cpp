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

## P4.32b, measured 2026-10-14: the direct-conv tile in ggml (`ggml-0032`)

**What shipped.** `scripts/gen_armv6_conv_f32.py` generates one kernel per width k = 1..11 into
`arch/arm/armv6-conv-f32.inc`, a top-level asm block in a header, so the patch needs no build change. It is
bench49's tile with the dilation passed as an argument (one `add` per tap, issued while a vector `vmla`
iterates), and the time-tile loop sits inside the harness region. Ahead of ggml's own paths, `ops.cpp`
takes every 1-D, stride-1, F32 convolution with k = 2..11; a Q4_0 kernel arrives dequantized. k = 1 stays
with the GEMM paths until P4.32d. Per block of TB = 64 steps (fewer while the window would pass 24 KB),
the path copies the window in C, applying padding and the fused LEAKY_RELU. Then it runs the kernel per
channel quad, with the bias in the accumulators. Output goes straight to `y` unless one of three cases
stages it: a fused residual, a partial last quad (OC = 1 becomes a padded quad), or an output over the
input. In the last case the block is held until the next window is copied. `GGML_CPU_ARMV6_CONV=0` turns
the path off.

**Correctness.** `tests/ci/test_conv_1d_sweep.cpp` runs 16 shapes x {plain, fused leaky+bias+residual,
resblock through ggml-alloc with the output over its input (22 cases)} x {1, 2 threads} against a double
reference: 196/196 on x86, in qemu, and on the board. A sabotaged tile (two W lanes swapped in the odd-k
fixup) fails 63 checks.

**Gate 2** (`scripts/bench50.cpp`: bench49's seven shapes as ggml graphs, conv + fused bias). Pi Zero W,
one thread, three quiet runs: **1.157-1.162 GOP/s**, against bench49's 1.103-1.115 in the same window and
0.43 for ggml today (13.4 s → 5.04 s, 2.67x). Every shape is at or above its bench rate (de051 +11%, amy d5
+10%, VITS and amy 48ch k7 level). Earlier runs had one shape per run 30-75% slow. The cause was ssh polling
of the board, and the runs taken without it agree within 0.5%.

**Gates 3 and 4** (`scripts/armv6_runfast_gate.sh` with `VAR=GGML_CPU_ARMV6_CONV`, eleven models, two
rounds ABBA, one inference per launch):

| model | off | on | speed-up | RTF off → on |
|---|---|---|---|---|
| piperlite amy (5.9 s) | 39.8 s | 22.9 s | **1.74x** | 6.75 → 3.88 |
| piperlite de051 (4.4 s) | 8.41 s | 4.84 s | **1.74x** | 1.91 → 1.10 |
| VITS (3.2 s) | 60.3 s | 34.8 s | **1.73x** | 18.9 → 10.9 |
| sanoTTS heart (3.2 s) | 5.52 s | 5.26 s | 1.05x | 1.73 → 1.64 |
| sanoTTS heart-nano (3.1 s) | 2.90 s | 2.80 s | 1.03x | 0.93 → 0.90 |
| Citrinet F32 / Q8_0, Moonshine F32 / Q8_0, conformer, NER | | | 0.99-1.02x | unchanged |

The six models without such convs give bit-identical output tensors. The five TTS waveforms move by
0.6-3.9e-5 of their peak, and whisper-small transcribes each one identically in both arms. Every run is
deterministic across rounds.

**Against the projection.** amy and de051 land near their P4.32b share (projected ~3.4 and ~1.14). VITS
falls short: projected ~7.7, measured 10.9. Its convolution seconds now run at ~0.7 GOP/s, not 1.16. Its
wide-input stages (128-768 channels, a few hundred steps) shrink TB and re-stream large weights, and its
OC = 1 post-conv uses a quarter of each tile. Those shapes are the open part of this item.

## P4.32b's wide and dilated shapes, and P4.32c (the GEMV), measured 2026-10-15

**Why VITS fell short.** A node-level board profile showed the dilated half of every resblock at half
rate: amy 48ch k7 ran 0.53 GOP/s against 0.89 for its d = 1 twin, and VITS 64ch k7 0.40 against 0.80. The
wide stages ran 0.35-0.58. The window `[IC][TB + (k-1)d]` plus one quad's weights came to 20-55 KB against
the 16 KB L1. **Fix:** the input channels go in chunks sized so a chunk's window rows and weights fit
12 KB. Chunk 0 starts from the bias, and each later chunk continues from the partial sums in the output
tile (a NULL bias in the kernel). A ragged block runs whole tiles into the staging buffer and flushes only
its steps, because an overlapping tile would add a chunk twice. TB went to 128 on a board sweep (TB
64/96/128/192/256 = 1.12/1.15/1.16-1.17/1.15/1.12 GOP/s over the models' shapes; the L1 budget is flat
from 8 to 16 KB). On `bench50 model` (the profile's slow shapes) the set goes 8.73 → 5.56 s (0.75 → 1.17
GOP/s), and bench49's set 1.16 → 1.19. **The arithmetic per output is unchanged**, so the outputs are
bit-identical to the first P4.32b.

**P4.32c.** `llamafile_sgemm` declines n = 1, so on this core every one-column F32 `mul_mat` ran
`ggml_vec_dot_f32`'s scalar branch: float products summed in double, one chain per output, 91 MB/s.
`ggml-0033` runs four rows a pass with exactly that arithmetic, so the output is **bit-identical**.
`scripts/bench51.cpp`, with tensors in a CPU backend buffer as in the engine: 274-275 MB/s on Moonshine's
shapes (3.0x). Gate 2 holds against the equal-arithmetic microbench (274), not against bench48's float sum
(316). A float sum would be ~15% faster on the GEMV, about 2% of a Moonshine run, and every decoder on this
core would need a numerics gate for it. A PLD per row and line cost 10%, and 8 rows a pass changed nothing.
**A placement effect, recorded so it is not mistaken for the kernel:** the same function and bytes ran 216
MB/s from a 32-bit `ggml_init` context allocation (4-byte tensor alignment, page offset 340) and 274 from an
8-byte-aligned one.

**Gates 3 and 4** (the P4.32b wheel against the new one, `armv6_runfast_gate.sh` with `PY0`/`PY1`, 13
models, two rounds ABBA): **every output bit-identical**, and every run deterministic.

| model | P4.32b | now | speed-up | RTF |
|---|---|---|---|---|
| VITS (3.2 s) | 34.6 s | 25.3 s | **1.37x** | 10.8 → 7.91 |
| Silero VAD (jfk) | 4.39 s | 2.99 s | **1.47x** | 0.40 → 0.27 |
| Moonshine tiny F32 (jfk) | 61.9 s | 47.0 s | **1.32x** | 5.62 → 4.27 |
| piperlite amy (5.9 s) | 22.9 s | 18.6 s | **1.23x** | 3.89 → 3.16 |
| piperlite de051 (4.4 s) | 4.91 s | 4.56 s | 1.08x | 1.12 → 1.04 |
| sanoTTS heart (3.2 s) | 5.19 s | 5.02 s | 1.04x | 1.62 → 1.57 |
| the other seven | | | 0.99-1.02x | unchanged |

Against the session's starting point (ggml-0031 only), the vocoders are now 1.84-2.39x: VITS 60.3 → 25.3 s,
amy 39.8 → 18.6 s, de051 8.41 → 4.56 s.

## P4.32d, measured 2026-10-15: the GEMM, 1x1 and strided convolutions (`ggml-0034`)

**One kernel, three new users.** The k = 1 conv kernel is already a 4x4 GEMM tile: per input row, a
vector of four positions times four scalars. (1) **1x1 convolutions** go to the conv path, with TB = 32,
because a tap-less window row lets a chunk take many more channels (Citrinet's shapes, TB 32/64/128:
0.770/0.768/0.652 GOP/s). (2) **F32 `mul_mat` with n >= 24 and M % 4 == 0**, graph nodes only, runs the same
kernel. The window is a transposed block of the weights, `[K chunk][32 rows]`, and the activation columns
are packed `[N/4][K][4]` as the scalars. Each accumulator vector is a contiguous piece of a `dst` row,
written in place. The plan sizes the work buffer. *Graph nodes only* because the convolutions' im2col
fallback calls `mul_mat` with its own patches in `wdata`, which the packing overwrote; the conv sweep caught
that on 2 threads. (3) **Any stride, by phases**: stride S over IC channels is stride 1 over IC * S virtual
channels with ceil(k / S) taps. The window copy gathers the phases, so the kernels need no change.
Separately, **odd-k kernels take input channels in pairs**, because the per-channel buffer copy cost as many
FMAC cycles as the `vmla`s at k = 1. That change is bit-identical, and the vocoder shapes gained 5-7%.

**Gate 2.** `bench51 gemm`: Moonshine's prefill GEMMs 0.29-0.38 → 0.76-0.79 GOP/s against tinyBLAS (2.1-2.6x),
and the conformer FFN 0.54 → 0.76. That is above the ADR-077 bench tile's 0.72. The n threshold comes from a
sweep: n = 16 0.47 vs 0.49, n = 24 0.56 vs 0.49. `bench50 citrinet`: 0.51 → 0.77. Moonshine's stem k7 s3
went 0.235 → 0.82, and k3 s2 0.39 → 0.66.

**Gates 3 and 4** (wheel A/B against `ggml-0032` + `0033`, 13 models, two rounds ABBA; the stride half on
Moonshine as a second A/B):

| model | before | after | speed-up | RTF |
|---|---|---|---|---|
| Citrinet-256-ls Q8_0 (jfk 11 s) | 29.0 s | 16.9 s | **1.72x** | 2.64 → 1.54 |
| Citrinet-256-ls F32 | 20.0 s | 16.2 s | 1.23x | 1.82 → 1.48 |
| Moonshine tiny F32 (GEMM, then stem) | 47.0 s | 39.3 → 32.8 s | **1.43x** | 4.27 → 2.98 |
| Moonshine tiny Q8_0 (stem) | 65.7 s | 58.9 s | 1.12x | 5.98 → 5.35 |
| conformer-ctc-small | 11.0 s | 9.33 s | 1.18x | 3.68 → 3.11 |
| sanoTTS heart | 5.04 s | 4.33 s | 1.16x | 1.57 → 1.35 |
| WakeHuBERT / Silero VAD | 3.12 / 3.00 s | 2.80 / 2.78 s | 1.12x / 1.08x | |
| VITS / amy / de051 | 25.4 / 18.7 / 4.58 s | 23.4 / 17.7 / 4.40 s | 1.04-1.08x | de051 1.04 → **1.00** |

**Numerics.** These changes are not bit-identical. Every transcript is unchanged, and so is every CTC
argmax frame (Citrinet both cards, the conformer). The TTS waveforms move by 1e-6 to 3e-5 of their peak, and
the ASR oracle hears the same words. **On a dynamically quantized card the tensor oracle's floor is the
card's own sensitivity, not 1e-6.** Moonshine Q8_0's encoder output moved 4.5e-2 between the arms, and on
x86, where no ARMv6 code runs, scaling the input by (1 + 1e-7) moves it 4.9e-2. Every board arm, including
the one from before this ADR's kernels, sits 6e-2 from x86. The conformer behaves the same way: 2.3e-3 from a
1e-7 input perturbation, 6.4e-3 between arms, and identical argmax. The F32 cards stay within 4e-6 of x86.
**And the first wheel A/B's tensor oracle compared a wheel with itself**, because one `libggml-cpu.so` beside
the probe served both arms. `armv6_runfast_gate.sh` now gives each arm's probe its own backend
(`GGML_BACKEND_PATH`), and the gate-3 claims above rest on that rerun.

**Since the session's start** (ggml-0031 only), on the Pi Zero W: VITS 60.3 → 23.4 s (2.58x), amy 39.8 →
17.7 s (2.25x), de051 8.41 → 4.40 s (1.91x, RTF 1.00), Moonshine tiny F32 61.9 → 32.8 s (1.89x), Silero VAD
4.39 → 2.78 s, Citrinet Q8_0 29.0 → 16.9 s, conformer 11.0 → 9.33 s. Left of the F32 family: P4.32e
(depthwise, Citrinet's 4.1 s), P4.32f (epilogue fusion), P4.32g (an FFT for the DFT-basis convolutions,
heart's 1.9 s of 4.3).

## P4.32c revised 2026-10-15: the GEMV sums in float

**User decision 2026-10-15: take the float sum** that P4.32c had declined for bit-identity. `ggml-0033` now
runs four rows a pass, each a float sum of float products in k order. That is what every SIMD build of
`vec_dot_f32` already does, in another order. Eight rows a pass was measured too and lost: eight row streams
thrash the 4-way 16 KB L1 at most strides. `bench51` (MB/s of weights, Pi Zero W, one thread, RunFast), at
row strides of 1152 B (Moonshine) / 1 KB / 2 KB (the LSTM-sized shape) / 3 KB / 4 KB:

| GEMV | 1152 B | 1 KB | 2 KB | 3 KB | 4 KB |
|---|---|---|---|---|---|
| ggml, `vec_dot_f32` | 92 | 92 | 92 | 93 | 92 |
| double sum, 4 rows (the bit-identical first version) | 274 | | 236 | | |
| float sum, 8 rows | 317 | 135 | 147 | 124 | 50 |
| **float sum, 4 rows** | **318-325** | **294-298** | **270** | **267-269** | **108-119** |

`tests/ci/test_mul_mat_f32_sweep.cpp` now holds the ARMv6 GEMV to its own contract's bits (a float sum in k
order); with the GEMV switched off, that check fails on every GEMV shape, so it can tell the two apart.

**Gates 3 and 4** (one wheel, `VAR=GGML_CPU_ARMV6_GEMV`, so arm 0 is stock ggml's double sum, whose bits are
the first version's; 13 models). The board rebooted during round 2, so the timings are round 1 plus the
seven round-2 runs that finished. **Twelve of the 13 outputs are byte-identical between the arms**,
Moonshine's transcripts (the token ids) included. **Silero VAD differs, by at
most 4.8e-7 in a speech probability, and no frame changes its decision** at 0.5 or 0.35. Against the double-sum
kernel (P4.32d's A/B), Moonshine tiny F32 goes 32.8 → 31.7-32.0 s and Silero 2.78 → 2.70 s, the ~2-3% the
GEMV's share predicted. Against stock ggml, those are 47.4 → 31.7-32.0 s and 4.15 → 2.70 s.
**Tensor oracle** (per-arm backends): Moonshine's encoder (both cards), the conformer and NER are bit-identical;
Citrinet moves 1.3e-6 (F32) and 1.2e-7 (Q8_0) of peak, through its squeeze-excitation FCs, which are one-column
GEMVs. Its CTC ids are unchanged.

## P4.32e, measured 2026-10-15: the depthwise kernel (`ggml-0035`)

**The kernel.** `scripts/gen_armv6_dw_f32.py` generates `arch/arm/armv6-dw-f32.inc`: one channel, stride
1, dilation 1, K >= 2, eight outputs a block. A depthwise conv has no input channels to reuse a weight
across, so the 4x4 tile does not apply; what it reuses is the input window, across taps. Bank 1 holds a
**ring** of eight inputs, and a vector operand starting at slot `k % 8` wraps around the bank, which
is exactly `x[t + k .. t + k + 3]`: the unroll renames the window and never moves it. Bank 0 holds eight
taps' weights as scalars, and a tap is two vector `vmla` and two scalar loads. `armv6_shortvec.lint`
rejected that wrap as never intended. A line now opts in with a marker, and only for its Sn; a wrapping
destination still raises.

**Why each output is two sums.** The first version kept one sum per output, in k order, so it gave
ggml's sweep bit for bit. It ran 0.333 GMAC/s on the board, against 0.289 for `bench43`'s C arm. With its
loads removed it still took 14 cycles a tap, against 8 for the FMAC: a vector `vmla` waits for the whole
previous one on its destination. Four chains ran a `vmla` per ~5 cycles, but the ring leaves room for
only two accumulator vectors. So the odd taps go to a second pair in bank 3, and the two sums are added
at the end. That costs bit-identity with ggml's order: the contract is `(w0 x0 + w2 x2 + ...) + (w1 x1 +
w3 x3 + ...)`, 3.2e-7 of peak from ggml's order on the jfk schedule.

**Gate 2** (`scripts/bench52.cpp`: `bench43`'s jfk schedule, the 104 stride-1 calls, 171.8 MMAC, Pi
Zero W, one thread, RunFast; every arm checked bit for bit against its contract):

| arm | ms | GMAC/s |
|---|---|---|
| ggml as shipped (`ggml_vec_mad_f32` per tap) | 4,136-4,159 | 0.041 |
| `bench43`'s C arm (eight outputs, the window renamed) | 592-597 | 0.289 |
| the kernel, one sum per output | 516 | 0.333 |
| two sums | 360 | 0.477 |
| two sums, next block's loads under the final adds; PLD of the next row (shipped) | 341-355 | 0.48-0.50 |
| the same without the PLD | 455-473 | 0.37 |
| the engine's per-channel work around it (row and weights copied, padded) | 351-357 | 0.48-0.49 |
| inside ggml (`ggml_conv_2d_dw_direct`, tensors in a CPU backend buffer) | 376-379 | 0.453-0.457 |

**Gate 2 is missed by about a point:** ggml runs 5.6-6.6% behind the bench arm that does the same work
per channel. One part was a 64-bit `i % channels` per row, which is a library call on this core. A copy of
the path's code ran 368 ms, and 357-361 with the channel counted instead. The patch now counts it, which
took ggml from 380-384 ms to 376-379. ggml's own cost per graph compute is 0.027 ms (2.8 ms over the 104
calls). Placing ggml's tensors on the bench's own arrays changes nothing (373-376 ms). The remaining ~4%
is not attributed. The path stays on because gate 4 decides that, and it delivered (below). Measured out:
two prefetch lines a block (0.45 GMAC/s) and a PLD on the output lines (0.39; the stores stream without
allocating).

**Correctness.** `tests/ci/test_conv_1d_dw_sweep.cpp`: every K % 8 on both sides of 8, Citrinet's and
the conformer's shapes, a valid conv, a wide pad, and the four shapes the kernel declines (K = 1, fewer
than 8 outputs, stride 2, dilation 2). Each runs bare and as the fused PAD -> conv -> bias -> RELU chain
with the output over its input, on 1 and 2 threads, against a double reference and, on ARMv6, the
contract's bits: 400/400 in qemu and on the board, 92 runs held to the bits. **Both sabotages go red.** A
skipped ring load fails 140 checks. With the path off (`GGML_CPU_ARMV6_DW=0`), ggml's order fails the
bit check on 88 runs at 7.5e-8, so the check can tell the two orders apart. The x86 `ci` label passes 121/121.

**Gates 3 and 4** (one wheel, `VAR=GGML_CPU_ARMV6_DW`, so arm 0 is ggml's sweep; the seven models with
depthwise convolutions; two rounds ABBA; one inference per launch; per-arm backends for the tensor oracle):

| model | off | on | speed-up | output |
|---|---|---|---|---|
| Citrinet-256-ls F32 (jfk 11 s) | 17.9 s | 14.0 s | **1.28x** | CTC ids identical; logits 1.4e-6 of peak, no argmax frame changes |
| Citrinet-256-ls Q8_0 | 18.1 s | 14.3 s | **1.27x** | CTC ids identical; logits 1.0e-6 |
| conformer-ctc-small | 10.1 s | 10.1 s | 1.01x | same text; logits 1.5e-3, inside the card's own floor (2.3e-3, P4.32d), no argmax change |
| sanoTTS heart / heart-nano | 4.97 / 3.11 s | 4.91 / 3.06 s | 1.01-1.02x | 1.2e-6 / 1.5e-5 of peak; whisper-small hears the same sentence in both arms |
| WakeHuBERT tiny | 2.77 s | 2.72 s | 1.02x | 4.2e-7 (its dilated convs stay on ggml's path) |
| VITS | 26.1 s | 26.3 s | 0.99x | identical: none of its depthwise convs takes the path |

Every run is deterministic across rounds. Citrinet's saving, 3.8-3.9 s, matches the depthwise share the
profile gave it (4.1 s at 0.043 GMAC/s, now ~0.4 s). **Both arms ran 7-10% slower in seconds than P4.32d's
on the same board** (Citrinet F32 16.2 s then, 17.9 s for this off arm). The board was not throttled
(42.8 °C, 1 GHz), and the ratios come from paired runs. The first launch (Citrinet F32 off, round 1:
21.4 s) overlapped an ssh login and is left out of the ratio. Citrinet-256-ls on the Pi Zero W is now RTF
~1.27 (F32) and ~1.30 (Q8_0), against 1.48 and 1.54 after P4.32d.


## P4.32f, measured 2026-10-15: the convolution epilogue (`ggml-0036`) and the exporter's topology CSE

**The budget was mostly spent already.** The decision table's "1.3-3.7 s in seven models" came from
ADR-077's node-by-node profile, where no CPU fusion fires ([Retro-073](../retros/retro-073-the-profiler-cannot-see-a-fusion.md)).
A fusion log of the real path (an x86 build of the dev tree that prints, per node, whether it ran fused
and why a fusion declined; the detectors are ISA-independent) joined with the board's per-node seconds:

| model | elementwise s still paid | already fused (ggml-0005/0007/0022) |
|---|---|---|
| Citrinet-256-ls F32 / Q8_0 | **2.11 / 2.10** | 0.44 |
| VITS / amy / de051 | 2.33 / 1.59 / 0.60 | 3.52 / 4.50 / 1.65 |
| Moonshine tiny F32 | 2.61 (0.79 of it is the GELU itself) | 0.01 |
| conformer / heart / Silero / NER / WakeHuBERT | 1.44 / 0.47 / 0.37 / 0.15 / 0.11 | ≤ 0.24 |

Citrinet's 2.11 s is the one near a real-time line. Each sub-block ends `MUL(RELU(ADD(conv, bias)), mask)`
and each depthwise conv is followed by `MUL(·, mask)`: RELU 0.45 s, the two mask MULs 0.44 + 0.43 s, and
53 of 129 bias ADDs (0.21 s) that ggml-0005/0007 declined because the allocator put the ADD's result
4384-4416 bytes ahead of the convolution's input. The work split three ways (f1 Citrinet's epilogue, f2 the
vocoders' leftovers, f3 a GEMM bias epilogue); **the user took f1**.

**Why the masks needed the exporter.** NeMo rebuilds `arange(T) < length` before every use, and ggml
expands a node's sources depth first, so each rebuilt mask lands between the RELU and its MUL and the MUL
is never the next node. The 239 RANGE_1D chains are 4 distinct ones, but only on the topology, where T is
an expression over the root axis; in MIL each ends at a different tensor's `shape`. The user chose a
generic pass in the exporter over one in the engine's graph build (re-export needed, but the runtime
stays lean): `loom_exporter/topology_cse.py` merges whitelisted pure nodes with equal op, inputs and
attributes and never removes a declared output (Epic-02). Citrinet: 2,410 nodes → 1,247. A fresh
export with the pass off is byte-identical to the GGUF the board had.

**`ggml-0036`.** `ggml_cpu_conv_2d_fusion` takes a RELU and then a per-position MUL after the bias (and
residual); the depthwise fusion takes the MUL. A result that partially overlaps the input is computed
into the convolution's own tensor and carried to the output in one pass that adds bias, residual, RELU
and mask, which is bit-identical to the unfused nodes. `GGML_CPU_CONV_EPILOGUE`, on by default on ARMv6 only.
The first ARMv6 epilogue, a per-element C function, cost as much as the passes it removed (42 ns an
element after a 1x1 conv, 54 after a depthwise one, against ~48 for a pass): the RELU's `vcmpe` + `vmrs`
drained the VFP pipeline every element. The shipped one tests the RELU on the integer bits, keep iff
`lo <= u <= 0x7f800000` (lo = 0x00800000 under flush-to-zero, 1 in IEEE mode), exact against the VFP
compare over 17.1M bit patterns in both FPSCR modes on the board (a sabotaged lo: 33,425 differ), and
unrolls the add and multiply by four. The depthwise row goes through an L1 scratch row first: the
ARM1176 does not allocate on a store, and reading back a just-stored row cost 43 ns an element against
10 cached (the same pass run twice).

`scripts/bench53.cpp` (Pi Zero W, one thread, RunFast, 256 channels, ms over the conv+bias or bare-dw floor):

| T | RELU + mask, unfused | first epilogue | shipped | dw mask, unfused | first | shipped |
|---|---|---|---|---|---|---|
| 551 | +13.5-17 | +6.0 | +3.7 | +7.0 | +7.6 | **+1.55** |
| 276 | +7.4-8.8 | +4.0 | +2.8 | +3.9 | +4.0 | +0.8 |
| 138 | +4.2-4.4 | +1.9 | +1.2 | +2.1 | +1.8 | +0.45 |

**Correctness.** `tests/ci/test_conv_epilogue_fusion.cpp`, registered with the epilogue off and on: 1x1,
k5 d2, OC % 4 != 0, a residual, both partial-overlap offsets, three depthwise shapes; a double reference,
a poisoned RELU (and depthwise) result that must survive exactly when the epilogue is on, and the two runs
compared byte for byte. x86 and ARMv6 (qemu) pass. Five sabotages each go red (the mask skipped past
t = 500, the RELU not matched, the depthwise mask misindexed, the overlap pass's bias dropped, one row of
each ARMv6 quad skipped). The x86 `ci` label passes 123/123; the exporter's `tests/ci` 1,246 passed.

**Gates 3 and 4** (wheel with ggml-0036, `VAR=GGML_CPU_CONV_EPILOGUE`; on x86 only Citrinet, VITS,
WakeHuBERT and Silero fuse anything new, so those are the rows). Every output hash is identical between
the arms, and Citrinet's logits through the tensor oracle are byte-identical off and on, in both
precisions and both GGUFs. The CSE'd graph differs from the old one by 1.3e-6 of peak (with the masks
gone from between them, more conv + bias pairs take ggml-0005's accumulator-start bias); CTC ids identical.
Four rounds (round 1 off first, then on first), one inference per launch, 45 s settle:

| Citrinet-256-ls F32, jfk 11 s | off | on |
|---|---|---|
| old GGUF | 14.09 s | 13.92 s (1.012x) |
| CSE'd GGUF | 13.68 s | 13.21 s (1.036x) |

**f1 end to end: 14.09 → 13.21 s (1.067x), RTF 1.28 → 1.20** (the CSE 0.42 s, the epilogue 0.47 s). Q8_0, two
rounds, two gates: 14.34-14.46 → 13.52-13.57 s (1.06-1.07x). VITS 26.24 → 26.33 s over three clean pairs
(the fourth "on" launch, 33.2 s, overlapped an ssh login); WakeHuBERT 1.03x; Silero 0.98-1.00x.

**Against the projection.** 0.89 s of the ~1.5 s the node-level seconds promised. The shipped epilogue
did not move the end-to-end number from the first epilogue's gate (0.88 s, two rounds), although bench53
prices its depthwise mask at a fifth of the first one's. Not attributed; one candidate is that in the
model's graph the depthwise output is allocated over its own just-read input row, already cached,
which bench53's separate output tensor is not. **Left for Citrinet's real-time line:** the DFT-basis
convolution (1.19 s, P4.32g). **Left of P4.32f:** f2, the vocoders' unfused leftovers (resblock sum and
its 1/3, the LEAKY_RELU fed from a CONT, the transposed conv's bias: VITS 2.3 s, amy 1.6, de051 0.6), and
f3, a bias epilogue on the ARMv6 GEMM (Moonshine 0.6 s, conformer). **The CSE reaches users only through a
re-export**: Citrinet's published GGUFs are unchanged until republished, and every family's next export
merges whatever its topology repeats, so the exporter's snapshot gate will show topology diffs.

## P4.32f f2, measured 2026-10-16: the elementwise passes (`ggml-0037`) and the transposed convolution (`ggml-0038`)

**What f2 was, and what it turned out to be.** f2 was scoped as the vocoders' unfused leftovers (VITS 2.3 s,
amy 1.6, de051 0.6): the resblock sum and its 1/3, the LEAKY_RELU fed from a crop's CONT, the transposed
convolution's bias. Two findings changed it. **ggml's elementwise loops are compute-bound on the VFP11, not
pass-bound:** 50-64 ns an element against 12 for `memcpy`, so the leftovers were slow passes more than
extra ones, and every model pays the same rate on whatever it does not fuse (Moonshine 1.8 s, conformer 1.4).
**The transposed convolutions never got an ARMv6 kernel.** The decision table above lists them under P4.32b,
but `ggml-0032` takes CONV_2D only, and ggml-0009's GEMM reaches tinyBLAS through `ggml_call_mul_mat_ldc`,
which `ggml-0034` declines (`dst->op != MUL_MAT`). They ran at ~180 MMAC/s: amy 3.5 s, VITS 3.0, de051 0.74,
larger than the leftovers. **The user took all three, in order:** the elementwise kernels, then the
transposed convolution, then whatever chain fusion is still worth it (step 3, open).

**`ggml-0037`: ADD, SUB, MUL, SCALE, RELU and LEAKY_RELU at memory speed, bit-identical.** Stock, every
loop is one load, one VFP op and a store that waits on it, with no prefetch, and a `[1, C]` bias or a
`[1, 1]` scale calls ggml's row loop once per ELEMENT. The RELUs pay a `vcmpe` + `vmrs` per element.
`armv6-eltwise.h` runs eight floats a step through `vldmia`/`vstmia` with eight independent VFP ops and a
`pld` per input line, has a scalar form for a one-float src1 row, and does the RELUs' compares on the
integer bits (P4.32f f1's test, and LEAKY_RELU's own `max + slope * min` arithmetic, so NaN and -0 come out
as stock). The input is first brought to a 32-byte line: a `vldmia` that straddles two lines nearly
doubles a cold stream on the ARM1176. `scripts/bench55.cpp` (Pi Zero W, one thread, a cold 12.5 MB
tensor, ns an element, the same in RunFast and IEEE mode, every arm bit-identical):

| op | stock | `ggml-0037` |
|---|---|---|
| ADD, same shape | 68.3 | 36.4 |
| ADD of a `[1, C]` bias, MUL by a `[1, 1]` scale | 43.3 | 11.6 (`memcpy` 11.9) |
| LEAKY_RELU | 63.8 | 23.0 (19 with the data in L1: compute-bound) |

What a stream costs on this core (`scripts/bench55_streams.cpp`): a cold read 10-13 ns an element, a write
2.6 (the store buffer), `vldmia` and `ldm` alike; a copy from a buffer 8 bytes off a line 21.0 against 11.1
aligned; one, two or three lines a step 11.6 / 11.4 / 14.1; and the `pld` distance does not matter, 96 to
1024 bytes within 0.3 ns. **Correctness**: `tests/ci/test_eltwise_armv6.cpp`, registered off and on: every
op and broadcast form (tails, `[1, C]`, `[1, 1]`, `[T, 1]`, a repeated src1 row, in place) exact against the
test's own float operations, then all of them again with ±0, denormals, inf and NaN spliced in and the two
runs compared byte for byte. qemu and x86 pass; four sabotages go red (a NaN kept by the LEAKY, SUB's
scalar form mapped to ADD, the eight-float tail cut to four, the RELU's FZ bound ignored).

**Gate (wheel v10, `VAR=GGML_CPU_ARMV6_ELTWISE`, 12 models, two ABBA rounds, 20 s settle): every output
byte-identical between the arms.** Means of the two launches per arm:

| model | off | on | |
|---|---|---|---|
| amy / de051 / VITS | 19.29 / 4.66 / 25.88 s | 18.45 / 4.37 / 25.21 | 1.045 / 1.067 / 1.027x |
| Moonshine tiny F32 / conformer / NER | 34.50 / 9.95 / 6.14 | 33.70 / 9.45 / 5.84 | 1.024 / 1.053 / 1.051x |
| Citrinet-256-ls F32 (CSE) / Q8_0 | 13.20 / 14.27 | 12.91 / 13.39 | 1.022 / 1.065x |
| heart / heart-nano (FFT exports) | 2.76 / 1.01 | 2.65 / 0.99 | 1.04 / 1.02x |
| Silero VAD / WakeHuBERT | 2.91 / 2.58 | 2.90 / 2.60 | 1.00x |

**`ggml-0038`: CONV_TRANSPOSE_1D as polyphase direct convolutions on the `ggml-0032` tiles.** With
K = J·s, output phase p is an ordinary J-tap convolution over the input padded by J - 1 zeros:
`y[oc][q·s + p] = Σ_j Σ_ic w[ic][oc][p + (J-1-j)·s] · xpad[ic][q + j]`. Per block of phase positions and
chunk of input channels (the 0032 blocking), the window is copied once, each quad of output channels runs
its s phases through the J-tap tile into a per-thread scratch, and the last chunk interleaves into dst: each
output is written once, and there is no overlap-add. The zoo's upsampling stages are all J = 2 (K = 2s).
A considered alternative, ggml-0009's GEMM moved onto `ggml-0034`, keeps the overlap-add, which adds K/s
partial results into every output. `scripts/bench56.cpp`, amy's stages against ggml-0009 + the bias ADD
(what the model pays today), ms:

| stage | ggml-0009 + bias | `ggml-0038` (TB 64; TB 32 is 3-7% slower) | |
|---|---|---|---|
| 192 → 96, s 8, k 16 | 658 | 367 | 1.79x |
| 96 → 48, s 8, k 16 | 1379 | 751 | 1.84x |
| 48 → 24, s 4, k 8 | 1870 | 763 | 2.45x |

Not bit-identical: the tile's summation order, 5e-7 to 1e-6 of the peak against an f64 reference (ggml-0009
1.4e-7 to 2.5e-7). `tests/ci/test_conv_transpose_1d_sweep.cpp` checks every output against f64 over J = 1-4,
OC % 4 != 0, inputs of 1 and 3 positions, a chunked 300-channel input, several blocks with a ragged last one, a
K that does not divide by s (the fallback), one and three threads, with the driver on and off. Three
sabotages go red (the taps reversed, the last block's last position not interleaved, the padding off by one).
The bias is still the graph's ADD; the tile could seed it for free, which is step 3's.

**Gate (wheel v11, `VAR=GGML_CPU_ARMV6_CONVT`, two ABBA rounds):** amy 18.39 → 16.22 s (**1.134x**), de051
4.42 → 3.86 s (**1.145x**, RTF 0.88), VITS 25.12 → 23.32 s (**1.077x**). The waveforms move by at most 1.6e-6 of
the peak (one int16 LSB), each arm is deterministic across rounds, and whisper-small transcribes both arms
of all three identically. heart and heart-nano, which no longer have a transposed convolution, are
byte-identical.

**Left of f2 (step 3)**, at the new rates, on amy's last stage (3.1M elements): the two resblock-sum ADDs
~225 ms, the LEAKY after the crop ~72, the 1/3 and the LEAKY after it ~110, the crop's CONT ~44, the bias
ADD ~36. About 0.65 s over amy's three stages; the transposed convolution can take the bias, the 1/3 and the
LEAKY in its window copy and accumulator start.
