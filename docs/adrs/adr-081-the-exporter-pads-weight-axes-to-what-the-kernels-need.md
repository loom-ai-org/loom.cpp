---
type: adr
status: proposed
date: 2026-10-22
domain: performance
tags: [exporter, padding, alignment, q8_0_pc, quantization, armv6, aarch64, pi-zero, pi-4, vocetta2]
---

# ADR-081: The Exporter Pads Weight Axes to What the Kernels Need

P4.34 in the [hub](../backlog/active-index.md#exporter--mil-compiler). Serves [Epic-05](../epics/epic-05-edge-performance.md). Builds on
[ADR-003](adr-003-per-model-complexity-in-the-exporter.md) (per-model complexity in the exporter) and the
exporter's conv fold (P4.13, `_fold_conv_kernels_for_quantization`), which is the precedent for a pack-time
rewrite of weights and topology together. Extends the reach of `Q8_0_PC`
([ADR-076](adr-076-quantized-gemms-reach-xnnpacks-microkernels-through-a-ggml-buffer-type.md),
[ADR-079 P4.33](adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md)). **Proposed:**
accepted once the pass ships and passes the gates below.

## Context

Vocetta2 (`loom-exporter` #63) exported with `--quantize Q8_0_PC` came out **byte-identical** to F32 for
the 276k and the 1m and only 50% quantized for the 500k: ggml lays Q8_0 blocks of 32 along ne[0], and none
of their contraction widths is a multiple of 32 (decoder 62 / 67 / 130, pointwise hidden 186 / 197 / 337,
embed conv 100 x 7, acoustic 31 x 5 and 72 x 5). The exporter leaves such a tensor F32
(`exporter.py`, `aligned = shape[-1] % block_size == 0`).

Zero-padding the axis to the next multiple is exact in real arithmetic, so a scratch prototype
(`~/.claude/tmp/vocetta2/pad/pad_export.py`: zero-padded torch modules through the real `main_export`)
measured what it buys, on the released rc18 wheel, 5 paired rounds per arm. Three things came out of it,
and only the first was the question that was asked.

**1. Padded to 32, Q8 pays on the Pi Zero and never on the Pi 4.** Pi Zero, one 11.2 s utterance: 276k
4117 -> 3800 ms, 500k 4500 -> 4198 (the published unpadded 500k Q8 is 4514: nothing it quantizes is on the
hot path), 1m 7237 -> 5266 (1.377x). Pi 4: every Q8 arm ties or loses to F32 (0.86-1.01x at one thread).

**2. Padding pays at F32, on both boards, and to 4, not 32.** The axis a kernel wants aligned is not the
same on every board, and the source says so:

| kernel | predicate | file |
|---|---|---|
| tinyBLAS F32 GEMM, aarch64 (NEON `KN = 4`) | a ragged `k` takes the scalar k-tail | `llamafile/sgemm.cpp`, `tinyBLAS::matmul` |
| tinyBLAS F32 GEMM, x86 (AVX2 `KN = 8`, AVX-512 16) | the same, with its own tail | same |
| ARMv6 F32 GEMM (`ggml-0034`) | `src0->ne[1] % 4 == 0`, else ggml's generic kernel | `ops.cpp`, `ggml_armv6_mm_ok` |
| direct 1-D conv, every ISA | `OC % GGML_CONV1D_OCB (4) == 0`, else im2col | `ops.cpp`, `ggml_conv_1d_direct_ok` |
| Q8_0 / Q8_0_PC weight | `ne[0] % 32 == 0`, else the exporter leaves it F32 | `exporter.py`, `pack_weights` |
| `CPU_ARMV6_I8` (Q8_0_PC on ARMv6) | none on N: it pads rows itself (`padded_rows`) | `armv6-i8.cpp` |

So F32 wants the contraction K (ne[0]) a multiple of 4 on aarch64 and the output count N (ne[1], or OC)
a multiple of 4 on ARMv6. Padding K alone to 4 gave the Pi 4 1.13 / 1.09 / 1.15x at one thread (1.04 /
1.03 / 1.05x at four) and the Pi Zero 1.05 / 1.04 / 1.19x; padding N as well took the Pi Zero to
**1.105 / 1.102 / 1.344x** and left the Pi 4 where it was. Padding to 32 instead of 4 bought the F32 arms
nothing more and cost the extra MACs (the 1m's decoder 130 -> 160 vs 132).

**3. With F32 padded properly, Q8's lead on the Pi Zero is gone -- for Vocetta2.** F32 padded K+N to 4
runs 3766 / 4110 / 5355 ms; the best padded Q8 runs 3754 / 4189 / 5260, inside each other's spread. Most
of finding 1 was alignment, not int8: `CPU_ARMV6_I8` pads its rows to 4 internally, so quantizing a weight
quietly fixed the N alignment its F32 form was missing. That is a statement about models this small
(0.3-1 M parameters, compute in 62-337-wide matmuls); the models where `Q8_0_PC` was measured to pay --
Pocket-TTS, Moonshine tiny, Citrinet -- are larger and stay as measured. The Q8 half of the pass still
matters for them and for the zoo's Q8 stragglers below. Nor does Q8 buy Vocetta2 memory: peak RSS on the
Pi Zero for one load + one 11.2 s synthesis is 54-56 MB for every 1m arm (F32 54, F32 K+N 55, Q8 linears
56, Q8 all eligible 52) against 14 MB for `import loom` alone; the weights are 1-4 MB of it. The memory
case for Q8 on a 512 MB board is real where weights dominate -- Pocket-TTS's 405 MB F32 does not fit
there, its 130 MB Q8 does -- and is a smaller download everywhere.

**One axis does not pad for free.** The decoder width C feeds LayerNorms, which normalise over C: zero
channels change the mean and variance. So the activation stays C wide and is padded (`PAD_1D`) right
before the two matmuls that read it, and an output padded past C is sliced back (`VIEW`) before the norm.
The pointwise hidden width, the mel and the acoustic channels pad exactly through zero rows, zero biases
and activations with f(0) = 0.

**The zoo.** A header scan of the staged Hub GGUFs (`zoo/scan.py`) finds weight bytes the F32 rule would
pad in most families, in small amounts, and large shares in the small TTS models and the CTC/vocab heads:
Kokoro 22.8%, StyleTTS2 18.2%, sanoTTS heart-nano 100% and heart 9.8%, WakeHuBERT 20.5%, whisper-small
16.6% (its vocab head), SenseVoice 5.5%. Under the Q8 rule: conformer-ctc-small 64.7% (its K = 176),
sanoTTS 0.51m 60.5%, Moonshine streaming small 40%. Eleven GGUFs, the LLMs among them, are fully aligned
and would not change.

## Options

1. **Pad in the exporter** (this ADR). Exact, no engine change, files run on every released runtime, one
   implementation for every backend; costs zeros in the file and a re-export per model.
2. **Give the kernels tails.** An N tail for the ARMv6 GEMM and the direct conv, a vector k-tail for NEON.
   Helps every model without a re-export, but it is one ggml patch per kernel and ISA, and it cannot help
   Q8: a block of 32 is the format, not a kernel choice. Not exclusive with option 1; kept as a follow-up
   for the F32 half.
3. **Pad at load time in the engine.** The engine would rewrite the graph (insert pads, slice outputs) on
   every load. Rejected: [per-model complexity belongs in the exporter](adr-003-per-model-complexity-in-the-exporter.md) -- this is
   exactly the per-model graph surgery the exporter exists to do once.

## Decision

**One pass in the exporter, `_pad_weight_axes`, run in `pack_weights` before the conv fold**, over every
topology of the export (per phase for a multi-phase export, which is consistent for the same reason the
fold is: phase weights are namespaced, so no other phase reads them).

**Targets, per weight read as the first input of `MUL_MAT`, `CONV_1D` or `CONV_2D` with `groups == 1`:**

| the weight will be written as | K (ne[0]; IC x K for a conv: pad IC) | N (ne[1]; OC) |
|---|---|---|
| quantized (eligible, not `keep_float`, block size B > 1) | multiple of B (32) | untouched |
| F32 (everything else, including an F32 export) | multiple of 4 | multiple of 4 |

A pad is skipped when it would grow the weight by more than 25% (`(Kp x Np) / (K x N) > 1.25`): padding
the 1m's acoustic stream 72 -> 96 as well as its decoder (+78% MACs in those convs) ran 7-10% slower than
padding the decoder alone, on both boards. F16/BF16 exports follow the F32 row.

**Mechanisms, all exact in real arithmetic (F32 output equal to unpadded up to summation order):**

* **N side.** Pad the weight's rows with zeros, and the bias the node's output is summed with (the
  `ADD`, or the conv form's `RESHAPE` + `ADD`). Then slice back to N with a `VIEW` (offset 0) right after
  the bias -- unless the next rule consumes the padded rows.
* **K side, carried by the producer (free).** If the consumer's input comes from an N-padded producer
  through nodes that each have ONE consumer and map 0 to 0 per channel -- the bias `ADD`/`RESHAPE`,
  `GELU`, `SILU`, `RELU`, `TANH`, `LEAKY_RELU`, a `MUL` by a per-channel weight (padded with zeros) --
  the padded channels arrive as zeros: no `VIEW`, no `PAD_1D`. This is `pw0 -> GELU -> pw1` and the
  mel conv feeding the decoder's embed conv.
* **K side, supplied by the consumer, `MUL_MAT` only.** Otherwise insert `PAD_1D(lp0 = 0, rp0 = Kp - K)`
  on the input, which pads ne[0] -- the LayerNorm-fed matmuls. One copy of a `[K, T]` activation per
  call, against a `[K, N, T]` product.
* **K side of a convolution with no producer route: not padded in v1.** A conv's channel axis is ne[1]
  and no engine op pads it. Padding a residual stream as a whole (every producer's rows, every reader's
  columns, the per-channel weights on its elementwise ops) is a v2: on the 276k's acoustic stream it was
  worth 1.04x on the Pi 4 at one thread and nothing measurable on the Pi Zero (below).

**Recorded in the file.** `loom.pad.weights`: each padded tensor's name with its true ne[0] and ne[1], so
a tool that reads weights by name (card builders, quantize scripts, the gate's oracles) can strip the
zeros.

**Exposure.** `--pad-axes {auto,off}` on `main_export` and a config attribute, **default `auto`**: an
export with every axis aligned is byte-identical either way, and an unaligned one gets the same outputs
faster. `off` is the escape hatch, not an instrument.

**No new ops.** `PAD_1D`, `VIEW` and `ADD` predate rc4, so a padded file needs no newer runtime and no
`OP_FIRST_RUNTIME` row.

**Gates.**
1. Exporter CI on synthetic graphs, one per mechanism, each with a sabotage that must go red (a nonzero
   padded bias; a missing `VIEW`; a producer route through a non-zero-preserving op, e.g. `SIGMOID`).
2. Per model: F32 padded vs unpadded within the reference's own f32-vs-f64 spread (Vocetta2: 2.0e-5 from
   f64 against upstream f32's 1.6e-5; 99-123 dB SNR).
3. Per quantized card: the model's WER bar on the padded Q8 file.
4. Board: the paired per-launch protocol, F32 padded vs unpadded, before any republish.

## Consequences

* Files grow by their zeros: 1.0-4.0% for the F32 rule on Vocetta2 (the 276k most, its every width being
  one or two short of a multiple of 4), more for Q8 (still smaller than F32).
* A tensor's GGUF shape is no longer the checkpoint's; `loom.pad.weights` is the map back.
* Graphs gain a `PAD_1D` per LayerNorm-fed matmul and a `VIEW` per N-padded output that is not carried.
* Republishing is per model and goes through the republish recipe: an F32 file's bytes change wherever an
  axis was unaligned.
* x86's AVX2 tinyBLAS wants K % 8; the F32 row's 4 is the ARM boards' number, and x86 keeps its tail.

## Measured 2026-10-21/22: the Vocetta2 prototype

Released rc18 wheel (Pi 4: from PyPI; Pi Zero: the board's rc18 venv, `bluetooth-km-switch` left running,
which is fine within a run and not for a published number). One launch per arm per round, arms rotated per
round, 5 rounds; Pi 4 cooled to 60 C before every arm; Pi Zero launched detached and never polled. Unit of
work: upstream's ids for eval sentence 5 (175 ids, 11.2 s of audio), one untimed warm-up, then 5 (Pi 4) or
3 (Pi Zero) timed calls. Every arm of a model emitted the same sample count. Mean ms per utterance. The
rows come from four runs per board; each run carried its own unpadded F32 arm, and those agree within 1%
across runs (Pi Zero 1m 7179-7262, Pi 4 276k 404-406), so rows from different runs compare.

| arm | Pi Zero 276k | 500k | 1m | Pi 4 1 thr 276k | 500k | 1m | Pi 4 4 thr 276k | 500k | 1m |
|---|---|---|---|---|---|---|---|---|---|
| F32 (as published) | 4115 | 4515 | 7262 | 405 | 441 | 633 | 243 | 257 | 300 |
| F32, K to 4 | 3978 | 4323 | 6008 | 357 | 404 | 547 | 234 | 251 | 285 |
| **F32, K and N to 4** | **3766** | **4110** | **5355** | **356** | **406** | **552** | **231** | **253** | **285** |
| F32, K to 32 (decoder) | 3984 | 4557 | 6312 | 367 | 429 | 581 | 235 | 257 | 294 |
| Q8_0_PC as exported today | = F32 | 4514 | = F32 | = F32 | 439 | = F32 | = F32 | 255 | = F32 |
| Q8_0_PC, K to 32, linears only | 3754 | 4189 | 5260 | 399 | 492 | 679 | 242 | 269 | 311 |

The 276k's acoustic stream (31 -> 32, the residual-stream case v1 skips) is worth 1.138x against 1.097x
without it on the Pi 4 at one thread, 1.049x against 1.038x at four, and nothing measurable on the Pi Zero.

**Exactness.** Padded F32 against unpadded: 99-123 dB SNR over six sentences per size; the 276k's longest
sentence is 2.0e-5 from upstream at f64, against upstream's own f32 at 1.6e-5.

**Quality of the padded Q8 files** (whisper-small, 60 x86 syntheses per size, 30 through upstream's ids
and 30 through the text door; bar = F32 + 1 pp, the Pocket-TTS bar):

| size | F32 (bar) | Q8, K to 32, every eligible weight | Q8, K to 32, linears only | Q8 linears + F32 N to 4 |
|---|---|---|---|---|
| 276k | 46/470 = 9.79% (10.79%) | 44 = 9.36% | 44 = 9.36% | 45 = 9.57% |
| 500k | 55/470 = 11.70% (12.70%) | 55 = 11.70% | 52 = 11.06% | 52 = 11.06% |
| 1m | 40/470 = 8.51% (9.51%) | 39 = 8.30% (41 = 8.72% with the acoustic net too) | 40 = 8.51% | 40 = 8.51% |

All ten pass. Waveform SNR against F32 says less than it seems to: the 1m with its acoustic net quantized
sits at 11.8 dB and still passes.
