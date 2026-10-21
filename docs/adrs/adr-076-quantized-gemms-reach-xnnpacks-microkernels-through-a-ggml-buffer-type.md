---
type: adr
status: proposed
date: 2026-10-11
domain: performance
tags: [cpu, xnnpack, int8, qd8, qs8, q8_0, extra-buffer-type, ggml-patch, quantization, exporter]
---

# ADR-076: Quantized GEMMs Reach XNNPACK's Microkernels Through a ggml Buffer Type, on Per-Channel Weights

Step 3 of the LiteRT/XNNPACK queue in the [hub](../backlog/active-index.md#engine--performance). Serves
[Epic-05](../epics/epic-05-edge-performance.md). Builds on
[ADR-075](adr-075-quantized-weights-reach-ggmls-repack-by-placement-and-routing.md) (placement and
routing), [ADR-014](adr-014-patch-ggml-rather-than-write-kernels.md) (patch ggml, write no kernels) and
[ADR-003](adr-003-per-model-complexity-in-the-exporter.md). [ADR-077](adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md)
reuses its weight format. **Proposed:** it is accepted once the first ISA ships behind its A/B switch
and passes the gates below.

## Context

`scripts/bench_xnnpack.cpp` (hub, the XNNPACK item) compared XNNPACK against this tree's ggml on loom's
own shapes, on four machines. At F32 there was no case on any of them. At int8 there is one:
XNNPACK's dynamic int8 (`qd8`: per-channel int8 weights, activations quantized per row at run time,
F32 out) beats ggml Q8_0 by **1.5-2.8x on GEMMs at one thread on every machine**.
[ADR-075](adr-075-quantized-weights-reach-ggmls-repack-by-placement-and-routing.md) has since closed
part of that with ggml's own kernels, and what is left is narrower:

| core | ggml's best Q8_0 GEMM today | `qd8` over it (VITS / whisper GEMMs, 1 thread) |
|---|---|---|
| M1 (dotprod) | `CPU_REPACK` q8_0_4x4 | 1.32x / 1.58x (8 threads: 2.15x / 1.62x) |
| x86 AVX2 | plain Q8_0 (no Q8_0 repack exists) | 1.48-1.52x / 1.57-1.69x |
| Cortex-A72 (Pi 4, no dotprod) | plain Q8_0 | **2.69x / 2.41x** |
| ARM1176 (Pi Zero) | plain Q8_0 | 1.8x, but ggml's own F32 beats both (19.0 s against 22.4 s) |

Five more facts constrain the design:

* **The operator API is the wrong unit.** It links every ISA's kernels (1.46 MB of x86 microkernels,
  471 KB on ARMv6; 2.2 / 0.97 MB stripped with pthreadpool and cpuinfo), and it runs on its own
  pthreadpool, a second pool that oversubscribes beside ggml's.
* **The microkernels are reachable only from source.** XNNPACK's per-ISA kernel tables
  (`xnn_init_qd8_f32_qc8w_gemm_config` and its siblings) are `XNN_INTERNAL`, hidden in the built
  library. What one ISA needs is small: on AVX2 the `qd8` GEMM is 0.6 KB (1x8c8) plus 1.6 KB (4x8c8)
  of code, and its weight packer is 3.5 KB.
* **Q8_0 is not XNNPACK's int8.** Q8_0 has one scale per 32-element block; XNNPACK's int8 weights have
  one scale per output channel, and it has no block-scaled int8 type.
* **4-bit is not a target.** XNNPACK's dynamic 4-bit GEMMs are 1.0-1.8x slower than its own int8 on
  every machine, and `qb4w` cannot take a Q4_0 block at all (it needs a positive scale; Q4_0's is
  signed).
* **Convolutions pay a layout tax.** XNNPACK is channels-last and loom's activations are not. Over
  VITS's convs, XNNPACK's int8 conv beat ggml's Q8_0 conv 2.2-3.9x on the M1 and the Pi 4. With a naive
  transpose at each edge that became 0.70-2.40x, below 1 at the Pi 4's four threads and the M1's eight.

## Options

1. **Link the operator API.** Rejected for size and for the second thread pool.
2. **Call the microkernels from loom's own C++.** That computes a GEMM in this repository, which is
   what ADR-014 rules out, and it would need ggml custom ops that ggml's scheduler cannot see.
3. **A ggml patch that adds an extra buffer type, `CPU_XNNPACK`, built like upstream's KleidiAI one.**
   It vendors the selected kernel sources, packs weights in `set_tensor`, and runs `compute_forward`
   over ggml's own `ith`/`nth`. Chosen.

For the weight format, given that XNNPACK wants per-channel scales:

* (a) Requantize Q8_0 to per-channel at load: a second rounding of already-quantized weights.
* (b) A new ggml/GGUF type: patches to ggml, every GPU backend, and gguf-py.
* (c) **Row-uniform Q8_0.** The exporter quantizes per channel and writes it as ordinary Q8_0, every
  block of a row carrying the same F16 scale. Chosen.

## Decision

* **Weights.** `loom-export --quantize Q8_0_PC` writes per-channel int8 as row-uniform Q8_0 and lists
  the tensors in a GGUF key. Every backend reads it as ordinary Q8_0, unchanged. The bridge converts
  it to XNNPACK's layout losslessly, and refuses a tensor whose scales are not row-uniform.
* **Kernels.** A ggml patch, `ggml-cpu/xnnpack/`, holds the pinned XNNPACK commit's kernel sources
  (BSD-3), only those selected per ISA: the `qd8` GEMM (1xN and MxN tiles), its packer, and the
  `f32 -> qd8` convert. `compute_forward` quantizes the activation rows into the work buffer, waits at
  ggml's barrier, then tiles the GEMM across ggml's threads. This mirrors `CPU_REPACK`'s own
  `from_float` pass.
* **Placement and dispatch.** Weights reach the buffer through ADR-075's loader and routing,
  unchanged (`LOOM_CPU_REPACK=xnnpack`). The default becomes a dispatch table keyed on (op, dtype,
  CPU features) that picks `CPU_XNNPACK`, `CPU_REPACK` or `CPU_KLEIDIAI` per weight from measurements.
  The environment override is for A/B only. `LOOM_XNNPACK=OFF` compiles the bridge out.
* **Scope of the first cut:** `mul_mat` on Q8_0 weights. Not F32 (no case, and its two ARM defects
  are ggml fixes). Not 4-bit. Not convolutions, until an export emits channels-last activations.
  Not ARMv6, where XNNPACK has no SIMD kernel for `qd8` (ADR-077 covers that core).

## Consequences

* **Gates, before any default flips:** (1) a GEMM-level A/B against the best ggml path on each target
  ISA, inside one binary; (2) a model-level A/B on the M1, the Pi 4 and x86 (`paired_arms.py --env`);
  (3) an accuracy gate per quantized model card, oracle or WER, because per-channel is coarser than
  block-32. The harness only shows it comparable on synthetic data (2e-3 to 1.7e-2 against F32).
* **Expected gain is per core.** The A72 is the main case (2.4-2.7x at the GEMM) and x86 Q8_0 the
  second (1.5-1.7x). On dotprod Arm the margin over ADR-075 is 1.3-1.6x, and KleidiAI may take it
  first.
* **Cost:** a vendored kernel set to re-pin and re-select on each XNNPACK bump; a second ggml patch
  family to rebase on each ggml bump; and the `Q8_0_PC` export flavour.
* **What stays true:** a row-uniform Q8_0 file is a valid Q8_0 file, so a GPU, a build without the
  bridge, or an older engine all run it as before.

## The weight format, built 2026-10-16 for ARMv6 (ADR-079 P4.33)

`loom-export --quantize Q8_0_PC` and the `loom.quantize.per_channel` key exist now, as decided above:
row-uniform Q8_0, the row's absmax / 127 as every block's F16 scale, written by `gguf.quants`' arithmetic
with the row in place of the block (a row whose blocks all hold its absmax comes out byte-identical to
Q8_0). Their first user is not this ADR's bridge but ARMv6's own int8 kernels, behind the same kind of
extra buffer type (`CPU_ARMV6_I8`, [ADR-079 P4.33](adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md#p433-measured-2026-10-16-dynamic-int8-ggml-0041-ggml-0042-ggml-0043)).
The engine places a listed weight there only if the buffer type serves every convolution that reads it,
because a kernel in a non-host buffer loses the direct conv path. Any bridge built under this ADR will need
the same rule. The XNNPACK bridge itself is not built, so this ADR stays proposed.
