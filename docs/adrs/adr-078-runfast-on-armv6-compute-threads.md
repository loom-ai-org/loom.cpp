---
type: adr
status: proposed
date: 2026-10-14
domain: performance
tags: [armv6, pi-zero, vfp, runfast, flush-to-zero, numerics, ggml-patch]
---

# ADR-078: RunFast on ARMv6 Compute Threads, Behind a Gate Across the Zoo

Surfaced by [ADR-077](adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md)'s gate 1. Serves
[Epic-05](../epics/epic-05-edge-performance.md) on the tier
[ADR-026](adr-026-armv6-is-the-floor-and-gets-its-own-kernels.md) makes the engine's floor.
**Proposed:** each gate below can stop it, in order.

## Context

The ARM1176's VFP11 has two modes. In its default IEEE mode it handles denormals and NaNs fully,
some of them by trapping to the kernel's VFP support code. In **RunFast** (FPSCR `FZ | DN`, with the
exception traps off, which is Linux's default) it flushes denormals to zero, returns the default NaN
for any NaN result, and does everything in hardware.

Two things measured on the Pi Zero W make that mode worth a decision:

* **`vmla.f32` runs 1.65x faster in RunFast**, on the same instruction stream: 0.41 GMAC/s against
  0.59-0.68 (`scripts/bench44.c`, eight independent accumulators). IEEE mode does not pipeline
  back-to-back `vmla`. On Citrinet's depthwise schedule a blocked F32 kernel went from 0.67 s to 0.50 s
  (`scripts/bench43.cpp`). This raises the ceiling of every F32 kernel on the board, and ggml's ARMv6
  F32 GEMM (`tinyBLAS_F32_ARMV6`) is the incumbent in most of the board's models.
* **Denormal and NaN arithmetic trap to software** in IEEE mode. A `-inf` in a branchless exp made
  softmax 18x slower (ggml-0029), and the fix was a predicated move added to every element. In RunFast,
  denormals never reach software. Whether the NaN case also stops trapping has to be measured.

**End to end, before any patch**, RunFast set from Python on the calling thread (ggml computes there
at one thread) takes jfk on Citrinet-256-ls's F32 card from 59.7-62.5 s to **20.0-22.5 s**, and its
Q8_0 card from 33.3-34.9 s to 29.3-30.9 s, with the same transcript ids as IEEE mode
([ADR-077](adr-077-static-int8-for-citrinet-256-on-the-pi-zero.md#found-while-measuring-gate-2-the-f32-card-was-paying-for-denormal-traps)).
Most of the F32 gain is denormal traps: BatchNorm-folded dead channels leave weights at ~1e-33, whose
products underflow. That checkpoint shape is common, so other models on this board may pay the same.

The int8 SMLAD path does not use the VFP, so it gains nothing (~1% on ADR-077's GEMM).

The cost is numerics. A result below 1.18e-38 becomes 0, an input below it is read as 0, and a NaN
loses its payload (it is still a NaN). Every other result is bit-identical. FPSCR is per thread, and
Linux saves it on a context switch.

## Options

1. **Keep IEEE mode.** No numerics change, and the 1.65x stays out of reach.
2. **RunFast for the duration of a graph compute, on every compute thread**: set on entry to ggml's
   compute-thread function, and restored on exit. This is ARMv6 builds only, in a ggml patch. The caller's
   thread is thread 0, so the restore is what keeps the host (Python, numpy, the Lua driver) in IEEE mode.
3. **RunFast process-wide at engine init.** One line, but it changes the host's floating point too.
   Rejected for that reason.
4. **RunFast inside loom's ARMv6 kernels only**, set and restored per call. This is the tightest scope.
   But every op without a loom kernel (softmax, the norms, the unary ops) keeps trapping, and each call
   pays two FPSCR writes.

## Decision

**Option 2, with a runtime switch to turn it off** (an escape hatch for a model the gate does not
cover, not an instrument). Whether it is on by default is decided by the gates, not now.

**Gates, in order. Each one can stop the work:**

1. **Does the emulator agree?** The ARMv6 numerics gate runs in qemu, which is bit-identical to the
   board in IEEE mode. Confirm that it honours `FZ | DN` bit for bit too, on a microbenchmark whose
   output depends on both (a denormal product, a NaN payload). If it does not, gate 2 runs on the board.
2. **Numerics across the zoo.** Every model with an ARMv6 oracle runs with and without the switch, at
   today's tolerances: VITS's hash, the ASR transcripts and their tensors, the NER labels. Any model that
   changes gets its difference recorded and explained. Each ASR model also needs the tensor oracle,
   not just the transcript, and each TTS model needs the ASR oracle on its audio.
3. **An audit of what depends on IEEE mode.** Loading Citrinet and running jfk leaves the
   invalid-operation, overflow and underflow flags set in FPSCR in either mode, so something in that
   window already makes NaNs and infinities; the audit names what. It covers `isnan`/`isinf` checks (still work), NaN
   payloads (none known), denormal-range constants such as an epsilon below 1.18e-38, and the
   predicated moves added to keep NaN and denormals out of the VFP (ggml-0029), which may become
   removable.
4. **Speed on the board, cooled and paired**, per model: the Pi Zero set (Citrinet-256-ls, Moonshine
   tiny, sanoTTS heart-nano, the piperlite voices) and the P7 set (VITS, conformer-ctc,
   distilbert-ner). An isolated op's 1.65x is a ceiling. The end-to-end number decides the default.

## Consequences

* If it goes through, every F32 op on ARMv6 gains without a kernel change, and the F32 arms of
  ADR-077's gate 1 analysis have to be re-measured in RunFast. That comparison is already done for the
  depthwise conv.
* The engine's ARMv6 output then differs from its x86/aarch64 output wherever a denormal or a NaN
  payload occurs, and nowhere else. Gate 2 says how often that is.
* A ggml patch to carry, ARMv6 only, of a few lines. It touches the compute-thread entry, which every
  other patch leaves alone.
