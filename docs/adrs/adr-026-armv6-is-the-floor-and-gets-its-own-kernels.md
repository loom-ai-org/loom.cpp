---
type: adr
status: accepted
date: 2026-09-05
domain: performance
tags: [armv6, arm32, kernels, simd32, raspberry-pi, scope]
---

# ADR-026: ARMv6 Is the Floor, and the Floor Gets Its Own Kernels

## Context

[ADR-025](adr-025-armv6-is-built-in-its-own-emulated-userland.md) made ARMv6 a supported target. The
question that followed was whether it may have **code of its own** — kernels written for ARM1176 and
run nowhere else, carried in `cmake/patches/` indefinitely because ggml would never take them.

Two rounds of scoping in [Epic-08 §6](../epics/epic-08-packaging-and-release.md) answered no, on carry
cost. Both were wrong, and the argument that corrects them is about *where the tier sits* rather than
about any one kernel: **ARMv6 is the bottom of this engine's range.** Below it is microcontroller
territory — no MMU worth the name, no dynamic loader, megabytes rather than hundreds of them — which
has none of the machinery this engine assumes and is not a target it will ever grow into. A kernel
written for ARMv6 therefore serves a **permanent tier**, not a passing one, and "we would carry it
forever" describes every other floor-defining piece of code in the project.

The measurements then showed the cost of saying no. On a Pi Zero W, at the shape of VITS's largest
convolution bucket (`scripts/bench27-29.c`, all arms verified identical):

| | MMAC/s |
|---|---|
| F32 GEMM, what runs today | 29.7 |
| q4_0 x q8_0, shipped `vec_dot` | 66.3 |
| `__smlad` swapped into `vec_dot` — the most a generic patch can do | 78.3 |
| **a dedicated kernel: order B + 2x2 register tile + `__smlad`** | **167.9** |

The drop-in is worth 1.18x over the shipped kernel. The dedicated one is worth **2.53x**. The
difference is not the instruction — it is the loop order and the register tile, and **neither is
expressible inside a `vec_dot`**, which computes one output element from one row and one column and
does not own the loop that calls it.

**That the difference is the TILE and not the instruction is the load-bearing part, and it was only
established later.** `__smlad` is emitted (192 in the shipped library) and does two MACs per
instruction, but the same 2x2 tile applied to plain **F32** reaches 153.7 MMAC/s against the quantized
kernel's 167.9 — within 10%. The dual-MAC advantage is spent on unpacking Q4_0 nibbles. So the case
for ARMv6-specific kernels does not rest on integer arithmetic at all; it rests on being allowed to
own the loop, which is equally true for the F32 kernel that conformer-ctc needs (99.7% of its weights
cannot be block-quantized: d_model is 176 and QK is 32). See Epic-08 §6.5-6.6.

## Options

**Generic code only.** What the first two scopings assumed. Caps ARMv6 at whatever a shape-blind
`vec_dot` can reach, which is measured: 78.3 of 167.9.

**Upstream it.** ggml has no int8 convolution for any architecture and no ARMv6 SIMD32 path at all;
there is no upstream demand to satisfy, and waiting for one is a decision not to have the performance.

**ARMv6-specific kernels, guarded and carried here.** Behind
`__ARM_FEATURE_SIMD32 && !defined(__aarch64__)`, which the stock Raspbian compiler defines and no
64-bit target does.

## Decision

**ARMv6 may have kernels written for ARMv6 alone**, in `cmake/patches/`, guarded so that every other
architecture compiles them out. Being unupstreamable is not an argument against one; being unmeasured
is.

The guard is compile-time and one-sided, so the cost to every other target is zero, and the cost here
is a patch to rebase when the ggml pin moves — which `cmake/patches/UPSTREAM.md` already governs for
sixteen other diffs.

## Consequences

* **A naming rule, because the tier now needs one.** In docs, comments, headers and the hub,
  **"ARM" unqualified means the 64-bit ARMs** — aarch64 Linux and arm64 macOS, the rungs
  `GGML_CPU_ALL_VARIANTS` builds a ladder for and everything Epic-05 was measured on. **ARMv6 is
  always spelled out**, never folded into "ARM", and never described as "32-bit ARM" where armv7
  could be meant. It has its own guard (`__ARM_FEATURE_SIMD32 && !defined(__aarch64__)`), its own
  kernels, its own benchmarks and its own section; treating it as a variant of the others is how a
  measurement taken on a Neoverse ends up quoted for an ARM1176.
* **ARMv7 is not ARMv6 either, and the guard has to say so.** SIMD32 is not an ARMv6 feature ARMv7
  dropped — ARMv7-A has it — so `__ARM_FEATURE_SIMD32` alone would capture every 32-bit Raspberry Pi
  OS install on a Pi 2/3/4, and there it would be a **regression**: ARMv7-A has NEON, whose
  `vmull_s8` does eight 8-bit MACs against `__smlad`'s two, and `vec_dot_q4_0_q8_0` already has a NEON
  arm those boards take. Hence `__ARM_FEATURE_SIMD32 && !defined(__ARM_NEON) && !defined(__aarch64__)`
  — "32-bit ARM with no vector unit at all". The tiling is what ARMv7 would want; as a NEON kernel,
  which is different work and is not this decision.

* **The floor gets a maintained fast path.** Epic-08 §6.7 holds the order: a dedicated `MUL_MAT`
  first, then the fp16 change, then routing the declined convolutions through it, then an int8 direct
  convolution.
* **`UPSTREAM.md`'s rule needs one amendment.** It requires a number from an x86 box *and* one from
  the Pi before a patch is called done ([Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md) is
  the register of what happens without it). An ARMv6-only patch is inert on x86 by construction, so for these the x86 number is a
  statement that nothing changed, not a speedup — state it that way rather than skipping it.
* **A second correctness burden.** These kernels change no output today (every prototype is
  `memcmp`-identical), but an int8 convolution would. Nothing in CI compares ARMv6 output bit-exactly
  — the board cannot hold the fixtures — so the check is the ASR oracle on the board, run by hand.
  That is weaker than the gates every other architecture gets, and the epic says so.
* **It does not make a Pi Zero fast.** VITS lands at ~29x real time with everything built. The models
  worth the effort are the matmul-bound ones (§6.4): distilbert-ner under four seconds, conformer-ctc
  at ~10x real time.
