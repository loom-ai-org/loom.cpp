---
type: retro
date: 2026-10-14
domain: performance
tags: [armv6, pi-zero, qemu, vfp, denormals, numerics-gate, oracle]
---

# Retro-086: qemu Matches the Board Only Where the VFP Never Bounces

## The Issue

The ARMv6 numerics gate runs in the emulated build container, not on the Pi Zero
(`scripts/armv6_oracle.sh`). The premise was "qemu's softfloat and the ARM1176's VFP agree bit for bit",
and the evidence was one calibration: P7.1's wheel gave the same VITS hash in both places.

[ADR-078](../adrs/adr-078-runfast-on-armv6-compute-threads.md)'s gate 1 asked the same question about
RunFast and wrote a microbenchmark to answer it (`scripts/bench45.c`: 22 VFP instructions, 200,000
operands each, drawn at denormals, underflowing products and NaNs). Its IEEE-mode control was meant to
agree everywhere. It did not.

## Root Cause Analysis

**The VFP11 executes only the easy cases in hardware.** Any operation near the bottom of the exponent
range (a denormal operand, or a result that might underflow), and some NaN cases, are bounced to the
kernel's VFP support code (`arch/arm/vfp`). That code is a separate implementation with its own
choices, and qemu agrees with the hardware, not with it:

* `vmla`, `vmls` and `vnmla` round once in the support code, close to a fused multiply-add. The
  hardware and qemu round the product and then the sum. The two differ by 1-3 ulp whenever the operands
  are small enough to bounce, even with every operand and the result normal: 12 of 20,000 such
  `vmla.f32` cases.
* A NaN keeps its sign through `vsub` and `vnmul` on the board; qemu flips it. A NaN accumulator plus
  an overflowing product gives +inf on the board. `vcvt.f32.f64` returns a signalling NaN unquieted.
* The underflow flag follows different rules.

VITS never produces values that small, so its hash matched and the calibration was true for VITS.
Citrinet-256-ls's F32 card is the opposite case: BatchNorm-folded dead channels put weights at ~1e-33
([Retro-085](retro-085-the-f32-card-was-slow-because-of-its-dead-channels.md)), so every one of its
1x1 convs multiplies in exactly the range that bounces.

## Resolution & Lesson Learned

`armv6_oracle.sh` now says where the calibration holds. In RunFast nothing bounces, and qemu then differs from the board only in the sign of a flushed zero and of one default NaN
(ADR-078, gate 1).

**Lesson: one matching hash calibrates an emulator for that workload's value range, not for the
machine.** Before using an emulator as the oracle for a model, ask whether the model reaches the cases
the real hardware hands to software (on the VFP11: denormals, near-underflow, NaN operands). Check those
cases with a microbenchmark that covers them, run in both places. A matching end-to-end hash says
nothing about a path the workload never took.
