---
type: retro
date: 2026-10-28
domain: backends
tags: [vc4, qpu, emulator, tmu, hazard, raspberry-pi-zero, verification]
---

# Retro-095: An Emulator That Allows What the Hardware Breaks

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (P4.44); the decision is
[ADR-090](../adrs/adr-090-the-pi-zero-qpus-are-an-accelerator-beside-the-cpu.md).

## Issue

The first QPU GEMM kernel was exact in the emulator and wrong on the Pi Zero: some output columns
came back computed from the activation column `ahead` TMU requests earlier. It happened only when
more than one QPU shared a slice's TMU, and more often the shallower the prefetch, so a single-QPU
hardware run would have passed too.

## Root cause

The per-k instruction did two things at once: it wrote the next TMU address (`tmu0_s = r2`) and
signalled `ldtmu0` to pop the oldest result. The reference guide's semantics give that instruction a
well-defined meaning and the emulator modelled exactly that meaning. The hardware does not honour it
under contention. py-videocore's assembler has a checker that forbids the pairing, and the warning
was there to read; the emulator encoded the documented semantics, not the checker's restrictions.

## Takeaway

**An emulator that refuses what a reference toolchain warns about is worth more than one that
models it.** Where a vendor's assembler or validator rejects a construct, the emulator must reject
it too (`qpu_emu.h` now throws on an `ldtmu` in an instruction that writes a TMU register), because a
faithful model of a forbidden construct is the one place emulator and hardware are guaranteed to
disagree. Then confirm on hardware across the axes that change contention -- QPU count and prefetch
depth -- not one configuration.

## Record

* Both fixed schedules (a separate pop, and the pop on the last fmac of the k step, whose own r4
  read sees the old value) are exact on hardware at TMU depths 1..4: 8 launches x 12 QPUs each. The
  17-instruction one is the default (`qpu_gemm.h`, `GemmVariant`); `qpu_gemm_wt.h` follows it.
* `test_vc4_gemm_emu` runs both variants at every depth in the emulator, which now also refuses the
  original pairing.
