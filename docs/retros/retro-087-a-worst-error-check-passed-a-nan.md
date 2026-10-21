---
type: retro
date: 2026-10-16
domain: performance
tags: [armv6, pi-zero, vfp, numerics-gate, nan, activations, ggml-patch, oracle]
---

# Retro-087: A Worst-Error Check Passed a NaN

## The Issue

`ggml-0039` (ADR-079 P4.32f f2 step 3a) put SIGMOID, SILU and TANH on the VFP11's short vectors with an
own e^v: Cody-Waite reduction, a polynomial for e^r, and 2^n added to the result's exponent field in the
integer unit. Before the board gate it had passed every check it was given:

* the kernels matched their scalar C reference bit for bit over ~70M bit patterns, in qemu and on the board;
* the reference measured 2.40 ULP (sigmoid) and 1.30 ULP (tanh) against f64 over every 11th float with
  |x| <= 90, as good as glibc;
* `tests/ci/test_activations_armv6.cpp` passed with x = +-87, +-100 among its inputs.

The 12-model board gate then gave **heart (sanoTTS) all NaN** with the kernels on, and **changed the
conformer's transcript** ("and so are my fellow americans" -> "and so am my fellow\<unk\>").

## Root Cause Analysis

**The range check was symmetric and the representable range is not.** p = e^r lands in [0.7, 1.42], so
its exponent field is 126 or 127, and the sum with n has to stay in 1..254 to be a normal float. The kernel
returned inf or 0 for |v| >= 88.37, which keeps n <= 127 above. Below, n = -127 already happens from
v = -87.68, and with p < 1 (r < 0, from v = -88.03) the field goes to -1 and **borrows from the sign bit**:
a negative NaN. SIGMOID and SILU compute e^-x, so every x in (88.03, 88.37) came out NaN, and from
(86.98, 88.03] a wrong tiny value (harmless there). One NaN in heart's SILU spread through the rest of the
network.

**Three checks were each blind to it, each for its own reason:**

1. **Asm against its own reference proves the transcription, not the math.** The reference had the same
   symmetric bound, so they agreed on the NaNs too.
2. **`std::max` and `std::fmax` drop NaN.** The accuracy sweep kept `max(worst, err)` and the CI test
   `fmax(w, e)`; with err = NaN both return the old worst. The sweep's NaNs at x in (88.03, 88.37) never
   reached its maximum, and the test's input 88.5 was past the window while 87 and 88 were before it.
3. **The window was 0.34 wide.** A sweep of every 11th float covers it; a hand-picked list of
   "large values" did not.

A harness that drove each kernel through ggml (under qemu) at heart's shapes with |x| up to 200 found it in one
run (16 NaNs of 28,704 SILU outputs).

## Takeaways

* **A worst-error check must fail on NaN.** Write `if (!(e <= w)) w = isnan(e) ? INFINITY : e`, or test
  `isnan` explicitly. `max`/`fmax`/`>` comparisons all treat NaN as "not worse".
* **A bit-exact match against your own reference is a transcription check.** Pair it with an independent
  oracle (f64, the library) over the whole input range, including both ends of every range reduction.
* **Derive a range reduction's bounds from the representation, per side.** With e^v built from an exponent
  add, the low bound comes from the field's floor (n >= -125 here), not from the high bound mirrored.
* **Put the boundaries into the test as exact values, not as round numbers near them.** The test now feeds
  +-86.9, +-87, +-88, +-88.2, +-88.3, +-88.5 and +-100, and with the NaN-aware maximum it fails on the
  old kernels.
* **The end-to-end gate earned its cost.** The model with the largest activations (heart) was the one that
  caught it; Retro-086's lesson (numerics on the board, over many models) applies to kernels you wrote
  yourself as much as to emulators.

## The record

* Defect: `scripts/gen_armv6_act_f32.py`'s integer pass, `cmp |x|, 88.37` -> `andhs r3, r8, r4, asr #31`.
  Fix: `cmp |x|, 86.98` -> `andhs r3, r3, r4, asr #31` (0 for x > 0, i.e. v <= -86.98), then the 88.37
  test for inf. Same bounds in `scripts/bench57.cpp` and `scripts/bench57_accuracy.cpp`.
* Gate 12 (wheel v12, defect): heart-fft 76,288 of 76,288 samples NaN with `GGML_CPU_ARMV6_ACT=1`;
  conformer text changed. Gate 13 (wheel v13, fixed): no NaN, every transcript and CTC id identical,
  waveforms within one int16 LSB, whisper-small hears the same sentences. See
  [ADR-079](../adrs/adr-079-armv6-f32-kernels-on-the-vfp11s-short-vectors-then-dynamic-int8.md).
