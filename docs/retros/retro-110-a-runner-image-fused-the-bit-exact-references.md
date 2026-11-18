---
type: retro
date: 2026-11-18
domain: backends
tags: [ci, floating-point, fma, vc4, emulator, testing]
---

# Retro-110: A Runner Image Fused the Bit-Exact References

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md).

## Issue

From November 2026, loom.cpp's CI failed two hermetic tests on every PR, whatever the PR changed:
`test_vc4_conv1d_emu` (hundreds of edge outputs "wrong") and `test_vc4_gemm_qb_emu` (thousands "not
bit-exact", every one inside its float64 bound). Main's last run had passed them. The code was unchanged,
the tests pass locally, and they passed in an `ubuntu:26.04` container with the same GCC 15.2 and CMake 4.2.3.

## Root cause

The failing runs were the first on GitHub's Ubuntu 26.04 image (GCC 15.2). The passing ones were on Ubuntu
24.04 (GCC 13.3). The emulated QPU has no FMA: it rounds the fmul and the fadd apart, and the tests compare
it bit for bit with a C++ reference written `acc = acc + w * v`. GCC's default `-ffp-contract=fast` fuses
that into one FMA whenever the target CPU has one. Compiled for `-march=x86-64-v3` or `-march=native`,
both tests fail exactly as on the runner, and with `-ffp-contract=off` they pass. Compiled for baseline
x86-64, which is what the local machine and the plain container give, nothing fuses. So the runner's builds
target an FMA-capable level. Why the image does that was not established; the test should not depend on it.

## Takeaway

**A bit-exact comparison against hand-written float arithmetic must pin `-ffp-contract=off`.** The rounding
of `a + b * c` is a property of the compiler's target, not of the source. A test that models hardware
without FMA (the VC4's QPUs, an ARMv6 VFP) is only as exact as that flag. The five VC4 emulator tests now
set it per target (`tests/CMakeLists.txt`).

## Record

* Main `f6fe5ee` passed on Ubuntu 24.04.5 / GCC 13.3.0. #132's run 44181776292 failed on Ubuntu 26.04.1 /
  GCC 15.2.0, and so did its re-run.
* `ubuntu:26.04`, GCC 15.2.0-16ubuntu1, defaults to `-march=x86-64`, `-mfma` disabled: both tests pass at
  the default, `-march=x86-64-v2`, `-O0` and `-fno-strict-aliasing`. They fail at `-march=x86-64-v3` and
  `-march=native`, and pass at `-march=x86-64-v3 -ffp-contract=off`.
* With the flag, all five emulator tests pass in the container under a CMake build with
  `CMAKE_CXX_FLAGS=-march=x86-64-v3`.
