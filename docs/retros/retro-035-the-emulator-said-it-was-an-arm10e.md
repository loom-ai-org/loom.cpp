---
type: retro
date: 2026-09-05
domain: packaging
tags: [armv6, qemu, verification, ggml-native, gate-that-could-not-fail]
---

# Retro-035: The Emulator Said It Was an ARM10E

## Issue

Two ARMv6 kernels were added behind `__ARM_FEATURE_SIMD32 && !defined(__ARM_NEON) && !defined(__aarch64__)`,
built in the emulated Raspbian, and `ctest -L ci` came back **81/81**. That number was reported as
evidence the kernels were correct. It was not evidence of anything about them: they were not in the
binary.

```
$ nm -C build/_deps/ggml-build/src/libggml-cpu.so | grep -c ARMV6
0
$ readelf -A .../ggml-cpu.dir/ggml-cpu/llamafile/sgemm.cpp.o | grep Tag_CPU_arch
  Tag_CPU_arch: v5TE
```

The whole ggml CPU backend had been compiled for **ARMv5TE** — one ISA rung below the target — for
every emulated build in the session.

## Root cause

`GGML_NATIVE` defaults **ON** for a standalone ggml build, and its ARM path asks the compiler:

```
$ gcc -mcpu=native -E -v - </dev/null      # inside the emulated container
-mcpu=arm10e  -march=armv5te+fp
$ gcc -mcpu=native -dM -E - </dev/null | grep __ARM_ARCH
#define __ARM_ARCH 5
```

**QEMU's guest `/proc/cpuinfo` is not the board's**, and GCC's native detection believes it: it reads
an ARM10E, which is ARMv5TE, which has neither SIMD32 nor the ARMv6 baseline. So the guard was false,
the kernels compiled out, and the suite passed on the generic path it had always been passing on.

Two things hid it. `Tag_CPU_arch` on the linked **`.so` reported v6** — it is the maximum over the
input objects, and the engine's own sources (which take no `-march`) are v6, so the one object that
mattered was masked. And the *wheel* build was fine all along, because `loom-py/CMakeLists.txt` forces
`GGML_NATIVE=OFF` so that a wheel is never tuned to its builder — which is why `nm` on the shipped
wheel found the kernels and a gdb breakpoint on the board hit them. **The artifact was right and the
gate was wrong**, which is the reverse of the usual shape and is why it survived.

## Takeaway

**An emulated build must pass `-DGGML_NATIVE=OFF`.** Under emulation `-mcpu=native` describes the
emulator, not the target, and it silently selects a *lower* rung — silently because a lower rung
compiles and passes. Every ARMv6 build in `.github/docker/Dockerfile.armv6`'s recipe and in Epic-08
§6 now says so.

**Check the object, not the library.** `readelf -A` on a `.so` reports the maximum `Tag_CPU_arch` over
everything linked into it, so one file built for the wrong ISA is invisible there. The per-`.o` check
is the one that answers the question.

**"The suite is green" is not "the code ran".** For a change behind an ISA guard the first check is
`nm` for its symbols, and the second is a breakpoint. Both were available and cheap here, and were
run only after a measurement came back flat — which is the wrong order.
