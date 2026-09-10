---
type: adr
status: accepted
date: 2026-09-04
domain: packaging
tags: [armv6, arm32, raspberry-pi, packaging, qemu, wheels, luajit]
---

# ADR-025: The ARMv6 Wheel Is Built in an Emulated Raspbian, and Shipped Outside PyPI

## Context

[Epic-08 §6](../epics/epic-08-packaging-and-release.md) scoped 32-bit ARM from a source read and
called ARMv6 — Raspberry Pi Zero / Zero W / Pi 1, an ARM1176JZF-S with VFPv2 and **no NEON** — a
declared non-goal, on the grounds that only the smallest TTS models are plausible on it and that no
PyPI wheel tag exists for `linux_armv6l`. It also said to reopen it if a real user named the board.
A real user named the board.

Two things then have to be decided, and they are separate questions that look like one:

1. **What machine compiles it?** There is no 32-bit ARM hosted runner on any provider and there will
   not be one. The dev box is x86-64. The board itself is a single 1 GHz core with 427 MB of usable
   RAM.
2. **How does it reach a user?** PyPI accepts only `manylinux*` and `musllinux*` platform tags for
   Linux, and the manylinux policy's floor is armv7. `linux_armv6l` is rejected at upload.

## Options

**Compile on the board.** Honest, and unusable as a release step: one core, no swap configured,
1.9 GB free. A full build there is measured in hours and cannot be a CI job at all.

**Cross-compile from the dev box** with `arm-linux-gnueabihf-gcc`. Rejected for three reasons that
are each sufficient, and the third is the one that would have been found late:

* the wheel's tag comes from `sysconfig.get_platform()` of the interpreter that *builds* it, so a
  cross build tags the host and nothing downstream notices;
* `GGML_NATIVE` is off and the 32-bit path passes no `-march`, so the baseline is the compiler's
  own default — a Debian armhf toolchain's is `armv7-a+neon`, which is not this board;
* even with the right `-march` on our own translation units, the **static parts of libgcc** are
  linked in from the toolchain. Debian's armhf libgcc is built for ARMv7 and contains `movw`/`movt`,
  which is ARMv6T2 — and an ARM1176 is ARMv6Z, which is not.

LuaJIT settles it independently: it bootstraps through `minilua` and `buildvm`, which run on the
*host*, so cross-building it needs a `HOST_CC`/`TARGET_CFLAGS` split that `cmake/Dependencies.cmake`
deliberately does not set up.

**Build natively inside an emulated userland of the target** — `docker run --platform linux/arm/v6`
against a Raspbian bookworm image, with `qemu-arm` under binfmt. The host *is* the target, so all
three cross-compilation hazards and the LuaJIT bootstrap stop being questions rather than being
answered.

## Decision

**The ARMv6 wheel is built in an emulated Raspbian bookworm container
(`.github/docker/Dockerfile.armv6`), and shipped as a GitHub release asset rather than to PyPI.**

Raspbian rather than Debian, specifically: Debian's armhf port has an ARMv7 baseline, and Raspbian
exists because the Raspberry Pi Foundation rebuilt that archive down to the Pi 1. Its gcc is
configured `--with-arch=armv6 --with-fpu=vfp --with-float=hard`, and its glibc is 2.36 — the same
version the board runs.

The build is one un-split `libggml-cpu.so`: `GGML_CPU_ALL_VARIANTS` is gated on
`CMAKE_SIZEOF_VOID_P EQUAL 8` in `loom-py/CMakeLists.txt`, because ggml's ARM variant ladder is
aarch64-only and a 32-bit build walks into `-march=armv8-a` at a compiler that has no such
architecture. Nothing is lost by that: every rung of the ladder distinguishes dotprod / FP16 / SVE /
i8mm, and none of those exists below ARMv8.

## Consequences

* **`pip install loom-py-rt` does not work on this board, and cannot be made to.** The install is
  `pip install <release-asset URL>`. README says so rather than letting pip fall through to an sdist
  build on a 1 GHz core.
* **The CI job must not name its artifact `wheels-base-*`.** `publish-pypi` globs that pattern into
  one directory and uploads it; one file with a tag PyPI refuses fails the whole `loom-py-rt`
  upload, including the wheels that were fine. It is `wheel-armv6l`.
* **`QEMU_CPU=arm1176` is what makes the emulated check a check.** qemu-user runs whatever CPU model
  it is given, and under a permissive one an ARMv7 build imports and passes — then takes SIGILL on
  the board. Measured rather than assumed: a program containing one `movw`/`movt` pair (ARMv6T2,
  which an ARM1176 is not) prints its answer under `QEMU_CPU=cortex-a15` and dies with "Illegal
  instruction" under `QEMU_CPU=arm1176`. Docker's own `--platform linux/arm/v6` already selects an
  ARMv6 model, so the variable is belt-and-braces — but it is the half that *says* what is being
  claimed, and a `--platform` default is not something to rest an ISA guarantee on.
* **The gate is emulated, and the artifact is verified on real hardware before a release claims it.**
  A Pi Zero W is the reference board.
* **It is slow, and the docs say by how much rather than implying parity.** Every optimisation in
  [Epic-05](../epics/epic-05-edge-performance.md) is `__aarch64__`- or AVX2-guarded, so this rung
  runs at the generic-C level with no SIMD at all.
