---
type: adr
status: accepted
date: 2026-11-10
tags: [performance, x86, sse, windows, msvc, denormals, ggml-patches]
supersedes: []
---

# ADR-105: x86 Without AVX2 Gets SSE Paths, and RunFast Covers x86-64

## Context

On a Celeron N4100 (Goldmont Plus: SSE4.2, no AVX) under Windows, onnxruntime 1.28.0's PyPI wheel ran
Matcha 5.1x, VITS 3.5x, Citrinet-256 3.1x and Zipformer small 1.6x faster than loom's CPU path, at F32
([ADR-096](adr-096-windows-builds-with-msvc-and-luajit-is-built-by-cmake.md)). onnxruntime's MLAS ships
hand-written kernels dispatched by CPUID. ggml compiles one DLL per CPU level, and the vector code inside each
is selected by compiler macros. Four things left ggml's `sse42` DLL mostly scalar:

1. **MSVC defines none of GCC's `__SSE*__` macros, even under `/arch:SSE4.2`.** ggml shims `__SSE3__` for
   MSVC only when AVX is on, so every `GGML_SIMD` path in the Windows `sse42` variant compiled as scalar C.
2. **The direct conv1d tile (ggml-0006) had NEON and AVX2+FMA arms only.** Convolutions are 80-90% of these
   models' CPU time.
3. **tinyBLAS's F32 GEMM had no `__m128` instance**, though every helper it needs exists. Without AVX, F32
   matmuls and im2col convolutions ran one `ggml_vec_dot_f32` per output.
4. **Denormal products.** BatchNorm-folded dead channels leave Citrinet with 138k weights below 1e-19 (down to
   3e-36). No stored tensor holds a denormal, but their products with ordinary activations do. x86 takes
   microcode assists on those, so three 1x1 convolutions ran 150 ms against 5 ms for same-shape siblings.
   [ADR-078](adr-078-runfast-on-armv6-compute-threads.md) fixed exactly this on ARMv6 and left later cores
   out, assuming they handle denormals in hardware.

## Options

**1. A clang-cl Windows build** (ADR-096 Option 2). clang-cl defines the macros, so it fixes item 1 alone,
and nothing for Linux or for items 2-4.

**2. Hand-written SSE kernels per op**, MLAS-style. The most speed, and the most code to own; nothing measured
says the remaining gap needs it.

**3. Fill the three gaps where ggml already has the structure, and extend RunFast.**

## Decision

Option 3, as two patches:
* **`ggml-0060-x86-sse-without-avx2.patch`**:
  * the SSE macros for MSVC's `GGML_SSE42` build, next to ggml's AVX shim;
  * an SSE arm of the direct-conv tile (4 channels x 2 `__m128`, multiply then add, the AVX2 tile's
    epilogues) and of its fused `LEAKY_RELU` copy;
  * `tinyBLAS<4, __m128, ...>` after the AVX arm, x86-64 only.
* **`ggml-0061-x86-runfast-ftz-daz.patch`**: ADR-078's `GGML_CPU_HAS_RUNFAST` on x86-64, with MXCSR
  FTZ | DAZ as the mode bits. It is set on entry to each compute thread's graph and restored on exit, so the
  caller's thread is unaffected after the compute. `GGML_CPU_RUNFAST=0` turns it off. It applies to every x86-64
  variant: assists cost there too (Citrinet on a native AVX2 build, Core Ultra 9 285K: 1.12x).

## Consequences

* **Celeron N4100, Windows, wheel rebuilt with both patches, F32 / Q8_0_PC RTF, before → after:**

  | | before | after | onnxruntime F32 / int8 |
  |---|---|---|---|
  | Citrinet-256 | 0.179 / 0.109 | 0.042 / 0.040 | 0.058 / 0.380 |
  | Zipformer small | 0.520 / 0.449 | 0.170 / 0.334 | 0.315 / 0.159 (9 clips) |
  | VITS (miro) | 0.917 / 0.936 | 0.304 / 0.306 | 0.259 / 1.324 |
  | Matcha | 13.26 / 13.03 | 3.78 / 4.60 | 2.59 / 14.36 |

  * loom now leads onnxruntime on both ASR models at equal WER, and trails on TTS F32 (VITS 1.17x, Matcha 1.46x).
  * The CPU now beats the UHD 600 iGPU on every arm except Zipformer Q8_0_PC.
  * Per step: the conv tile (VITS 1.54x, Matcha 2.15x), the macros (Zipformer 2.8x), tinyBLAS (1.04-1.17x),
    RunFast (Citrinet 2.9x on the Celeron).
* **Not bit-identical.** The SSE tile does the scalar tile's arithmetic per element, but it covers 8 positions
  rather than 4, and `ggml_conv_1d_direct_ok`'s shape rules are written in that width, so some convolutions
  change path.
  * VITS audio is within 104-120 dB SNR of the old output, and Matcha within 78-104 dB.
  * ASR F32 transcripts are identical on all 73 clips.
  * Zipformer Q8_0_PC changed 3 of 73 transcripts, WER 7.57 → 7.65: the macros also vectorise ggml's int8
    quantization of activations.
  * RunFast flushes values below 2^-126; ADR-078's caveat stands (`logf` of a denormal gives -inf, not -103.97).
  * Gate (`v5`): 21 of 22 real-model tests pass on a native build. CosyVoice3's cloned-voice free-running
    arm fails with or without these patches.
  * On an SSE4.2 build, 96 of 98 pass. CosyVoice3 fails the same way. VoxCPM2's free-running arm drifts to
    6.4e-4 RMSE (bound 1e-4, 9.2e-6 unpatched) while its teacher-forced arms are unchanged. That is the
    accumulation-order case its header warns about.
* **Linux gains too:** an SSE4.2-only build on a Ryzen 3 3250U, 0060 alone, Citrinet 1.61x, Zipformer 1.39x,
  VITS 1.72x, Matcha 1.55x. `ctest -L ci` passes on SSE2-only, SSE4.2, AVX-only and native builds.
* **Q8_0_PC matmuls don't reach tinyBLAS on this path**, so after the fix Zipformer Q8_0_PC is 2x slower than
  its F32 and Matcha's 1.2x. That's in the hub.

## Related

* [ADR-096](adr-096-windows-builds-with-msvc-and-luajit-is-built-by-cmake.md) — the Windows port and its first
  benchmark. [ADR-078](adr-078-runfast-on-armv6-compute-threads.md) — RunFast on ARMv6.
* [Epic-05](../epics/epic-05-edge-performance.md). `cmake/patches/UPSTREAM.md` PRs 55 and 56.
