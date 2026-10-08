---
type: retro
date: 2026-10-08
domain: backends
tags: [backends, vulkan, metal, precision, mul_mat, sensevoice, ggml, oracle]
---

# Retro-074: A GPU's F32 Matmul Was Half Precision

## The Issue

`sensevoice-small.gguf` transcribed `samples/jfk.wav` to `''` on Vulkan (Radeon Vega 3) and Metal
(M1 Pro), and correctly on the CPU and CUDA. Both engines failed identically: P4.31's and rc14's. The
hub's starting suggestion was to diff the three backends' `supports_op`, on the theory that one backend
was missing an op the others had.

## Root Cause

It was not coverage. Every op was supported; the difference was **precision**. For an F32 x F32
`MUL_MAT`, Vulkan's `matmul_f32_f32` stages both operands in `float16_t` shared memory on any
fp16-capable device, and Metal's `kernel_mul_mm_f32_f32` stages them in `half`. SenseVoice's kaldi
front end multiplies a power spectrum by the mel filterbank, and that spectrum peaks near 1e10, far
past half's 65504. The tile overflowed to inf, `inf * 0` gave NaN, 87,440 of the projection's 87,840
outputs were NaN, and CTC decoded blanks. CUDA's cuBLAS SGEMM computes in F32, which is why it alone
was fine. The rest of the model was exact: with the matmul fixed, no node differs from the CPU by more
than 1e-5 in abs-sum, and the logits agree to six significant figures.

Neither backend honours `GGML_PREC_F32` for F32 inputs, so the existing escape hatch did nothing even
where loom had set it (attention scores).

## The Fix

[ADR-069](../adrs/adr-069-an-f32-matmul-asks-for-f32-precision.md): `GraphBuilder` tags every F32 x F32
`MUL_MAT` with `GGML_PREC_F32`, and `ggml-0023`/`ggml-0024` make Vulkan and Metal honour it.

**How it was found, which is the reusable part.** One process holds both devices, so the CPU is the
oracle. A scratch hook in the profiler's eval callback (`LOOM_PROFILE` runs node by node) wrote each
node's NaN count, sum and abs-sum to a file. Running the model once per device and joining the two
files on (op, name, shape, occurrence) named the first divergent node in one pass. Node-by-node
execution on Vulkan still failed, which ruled out fusion before anything was read. `GGML_VK_DISABLE_F16=1`
then confirmed the mechanism in one run. The same hook found the Kokoro-on-Metal defect
([Retro-075](retro-075-upstream-had-already-fixed-the-norm.md)), the CUDA abort's node, and a fourth
defect the verification sweep turned up: Soprano near-silent on Vulkan, a transposed convolution
overrunning its fixed shared window (`ggml-0027`).

Verified: SenseVoice correct on every Vulkan path (coopmat2, KHR coopmat, plain fp16) and on `MTL0`; the
card gate green on Vulkan on the Vega 3 (40 models), Metal (7) and CUDA (7), and 37/40 on the 5090's
Vulkan (three pre-existing NVIDIA-only failures, on the hub). The cost is nil on Metal, +16-20% on the
5090 and +28% to +36% on the Vega 3
([Epic-04 §6](../epics/epic-04-backends-and-accelerators.md#6-gpu-correctness-f32-matmul-precision-the-metal-norm-the-cuda-mat-vec-stride-2026-10-08)).

## Takeaway

**On a GPU backend, F32 weights do not mean F32 arithmetic unless the graph asks for it.** ggml's
default precision is chosen for LLMs, whose activations are normalised and whose weights are rarely
F32. Audio front ends are neither: a power spectrum, an unnormalised filterbank or a raw energy can
exceed 65504 while every downstream op is well conditioned. When two GPU backends fail and a third
works, suspect precision as readily as coverage, and test it first with the backend's own
precision switch (`GGML_VK_DISABLE_F16`), which costs one run.
