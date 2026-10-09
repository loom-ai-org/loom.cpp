---
type: adr
status: accepted
date: 2026-10-08
tags: [backends, vulkan, metal, precision, mul_mat, conv_2d, ggml, sensevoice, f5-tts]
supersedes: []
---

# ADR-069: An F32 x F32 Matmul Asks for F32 Precision

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md). The failure that forced it is
[Retro-074](../retros/retro-074-a-gpus-f32-matmul-was-half-precision.md).

## Context

ggml's `MUL_MAT` carries a precision request, `op_params[0]`, set with `ggml_mul_mat_set_prec`. At the
default, two GPU backends stage an F32 x F32 product's tiles in half precision:

* **Vulkan**: `matmul_f32_f32` is compiled with `FLOAT_TYPE = float16_t` shared memory on any device
  with fp16 support, and its coopmat/coopmat2 paths convert both operands to f16 outright. A
  non-contiguous F32 operand is also copied to f16 before the multiply.
* **Metal**: `kernel_mul_mm_f32_f32` loads both operands into `half` threadgroup tiles.

Neither honours `GGML_PREC_F32` for F32 inputs (pinned v0.19.0 and upstream master, 2026-10-08). The
CPU and CUDA (cuBLAS SGEMM) compute in F32. So any value past 65504 reaching a matmul becomes inf on
two backends, and `inf * 0` becomes NaN. SenseVoice's kaldi power spectrum peaks near 1e10. Its mel
projection came back 99.5% NaN and the transcript was `''`.

## Options

1. **Rescale SenseVoice's front end in the exporter.** Since log(mel(s·P)) = log(mel(P)) + log(s),
   divide the spectrum by a constant and add the log back, adjusting the clamp floor to match. This
   fixes one model and leaves the class open: every F32 model is one large activation away from the
   same NaN, and nothing would say which.
2. **`GGML_VK_DISABLE_F16=1`.** This is Vulkan only (Metal has no equivalent), and it turns off fp16
   for every shader, which slows F16 and quantized exports that never had the problem.
3. **Patch both backends to always stage F32 x F32 in float.** This is simplest for loom, but it
   silently overrides a choice upstream made on purpose (LLM weights are rarely F32, and half tiles
   are faster). It is also a harder patch to send upstream.
4. **Call `ggml_mul_mat_set_prec` at each of the twenty `ggml_mul_mat` call sites.** Three attention
   primitives already do this for their scores. The next primitive written would forget it.
5. **Tag every F32 x F32 `MUL_MAT` with `GGML_PREC_F32` on the finished graph, and patch Vulkan and
   Metal to honour the tag.**

## Decision

Option 5. `GraphBuilder::build` walks the graph once (`request_f32_precision`, until 2026-10-09
`request_f32_matmul_precision`, in
`src/core/graph_builder.cpp`) and sets `GGML_PREC_F32` on every `MUL_MAT` whose two operands are F32.
`ggml-0023` makes Vulkan honour it: a float-shared-memory scalar pipeline, and F32 rather than f16
copies of non-contiguous operands. `ggml-0024` does the same for Metal: a float-tiled
`kernel_mul_mm_f32_f32_prec`, and the all-F32 mat-vec kernel on tensor-API devices, which have no float
variant. Both patches use ggml's own precision vocabulary, so each is a self-contained upstream PR
([UPSTREAM.md](../../cmake/patches/UPSTREAM.md) PRs 20 and 21).

A weight exported at F16 or quantized is left at the default. That precision was the export's choice,
and the CPU already converts `b` to the weight's `vec_dot_type` on those paths, so it would overflow
there too.

## Consequences

* **Cost.** Measured on the M1 Pro (`MTL0`, median of 10 calls, three alternating rounds): whisper-small
  587.9 → 586.8 ms, SenseVoice 114.6 → 115.6 ms. **On Vulkan (Radeon Vega 3) it is +28% to +36%**:
  whisper-small 6007 → 7713 ms, Parakeet-TDT 2706 → 3673 ms, because Vega's packed fp16 is twice its F32
  rate. The GPU still beats that box's CPU (whisper-small 13489 ms). On an RTX 5090 under Vulkan it is
  +16-20% (whisper-small 122.0 → 141.3 ms), the coopmat2 path given up for the scalar F32 shader. This is the trade the decision
  makes: an F32 export is correct on every backend, and an export that wants the half-rate path can
  ship F16 weights, which keep the default precision (not yet measured on Vulkan).
* **Verified on every Vulkan path.** `''` before the patches and the correct transcript after, on all
  three: the RTX 5090's coopmat2, its KHR coopmat (`GGML_VK_DISABLE_COOPMAT2=1`), and the plain fp16
  shaders (Intel Arc iGPU, Radeon Vega 3). **Not run:** Metal's tensor-API fallback, since no M5-class
  device is available; a hub item tracks it.
* The CPU and CUDA ignore the tag for F32, so their numbers and outputs are unchanged.
  `test_graph_builder_shapes` checks the tag itself (and goes red with the pass removed), since no CI
  machine has a GPU.
* **Amendment 2026-10-09: an F32 x F32 `CONV_2D` asks too.** Vulkan's conv2d coopmat2 and KHR-coopmat
  shaders stage both operands in `float16_t` and accumulate in it. An F32 `CONV_1D` lowered to ggml's
  direct `CONV_2D` therefore ran at half precision on an RTX 5090, while the same convolution through
  im2col + `MUL_MAT` was already tagged. F5-TTS's DFT-basis STFT came out with an fp16 noise floor that
  `log` amplified into speech twice as loud
  ([Retro-077](../retros/retro-077-a-gpus-f32-convolution-was-half-precision-too.md)). The pass now
  tags `CONV_2D` the same way (op_params slot 9), and `ggml-0028` gives a tagged node the scalar
  float shader ([UPSTREAM.md](../../cmake/patches/UPSTREAM.md) PR 25). Metal, CUDA and the CPU compute
  conv_2d in F32 already. Cost on the 5090 under Vulkan: Supertonic-2 39.8 → 44.8 ms, Citrinet-1024
  14.1 → 15.9 ms, F5-TTS 2558 → 2598 ms, whisper-small and Parakeet-TDT unchanged. Devices without
  a coopmat conv path (Vega 3, Arc iGPU) already ran the scalar shader and are bit-identical.
