---
type: retro
date: 2026-10-09
domain: backends
tags: [backends, vulkan, precision, conv_2d, coopmat, ggml, f5-tts, oracle]
---

# Retro-077: A GPU's F32 Convolution Was Half Precision Too

## The Issue

F5-TTS on Vulkan on an RTX 5090 failed the card gate's clipping check: peak 1.2055 against a bound of
1.001. The CPU gave 0.756, and Vulkan on the Radeon Vega 3, CUDA on the same 5090 and the Intel Arc iGPU
all passed. It was not loud in a few places only. The whole utterance was twice as loud (RMS 0.340
against the CPU's 0.179), and its correlation with the CPU waveform was 0.08.

The cheap bisect suggested by [Retro-074](retro-074-a-gpus-f32-matmul-was-half-precision.md) found
nothing. `GGML_VK_DISABLE_COOPMAT2`, `..._COOPMAT`, `..._F16`, `..._FUSION`, `..._GRAPH_OPTIMIZE` and
`..._MULTI_ADD`, one at a time, each gave **bit-identical** output.

## Root Cause

The per-node CPU-oracle diff ([Retro-074](retro-074-a-gpus-f32-matmul-was-half-precision.md#the-fix))
found the first divergence at node 22 of the reference clip's mel, long before the DiT. F5's STFT is a
1024-tap DFT-basis `CONV_1D`, which loom lowers to ggml's direct `CONV_2D`. On the 5090 that convolution
had fp16-sized error: a maximum of 123.5 where the CPU has 123.397, and exactly 0 where the CPU has
2.2e-11. The power spectrum and the mel projection (an F32 matmul, already tagged by
[ADR-069](../adrs/adr-069-an-f32-matmul-asks-for-f32-precision.md)) carried that error through almost
unchanged (abs-sum 220969 against 220869). The next op, `log(clamp(x, 1e-5))`, amplified it. Most of a
mel's bins sit near the floor, and there the half-precision noise floor is orders of magnitude above
the true value. The mean log-mel moved from -2.23 to -1.50. The model was conditioned on a reference
about twice as loud as the one it was given, and it spoke at that loudness.

The mechanism is ADR-069's again, in a different op. `ggml-vulkan`'s conv2d has three builds: a scalar
shader with float shared memory and float accumulators, a coopmat2 shader, and a KHR-coopmat shader.
The two coopmat builds stage both operands in `float16_t` **and accumulate in `float16_t`**
(`conv2d_use_fp16_shmem = device->coopmat2 || conv2d_use_cm1`, `ACC_TYPE float16_t`). The 5090 has both
coopmat flavours. The Vega 3 and the Arc iGPU have neither, so they run the scalar shader, which is why
they passed. Upstream master is the same (2026-10-09).

The single-switch bisect could not see this. With coopmat2 disabled, conv2d falls through to the KHR
coopmat build, which is just as half precision. With coopmat disabled, coopmat2 is still there. Only
both switches together moved the output (peak 0.748).

## The Fix

The tag ADR-069 put on F32 x F32 `MUL_MAT` now goes on F32 x F32 `CONV_2D` too
(`request_f32_precision` in `src/core/graph_builder.cpp`). `ggml-0028` makes `ggml_prec_set_acc`
accept `CONV_2D` (op_params slot 9, which no conv-2d builder writes). It adds a `prec_f32` bit to
Vulkan's conv2d pipeline key, and a tagged node gets a pipeline built from the scalar SPIR-V with the
scalar tile configuration, whatever the device would otherwise pick.

After it, on the 5090:

| path | peak | RMS | correlation with CPU |
|---|---:|---:|---:|
| CPU | 0.7565 | 0.1789 | 1 |
| Vulkan coopmat2, before | 1.2055 | 0.3396 | 0.08 |
| Vulkan coopmat2, after | 0.7478 | 0.1715 | 0.950 |
| Vulkan KHR coopmat (`GGML_VK_DISABLE_COOPMAT2=1`), after | 0.7478 | 0.1715 | 0.950 |
| CUDA (unchanged) | 0.7499 | 0.1714 | 0.949 |

The remaining gap to the CPU is ordinary GPU drift over 32 guided Euler steps: CUDA, which computes
conv_2d in F32, shows the same gap. On the Arc iGPU, the output before and after is bit-identical.

## Takeaway

**A feature switch that changes nothing has not cleared that feature. It may have handed the work to a
sibling that does the same thing.** Six bit-identical bisect runs read as "not a coopmat problem", but
coopmat2 and KHR coopmat were each the other's fallback. When a backend has several implementations of
one op, check which one actually ran before trusting a one-switch-at-a-time bisect. Turn off every
fast path together first. If that changes the output, bisect from there.

And **a precision contract has to name every op that can do the arithmetic.** ADR-069 covered
`MUL_MAT` because SenseVoice failed in a matmul. A `CONV_1D` with an F32 kernel takes the im2col +
`MUL_MAT` path (tagged) or the direct `CONV_2D` path (untagged) depending on a build flag. So the same
convolution was F32 on one lowering and half on the other.
