---
type: adr
status: accepted
date: 2026-11-12
tags: [gpu, cuda, performance, decode, rms-norm, cuda-graphs, exporter]
supersedes: []
---

# ADR-113: A GPU Decode Step Takes Fewer Kernels, and CUDA Replays It as a Graph

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (backends) and
[Epic-05](../epics/epic-05-edge-performance.md) (performance). Found while putting CosyVoice3's stream on a
GPU ([ADR-112](adr-112-cosyvoice3-streams-the-way-its-reference-does-by-re-running-the-flow-a-hop-at-a-time.md)).

## Context

After `RSQRT` became native, CosyVoice3's Qwen2-0.5B LM decoded at 2.8 ms a token on an RTX 5090. Timers
around the bridge calls split one token as 0.10 ms graph build, 0.03 ms inputs, 0.06 ms sampling, and
**2.63 ms on the GPU**. So the GPU was the bottleneck, not the host. The step was ~2,100 nodes, ~1,420 of
them kernels (the rest views), at ~1.85 us each. F32 weights put a floor of about 0.8 ms a token on
memory reads (~1.4 GB at 1.8 TB/s), so roughly 1.8 ms was per-kernel cost.

Per layer the LM ran ~59 kernels. Two patterns accounted for a third of them:

* **RMSNorm, 9 kernels twice a layer.** The exporter's `fuse_rms_norm` turns PyTorch's chain into one
  `RMS_NORM`, which ggml-cuda fuses with the weight `MUL` into a single kernel. But the matcher knew the
  square only as `pow(x, 2)`, and this LM traced `x * x`.
* **Zero biases, 4 kernels a layer.** MIL gives a bias-free `nn.Linear` an all-zero bias. The exporter
  lowered every `linear` to `MUL_MAT` + `ADD`, so `o_proj`, `gate`, `up` and `down` each added 0.0.

Separately, a standalone ggml builds without CUDA graphs (`GGML_CUDA_GRAPHS` defaults OFF; llama.cpp turns
it on). The engine already reuses a decode step's graph within a KV bucket (P4.0.15), which is the
condition ggml-cuda needs to capture a step once and replay it.

## Options

1. **CUDA graphs alone.** A build flag, no change to any file's numerics.
2. **Fewer kernels from the exporter**, where the rewrite is the same arithmetic: the square in any
   spelling, and no `ADD` of a zero constant.
3. **ggml's fused rotary and flash attention.** These would remove more kernels: the rotary is ~10 a
   layer, written out as the reference writes it. But `ggml_rope_ext` computes its angles itself instead
   of reading the reference's cos/sin tables, and flash attention reorders the softmax reductions. Both
   move numbers that today match the reference's tables bit for bit on the CPU, and CosyVoice3's LM
   draws under pinned uniforms with margins down to 1.9e-5.
4. **F16/BF16 weights.** This halves the memory floor, but it is an export-precision decision per model,
   not an engine fix.

## Decision

**1 and 2.**

* `fuse_rms_norm` accepts `pow(x, 2)`, `square(x)` and `mul(x, x)` as the square (`_squares`). It keeps the
  identity check, so the multiply must feed back the very `x` that was squared, and the
  single-consumer check.
* The `linear` lowering leaves out the `ADD` when the bias is a constant that is zero everywhere. One
  non-zero element keeps it. `x + 0.0` is `x` except for the sign of a zero.
* `LOOM_CUDA_GRAPHS` (default ON) sets `GGML_CUDA_GRAPHS`, and loom-py's CUDA wheel sets it too.
  `GGML_CUDA_DISABLE_GRAPHS=1` turns replay off at run time.

## Consequences

* RTX 5090, CosyVoice3, the same 224 speech tokens into the flow, three launches each. LM time per token:

  | | graphs off | graphs on |
  |---|---|---|
  | export before | 3.1 ms | 2.45 ms |
  | export after | 2.35 ms | **2.1 ms** |

  The 8.92 s sentence takes 1.29 → 1.05 s, 8.5x real time (RTF 0.117). The stream's first piece comes after
  0.4 s. 216 of 224 LM steps replay a captured graph; the rest capture once per KV bucket.
* The LM topology goes from 1,436 to 1,094 nodes: 49 `RMS_NORM` and 146 fewer `ADD`. The DiT and the
  vocoder lose their zero-bias adds too, 22 and 1.
* **Numerics.** Dropping a zero bias is exact. `RMS_NORM` sums the squares in another order than the
  chain did, so a model that spelled `x * x` moves at the ulp level on re-export. CosyVoice3's LM still
  draws the reference's tokens exactly on the dev box's CPU, the workstation's CPU and the 5090. Its
  gates are unchanged: one pass 139/139 (flow and vocoder numbers identical), stream 44/44, with CUDA
  graphs on.
* A re-export picks this up; a file already exported keeps its graph. The exporter's structure baselines
  move for every model with a bias-free `linear`.
* What is left. The flow is now the largest stage (0.42 s): 20 estimator calls over ~620 frames, with the
  guided and unguided passes run separately rather than as one batch of two. The LM's rotary (~10 kernels
  a layer) is the next per-token target if a rewrite can keep the reference's tables. Both are in the hub.

## Related

* `loom-exporter/loom_exporter/passes.py` (`fuse_rms_norm`, `_squares`), `topology_ops.py` (`_op_linear`),
  `tests/ci/test_passes.py`, `tests/ci/test_linear_bias_lowering.py`
* `cmake/Dependencies.cmake` (`LOOM_CUDA_GRAPHS`), loom-py `packaging/rt-cuda/CMakeLists.txt`
* ggml: `ggml-cuda.cu` `ggml_backend_cuda_graph_compute` (capture, warmup, replay), `GGML_CUDA_DISABLE_GRAPHS`
