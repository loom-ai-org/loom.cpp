---
type: adr
status: accepted
date: 2026-11-13
tags: [performance, guidance, cfg, kv-cache, chatterbox, cpu, cuda, hybrid-cpu]
supersedes: []
---

# ADR-118: Chatterbox Runs Each Guidance Pair as One Batch of Two

Serves [Epic-05](../epics/epic-05-edge-performance.md) (performance). Extends
[ADR-113](adr-113-a-gpu-decode-step-fewer-kernels-and-cuda-graphs.md)'s guidance-in-the-estimator to
Chatterbox's LM, where the two streams also carry KV caches; found by
[ADR-117](adr-117-chatterbox-streams-a-hop-at-a-time-by-re-running-its-full-attention-flow.md)'s timings.

## Context

On the workstation's CPU (Core Ultra 9 285K: 8 performance and 16 efficiency cores, dual-channel DDR5)
Chatterbox's one pass over an 8 s sentence took 20.2 s, and upstream PyTorch took 12.5 s. Per-module
timers (wall time per module call, around the bridge's compute) split loom's run:

| module | calls | s | ms/call |
|---|---|---|---|
| `t3_lm` | 201 | 7.5 | 37 |
| `t3_lm_uncond` | 201 | 7.3 | 36 |
| estimator | 20 | 4.0 | 198 |
| T3 embeddings, vocoder, flow encoder | | 1.5 | |

T3 ran classifier-free guidance as two calls on two KV caches (`extra_streams`, ADR-023), each one row
through 2 GB of F32 weights at about 55 GB/s, the memory's bandwidth. Upstream runs both guidance pairs,
T3's and the flow's, as one batch of two (46 ms a token for T3).

## Options

1. **Both streams as a batch of two in one graph.** The weight matmuls see two rows. T3's two streams have
   the same length and step together, so their attention is one attention over twice the heads (the
   conditional stream's 16, then the unconditional stream's) against one cache of twice the width under
   one causal mask: the generic attention `fuse_loom_attention` already matches, so the engine needs no
   batched-cache support. The logits are combined in the graph with a `cfg` input.
2. **Quantize T3** (F16 or Q8_0): half or a quarter of the traffic, but the gate's greedy tokens may move,
   and it is a separate decision for the published card.
3. **Fewer threads.** T3 alone was faster at 8 threads (31.5 ms a pass), but the estimator was 1.9x
   slower.

## Decision

**1**, for both pairs. An exporter change and a re-export; no engine change.

* `T3LMPhase` spells Llama's decoder out over HF's own modules: norms, projections and the MLP over
  `(2, n, 1024)`, the heads regrouped to `(1, 32, n, 64)` for the attention and back by slices (two
  reshapes would merge, and the fusion would replace a `(n, 2, 1024)` var with its `(1, n, 2048)` output).
  `rotate_half` is a constant signed permutation matmul, exact. `cond + cfg * (cond - uncond)`, the
  reference's combination, is in the graph; `t3_lm_uncond` is gone. `t3_prefill_embed` returns both
  streams and `t3_step_embed` the draw twice. An export check refuses a `t3_lm` with other than 30 cached
  attention nodes: an attention the fusion missed would still run, uncached.
* `EstimatorPhase` takes `cfg` and runs `solve_euler`'s batch and combination itself, as CosyVoice3's does.
* Ruled out: the streams as `(2n, 1024)` rows. I thought ggml's F32 matmul read the weights once per
  batch index; a microbenchmark of T3's shapes (2 GB per pass) said not: 16 threads, 29.2 ms for one
  column, 31.3 for two, 28.6 for one column in a batch of two. Rows also cost the sequence axis its
  symbol, so the rotation's slice views reached the engine as length 0.

## Consequences

* f64, random small models (`test_chatterbox_export.py`): the batched T3 is HF's two `LlamaModel` runs
  combined to < 1e-12, and either stream alone does not move the other; the estimator is `solve_euler`'s
  step to < 1e-12. HF's eager attention runs its softmax in f32 even in an f64 model (2e-08 off), so the
  reference there is HF's SDPA path.
* Gates on the re-export, all passing: the one-pass gate 45/45 on the dev box; one pass 45/45 and stream
  49/49 on the workstation's CPU and on its RTX 5090. The one-pass waveform is 2.83e-05 from the reference,
  as before; greedy T3 still draws the reference's tokens.
* Sabotage, in place and reverted: T3's `cfg` zeroed, T3 draws another sequence and the pinned noise no
  longer covers the grid (the gate fails); the flow's `cfg` zeroed, the waveform is 0.897 off (44/45).
* Workstation CPU, 8 s of audio, three interleaved runs each (wall, T3, estimator):

  | threads | before | after |
  |---|---|---|
  | 16 | 20.95 s, 13.84, 4.64 | **16.25 s**, 8.77, 4.90 |
  | 24 (the default) | 21.78 s, 14.78, 3.90 | 20.36 s, 12.33, 4.08 |

  A T3 step is 44 ms at 16 threads, under upstream's 46; the weights' one read is 29 ms of it, and the
  rest is about a thousand small nodes, each a barrier across every thread. At 24 threads the efficiency
  cores make every barrier slower and the step is 61 ms. The estimator is compute-bound and gained
  nothing (5% slower on this CPU), where upstream's flow takes 2.6 s: a gap in ggml's GEMM and
  convolutions on these shapes, which is in the hub with the thread default.
* RTX 5090: the one pass 1.8 s -> 1.3 s (6.2x real time, RTF 0.16); the stream 3.6 s -> 2.7 s, its first
  piece after 0.5 s.
* The KV cache is one cache of 2 x 1024 per layer instead of two of 1024, the same memory.

## Related

* `loom-exporter/loom_exporter/chatterbox_export.py` (`T3PrefillEmbedPhase`, `T3LMPhase`,
  `T3StepEmbedPhase`, `EstimatorPhase`, `_check_t3_attention`), `chatterbox_driver/00_stream.lua`,
  `tests/ci/test_chatterbox_export.py`
* `loom-exporter/loom_exporter/passes.py` (`fuse_loom_attention`)
