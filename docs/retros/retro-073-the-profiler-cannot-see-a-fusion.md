---
type: retro
date: 2026-10-03
domain: performance
tags: [performance, profiler, fusion, ggml, conv, wakehubert, p4.31]
---

# Retro-073: The Profiler Cannot See a Fusion

## The Issue

After the depthwise convolutions were fixed (P4.31 in
[Epic-05](../epics/epic-05-edge-performance.md#p431--depthwise-convolutions-a-direct-kernel-a-simd-interior-and-the-causal-block-fused--done-2026-10-03)),
WakeHuBERT tiny's `$LOOM_PROFILE` table put `ADD` at 12.7% and `PAD` at 11.3% of engine time, and the
user asked for the "unfused bias and residual adds" to be fused. Read at face value, the table said
all 25 `[125, 256]` ADDs per call were separate passes -- including the 1x1 convolutions' bias and
residual ADDs, which `ggml-0005`/`ggml-0007` exist to absorb.

## Root Cause

The profiler times nodes through `ggml_backend_sched_set_eval_callback`, which makes the scheduler run
the graph **one node at a time** (`include/loom/core/profile.h`). `ggml_cpu_try_fuse_ops` only fuses
nodes that one compute call sees together, so **under the profiler no CPU fusion ever fires**, and every
fused node shows up with its own unfused cost. On the real path the 1x1 conv chain was already fused: a
`GGML_CPU_DISABLE_FUSION=1` A/B costs 0.5 ms (12.5 → 13.1 ms).

The table did contain the real target, one level down. Per node, the residual ADDs cost 0.012 ms and the
**bias** ADDs 0.075 ms for the same shape -- a src1 with `ne0 = 1` is a broadcast along the row, which
ggml's binary op runs per element. The depthwise conv's bias ADD and its causal PAD were the passes no
detector covered; fusing those (`ggml-0022`) was 12.67 → 11.29 ms.

## The Fix

Nothing to fix in the profiler: running node by node is what makes per-node timing possible. The
practice changed: an elementwise row in a profile is a *candidate* until a timing A/B with
`GGML_CPU_DISABLE_FUSION=1` says how much of it the real path still pays, and the per-node table
(`$LOOM_PROFILE_NODES`) is read for which instances are slow, not just the op total.

## Takeaway

**A per-node profile measures the graph as written, not the graph as run.** Anything the backend does
across nodes -- fusion today, graph-level scheduling tomorrow -- is invisible to it, and the profile will
attribute to separate nodes a cost the real run no longer pays. Before scoping work from an elementwise
row, time the real path with the cross-node optimisation switched off and on. Same family as
[Retro-026](retro-026-three-nodes-were-half-the-runtime.md)'s profiler-overhead lesson: the instrument
changes what it measures.
