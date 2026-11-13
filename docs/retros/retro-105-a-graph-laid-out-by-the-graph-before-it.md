---
type: retro
date: 2026-11-13
domain: engine
tags: [graph-builder, gallocr, determinism, cpu-fusion, stream-state, streaming, chatterbox]
---

# Retro-105: A Graph Laid Out by the Graph Before It

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming), from
[ADR-117](../adrs/adr-117-chatterbox-streams-a-hop-at-a-time-by-re-running-its-full-attention-flow.md).

## Issue

Chatterbox's stream gate passed every check but one: in park mode, a short sampled stream interleaved with
the long pinned one came out different from itself alone (48/49); in per-stream mode it was bit-identical.
A repro with both streams returning their MEL was bit-identical, so T3, the KV parking and the flow were
clean, and the difference entered at the vocoder, which has no cache. HiFT alone then showed it with no
stream at all: one mel vocoded, a longer one vocoded, the first again -- 1.1e-07 apart. Repeating the
first without the longer call in between gave 0, and so did running another module in between. One
thread or four made no difference; `GGML_CPU_DISABLE_FUSION=1` removed it.

## Root cause

`ggml_gallocr_alloc_graph` re-plans only when a graph does not fit the last plan. `GraphBuilder` called
it directly, so a graph built after a larger one of the same topology reused the larger one's offsets:
the same graph had another layout depending on what ran before it. The CPU backend's conv fusion
(ggml-0036) reads that layout: it falls back to the unfused nodes when its result partially overlaps its
input, and the fused kernel starts its sums at the bias where the unfused ADD adds it after, so the two
paths round differently. Park mode and per-stream mode ran the graphs in different orders, so they
landed on different layouts.

## Takeaway

**A cached allocation plan is history; a kernel that looks at addresses turns history into numbers.**
Each was reasonable alone: gallocr's reuse saves a planning pass, and the fusion's overlap check keeps
it correct under any layout. Together the result depended on the previous call. `GraphBuilder::build` now
calls `ggml_gallocr_reserve` for every rebuild, so the offsets are the graph's own (the buffer still only
grows). `test_graph_layout_is_its_own` builds a shape before and after a larger one and checks every
node's offset: 12 of 22 move without the fix. The cost is a planning pass per rebuild and none per reuse;
Soprano's one pass took 2.86 s with it and 3.00 s without (five interleaved runs each), and Soprano's stream
stayed bit-identical to `infer`. The hybrid path (`ggml_backend_sched`) reuses its plans
the same way and is not changed; it is in the hub.

**Bisect a numeric difference by the order of calls before the code.** "Same input, different history"
was the experiment that found it: it took the streams, the caches and the driver out of the
question at once, and the fusion switch then named the kernel.
