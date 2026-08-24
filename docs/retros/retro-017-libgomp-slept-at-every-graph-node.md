---
type: retro
date: 2026-08-23
domain: performance
tags: [threading, openmp, libgomp, ggml, barrier, scaling, p4.17]
---

# Retro-017: 24 Threads Ran At One Thread's Speed Because libgomp Slept 2789 Times Per Synthesis

## The Issue

VITS on a 24-core Core Ultra 9 285K got *slower* past 8 threads and at 24 was exactly as slow as at 1
(0.191 s against 0.198 s), while onnxruntime on the same box kept improving across the whole range.
The causal LM did the same thing: 21.8 tok/s at 4 threads, 10.6 at 24.

## Root Cause Analysis

**loom built ggml with `GGML_OPENMP=ON` and had never chosen to.** It is ggml's own default
(`ggml/CMakeLists.txt:245`) and loom's CMake did not mention OpenMP at all.

In that configuration `ggml_barrier` is `#pragma omp barrier`, and ggml runs one after **every
non-empty graph node** — 2520 of them in one VITS synthesis. **libgomp's default wait policy makes
every thread sleep on a futex at every one.** Over 5 syntheses at 24 threads that is **334,609
voluntary context switches**, against **160** with OpenMP compiled out: 2789 sleeps per thread per
synthesis, one per graph node plus the few ops that barrier internally.

The cost grows with thread count and shrinks with work-per-call, which is why the per-op profile
looked like "every op degrades and the cheapest degrade worst" and why the curve inverted rather than
flattening.

## Resolution & Lesson Learned

`GGML_OPENMP=OFF` — ggml's own threadpool spins a bounded ~6.5M rounds (`poll = 50`) and then sleeps,
so it spins through a graph's barriers and still sleeps between inferences. **0.189 → 0.040 s at 24
threads, 4.8x**, monotonic curve restored, nothing regressed at any thread count.

**ggml upstream knows.** `ggml-cpu.c:4114` sets `KMP_BLOCKTIME=200` at init "to wait before sleeping
a thread", right below a **commented-out** `setenv("OMP_WAIT_POLICY", "active", 0)` whose own comment
reads "so that OpenMP threads don't sleep". `KMP_BLOCKTIME` is Intel/LLVM `libomp`'s knob and GNU
`libgomp` ignores it, so **every gcc-built ggml gets no mitigation** — and the commented-out line could
never have worked, because libgomp reads `OMP_WAIT_POLICY` in its load-time constructor, before
`ggml_init`. The bug is upstream's, half-fixed for one OpenMP runtime.

**Four lessons, in the order they cost time:**

1. **Check which threadpool is actually compiled in before reading any of it.** The scoping named
   "ggml's CPU threadpool spins then sleeps" as candidate (3) and pointed at `threadpool->poll` and
   `ggml_graph_compute_poll_for_work`. That code is inside `#ifndef GGML_USE_OPENMP` and is **dead in
   every build loom ships**. A round of source reading went into a threadpool that does not run.
2. **The cheapest experiment was ranked last and was the whole answer.** The scoping put barrier cost
   and `n_tasks` clamping first — "the same fix from different ends and the likely answer together" —
   and (3) last as "cheapest to rule out". (3) was it, alone. **Order candidate experiments by cost to
   run, not by how convincing the mechanism sounds**; this is the fifth time on this thread that
   reasoning from source lost to a measurement.
3. **A coincidence that fits the theory is not evidence for it.** The box is 8 P-cores + 16 E-cores
   and the curve peaked at exactly 8. That is a compelling story and it is wrong — `OMP_PROC_BIND`
   changes nothing once spinning is on. The number matched for an unrelated reason.
4. **Count context switches.** One `getrusage` call separated "per-op synchronisation, probably" from
   "every thread sleeps once per graph node, here is the count". `perf` was not installed and was not
   needed.

**What this does NOT change:** a Pi 4 at its 4 cores is unchanged — `GGML_OPENMP=OFF` against `ON`,
same source tree, arms interleaved with cooling gaps, three rounds: **1.011 / 1.015 / 0.997**. The
whole barrier bill there is a couple of ms of a 1.1 s synthesis. **This is a many-core fix**, and the
thing worth checking was the downside rather than the upside: ggml's threadpool *spins* the workers
libgomp slept, and on a 4-core box three spinners could have starved the main thread. They do not. The default thread count is
still 4 and raising it remains its own question with its own benchmark.

**What NOT to re-propose** (adding to
[Retro-012](retro-012-optimizations-that-were-measured-out.md)):

| idea | verdict |
|---|---|
| clamping `ggml_get_n_tasks` by work size so small elementwise ops use fewer threads | **not the mechanism.** Nothing was clamped and the curve was fixed by the wait policy alone. No ggml patch needed |
| a cheaper `ggml_barrier` | real but a fifth of it: 2520 barriers × 11.4 µs = 28.7 ms of a ~124 ms delta. The rest is threads sleeping *inside* ops |
| `OMP_PROC_BIND` / `taskset` pinning | no effect once spinning is on; `bind=close` without it is far worse (0.456 s at 24 threads) |
| shipping `OMP_WAIT_POLICY=active` | fastest arm (0.031 s) and unusable: libgomp reads it in its load-time constructor, so a library cannot set it for its host, and it spins forever between inferences |

## Full record

Measurements, the microbenchmark, and the reason the fix is the build flag rather than the environment
variable are in [Epic-05 §2](../epics/epic-05-edge-performance.md).
