---
type: retro
date: 2026-08-30
domain: performance
tags: [ggml, threading, profiling, measurement-hygiene, p4.25, p4.27]
---

# Retro-023: A Bench Whose Graph Was The Treatment, And A Profiler That Made The Symptom

## The Issue

Three items were built on one belief: that ggml gives `TANH`, `SIGMOID` and `EXP` one thread while
`GELU` and `SILU` get all of them, so VITS's WN gate ran on one core with three idle.

* **P4.16** measured the gate at "30.4 ms per synthesis at one thread and 30.4 ms at four" and called
  it the one thing in VITS not against a roofline.
* **P4.25** wrote the patch, measured the op at **3.92x** through ggml's own threadpool, predicted
  26 ms of a 1130 ms synthesis, measured the model at **1.005x** over twelve paired rounds, and
  correctly refused to ship it.
* **P4.27** was opened to find where the 26 ms went, with ggml's threadpool sleeping between nodes as
  the named suspect. It would have been a real piece of work: that mechanism would tax every threaded
  node in every model.

There was no missing 26 ms. The gate was threaded before the patch, during it and after it.

## Root Cause Analysis

**`ggml_get_n_tasks` does not decide whether a node threads. It decides whether a graph does.** In
ggml v0.19.0 it is called in exactly one place, `ggml_graph_plan`, where it sizes the work buffer and
feeds `max_tasks` — and then `cplan.n_threads = MIN(max_tasks, n_threads)`. There is no per-node thread
count at all: `ggml_graph_compute_thread` runs every node on every thread with `params.nth =
cplan.n_threads`, and an op that must stay serial says so itself (`if (params->ith != 0) return;`).
`apply_unary_op` does not say so — it splits over rows like anything else.

So `n_tasks` is a graph-level clamp that only bites when **no** node in the graph declares more than
one task. Two things sat in exactly that hole:

1. **`scripts/bench18.cpp`'s graph is 256 `TANH` nodes and nothing else.** Unpatched it plans one
   thread; patched it plans `n_threads`. Its "1 thread against 4" was really "this graph cannot thread
   against this graph can", and the 3.92x is a property of the bench. Adding **one** `MUL_MAT [32x8]`
   — 0.03% of the work — makes the unpatched graph plan 4 and the `TANH` nodes run 3.94x faster
   (`scripts/bench20.cpp`, and the same on a 285K at 24 threads and a Ryzen at 2).
2. **`$LOOM_PROFILE` runs every node as a graph of its own** (`profile::compute`,
   `ggml_graph_view(graph, i, i + 1)`), which is the only way to get a per-node hook out of
   `ggml_backend_graph_compute`. So every node whose op declares `n_tasks = 1` is *planned at one
   thread* inside the profile, at any thread count. On VITS at 4 threads that is 122 `UNARY`, 126
   `SUB`, 106 `SCALE`, 42 `SUM_ROWS`, 32 `LEAKY_RELU`, 12 `CLAMP` and 6 `SQRT` nodes. "30.4 ms at one
   thread and 30.4 ms at four" is that, exactly.

The model A/B that closed P4.25 was therefore two identical arms, and 1.005x is the correct measurement
of no change.

## What The Right Measurement Says

The honest way to price something already happening is to take it away. A probe that forces every row
of an `apply_unary_op` onto thread 0 (`LOOM_UNARY_SERIAL=1`), paired ABBA against the real model:

| | rounds | serial / threaded | p10 | p90 |
|---|---:|---:|---:|---:|
| Raspberry Pi 4B, VITS, 4 threads | 12 | **1.025** | 1.016 | 1.037 |
| Core Ultra 9 285K, VITS, 4 threads | 15 | **1.046** | 1.043 | 1.060 |

2.5% of 1115 ms is **28 ms** — against the **26 ms** P4.25 predicted. The prediction was right to
within the board's resolution. It was a prediction of something the model was already collecting.

## Takeaway

**A bench measures its own graph, not the model's.** Before believing an op-level number, ask what
differs between the harness's graph and the model's — not just in shape and thread count, but in what
the surrounding nodes cause the runtime to *do*. Here one unrelated node changed the thread count of
every other node in the graph.

And the sharper, more portable one:

* **A profiler that decomposes a computation can change it.** The per-node floor was already documented
  in `include/loom/core/profile.h`; that is noise with a known sign and a correction column. This is
  different in kind — running node `i` alone made node `i` *single-threaded*. A tool that isolates
  something to measure it must be asked what the isolation itself changes. `ggml_backend_sched`'s own
  eval-callback path has the same property, so this is not a loom-only trap.
* **"It did not scale with threads" is a claim about the measurement first.** The observation that
  survived four days — identical at one thread and four — is exactly what a clamped one-node graph
  produces, and it should have been checked against a whole-graph A/B the moment it was surprising.
* **Price a suspected win by removing it, not only by adding it.** Adding threading that is already
  there measures zero and looks like a mystery. Taking it away measures 2.5% and looks like an answer.

`report()` now prints the caveat next to the per-node floor, and the header of
`include/loom/core/profile.h` carries both failure modes.
