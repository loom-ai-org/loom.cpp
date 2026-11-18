---
type: adr
status: accepted
date: 2026-11-18
domain: edge-performance
tags: [performance, cpu, threads, lstm, gru, recurrent, raspberry-pi, paradee]
supersedes: []
---

# ADR-136: A Recurrent Sweep Runs Its Cells on One Thread

Serves [Epic-05](../epics/epic-05-edge-performance.md). Found by Paradee's Pi 4 row
([ADR-135](adr-135-paradee-reuses-kokoros-phases-and-carries-its-voices-as-rows.md)).

## Context

ggml has no LSTM or GRU op, so the engine sweeps a recurrence one timestep at a time
(`LoomLuaBridge::cell_sweep`). Each step builds once (the graph is retained) and computes once: a
`4H x (in + H)` matrix-vector product and a few elementwise ops. Paradee's six BiLSTMs are about 1 480 such
graphs per synthesis. Kokoro, StyleTTS2, AuroraTTS (GRU) and pyannote sweep the same way.

The CPU backend holds no threadpool, so `ggml_graph_compute` creates one for every compute and joins it
afterwards. N threads cost N - 1 thread creations per timestep, for work one core does in microseconds. On
the Pi 4 the whole Paradee synthesis took 1.50 s at four threads, 1.14 s at two and 1.55 s at one (where the
convolutions are slower). onnxruntime took 0.74 s.

## Decision

While a sweep runs, the CPU backend's thread count is `recurrent_cpu_thread_count()`, which is
`$LOOM_RECURRENT_THREADS` or else 1. A scoped guard (`ScopedCpuThreads`) restores the engine's count
afterwards. The guard applies to the primary and the fallback backend, and does nothing for anything that
is not a CPU. A thread count changes no number in these graphs: ggml splits a product's rows across
threads, never its reduction. Kokoro's, StyleTTS2's and Paradee's gates, and Paradee's checksum, are
identical at 1 and 4.

## Options not taken

* **A persistent threadpool for the CPU backend** (`ggml_threadpool_new` +
  `ggml_backend_cpu_set_threadpool`). It would remove the per-compute creation for every graph of every
  model. But it changes the threading of every family, and ggml's pool spins between graphs while the
  driver does host work. That is a separate measurement across the zoo, not a fix for the sweep.
* **The input projection hoisted out of the sweep** (`W_ih x` for every timestep in one GEMM). It saves
  under ~45 ms here (86k multiply-adds a step), changes every LSTM family's exported topologies and numbers,
  and leaves the per-graph cost, which is what dominated.

## Consequences

* Paradee on the Pi 4: 1.505 -> 0.764 s (F32) and 1.293 -> 0.679 s (Q8_0_PC), against onnxruntime 0.97x and
  1.20x where it was 0.49x and 0.63x. Dev box 362 -> 282 ms. The output is bit-identical.
* A board with one core (the Pi Zero) ran every graph on one thread already, so it does not move.
* `test_default_thread_count` pins the knob's parsing and the guard's no-op on a non-CPU backend.
