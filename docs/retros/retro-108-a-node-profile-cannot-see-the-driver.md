---
type: retro
date: 2026-11-17
domain: performance
tags: [profiling, lua, driver, per-sample-loops, tiny-voca, pi4, measurement]
---

# Retro-108: A Node Profile Cannot See the Driver

Serves [Epic-05](../epics/epic-05-edge-performance.md); found closing tiny-voca's Pi 4 gap
([ADR-134](../adrs/adr-134-tiny-voca-splits-its-pulse-train-at-the-frame-and-draws-torch-noise-in-the-engine.md)).

## Issue

tiny-voca ran at 0.77x onnxruntime on the Pi 4 at F32. `LOOM_PROFILE` listed the graphs' nodes, and the
small nodes of the interleaved rotary looked expensive, so the q/k rows were reordered to make the
rotation two halves. It ran at 664 ms against 667 ms: measured out. The gap stayed open as "a graph
problem".

## Root cause

The profile times graph nodes and nothing else. 44% of the run was in the Lua driver: torch's noise
stream drawn in doubles (202 ms) and the pulse train rendered sample by sample (85 ms), 130k samples
each. A wall timer wrapped around the driver's `loom.*` calls, from Lua through LuaJIT's `ffi` clock,
split the run in one try: graphs 373 ms -- already faster than onnxruntime's whole 516 ms -- and the
rest Lua. The loops were in Lua by an untested premise (the pulse train "needs float64"); measured, a
frame-level split in float32 is 94 dB from upstream. Moved to the graph and the engine, the run is 391 ms,
1.30x.

## Takeaway

**Before optimising a graph, split the wall time into graph calls and driver time.** Wrap `loom.*` in the
driver with a wall clock (`phase_timer.cpp` in the record); the node profile's sum is not the run. A
driver loop over samples is the first suspect on an edge board: LuaJIT pays for every libm call and
float32 rounding per sample, and every table crosses the boundary. A precision argument for keeping one
there is a measurement to make, not a premise.

## Record

* Per-call timer: `LoomLuaBridge::load_script` a chunk that replaces `loom.run_subgraph`,
  `run_subgraph_and_retain` and `get_output` with wrappers timing them by `clock_gettime(CLOCK_MONOTONIC)`,
  then `dofile` the driver extracted from the GGUF (`model.driver_script`) with its local functions
  wrapped the same way. Before: encode 17.8, prosody 66.8, acoustic 160.8, synth 126.7, `aten_randn`
  208.8, `voca_pulses` 85.9 ms (673.9 total, 5 runs). After: 17.3, 65.8, 157.9, 135.2, Lua 15.0 (391.4).
* Standalone LuaJIT on the same Pi ran the same loops faster (146 and 51 ms): driver constants are
  globals, and the engine's Lua state holds large tables. Time inside the engine.
