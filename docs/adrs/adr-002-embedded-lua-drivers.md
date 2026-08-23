---
type: adr
status: accepted
date: 2026-06-01
tags: [edge-ai, scripting, orchestration, control-flow, luajit]
---

# ADR-002: Orchestration Is an Embedded Lua Driver, Not a C++ Driver

## Context

[ADR-001](adr-001-data-driven-gguf-topologies.md) makes a *single-module* network data-driven. Real
architectures — ASR, TTS, diffusion — are multi-modular: they need dynamic pipeline routing between
subgraphs, autoregressive control flow (decode loops, ODE integration steps), and host-side data
wrangling (attention-mask generation, column replication, noise injection, relative-position padding).

That work was done by bespoke C++ drivers — `WhisperDriver`, `VitsDriver`, and one per family. Each new
model meant a new C++ file, which is exactly the growth ADR-001 exists to prevent, one level up.

## Options Considered

1. **A Python C-API runtime.** Too heavy for edge targets, and difficult to cross-compile for embedded
   ARM.
2. **WebAssembly.** Good isolation, but complex host bindings for direct memory manipulation.
3. **Structured control-flow operators in the graph format itself** (the TFLite/ONNX approach: `While`
   and `If` ops pointing at nested subgraphs). Expressive, but it puts a compiler-shaped problem inside
   the runtime and still cannot express arbitrary host math.
4. **LuaJIT embedded in the engine**, with the driver script shipped inside the GGUF.

## Decision

Embed **LuaJIT**. Each GGUF carries a driver script; the engine exposes topologies and host math to it
(`loom.run_subgraph`, `loom.run_subgraph_and_retain`, `loom.argmax_rows`, …) and the script owns the
orchestration. Every model reaches inference through one entry point, `infer`.

**A driver reads no GGUF metadata.** The engine hands it topologies and host math, not hparams — see
[ADR-006](adr-006-model-constants-belong-to-the-export.md) for where numbers come from instead.

Full design: [`docs/LOOM_PROCEDURAL_GENERALIZATION.md`](../LOOM_PROCEDURAL_GENERALIZATION.md).

## Consequences

* **Positive:** minimal binary-size cost; a high-performance FFI straight to C structs. A decode loop,
  a CTC collapse or a transducer's double loop is Lua, not C++ — see
  [ADR-003](adr-003-per-model-complexity-in-the-exporter.md) for what that retired.
* **Negative:** the C++↔Lua boundary has capacity limits that scale with the model rather than the
  code — see [Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md). Reductions belong on
  the side that owns the memory.
* **Negative:** a hand-written `.lua` file gets none of the generated path's validation. Driver
  generation (`driver_ir`, `DriverBuilder`) exists to close that gap.

## Related

* Epic: [Epic-01: Inference Engine Core](../epics/epic-01-inference-engine-core.md)
