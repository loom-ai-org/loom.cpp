---
type: adr
status: accepted
date: 2026-05-01
tags: [edge-ai, execution-graph, gguf, ggml, foundational]
---

# ADR-001: The Model Carries Its Own Graph; the Engine Hardcodes None

## Context

A `ggml` engine normally hardcodes each architecture in C++: a new model family is a new C++ file,
and the engine grows without bound. loom targets edge devices, where binary size and cross-compilation
cost are first-order constraints, and the project's stated goal is that adding an architecture should
cost a Python change rather than a C++ patch.

The corollary problem is synchronisation: if a graph definition ships separately from the weights it
describes, the two can drift.

## Options Considered

1. **Per-architecture C++ builders**, as most `ggml` engines do. Simple, fast, and the thing this
   project exists not to be.
2. **A separate topology file beside the GGUF.** Solves the C++ growth, creates a drift hazard and a
   second artifact to distribute.
3. **The topology serialized as GGUF key-value metadata**, alongside the weights it describes.

## Decision

A model is **one GGUF** that carries its own graph topologies as JSON metadata
(`model.graph_topology.<name>`) and its own driver script as embedded Lua, alongside the weights those
describe. The engine parses them and builds the `ggml` graph at run time, through a **symbol table**
(name → `ggml_tensor*`) and a **primitive registry** (JSON op string → `ggml` call).

Dynamic sequence lengths are handled by rebuilding the graph when the length changes — and *only* then:
`GraphBuilder` retains the last graph and returns it unchanged when the axes repeat.

Full design: [`docs/SPECIFICATION.md`](../SPECIFICATION.md).

## Consequences

* **Positive:** adding an architecture is an export. Weights and graph cannot desynchronise, because
  they are the same file. One artifact to distribute. The engine stays small enough for edge targets.
* **Positive:** the same GGUF runs on any backend, which is what makes
  [ADR-007](adr-007-backend-capability-negotiation.md) both possible and necessary.
* **Negative:** errors that a C++ builder would catch at compile time surface at run time, as a parse
  or shape failure. Informative error messages are therefore load-bearing rather than a polish step —
  see [Retro-013](../retros/retro-013-retrofitting-eight-bespoke-converters.md).
* **Negative:** the engine sees shapes, not intent, which is the root of
  [Retro-001](../retros/retro-001-layout-healing-heuristics.md).

## Related

* Epic: [Epic-01: Inference Engine Core](../epics/epic-01-inference-engine-core.md)
* Follows-on: [ADR-002](adr-002-embedded-lua-drivers.md),
  [ADR-003](adr-003-per-model-complexity-in-the-exporter.md)
