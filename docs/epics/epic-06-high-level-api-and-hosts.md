---
type: epic
status: active
domain: host-api
last_updated: 2026-08-22
---

# Epic-06: The High-Level API and Its Hosts

## 1. Context and Scope

Two hosts consume the engine: `loom_cli` and `loom-py`. This epic covers the API they present, the
layer each piece of behaviour belongs in, and what a GGUF must declare for a host to dispatch on it
without knowing the architecture.

The authority is [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md); this epic is the orientation.

## 2. Architectural Overview

**One door per task, chosen by the modality pair the file declares** —
[ADR-013](../adrs/adr-013-one-door-per-task.md). X2Y interfaces come off `loom.task` plus the declared
input/output modalities, with canonical input names. A knob with no role in the declared contract stays
`infer`-only.

**The layering rule:** in the **FILE** when it is a property of the checkpoint; in the **ENGINE** when
it is a property of the task; in the **HOST** when it needs the host's ecosystem. Corollary, which
settles the hard cases: anything shipped inside a GGUF can only be fixed by re-exporting every model,
so an evolving policy must not be baked into files even when Lua could express it.

Net: **per-task code may live in any layer; per-architecture code only in the exporter.**

### What the engine owns

| | |
|---|---|
| `include/loom/core/model_contract.h` | the one place that knows the declared KV names. Every reader is absence-tolerant; `declared()` separates a file that states its contract from one a caller must know about |
| `include/loom/core/text_generate.h` | `loom::text::generate` — one LM loop, both driver shapes, the file's own EOS |
| `include/loom/core/session.h` | topologies registered and caches attached once, owned in an order that cannot dangle |
| `transcribe.cpp` | reads the declared ASR table; Whisper's spellings survive only as a flagged legacy fallback |

### Resolution order for anything a model can infer itself

**Optional argument → autodetect from the file → default.** Language is the worked example. Capability
is declared by the **export**, not by the host.

### Compatibility

Pre-contract GGUFs still work through the legacy fallback, and **the fallback must stay tested** — a
declared-only test would stay green while the path every GGUF on disk depends on rotted. The `v4`
fixture set is kept for exactly this reason; `v5` is the contract-declaring set.

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Authority | [`docs/HIGH-LEVEL-API.md`](../HIGH-LEVEL-API.md) |
| Decisions | [ADR-013](../adrs/adr-013-one-door-per-task.md), [ADR-006](../adrs/adr-006-model-constants-belong-to-the-export.md), [ADR-002](../adrs/adr-002-embedded-lua-drivers.md) |
| Retros | [Retro-006](../retros/retro-006-kokoro-shipped-noise.md), [Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md) |
| Active tasks | [Backlog → Host API](../backlog/active-index.md#host-api) |
