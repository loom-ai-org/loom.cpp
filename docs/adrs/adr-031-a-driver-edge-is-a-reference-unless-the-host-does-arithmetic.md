---
type: adr
status: accepted
date: 2026-09-12
tags: [engine, exporter, drivers, performance, accelerators]
---

# ADR-031: A Driver Edge Is a Reference Unless the Host Does Arithmetic On It

## Context

`output_store.h` has stated the rule since P4.0.12: **marshal only when a value is genuinely
host-side** — a final result, a control decision, or host math the driver actually performs. Retained
outputs and `{from = 'module'}` references are the mechanism, and Whisper, Dia and T5 use them for
their large edges.

Nothing checked the rest. An audit of every driver in the zoo found four that pushed intermediates
through Lua for no host reader, and one binding — `loom.run_recurrent` — that predated the rule and
had no way to obey it.

**On CPU this is nearly free, which is why it went unnoticed.** Measured on EnCodec: the marshalling
round trip is ~0.6% of a decode, and an ABBA comparison of the two builds is inside the noise of a
two-core box. On an accelerator it is not free at all: a Lua-table edge is a device→host→device round
trip, and the values in question are megabytes.

## Decision

**Every driver edge whose consumer is a graph is a retained reference.** A value crosses the boundary
only when the host reads it: a returned result, a loop condition, or arithmetic the driver performs.

Three engine additions make that expressible where it was not:

* **`loom.run_recurrent_and_retain`**, and `sequence` may be a reference. A stacked LSTM is N calls
  with nothing crossing between them.
* **`{from = ..., row = t, rows = n}`** — a retained reference can name a row RANGE, not just a
  prefix. A transducer's joint consumes one encoder frame per call.
* **`loom.output_shape(module, index)`** — a retained output's shape without its data, so a loop can
  be driven by a tensor it never reads.

## What the Audit Found

| driver | crossed | now |
|---|---|---|
| EnCodec | `pre`→LSTM→LSTM→`post` sequences, 7·T·1024 doubles per call | codes in, waveform out |
| transducers (parakeet ×2, gigaam) | the whole encoder output — 1.1 MB for 11 s of audio — plus `embed`, per-layer h/c, `top_h` | one row named per frame; only the first pass's zeroed h/c cross |
| `run_bi_lstm` (5 in Kokoro, 5 in StyleTTS2) | `layer_input`, `h_prev`, `c_prev` in and `h_new`/`c_new` out, per timestep per direction — 4T calls | two C++ sweeps per BiLSTM |
| `run_resblk_stack` (2 per model) | each block's output rebuilt as a Lua table and written straight back — and the two conversions around it are exact inverses | blocks chain by reference; one conversion in, one out |
| Whisper, Dia, T5 | — | already retained when written |

**Two classes stay host-side, and they are not oversights.** A BiLSTM's two directions are
*interleaved* per row by the driver, and Kokoro's and StyleTTS2's duration encoders *concatenate the
style vector into every row* between stages. Those are arithmetic, so the values are genuinely
host-side under the current topology split. Moving them into the engine means changing what the
traced graphs accept — a re-trace per phase, not a driver fix — and it is a real option, priced in
[Epic-03](../epics/epic-03-model-coverage.md) rather than taken here.

## Consequences

* The rule is now enforceable where it was advisory: a driver that marshals an intermediate is visible
  as a `run_subgraph` whose output is bound to a local nothing but the next call reads.
* **Shape checking got stricter for free, and it caught a real bug immediately.** A retained copy
  asserts `ggml_are_same_shape`; marshalling only ever compared element counts. Chaining
  `run_resblk_stack` through the store failed on `[66,512]` vs `[512,512]` — same element count,
  different tensor — which the Lua path would have accepted silently.
* Five models were re-exported and re-gated (parakeet-tdt, parakeet-rnnt, gigaam-v3, kokoro,
  styletts2); the whole 24-model card gate passes, including the ASR-oracle rows that are the only
  real test for the TTS pair.
* **The accelerator benefit is argued, not measured.** The dev box has no GPU; the CPU numbers are a
  wash. What is measured is that nothing regressed and that the crossings are gone.
