---
type: adr
status: accepted
date: 2026-08-07
tags: [kv-cache, graph-reuse, attention, memory, scope]
---

# ADR-016: The KV Cache Is Single-Sequence, F32, and Addressed by an Index Tensor

## Context

A cached attention block has to reach the engine from an export
([`docs/KV-CACHE.md`](../KV-CACHE.md) is the full design). Two questions came with it: how a write is
addressed, and how much of a general cache to build before a model needs it.

The addressing question turned out to gate something bigger. `n_past` was baked into each layer's
`ggml_view_2d` write offset, so consecutive decode steps differed *in the graph* — which meant
autoregressive decode could never reuse a retained graph, no matter what `n_kv` was rounded to.

## Options Considered

1. **Offset-in-the-graph writes.** Simplest; makes every decode step a distinct graph.
2. **Bucketed `n_kv` alone.** Rounding `n_kv` up to a boundary was the original plan, and it would not
   have worked, because the write offset was the thing that varied.
3. **Make the write destination *data*.**

## Decision

* **Writes are addressed by a cell-index tensor** (`ggml_set_rows`), so the write destination is data
  rather than graph structure. `n_kv` is bucketed at 32 and the mask padded host-side. A prefill plus
  40 decode steps is **three graphs**.
* **`KvCache::fill_cell_index` is the single place a second addressing policy would go.** Only the
  contiguous append `[n_past, n_past + n_tokens)` is written today.
* **Storage is always F32.** Weight quantization is the exporter's `quantize=` kwarg; KV-cache
  quantization is a separate runtime concern at a different point in the pipeline.
* **Single-sequence.** No ring buffer, no multi-stream, one `kv_size` for every layer.
* **`ATTENTION` always uses the composite path** (`MUL_MAT` → `soft_max_ext` → `MUL_MAT`).
  `ggml_flash_attn_ext` forces an F16 K/V cast that fights exact-fp32 verification.

## Consequences

* **Positive:** graph reuse works for autoregressive decode, which is what made bucketing worth
  anything.
* **Positive:** a second addressing policy has one place to land rather than N.
* **Negative:** no batching, no multi-sequence serving. This engine targets edge devices, where one
  sequence at a time is the normal case; a serving deployment would need a redesign, not an extension.
* **Negative:** `FLASH_ATTENTION` stays unbuilt, and the blocker is **the gate suite's exact-fp32
  comparisons, not the hardware**. A GPU exists now and the trade still has not been made — it is a
  decision about verification. Tracked in [the backlog](../backlog/active-index.md#engine--performance).

## Related

* Authority: [`docs/KV-CACHE.md`](../KV-CACHE.md)
* Epic: [Epic-01: Inference Engine Core](../epics/epic-01-inference-engine-core.md)
