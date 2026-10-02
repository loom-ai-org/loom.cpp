---
type: adr
status: accepted
date: 2026-10-02
tags: [engine, kv-cache, attention, sliding-window, streaming, kyutai-stt]
supersedes: []
---

# ADR-066: A Uniform Sliding Window Is a Ring KV Cache

## Context

Kyutai STT 1B is a streaming ASR: one LM step per 80 ms frame, every layer attending to the last 750
positions (`context: 750`), for as long as the audio runs. Kyutai's own `moshi` keeps those 750 keys in
a `RingKVCache`: position p goes to slot `p % 750`, K stored after RoPE.

loom.cpp's `KvCache` was linear: cells `[n_past, n_past + n_tokens)`, and a step reads `[0, n_kv)`. A
window was a banded MASK over that (Gemma 3's sliding layers, `loom.causal_mask(n, n_past, window)`), so
memory and per-step work grew with the whole sequence, and a clip longer than `kv_cache_size` positions
could not be run at all. For Kyutai at F32 that is 256 KB per position: 10 minutes of audio, 2 GB of
cache, of which 750 positions are ever read.

## Options

1. **A linear cache and a duration limit** (Moonshine's 81.9 s precedent). Throws away the model's one
   design point, unbounded streaming, and still costs memory proportional to the clip.
2. **Shift the cache in the driver** when it fills: copy the last 750 rows to the front. Needs a new
   cache-move binding, and every shift moves 750 rows of 16 layers.
3. **A ring, declared by the file**: `loom.kv_cache_ring = true` makes `KvCache` write cell
   `p % kv_cache_size`, and a step past capacity reads every cell. `kv_cache_size` is the window.

## Decision

Option 3. It is exact, not approximate: scores depend on the q-k rotation DIFFERENCE (RoPE is applied
before the write) and softmax on no order, so which cell holds which position does not matter; once
the ring is full every cell is inside the window, and before that the driver's causal mask covers the
written prefix. `GraphBuilder` caps `n_kv` at the capacity (`real_n_kv`), which also places the mask in
the bucket. A call of several tokens that would WRAP is refused: its later rows would overwrite cells
its earlier queries read, since all rows are written before any is read. moshi's own LM steps one token
at a time, as the driver does; a multi-token prefill inside the first `kv_cache_size` positions is fine.

The file opts in; a file without the key is linear, as every file before this one. The header comment
in `kv_cache.h` had named a ring as the second filler this design would take.

## Consequences

* Kyutai STT runs any length in a fixed 750-cell cache (98 MB at F32), id for id with moshi on a 196 s
  clip that wraps the ring three times.
* Only a model whose EVERY cached layer has the same window can use it. Gemma 3 (5 sliding layers per
  full one) stays linear with banded masks.
* Pinned by `tests/ci/test_ring_kv_cache.cpp`: a 4-cell ring whose steps return the mean of exactly
  their window, a multi-token call before the wrap, and the refusal after it; red with the flag off.
* Needs an engine with this (1.0.0-rc14); an older one ignores the key and refuses the first step past
  capacity.

## Related

* [ADR-068](adr-068-kyutai-stt-follows-moshi-not-the-transformers-port.md): the model that needed it
* `loom-exporter/loom_exporter/kyutai_stt_export.py`: the declaration (`kv_cache_ring=True`)
