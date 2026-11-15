---
type: adr
status: accepted
date: 2026-11-08
tags: [streaming, live-api, stream-state, kv-cache, conv-state, lm, generation, backend]
supersedes: []
---

# ADR-099: A Stream Leases the Model's Caches, or Owns Its Own

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Extends
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md), whose `StreamState` keeps a stream's
tensors and values apart from every other stream's, to the two caches it did not cover.

## Context

ADR-098's first model, Zipformer streaming, carries its history as retained outputs, which a
`StreamState` copies into its own stores. Most of the other models in the survey carry it in the
model's `KvCache` (every LM, Kyutai STT, Moonshine streaming, the AR TTS models) or `ConvStateCache`
(LFM2's ShortConv layers). The session allocates one of each from the file's hparams and lends it to
every module, so two token streams on one model -- or a stream and a one-pass `generate` between its
pulls -- would write the same cells.

A first door also had to exist for tokens out: an LM's loop runs inside one `infer_with_past` call
and returns the whole generation.

## Options

For the caches:

1. **Park.** One cache per model, leased to one stream at a time. When a different stream (or a
   one-pass call) runs a module on it, the owner's written rows are copied to the host and the
   newcomer's parked rows, if any, are copied back. One stream alone copies nothing.
2. **A cache per stream.** Nothing is copied; every stream allocates a full cache.
3. **One stream at a time.** Refuse a second stream, or a one-pass call while one is open.

Measured on the dev box (4 cores, a park plus a restore at ~8 GB/s), against warm decode steps:

| Model | KV per position | Switch at 200 / 1000 positions | Decode step |
|---|---|---|---|
| Qwen3-0.6B | 224 KiB | ~11 / ~57 ms | 155 ms |
| SmolLM2-360M | 80 KiB | ~4 / ~20 ms | 93 ms |
| Gemma-3-270M | 36 KiB | ~2 / ~9 ms | -- |
| LFM2-350M (+80 KiB conv state) | 24 KiB | ~1 / ~6 ms | -- |
| Kyutai STT (750-row ring) | 256 KiB | up to ~48 ms | 80 ms frame |

A cache per stream instead costs the file's whole `kv_cache_size`: 896 MiB per stream for Qwen3's 4096.

## Decision

**1 and 2, chosen per cache by a setting** (the user, 2026-11-08: lease and park on CPU and integrated
GPUs, a cache per stream on CUDA and Metal).

* `StreamCaches { Auto, Park, PerStream }` on the bridge (`LoomLuaBridge::set_stream_caches`; loom_cli
  `--stream-caches`; loom-py `model.stream_caches`). `Auto` resolves per cache from its backend's ggml
  device type: `GPU` gets a cache per stream; `CPU`, `IGPU` and `ACCEL` park. On a discrete GPU a park
  would have to stay in VRAM to avoid a PCIe round trip, so it would cost what a cache of its own costs
  and still copy. Metal reports `GPU` on unified memory and gets a cache per stream too, as asked.
* `LeasedCache` (`cache_lease.h`) is the base of both caches. The owner is held **weakly**, through the
  stream's parking slot: a dropped stream leaves nothing to park into, and a new stream at its address
  cannot be mistaken for it.
* A KV park copies rows `[0, rows_used)` per layer, where `rows_used` is a high-water mark every cell
  write raises (the graph's cell index, `set_rows`). Rows past it are not read unmasked: a bucketed
  `n_kv` reads them under the mask, as it already read earlier generations' rows. A conv-state park
  copies every layer's window (80 KiB for LFM2); a step at `n_past == 0` reads the zero slot.
* The cache a module runs on is resolved per call (`LoomLuaBridge::stream_cache`), and handed to its
  builder (`GraphBuilder::set_caches`), which drops its retained graph when the cache's **serial**
  changes. A one-pass call's state always leases, so it costs nothing extra in either mode.
* **A seeded stream draws from a generator of its own** (`StreamState::seed_rng`), swapped in for each
  call, so a sampled stream gives its seed's sequence however other streams interleave with it.

For tokens out, the door is `loom::text::GenerationStream` (`text_generate.h`). A driver whose loop runs
inside one call exports `stream_open` (prefill, first token), `stream_feed{max_tokens}` and
`stream_finish` beside `infer_with_past`, which stays byte for byte as it was. The cursor, count, last
token, ceiling, eos and sampling knobs are state values, and the driver sets the value `done` when its
own loop would have stopped, so the stop rule is written once, in the exporter's `PrefillDecodeLoop`. A
one-token driver (`infer` only) is streamed by the door's own loop, with no re-export.

## Consequences

* Qwen3-0.6B-Base, SmolLM2-360M-Instruct, Gemma-3-270M-IT and LFM2-350M-monolithic, re-exported: 77/77
  each. `GenerationStream` gives `generate`'s ids greedy at 1, 5 and 48 tokens per pull, and sampled with
  a seed; two streams pulled in alternation with a one-pass `generate` between their pulls each give
  their own `generate` ids, in both modes (LFM2's parks its conv state too); each published file is
  refused. loom-py's gate does the same through `text2text.stream()`.
* Streaming costs nothing alone: Qwen3, 64 tokens, 11.9 s streamed one per pull against 12.0 s one-pass,
  the first token after 0.23 s. Two Qwen3 streams alternating on the CPU, 32 tokens each, measured end
  to end: at ~10 positions Park is 5% FASTER than PerStream; at ~410 it is 6.9% slower switching every
  token, and even (-0.6%) switching every 8. Ids identical in every run. PerStream's cost on a switch is a graph rebuild -- the builder
  retains one graph, and it points into one cache -- which is what Park avoids at short contexts.
* Sabotaged, each mode fails only its own check. With the lease made a no-op, Park mode's two streams
  both differ and no rows are ever parked, while PerStream passes (72/75). With each stream's caches
  allocated but never given to the builder, PerStream's two streams differ while Park passes (73/75).
* A file exported before this has no stream functions: `generate` runs it as before, and the door
  refuses it with a `LoadError` naming the re-export.
* Not covered yet: the loops that carry per-call context (`bound` encoder output, a built prompt, a
  transcript marker), which are Whisper's, T5's and the speech LMs'. `PrefillDecodeLoop.stream_supported`
  names the field, and those loops export `infer_with_past` alone until a stream state holds it.
* lfm2-350m-modular's per-layer export carries no caches, so its one-token loop re-feeds the whole
  sequence every step. It streams, but each token costs more than the last.

## Related

* `include/loom/core/cache_lease.h`, `stream_state.h`, `text_generate.h`; `tests/ci/test_cache_lease.cpp`,
  `tests/gate/test_e2e_token_stream.cpp`
* `loom-exporter/loom_exporter/driver_components.py` (`PrefillDecodeLoop.stream_*`, `DecodeLoopStream`)
