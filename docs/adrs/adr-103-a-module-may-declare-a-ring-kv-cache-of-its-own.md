---
type: adr
status: accepted
date: 2026-11-09
tags: [engine, kv-cache, ring, streaming, audio-out, codec, moss-tts, dia]
supersedes: []
---

# ADR-103: A Module May Declare a Ring KV Cache of Its Own

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Extends
[ADR-066](adr-066-a-uniform-sliding-window-is-a-ring-kv-cache.md) (the ring) and
[ADR-023](adr-023-a-second-stream-is-declared-not-derived.md) (a private cache), for the codec pairs
[ADR-102](adr-102-a-codec-pair-streams-as-two-doors-and-a-causal-codec-carries-its-state.md) left: MOSS-TTS and Dia.

## Context

MOSS-TTS's codec (MOSS-Audio-Tokenizer-v2) decodes through six causal transformers and reshapes, no convolution:
1, 2, 4, 8, 16 and 32 positions per 25 Hz code frame, 32 x 1280 and then 5 x 12 x 768 layers, attending to the
last 125, 250 and then 400 positions. Its reference streams with that window per layer. The whole state is the
windows' keys and values, 338 MiB per stream at full context. ADR-102 carries a codec's state as graph inputs and
outputs, which here would copy all of it on every push.

The engine's KV cache is the obvious home -- written in place by the fused ATTENTION op, leased or owned per
stream (ADR-099) -- but a file had ONE cache geometry (`loom.*`), one width per layer, and a ring took one token
per call past its capacity (ADR-066), because a longer call overwrites cells its own earlier queries read.

## Options

1. **State as graph inputs and outputs** (ADR-102): no engine change; 338 MiB copied in and out per push.
2. **A module's own cache, declared by its topology**: the topology says the geometry; the engine allocates it
   privately for that module.
3. **One cache with per-layer widths**: a larger change to `KvCache` for one model.

## Decision

**2** (the user, 2026-11-09: "Engine ring caches").

* A topology may carry `"kv_cache": {n_layer, n_embd_k, n_embd_v, size, ring, window}`. `register_topologies`
  gives that module a private `KvCache` of exactly that geometry, whatever the file's `loom.*` keys say. A cache
  of its own is private by construction.
* A ring may serve a WINDOW smaller than itself. Past capacity it then takes calls of up to `size - window`
  tokens: the cells such a call overwrites hold positions outside every query's window, since the queries
  start at `n_past` and see back only `window - 1`. MOSS's rings are each window plus the most positions an
  8-frame push adds (133, 266, 432, 464, 528, 656 cells).
* `loom.causal_mask(n, n_past, window, ring)` masks over the ring's CELLS: past capacity, cell c holds the
  latest position written to it, `last - ((last - c) mod ring)`.
* The exporter: `ExportPhase.own_kv_cache = {ring, window}` writes that object into the phase's topology from
  its own fused blocks (`decomposition.declare_own_kv_cache`), leaves its layer indices at 0.. and keeps it out
  of the file's geometry. MOSS's codec is six such stage phases beside its unchanged one-pass decode
  (`moss_codec_stream.py`); the driver runs them in order per push.

**Dia** needs none of this. Its talker's loop is cut per decoder row like the others (ADR-102), a frame handed
back once every channel's delayed row has been drawn; its codec, DAC, is not causal, but its receptive field is
under 16 frames each side, so the stream decodes windows of the one-pass graph with 16 frames of context either
side (4.1e-07 from one call on random codes, against 5.9e-03 with 8) and runs 16 frames (186 ms) behind the codes.

## Consequences

* Pinned by `test_ring_kv_cache`: a module with its own 6-cell ring for a window of 3 takes uneven calls of up to
  3 tokens past capacity, each row the mean of exactly its window; a 4-token call is refused; the shared cache is
  untouched. With the mask's cell mapping removed, two rows fail (34/36).
* MOSS's streamed stages, in PyTorch, are 2.0e-05 from the one-call decoder on 24 frames pushed 1, 3, 8, 2, 5.
* Re-exported on the workstation (CPU): MOSS-TTS + its codec, Dia + DAC. One pass is unchanged -- MOSS's
  composition gate 61/61 against the reference (codes identical, waveform 5.9e-06), Dia 0/288 codes off
  `transformers` greedy and guided (26/26), the Dia + DAC composition 30/30 (5.6e-08).
* Streams: MOSS's codec on 137 real frames, every stage's ring wrapping, 1.56e-06 from one pass, interleaved
  streams bit-identical to the stream alone in both cache modes (19/19); the MOSS pair 2.5e-07 from decoding
  the codes in one pass (15/15). DAC's windows on 32 frames 2.3e-08; the Dia pair over 40 frames, across window
  seams, 2.9e-07 (15/15).
* `loom_cli --stream 1` on the workstation CPU: MOSS-TTS's first audio after 0.86 s, 9.6 s of audio at RTF
  3.3; Dia's after 6.1 s (15 delay rows before frame 1 is whole, then DAC's 16-frame lookahead), RTF 15.8 for
  the 1.6B talker.
* A file with such a topology needs an engine with this; an older one ignores the key, allocates the file's
  geometry and fails on the first stage.

## Related

* `include/loom/core/graph_topology.h` (`KvGeometry`), `kv_cache.h` (`window`), `src/core/session.cpp`,
  `graph_builder.cpp` (`real_n_kv`), `lua_bridge.cpp` (`l_causal_mask`); `tests/ci/test_ring_kv_cache.cpp`
* `loom-exporter/loom_exporter/moss_codec_stream.py`, `codec_stream_export.py`, `decomposition.py`,
  `codec_stream_driver/`, `moss_tts_driver/00_stream.lua`, `dia_driver/00_stream.lua`
