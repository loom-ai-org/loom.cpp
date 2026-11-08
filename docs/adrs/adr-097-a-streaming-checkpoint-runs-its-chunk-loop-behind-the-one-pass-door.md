---
type: adr
status: accepted
date: 2026-11-08
tags: [asr, streaming, zipformer, driver, lua-bridge, exporter]
supersedes: []
---

# ADR-097: A Streaming Checkpoint Runs Its Chunk Loop Behind the One-Pass Door

Serves [Epic-03](../epics/epic-03-model-coverage.md) (Zipformer streaming).

## Context

`pkufool/zipformer-small-streaming` is a chunk-causal Zipformer. Upstream decodes it chunk by chunk
(`zipformer inference --streaming 1`): 64 feature frames plus 13 of lookahead per encoder call, with 74 state
tensors in and out (six caches per layer, the subsampler's left pad, a processed-frame count). One `model.pt`
backs three released latencies: chunk 16 / left 64 (320 ms), 32 / 128 (640 ms) and 64 / 256 (1280 ms).

Every streaming model in the zoo so far ships offline: Nemotron 3.5 ASR runs its encoder once over the clip,
Moonshine Streaming windows its encoder over the whole clip, and Kyutai STT runs moshi's frame loop over a
finished clip. loom has no door that takes audio as it arrives.

## Options

1. **One full-clip encoder pass with a chunk mask**, the way icefall simulates streaming in training. Same
   graph shape as the non-streaming leaf, but it is not the computation upstream's `--streaming 1` runs (it
   goes through `forward`, not the `streaming_forward` paths and their caches), so that release could not be
   the oracle. Memory would grow with the clip again.
2. **Upstream's chunk loop in the driver, behind the existing one-pass `speech2text` door.** A fixed-shape
   `encoder_chunk` phase whose states are retained outputs fed back to the next call. No engine change.
3. **A live door now**: a session that takes audio pieces and returns partial transcripts. New per-task
   surface in the engine and loom-py, and each existing streaming model would need a re-export to join it
   (the driver is in the GGUF); Nemotron and Moonshine Streaming would also need incremental encoders.

And for the latency: one file at one setting, one file with all three, or three files.

## Decision

Option 2, at chunk 32 / left 128, in one file (the user, 2026-11-08). The driver is written as `init`,
`feed`, `finish`, and the one-pass door is init, a feed per chunk of the whole clip, finish, so a live door
can call the same three. The live door is a hub item.

## Consequences

* Token for token upstream's released `jit_model-chunk-32-left-128.pt` on 73/73 LibriSpeech-dummy clips.
* Cost is linear in the clip, and the encoder's memory is bounded. The front end is not: the fbank still runs
  once over the whole clip, about 1.5 MB per second of audio at F32, until a live door brings a chunked one.
* The other two latencies are an export constant away (`CHUNK_SIZE`, `LEFT_CONTEXT`) and are not shipped.
* A driver can carry 74 tensors of state through `{from = ..., index = k}` with no new bridge feature; the
  copy stays on the device.

## Related

* `loom-exporter/loom_exporter/zipformer_streaming_export.py`, `zipformer_streaming_driver/`
* [Retro-103](../retros/retro-103-an-argument-the-lowering-did-not-read.md): the strided slice this export found
