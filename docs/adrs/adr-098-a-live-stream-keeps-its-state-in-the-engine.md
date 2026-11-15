---
type: adr
status: accepted
date: 2026-11-08
tags: [streaming, live-api, stream-state, lua-bridge, driver, asr, zipformer, backward-compatibility]
supersedes: []
---

# ADR-098: A Live Stream Keeps Its State in the Engine

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Builds on
[ADR-097](adr-097-a-streaming-checkpoint-runs-its-chunk-loop-behind-the-one-pass-door.md), which ran
Zipformer streaming's chunk loop behind the one-pass door and left the live door as a hub item.

## Context

Every door takes a whole input and returns a whole output. A live door has the host push audio as it
arrives and read the transcript as it grows, so something must survive between the host's calls: a
transducer's encoder caches and decoder output, a feature cursor, audio not yet consumed. Two questions:
how a driver hands back a piece mid-run, and where the state lives between calls.

## Options

For the hand-back:

1. **Coroutines.** The driver's entry runs as a Lua coroutine; `loom.emit` yields a piece and a
   `loom.next_input` yields until the host pushes more. One driver serves both doors.
2. **Entry points.** The driver exports `stream_open` / `stream_feed` / `stream_finish` beside `infer`.
3. **A host callback** from `loom.emit`. Covers output only; audio in still needs 1 or 2.

For the state:

a. **Lua locals** captured by the three functions. Free, but it lives in the bridge's one VM: one stream
   per loaded model, a one-pass call in between clobbers it, and the host can neither see nor reset it.
b. **An engine-side store per stream**, beside `KvCache`, `ConvStateCache` and `OutputStore`.

## Decision

**2 and b** (the user, 2026-11-08: streaming is a separate door, and files already exported keep
working exactly as they do; the state belongs on the engine side, as output buffers already do).

* `StreamState` (`include/loom/core/stream_state.h`) holds named **tensor** entries -- `OutputStore`s
  on the device, filled by `loom.keep_state(name, module)` (a backend-to-backend copy of the module's
  retained outputs) and read by an input reference `{state = name, index = k, row, rows}` exactly as
  `{from = module}` is -- and named **host values** (`loom.state` / `loom.set_state`: numbers or flat
  arrays).
* `LoomLuaBridge::call(fn, args, state)` runs a driver function against a host's state. The plain
  `call(fn, args)` hands each call a fresh state of its own, so `infer` can be `stream_open`, one
  `stream_feed` with the whole clip, and `stream_finish`: the one-pass gate covers the live code.
* `TranscriptionStream` (`include/loom/core/transcribe_stream.h`) is the per-task door: `push(samples)`
  returns the ids a piece completed, the transcript so far and the text it added; `finish()` flushes;
  `reset()` reopens. It refuses a file whose driver lacks the three functions.
* A copy, not a hand-over, on `keep_state`: a module's cached graph ends in a `cpy` into its store's
  tensors, and swapping that store away is the dangling destination `OutputStore::layout_epoch`
  guards against. Reads are not copies: a referenced input is aliased in place per call (ADR-073).

## Consequences

* **Backward compatible both ways.** A file without `stream_*` runs one-pass as before and is refused
  only by the live door. A new file on an older runtime still has `infer`, but Zipformer streaming's
  calls `loom.keep_state`, so that file needs this runtime -- it publishes on rc21 anyway (ARGMAX).
* Streams on one loaded model are independent: weights and compute buffers are shared, state is not.
  The gate pushes two streams in alternation with a one-pass call between their pushes. Sabotaged to read
  the module's own retained outputs instead (`{from = ...}`), every single-stream check still passed and
  that one failed on all three clips (299/305): it is the check that sees a leak.
* The driver functions keep nothing. A host can drop a stream, reset it, or report its size
  (`StreamState::tensor_bytes`, `value_bytes`).
* The front end is chunked in the driver, not in the engine: Zipformer runs its fbank phase per chunk
  over a window starting one hop before the chunk's first frame, which gives the clip's own frames under
  kaldi's `snip_edges=False` reflection. No topology changed.
* The one-pass path keeps no whole-clip feature matrix any more.
* Not covered yet: tokens out for LMs, audio out for TTS, and the other ASR families. Each is a driver
  with the three functions (a re-export); the epic's survey says which.

## Related

* `include/loom/core/stream_state.h`, `transcribe_stream.h`; `loom_cli --wav ... --stream MS`
* `loom-exporter/loom_exporter/zipformer_streaming_driver/00_stream.lua`
* [ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md),
  [ADR-073](adr-073-a-referenced-input-is-read-in-place.md)
