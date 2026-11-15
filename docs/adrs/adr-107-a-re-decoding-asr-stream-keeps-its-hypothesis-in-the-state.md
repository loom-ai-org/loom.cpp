---
type: adr
status: accepted
date: 2026-11-11
tags: [engine, streaming, audio-in, asr, moonshine, encoder-decoder, windows]
supersedes: []
---

# ADR-107: A Re-decoding ASR Stream Keeps Its Hypothesis in the State

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Extends
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md)'s transcript door to a model whose
transcript changes as audio arrives, and reuses
[ADR-103](adr-103-a-module-may-declare-a-ring-kv-cache-of-its-own.md)'s windows over a non-causal stack.

## Context

Moonshine Streaming (tiny, small) is the survey's "exact encoder, decoder re-run per update" row. Its
encoder was trained to stream. It uses per-frame CMVN, two causal stride-2 convolutions and attention bands
with no position inside them: `(16, 4)` on the first two and last two layers, `(16, 0)` between. The
position table is added only at the end. Its decoder is an ordinary autoregressive decoder that
cross-attends to every encoder row. So a new row changes what every token sees, and no prefix of the
transcript is ever final.

The transcript door appends the ids each `stream_feed` returns and only grows the text (Zipformer's
emissions are final). transformers' Moonshine Streaming has no streaming loop of its own. The card says
"the current Transformers code path does not yet implement fully efficient streaming", and the paper's
scheme is the encoder incrementally and the decoder re-run.

## Options

The transcript:
1. **Commit a stable prefix** (local agreement, as whisper_streaming does). That rule is ours, not the
   model's, and the text at the end would depend on where the pushes broke.
2. **The driver keeps its whole hypothesis** in the stream's state. The door shows it as the transcript and
   says `revised` when it changes earlier text.

The encoder:
1. **Per-layer caches** (ring K/V of 15 rows, conv tails, layer-by-layer look-ahead): one graph per layer
   per push, and a new export structure.
2. **Windows over the one-pass graph.** The receptive field is finite and position enters only at the end,
   so a window that starts at a row boundary, with the row's index as a new `row_offset` input to the
   position table, gives the clip's own rows. Its left context must cover the receptive field and the row
   must be far enough from the window's right edge.

## Decision

Transcript **2**, encoder **2**.

* `TranscriptionStream`: when the driver has set the state value `hypothesis`, it is the stream's ids after
  every call, and the returned array is ignored. The constructor takes `stream_open` inputs, as
  `SpeechStream` and `CodecStream` do. loom-py passes them as `speech2text.stream(**options)`.
* Exporter: `encoder` takes `row_offset` (the one pass passes 0). `receptive_rows()` reads the margin off
  the checkpoint: 15 rows back per band, plus 3 rows for the convolutions' 12 frames, gives 93 for tiny
  and 153 for small. Each look-ahead band adds 3 rows ahead, so 12 rows (240 ms). At f64 a window with
  these margins is the one pass to 1e-15 on both checkpoints, and one row less on either side is 1e-07 to
  1e-05.
* Driver (`moonshine_driver/03_stream.lua`): rows are encoded only when a decode needs them, every
  `update_seconds` (0.5 s by default). A mid-stream decode teacher-forces the last hypothesis in one call
  and steps on from the first token whose argmax differs. `stream_finish` frames the clip as `infer` does
  (the partial frame zeroed) and decodes token by token from `<s>`, so it ends on one call's transcript.

## Consequences

* **The stream ends on one call's ids.** On 8 clips (jfk, six TTS references resampled to 16 kHz, and a 60 s
  concatenation), both sizes give one call's ids exactly at every cut tried: 20 ms, 100 ms, 0.5 s, the whole
  clip, and a 1.0 s cadence. The final encoder rows are within 7.7e-06 of one call for tiny and 1.0e-05 for
  small (111/111 each). Two interleaved streams, with a one-pass call between them, each end where they end
  alone. The published files decode the same ids in one call, their encoders are bit-identical to these at
  offset 0, and the live door refuses them. Sabotage with `row_offset` fixed at 0 in the windows puts the
  rows 0.74-1.14 off. f5's ids still came out right, so only the tensor oracle catches it there.
* **The cost is the re-decoding.** On the dev box CPU, at 100 ms pushes and the default cadence:

  | model | clip | stream speed | one pass |
  |---|---|---|---|
  | tiny | jfk, 11 s | 2.84x real time (RTF 0.352) | 0.80 s |
  | small | jfk, 11 s | 0.81x real time (RTF 1.234) | 2.64 s |

  Over the whole clip set the stream averages RTF 1.39 for tiny and 3.02 for small. The two speed-ups each
  pay for themselves: encoding every push cost 22x the one pass, and token-by-token
  re-decodes made tiny 0.88x real time. Each update re-decodes all the audio since the stream opened, so
  the cost grows with the stream, and most of the clip set's time is the 60 s clip.
  A live host should open a new stream at each pause, which Silero VAD's `FrameStream`
  ([ADR-106](adr-106-a-frame-model-streams-rows-through-a-door-of-its-own.md)) finds. A decoder step costs
  7-20 ms whatever its length, which is per-call overhead in the engine, and the one pass pays it too.
* `update_seconds` trades how fresh the partial text is against cost, and does not change the final
  transcript.
* Publishing needs a re-export (the encoder's new input and the stream functions) on an engine with this
  door, which is rc21 or later.
