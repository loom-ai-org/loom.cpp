---
type: adr
status: accepted
date: 2026-11-11
tags: [engine, streaming, audio-in, vad, classifier, embedder, silero, lstm]
supersedes: []
---

# ADR-106: A Frame Model Streams Rows Through a Door of Its Own

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Uses
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md)'s three driver functions and
[ADR-104](adr-104-a-recurrence-carries-across-pushes-and-noise-belongs-to-its-frame.md)'s carried recurrence,
and is the live half of [ADR-062](adr-062-a-classifier-says-how-many-answers-it-gives.md)'s
`classify` / `embed`.

## Context

Silero VAD is the first audio-in row of the survey that is not ASR. Its reference streams by construction:
upstream calls the JIT once per 512-sample frame with the previous frame's last 64 samples in front and
carries an `LSTMCell` state. The export runs the same arithmetic over the whole clip: `frontend` (a
stride-512 convolution with a 64-sample left pad), one `run_recurrent` sweep, and `head`.

The four doors built so far each hand back something else: ids that become text (`TranscriptionStream`),
ids (`GenerationStream`), samples (`SpeechStream`, `CodecStream`). A VAD hands back one row of label
probabilities per frame. MarbleNet (sliding window), WakeHuBERT (frame embeddings) and pyannote (rolling
windows) are later rows with the same shape of answer.

## Options

The door:
1. **Reuse `TranscriptionStream`.** It calls `stream_feed({waveform})` already, but it turns the answer into
   token ids and text.
2. **A raw stream that returns flat numbers** and lets each host cut them into rows. Two hosts cutting on
   their own is how they come to disagree about which frame is which, the reason `classify` cuts in the
   engine (ADR-062).
3. **`FrameStream`: audio in, rows out,** cut by the contract the way `classify` and `embed` cut, for any
   frame classifier or frame embedder.

Silero's live path:
1. **A per-frame step topology** with h and c as inputs. That would be a new export phase, and it would make
   one Lua call per 32 ms frame.
2. **`infer`'s own graphs over a window**, with the cell's state carried by `run_recurrent`'s `carry`.

## Decision

Door **3**, Silero **2**.

* `loom::audio::FrameStream` (`frame_stream.h`): `push(samples)` returns a `FrameUpdate` with the rows of
  the frames that push completed, `first_row` and the samples so far. It also has `finish()` and `reset()`.
  Row width is the labels' count or `loom.output.embedding_dim`. A row's time is its index plus the declared
  `frame_rate` / `frame_offset`. An answer that is not whole rows throws with both numbers. A clip model, a
  contract that is not audio in, and a file without a declared width or rate are refused by name. A clip
  answer needs the whole clip, so its door stays `classify`.
* Silero (`silero_driver/00_stream.lua`): a push of k whole frames is one `frontend` call over a window that
  starts one whole frame back (`lead`, zeros at first), so 512 (k + 1) samples. The 64-sample left pad then
  puts every later row on the clip's own frame grid, and the lead's own row is dropped. One `run_recurrent`
  continues h and c, and one `head` call follows. Samples short of a frame wait in `pending`. At
  `stream_finish` they become the last frame, zero-padded on the right by `frontend` as one call pads the
  clip. `infer` is unchanged.
* `loom_cli --wav x.wav --stream MS` on a frame classifier prints each change of the most probable label
  when the push that revealed it returns. loom-py's `speech2class.stream()` and `speech2embeddings.stream()`
  return a `FrameStream` whose pushes are `AudioClasses` / `FrameEmbeddings`, with times measured from the
  start of the stream.

## Consequences

* **Silero streams exactly.** On jfk (344 frames, the last one partial), pushes of 1, 160, 512 and 1000
  samples are 1.9e-06 from one call. The whole clip in one push is 3.0e-07. The difference is in the
  summation order: a push's `frontend` runs over fewer frames than one call's, so its matmuls sum in
  another order. Against the JIT run frame by frame (upstream's own loop), the stream is 3.3e-06 and one call
  is 4.4e-06. Two streams alternated, with a one-pass call between them, each give bit-identical rows to
  that stream run alone. The file exported before the door answers one call bit-identically and is
  refused. Sabotage (carry dropped): 0.98 at every piece size. At 32 ms pieces it runs 46x real time (RTF
  0.0216) on the dev box, with 6.1 KB of stream state.
* The cost of the lead is one extra frame of `frontend` per push, which is double that phase when pushes
  are one frame each. Option 1 would remove it at the price of a new phase. Measured, the stream is still
  46x real time, so it stays.
* **The door is ready for the other frame rows.** MarbleNet and WakeHuBERT need only a driver, and pyannote
  needs its windows. Each one is a re-export.
* A file exported before this ADR keeps its one-pass doors. Publishing the live path means a re-export on an
  engine that has `FrameStream`, which is rc21 or later.
