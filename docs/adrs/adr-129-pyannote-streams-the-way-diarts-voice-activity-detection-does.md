---
type: adr
status: accepted
date: 2026-11-15
tags: [engine, streaming, audio-in, vad, pyannote, diart, frame-stream]
supersedes: []
---

# ADR-129: pyannote Streams the Way diart's Voice Activity Detection Does

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The frame door of
[ADR-106](adr-106-a-frame-model-streams-rows-through-a-door-of-its-own.md) for pyannote-segmentation-3.0,
the last audio-in row of the survey.

## Context

pyannote-segmentation-3.0 answers seven powerset classes per frame over 10 s chunks: no speaker, each of
three LOCAL speakers, each pair. pyannote.audio has no streaming mode. Its sliding-window inference and
its pipelines are offline. Speaker 1 of one chunk is not speaker 1 of the next, so nothing about who is
speaking carries across chunks without a speaker-embedding model to link them.

**diart** (github.com/juanmc2005/diart) is the streaming reference. It offers two pipelines:

* **`SpeakerDiarization`**: segmentation, an embedding model (pyannote/embedding or WeSpeaker), and online
  clustering into global speakers.
* **`VoiceActivityDetection`**: segmentation alone.
  * 5 s chunks every 0.5 s.
  * The hard powerset -> multilabel mapping, then the max over the three local speakers.
  * Each output region aggregated over the last `latency / step` chunks, each weighted by a Hamming window
    over its own chunk.
  * Turns where the score exceeds tau 0.6, joined across gaps under 0.05 s.

Several of its behaviours had to be read off the code, not its docs:

* **Window crops are floating-point arithmetic.** A crop is `ceil((start - res - buffer_start) / res)` frames
  in, with `res` = chunk duration / 293. The first output therefore has 293 rows where exact arithmetic says
  294, and starts with a repeat of frame 0.
* **The first output is prepended back to 0** (`_prepend`): the first chunk's own rows, with the aggregated
  tail written over them. That array is float32, so those rows are float32-rounded.
* **A chunk that is not a whole number of steps** (4 s at 0.3 s) is first emitted one step late.
  Its start is stamped at `step` although its audio began earlier, and nothing before 4.0 s is ever
  answered.
* **A file is padded**: `latency - step` of zeros after the audio, enough zeros before a short clip to fill
  one chunk, and the last block zero-filled to a step. Every time then moves back by the leading zeros.

## Options

1. **diart's `SpeakerDiarization`**: needs a new embedding export and a two-model composite with a turn
   output. The most useful, and much the largest.
2. **diart's `VoiceActivityDetection`**: segmentation alone; diart's own code is the oracle.
3. **Not streamable**: the seven classes mean nothing across chunks.

## Decision

Option **2**, the user's choice.

* **The driver** (`diart_vad_stream_driver/00_stream.lua`, beside the unchanged synthesized `infer`) ports
  diart operation by operation, in the same float64 arithmetic:
  * `rearrange_audio_stream`'s scan;
  * `VoiceActivityDetection.__call__`: `infer` on each chunk, then first-maximum argmax on the
    probabilities;
  * pyannote.core's `SlidingWindow.crop` (`loose`, `fixed`) and the out-of-range edge repeat;
  * the Hamming aggregation and `_prepend`. Its float32 rounding is not restated: the door hands rows
    back as float32, and rounding twice is rounding once. A sabotage dropping a port of it stayed green,
    so the port was removed;
  * `get_file_padding` in `stream_finish`. With leading zeros nothing has been answered yet, so the stream
    is re-run behind them.
* `stream_open` takes `duration_seconds` (5), `step_seconds` (0.5) and `latency_seconds` (the step). It
  refuses a latency outside [step, duration].
* **The rows are a speech score, not `infer`'s classes.** The export declares `loom.stream.labels`
  (`["speech"]`), and `FrameStream` answers with those labels when present.
* **Rows on a grid of their own.** Each call's region has its own resolution: 30 rows over 0.5 s, and
  293 over the first 5 s. The driver states every row's start and duration in state `row_times`.
  `FrameUpdate` now carries `row_start` / `row_duration` for every stream, from the frame grid when the
  driver states none. loom-py's `AudioClasses.starts` carries them, and `.times` reads them.
* **diart's `Binarize` and its 0.05 s patch are the host's**, as a VAD's thresholding is for every frame
  model. The gate ports them to show the turns are diart's.

## Consequences

* **The stream is diart.** `test_e2e_diart_vad_stream` compares against
  `scripts/diart_vad_reference.py`. That script runs diart's own code (clone 392d53a) with its RxPY
  plumbing stubbed by a recorder, so diart's `accumulate` scan, filters and map run unchanged.
  * 35 cases: 5 clips (3-60 s, two under 5 s, one of four voices with gaps) at 5/0.5 s (latency 0.5, 2
    and 5), 10/1 s (1, 3) and 4/0.3 s (0.3, 1.2).
  * Every row equals diart's score rounded to float32; every row's start is exact; every speech turn is
    diart's (7193 checks). Not one argmax flipped over ~1,300 chunks.
  * 20 ms pushes give the same rows bit for bit. Two streams with different settings interleaved with a
    one-pass call each answer as alone.
  * The published file's one pass is unchanged, and the door refuses that file.
* **Sabotage** (same-length byte patches of the driver in a GGUF copy, on the 21 jfk and short-clip
  cases): each of these turned the gate red:
  * no first-output prepend;
  * Hamming weights made flat (a mean);
  * the crop's index shifted by a frame;
  * no padding before a short clip (row count wrong);
  * no padding after the audio;
  * the wrong class read as non-speech;
  * the buffer never pruned;
  * the start time stepped by 0.5 s whatever the step (caught on the 0.3 s and 1 s cases).

  Rounding a half down instead of to even stayed green: no setting puts a value on a half, as in ADR-126.
* **Cost: one pass over the whole chunk every step.** On the dev box (Ryzen 3 3250U, 2 cores / 4
  threads), 60 s at 0.5 s steps, each run alone:

  | threads | real time | RTF |
  |---|---|---|
  | 1 | 3.9x | 0.25 |
  | 2 (the engine's default, physical cores) | 2.5x | 0.40 |
  | 4 (the test helper's ggml default) | 0.8x | 1.21 |

  The BiLSTM is ~2,300 tiny per-step graphs per chunk, so threads add barrier cost and no work.
  Two engine processes oversubscribing the cores were 40x slower still: a 0.25 s pass took 10.8 s
  beside a 4-thread stream.
* **`FrameUpdate::values` is float32**, so a host sees diart's scores to float32. Thresholding at 0.6
  could differ only for a score within float32 rounding of 0.6.
* Publishes with the rc21+ engine, after a re-export.
* **Out of scope:** diart's `SpeakerDiarization` (option 1). It needs an embedding model in the zoo first.
