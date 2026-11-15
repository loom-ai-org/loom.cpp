---
type: adr
status: accepted
date: 2026-11-14
tags: [engine, streaming, audio-in, asr, vad, silero, moonshine, granite, lfm2, sensevoice, paraformer, gigaam]
supersedes: []
---

# ADR-127: A Recogniser Without a Streaming Mode Streams in Moonshine's VAD Lines

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The survey's "re-decode" rows
whose upstream has no streaming loop of their own, or a VAD-cut one: moonshine-tiny, granite-speech-4.0-1b,
lfm2.5-audio-1.5b-asr, sensevoice-small, paraformer-zh, gigaam-v3-rnnt. HuBERT and data2vec stream in the
transformers pipeline's chunks ([ADR-126](adr-126-a-wav2vec2-style-ctc-model-streams-in-the-transformers-pipelines-strided-chunks.md));
Whisper in SimulStreaming's AlignAtt (ADR-128).

## Context

What upstream offers for these six:

* **moonshine-tiny**: Moonshine's own library (moonshine-ai/moonshine `core/transcriber.cpp`) streams its
  non-streaming models in *lines*. Silero VAD cuts the audio, and the line still being spoken is
  re-transcribed from scratch every 0.5 s.
* **sensevoice-small, paraformer-zh**: FunASR cuts with its fsmn-vad (lines of at most 30 s) and
  transcribes each line. paraformer's streaming model is a different checkpoint.
* **gigaam-v3-rnnt**: `transcribe_longform` cuts with pyannote's segmentation-3.0, offline (15-22 s
  chunks).
* **lfm2.5-audio-1.5b-asr**: Liquid's cookbook runs fixed 2 s windows with 0.5 s of overlap, each on its
  own, and concatenates them, duplicates included.
* **granite-speech-4.0-1b**: nothing.

Every one of these that has a reference cuts by voice activity, and the zoo already streams a VAD
([ADR-106](adr-106-a-frame-model-streams-rows-through-a-door-of-its-own.md)). Moonshine pins exactly the Silero ONNX
the zoo's silero-vad was exported from (sha256 `1a153a22…`).

## Options

1. **Each model's own upstream**, where it has one: offline VAD for three, duplicated words for LFM2.5,
   nothing for Granite.
2. **A growing window per driver with local agreement**: one file each, but no reference to hold it to.
3. **Moonshine's lines for all six**, composed in the engine from the zoo's Silero and any recogniser's one
   pass.

## Decision

Option **3**, chosen by the user. It is upstream's loop for moonshine-tiny and the same kind of loop for the
other five, with Silero standing in for their VADs.

* **`loom::audio::LineStream`** (`line_stream.h`) composes two loaded models on two sessions: the VAD's
  `FrameStream` and the recogniser's `transcribe`. It is per task, not per model, so it is engine C++,
  not a driver. Neither file changes, and the recogniser needs no re-export: any file `transcribe` takes
  will do.
* **It is Moonshine's loop**, in Moonshine's float arithmetic, with defaults from Moonshine's
  `TranscriberOptions`. Per push:
  * the audio waits until `update_seconds` (0.5) has gathered;
  * the VAD takes it in 512-sample hops, Silero's row per hop;
  * each probability is averaged over a ring of the last 16 (a float sum in ring order), faded over a long
    line, and compared `> 0.5`;
  * a line opens with the last 8192 samples, grows a hop at a time, and closes when voice ends or the
    stream finishes;
  * every line the push touched is transcribed again, the open one included.
* **The fade is upstream's code, not its comment.** The comment says the score reaches 0 at the 15 s
  maximum. The code scales by `(len - 2/3 max) / (2/3 max)`, which is near 0 just past 10 s, so a line of
  continuous speech closes at about 10 s. The port keeps the code's behaviour.
* `max_tokens_per_second` gives Moonshine's per-line budget, `ceil(seconds * 6.5)`, handed to the driver as
  the new `TranscribeOptions::max_new_tokens`. 0, the default, keeps the recogniser's own budget.
* `loom_cli --stream MS --vad silero.gguf` prints each touched line as it changes. Line options go through
  `--input`.

## Consequences

* **The oracle** is `scripts/vad_lines_reference.py`, a line-for-line port of Moonshine's
  `VoiceActivityDetector` and `transcribe_stream`. It runs over onnxruntime and the pinned Silero, with
  transformers for moonshine-tiny's text.
  Installing Moonshine's own `moonshine-voice` package was blocked in this session, so the library itself
  has not been run against it.
* **moonshine-tiny is Moonshine's loop, at every update.** Gated by `test_e2e_vad_line_stream` with the
  reference's texts from transformers under Moonshine's budget (`LOOM_VAD_LINES_TOKENS_PER_SECOND=6.5`): five
  clips (jfk, f5_tts_ref, pocket_tts_ref, a 60 s joined clip and a 48 s clip with its pauses cut out) at
  pushes of 512, 1600 and 8000 samples and the whole clip.
  * 4332/4332 on the dev-box CPU and 4332/4332 on the RTX 5090.
  * Silero's probability within 6.7e-06 of onnxruntime's.
  * Every hop on the reference's side of the threshold (smallest margin in the references: 2.4e-03).
  * Every touched line's place, length and closedness, and every text, the reference's at every push.
* **The other five run the same lines, each line's text that recogniser's own one pass** over the
  reference's line (`--asr none` references; no reference text for them exists). On the RTX 5090:
  * sensevoice-small 1106/1106 (jfk, zh, ja, the 60 s clip);
  * paraformer-zh 530/530 (zh, jfk);
  * gigaam-v3-rnnt 1422/1422;
  * granite-speech-4.0-1b 1299/1299;
  * lfm2.5-audio-1.5b-asr 1299/1299.

  GigaAM's published file declares no sample rate, so a file without one is read at 16 kHz, as
  `transcribe` reads it.
* **Sabotage** (six patches of `line_stream.cpp`, rebuilt, on three moonshine-tiny cases), all caught:
  * look-behind one hop short: 527/730;
  * no fade: 557/718;
  * half the smoothing window: a crash (the ring runs past its vector);
  * `finish` re-running a stream with nothing new: 730/731;
  * the budget floored instead of ceiled: 720/730;
  * the open line not re-transcribed: 608/730.
* **Speed** of the gate's own stream, which also runs a reference one pass per checked line, so the
  stream alone is faster:
  * moonshine-tiny 24x real time on the RTX 5090 (RTF 0.041), 0.81x on the dev-box CPU while two other
    jobs shared it;
  * on the 5090: SenseVoice 14.7x, Paraformer 19.6x, GigaAM 16.4x, LFM2.5-Audio ASR 2.4x,
    Granite 1.6x (RTF 0.629).

  Re-transcribing the open line every 0.5 s costs a speech LM most: its one pass per update grows with
  the line, up to ~10 s.
* loom-py: `model.speech2text.stream(vad=silero)` (or `Model.line_stream`) gives a `loom.LineStream`; its
  pushes return the `loom.Line`s that changed. On the RTX 5090 it gives the gate's lines for jfk.
* **Lines are final once closed; the open line is revised** at every push that touches it.

## Related

* `include/loom/core/line_stream.h`, `src/core/line_stream.cpp`, `tools/loom_cli/main.cpp` (`--vad`)
* `scripts/vad_lines_reference.py`, `tests/gate/test_e2e_vad_line_stream.cpp`
