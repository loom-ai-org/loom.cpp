---
type: adr
status: accepted
date: 2026-11-09
tags: [streaming, live-api, stream-state, tts, audio-out, codec, qwen3-tts]
supersedes: []
---

# ADR-102: A Codec Pair Streams as Two Doors, and a Causal Codec Carries Its State

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The audio-out door
([ADR-100](adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md),
[ADR-101](adr-101-pocket-tts-streams-a-frame-at-a-time-through-a-state-carrying-decoder-step.md)) for the
models that are two files ([ADR-022](adr-022-dia-and-its-codec-stay-two-files.md)): an LM that emits frames of
codes and a codec that decodes them. Qwen3-TTS first; Audio8, MOSS-TTS and Dia have the same shape.

## Context

A pair runs as two calls with the codes between them, so a stream has two halves: the LM handing back each
frame as it is drawn, and the codec decoding frames as they arrive. Qwen3-TTS's 12 Hz codec is causal end to
end (causal convs, transposed convs, ConvNeXt blocks, an 8-layer transformer over a 72-frame causal window).
Upstream ships no streaming decode: `qwen-tts` 0.1.1's `decode` is `chunked_decode(300, 25)`, equal to one
whole call through 299 frames and an approximation past it (ADR-034), and its `non_streaming_mode` flag only
"simulates streaming text input".

## Options

For the pair: **one engine door over both files**, or **the two doors composed by the host**, each streaming
on its own file. For the codec: **recompute each chunk with left context** (the receptive field is 8 x 71
frames plus the convs, so ~570 frames a chunk), or **a step that carries every piece of state**, ADR-101's
answer for Mimi.

## Decision

**Two doors, composed by the host; the codec carries its state.**

* The LM's half is `SpeechStream` as it is: a text-to-codes driver's pieces are frames of codes. The talker's
  `stream_open` builds the prompt, prefills and draws frame 0; each `stream_feed` runs the previous frame's
  feedback step and draws the next -- the one-pass loop cut where one iteration's feedback meets the next
  one's draw, so the draws keep their order and `infer` (the stream in one call) gives the codes it always did.
  The prompt's text schedule, which every step reads, is kept in the stream's state (`keep_state('prompt')`):
  as a retained output it would be another stream's after any call in between.
* The codec's half is a new door, `loom::audio::CodecStream`: `push(codes)` (whole frames) -> samples;
  `finish()`; `reset()`. Its driver decodes each push with `codec_step(codes [1, n, w], positions,
  key_positions, state_1..state_37)`, the decoder state (16 key/value windows, 21 conv tails and overlaps,
  1.32 M floats) kept on the device, a fresh one from `codec_init`. The one-pass graph and its 300 + 25 chunk
  loop are unchanged (`codec_stream_export.py`; the conv steps are `stream_steps.py`, shared with Mimi).
* Hosts push in GROWING groups, 1 frame, then 2, 4 and 8: one frame per push decoded at RTF 2.56 on the dev
  box, four or more beat the one-pass decode (1.82 and 1.58 against 1.90), since each call attends over 72
  frames rather than 300. loom-py: `codec.codes2speech.stream(lm.text2codes.stream(text))` iterates `Audio`;
  `loom_cli --codec c.gguf --out x.wav --stream 1` the same, printing when each push's audio is ready.

## Consequences

* Codec, on real talker codes (`test_e2e_codec_stream`): 99 frames streamed in pieces of 1 to 13 are 7.6e-07
  from one pass; 400 frames are 9.5e-07 over the first 300 and 0.092 past them, where one pass is the
  reference's chunked decode; reset, and two interleaved streams with a one-pass call between, are each
  bit-identical to the stream alone (21/21 twice); the one-pass decode is bit-identical to the published file,
  which the door refuses.
* Talker: one pass bit-identical to the published file; the stream bit-identical to one pass; the pair's audio
  1.2e-06 from decoding the same codes in one pass.
* Dev box CPU, 30 frames: first audio after 1.58 s, against the whole generation (28 s) before any audio one
  pass. The talker itself runs at about RTF 11 here (0.9 s a frame: one 28-layer step and fifteen predictor
  passes), so the pair is RTF 13 -- live on a GPU, not on this CPU.
* Two files stay loaded at once while streaming, which the one-pass `loom_cli` path avoided by releasing the
  LM first (MOSS-TTS is 16 GB at F32).
* Audio8-TTS, the second pair, on the same machinery: its codec is causal throughout too (the windowed
  transformer's `look_ahead_conv` is an identity), and its reference decodes in ONE call, so the stream is that
  call at any length. `Audio8CodecStep` carries 36 states (8 layers of 8 key/value heads over 128 frames, and
  the DAC decoder's tails and overlaps; 1.19 M floats). Talker and codec one pass bit-identical to the
  published files; the talker's stream bit-identical to its one pass; the codec stream 7.9e-07 from one pass at
  pushes of 1, 4 and 8; the pair 5.2e-07, its first audio after 1.30 s on the dev box (the talker 0.32 s a
  frame, the 44.1 kHz codec RTF 3.7 alone: RTF 11.4 together).
* Not covered yet: MOSS-TTS's and Dia's pairs (Dia's DAC is not causal, so its stream would be approximate),
  and a GPU run.

## Related

* `include/loom/core/speech_stream.h` (`CodecStream`); `tests/gate/test_e2e_codec_stream.cpp`,
  `tests/gate/test_e2e_codec_pair_stream.cpp`
* `loom-exporter/loom_exporter/codec_stream_export.py`, `stream_steps.py`, `codec_stream_driver/`,
  `qwen3_tts_driver/00_stream.lua`
