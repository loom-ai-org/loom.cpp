---
type: adr
status: accepted
date: 2026-11-11
tags: [engine, streaming, audio-in, asr, kyutai-stt, mimi, stream-state]
supersedes: []
---

# ADR-108: Kyutai STT Streams Its Encoder With moshi's State Carried

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Takes
[ADR-101](adr-101-pocket-tts-streams-a-frame-at-a-time-through-a-state-carrying-decoder-step.md)'s state-carrying step from
Mimi's decoder (Pocket-TTS) to Mimi's encoder, for the transcript door of
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md).

## Context

Kyutai STT 1B is a streaming model, and moshi runs it as one: each 80 ms frame goes through Mimi's encoder with
its streaming state (every causal conv's last inputs, the 8-layer transformer's ring of 250 positions written two
per frame), then the split RVQ, then one step of the 16-layer LM over a 750-cell ring. Its text runs 0.5 s behind
the audio. Our `infer` gives moshi's ids but recomputes the encoder in chunks of 250 frames with 1008 frames of
left context, the receptive field of 8 layers of 249 positions. A stream doing that per push would encode about
1000 frames for every new one.

## Options

The encoder:
1. **Windows over the one-pass graph** (ADR-107's way): about 1000 frames of left context per push.
2. **Every state as an explicit graph input and output** (ADR-101): 27 states, about 8 MB, nearly all of it the
   transformer's keys and values, kept on the device by `loom.keep_state`.
3. **A ring KV cache per module** (ADR-103): the cache would hold the keys, but the first position of each pair sees
   249 keys and the second 250, which `loom.causal_mask`'s one window cannot say.

The RVQ, at a frame or two per push:
1. **`infer`'s chain**, a stage per call with the argmax on the host: 34 calls a push.
2. **One graph**, the argmax of each stage on the device (P4.53's ARGMAX) and its centroid gathered by it.

## Decision

Encoder **2**, RVQ **2**. `infer` is unchanged.

* `mimi_encode_step(waveform, positions, key_positions, first, state_1..27) -> (rows, state')` and
  `mimi_encode_init(zero)`. The mask is `ring_pair_mask` over the carried keys' own positions; a key at a negative
  position is a slot no frame has written. The x2 downsample pads by replicate: on moshi's first call its state is
  the first input column, so the step takes `first` and uses `state * (1 - first) + x[:, :, :1] * first`.
* `rvq_encode(rows) -> codes`: the semantic stage, then the 31 acoustic stages chained, each `2 r.e - |e|^2`, its
  argmax and the gathered centroid subtracted. The codes come out as floats, frame-major.
* Driver (`kyutai_stt_driver/00_stream.lua`): `stream_feed` encodes the whole frames pushed so far, up to 25 a
  call, and runs one LM step per frame. Each step's id is final, so the feed returns the new ids and the text only
  grows. `stream_finish` pushes `stt_config`'s 1.5 s of trailing silence and keeps whole frames, as `infer` does.
  The LM's history is the model's ring KV cache, leased per stream (ADR-099).

## Consequences

* **The step is moshi's streamed encoder.** In PyTorch at f64 it is the one-call encoder to 1.4e-15 on jfk, at
  pieces of 1 frame, of 1, 3, 7, 2, 16 and 5 frames, and of the whole clip, past the 250-position wrap. The codes
  from its rows equal moshi's own frame-by-frame codes, 156/156 frames. A hermetic test builds a toy encoder from
  moshi's own modules and grades the step against moshi's streaming. It fails with the pair parity removed from the
  mask, and without `first`.
* **The fused RVQ draws the chain's codes.** In PyTorch it equals moshi's quantizer on jfk. In the engine it drew
  0 codes differently from the chain out of 37,728 on the gate's five clips. No codebook of this checkpoint repeats a
  row, which matters because ggml's ARGMAX keeps the last tied index and torch the first. A hermetic test against
  moshi's `SplitResidualVectorQuantizer` fails with the subtraction removed.
* **The stream ends on one call's ids.** The gate runs on five clips, 94 s in all: jfk, three TTS references
  resampled to 24 kHz, and 70 s of them joined, which is past the LM's 750-step ring. It passes 79/79. The encoder
  rows, streamed in uneven pieces, are within 1.7e-05 of one call. The ids are `infer`'s and the text is
  `transcribe`'s at 100 ms pieces and the whole clip on every clip, and at 20 ms, one frame and 0.5 s on the
  first. Text arrives before the end (228 updates on the 70 s clip) and is never revised. Two streams interleaved
  with a one-pass call between them, the 70 s clip one of them, each end where they end alone, so the ring cache
  parks and restores past its wrap. A driver that starts every call from a fresh encoder state fails it, 21/25
  on jfk: the ids differ at 3 of the 5 cuts. Its text still reads well ("you, ask" for "you. Ask"), so only
  the ids catch it.
* **Speed, dev box CPU, jfk (11 s):**

  | file | one pass | stream, 80 ms pushes | stream, 100 ms pushes | stream, 0.5 s pushes |
  |---|---|---|---|---|
  | F32 | 0.26x real time (RTF 3.83) | | 0.25x (RTF 4.04) | |
  | Q8_0 | 0.70x (RTF 1.44) | 0.68x (RTF 1.48) | 0.68x (RTF 1.47) | 0.74x (RTF 1.36) |

  The LM step sets the pace (238 ms at F32 and 68 ms at Q8_0, per 80 ms frame); the stream adds little to it.
  Before the fused RVQ, the 34 RVQ calls cost 80-98 ms per push, more than a frame lasts, and the F32 stream was
  RTF 5.05. Neither file is live on this 2-core CPU.
* Publishing needs a re-export (three new graphs and the stream functions) on an engine with ARGMAX and the door,
  which is rc21 or later. The published rc11 file gives the same ids in one call, and the live door refuses it.

## Related

* `loom-exporter/loom_exporter/kyutai_stt_export.py` (`MimiEncodeStepPhase`, `ring_pair_key_mask`,
  `RvqEncodePhase`), `kyutai_stt_driver/00_stream.lua`, `tests/ci/test_kyutai_stt_export.py`
* `tests/gate/test_e2e_kyutai_stream.cpp`
