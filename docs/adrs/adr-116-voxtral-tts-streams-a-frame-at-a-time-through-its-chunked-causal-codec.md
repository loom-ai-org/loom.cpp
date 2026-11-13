---
type: adr
status: accepted
date: 2026-11-13
tags: [streaming, live-api, stream-state, tts, audio-out, voxtral]
supersedes: []
---

# ADR-116: Voxtral-4B-TTS Streams a Frame at a Time Through Its Chunked Causal Codec

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Puts Voxtral-4B-TTS behind
[ADR-100](adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md)'s audio-out door.

## Context

Voxtral's loop draws one frame of 37 codes per LM step (80 ms of 24 kHz audio) until the semantic code
is [END_AUDIO]; the codec then decodes the frames. The codec is causal with a finite receptive field
(~18 frames), and the one pass already decodes in chunks of 256 frames that each re-read 32 frames
before them (`voxtral_decode`), because one call over a minute of audio would build gigabytes of
attention bias. On the workstation's 24 cores at the export's precision, 11.3 s of audio took 62 s, all
of it before the first sample.

## Options

1. **A state-carrying codec step** (ADR-101's `mimi_step`, ADR-103's ring KV): exact and cheaper per
   piece, but new graphs for a codec that the one pass already decodes in windows.
2. **The one pass's own chunked decode, started at the first frame not yet handed out**: a piece's frames
   decoded from 32 frames before them. Causal, so no frame needs a later one, and nothing is held back.

## Decision

**2.** A driver change only.

* `stream_feed` draws `piece_frames` frames (default grows 1, 2, 4, then 8), then runs `voxtral_decode`
  over the frames from `CODEC_CONTEXT` before the first new one and keeps the new frames' samples
  (`voxtral_decode` took a start frame for it). `0` is the whole turn in one feed and one decode, which
  `infer` asks for, so the one pass is unchanged; it is also the only cut that returns `return_codes`.
* The loop is cut per frame, the previous frame fed back before the next is drawn (the prompt on the first
  step), so the flow head always reads an LM row from a graph run in the same call. The prompt, voice rows,
  pinned noise and teacher frames are state values; codes are kept from 32 frames before the first frame
  not yet decoded.

## Consequences

* Stream gate 21/21 on the workstation CPU with two short texts (no pre-stream file was copied there for
  the refusal check). The join is 140,160 samples, the one pass's count, at most 4.73e-07 from
  it; each stream after a reset, and interleaved with another stream and a one-pass call under both cache
  modes, is bit-identical to itself. Not bit-identical to `infer`: a piece's codec window holds another
  number of rows than the one pass's chunk.
* One-pass gate `test_e2e_voxtral_tts_lua_driver` 64/64 on the re-export: codes 142/142 identical to
  vllm-omni's free-running and teacher-forced, waveform 5.7e-07 (the reference's own f32/f64 spread is
  7.5e-07), and the same with `codec_chunk = 40`.
* Sabotage: an in-place same-length byte patch `CODEC_CONTEXT = 32` -> `02` (each piece decoded from 2
  frames before it, inside the codec's ~18-frame receptive field) puts the join 0.15 off the one pass, and
  the gate fails exactly its two stream-versus-one-pass checks (19/21). The one pass, a single 256-frame
  chunk at these lengths, does not move. Reverted, sha256 verified.
* Workstation CPU (24 cores, the export's 16 GB file): the first 80 ms is ready after 4.4 s, against 38 s
  for the one pass over 5.84 s of audio (0.15x real time, RTF 6.6). RTX 5090, `loom_cli` including the
  file's load: the first 80 ms after 0.19-0.22 s; the whole 5.44 s clip 4.7 s streamed against 4.5-4.8 s in
  one pass, so the stream costs nothing measurable there and runs faster than real time.
* Re-export needed: the driver is in the GGUF (85 s and 32 GB of RAM on the workstation). Publishing waits
  for an rc21+ engine like the other streamed re-exports. No engine change.

## Related

* `loom-exporter/loom_exporter/voxtral_tts_export.py` (`PIECE_FRAMES_GROWN`), `voxtral_tts_driver/00_header.lua`
  (`voxtral_decode`'s start frame), `voxtral_tts_driver/00_stream.lua`, `voxtral_tts_driver/01_generate.lua`
* `tests/gate/test_e2e_speech_stream.cpp`, `tests/gate/test_e2e_voxtral_tts_lua_driver.cpp`
