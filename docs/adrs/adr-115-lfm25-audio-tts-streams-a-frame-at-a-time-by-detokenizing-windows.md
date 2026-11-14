---
type: adr
status: accepted
date: 2026-11-12
tags: [streaming, live-api, stream-state, tts, audio-out, lfm2.5-audio, detokenizer]
supersedes: []
---

# ADR-115: LFM2.5-Audio's TTS Streams a Frame at a Time by Detokenizing Windows

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Puts LFM2.5-Audio's TTS
behind [ADR-100](adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md)'s audio-out door, with
the recompute-a-window cut of [ADR-114](adr-114-soprano-streams-within-a-sentence-by-decoding-windows-of-hidden-rows.md).

## Context

The TTS door is `generate_sequential` with a "Perform TTS." voice prompt: the 1.5B LM reads the text
(greedy) until `<|audio_start|>`, then draws one frame of 8 codes per step through the depthformer (80 ms
of 24 kHz audio) until a frame opens with end-of-audio, then text again until `<|im_end|>`. The one pass
then runs the detokenizer over every frame: each frame's codes repeated x6 into rows, an 8-layer LFM2
backbone, a 1282-wide head and a 'same'-padded ISTFT (n_fft 1280, hop 320). On the dev-box CPU at F32 a
4.4 s sentence takes 41 s, and nothing plays until it is done.

The backbone is causal: 5 short convolutions read 2 rows back, 3 attention layers a causal window of
30 rows. Only the ISTFT looks ahead, and only by the 480-sample overlap of the next row's frame.
liquid-audio's demo decodes frames as they are generated; there is no streaming entry point in the
library to match beyond that.

## Options

1. **A state-carrying detokenizer step**: per attention layer a ring KV cache of 29 rows, per
   convolution its last 2 rows, the ISTFT's overlap. Exact and cheapest per piece, but a new step graph
   with ~16 states and ring masks (ADR-103's machinery), for a 512-wide module beside the 1.5B LM that
   draws each frame.
2. **Recompute over a window**: the existing `detokenizer` phase over the rows from the backbone's
   receptive field before the first row a piece's samples read, to the last row drawn. No new graph.

## Decision

**2.** A driver change only.

* `detokenizer_context_rows` derives the reach from the detokenizer's config: the sum over layers of
  `conv_L_cache - 1` per convolution and `sliding_window - 1` per windowed attention, **97 rows** (about
  16 frames). A sample `n` reads the ISTFT frames (rows) from `ceil((n + 480 - 1279) / 320)`, so a piece
  starting at sample `s` is detokenized from row `ceil((s - 799) / 320) - 97` (or 0). It hands back every
  sample up to 480 before the last drawn row's end; the end of the turn flushes them.
* `stream_feed` draws `piece_frames` frames (default grows 1, 2, 4, then 8), then detokenizes. `0` is the
  whole turn in one feed and one detokenizer call, which `infer` asks for, so the one pass is unchanged.
  Codes are kept only from the first frame a later window reads.
* The LM loop is cut per step. Each step feeds the previous draw back (a text id through `embed`, a frame
  through `audio_embed`; the prompt on the first step) before it draws, so a draw reads the decoder's row
  from a graph run in the same call. The text after end-of-audio still runs to `<|im_end|>`, as the one
  pass does, so a second `<|audio_start|>` would continue the same audio.

## Consequences

* Stream gate 23/23 on the re-export (two short texts, seeded at the README's T = 0.8, top-k 64): the join
  is 105,600 samples, the one pass's count, at most 1.63e-06 from it. Not bit-identical as Soprano's is
  (ADR-114): a window's attention runs over fewer rows than the one call's. Each stream after a reset, and
  interleaved with another stream and a one-pass call under both cache modes, is bit-identical to itself.
  The v5 file (no stream functions) is refused by the door.
* One-pass gate `test_e2e_lfm25_audio_tts_lua_driver` 20/20 on the re-export: codes 65/65 identical to
  liquid-audio's, waveform 3.0e-06.
* In f64 on a random detokenizer of the checkpoint's layout (`test_lfm25_audio_export.py`): a window from
  row `a` gives the one-call hidden rows from `a + 97` on exactly, and row `a + 96` is 2.2e-09 off; every
  piece of a 40-frame turn is the one-call samples to 1.0e-15, and holding back one hop less is 0.29 off.
  The test builds the window mask in f64: an f32 mask handed to an f64 transformers model is replaced by
  a plain causal one with no window, which silently gives the backbone an unbounded receptive field.
* Sabotage: an in-place same-length byte patch `DETOK_CONTEXT_ROWS = 97` -> `07` (every window after the
  first ~100 rows starts 90 rows too late) puts the join 0.755 off the one pass, and the gate fails exactly
  its two stream-versus-one-pass checks (19/21); reverted, sha256 verified.
* Dev box CPU at F32: the first 60 ms is ready after 2.1 s, against 41 s for the one pass (0.11x real
  time, RTF 9.4). loom-py `text2speech.stream()` on 4.32 s of audio: first piece 2.35 s, the whole stream
  34.6 s against 33.9 s for the one pass (the windows cost ~2%), 8.3e-07 from `infer` (9.3e-07 at
  `piece_frames=3`). The stream cannot be faster than the LM, so it is live only on faster hardware; no GPU run
  yet, as for most streams.
* Re-export needed: the driver is in the GGUF. Publishing waits for an rc21+ engine like the other streamed
  re-exports. No engine change; the speech-stream gate learned the BPE text front end.

## Related

* `loom-exporter/loom_exporter/lfm25_audio_export.py` (`detokenizer_context_rows`, `DETOK_TRIM`,
  `PIECE_FRAMES_GROWN`), `lfm25_audio_driver/00_stream.lua`, `lfm25_audio_driver/02_speak.lua`,
  `tests/ci/test_lfm25_audio_export.py`
* `tests/gate/test_e2e_speech_stream.cpp`, `tests/gate/test_e2e_lfm25_audio_tts_lua_driver.cpp`
