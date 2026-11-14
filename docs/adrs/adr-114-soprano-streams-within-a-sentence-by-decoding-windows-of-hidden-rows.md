---
type: adr
status: accepted
date: 2026-11-12
tags: [streaming, live-api, stream-state, tts, audio-out, soprano, vocos]
supersedes: []
---

# ADR-114: Soprano Streams Within a Sentence by Decoding Windows of Hidden Rows

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Cuts Soprano finer than
[ADR-100](adr-100-a-tts-streams-the-pieces-its-reference-already-makes.md)'s sentence, the way upstream's
`infer_stream` does.

## Context

Under ADR-100 Soprano streams a sentence per piece: each sentence is its own generation and its own decoder
call, so the pieces join to the one pass bit for bit, but the first audio waits for a whole sentence (2.4 s
on the dev box for a 5.7 s sentence).

Upstream's `SopranoTTS.infer_stream` hands back audio every `chunk_size` hidden rows (default 1, 64 ms of
32 kHz audio). It keeps a sliding buffer of up to `2 * RECEPTIVE_FIELD + chunk_size` rows
(`RECEPTIVE_FIELD = 4`), runs the decoder over it, and keeps the samples that the rows on either side
cover. That works because the decoder is local in time: a x4 linear upsample (a frame reads only the two
rows either side of it), 8 ConvNeXt blocks with depthwise kernel 3 (one frame either side each), and a
centred ISTFT with `n_fft` 2048 at hop 512 (a sample reads the frames whose 2048-sample window covers it).

## Options

1. **A state-carrying decoder step**, as ADR-101 and ADR-111 did for Pocket-TTS and VoxCPM2. These
   convolutions are not causal, so the state would be the last few rows plus a lookahead. That is option 2
   with extra graphs.
2. **Recompute over a window with the receptive field either side**, as upstream does and as Dia's DAC
   windows do (ADR-103). The `decoder` phase already takes any number of rows, so this is a driver change
   with no new graph.

## Decision

**2.**

* The margins come from the decoder's config (`decoder_context_rows`): the first sample of row `e` reads
  frames from `4(e - a) - h + 1` of a window starting at row `a` (h = n_fft / 2 / hop = 2), and a window's
  zero padding spoils its first and last `C` frames (C = 8, the sum of the kernels' half-widths). That gives
  **3 rows before a piece and 4 after**. Upstream uses 4 both sides.
* `stream_feed` hands back `piece_tokens` rows of audio (2048 samples each) once 4 more rows are drawn past
  them. The default grows: 1 row, then 2, 4, 8, and 16 from then on. `0` is one sentence through one decoder
  call: ADR-100's cut, which `infer` asks for, so the one pass is unchanged. The sentence's end flushes
  the rows held back.
* The LM loop is cut per draw. Each step feeds the previous draw back before drawing, so a draw always reads
  logits and a hidden row from a graph run in the same call (as ADR-111 arranged). Between feeds the state
  values are the open sentence's position, step, budget, the penalty history, the draw not yet fed back, and
  only the rows a later window still reaches (`row_base` onward). The LM's history is the model's KV cache
  (ADR-099).

## Consequences

* The join is **bit-identical** to `infer`, not just within rounding: the speech-stream gate measures
  max|d| = 0 over 532,480 samples (21/21, both cache modes, a one-pass call between pulls). Every piece size
  (1, 4, 16, the growing default, a sentence) writes the same WAV bytes as the one pass through `loom_cli`.
  A windowed call computes each frame with the same per-row arithmetic as the whole-sentence call.
* `infer` on the re-export writes the same bytes as the ADR-100 export. `test_e2e_soprano_lua_driver` passes
  187/187.
* In f64 on the reference decoder with random weights (`test_soprano_export.py`), the 3/4 margins reproduce
  the whole call to 1.2e-15 relative. One row fewer on either side gives ~1e-10: nonzero, but below f32,
  because the lost frame reaches the kept samples only through the periodic Hann window's near-zero edge
  taps. So 3 and 4 are the exact margins, and 2 would pass any f32 test.
* Sabotage: a same-length byte patch `DECODER_RIGHT_ROWS = 4` -> `1` in a GGUF copy puts the join 0.731 off
  the one pass, and the gate fails its two stream-versus-one-pass checks (19/21).
* Dev box CPU (4 threads), two sentences, 11.26 s of audio: the first 64 ms is ready after 0.32 s, against
  2.4 s for the first sentence under ADR-100 and 4.3-4.6 s for the whole one pass (`loom_cli` 4.9-5.3 s
  including a 0.66 s load: ~2.2x real time, RTF ~0.45). The cost of the margins: growing to 8 rows took
  ~14% longer than the one pass overall, growing to 16 ~7%, fixed 1-row pieces 2.1x. loom-py:
  `text2speech.stream()` 7.3 s of audio, first piece 0.25 s, whole stream 3.4 s against a 3.1 s one pass,
  max|d| = 0.
* Re-export needed: the driver is in the GGUF. Publishing waits for an rc21+ engine like the other
  streamed re-exports. No engine change.

## Related

* `loom-exporter/loom_exporter/soprano_export.py` (`decoder_context_rows`, `PIECE_TOKENS_GROWN`),
  `soprano_driver/00_stream.lua`, `soprano_driver/01_generate.lua`, `tests/ci/test_soprano_export.py`
* `tests/gate/test_e2e_speech_stream.cpp`, `tests/gate/test_e2e_soprano_lua_driver.cpp`
* Upstream: `soprano/tts.py` `SopranoTTS.infer_stream`
