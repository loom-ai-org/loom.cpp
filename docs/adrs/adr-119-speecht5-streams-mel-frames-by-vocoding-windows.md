---
type: adr
status: accepted
date: 2026-11-13
tags: [streaming, live-api, stream-state, tts, audio-out, speecht5, hifigan]
supersedes: []
---

# ADR-119: SpeechT5 Streams Mel Frames by Vocoding Windows

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The last audio-out row of the
survey. It uses [ADR-114](adr-114-soprano-streams-within-a-sentence-by-decoding-windows-of-hidden-rows.md)'s
cut, applied to a mel spectrogram.

## Context

SpeechT5's decoder loop makes two mel frames per step, with a stop head. After the loop, a five-layer
convolutional postnet (kernel 5) runs over the whole spectrogram, and HiFi-GAN turns it into 16 kHz audio,
256 samples per frame. Upstream has no stream. Its `generate_speech` returns the whole waveform. The
one pass is the oracle.

The postnet and HiFi-GAN are both non-causal but local. A sample of frame t reads frames
`[t - L, t + R]` and nothing further.

## Options

1. **A state-carrying vocoder step**, as Pocket-TTS's Mimi has (ADR-101). Every convolution is
   non-causal, so the state would be a tail plus a look-ahead at each of HiFi-GAN's four rates, about 40
   tensors across the postnet and HiFi-GAN. That is option 2 with extra graphs and an exporter rewrite.
2. **Recompute over a window holding the receptive field**, as Soprano (ADR-114) and LFM2.5-Audio
   (ADR-115) do. The `postnet` and `vocoder` phases already take any number of frames.

## Decision

**2.** It is a driver-only change.

* **The margins come from the layers.** `mel_context_frames` walks the dependency interval of one frame's
  samples back through `conv_post`, each upsampler's residual blocks (the widest one), its transposed
  convolution, `conv_pre` and the postnet's convolutions. It reads each layer's kernel, stride, padding and
  dilation. For `microsoft/speecht5_hifigan` the result is 24 frames each side, plus the postnet's 10:
  **34 and 34**. The walk is exact, not an estimate. A window's zero padding only stands in for positions
  outside it, so every position a kept sample reads is computed from the same inputs.
* `stream_feed` hands back `piece_frames` frames of audio once 34 more frames have been made past them.
  The default grows: 2 frames (32 ms), then 4, 8, and up to 256 (4.1 s). `0` runs the whole loop and then
  the postnet and the vocoder over every frame in one call each. That is the old one pass, and `infer`
  asks for it. The loop's end flushes the frames held back.
* The loop is cut per decoder step. The cross-attention K/V go into the stream's state (`loom.keep_state`),
  as Dia's do. The speaker, the stop rule's numbers and any pinned `masks` or `teacher_frames` are
  state values, as are the step, the frame fed back next, and only the frames a later window still reaches.
  The prenet's dropout masks are drawn per step, in the one pass's order, from the stream's own generator.

## Consequences

* **`infer` is unchanged**: the same WAV bytes as the published export (seeded, through `loom_cli`), and
  the one-pass gate passes 88/88 with the same numbers.
* **The join is not bit for bit**, unlike Soprano's. ggml's CPU convolution runs im2col + GEMM in batches
  of patches. A long one-pass postnet call ends on a remainder batch that another GEMM tile sums, so its
  rows differ from a window's by ~1e-6 (a probe: 0 at 268 frames, 9.5e-7 at 1340). HiFi-GAN amplifies
  rounding about 40x (in PyTorch its own f32 is 2.7e-5 from f64 on a 4 s clip). On a 21 s text the join is
  9.0e-5 to 9.7e-5 from `infer`; on short texts, 3.5e-6. The speech-stream gate holds SpeechT5 to 1e-3
  instead of 1e-4, and the comment there records why.
* **No f32 gate pins the last frames of the margin.** Patching `MEL_RIGHT_FRAMES` in a copy of the file:
  34 -> 4 puts the join 0.27 off (19/21, red), 10 gives 0.12 and 14 gives 5.8e-2, all red. 20 gives only
  1.4e-4, because real speech barely reaches the far taps (in f64, one frame short moves samples by
  ~1e-13). The exporter's CI test pins the exact numbers instead. A NaN in one frame must mark exactly the
  frames `[j - R, j + L]` (structural, whatever the weights), and f64 windows reproduce one call to 1e-12.
* Gates on the re-export: speech-stream 23/23 (both cache modes, a one-pass call between pulls, the file
  without the stream functions refused). loom-py `text2speech.stream()`: `exact`, 1.1e-5 from `infer`.
* Dev box CPU (4 threads), a 21 s text: the one pass takes 22.6 s (RTF ~1.07). The stream's first 32 ms is
  ready after 1.7 s and the whole takes 24.8 s (RTF 1.18). The margins cost: each window carries 68 frames
  (1.1 s) beyond its piece, and the vocoder is ~30% of a one pass. Fixed pieces of 32, 64, 128 and 256
  frames streamed at RTF 1.69, 1.42, 1.20 and 1.09, so the growth stops at 256 rather than at Soprano's 1 s.
* Re-export needed: the driver is in the GGUF. Publishing waits for an rc21+ engine, like the other
  streamed re-exports. No engine change.

## Related

* `loom-exporter/loom_exporter/speecht5_export.py` (`mel_context_frames`, `PIECE_FRAMES_GROWN`),
  `speecht5_driver/01_stream.lua`, `speecht5_driver/02_speech.lua`, `tests/ci/test_speecht5_export.py`
* `tests/gate/test_e2e_speech_stream.cpp` (`kSpeechT5Tolerance`), `tests/gate/test_e2e_speecht5_lua_driver.cpp`
* ADR-057 (SpeechT5's vocabulary), ADR-042 (the driver draws graph-input noise)
