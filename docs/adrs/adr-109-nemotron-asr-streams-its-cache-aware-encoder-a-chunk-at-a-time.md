---
type: adr
status: accepted
date: 2026-11-11
tags: [engine, streaming, audio-in, asr, nemotron, transducer, stream-state]
supersedes: []
---

# ADR-109: Nemotron 3.5 ASR Streams Its Cache-Aware Encoder a Chunk at a Time

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). Takes
[ADR-108](adr-108-kyutai-stt-streams-its-encoder-with-moshis-state-carried.md)'s state-carrying encoder step to a
cache-aware FastConformer, for the transcript door of
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md).

## Context

Nemotron 3.5 ASR was trained to stream. Its attention is chunked: 4 frames of 80 ms, each chunk seeing itself
and the 14 before it. Every convolution is causal, and the subsampling is three k=3, s=2 causal stages.
`transformers` streams it with conv caches and the last 56 keys and values per layer (`use_cache=True`). Our
published file runs the offline mode: the whole clip through the encoder with the chunked mask, then
Parakeet's greedy RNN-T loop. The offline encoder and the cache-aware one compute the same rows, because
the mask is what the caches carry.

`transformers`' streaming recipe differs from its one pass in two places, though. It computes each chunk's
mel with its own STFT call (frame-exact by construction). It also **drops the clip's last partial chunk**:
its generator stops once a whole chunk no longer fits, so up to 320 ms of trailing audio is never encoded.

## Options

1. **Windows over the one-pass encoder** (ADR-107's way): the left context is 24 layers of 56 frames,
   about 107 s, so a window would recompute nearly the whole clip per push.
2. **The reference's streaming as it is**: the same state, but the stream ends short of the clip, so it
   cannot end on `infer`'s ids.
3. **Every state as an explicit graph input and output** (ADR-101, ADR-108), whole chunks per call, with the
   clip's end handled as the one pass handles it.

## Decision

Option **3**. `infer` is unchanged.

* `encoder_step(waveform, frame0, n_valid, prompt, state_1..75) -> (rows, state')`, and `encoder_init(zero)`.
  The states are each subsampling stage's last input column, then per layer its last 56 keys, its last
  56 values and its depthwise conv's last 8 inputs: about 12 MB.
* **A call is whole chunks.** Encoder frame `t` reads mel frames up to `8t`, so chunk `c` runs once the
  samples reach mel frame `32c + 24`. The call's waveform is the centred clip from one sample before mel
  frame `32c - 7` (that sample is the preemphasis's neighbour): `160 M + 353` samples for `M` mel frames.
  The mel is computed in the graph.
* **The first call needs no special case.** Its first mel frame is `-7`. The rows before time 0 (mel
  frames, and the first two stages' outputs) are zeroed in the graph: that is exactly the zero padding
  the one-pass causal stages see there. The stage caches start at zero.
* **The last call is the clip's end.** `n_valid` says how many preemphasised samples are the clip's; the
  rest are the front end's right padding, which is zero after preemphasis. Every stage appends the one
  zero column the one-pass causal padding puts on its right. On whole chunks each stage's input count is
  odd, so that column never enters a window, and every call can append it: no flag.
* **Driver** (`nemotron_asr_driver/01_stream.lua`): `stream_feed` runs the whole chunks the samples allow,
  up to 25 per call, then the template's greedy RNN-T loop over their frames. Each emission is final, so
  the feed returns the new ids and the text only grows. The loop's state crosses calls as state: the
  last label, whether the prediction network must run again, and each LSTM layer's h and c on the device.
  `stream_finish` runs the rest of the clip.
* **The language** is `stream_open`'s `language` input, the prompt id; omitted, the file's default (`auto`).
  `TranscriptionStream::language_input(model, name)` resolves a name to that id from the file's
  declared table, as `transcribe` does. `loom_cli --stream --language` and loom-py's
  `stream(language=...)` use it.

## Consequences

* **The step is the one-pass encoder.** In PyTorch at f64 it equals `encoder` to 1.9e-14 on two clips, at
  one chunk per call, at 3, 1 and 2, and at the whole clip at once, the last partial chunk included.
  (`transformers`' eager attention takes its softmax in f32 whatever the dtype, so the reference arm needs
  an f64 softmax patched in. Without it the two differed by 1.6e-7, which was the reference's precision
  and not a bug in the step.) A hermetic test builds a toy encoder from stand-ins transcribing
  `transformers`' offline modules and grades the step against it. It fails without the appended zero
  column (12 of 15 cases), without the zeroing before time 0, and with the clip's padding taken for samples.
* **The stream ends on one call's ids.** The gate runs on eight clips, 107 s in all: jfk, six TTS
  references and 60 s of them joined. It passes 99/99. The encoder rows, streamed in groups of 1, 3, 2 and 7
  chunks and then the end, are within 5.4e-06 of one call. The ids are `infer`'s and the text is
  `transcribe`'s at 100 ms pieces and the whole clip on every clip, and at 10 ms, one chunk and 0.5 s on the
  first. Text arrives before the end (131 updates on the 60 s clip) and is never revised. A stream opened
  with `en-US` ends on `transcribe(language="en-US")`. Two streams interleaved with a one-pass call between
  them each end where they end alone. The published rc16 file gives the same ids in one call and is
  refused by the live door.
* **Two other oracles agree.** Through loom-py, the stream's ids equal `transformers`' offline `generate` on
  all eight clips. `transformers`' own cache-aware streaming `generate` gives a strict prefix of its offline
  ids on every clip (it drops the last partial chunk), so its streaming and its one pass agree everywhere
  both run, as this stream assumes.
* **Sabotage.** A driver that starts every call from a fresh encoder state fails the gate, 23/35 on two
  clips: the transcript comes back empty. One that re-runs the prediction network at each call's start,
  advancing its LSTM a step across calls, fails 27/35. Its text still reads well ("quip brown fox jumps
  over the lazy dog"), so only the ids catch it.
* **Speed, dev box CPU, jfk (11 s), through loom-py:**

  | file | one pass | stream, 80 ms pushes | 100 ms | 320 ms | 0.5 s | 1.28 s | whole clip |
  |---|---|---|---|---|---|---|---|
  | F32 (2.55 GB) | 2.90x real time (RTF 0.344) | 1.01x (RTF 0.988) | 1.02x (RTF 0.984) | 1.00x (RTF 1.004) | 1.05x (RTF 0.951) | 1.99x (RTF 0.503) | 2.14x (RTF 0.468) |
  | Q8_0 (0.71 GB) | 2.07x (RTF 0.483) | 0.77x (RTF 1.295) | 0.81x (RTF 1.230) | 0.81x (RTF 1.239) | 1.00x (RTF 0.998) | | |

  A call costs about the same whatever it carries, because it reads the encoder's weights once: at one
  320 ms chunk per call a 4-row matmul is bound by memory, not arithmetic. So a live stream, which gets a
  chunk every 320 ms, runs at about real time on this 2-core CPU at F32. Larger pushes approach the one pass
  (1.99x at four chunks a call). Q8_0 is slower than F32 here in both modes: this box's small-row int8
  kernels cost more than the bandwidth they save. The first text arrives after 0.9 to 2.2 s of audio.
* **At Q8_0 a one-chunk call is less accurate than the rest.** The gate's row bound is F32's. On the Q8_0 file
  the ids and text still equal `infer`'s and `transcribe`'s at every cut on two clips (33/35; the two row
  checks fail). Against the F32 encoder, though, the Q8_0 one call is 0.148 off on f5_tts_ref, and the
  stream at 2, 4, 8 or 25 chunks per call is 0.148 off too, but at one chunk per call (4 rows) it is 0.376
  off. Some op is less accurate at 4 rows; both of ggml's CPU matmul paths quantize the activations the same
  way, so the op is not identified yet (hub item). The published file is F32.
* Publishing needs a re-export (two new graphs and the stream functions) on an engine with the door, rc21 or
  later. The published file gives the same ids in one call, and the live door refuses it.

## Related

* `loom-exporter/loom_exporter/nemotron_asr_export.py` (`NemotronEncoderStepPhase`, `chunk_key_mask`),
  `nemotron_asr_driver/01_stream.lua`, `tests/ci/test_nemotron_asr_export.py`
* `tests/gate/test_e2e_nemotron_stream.cpp`; `TranscriptionStream::language_input`
