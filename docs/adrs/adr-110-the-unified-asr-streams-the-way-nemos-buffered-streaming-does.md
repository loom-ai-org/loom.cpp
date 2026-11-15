---
type: adr
status: accepted
date: 2026-11-12
tags: [engine, streaming, audio-in, asr, nemotron, transducer, nemo]
supersedes: []
---

# ADR-110: The Unified ASR Streams the Way NeMo's Buffered Streaming Does

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). The transcript door of
[ADR-098](adr-098-a-live-stream-keeps-its-state-in-the-engine.md) for Nemotron-Labs-Unified-ASR, the decode of
[ADR-109](adr-109-nemotron-asr-streams-its-cache-aware-encoder-a-chunk-at-a-time.md) shared.

## Context

Nemotron-Labs-Unified-ASR is one checkpoint for offline and streaming decoding. Our published file runs the
offline mode: full-context attention and the non-causal convolutions it was trained with. Its streaming mode is
not cache-aware. NeMo's streaming script (`speech_to_text_streaming_infer_rnnt.py`, the recipe the model card
gives) is **buffered**. Every step runs the whole model on a buffer of up to `left + chunk + right` seconds
(5.6 + 0.56 + 0.56 by default; latency is chunk + right, from 0.16 s), under the chunked attention mask
`set_default_att_context_size([left, chunk, right])` selects. It then decodes the buffer's chunk frames with a
stateful greedy decoder. The mel is normalised per feature over the buffer, so nothing carries from one step to
the next. The convolutions stay full-width: the checkpoint says `conv_context_style: dcc`, but NeMo 3.0 and main
store that and never read it.

So the streaming mode is not the one pass. Its ids differ from `infer`'s on most clips (jfk: 37 against 38, the
final "." gone at every latency tried), and the oracle is NeMo's streaming script, not `infer`.

## Options

1. **Windows over the one-pass encoder** (ADR-107's way): the stream ends on `infer`'s ids, but it is not the
   streaming mode the model was trained for and its card measures, and with full-context attention every
   window's last frames would change as audio arrives.
2. **Cache-aware decoding** of this checkpoint: the card says it "breaks that match and leads to accuracy
   degradation"; there is no reference to grade it against.
3. **NeMo's buffered loop as it is**: a step graph with the chunked mask, and the script's buffer arithmetic in
   the driver.

## Decision

Option **3**. `infer` is unchanged.

* `encoder_step(waveform, length, prompt, att_left, att_chunk, att_right) -> rows` is the encoder phase
  (preprocessor, encoder, prompt fold) with NeMo's `chunked_limited_with_rc` mask built in the graph from its
  three sizes in encoder frames, so one graph serves every latency. The mask goes in by shadowing the encoder's
  `_create_masks` while the step traces; the trace runs NeMo's own layers. Every weight is the `encoder`
  phase's (1055 aliased; the file grows 0.4 MB).
* The mask's comparisons are between `[T, T]` outer products against a ones vector, and both vectors are f32
  through an input. ggml broadcasts only a binary op's second operand, so a `[1, T]` against a `[T, 1]`
  aborted the engine, and a bare float `arange` reached it as an int32 matmul, which ggml's CPU kernel refuses.
  A CI test lowers the mask and fails on an integer matmul operand.
* **Driver** (`nemotron_asr_driver/02_unified_stream.lua`): `StreamingBatchedAudioBuffer`'s arithmetic at
  batch 1, in samples. The old chunk joins the left context, the new samples the right, the left is cut to the
  buffer's size, the left context's rows are dropped by floor (a last step's left part need not be whole
  frames), and the chunk's rows are decoded, or on the last step every row after the left.
* **One difference, forced by liveness:** the script knows the clip's length and runs its last step (chunk and
  right context together) when a step reaches the end. A live feed cannot know, so a step runs only once a
  sample past its end has arrived, and `stream_finish` runs the step that reaches the end as the script's last.
* `stream_open` takes `left_seconds`, `chunk_seconds` and `right_seconds` (the card's defaults), rounded to
  frames as the script rounds them, and `language`. It refuses a right context under one frame: without one
  NeMo's batch and scalar bookkeeping part ways and the script decodes the first chunk twice, so there is
  nothing to match. A chunk under one frame and a buffer longer than the step takes are refused too.
* The decode is ADR-109's loop, moved to `00_stream_decode.lua` and shared by both Nemotron streams (a row
  offset added).
* `loom_cli --stream MS --input name=v` passes stream options to `stream_open`; loom-py's `stream(**options)`
  already did.

## Consequences

* **The step is NeMo's step.** In PyTorch at f32 the step's rows over the driver's buffers are within 4.8e-07
  of NeMo's on all 19 of jfk's steps. The buffer sizes, left frames and decoded counts equal the reference's.
* **The stream ends on NeMo's streamed ids.** The gate runs nine cases, 128 s in all: jfk at four latencies
  (5.6/0.56/0.56, 5.6/0.16/0.4, 5.6/0.08/0.08, 2/1.04/1.04 s) and under the `en` and `auto` prompts, and four
  more clips, among them 60 s of joined references (107 steps), at the default. It passes 1280/1280. Every
  step's rows, over NeMo's own buffers, are within 7.7e-06 of NeMo's relative to their absmax (`en` and `auto`
  are 0.18 apart there, so a lost prompt fails). The ids are NeMo's at 100 ms pieces and the whole clip on every
  case, and at 10 ms, one chunk and 0.5 s on the first. Text arrives before the end and is never revised. Two
  streams interleaved with a one-pass call between them each end where they end alone. The published file is
  refused by the live door.
* **Sabotage.** A driver that hands the mask the right context's size as the left context's fails the gate
  (115/117): jfk's text loses its commas and still reads well, so only the ids catch it. One whose finish skips
  the last step fails 110/117 (the last 1.1 to 1.5 s of each transcript is missing). In CI, a mask window one
  frame wide fails 23 tests, and the float-less mask fails the lowering test.
* **Speed is the reference's design.** Every step re-encodes the whole buffer, so the cost per second of audio
  is about the buffer over the chunk. Dev-box CPU, F32, 100 ms pushes:

  | left / chunk / right (s) | latency | buffer / chunk | jfk |
  |---|---|---|---|
  | 2 / 1.04 / 1.04 | 2.08 s | 4x | 0.84x real time (RTF 1.19) |
  | 5.6 / 0.56 / 0.56 | 1.12 s | 12x | 0.33x (RTF 3.0) |
  | 5.6 / 0.16 / 0.4 | 0.56 s | 38.5x | 0.11x (RTF 9.1) |
  | 5.6 / 0.08 / 0.08 | 0.16 s | 72x | 0.06x (RTF 16) |

  None of it is live on this 2-core CPU; the card's own advice is a shorter left context (2 to 3 s) for speed.
  A GPU run has not been done.
* Publishing needs a re-export on an engine with the door, rc21 or later. The published file gives the same
  text in one call, and the live door refuses it.

## Related

* `loom-exporter/loom_exporter/nemotron_unified_export.py` (`UnifiedEncoderStepPhase`, `chunked_masks`),
  `nemotron_asr_driver/00_stream_decode.lua`, `02_unified_stream.lua`, `tests/ci/test_nemotron_unified_export.py`,
  `fixture_gen/reference_stream_nemotron_unified.py`
* `tests/gate/test_e2e_nemo_buffered_stream.cpp` (was `test_e2e_nemotron_unified_stream.cpp`); `loom_cli --input` on `--stream`
* [ADR-122](adr-122-parakeet-streams-the-way-nemos-buffered-script-does.md) moved the loop and the decode to
  `transducer_driver/11_buffered_stream.lua` and `10_stream_decode.lua`, shared with Parakeet; the oracle is now
  `fixture_gen/reference_stream_nemo_buffered.py`.
