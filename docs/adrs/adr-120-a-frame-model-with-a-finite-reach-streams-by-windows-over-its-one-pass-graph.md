---
type: adr
status: accepted
date: 2026-11-13
tags: [streaming, live-api, audio-in, vad, embedder, wakehubert, marblenet, frame-stream]
supersedes: []
---

# ADR-120: A Frame Model With a Finite Reach Streams by Windows Over Its One-Pass Graph

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming). It is the second and third
rows on [ADR-106](adr-106-a-frame-model-streams-rows-through-a-door-of-its-own.md)'s `FrameStream`, after
Silero VAD.

## Context

WakeHuBERT and frame-VAD MarbleNet are single-graph exports (`Flattened`, synthesized `infer`). Each
answers one row per 320 samples (50 per second). Neither carries a recurrence, and each frame reads a
finite span of samples:

* **WakeHuBERT** is strictly causal: a causal log-mel, a stride-2 stem and eight dilated causal blocks.
  Frame t reads samples `[320t - 38960, 320(t + 1))`. Upstream's own advice for streaming is to keep
  40,000 samples (2.5 s) of context, so that streamed frames equal offline ones.
* **MarbleNet** is centred: a centred STFT, then separable 'same' convolutions (kernels 11 to 17, the
  first at stride 2), with pre-emphasis and `normalize: None` (so nothing global). Frame t reads 23,457
  samples before it and 23,136 after it.

## Options

1. **A state-carrying step per model** (ADR-101's or ADR-108's shape): every convolution's tail as a
   state, about 20 tensors each, a new phase, and per-model exporter work. It is cheapest per push.
   MarbleNet's centred convolutions would still need the look-ahead.
2. **Windows over `infer`'s own graph**, as SpeechT5 (ADR-119) and Soprano (ADR-114) do for audio out:
   frames `[e, f)` are the graph over the samples from `320 (e - L)` to `320 (f + R)`, with their rows
   kept. One driver fragment serves any frame model whose reach is finite.

How the stream reaches a single-graph file:

* **a. Wrap it as a multi-phase export**, as the codec streams did (ADR-102). That renames the graph and
  every weight, and WakeHuBERT ships four quantized files with `keep_float` names.
* **b. Let the synthesized driver take a stream.** `CodecDecodeBuilder` gains an optional `stream` of
  top-level components, ahead of the unchanged `infer`.

## Decision

**2 and b.**

* **The reach is measured, not walked.** `frame_window_stream.frame_reach` puts a NaN in one sample of a
  clip and bisects for the first and last samples a middle frame reads. A frame reads one contiguous
  span. The probe runs on the very wrapper the trace runs, front end included (pre-emphasis, the STFT's
  padding, the mel). A NaN marks a dependency whatever the weights. A value probe would under-count:
  SpeechT5's far taps move a sample by ~1e-13. In frames: WakeHuBERT 122 + 0, MarbleNet 74 + 73.
* `frame_window_stream_driver/00_stream.lua` answers frames `[e, f)` once the samples
  `HOP (f + RIGHT_FRAMES)` are in, from a window starting `LEFT_FRAMES` back. A long push is cut into
  windows the graph takes (at most `MAX_PIECE` kept frames, from its `RangeDim`), so a stream has no
  length limit; MarbleNet's one call stops at 20 s. `stream_finish` runs a window ending at the
  stream's last sample, as one call's clip ends, unless the model is causal. A causal model answered
  every whole frame on arrival, and a trailing partial frame is not a frame in one call either.
  The state is `context` (touched only when a window runs) and `pending` (the samples since). Keeping
  one buffer made a 1-sample push copy WakeHuBERT's whole 2.5 s context.
* **`lookahead_frames`** (a `stream_open` option, 0 to `RIGHT_FRAMES`) answers a frame from a window
  ending that many frames after it. For MarbleNet that trades exactness for latency. `FrameStream` now
  takes `stream_open` inputs, as `TranscriptionStream` does: `loom_cli --stream MS --input
  lookahead_frames=N`, and loom-py's `speech2class.stream(lookahead_frames=N)`.

## Consequences

* **The graph, weights and every metadata key are byte-identical to the published files.** `infer` is
  the published text, so one-pass answers do not move. Only `model.driver_script` grows.
* Frame-stream gate (jfk, and jfk cut 100 samples short so that the stream ends inside a frame), pieces
  of 1, 160, 512, 1000 and 16000 samples, a ragged cycle and the whole clip:
  **WakeHuBERT** 4.8e-06 from one call at every piece size (features of norm up to ~40; the whole clip in
  one push is one window and is 0), 549 rows on the cut clip with no row for its partial frame;
  **MarbleNet** 2.4e-07, 550 rows on both clips, its last frames answered by `stream_finish`. Both cases
  pass: reset is bit-identical, interleaving with a one-pass call between pushes is bit-identical, and
  the published file answers the one pass bit for bit and is refused by the door. MarbleNet with
  `lookahead_frames = 10` answers its first row after 3,520 samples, where the exact stream has
  answered nothing, and is 0.445 off (PyTorch's number), with the same row count.
* Sabotage by same-length byte patches in copies: WakeHuBERT `LEFT_FRAMES` 122 -> 100 puts it 1.03 off;
  MarbleNet `RIGHT_FRAMES` 73 -> 50 gives 0.096 and `LEFT_FRAMES` 74 -> 50 gives 0.158. All red.
* MarbleNet uses its far context, unlike SpeechT5's tails (PyTorch, jfk, 550 frames):

  | look-ahead | worst \|dp\| | decisions flipped at 0.5 |
  |---|---|---|
  | 73 frames (1.46 s, exact) | 0 | 0 |
  | 40 (800 ms) | 0.137 | 2 |
  | 20 (400 ms) | 0.33 | 7 |
  | 10 (200 ms) | 0.45 | 11 |
  | 0 | 0.96 | 52 |

  So the default stays exact, with 1.46 s of latency, and `lookahead_frames` is an explicit trade.
* **The cost is the recomputed context.** On the dev box (`loom_cli`, jfk, 11 s), one call takes ~0.04 s
  (~275x real time) for either model. The stream runs at 2.0x real time (RTF 0.50) with 20 ms pushes,
  where each push recomputes 123 (WakeHuBERT) or 148 (MarbleNet) frames to answer one, and at 8-10x
  with 100 ms pushes. Upstream streams WakeHuBERT the same way. Option 1 would cut a 20 ms push's cost
  roughly a hundredfold; it is a hub item for a device where this is too slow, not built here. The
  stream state is 312 KB (WakeHuBERT) and 190 KB (MarbleNet).
* Exporter CI: the probe reproduces hand-derived reaches (a causal and a centred toy, through their own
  front ends); f64 windows cut as the driver cuts equal one call to 1e-12; one frame less context on
  either side moves rows; the toy WakeHuBERT export carries the stream beside the old `infer`.
* Re-export needed: the driver is in the GGUF. Publishing (all four WakeHuBERT precisions) waits for an
  rc21+ engine.

## Related

* `loom-exporter/loom_exporter/frame_window_stream.py`, `frame_window_stream_driver/00_stream.lua`,
  `audio_classification_export.py` (`_probe_stream`), `driver_components.py` (`CodecDecodeBuilder.stream`),
  `tests/ci/test_frame_window_stream.py`
* `include/loom/core/frame_stream.h` (`Inputs`), `tests/gate/test_e2e_frame_stream.cpp` (section 5),
  `tools/loom_cli/main.cpp`
