---
type: adr
status: accepted
date: 2026-09-25
tags: [exporter, model-coverage, family-11, codec, attention, drivers, moss]
supersedes: []
---

# ADR-049: A Codec Whose Windows Outrun Any Chunk Decodes in One Call, With Its Attention Blocked

## Context

MOSS-Audio-Tokenizer-v2 is the codec MOSS-TTS emits codes for. Its decoder has no convolutions at all.
It is six causal transformer stacks (32 layers at 12.5 Hz, then five 12-layer stacks, each at twice the
previous rate, up to 400 Hz) joined by patch reshapes. Every stack's attention sees a window: 125
frames at 12.5 Hz, and 400 positions from the 50 Hz stack up.

[ADR-034](adr-034-a-chunked-decode-is-the-drivers-loop-not-a-longer-call.md) gave family 11 a
second call shape for a codec with attention over the frame axis: a driver loop over bounded chunks,
each re-decoding some left context and dropping it. So the question here was which of the two shapes
this codec takes.

## Options

1. **ADR-034's loop.** Measured on 30 s of real speech against the reference's whole-sequence decode:
   25 frames (2 s) of carried context is **58%** away in relative RMS, and 100 frames (8 s) is still
   **48%**. Ninety-two layers of stacked windows reach much further back than any affordable context.
   This would not be an approximation of the model. It would be a different model.
2. **One call, masked-dense, like Pocket-TTS's Mimi.** This is exact, but a `T x T` score matrix at the
   400 Hz stack is 12 heads × T² floats: 0.8 GB per layer for 10 s of audio, 7 GB for 30 s.
3. **The reference's streaming path: a ring KV cache per layer.** This is exact. MOSS-TTS calls it,
   with `chunk_duration=8`, and it is 1.1e-06 from the whole-sequence pass. But the engine's KV cache
   grows without bound and attends to every cached key, so it would cost quadratic time, 92 caches
   and a sliding-window cache primitive that does not exist.
4. **One call, attention BLOCKED.**

## Decision

**Option 4.** The graph cuts each stack's sequence into blocks of `m` codec frames (`m · 2^stage`
positions). The queries of block `b` attend only to the keys of blocks `b − p … b`, where
`p = ⌈(W − 1) / B⌉`. The keys are fetched with one row gather per layer, from a clamped block index.

Measured against the span of blocks `b − p … b`, the window test `0 ≤ q − k < W` is the same for every
block. So it is one constant `[B, (p+1)B]` mask. The only fact that depends on `b` is that blocks
before 0 do not exist, and a second `[nb, (p+1)B]` mask built in the graph covers it. At `m = 4` the
400 Hz stack gathers 640 keys for a 400-wide window, and memory is linear in the clip.

**The driver pads the codes to a whole number of blocks and trims the output**
(`PaddedCodecCall`, the family's third call shape). The graph cannot pad its own dynamic axis, because
coremltools refuses dynamic padding. Padding at the end is exact because the decoder is causal end to
end. The pad id is the codec's absent id ([ADR-050](adr-050-a-codec-declares-its-absent-id-and-its-channels.md)).

## Consequences

* **Verified, engine against the reference's own decode, on real codes:** 375 frames (30 s, padded to
  376) come out at max |Δ| **2.6e-06** and relative RMS **1.2e-06**, and the same against the 8 s
  stream MOSS-TTS calls. The sabotage arm, the reference shifted by one frame, is 1.43. The engine
  takes 125 s for 30 s on the 2-core dev box, where PyTorch's own decode takes 116 s.
* **The window is exact at every block size.** The CI test checks the blocked stage against dense
  windowed attention with blocks narrower than, equal to and wider than the window. `m` trades padding
  granularity (up to 0.24 s) against gathered keys, and changes no output.
* **One call has a memory ceiling.** At `max_frames = 4096` (5.5 min, MOSS-TTS's own generation
  budget) the 400 Hz stack's scores are about 4 GB. A clip of the lengths TTS produces is a fraction
  of that.
* **ADR-034 still holds for its own case.** There the reference itself chunks, so the loop is the
  reference's answer. The rule the two ADRs make together: **copy the reference's call shape when it
  is exact, and when it is not, prove the exact shape affordable before choosing an inexact one.**

## Related

* [Epic-03](../epics/epic-03-model-coverage.md), family 11.
* [Retro-060](../retros/retro-060-a-sequence-at-axis-zero-was-read-as-a-batch.md): what went wrong
  on the way to this.
* `loom-exporter/loom_exporter/moss_audio_tokenizer_export.py`, `driver_components.PaddedCodecCall`.
