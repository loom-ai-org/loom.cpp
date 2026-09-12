---
type: adr
status: accepted
date: 2026-09-12
tags: [exporter, model-coverage, family-11, drivers, qwen3-tts, edge]
---

# ADR-034: A Chunked Decode Is The Driver's Loop, Not A Longer Call

## Context

Qwen3-TTS's 12 Hz speech tokenizer is family 11's fourth leaf and the first that is **not
convolutional throughout**. DAC, SNAC and EnCodec are conv stacks; this one puts an 8-layer
transformer between the quantizer and the upsampling stack, attending over the frame axis inside a
72-frame sliding window.

Its reference implementation does not decode a whole sequence. `Qwen3TTSTokenizerV2Model.decode`
calls `self.decoder.chunked_decode(codes)`, whose defaults are `chunk_size=300` and
`left_context_size=25`: each call re-decodes 25 frames it has already emitted, then discards
`25 * 1920` samples from the front of its output.

`encodec_export.py` had already written the sentence this decision answers, while refusing a
checkpoint that declared `chunk_length_s`:

> a chunked one is a different driver, not a longer call.

This is the leaf that makes that concrete. The question is whether loom's answer should be one call
over the whole sequence — which is what every other member of the family emits, and what the
checkpoint's own `decoder.forward` computes — or the reference's loop.

## Decision

**The driver loops.** A new `ChunkedCodecCall` component replaces `MonolithicCall` for a leaf that
declares `chunk_frames`, and `max_frames` for that leaf is `chunk + context` — 325 — rather than the
family's usual 4096.

## Why

**It is what the reference computes.** Measured on real codes from this model's own talker:

| frames | max abs difference | relative RMS |
|---:|---|---|
| 42 | 0.000e+00 | 0.0000 % |
| 299 | 0.000e+00 | 0.0000 % |
| 301 | 1.326e-01 | 1.1 % |
| 700 | 7.421e-01 | 8.9 % |

Whole-sequence and chunked are **bit-identical through 299 frames** — one chunk, no carried context,
literally the same call — and then part company. 8.9 % is ~21 dB down, which is the band
[Retro-043](../retros/retro-043-the-band-was-20db-down-and-audible.md) records a listener hearing on
the first play after every metric had said it was inaudible. That retro is the reason this is not
argued from "more context must be better".

**And it is what makes the leaf runnable at all.** Attention is quadratic in the frame axis. A
whole-sequence call at a 4096-frame ceiling builds a 4096×4096 score matrix in each of 8 layers — on
the order of a gigabyte — plus a 67 MB mask, on an engine whose reason for existing is edge devices.
Chunked, no call exceeds 325 frames and the cost is flat in clip length. **The 325 ceiling is a
property of the driver, not of the checkpoint**: the graph is never asked for more because the driver
never asks.

So the two arguments point the same way, and either alone would be enough. That matters, because the
fidelity argument on its own is the weaker one — a case could be made that the whole-sequence pass is
the model's real definition and `chunked_decode` its memory ceiling. The resource argument settles it
independently.

## What It Cost

One driver component, no engine change, no new binding. The loop is a `SubgraphCall` inside a `While`
because **a chunk is a self-contained call**: no cache, no overlap-add, no window function, nothing
carried between iterations. The left context is dropped from the *output*, not from the input, which
is what makes the seam continuous — each chunk re-decodes frames it has already emitted so that its
first kept frame has a populated receptive field, and every output sample is then produced exactly
once, by the call that had the most history for it.

The waveform crosses the Lua boundary either way — [ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md)'s
rule is that an edge is a reference *unless the host reads it*, and here the host's answer IS the
waveform — so chunking adds only the per-chunk table churn and a small `array_slice` of the codes.

## Verification

Against the reference's own `chunked_decode`, on the engine's floats rather than through
`loom_cli`'s PCM16 wav (which has a 1.5e-05 quantisation floor of its own and would have hidden the
real number under a worse one):

* 42 frames: 80,640 samples, exact count, **max abs difference 4.167e-06**, 0.0003 % relative RMS.
* 700 frames — past two chunk boundaries: 1,344,000 samples, exact count, **max abs difference
  1.699e-05**, 0.0004 % relative RMS, against a whole-sequence answer that is 8.9 % away.

That second row is also the sabotage arm: the comparison discriminates between the two hypotheses by
four orders of magnitude, so it is a check that can fail.

The ASR oracle reads the 42-frame decode back as *"The quick brown fox jumps over the lazy dog."* —
the sentence the talker was asked for.

## Alternatives

* **Decode whole-sequence and document the divergence.** Rejected on the resource argument: it is not
  merely less faithful past 24 s, it is unrunnable on the hardware this engine targets.
* **Chunk inside the graph.** The chunk count is data-dependent, which is what a driver loop is for;
  a topology is a pure graph built once and reused.
* **An engine binding for chunked decode.** Nothing about this is engine work. There is no state to
  carry, so it is a `While` around a call the driver already makes — and the engine stays free of the
  knowledge that this codec chunks, exactly as it stays free of Dia's delay pattern
  ([ADR-020](adr-020-audio-codes-is-its-own-modality.md)).
