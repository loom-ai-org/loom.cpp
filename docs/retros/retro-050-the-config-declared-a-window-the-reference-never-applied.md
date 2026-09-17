---
type: retro
date: 2026-09-17
domain: exporter
tags: [verification, family-10, family-11, masks, qwen3-tts]
---

# Retro-050: The Config Declared a Window the Reference Never Applied

## The Issue

Qwen3-TTS's ICL mode replays a reference clip's own codec codes, so the talker's export had to grow
the codec's **encoder**: a Mimi conv stack, an eight-layer transformer over its output, a downsampling
convolution, and a sixteen-stage residual vector quantizer.

The transformer's mask was written from the checkpoint's own configuration, which says:

    encoder_config.sliding_window = 250

So the export prepared a windowed causal mask — `0 <= i - j < 250` — the same shape
`audio_codec_export._qwen3_tts_sliding_causal_mask` already builds for this codec's DECODER.

Against the reference's own `encode`, the result was **2192 of 2192 ids identical at 25 frames and at
6 frames, and 2125 of 2192 at 137**. Sixty-seven ids moved, and only in the long clip.

## Root Cause Analysis

137 frames is 274 transformer rows, and 274 is the first length in the test set that outruns a
250-wide window. Everything shorter never reaches the window, so the two masks are the same matrix and
the export looks exact.

Dumping the reference's own mask settled it in one call:

    row 273: reference j in [0, 273] (274 keys)    export j in [24, 273] (250 keys)

`MimiTransformerModel` asks `create_causal_mask` for its mask, and with `layer_types` unset that
function returns a **plain causal** mask. The window in the config is never applied on this path. The
checkpoint's two halves genuinely differ: the decoder's transformer declares
`attention_type = "sliding_attention"` and IS windowed; this one does not and is not.

## What Went Wrong

The mask was derived from **what the checkpoint declares** rather than from **what the reference
computes**. Those are different questions: a config records what a model was trained with, and the
reference implementation is what produced every output anyone has compared against. Where they
disagree, the second one is the oracle — [Retro-049](retro-049-being-more-precise-than-the-reference.md)
one model over, where reproducing FunASR meant reproducing its float32 rounding rather than improving
on it.

The near miss is the part worth keeping. Two of the three verification lengths passed at 100%, and a
suite that ran only those would have shipped a voice-cloning model that clones a different voice on any
reference clip longer than about eleven seconds — with no error, no shape mismatch and a plausible
waveform.

## The Fix

`encoder_attention_mask` builds a plain causal mask and carries the finding, the measurement and the
decoder's contrasting declaration in its docstring. `tests/ci/test_qwen3_tts_export.py` asserts that
the last row of a 300-row mask sees every earlier row — an assertion that reads as obviously true
until you know the config argues the other way, which is why it is written down.

## The Takeaway

**Dump the reference's mask; do not derive it from the config.** It is one call, it is decisive, and
the alternative is an export that is exact on every short input.

And: **a verification length that never reaches a mechanism does not test it.** Three clip lengths
looked like three samples and were two — the window only exists past 250 rows. Pick lengths that
straddle every constant in the model ([[feedback-a-predicate-right-at-both-ends]] is the same shape of
mistake one op over).
