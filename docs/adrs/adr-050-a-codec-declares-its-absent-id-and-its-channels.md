---
type: adr
status: accepted
date: 2026-09-25
tags: [exporter, contract, family-10, family-11, codec, stereo, moss]
supersedes: []
---

# ADR-050: A Codec Declares Its Absent Id and Its Channels

## Context

MOSS-Audio-Tokenizer-v2 differs from every codec before it in two ways that a HOST has to know about.

* **It has 32 codebooks, and the LM that feeds it emits 12.** MOSS-TTS-Local-Transformer-v1.5 is
  trained at a fixed RVQ depth of 12. The reference decodes its output with
  `decode(num_quantizers=12)`, which sums the first 12 codebooks and never looks at the others.
  `codec.n_codebooks` has always been an exact row width, and loom-py refuses a row of any other width.
* **It is stereo.** It models `L R L R …` as one 96 kHz stream, so what it outputs *is* the
  interleaved waveform. Every consumer so far assumed mono: `Audio`, `loom_cli`'s WAV writer and
  `ModelContract`.

## Decision

**`codec.absent_code`: an id per column that contributes nothing.** The exporter folds each
codebook's 1×1 projection into its codebook, bias included, which turns the quantizer into one gather
and one sum. It also gives every codebook one extra row, id 1024, and that row is all zero. So a row
whose last 20 columns are 1024 decodes exactly as the reference's 12-codebook prefix does. This is a
property of residual quantizers, not a trick: a prefix of the sum IS a coarser reconstruction. It is
also MOSS's own spelling, since `audio_pad_code` is already 1024. The file declares the id. Given
narrower rows, loom-py fills them with it. A codec that declares none still refuses narrow rows.

**`channels`: interleaved channels in the audio.** It is written only when it is not 1, and it is not
a `codec.` key, because it describes the waveform: a stereo TTS model would declare it too.
`ModelContract::channels` defaults to 1, `Audio` gains `channels` (duration, `save()` and a
`[frames, channels]` array view), and `loom_cli` writes the matching WAV.

## Options not taken

* **Export the codec at the LM's width** (12). The graph is simpler, but this ships a 4.3 GB file per
  width, and one codec serving many LMs is the point of
  [ADR-022](adr-022-dia-and-its-codec-stay-two-files.md).
* **A dynamic width axis in the graph.** A second dynamic axis would need new driver plumbing, and the
  driver would also need the width passed in, since a flat run cannot say it. Padding on the host,
  which already knows the width, costs one list operation.
* **Downmix to mono in the graph.** That would be a different model from the one that was published.

## Consequences

* Verified: the 12-codebook prefix decodes at relative RMS **1.3e-06** against the reference's
  `decode(num_quantizers=12)`, and all 32 codebooks on the same frames are 36% away.
* `test_codec_pair`'s equality check between an LM's width and its codec's becomes "≤, when the codec
  declares an absent id". That lands with MOSS-TTS, the first pair it applies to.

## Related

[ADR-049](adr-049-a-codec-whose-windows-outrun-any-chunk-decodes-in-one-blocked-call.md),
[ADR-020](adr-020-audio-codes-is-its-own-modality.md),
[Epic-03](../epics/epic-03-model-coverage.md) family 11.
