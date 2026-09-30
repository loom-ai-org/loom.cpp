---
type: retro
date: 2026-09-30
domain: exporter
tags: [exporter, nemo, masks, tile, layout-healing, family-1, conformer, parakeet]
---

# Retro-065: NeMo's Masks Were Baked All-True

## The Issue

While measuring the DFT-basis re-export ([Retro-064](retro-064-the-dft-basis-was-built-in-fp32.md)),
conformer-ctc-small matched NeMo at f64 on two clips and missed it on three. On those three,
every frame's log-probs were 0.1-0.5 off and the last frame up to 18. Parakeet-TDT's encoder was 8-32%
off (relative to max), and Parakeet-RNNT's 1-10%. The transcripts did not change, so no token-level
check could have seen it ([[feedback-tensor-oracle-not-token-oracle]]).

## Root Cause

There were two defects, and the first one hid the second.

**1. The exporter baked every NeMo length mask all-true.** `_less_is_always_valid_mask` replaced each
`arange(T) < f(length)` with a constant, except the one mask whose range was its bound plus one at every
probe (CMVN's). [Retro-013](retro-013-retrofitting-eight-bespoke-converters.md) justified that with
"a single utterance is never padded". That is false, because the padding is made *inside* the model.
The mel front end's valid length is `floor(n/160)`, one less than the frame count.
`MaskedConvSequential` re-masks after every stride-2 conv with `floor((L-1)/2) + 1`, and the conv's
output is one frame longer than that whenever `L` is even. The encoder's `pad_mask` takes the last
stage's length. It is therefore one short whenever every stage's input was even, which for a factor-4
model means `floor(n/160) % 4 == 0`. That includes jfk.wav, which the earlier note had listed as
"final length matches". NeMo's own lengths on the sweep confirm the rule: jfk has 276 frames with
length 275.

Retro-013 records why forcing every mask real once measured *worse* (2.09 against 0.13): the NeMo of
that time traced `calc_length` with a wrong constant. The shipped NeMo (2.6.2) derives each stage with
`calculate_conv_output_size`, and the exporter's facts now derive exactly NeMo's per-stage `(T, L)` at
every probe. The hub item's claim that "the traced arithmetic is the broken part" was carried over from
Retro-013 and was no longer true.

**2. With the masks real, `pad ∧ padᵀ` lost its transpose.** NeMo builds the attention mask as
`pad.unsqueeze(1).repeat([1, T, 1])` ANDed with its own transpose. coremltools packs a shape-read rep as
`concat(1, gather(shape(x), 1), 1)`. `_op_tile` resolved only the `expand_as` form
([Retro-063](retro-063-an-expand-as-was-lowered-as-an-identity.md)) and read these reps as ones, so the
REPEAT was an identity, and the MUL got `[T,1,1]` by `[1,T,1]`. Those shapes do not broadcast. The
engine's "axis 0/1 swapped" layout healer ([Retro-001](retro-001-layout-healing-heuristics.md)) then
permuted one operand onto the other, and the product was `pad·pad`. Only the padded query *row* was
masked, and the padded key *column* still took part in every softmax. Re-running NeMo with exactly that
mask (rows only) reproduced loom's output to 5e-5 in log-prob. NeMo's real mask left it 1.1-1.5 away, and the other variants 4.6-17.8.

## The Fix

* The guard bakes a mask only when the range and the bound are the **same symbolic expression**, which
  is a proof rather than a probe. Every other length mask lowers as a real LESS. In the NeMo encoders
  the only baked mask left is the waveform mask (`n_samples < n_samples`).
* `ValueFacts.tile_target_exprs` resolves the `concat` form: each axis's target is the input's extent
  times the rep, and every rep must be a real derivation, not the root-axis guess. The exporter's shape
  walk answers a live tile from the same derivation, so the walk and the REPEAT cannot disagree.
* Tests: `test_length_masks.py` (a stride-2 stage mask must stay a LESS; the waveform mask must still be
  baked) and `TestRepeatWithAShapeReadRep`. Reverting either fix turns its test red on the assertion.

## Measured Effect

Measured against NeMo at f64 on 20 lengths: jfk truncated to every residue of `floor(n/160) mod 8`,
plus the five full clips. Error is max |Δ| over max |ref|, and the rightmost column is NeMo's own
f32 run for scale:

| model | before (even lengths) | after (all lengths) | NeMo f32 |
|---|---|---|---|
| conformer-ctc-small (log-probs) | 2.3e-3 - 2.9e-1 | ≤ 4.7e-5 | ≤ 3.2e-5 |
| parakeet-tdt (encoder) | 1.9e-2 - 5.5e-1 | ≤ 1.6e-5 | ≤ 3.5e-5 |
| parakeet-rnnt (encoder) | 8.8e-3 - 1.05e-1 | ≤ 4.1e-6 | ≤ 6.9e-6 |

At odd lengths, the outputs were identical before and after. Transcripts of jfk.wav and four
LibriSpeech clips are identical between the published and fixed files, for all four models. gigaam-v3-rnnt has no NeMo-style length
mask, and its topology is unchanged.

The `repeat` fix is generic, so every other catalogued model the local box can export (32 of them)
was exported with a probe on `tile_target_exprs`. Only Qwen3-TTS resolves any live tile (its 8
`expand_as`, the Retro-063 form), and its GGUF is byte-identical with and without this change. The
`concat` form occurs only in the NeMo encoders. Voxtral-4B-TTS and MOSS-TTS need the workstation and
were not run.

**Published 2026-09-30** at loom-exporter `2640123`: conformer-ctc-small, parakeet-rnnt, parakeet-tdt,
and gigaam-v3-rnnt. GigaAM rode along because the DFT-basis re-export
([Retro-064](retro-064-the-dft-basis-was-built-in-fp32.md)) had missed it: its fresh export differs from
the old file only in the `[320,1,161]` basis, by 5.5e-5. Each file passed the card gate (2 passed, 6
skipped) and the engine gate (parakeet 70/70, gigaam 83/83). On the Hub, `x-linked-etag` equals the
local sha256 and every card is byte-identical.

## Takeaway

**A bypass justified by an invariant needs the invariant checked at the seam where it can break, not
at the input.** "Never padded" was true of the waveform and false one conv later. The bypass was also
self-concealing: while every mask was all-true, the broken REPEAT behind it had nothing to multiply.
Fixing one masking defect is the moment to re-run the oracle, not to expect the numbers to follow.

## Related

* [Retro-013](retro-013-retrofitting-eight-bespoke-converters.md): the original bypass, now corrected
* [Retro-063](retro-063-an-expand-as-was-lowered-as-an-identity.md): the first live-reps `tile`
* [Retro-001](retro-001-layout-healing-heuristics.md): the healer that turned an identity into a wrong answer
