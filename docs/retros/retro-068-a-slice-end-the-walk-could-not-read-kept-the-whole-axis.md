---
type: retro
date: 2026-10-01
domain: exporter
tags: [exporter, shape-walk, value-facts, slice, nemo, ctc, citrinet, conformer, canary]
---

# Retro-068: A Slice End the Walk Could Not Read Kept the Whole Axis

## The Issue

Citrinet's CTC output (loom-exporter, `feat/p5-canary-citrinet`) was one frame longer than NeMo's
`encoded_len` at about half of all input lengths: 39 frames for 48,777 samples where NeMo decodes 38.
The extra frame is computed from masked context and NeMo never reads it; the CTC driver reads every
row. It decoded to blank on every clip tried, which is luck, not a property.

Cutting the output to `encoded_len` in the traced wrapper (`log_probs[:, :encoded_len[0]]`) exported
cleanly and **changed nothing**: the topology ended in a `VIEW` whose length was the full frame count.

The published **conformer-ctc-small** has the same extra frame without any cut being attempted: 276
frames for JFK where NeMo decodes 275.

## Root Cause

Two mechanisms in series, neither of which raises.

1. **The length was unreadable.** NeMo's `MaskedConv1d.get_seq_len` uses
   `torch.div(..., rounding_mode='trunc')`, which reaches MIL as `sign(x) * floor(x * sign(x))`, and
   `encoded_len[0]` is a `slice_by_index` of a one-element tensor. `value_facts.scalar_expr` knew none
   of `sign`, unary `floor` or that slice, so the whole chain resolved to `None`.
2. **`_op_slice_by_index` reads `None` as "no bound".** `if e_val is None: e_val = dim_size` -- the
   axis's full extent. Correct for a slice that genuinely has no end; here the end existed and could
   not be read, and the two are indistinguishable at that line.

This is [Retro-044](retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md)'s shape
(an unknown symbol becomes the root axis) one layer over: an unknown slice BOUND becomes the whole axis.

The Conformer case has a different first cause and the same outcome: nothing ever asked for the cut,
so the family emitted every frame the convolutions produce, and the mel front end's `floor(n/160)`
valid frames out of `floor(n/160) + 1` plus each stride-2 stage's rounding leaves one past
`encoded_len` whenever every stage's input is even.

## The Fix

* `scalar_expr` resolves unary `floor`, a one-element `slice_by_index`, and the trunc pattern --
  matched by identity on its `sign` var and written exactly in the engine's grammar as
  `floor(Max(x, 0)) - floor(Max(-x, 0))`.
* `EncoderOutput.select` cuts CTC log-probs to `encoded_len` for every NeMo CTC leaf.

Measured: Citrinet and Conformer-CTC both emit exactly `encoded_len` frames at eight lengths, including
three where every stage's input is even. Canary's encoder phase relies on the same cut, so its
cross-attention needs no mask.

## Takeaway

**A fallback that substitutes "the whole axis" for "I could not read the bound" is a silent wrong
answer, and its only symptom is a length.** After an export that cuts by a computed length, read the
emitted `VIEW`'s shape expression and evaluate it at a length where the cut should bite -- an output
that is never shorter than its input was not cut. The silent fallback itself was not removed here:
other models rely on it for slices whose bound is genuinely absent, and telling those apart needs the
`is_guess`-style provenance `scalar_expr` already carries. That is the layout/shape-walk audit's work.

The transducer leaves (Parakeet-TDT/RNNT, GigaAM) emit their encoder frames untrimmed too, and their
decode loops read every row. Open in the hub.
