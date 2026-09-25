---
type: retro
date: 2026-09-25
domain: exporter
tags: [exporter, shape-walk, rope, verification, family-11, moss]
---

# Retro-060: A Sequence at Axis Zero Was Read as a Batch, and the Export Ran 0.1% Off

## The Issue

MOSS-Audio-Tokenizer's blocked decode ([ADR-049](../adrs/adr-049-a-codec-whose-windows-outrun-any-chunk-decodes-in-one-blocked-call.md))
was first written in a flat `[T, C]` layout. The torch wrapper matched the reference at 4e-07. On a
tiny random checkpoint the export converted, loaded and ran, with the right output length at every
frame count. In the engine it was **1e-03** relative RMS away. The sabotage arms were 0.4–3.0, so the
check could fail, and it said "nearly right".

## Root Cause

A truncated-export bisection put the error in the attention sublayer (the FFN was 3e-07). Probes on
the one-sided halves put it in the RoPE: the rotated Q alone was **65%** off, the pair rotation alone
was exact, and the position ramp was wrong. The topology said why: `RANGE_1D {"end": 1}`.

`value_facts.gather_shape_value` reads `x.shape[0]` on a rank≥2 tensor as a **batch size** whenever
the walk resolves it to the root axis, and answers 1. The exporter targets batch=1 everywhere, and
that guess fixes the common idiom. In a flat `[T, C]` layout, though, the first stack's length is
exactly `x.shape[0]` and exactly the root axis. So `arange(T)` became `arange(1)`, and ggml broadcast
position 0's rotation over every frame. That is GigaAM's rotary-crop failure (P4.2) from the other
side. There, the provenance test was added so a derived axis-0 length survives. Here the length is
the root axis itself, which is what the batch reading also resolves to.

It was only 1e-03 end to end because the tiny model's attention barely depends on position. The real
model's would not have.

## The Fix

The wrapper keeps a leading batch axis (`[1, T, C]`) at every stage, and every reshape whose length a
later shape read depends on is written with that length (`T * patch`, `T // B`), never with `-1`
([Retro-047](retro-047-an-inferred-dimension-outlives-the-reshape.md)). Engine against reference is
then 4e-07 to 1.2e-06 on the tiny model and 1.2e-06 on the real one.
`test_every_length_the_blocking_reads_stays_symbolic` asserts that every `RANGE_1D` bound in the
export is an expression in `n_codes`. With the flat layout it fails.

## Takeaway

**When you write a wrapper for the trace, the sequence axis must never be axis 0.** The exporter
reads axis 0 as the batch, and when it is wrong it does not fail, it writes a 1. And **a relative
error of 1e-03 on a random model is not "float noise"**: the torch wrapper had already shown what
noise looks like here (4e-07), so the gap was a defect, and bisecting it took five truncated exports.
Know the floor before calling a number close to it, which is the measurement
[Retro-049](retro-049-being-more-precise-than-the-reference.md) also turned on.

## Related

[ADR-049](../adrs/adr-049-a-codec-whose-windows-outrun-any-chunk-decodes-in-one-blocked-call.md),
[Retro-047](retro-047-an-inferred-dimension-outlives-the-reshape.md),
[Retro-044](retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md).
