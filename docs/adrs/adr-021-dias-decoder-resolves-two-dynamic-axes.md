---
type: adr
status: accepted
date: 2026-09-03
domain: exporter
tags: [model-coverage, exporter, dynamic-shapes, cross-attention, family-10]
---

# ADR-021: Dia's Decoder Resolves Two Dynamic Axes Rather Than Padding Its Text

## Context

Family 10 — an autoregressive LM that emits neural codec tokens — is structurally family 2's shape:
an encoder run once, then a KV-cached decoder cross-attending to its output. Whisper is the precedent,
and `whisper_export` splits that into three phases (`encoder`, `cross_kv`, `decoder`) for reasons that
transfer unchanged.

**One thing does not transfer.** Whisper's encoder always emits exactly 1500 frames — it sees 30 s of
audio, always — so its decoder can declare fixed-shape `xk_i`/`xv_i` inputs and has exactly one dynamic
axis, its token count. Dia's encoder is byte-level and emits **one frame per input byte**, so the
cross-attention K/V handed to its decoder carry a second, genuinely independent dynamic quantity: a
sentence's byte count says nothing about how many codec frames it becomes.

The exporter's own `_validate_input_axes` states the constraint plainly: every dynamic input axis must
either be declared, or share one MIL symbol with every other undeclared one. `_sub_symbol` rewrites any
symbol it was not given an override for into `root_axis` — so two independent dynamic quantities
**silently collapse into one name**, and the emitted shape expressions are wrong rather than malformed.
No downstream gate catches that.

The received wisdom said this was not available. [Retro-013](../retros/retro-013-retrofitting-eight-bespoke-converters.md)
records, of Supertonic's `vfe` phase, that *"`GraphBuilder::build` resolves only ONE dynamic-length
symbol per topology, so `T_lat` gets `$n_tokens` and `T_TEXT` must be static"* — and
[Epic-05](../epics/epic-05-edge-performance.md) repeats it. Taken at face value, family 10 would have to
pad or bucket its text axis the way Supertonic does.

## Options Considered

1. **A statically-sized text axis.** Trace at a fixed width, pad every sentence to it. What Supertonic
   does. It makes the encoder run at the padded width on every utterance, and — the part that actually
   bites — padding is only inert if every op respects a mask, which
   [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md) is the record of getting wrong.
   Dia's cross-attention would attend over padded encoder frames unless a cross-attention mask became a
   second decoder input, which is a dynamic-width tensor rebuilt per step.
2. **Buckets.** Supertonic's eventual answer: trace at five widths, the driver picks one and pads. Five
   copies of a 12-layer encoder, and the same masking obligation.
3. **Declare the second axis.**

## Decision

**Option 3.** The decoder declares `n_enc_frames` for every one of its 36 cross-attention inputs
through `ExportPhase.declared_axes`, and the driver binds it alongside `n_tokens`/`n_past` on every
call.

Both halves already supported this; the received wisdom was a statement about one model, not about the
machinery:

* **The exporter** has `declared_axes` for exactly this, and `axes.py` already declared
  `N_ENC_FRAMES` — Kokoro's `decoder_vocoder` phase uses it for the structurally identical job, the
  encoder-output length a downstream phase consumes.
* **The engine** takes an axis *map*: `GraphBuilder::build(axes, ...)` evaluates every declared input
  dim as a `SymbolEnv` expression over whatever the caller bound, and `loom.run_subgraph`'s axes table
  is `{name = value, ...}` with no fixed arity.

Measured rather than argued: the traced program comes back with `codes (1, is0, 9)` and
`xk_0 (1, is1, 2048)` — two distinct coremltools symbols — and the emitted topology resolves them as
`n_tokens` and `n_enc_frames` respectively.

**Supertonic's limitation is real and is not this one.** Its `vfe` phase needs two independently-sized
sequences within *one traced attention block*, and it has a second, independent blocker anyway — a
length-derived pad coremltools refuses outright. Dia's two axes live on different inputs of different
attention blocks (self vs. cross), and neither is derived from the other.

**All 36 inputs must be declared, not one.** They share a single `ct.RangeDim` instance, so they share
one MIL symbol, and substitution is per symbol rather than per input.
`_reject_shared_symbol_overrides` raises unless every input carrying the symbol is named — which is the
check doing its job, not an obstacle.

## Consequences

* **Positive: Dia's text axis is fully dynamic**, with no padding, no buckets, no mask input, and no
  per-width copies of the encoder. The encoder runs at the caller's own byte count.
* **Positive: the "one dynamic symbol per topology" claim is now bounded.** It was carried forward as a
  property of `GraphBuilder`; it is a property of a *trace* in which two axes meet inside one attention
  block. The next family that needs two independent lengths should check which of the two it has.
* **Negative: the cross-attention K/V are copied into the decoder's inputs on every step** — 36
  tensors of `n_bytes × 2048` floats. That is Whisper's arrangement and Whisper's cost argument (it
  beats re-projecting them per step by an order of magnitude), but Dia's step count is ~10× Whisper's
  for the same wall-clock output, so it is the first thing to profile if the decode loop is ever slow.
  The mechanism to fix it, if needed, is `whisper_export.hoist_cross_v_transpose`, unchanged.
* **Negative: `declared_axes` is now load-bearing for a family rather than a convenience for one
  Kokoro phase.** A change to `_sub_symbol`'s substitution rule would break Dia's shapes silently, so
  `tests/ci/test_dia_export.py` asserts on the emitted `xk_0`/`xv_0` shape strings directly.

## Verification

`test_the_decoder_carries_two_independent_dynamic_axes` asserts that `codes` is sized by `n_tokens`,
that every cross-attention input is sized by `n_enc_frames`, and that neither carries the other's
symbol; `test_the_traced_lengths_do_not_reach_the_graph` exports the same checkpoint at two *pairs* of
trace lengths and requires an identical topology, because one baked axis is enough and either could be
the one. The end-to-end arm is `tests/gate/test_e2e_dia_mil_export.cpp`.

## See Also

* [Epic-03](../epics/epic-03-model-coverage.md) — the family roadmap
* [ADR-020](adr-020-audio-codes-is-its-own-modality.md) — why codec tokens are their own modality
* [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md) — the fixed text axis this avoids
* [Retro-030](../retros/retro-030-a-guard-that-could-not-fire.md) — the other tracing finding from
  this family
