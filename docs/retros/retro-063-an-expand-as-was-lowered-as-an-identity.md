---
type: retro
date: 2026-09-29
domain: exporter
tags: [exporter, lowering, tile, expand, shape-walk, family-10, qwen3-tts]
---

# Retro-063: An `expand_as` Was Lowered as an Identity

## The Issue

Qwen3-TTS's `text2codes.infer(..., waveform=<clip>)` killed the process:
`GGML_ASSERT(a->ne[d] == b->ne[d])` in `ggml_concat`, on the driver's first call (`speaker_encoder`).
It did so on every clip, on the rc10 and rc11 engines, and on the published and staged GGUFs. The
card's clip step was withdrawn on 2026-09-27 and the card moved to `x_vector=`. ICL's own verification
had passed an x-vector, so nothing had ever run the speaker encoder.

## Root Cause

ECAPA's attentive-statistics pooling is `cat([h, mean.expand_as(h), std.expand_as(h)], dim=1)`.
coremltools lowers `expand_as(h)` to

    tile(x, reps = real_div(select(equal(shape(h), -1), x.shape, shape(h)), x.shape))

The reps come from `h`'s live shape, so they never fold to a constant. `_op_tile` read them with
`static_value(reps, [1])`: **no constant meant "all ones", so the REPEAT became an identity.** The
mean reached the CONCAT as one frame against a T-frame hidden state.

The engine made it worse. `op_concat` handed its operands straight to `ggml_concat`, whose assert
aborts. A Python caller could not catch it, and the message named no node. A mismatched ADD throws a
`SchemaError` that `build_node` labels. CONCAT had no such check.

It had stayed hidden because **an identity REPEAT is harmless wherever a broadcasting op consumes it**.
The talker's decorative mrope `position_ids.expand(3, ...)` is a live-reps tile too, and every reader of
it takes row 0. Only a CONCAT refuses to broadcast.

## The Fix

* **Exporter** (`ValueFacts.tile_target_exprs`): recognise coremltools' expand pattern and take the
  target from `shape(h)` axis by axis, reusing `_reshape_shape_uncached`'s whole-shape walk (now
  `whole_shape_exprs`). Any other live-reps tile keeps the old reading. Making them raise broke the
  talker's `position_ids` at export, which is how that second case was found.
  `tests/ci/test_tile_lowering.py` fails without the fix (`['1','4','1']`).
* **Engine** (`op_concat`): check every off-axis extent and throw a `SchemaError`. The same failure
  now reads `node 'CONCAT' (... outputs=[input_145_concat_temp_1]): CONCAT: along dim 1,
  a=[1031,1536,1,1] and b=[1,1536,1,1] differ on dim 0`, and the process survives
  (`test_concat_rejects_mismatched_operands`).

The re-exported talker differs from rc11's in exactly those two REPEAT nodes. Every tensor and every
other topology is byte-identical.

## Takeaway

**A lowering that defaults an unreadable operand to its identity value turns "I don't know" into a
silent wrong graph.** The default lived in one `static_value(..., [1])` call and was right for most of
the zoo. When an operand is not a constant, either resolve it or keep the old reading deliberately
and say which cases do that. Do not fall back to the identity by accident.

**An engine op that wraps a ggml assert should check the assert's condition first.** A labelled
`SchemaError` would have named the node on day one, and this bisect started from a gdb backtrace.

## Related

* [Retro-064](retro-064-the-dft-basis-was-built-in-fp32.md): the second defect in the same speaker
  encoder, found once this one stopped the abort
* [Retro-047](retro-047-an-inferred-dimension-outlives-the-reshape.md): the same shape of failure
  (export, write and load pass; only running fails)
* [ADR-038](../adrs/adr-038-the-codecs-encoder-ships-inside-the-talker.md)
* [Epic-03](../epics/epic-03-model-coverage.md) family 10, Qwen3-TTS
