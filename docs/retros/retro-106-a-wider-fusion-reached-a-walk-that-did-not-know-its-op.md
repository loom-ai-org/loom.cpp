---
type: retro
date: 2026-11-16
domain: exporter
tags: [shape-walk, symbolic-axes, fusion, rms-norm, voxcpm2, streaming, verification]
---

# Retro-106: A Wider Fusion Reached a Walk That Did Not Know Its Op

Serves [Epic-06](../epics/epic-06-high-level-api-and-hosts.md) (streaming) and
[Epic-03](../epics/epic-03-model-coverage.md); found by the layout-heal survey
([Retro-001](retro-001-layout-healing-heuristics.md)'s hub item). The third time
[Retro-044](retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md)'s lesson has cost
a model; [Retro-048](retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md) was the second.

## Issue

The heal survey re-exported every stream model from exporter main. VoxCPM2's fresh export failed its
first `stream_feed`, and `infer` failed the same way: the retained `feat_encode` came back
`[2048, 1]` where the next phase's input `feat_embed` is `[2048, n_text]`. [ADR-111](../adrs/adr-111-voxcpm2-streams-a-patch-at-a-time-through-a-state-carrying-vae-step.md)
had measured this stream at 7.0e-07 from `infer` on an export made the same day, 2026-11-12. The
published rc20 file predates both and is unaffected (its card ran clean in the survey).

## Root cause

VoxCPM2's feature encoder normalises each token's 5-row patch and keeps row 0 (`h[:, 0, :]`). MIL mints
a fresh symbol for a shape-derived slice, and `_infer_dynamic_dim_expr` relates it back to `n_tokens`
by walking producers through `_UNARY_PASSTHROUGH_OPS`. `loom_rms_norm` had never been in that set.
Exporter 94416f4 (2026-11-12, for CosyVoice3's GPU kernel count, loom.cpp ADR-113) taught
`fuse_rms_norm` to match `x * x` as well as `pow(x, 2)`. VoxCPM2 spells its square `x * x`, so its norms
started fusing, the walk stopped at the first one, and the VIEW and RESHAPE after the slice got a literal
1. 94416f4 was gated on the model it was written for (CosyVoice3, 139/139); VoxCPM2 was not re-run.

## Takeaway

**Widening a matcher moves an op into models that never had it.** Retro-048's rule, that a pass that
*introduces* an op owes the walk an entry, did not fire: `loom_rms_norm` was old, and only the set of
graphs it appears in changed. VoxCPM2 had worked with the gap since its export because its norms never
fused. A pass change is a change to every model it now matches, so its gate is the models it
newly matches, not the one it was written for. A count of fused nodes per model before and after would
have named VoxCPM2.

**A hand-kept list fails silently by omission.** The walk falls back to the root axis or a literal; no
error is raised. The fix adds `loom_rms_norm`, `loom_rope` and `loom_spline` (each types itself as its
`x`, `dialect.py`) to the passthrough set. The last two are the same gap waiting for a slice. The
structural fix, a CI check that every dialect op is either known to the walk or declared opaque, is in
the hub.

## Fix and verification

Exporter branch `fix/rms-norm-axis-walk`:
* `tests/ci/test_axis_walk_through_rms_norm.py` is the feature encoder in miniature, traced with a
  `RangeDim` token axis, once per spelling. It asserts that the norm fused and that the output RESHAPE
  is `[12, n_tokens, 1]`. Both cases fail without the fix.
* Re-export (9.3 GB, 7 topologies): `test_e2e_voxcpm2_lua_driver` 47/47.
* loom-py's `test_a_speech_stream_is_what_infer_returns` passes on the fixed file (72 min on the dev
  box).
* Exporter CI: 1532 passed on the workstation. The one failure is
  `test_chatterbox_export.py::test_the_vocoder_holds_back_exactly_its_reach`, which fails there with and
  without the fix (that machine's torch build) and passes on the dev box both ways.
