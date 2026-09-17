---
type: retro
date: 2026-09-12
domain: exporter
tags: [tracing, coremltools, shapes, family-11, verification]
---

# Retro-044: A Slice Retired the Algebra, and the Walk Quietly Substituted the Root Axis

## The Issue

EnCodec's first working export decoded 4 seconds of audio into **200 samples**. Not an error: the
export succeeded, the GGUF loaded, all four topologies built, the driver ran, and the first 200
samples were numerically right (cosine 0.9997 against `transformers` over that prefix).

This is family 11's third export and its **second silent wrong length**. DAC's first export returned
one frame's audio forever (Retro in Epic-03 §2); this one returned one sample per frame.

## Root Cause Analysis

Two mechanisms in series, neither of which raises.

**1. MIL retires symbolic algebra through a shape-derived slice.** EnCodec's decoder crops after every
transposed convolution with `hidden_states[..., padding_left : hidden_states.shape[-1] - padding_right]`.
MIL cannot express "this length minus a constant" through that, so it stops propagating and mints a
fresh opaque dim:

    conv_transpose  in=(1, 1024, is0)        out=(1, 512, 8*is0 + 8)     <- algebra intact
    slice_by_index  in=(1, 512, 8*is0 + 8)   out=(1, 512, is118)         <- algebra gone

Every length downstream is then a symbol with no stated relationship to the root axis.

**2. The exporter's rule for an unknown symbol is to substitute the root axis.** That rule is correct
for the common case — a pure pass-through of the topology's one dynamic input — and it is what
`_validate_input_axes` protects for *inputs*. For an internal symbol it has no such guard, so
`is118` became `n_codes`, and a crop that should have read `8*n_codes + 2` was emitted as
`n_codes + 2`. The graph is consistent, buildable and wrong.

The walk *does* derive these lengths itself rather than trusting MIL — that is what
`_infer_dynamic_dim_expr` is for — but it stops at the first producer it does not know, and it did not
know two of EnCodec's: **`pad`** (reflect-padding before every convolution in a residual block) and
**`elu`** (between every upsampling stage). Both were missing from a walk that already handled `conv`,
`conv_transpose`, `slice_by_index`, `matmul`, `linear` and eighteen unary passthroughs.

## The Fix, in Two Halves

* **Make the model's own arithmetic static** where it can be: the unpad becomes
  `[..., padding_left : -padding_right]` (the same elements, a compile-time-constant slice), and
  `_pad1d`'s trailing re-slice disappears because the extra padding is provably 0.
* **Teach the walk the two missing producers.** `pad` adds its constants to the padded axis; `elu`
  changes nothing.

After both, the emitted crops read `8*n_codes`, `40*n_codes`, `160*n_codes`, `640*n_codes` — the
running product of the upsampling ratios, whose last term is the checkpoint's own `hop_length`.
Verified against `transformers` at **max |Δ| 5.1e-07**.

## Takeaways

* **A shape-derived slice is where symbolic algebra goes to die.** When a model computes an index from
  `shape[-1]`, rewrite it to a static (negative) index at the tracing patch rather than letting the
  converter mint a fresh symbol. Two of EnCodec's three patches exist only for this, and neither
  changes a single output element.
* **"Fall back to the root axis" is a silent-wrong-answer generator, and it is load-bearing.** It
  cannot simply be removed — it is right for the common case — but every new op the walk does not know
  is a new way to hit it. The failure mode is always the same: no error, a plausible graph, a wrong
  length.
* **Assert on the emitted crop EXPRESSIONS, not on the call.** This is the third time in this family
  that the only symptom was a sample count. `tests/ci/test_encodec_export.py` pins the four crops by
  their expressions, and reverting either half of the fix turns it red.
* **When a new activation arrives, the op map is not the only place it belongs.** `elu` needed three
  entries: a primitive, a topology rule, and a line in the shape walk. The first two fail loudly if
  missed; the third does not.
