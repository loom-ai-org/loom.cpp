---
type: retro
date: 2026-09-26
domain: exporter
tags: [exporter, verification, codec, family-9, voxtral, f64]
---

# Retro-062: An f32 Wrapper Check Could Not Tell a Spelling From a Defect

## The Issue

Voxtral-4B-TTS's export re-spells vllm-omni's codec decoder for the trace. The export checks each
wrapper against the upstream module before tracing ([[feedback-a-wrapper-owes-the-tensor-it-took-over]]).
On random codes the codec wrapper was **2.15e-05** from upstream, relative to the peak, against a
1e-5 bar. The same comparison on another draw of codes gave 2.6e-06.

The first reading was "f32 noise": the decoder is eight windowed-attention layers and four
convolutions deep. The bar could have been raised until the check passed.

## Root Cause

The comparison was repeated at **float64**. If the gap were rounding, it would shrink to ~1e-14 there.
It stayed at **2.2e-06**, so it was a real difference. A block-by-block bisect showed where:

* The **quantizer's output** differed by 4.8e-08 at f64. Upstream's `_rescale` computes
  `codes * 2 / (levels - 1) - 1` in f32 whatever the model's dtype: an integer code, times two, divided
  by 20. The wrapper spelled it `c * 0.1 - 1`. `0.1` is not exact in binary, so some codes land one
  ulp away.
* Fed upstream's exact quantizer output, **every block agreed to 1e-14**.
* The decoder then carried that one ulp at its input to **~60x** at its output (3e-6).

So the wrapper had a spelling difference, not a defect in the network. But no f32 tolerance could have
separated the two: a check loose enough to pass the spelling would pass a real 1e-5 error just as well.

## The Fix

The rescale is spelled as upstream spells it (`c * 2 / 20 - 1`). The wrapper check now runs at
**float64** on separately built f64 copies of the upstream modules, with a **1e-9** bar
(`voxtral_tts_export.check_wrappers` / `compare_wrappers`). It measures 0.0, 0.0 and 2.3e-14 on the real
weights, and a 0.1% skew fails it by name (`test_the_wrapper_check_catches_a_scaled_rescale`). The
export's own phases are still built at f32.

## Takeaway

**Check a re-spelling at f64 whenever the module amplifies its input's rounding.** At f32 such a
check measures the amplifier, not the wrapper. At f64 an exact spelling is exact, so any gap is a
finding. Before raising an f32 tolerance, run the comparison at f64 once: if the gap does not
collapse, it is not noise. This is [[feedback-run-the-reference-at-f64]]'s rule applied to the
exporter's own checks.

The bisect also showed what the lowering must preserve. An integer-to-float conversion that the
reference does in f32 must be done in f32, with the reference's operation order.

## Related

* [Epic-03](../epics/epic-03-model-coverage.md) family 9, Voxtral-4B-TTS
* [Retro-055](retro-055-a-feedback-loop-cannot-be-gated-free-running.md): the f64 arm's other use, as a
  drift bound
