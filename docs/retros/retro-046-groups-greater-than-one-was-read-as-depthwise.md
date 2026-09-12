---
type: retro
date: 2026-09-12
domain: exporter
tags: [conv, lowering, family-4, ggml, verification]
---

# Retro-046: `groups > 1` Was Read As "Depthwise", And Then A View Read The Wrong Layout

## The Issue

Family 4's first export — `data2vec-audio-base-960h`, a CNN + transformer + CTC model — aborted the
**engine**, not the exporter:

    ggml.c:4468: GGML_ASSERT(b->ne[1] == a->ne[1]) failed
    #3 ggml_im2col
    #4 loom::op_conv_1d_dw

No op name, no model name, no channel count. Fixing that produced a second failure of the opposite
kind: the model ran, every shape was right, and the logits were wrong by 21 with **80% of the argmaxes
still agreeing**.

## Root Cause Analysis

**1. The exporter's depthwise test was `groups > 1`.** `topology_ops._op_conv` mapped any grouped
convolution to `CONV_1D_DW`/`CONV_2D_DW`. Depthwise actually means *one input channel per output
channel* — `IC/groups == 1` — and the two conditions coincide at both ends of the range:

| | `groups` | `IC/groups` | emitted |
|---|---|---|---|
| dense | 1 | IC | CONV_1D ✓ |
| depthwise | IC | 1 | CONV_1D_DW ✓ |
| **grouped** | **16** | **48** | CONV_1D_DW ✗ |

Every convolution converted in eight families sat in one of the first two rows — verified over the
shipped topologies, where every `CONV_1D_DW` has `groups` equal to a channel count. wav2vec 2.0,
HuBERT and data2vec put their positional convolution in the third, and it is the first model in the
zoo that does.

**The `groups` attr had been written on every conv node since the first export and read by neither
op.** It was correct metadata that nothing consumed, which is exactly why the mislabel could not be
caught by anything downstream.

**2. `ggml_view_3d` sets `nb[0]` to the type size, unconditionally.** With `CONV_1D` lowering a
grouped convolution as G slices and a concat, each group's activation slice was
`ggml_view_3d(data, IL, IC/G, N, data->nb[1], data->nb[2], offset)` — which takes nb1 and nb2 and
*nothing else*. The activation reaching a positional convolution is PERMUTED (this family transposes
`[T, C]` to `[C, T]` in front of every convolution), so `nb[0]` is not the type size and the view
silently reinterpreted the layout instead of failing. Every group ended up reading the same wrongly
strided window.

## The Fix

* **The exporter decides depthwise on the KERNEL's shape** — MIL declares a conv weight as
  `[OC, IC/groups, K]`, so `IC/groups == 1` is a structural read of the thing that knows. A grouped
  2-D convolution raises naming the gap rather than silently dropping the attr.
* **`CONV_1D` honours `groups`**, as G slices re-entering `op_conv_1d` and one `ggml_concat` —
  composition rather than a fourth code path, so each slice still gets whichever of the three existing
  lowerings it qualifies for (the direct sweep, the folded block-quantized sweep, im2col + mul_mat).
* **`conv_1d_grouped` makes the activation contiguous before slicing.** Also the cheaper spelling:
  `op_conv_1d` would otherwise materialise the same copy `groups` times.
* **`CONV_1D_DW` now rejects a kernel with `ne[1] != 1`**, naming the op and the number, instead of
  aborting the process inside ggml.

Verified against `transformers` on the LOGITS at 11 s of real speech: max |Δ| 1.9e-03 (data2vec),
1.7e-03 (HuBERT), 5.5e-04 (omniASR-CTC), cosine ≥ 0.99999982, 549/549 frames agreeing on the argmax,
sabotage arm 32.7.

## Takeaways

* **An attribute nothing reads is a claim nothing checks.** `groups` was emitted correctly by the
  exporter and ignored by both consumers for eight families. The value being right is not the same as
  the value being used, and there is no test that can tell the difference until a model needs it.
* **A predicate that is right at both ends of a range will be wrong in the middle, and the middle
  arrives late.** `groups > 1` was indistinguishable from `IC/groups == 1` across every model in the
  zoo. The fix is not a wider test suite — it is to write the predicate as the property it means
  (*one input channel per output channel*) rather than as the proxy that happened to separate the
  cases in hand.
* **`ggml_view_*` does not validate the layout it is cutting from.** It takes the strides it is given
  for ne[1] and ne[2] and assumes the type size for ne[0]. A view taken from anything that might be
  permuted has to be made contiguous first, and the failure mode is a plausible graph with a wrong
  answer — no assert, no shape mismatch.
* **80% argmax agreement is what a wrong tensor looks like.** The transcript was one letter. The
  standing *tensor oracle, not token oracle* rule paid for itself again, and so did the f64 arm: once
  it matched, torch's own f32 measured 7.0e-4 from f64 against loom's 1.2e-3, which is what says the
  remaining gap is accumulation over a deep graph rather than a defect.
