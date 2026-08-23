---
type: retro
date: 2026-08-22
domain: performance
tags: [benchmarking, reproducibility, stochastic-models, measurement-hygiene]
---

# Retro-010: An Unpinned Baseline Made Every Ratio 4% Optimistic

## The Issue

Every loom-vs-onnxruntime ratio recorded during P4.14 was derived from a single onnxruntime wall time
of **1.024 s**. That number is not reproducible, and every ratio built on it was ~4% optimistic.

## Root Cause Analysis

VITS's duration predictor is **stochastic** and the reference host does not seed it, so each run
synthesises a *different number of output samples* — while both engines' time is very nearly linear in
output samples. The 1.024 s timed an unknown length. Pinned, onnxruntime is **1.044 s at 72192 samples,
1.063 s normalised to loom's 73472**, repeatable to 0.5%.

## Resolution & Lesson Learned

Pin the stochastic path (`SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)`, or `scales=[0,1,0]`
driving the session directly) before quoting any ratio. The recipe is in
[Epic-05's operating notes](../epics/epic-05-edge-performance.md#operating-notes-benchmarking).

* **Actionable takeaway 1 — a generative model's output length is part of the measurement.** Normalise
  to samples produced, or pin the sampler. A wall-clock comparison between two engines producing
  different amounts of audio is not a comparison.
* **Actionable takeaway 2 — per-op *shares* survived this; the *wall time* they were apportioned over
  did not.** Structuring the profile as shares plus one separately-measured wall time limited the blast
  radius of a bad baseline to a single number.

---

## Full record (verbatim from the ledger)


> **CORRECTION (2026-08-22): the 1.024 s is not reproducible, and every ratio in this section derived
> from it is ~4% optimistic.** VITS's duration predictor is stochastic, phoonnx does not seed it, and
> each run therefore synthesises a DIFFERENT number of samples — while both engines' time is very
> nearly linear in output samples. 1.024 s timed an unknown length. Pinned
> (`SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)`, or `scales=[0,1,0]` driving the session
> directly) onnxruntime is **1.044 s at 72192 samples, 1.063 s normalised to loom's 73472**, repeatable
> to 0.5%. The per-op SHARES below are unaffected — only the wall time they are apportioned over moves,
> and by less than the 1.18x profiling cost already accounted for. **Pin it before quoting any ratio;**
> see the recipe in P4.15b's cold start.

Both engines profiled per-op on the SAME model and utterance — loom via P4.14, onnxruntime via its own
`enable_profiling` (shares only; profiling costs it 1.18x, so shares are apportioned over the
un-profiled 1.024 s).

| | loom | onnxruntime | delta | share of gap |
|---|---|---|---|---|
| convolution | 1.550 s | 0.605 s (`Conv`+`FusedConv`) | 0.95 s | **71%** |
| transposed conv | 0.189 s | 0.162 s (`ConvTranspose`) | 0.03 s | 2% |
| everything else | 0.61 s | 0.257 s | 0.35 s | 27% |

The convolution carries 15.578 GFLOP: loom runs it at **10.0 GFLOP/s**, onnxruntime at **25.7
GFLOP/s**, against a Pi 4 fp32 peak of ~57.6 GFLOP/s (4 cores x 1.8 GHz x 4 lanes x 2). So MLAS reaches
45% of peak and ggml's generic F32 `mul_mat` reaches 17%, on identical arithmetic. That single ratio is
most of the 2.2x, it is NOT the im2col materialisation (removing it changes nothing, above), and
`GGML_LLAMAFILE`'s tinyBLAS is present and does run for these shapes (`n >= 4`) and measures no better.
**The gap is F32 micro-kernel quality in the large-M / small-N (32-384) / medium-K regime**, which is
a ggml-side concern, not something the exporter or a new backend can reach. The remaining 27% is
onnxruntime fusing conv+bias+activation where loom emits separate ADD/activation nodes, plus loom's
165 ms of graph build, Lua driver and marshalling.

