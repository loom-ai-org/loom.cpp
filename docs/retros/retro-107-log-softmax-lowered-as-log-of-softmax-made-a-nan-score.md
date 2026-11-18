---
type: retro
date: 2026-11-17
domain: exporter
tags: [mil, log-softmax, nan, sorting, masked-diffusion, omnivoice, verification]
---

# Retro-107: log_softmax Lowered as log(softmax) Made a NaN Score

Serves [Epic-02](../epics/epic-02-mil-exporter-and-compiler.md) and [Epic-03](../epics/epic-03-model-coverage.md);
found building OmniVoice ([ADR-131](../adrs/adr-131-omnivoice-ships-as-one-file-and-streams-a-text-chunk-at-a-time.md)).

## Issue

The first OmniVoice export matched the reference on step 0 of 32: every id equal, scores within 7e-5.
The prompt and the time schedule matched too. The whole run still ended with 132 of 632 codes equal. A
bisect over `stop_after` found the first bad step: step 11, where one unmasked cell differed. But the
same graph, handed the same inputs from Python, picked the reference's cells, with a margin of 0.45.

## Root cause

One cell's score was **NaN**. coremltools lowers `torch.log_softmax` as `log(softmax(x))`. For a code
the model all but rules out, softmax underflows to 0 in f32 and `log` makes it -inf, in both guidance
halves. Guidance computes `c + g (c - u)`, which is `-inf - -inf` = NaN, and the next log-softmax spreads
it across the row. The scores' top-k then ran in the driver with a Lua comparator, where a NaN breaks
the ordering and won a slot. The Python check sorted with numpy's `argsort`, which puts NaN last, so it
picked the reference's cells and hid the value.

## Takeaway

**A comparison built on numpy can hide a NaN that the driver acts on.** Look for non-finite values in a
graph output before reasoning about margins; `np.isfinite(x).all()` is the first line. Separately,
**`log_softmax` is not stable after conversion.** Anywhere its -inf can meet arithmetic, spell it
`x - (m + log sum exp(x - m))` in the wrapper. That form needs a per-row max, which this work added
(POOL_1D over the last axis). With it, all 632 codes matched.

## Record

* Engine step 0 vs upstream: ids 632/632, scores 7.3e-5. `stop_after` bisect: steps 1-10 equal,
  step 11 picked cell 541 where the reference took 28.
* The same step with upstream's state after step 10, from Python: picks `[18, 28, 29, 141]` = the
  reference's. A repeat of the same call 11 times was identical, so call history was ruled out.
* Driver dump at step 11: ids equal, uniforms 5e-15 (print precision), scores 5e-14 apart, and
  `score[541] = nan`.
* Fix `omnivoice_export._log_softmax`; re-export: codes 632/632, waveform within 1.8e-5 (int16 flips).
