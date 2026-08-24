---
type: retro
date: 2026-08-24
domain: performance
tags: [benchmarking, methodology, estimator, onnxruntime, readme, p4.17]
---

# Retro-018: A Published Table Of Ratios That Nobody Could Re-Derive

## The Issue

`README.md` carried a four-machine loom-vs-onnxruntime table. Asked to refresh it after P4.17, the
obvious cheap route was to measure loom's own before/after factor and scale each published ratio by it
— the onnxruntime side had not changed, so its times should cancel.

**The scaled numbers did not survive their first check.** For TTS on the 285K at 4 threads, scaling
predicted 1.35x; measuring onnxruntime directly gave **1.03x**. The published ratios could not be
reproduced from either side, and there was no harness in the repository that produced them: loom's side
existed (`bench_vits_loom.cpp`, `bench_lm_loom.cpp`), the onnxruntime side did not, except for a
Pi-specific TTS script with a hardcoded model path.

## Root Cause Analysis

Two independent faults, and the second is the one worth remembering.

**1. Ratios were carried forward rather than re-measured.** Each table inherited numbers from the one
before it, so an error in any generation propagated silently and no single commit contained both halves
of any ratio.

**2. The two sides used DIFFERENT ESTIMATORS, and nothing said so.** `bench_lm_loom.cpp` timed **one
cold generation**; the onnxruntime harness warmed up and took the best of N. Both are defensible; the
pair is not. `bench_vits_loom.cpp` and `bench_asr_loom.cpp` both take a median over runs in a warm
process, so the LM harness was also the odd one out **within loom's own set** — the cold run is
reproducibly ~5% slower than the ones after it.

Correcting only that, changing nothing else, moved the LM column across every machine:

| machine | mismatched | matched estimators |
|---|---|---|
| 285K @4 | 0.99x | **1.02x** |
| 285K @24 | 0.95x | **1.03x** |
| Ryzen @4 | 0.97x | **1.05x** |

**It flipped the sign of the result on all three.** "loom loses the LM everywhere" and "loom is a few
percent ahead everywhere" differed by which run of loom's own harness was being timed.

## Resolution & Lesson Learned

`scripts/bench_onnx_tasks.py` now exists — the onnxruntime side of all three tasks, driving `ort`
directly rather than through `optimum` — alongside a new `scripts/bench_asr_loom.cpp`, so every cell of
the table can be re-derived from this repository. `bench_lm_loom.cpp` warms up and repeats like its two
siblings. The table is now measured with both engines run back to back on each machine.

**Four lessons:**

1. **Never scale a published ratio; re-measure both sides or quote neither.** A ratio whose two halves
   were measured at different times, by different people, with harnesses that may not both still exist
   is not a measurement — and this one was wrong by 30%.
2. **Matching the ESTIMATOR matters as much as matching the workload.** This thread already had rules
   for equal work — pin VITS's scales, compare ASR transcripts, difference the LM — and every one of
   them held here. A 5-7% estimator mismatch still flipped a column's sign. Warm-up, median vs best,
   and run count are part of the comparison, not the plumbing around it.
3. **A benchmark that ships only one side of its comparison will drift.** Whichever side has no script
   is the side that silently becomes unreproducible; keep both in the repository or expect to redo the
   work.
4. **Repeating a benchmark can be incorrect, so check before assuming it is not.** Repeating
   `infer_with_past` is only valid because it re-primes the KV cache on entry; a cache leaking state
   across calls would make run 2 both wrong and faster. The harness now asserts that every run returns
   identical tokens rather than trusting it.

**What NOT to re-propose:** scaling the table by a measured engine-side factor. It was tried, it was
30% out on the first cell checked, and the arithmetic is not the problem — the published baseline is.
