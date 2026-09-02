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

---

## CORRECTION (2026-08-28): the same bug was in `bench_asr_loom.cpp`, and this retro said it was not

The root-cause section above states that "`bench_vits_loom.cpp` and `bench_asr_loom.cpp` both take a
median over runs in a warm process, so the LM harness was also the odd one out". **The second half of
that was false.** `bench_vits_loom.cpp:63` and `bench_lm_loom.cpp:67` each discard a warm-up run;
`bench_asr_loom.cpp` had none. Its ASR counterpart in `bench_onnx_tasks.py` does
(`text = run()  # warm up`), so **the ASR column of the README table was loom's cold run against
onnxruntime's warm one** — the exact fault this retro is about, in the one column that still showed a
loss, left in place by a fix that assumed it had been checked.

There was a second fault in the same seven lines. `times[times.size() / 2]` on **two** samples is
`times[1]` — the **larger** of the two, not a median. At `nrun=2` the harness therefore reported the
max of a cold run and a warm one.

Measured on 2026-08-28, cold/warm for whisper-small: **1.25-1.7x at 24 threads on the 285K**, 1.02x at
four on the same box, and *below* 1.0 on the 2-core Ryzen, where the box heats up faster than the first
run pays for itself. It is a thread-count effect, so it hit hardest exactly where the README claimed
loom's one ASR win.

**Together the two faults are 1.43x** on the 285K at 24 threads: nine launches at `nrun=2` with no
warm-up give a median of **1.650 s**, nine at `nrun=5` with one give **1.157 s** — same binary, same
clip, same box, same afternoon.

**The lesson is not lesson 2 again — it is what "fixed" meant.** Lesson 2 was correct and was acted on;
what failed is that the audit behind it checked one harness and asserted the other two, and the
assertion was written into the retro as a finding. **When a retro's root cause is "these N things
disagree", open all N.** The fix is four lines in each file and the check is one `grep -n warm
scripts/bench_*_loom.cpp`.

`bench_asr_loom.cpp` now discards a warm-up run and prints it, so the asymmetry is visible in the
output rather than implicit in the estimator, and the comment there says to use `nrun >= 3`.

---

## SECOND CORRECTION (2026-09-02): the onnxruntime side had the same two faults

The correction above says the fix is *"one `grep -n warm scripts/bench_*_loom.cpp`"*. **That glob is
half the comparison.** `bench_onnx_tasks.py`'s `vits` task had **no warm-up** while
`bench_vits_loom.cpp:100` discards its first synthesis, and `bench_vits_loom.cpp` still took
`ts[ts.size()/2]` — the same not-a-median index this correction caught in the ASR harness and fixed
only there. Both were found by P4.30b while re-sampling the TTS column, five days later. Both are
fixed, and the check is now `grep -n 'warm' scripts/bench_*_loom.cpp scripts/bench_onnx_tasks.py`.

Neither was worth much on its own — a missing warm-up biases a median over nine runs by 1/9 of the
cold/warm gap — but the direction matters: it was in **onnxruntime's** arm, so it flattered this
repository, which is the direction a repository's own benchmark is least likely to question.

**The lesson is this retro's own, applied one level out.** Lesson 4 of the correction is *"when a
retro's root cause is 'these N things disagree', open all N"*, and the audit it prompted opened
loom's three harnesses. The other engine's three tasks are the fourth, fifth and sixth things, in the
same comparison, and nobody counted them. **N is every arm of the comparison, not every arm on your
own side.** [Retro-025](retro-025-the-arm-that-ran-second-paid-for-the-first.md) is where that landed,
along with the finding that actually moved cells: a benchmark that runs its two arms back to back
measures the hand-off between them as well.
