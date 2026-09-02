---
type: retro
date: 2026-09-02
domain: performance
tags: [benchmarking, methodology, estimator, placement, hybrid-cpu, onnxruntime, readme, p4.30b]
---

# Retro-025: The Arm That Ran Second Paid For The One Before It

## The Issue

P4.30b step 2 asked for one thing: re-sample the README's TTS and LM columns **per launch**, because
thread placement on the Core Ultra 9 285K is chosen once per process and a within-process median
cannot average it out. The protocol seemed settled — the ASR column had already been re-sampled that
way in August, and the tooling existed.

The first sample said the 285K's 24-thread TTS cell had moved from the published **1.17x to 1.65x**.
Nothing in the engine explained a 40% jump, and both engines were running the byte-identical models
the published cells were measured on.

## Root Cause Analysis

**Three faults, and only the third is new.**

**1. The onnxruntime VITS arm had no warm-up, and nobody had opened it.** `bench_onnx_tasks.py`'s `lm`
and `asr` tasks both warm up; its `vits` task did not, while `bench_vits_loom.cpp:100` discards its
first synthesis because that call is what builds the graphs. This is exactly
[Retro-018](retro-018-a-table-of-ratios-nobody-could-re-derive.md) and exactly its correction — and it
survived both. The correction's lesson was *"when a retro's root cause is 'these N things disagree',
open all N"*, and the audit behind it opened **loom's three harnesses**. The onnxruntime side is a
fourth thing, in the same comparison, and no one opened it. **N was never three.**

**2. A MEDIAN is the wrong estimator for a bimodal arm.** At 24 threads both engines' VITS time splits
into two per-launch modes about 1.4x apart at roughly even odds. A median over that is a coin flip
between the two modes, so the published cell could read 1.19x or 1.66x with nothing changed — and the
README's rule for every cell is "a median of repeated runs". The mean does not have that property.
`paired_arms.py` now prints both and warns when an arm's samples split into two weighted clusters.

**3. Back-to-back arms hand each other a contaminated machine, and ABBA does not cancel it.** This is
the new one. Interleaved with loom, onnxruntime landed in its slow mode in **29 of 31 launches**; run
alone in a plain loop on the same box in the same session, **10 of 20**. Inserting one second between
arms — `paired_arms.py --between "sleep 1"` — restored it to 21 of 31 and moved the cell from 1.41x to
**1.20x**, which is the published 1.17x back again.

The pairing machinery was built for *drift*, which moves both halves of a pair together and therefore
cancels. This does not: the first arm's threads are still winding down when the second launches, and
what it costs the second arm is a **placement** decision that then sticks for that whole process.
Alternating the order does not help, because both orders penalise whoever is second. It is not
symmetric between the two engines either — ggml's threadpool spins where onnxruntime re-decides
placement at session creation, so the same hand-off costs them different amounts.

**What the two modes actually are, on this part.** The 285K is 8 P-cores (`cpu_core` = 0-7) and 16
E-cores (`cpu_atom` = 8-23), no SMT. Pinned to one cluster, every number is reproducible to ~1%:

| VITS, 4 threads, 5 launches each | loom | onnxruntime | ratio |
|---|---:|---:|---:|
| P-cluster (`taskset -c 0-3`) | 0.0624-0.0628 s | 0.0646-0.0661 s | **1.034-1.053** |
| E-cluster (`taskset -c 8-11`) | 0.0916-0.0927 s | 0.1033-0.1075 s | **1.114-1.165** |

So the "1.48x lottery" the README described is not noise around a number — it is **which cluster the
process landed on**, and the two clusters give *different ratios*. Whichever cluster a four-thread
process gets, it keeps for its whole life.

## Resolution & Lesson Learned

`bench_onnx_tasks.py`'s `vits` task warms up. `bench_lm_loom.cpp` takes a `refeed` argument, so the
`infer` diagnostic arm — 1.4x the timed one, and 65 s a launch on a Pi — can be skipped where nothing
reads it. `bench_vits_loom.cpp` takes a real median rather than `ts[ts.size()/2]`, which at an even `nrun` is the larger of the two middle samples — the same index
Retro-018's correction caught in the ASR harness and left here. `bench_lm_loom.cpp` prints its first
five tokens, so the LM's equal-work check is *shown* rather than asserted, like the sample count VITS
prints and the transcript ASR prints. `paired_arms.py` grew a `--cmd` mode (two engines, not two builds
of one binary), per-arm `--metric`, a named `--ratio`, mean alongside median, and a bimodality warning.

**Four lessons:**

1. **On a hybrid CPU, WHICH CORES is part of the measurement, not noise around it.** A cell that does
   not say whether the process was on P-cores or E-cores is under-specified by 1.47x on this part, and
   the ratio it reports differs by a tenth between them.
2. **Let the machine settle between arms.** One second is enough here. A benchmark harness that runs
   its arms back to back is measuring the hand-off as well as the arms, and alternating the order does
   not cancel it because both orders punish whoever is second.
3. **Check the estimator against the SHAPE of the samples, not just against the other arm.** Matching
   estimators (Retro-018's lesson) was necessary and was not sufficient: two matched medians over two
   bimodal arms are still two coin flips.
4. **"Open all N" has to include the other engine.** Two of the three faults above are Retro-018's own,
   found in the half of the comparison its audit did not open. The check is one line —
   `grep -n 'warm' scripts/bench_*_loom.cpp scripts/bench_onnx_tasks.py` — and it has to name **both**
   globs.

**And the board with the most protocol produced the tightest numbers.** The Raspberry Pi 4B was off the
network when the above was measured and was re-sampled two hours later. It has a *stronger* settle than
anything invented here — `cool.sh 60`, waiting for the SoC to reach a fixed temperature before every
arm, introduced for thermal drift and not for placement at all. Cooled, it holds a **1.02-1.05x** spread
per engine across launches where the 285K gives 1.45x, and **both its cells resolve** (TTS 0.96x at
p10 0.947/p90 0.977, LM 1.03x at p10 1.027/p90 1.041) where only three of the eight x86 cells do. Two
consequences: the Pi's "report the fastest run" special case is retired, because a cooldown before every
arm removes the drift that estimator was working around; and the cheapest way to read this whole retro
is that **the Pi had the answer first, for the wrong reason.**

**What NOT to re-propose:** pinning both engines as the fix. It is how the P/E decomposition above was
*measured*, and it is the right way to state what the table means, but it constrains onnxruntime more
than loom (pinned to P-cores its whisper time is worse than its lucky unpinned launches), so a table
built on it would not be like-for-like. The settle is the fix; the pinning is the explanation.
