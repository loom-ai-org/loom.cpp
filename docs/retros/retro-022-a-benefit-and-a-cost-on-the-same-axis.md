---
type: retro
date: 2026-08-30
domain: performance
tags: [ggml-patches, aarch64, benchmarking, measurement-hygiene, p4.26]
---

# Retro-022: A Benefit And A Cost That Scale On Opposite Sides Of The Same Axis

## The Issue

`ggml-0012` shipped as **2.75x** at whisper's `QK^T` on a 24-core x86 box. It cost a Raspberry Pi 4
**2.4% on VITS** — 27 ms per synthesis, of which `$LOOM_PROFILE` put 22.6 ms in `CONV_2D`, an op the
patch's own reasoning never mentions. It moved the README's Pi TTS cell from 0.96x to 0.93x with no
change to the harness, and it was found the next day while re-measuring something else.

This is [Retro-019](retro-019-a-patch-measured-on-one-isa.md) one patch later, and the first reading
was the same: *a patch measured on one ISA, shipped to both*. That reading was wrong, and it would have
produced a wrong fix.

## Root Cause Analysis

The patch replaces `m % 16 == 0` with "take the largest 16-aligned prefix of `m`", so a ragged `m`
gets a job that owns a whole cache line of `C` instead of a quarter of one. What it removes is a
**per-output** cost: `m*n` contended stores, one per element of `C`, no matter how long the contraction
is. What it adds is **per-work**: a job's row block is four times taller, so one job holds
`RM*BM*k*4 = 16k` bytes of `A` instead of `4k`, and there are four times fewer jobs to spread across
the threads.

**The benefit therefore decays as `1/k` and the cost does not.** Sweeping `k` at `m = 284, n = 384`,
4 threads, both arms in one process (`scripts/bench19.cpp`), ratio against the schedule the branch
replaces:

| `k` | 64 | 128 | 192 | 256 | 384 | 512 | 768 | 960 | 2304 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Core Ultra 9 285K | **2.035** | **1.317** | 1.046 | 1.030 | 1.005 | 0.996 | 0.987 | 0.983 | 0.974 |
| Cortex-A72 | 1.014 | 1.011 | 1.005 | 0.997 | 0.995 | 0.977 | 0.959 | **0.912** | 0.901 |

Monotone on both machines, crossing 1.0 within a factor of two of each other. **The ISA is not the
axis. `k` is.** `ggml-0012` was measured at `k = 64` — an attention head dimension — and shipped for
every `k` there is.

The reason it reached a TTS vocoder at all is that `ggml-0004` and `ggml-0009` lower **convolution**
and transposed convolution through the same `sgemm`, where `k` is `in_channels * kernel_width` and runs
576 to 2304. A census of one VITS synthesis (`GGML_SGEMM_CENSUS=1`) puts **10.9 of its 14.4 GFLOP**
through matmuls this branch newly accepted.

## The Fix

One clause, `k <= 256`, on the ragged prefix only — an `m` that already divides 16 keeps the schedule
it always had. It keeps every x86 win, including a whisper `QK^T` that goes from 1.809x to **1.941x**
because the shapes it now declines were the ones dragging it down, and it returns the Pi to parity with
the pre-patch tree (12 paired ABBA rounds, cooled to a fixed 60 C: **1.003**, p10 0.985, p90 1.018).

**The obvious fix would have been wrong in both directions.** `!defined(__aarch64__)` keeps the
`k >= 512` losses on x86 and throws away the `k <= 192` wins on aarch64.

## Takeaway

**When a change trades a fixed cost against a scaling one, the ratio between them is an axis, and one
point on an axis is not a measurement of it.** Before shipping a scheduling or blocking change, name
what it removes and what it adds, work out which quantity each one scales with, and sweep that
quantity — not just the machine.

Two corollaries this cost real time to learn:

* **A shared kernel carries a patch into every caller.** `sgemm` here is attention *and* convolution.
  A predicate justified by one caller's shapes needs the other caller's shapes measured, and a census
  of what a model actually issues (`GGML_SGEMM_CENSUS=1`, ~40 lines) settles in one run what reading
  the patch cannot.
* **Put both arms in one process.** `scripts/probes/ggml-p426-sgemm-policy-probe.patch` makes the
  predicate a run-time switch, so an A/B is one binary, one allocation, one page cache and one branch —
  and `scripts/paired_arms.py --env` then pairs them. P4.22 and this item's first pass were both
  measured by building two whole trees, which cannot make that claim, and the resolution difference is
  the whole result: the clock witness holds 1% where two trees hold 3%.

The amendment to [Retro-019](retro-019-a-patch-measured-on-one-isa.md)'s "measure every ISA a patch is
enabled for" is **every regime it is enabled for**, and the ISA is only one of them.
