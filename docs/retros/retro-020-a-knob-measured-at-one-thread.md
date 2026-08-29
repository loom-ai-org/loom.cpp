---
type: retro
date: 2026-08-29
domain: performance
tags: [ggml-patches, threading, false-sharing, benchmarking, measurement-hygiene, p4.21, p4.22]
---

# Retro-020: A Knob Measured At One Thread, Shipped To Four

## The Issue

For four days the ledger recorded, as a settled fact, that **"`BM` is a cache knob, NOT the
mechanism"** — with evidence: `perf` instruction counts at whisper's `QK^T` shape are identical across
`m = 1496`, `1500` and `1504`, and all three call `mnpack<4, RN, BM>` with the same 4x`RN` register
tile. That entry was written to stop a plausible idea being re-proposed, and it was correct about
everything it measured.

It was measured at **one thread**. At four, `BM` is not a cache knob at all:

| `m` | `BM` | bytes of `C` one job writes | 1 thread | 4 threads | scaling |
|---:|---:|---:|---:|---:|---:|
| 1496 | 2 | 32 | 29.4-30.5 ms | 21.2-21.9 ms | 1.40x |
| **1500** | **1** | **16** | 29.6-30.9 ms | 30.0-31.5 ms | **0.98x** |
| 1504 | 4 | 64 | 29.4 ms | 10.6-11.0 ms | **2.75x** |

`gemm()` gives one job the rows `[ii, ii + BM*RM)`, `C` is `m`-contiguous, and `BM*RM*4` is therefore
how many **bytes of a 64-byte cache line** a job owns. At `BM = 1` four threads write four quarters of
the same line for every column of the matmul. `perf stat -e task-clock` reports **3.65 CPUs utilised
for a 1.02x speedup**: the threads run, they just pass a line back and forth.

whisper-small's encoder is `m = 1500`. **Four rows of padding were worth 2.8x**, and the ledger said
the knob was inert.

## Root Cause Analysis

**1. The measurement was single-threaded because the repository's own advice says to profile that
way**, and that advice is right for what it is for. `include/loom/core/profile.h` warns "profile with
ONE thread, or dispatch cost dominates", and every per-op attribution in Epic-05 follows it. But a
coherence protocol has no cost at one thread by construction. **The instrument was blind to this class
of bug, and the blindness is the instrument's design goal**, not a mistake in using it.

**2. Instruction counts cannot see it either, and that is what made the verdict feel solid.** The
counts across the three `m` really are identical — the same instructions, on the same tile, in the
same order. What differs is who owns the line they store to. An instruction count is a per-thread
quantity; false sharing is a between-thread one.

**3. Nothing else pointed at it.** The shape is 12% of a transcription, the engine defaults to 4
threads, and whisper's Pi numbers — the reference target — are healthy, because the Pi does not have
the problem at all. There was no symptom to chase.

## Resolution & Lesson Learned

Fixed as **P4.22** (`cmake/patches/ggml-0012`): run the `m - (m % 16)` prefix at `BM = 4` and finish
the <= 12 leftover rows in a column-split loop, guarded on `nth > 1`. In model the `QK^T` bucket goes
**391.2 -> 185.9 ms (2.10x)** and it is the only bucket of forty that moves; the transcription is
**4.050 -> 3.858 s at 4 threads**. Output is bit-identical.

* **Actionable takeaway 1 — a "no effect" verdict inherits the thread count it was measured at, and
  should say so.** [Retro-012](retro-012-optimizations-that-were-measured-out.md) is the register of
  things not to re-propose, and its entries are trusted precisely so nobody re-derives them. An entry
  that does not name its conditions is an entry that will be over-trusted. The `BM` line now says "at
  one thread"; **every negative result in that file should carry the axis it was swept on.**
* **Actionable takeaway 2 — anything whose unit is BYTES OF A CACHE LINE needs a multi-threaded
  measurement, always.** Tile heights, job granularity, chunk sizes, per-thread output strides: if a
  parameter decides how much of a line a thread owns, one thread cannot measure it. This is the
  threading analogue of [Retro-019](retro-019-a-patch-measured-on-one-isa.md)'s ISA rule, and it wants
  the same standing form: **a scheduling change is not measured until it has a number at 1 thread and
  a number at 4.**
* **Actionable takeaway 3 — it was found by an experiment aimed at something else, and only because
  both arms were in one harness.** `scripts/bench16.cpp` was built to gate P4.21, a proposal about the
  vector lane axis. It had to run the incumbent alongside the candidate at the same thread count, and
  the incumbent not scaling was visible in that comparison and nowhere else. **A measuring stick that
  only measures your own proposal cannot tell you your proposal is not the problem.** P4.21 was
  measured out; the experiment that killed it paid for itself several times over.
* **Actionable takeaway 4 — "no change" on the second box is a result, and getting it wrong is easy.**
  The Pi is neutral here (it threads 3.5x at every `m`; four small cores behind a shared 1 MB L2 do
  not pay for a contended line the way a 24-core mesh does). A first pass read it as **1.9% slower** —
  the box warmed 54 -> 82 C across the run and the loop sampled base first in every pair, so the
  thermal ramp was attributed to one arm. ABBA ordering made it 133.10 vs 133.65 ms, i.e. nothing.
  **Interleave ABBA or do not quote the number**, on the Pi especially.

**What NOT to conclude:** that P4.18's `BM` measurement was sloppy. It answered the question it was
asked — "does `BM` explain the single-threaded instruction count at `QK^T`?" — correctly, and the
answer is still no. The failure was in how the answer was *filed*: as a property of the knob rather
than as a property of the knob under one condition.
