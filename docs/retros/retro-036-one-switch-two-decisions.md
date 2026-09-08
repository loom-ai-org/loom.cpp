---
type: retro
date: 2026-09-08
domain: performance
tags: [armv6, convolutions, kill-switch, measurement, raspberry-pi]
---

# Retro-036: One Switch, Two Decisions — and a Control Arm That Controlled Nothing

## The Issue

P7's closing measurement said that on a Pi Zero W `GGML_CPU_DISABLE_CONV_HEURISTICS=1` was worth
**1.27x** on a VITS synthesis (130.6 -> 103.0 s). Both the epic and the backlog then wrote the
follow-up item around a constraint drawn from that one number:

> **Do not simply hardcode the predicate to false.** The switch disables it for every shape, and the
> direct path exists partly to avoid materialising im2col -- at VITS's L=70400 that is 15.8 MB on a
> 427 MB board.

That switch does not disable one thing. It gates **two independent decisions** in
`ggml_compute_forward_conv_2d_impl`:

1. `ggml_conv_1d_direct_ok` — whether a convolution takes the direct sweep at all;
2. `ggml-0004`'s patch-batch budget — whether the im2col fallback stages 512 KB of patches at a time
   or fills the whole work buffer.

One variable, one number out, and nothing in a whole-model timing says which half moved. The item was
written as though all of it were (1), with the risk attributed to (2) — which is the combination that
makes the 15.8 MB look real.

## Root Cause Analysis

**The first attempt to split it was also wrong, and that is the more useful half of this retro.** A
per-node profile of both arms — same wheel, same session, `LOOM_PROFILE_NODES=1`, one thread — gives:

| dst shape (OL,1,OC,1) | calls | weights | direct sweep | im2col + GEMM |
|---|---|---|---|---|
| 70400,1,32,1  | 6  |  28 KB | **22529.7 ms** | 23714.6 ms |
| 17600,1,64,1  | 6  | 112 KB | 35290.3 ms | **22806.2 ms** |
|  2200,1,128,1 | 6  | 448 KB | 26598.5 ms | **11837.2 ms** |
|   275,1,384,1 | 28 | 1.47 MB | 17520.1 ms | 17854.1 ms |
|    94,1,192,1 | 44 | 147 KB | 3800.7 ms | 3833.6 ms |

The bottom two rows decline the direct path in **both** arms, so they looked like a free control for
decision (2): 17520 against 17854 and 3800 against 3833 is nothing, across 36% of the synthesis, so
the batch budget must be worth nothing and the whole saving must be the predicate.

It is not. **Those two shapes never batch.** The budget only engages when
`patch_total * space_per_patch > 3 * patch_budget`, and at 275 positions of 5376 bytes that is 1.48 MB
against a 1.57 MB threshold — under it, in both arms, so `wbudget` is the whole work buffer either
way. The rows were identical because the decision was never taken, not because taking it costs
nothing. A control arm has to *exercise* the thing it is controlling for.

Measured properly — one wheel, one session, VITS on 3.24 s of audio, with a knob that moves decision
(1) alone (`GGML_CPU_CONV1D_BUDGET`, added for this):

| | predicate | batching | VITS |
|---|---|---|---|
| budget 524288 (the shipped floor) | accepts all three | on | 149.20 / 125.03 s |
| budget 0 | off entirely | on | 113.14 / 115.88 s |
| `DISABLE_CONV_HEURISTICS=1` | off entirely | **off** | 99.62 s |

**1.197x is the predicate and 1.149x is the batch budget**, and 1.197 x 1.149 = 1.376 is all of what
the switch had been measuring. Half of a number that had already steered two rounds of scoping
belonged to the decision nobody was looking at.

## Resolution & Lesson Learned

The predicate's discriminator is the weight footprint, and `ggml_conv_1d_direct_budget()` was already
the knob for it — it was simply returning a number from another machine. `sysconf` reports 0 for every
cache level on an ARM1176, so the function fell to its 512 KB floor on a core with a 16 KB L1 and no L2
it can use. The direct sweep re-reads the whole weight tensor once per four-position block, one byte
of weights per MAC, so past L1 it is DRAM-bound; the turn between 28 KB and 112 KB in the first table
is that boundary. An ARMv6 arm returning **32 KB** routes all five shapes the way the table says, and
is worth **1.371x** on its own — as much as the whole switch, without touching the batch budget,
because it keeps the sweep on the one bucket where the sweep still wins. Declining every shape instead
gives up 14%.

The batch budget is now a sized item of its own rather than a footnote: 512 KB is 32x this core's L1,
so the cap buys none of the cache residency it was designed for and costs 1.149x in barriers and a
narrower GEMM.

* **Actionable takeaway 1 — a kill switch that gates two decisions cannot attribute a win to either.**
  It is a fine escape hatch and a bad instrument. The moment a number from one is used to *scope work*,
  split it: one switch per decision. Adding the second switch took ten lines and moved half the win to
  a different item.
* **Actionable takeaway 2 — a control arm must exercise the decision it controls for.** "These buckets
  are identical between the arms" proves nothing until you have checked that the code path under test
  actually runs for them. Here the threshold that skipped it was three lines above the branch being
  reasoned about.
* **Actionable takeaway 3 — a heuristic that reads a machine property is wrong twice on a machine that
  reports nothing.** It takes its floor, and it does so silently. `ggml_conv_1d_direct_budget`'s own
  comment says the floor is "for the machines that report nothing at all, which is every aarch64 Linux
  box tried" — on those it happened to be about right, which is why nobody looked again when a
  16 KB-L1 core landed on the same line.

## See Also

* [Retro-012](retro-012-optimizations-that-were-measured-out.md) — carries the reversal this corrects;
  the entry there measured a switch, and a switch was not one idea
* [Epic-08 §6.8](../epics/epic-08-packaging-and-release.md) — where the remaining ARMv6 performance is
* [ADR-026](../adrs/adr-026-armv6-is-the-floor-and-gets-its-own-kernels.md) — ARMv6 may have code of
  its own, and what has to be measured before it does
