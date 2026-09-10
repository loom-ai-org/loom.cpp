---
type: retro
date: 2026-09-09
domain: performance
tags: [armv6, gemm, cache, conv-transpose, raspberry-pi]
---

# Retro-038: Two Panels, One Cache Way, and Three Hypotheses That Died First

## The Issue

`CONV_TRANSPOSE_1D` was the last unexamined item in the ARMv6 convolution path. The hub described it
as *"11.5% of a VITS synthesis and never looked at here — whether its inner loop holds an accumulator
array of the kind that cost 2.3x in `ggml_conv_1d_direct_tile_impl` is unknown. One benchmark answers
it."*

Three claims, and the investigation killed all three:

* it is **6.5%** of a synthesis, not 11.5%;
* there is **no accumulator array** — `ggml-0009` already made the compute a GEMM that routes through
  `ggml_compute_forward_mul_mat` and reaches `tinyBLAS_F32_ARMV6` like everything else;
* one benchmark did not answer it. It took three, and the first two answered the wrong question.

What was actually wrong is not ARMv6-specific and is not in the kernel.

## Root Cause Analysis

**Attribution first.** `scripts/bench35.c` times the op's three non-GEMM phases — kernel repack,
activation transpose, overlap-add — at VITS's three real shapes, read off the model rather than
guessed. Subtracting them from a per-node profile gives each node's GEMM rate:

| node | profile | non-GEMM | GEMM | MMAC | rate |
|---|---|---|---|---|---|
| K=16 Cout=128 Cin=256 L=275 | 1638.6 ms | 59.0 | 1579.6 | 144.2 | **91.3 MMAC/s** |
| K=16 Cout=64 Cin=128 L=2200 | 1521.2 ms | 146.2 | 1375.0 | 288.4 | 209.7 |
| K=8 Cout=32 Cin=64 L=17600 | 1653.4 ms | 339.7 | 1313.7 | 288.4 | 219.5 |

That killed the first guess made *during* the investigation as well: the first node repacks 2 MB of
kernel with a 1 KB-strided scatter on every call, which looks like the whole problem and costs
**28.4 ms**.

**Then the wrong axis.** The first node differs from the others in both `n` (32, from
`GGML_CONV_TRANSPOSE_1D_TILE / mk`) and `k` (Cin). `n` was the attractive hypothesis, because a cache
constant sized for a bigger machine is exactly what the two preceding P7.1 items turned out to be.
`scripts/bench36.c` swept both, and **raising `n` makes it worse** — 91.5 MMAC/s at n=32 down to 72.1
at n=272. The axis is `k`, and it is a cliff rather than a slope: at m=2048 n=64, **280.4 MMAC/s at
k=64, 220.9 at k=128, 81.4 at k=256**.

**The cause.** `gemm44` walks eight live streams — four rows of each operand, `lda`/`ldb` apart — and
`ggml_call_mul_mat_ldc` pins `lda = ldb = k`. This op packs its kernel at `wdata` and its transposed
activation at `wdata + nk`. An ARM1176 has a 16 KB 4-way L1, so a way is 4 KB, and **`nk` for that
node is 524288 floats — exactly 2 MB, a whole number of ways.** The two panels are placed on top of
each other by construction: the a-streams and the b-streams land in the same sets, four ways deep, and
evict each other on every access.

`scripts/bench37.c` confirms it and, more usefully, contains the arm that *fails*: **blocking the `k`
loop so the live window is smaller changes nothing at all** — 81.7 MMAC/s at KB=128, 83.1 at KB=64,
against a baseline of 82.1. A capacity miss would have moved. A set conflict does not.

## Resolution & Lesson Learned

Sixteen floats of skew between the two panels, in `ggml-0009`, with the matching reservation in
`ggml_graph_plan`. One addend; padding `lda`/`ldb` instead measures the same (up to 284.8 MMAC/s) and
costs a stride parameter threaded through `ggml_call_mul_mat_ldc`.

| | before | after | |
|---|---|---|---|
| node 1 | 1638.6 ms | **596.2 ms** | 2.75x |
| node 2 | 1521.2 ms | 1199.9 ms | 1.27x |
| node 3 | 1653.4 ms | 1439.7 ms | 1.15x |
| the op | 4813.1 ms | **3235.7 ms** | **1.49x** |
| VITS, ABBA | 75.82 / 76.04 s | 74.69 / 75.20 s | 1.013x |

**Bit-identical output** — the change is an address — verified on x86 by hash across a whole synthesis
and on the board across all four ABBA arms. No measurable change on x86 itself (min-of-7 on the op:
44.0 ms against 45.1, inside that box's spread); the win is where the L1 is small.

* **Actionable takeaway 1 — a roofline measured at one shape is not a roofline.** Nodes 2 and 3 were
  read as "92% and 97% of the 226.7 MMAC/s ceiling, effectively done" and then got **27% and 15%
  faster**. That 226.7 came from a different shape (m=704 n=75 k=176) and was never a property of the
  machine. Before calling an op finished against a ceiling, check the ceiling was measured at the
  shape in front of you.
* **Actionable takeaway 2 — when a fix works, run the arm that should NOT work.** k-blocking was the
  cheapest way to tell a set conflict from a capacity miss, and it cost one extra function in a bench
  that already existed. A fix that works tells you far less than a fix that works next to a
  plausible one that does not.
* **Actionable takeaway 3 — two buffers carved out of one allocation are aligned to each other by
  construction.** `wdata` and `wdata + nk` differ by whatever `nk` happens to be, and tensor sizes are
  overwhelmingly powers of two times small factors, so "a whole number of cache ways apart" is the
  common case rather than bad luck. Any op that packs two operands into one scratch buffer and then
  streams both is exposed to this.

## See Also

* [Epic-08 §6.8](../epics/epic-08-packaging-and-release.md) — the ARMv6 convolution path
* [Retro-012](retro-012-optimizations-that-were-measured-out.md) — the register of ideas that were
  measured out; the `n` sweep and the k-blocking arm belong to it
* [Retro-037](retro-037-ps-said-six-percent-top-said-fifty.md) — the board contamination that had to be
  cleared before any of these numbers meant anything
