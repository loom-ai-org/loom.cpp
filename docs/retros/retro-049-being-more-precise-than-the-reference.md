---
type: retro
date: 2026-09-16
domain: exporter
tags: [numerics, family-5, cif, verification]
---

# Retro-049: Being More Precise Than the Reference Is a Way of Being Wrong About It

## The Issue

Family 5's second leaf (Paraformer) decides how many tokens to emit with a **continuous
integrate-and-fire** predictor: it accumulates a per-frame `alpha` and fires a token every time the
running sum crosses an integer. The number of tokens depends on the VALUES, not on any shape — the
first such thing in this zoo — so the crossing has to be decided host-side and handed to the graph.

The obvious way to write that host loop is to accumulate in double and compare floors. Done that way,
the acoustic embeddings came out **2.6e-01** from FunASR's, the decoder logits **3.8e-01**, and the
transcript was still *almost* right.

## Root Cause Analysis

Two separate float32 facts, and neither is visible in the algebra.

**1. The reference deliberately throws precision away, and the boundaries are knife-edge.** FunASR
accumulates at float64 and then **casts to float32** before flooring. A pure-float64 host is strictly
more accurate and puts tokens on different frames: the measured crossings sit within **1.9e-06** of an
integer, and one frame in this utterance moved. *Reproducing a reference means reproducing its
rounding, not improving on it.*

**2. The ORDER of the remainder expression is part of the contract.** FunASR computes

    remainder = (indicator + prefix_sum) - floor(prefix_sum)

in float32, left to right. Adding the indicator to the FULL running total first is what loses the
precision: at `prefix_sum ≈ 32` the float32 spacing is 3.8e-06, so `1 + 31.999998` lands exactly on a
representable midpoint, rounds to `33.0`, and the remainder collapses from ~1 to **0**. The
algebraically identical `1 + (prefix_sum - floor(prefix_sum))` gives ~1 — and a visibly wrong embedding
for that token. Because the value lands on a midpoint, **round-half-to-even is load-bearing** rather
than theoretical.

Finding (2) took four wrong guesses. What ended them was printing every intermediate at the one frame
that disagreed instead of reasoning about which operation "should" round.

## The Fix

The driver accumulates in Lua doubles (which is the float64 half for free) and rounds explicitly with a
`to_f32` helper built from `frexp` + the existing `round_half_to_even` + `ldexp` — exact against a real
float32 cast on 20,304 values. Both the recipe and its ORDER are transcribed into `cif_fire.lua` with
the reasoning beside them.

Two properties fell out of getting this right, and both are now design rules rather than accidents:

* **A threshold crossing must be computed in exactly one place.** The first working design had the host
  compute the fire indices and the GRAPH re-derive the remainders from its own float32 cumsum. Those
  two agree everywhere except on the frames that sit within an ulp of an integer — which is precisely
  the set that matters. The graph is handed the answer instead.
* **CIF is a linear resampling, so the answer is a matrix.** Each token is a weighted sum of encoder
  frames whose weights depend on `alphas` alone, so the host hands over the whole
  `(n_tokens, n_frames)` matrix and the graph does one matmul — no cumulative sum, no gather, no
  threshold. That was forced as much as chosen (`ggml_cumsum` only sums over `ne[0]`, and
  `ggml_get_rows` needs a contiguous source), and it is also **more accurate than the reference**:
  differencing two cumulative sums is catastrophic cancellation by construction. Against an f64
  evaluation of the same algebra, FunASR's own f32 result is 7.18e-07 away and loom's is **4.43e-08**.

## Takeaways

* **"Run the reference at f64" has a converse.** [The standing
  rule](retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md)'s neighbour says to
  compute the truth at f64 before blaming float noise, and that is still right for grading an OUTPUT.
  But where a reference's own rounding decides a BRANCH, the f64 answer is not the target — the
  reference's f32 answer is. Ask which of the two a number is: a value to be compared, or a decision to
  be reproduced.
* **An arithmetic identity is not an implementation identity.** `(1 + a) - b` and `1 + (a - b)` are the
  same number in ℝ and different numbers in float32 whenever `a` is large. When transcribing a
  reference that decides something, transcribe the expression, not its meaning.
* **Print the intermediates at the one element that disagrees.** Four hypotheses about which operation
  rounded were wrong; the fifth came from dumping `prefix_sum`, its floor, and `fires` at the single
  frame whose remainder differed. Same lesson as
  [Retro-048](retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md), one layer down.
* **A wrong boundary does not look like a bug.** 2.6e-01 on the embeddings still transcribed 13 s of
  Chinese nearly correctly. The only thing that caught it was comparing a tensor to the reference's,
  which is why this family's test pins the fire positions and the weights rather than the transcript.
