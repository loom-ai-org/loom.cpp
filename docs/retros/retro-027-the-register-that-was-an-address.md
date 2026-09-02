---
type: retro
date: 2026-09-02
domain: backends
tags: [metal, gpu, convolution, occupancy, registers, roofline, benchmarking, p4.30d]
---

# Retro-027: The Register That Was An Address, And A Denominator That Was 2.5x Wrong

## The Issue

[Retro-026](retro-026-three-nodes-were-half-the-runtime.md) closed P4.30a and left one op behind:
`CONV_2D` at **86% of a VITS synthesis on Metal**, 241 ms for 16.9 GFLOP. That remainder became
P4.30d, and [Epic-04 §5.7](../epics/epic-04-backends-and-accelerators.md) scoped it with two named
candidate shapes — "a tiled implicit-GEMM kernel with threadgroup staging, or lower to `im2col` +
`mul_mat` and accept the expansion traffic" — and a ceiling, "1.3% of the part's peak".

**Both candidates lose. And the peak was the wrong number to divide by.**

The kernel that shipped is a third shape, and *half of its win has nothing to do with tiling*: it
comes from writing the addresses as 32-bit element indices instead of 64-bit pointers. Same
arithmetic, same tile, same loads — **68.8 ms to 45.6 ms**.

## Root Cause Analysis

**The registers were the budget, and addresses were spending it.** Giving each thread eight output
channels needs eight weight streams. Written the obvious way that is eight `device const TK *` —
sixteen registers of pure address, competing with the eight accumulators they exist to feed.
`maxTotalThreadsPerThreadgroup`, the only register-pressure signal Metal exposes without a GPU
capture, reads it off directly: **704 threads for the pointer form, 896 for the index form**, out of
1024 for a kernel under no pressure. On this GPU occupancy *is* latency hiding, and the kernel was
latency-bound the whole time — it issued 17 instructions per 8 FMAs and ran at 40% of what that mix
allows.

**The probe explains that gap and misses the next one, which is worth knowing before trusting it.**
Push past 32 accumulators per thread — 8 positions x 8 channels — and the kernel runs **302.7 ms,
0.73x, slower than the stock kernel it was meant to beat** — while `maxTotalThreadsPerThreadgroup`
still reads 896. The compiler spilled to scratch instead of raising the register count, and the probe
reports register *pressure*, not spilling.

And the ratio everyone reasons about was never the binding constraint. A 4-position x 4-channel tile
has **0.5 loads per FMA against the winner's 1.125** — less than half — and is **60% slower**,
because it gives back more occupancy than the ratio buys.

**And the denominator was a spec sheet.** "1.3% of the part's 5.31 TFLOP/s" is arithmetic on a number
no kernel here can reach. A pure FMA loop with sixteen independent chains and no memory traffic at
all measures **2.11 TFLOP/s** on this M1 Pro. The real headroom was 28x, not 77x — which is a
different scoping decision, and would have been a different one again if the measured roofline had
come back at 400 GFLOP/s.

## Resolution & Lesson Learned

`cmake/patches/ggml-0015-metal-conv-2d-register-tile.patch`, upstream as PR 15. **5.10x on the op set
(222.3 -> 43.6 ms), 241.3 -> 60.1 ms in the model, and an f32 VITS synthesis 278.1 -> 97.6 ms with a
bit-identical waveform.** `test-backend-ops -o CONV_2D` is 2026/2026 on MTL0, `ctest -L ci` 75/75.

* **Actionable takeaway 1 — on a GPU, count the registers an operand's ADDRESS costs, not just the
  operand.** A 64-bit pointer per accumulator stream is two registers that do no arithmetic, and at
  eight streams that is half the register budget of the thing they feed. 32-bit element indices off
  one base pointer are free by comparison, and the tensors have to be big enough to need 64 bits
  before it is even a correctness question — so decline those at a threadgroup-uniform test and keep
  the 64-bit loop as the general path.
* **Actionable takeaway 2 — `maxTotalThreadsPerThreadgroup` is a free occupancy probe, print it
  beside every variant in a Metal sweep, and know its blind spot.** It cost three lines and turned
  "why is the better ratio slower" from speculation into a reading. But it reports register
  *pressure*, not spilling: the worst variant in the sweep (302.7 ms, slower than stock) reads a
  healthy 896, because the compiler spilled to scratch rather than raise the register count. The
  probe tells you which variants are competing for registers; only the clock tells you which one
  gave up.
* **Actionable takeaway 3 — a GPU's spec FLOP/s is not a denominator.** Measure the achievable FMA
  rate with a kernel that has no memory traffic and enough independent chains to be an ISSUE rate
  rather than a latency rate, and quote fractions of *that*. This is
  [Retro-011](retro-011-chasing-the-gemm-and-convolution-gap.md)'s rule, already written down in this
  repo, and it did not stop §5.7 from dividing by 5.31 TFLOP/s six times — because on a GPU the spec
  number is published, and a published number does not feel like an assumption.
* **Actionable takeaway 4 — when a scoping note names candidate solutions, they are hypotheses and
  they get measured, not implemented.** Threadgroup staging was named first and is **worse on every
  activation length** (55.2 ms against 43.6): the uniform weight loads were already being served by
  one cache line per SIMD group, so there was nothing to save and the barriers were not free. Both
  named candidates were plausible, both were derived from what works on the CPU, and neither
  survived one afternoon of measurement.
* **Actionable takeaway 5 — build the harness that runs the model's real shape mix, not a
  microbenchmark of one shape.** `scripts/bench22.mm` runs the 31 distinct convolutions of a VITS
  synthesis at the multiplicity the graph issues them, from `conv_census.py`. It reproduces the
  stock kernel at 222.3 ms against the model's 241, so a ratio it reports is a prediction; it made a
  variant cost seconds instead of a ggml rebuild plus a profiled run; and it caught that the winner
  on the two longest activations is *not* the winner overall. A single-shape microbenchmark would
  have chosen the 4-position tile.

---

## Full record

Measured on an Apple M1 Pro (8 P-cores, 16-core GPU), macOS 15.7.9, loom `7167822`. Achievable
rooflines, measured rather than quoted: **2.11 TFLOP/s** F32 FMA and **180 GB/s** streaming read,
against 5.31 TFLOP/s and 200 GB/s on the spec sheet.

The sweep, over the 117 convolutions (16.884 GFLOP) of the utterance `bench_vits_loom.cpp` pins:

| | total | rate | occupancy |
|---|---:|---:|---:|
| stock | 222.3 ms | 76 GFLOP/s | 1024 |
| register tile (8 channels/thread), 64-bit pointers | 68.8 ms | 245 GFLOP/s | 704 |
| + 32-bit element indices | 45.6 ms | 370 GFLOP/s | 896 |
| + `KW` as a function constant | 44.8 ms | 377 GFLOP/s | 896 |
| + 128 threads per threadgroup | **43.6 ms** | **387 GFLOP/s** | 896 |
| *threadgroup-staged weights* | *55.2 ms* | *306 GFLOP/s* | *896* |
| *4 positions x 4 channels — 0.5 loads/FMA, less than half the winner's* | *69.7 ms* | *242 GFLOP/s* | *832* |
| *2 positions x 8 channels — wins on the two longest activations only* | *65.1 ms* | *259 GFLOP/s* | *832* |
| *8 positions x 8 channels — spills to scratch, probe says 896* | *302.7 ms* | *56 GFLOP/s* | *896* |

End to end, with the re-profile as the control — one op patched, one bucket moved:

| op | calls | Metal `+ggml-0014` | Metal `+ggml-0015` |
|---|---:|---:|---:|
| `CONV_2D` | 117 | 241.3 ms | **60.1 ms** |
| `CONV_TRANSPOSE_1D` | 3 | 7.8 ms | 7.7 ms |
| everything else | 2078 | ~30 ms | ~22 ms |
| **wall (un-profiled)** | | **278.1 ms** | **97.6 ms** |

It is not a VITS-only fix: whisper-small goes 0.874 -> 0.752 s, from the two convolutions its encoder
opens with.

**One thing this created rather than closed.** Metal declines loom's folded block-quantized
convolution kernel on its type test, so a Q4_0 export lowers through `im2col` + `mul_mat` and never
reaches the new kernel. That path is untouched, so **Q4_0 on Metal is now 1.53x SLOWER than f32**
(149.1 ms against 97.6) where before it was 1.86x faster. Quantizing a convolutional model for this
backend now costs time as well as accuracy. Filed on the hub; the fix is in the type test, not in the
quantization.

**What is still on the table, deliberately.** 387 GFLOP/s is 18% of the measured roofline, against
the stock kernel's 3.6%. A 2-position tile is worth ~6% on the two longest activations and was
declined because it costs a second pipeline family and a threshold on `OW`. Beyond that the ceiling
is a simdgroup-matrix implicit GEMM — this backend's own `mul_mat` reaches ~1.49 TFLOP/s — which is
another 2-3x and a much larger kernel than anything here.
