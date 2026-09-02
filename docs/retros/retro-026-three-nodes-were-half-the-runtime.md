---
type: retro
date: 2026-09-02
domain: backends
tags: [metal, gpu, profiling, convolution, threadgroup, occupancy, hypothesis, p4.30a]
---

# Retro-026: Three Nodes Were Half The Runtime, And The Named Hypothesis Was Backwards

## The Issue

P4.11 measured Metal on an M1 Pro and found it **5.24x slower than the CPU on VITS** while being
faster on whisper. The write-up ([Epic-04 §5.4](../epics/epic-04-backends-and-accelerators.md))
committed to a mechanism: loom's CPU convolution carries seven hand-written ggml patches "while its
Metal path runs stock `im2col` + `mul_mat`".

**Both halves of that clause were false.** At F32, ggml-metal does not lower a convolution through
`im2col` at all — it has a native `CONV_2D` kernel and runs it. And where it *does* take the `im2col`
route (a Q4_0 export, whose folded kernel it declines on its type test), that route is the **faster**
of the two, by about 2x.

The real answer was somewhere nobody had looked. **`CONV_TRANSPOSE_1D` — three nodes out of 1308 —
was 45% of the run**, at 6.7 GFLOP/s on a part that does 5.31 TFLOP/s: **0.13% of peak**, and 10x
slower than a *single* CPU core running the same three nodes.

## Root Cause Analysis

`ggml_metal_op_conv_transpose_1d` ends with

```c
ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);
```

`OL * OC` threadgroups of **one thread each**. Every threadgroup occupies a single lane of a 32-wide
SIMD group, so 31 of every 32 lanes on the GPU are idle by construction. `kernel_conv_2d`, 400 lines
above it in the same file, already does the opposite.

**Why nothing before the profile could have found it.** The op is fully supported by ggml-metal, so
it produces no split, no fallback and no warning — every diagnostic the investigation had reached
for was silent on it. And the op has *three nodes*, so every ranking by node count, split count or
"where is this model's work" put it near the bottom. §5.4 had already learned that **the split count
says where the graph went, not where the time went** — from the `PAD` prototype that moved 27 of 56
splits and bought 1.8%. This is the same lesson one level in: the *node* count does not say it
either.

The fix is three edits and no new kernel — `nth` threads on the x axis, the grid divided by the same
factor, `OL` in the args for the tail bounds check. `cmake/patches/ggml-0014-...patch`, upstream as
PR 14. **29x on the op; 494.1 -> 278.7 ms on an F32 VITS synthesis, with a bit-identical waveform**,
and `test-backend-ops -o CONV_TRANSPOSE_1D` 116/116 on MTL0 either way.

## Resolution & Lesson Learned

* **Actionable takeaway 1 — a hypothesis that names a mechanism has to be checked against the
  mechanism, not against the outcome it predicts.** "Metal runs stock im2col+mul_mat and the CPU
  carries seven patches" predicted the observed 5.24x correctly and was wrong about every fact in
  it. Confirming it cost one `grep` in `ggml-metal-ops.cpp`, which nobody ran for two weeks because
  the prediction kept coming out right.
* **Actionable takeaway 2 — on a GPU, read the dispatch before theorising about the kernel.** The
  threadgroup geometry is one line, it is the first thing that can be catastrophically wrong, and
  no profile, split count or op-support table will mention it.
* **Actionable takeaway 3 — a device profile is quotable once, and only once, you calibrate it
  against the graph's own no-ops.** `$LOOM_PROFILE` on a scheduler costs a `ggml_backend_synchronize`
  per node — 925 ms against a 494 ms wall here. But `RESHAPE`, `VIEW` and `PERMUTE` compute nothing,
  so whatever the report charges them *is* the per-node overhead (0.1998, 0.1926, 0.2111 ms — three
  independent estimates that agree). Subtract it and the remaining buckets sum to the un-profiled
  wall within 1%. This turns "ordering information only" into a real measurement, and it works on
  any backend. It does **not** work on the CPU above one thread, where the overhead is a thread-pool
  barrier that lands unevenly and the same subtraction drives no-op rows negative.
* **Actionable takeaway 4 — the cheapest experiment was one already sitting on disk.** The question
  "would `im2col` + `mul_mat` beat this kernel" needed no new build: the Q4_0 export already takes
  that path, because Metal declines a folded quantized kernel. Before writing a harness, check
  whether some existing artefact already runs the arm you want.

---

## Full record

Measured on an Apple M1 Pro (8 P-cores, 16-core GPU), macOS 15.7.9, loom `7782a30`,
`scripts/bench_vits_loom.cpp` median of 9. Arithmetic from `scripts/conv_census.py --syms
n_tokens=100 --syms flow_vocoder:n_tokens=286`: 18.39 GFLOP of convolution, of which the three
transposed ones are 1.50.

| op | calls | Metal, stock | Metal, `+ggml-0014` | CPU @ 1 thread |
|---|---:|---:|---:|---:|
| `CONV_2D` | 117 | 241.0 ms | 241.3 ms | ~212 ms |
| `CONV_TRANSPOSE_1D` | 3 | **224.7 ms** | **7.8 ms** | ~21.6 ms |
| everything else | 2078 | ~28 ms | ~30 ms | ~62 ms |
| **wall (un-profiled)** | | **494.1 ms** | **278.7 ms** | **296.0 ms** |

The re-profile is the control that makes the attribution a measurement rather than an inference:
the patch touched one op and exactly one bucket moved.

What is left is `CONV_2D` at **70 GFLOP/s, 1.3% of peak** — 86% of the remaining run — from a kernel
that is one thread per output element with two global loads per FMA and no reuse. Metal's own
`MUL_MAT` runs the same convolutions at about **1.49 TFLOP/s** through the Q4_0 lowering, so the
headroom is real and the arithmetic was never the problem. Tracked as **P4.30d**, and **CLOSED the
same day** by `ggml-0015` — a register tile whose other half is that sixteen registers of 64-bit
address were competing with the accumulators; 5.10x on the op, 278.1 -> 97.6 ms on the model. Full
record in [Epic-04 §5.7 and §5.8](../epics/epic-04-backends-and-accelerators.md). **Note while
reading the "of peak" columns above that 5.31 TFLOP/s is a spec number this part does not deliver:
the measured F32 FMA roofline is 2.11 TFLOP/s, so every percentage here is understated by 2.5x.**

Two smaller things fell out of the same pass and are fixed here rather than tracked:
`scripts/conv_census.py` rejected `Max(a, b)` — the clamp `SymbolEnv` gained in P4.28 and sympy now
prints into every VITS-family export — so the census tool could read only pre-P4.28 GGUFs; and
`scripts/bench_asr_loom.cpp` hardcoded `"cpu"`, so the ASR half of a per-model device comparison
could not be run at all.
