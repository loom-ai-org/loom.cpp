---
type: epic
status: active
domain: performance
last_updated: 2026-08-31
---

# Epic-05: Edge CPU Performance

## 1. Context and Scope

The engine's stated target is edge devices, so the reference question is: **how does loom compare to
onnxruntime running the same checkpoint on a Raspberry Pi 4?**

The answer went from **2.2x slower** to **1.03x** over this thread, and every step of it was a
measurement. In scope: profiling infrastructure, kernel-level performance in the pinned `ggml`,
graph-level work reduction, and quantization for artifact size.

**That 1.03x reads 0.96x today**, on a re-measurement with matched estimators. It briefly read
**0.84x**: `ggml-0011` was found and measured on x86, shipped to every architecture, and regressed this
board by 1.3-1.75x on every f32 GEMM for four days (P4.20, below — found and fixed 2026-08-29). It is
the only thing in this epic that has moved the reference number backwards.

## 2. Architectural Overview

### The profiler comes first

`$LOOM_PROFILE` gives per-node timing of the graph the engine **actually runs**. It exists because
three plausible explanations, each argued convincingly from the source, were each measured at a few
percent or less. Reasoning from code lost three times in a row before anything was built.

**What it cannot see:** fusion (fusion changes a convolution's *neighbours*, not the convolution), and
anything inside a single op. When the op is the suspect, phase-time inside it.

### Where the time actually went

| op | share of a 1-thread VITS synthesis |
|---|---|
| `MUL_MAT` | **70.4%** |

The gap against onnxruntime was **F32 micro-kernel quality**: `ggml`'s generic `mul_mat` at 17% of the
machine's fp32 peak against MLAS's 45%, on identical arithmetic.

### What shipped

Nothing in this repository computes a GEMM — the fixes are patches to `ggml`
([ADR-014](../adrs/adr-014-patch-ggml-rather-than-write-kernels.md)):

| change | worth |
|---|---|
| `GGML_LLAMAFILE=ON` + two tinyBLAS register-blocking patches | the bulk of the 71% convolution gap |
| `ggml-0005` bias fusion into `CONV_2D` | folds a 12% `ADD` row into the convolution |
| `ggml-0006` direct 1-D convolution behind a cache-size heuristic | wins where the activation is long and the weights small |
| tap-major (phase-major) traversal, a third path | dilation `d` is `d` dense convolutions; one contiguous run per channel instead of 896 prefetch streams |
| `ggml-0007` resblock `LEAKY_RELU` + residual `ADD` fusion | 1.441 → 1.345 s |
| `ggml-0008` / `ggml-0009` `conv_transpose_1d` prologue and GEMM | 1.314 → 1.202 s; the op itself 195.8 → 79.1 ms |
| the text encoder exported once, not twice | 1.196 → 1.099 s ([Retro-014](../retros/retro-014-the-text-encoder-was-in-the-graph-twice.md)) |

**loom lowers `CONV_1D` to `CONV_2D` on every architecture** — the `#if defined(__aarch64__)` guard is
gone, because the direct kernel is what x86 was missing.

### The same question on a 24-core x86 box

The Pi is the stated target, but every heuristic in the pinned `ggml` was tuned on it and on a 2-core
AVX2 laptop, so the many-core x86 class was the one that could plausibly have been regressed. Measured
2026-08-23 on an **Intel Core Ultra 9 285K (24 cores, 36 MB L3)**, same VITS checkpoint, same utterance,
scales pinned to `[0, 1, 0]` so both engines synthesise the same 73216 samples:

| | median | vs loom |
|---|---|---|
| **loom**, engine default threads | **0.0708 s** | — |
| onnxruntime, `intra_op_num_threads = 4` | 0.0650 s | **1.09x** |

So the 1.03x on the Pi is **1.09x here** — the thread's result holds on a machine nothing in it was
tuned for. The harnesses are `scripts/bench_vits_loom.cpp` and its onnxruntime counterpart; both print
their sample count, and a ratio taken without checking those match is measuring two different
utterances ([Retro-010](../retros/retro-010-an-unpinned-competitor-baseline.md)).

**Both convolution heuristics together are a 1.75x win here, not the feared regression** — 0.1242 s
with `GGML_CPU_DISABLE_CONV_HEURISTICS=1`, 0.0708 s without it. At the `bench9` level the same switch
gives 1.80x against 1.00x: **unpatched, ggml's `CONV_2D` is exactly level with im2col + `mul_mat` on
this machine, and the whole margin is patches 0004 + 0006.** Nothing here is dodging a slow path.

The **0.87x** that appears in `bench9.cpp`'s header is easy to misread as one. It was patch 0004
*alone* on a 2-core AVX2 Ryzen, and it is why this lowering used to be aarch64-only. It was superseded
by 0006 — a direct kernel that materialises no patch matrix — not by measuring a roomier x86 box, which
is why `LOOM_CONV1D_DIRECT` now defaults on for every architecture. `scripts/bench9.cpp`'s header still
says "the lowering is chosen by architecture"; `src/ops/primitives_conv.cpp` is the current word. That switch is new, and is the run-time
escape patches 0004 and 0006 previously lacked; `cmake/patches/UPSTREAM.md` carries the per-patch
numbers and the reason `bench10`'s kernel-only 0.84x does not contradict this.

**`n_threads` was never set anywhere in the engine**, so ggml's default of 4 ran on every machine.
`$LOOM_N_THREADS` now overrides it (unset changes nothing). Measuring what that buys produced the more
interesting result:

### loom stops scaling at 8 threads, and then goes backwards

VITS, Core Ultra 9 285K, median of 7:

| threads | 1 | 2 | 4 | **8** | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| synthesis | 0.198 s | 0.113 s | 0.093 s | **0.080 s** | 0.116 s | 0.157 s | 0.191 s |

**24 threads is as slow as one.** onnxruntime on the same box keeps improving over the same range
(0.087 s at 4 threads, 0.075 s at 24), so this is loom's curve and not the workload's. The same shape
shows in the LM: 21.8 tok/s at 4 threads, 10.6 at 24.

**Resolved below: it was libgomp's default wait policy, and the fix is `GGML_OPENMP=OFF`.** The table
above is the *broken* curve, kept because it is what the investigation started from. ggml's default of
4 happened to sit near the good part of it, which is why nothing noticed for so long.

#### P4.17: it was libgomp's default wait policy — RESOLVED 2026-08-23

**loom builds ggml with `GGML_OPENMP=ON`.** That is ggml's own default
(`ggml/CMakeLists.txt:245`); loom's CMake never mentioned OpenMP, so it was inherited, never chosen.

In that configuration `ggml_barrier` is `#pragma omp barrier` (`ggml-cpu.c:575`, `582`) and ggml runs one
after **every non-empty graph node** (`ggml-cpu.c:3370`) — **2520** of them in a VITS synthesis (4240
nodes, less the 1720 `RESHAPE`/`VIEW`/`PERMUTE` that `ggml_op_is_empty` skips). **libgomp's default
wait policy makes every thread sleep on a futex at every one of them.**

That is not an inference. 5 syntheses at 24 threads, `getrusage(RUSAGE_CHILDREN)`:

| build | voluntary ctx switches | user CPU | median |
|---|---|---|---|
| `GGML_OPENMP=ON`, as shipped | **334,609** | 2.90 s | 0.2050 s |
| `GGML_OPENMP=ON` + `OMP_WAIT_POLICY=active` | **101** | 13.82 s | 0.0314 s |
| `GGML_OPENMP=OFF` (ggml's threadpool, `poll = 50`) | **160** | 10.16 s | 0.0500 s |

334609 / 5 / 24 = **2789 sleeps per thread per synthesis** against 2520 barrier-carrying nodes — one
per node, plus the few ops that barrier internally.

**The curve with the fix.** VITS, same box, median of 7:

| threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| `GGML_OPENMP=ON`, as shipped | 0.186 | 0.114 | 0.071 | 0.079 | 0.087 | 0.118 | 0.189 |
| + `OMP_WAIT_POLICY=active` | 0.195 | 0.159 | 0.064 | 0.040 | 0.040 | 0.036 | **0.031** |
| `GGML_OPENMP=OFF` | 0.186 | 0.111 | 0.064 | 0.041 | 0.042 | 0.037 | **0.040** |

**4.8x at 24 threads from a build flag, and the curve is monotonic again.** The causal-LM decode loop
agrees — `infer_with_past` tok/s, best of 5:

| threads | 1 | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| as shipped | 9.66 | 21.9 | 24.7 | 14.0 | 11.0 |
| + `active` | 11.0 | 23.4 | 29.8 | 28.4 | 26.6 |
| `GGML_OPENMP=OFF` | 10.0 | 23.5 | 29.6 | 24.4 | 19.1 |

Nothing regresses at any thread count on either workload. A single LM run at 4 threads showed 17.7
tok/s for both fixed arms; repeated five times it is 23.4–23.5 against 21.9. **Benchmark the LM more
than once — its first run is cold by several tok/s.**

#### Which of the three candidates it was, and what that costs the other two

It was **(3), alone** — the one ranked last and called cheapest to rule out.

* **(3) wake-up latency — CONFIRMED.** `GOMP_SPINCOUNT=100` reproduces the broken curve exactly
  (0.068 / 0.070 / 0.128 / 0.138 at 4 / 8 / 16 / 24); `GOMP_SPINCOUNT=300000`, which is libgomp's own
  documented default, reproduces the fixed one. Every value at or above 300000 is identical.
* **(1) barrier cost — real, and a fifth of it.** Microbenchmarked directly: **12591 ns per barrier at
  24 threads against 1187 ns** with `active` (fork-join 36139 against 2853). 2520 barriers × 11404 ns
  = **28.7 ms of a ~124 ms delta**. The rest is threads sleeping *inside* ops, which is why
  `CONV_TRANSPOSE_1D` loses 89 ms over **6** calls — 6 barriers cannot buy that.
* **(2) `n_tasks` not clamped by work size — NOT the mechanism.** Nothing was clamped and the curve
  was fixed anyway. The earlier claim that "(1) and (2) are the same fix from different ends and are
  the likely answer together" was wrong: **no ggml patch to `ggml_get_n_tasks` is needed.**
* **Heterogeneous cores — NOT the mechanism.** The box is 8 P-cores + 16 E-cores and the peak sat at
  exactly 8, which made this look compelling. `OMP_PROC_BIND=close/spread` changes nothing once
  spinning is on; `bind=close` *without* spinning is far worse (0.456 s at 24).

**The candidate-(3) text above described the wrong threadpool.** ggml's hybrid poll/sleep loop
(`ggml-cpu.c:3421`, `threadpool->poll`) is inside `#ifndef GGML_USE_OPENMP` — **compiled out of every
build loom ships**. The spin policy that mattered was libgomp's, and `$GGML_CPU_WAIT_POLICY` does not
exist. When a threading question comes up, **check which threadpool is actually compiled in first**;
this one cost a full round of reading dead code.

#### ggml knows about this, and only fixes it for Intel's OpenMP

`ggml-cpu.c:4114` — in the pinned v0.19.0, inside `#ifdef GGML_USE_OPENMP`:

```c
//if (!getenv("OMP_WAIT_POLICY")) {
//    // set the wait policy to active, so that OpenMP threads don't sleep
//    setenv("OMP_WAIT_POLICY", "active", 0)
//}

if (!getenv("KMP_BLOCKTIME")) {
    // set the time to wait before sleeping a thread
    setenv("KMP_BLOCKTIME", "200", 0); // 200ms
}
```

**`KMP_BLOCKTIME` is Intel/LLVM `libomp`'s knob and GNU `libgomp` ignores it entirely**, so every
gcc-built ggml — which is what Debian, the wheels, and the Pi all produce — gets no mitigation at all.
The `OMP_WAIT_POLICY` line that *would* have covered libgomp is commented out, and it could not have
worked from there anyway: libgomp reads that variable in its **load-time constructor**, long before
`ggml_init` runs.

This is independent confirmation of the diagnosis, and it is the reason the fix cannot be an
environment variable set from inside the library.

#### Why the fix is the build flag and not the environment variable

`OMP_WAIT_POLICY=active` is the fastest arm and is still the wrong answer:

* **A library cannot set it.** libgomp reads it in its load-time constructor, before any loom code
  runs, so `setenv` from inside the engine is too late.
* **It spins forever**, including between inferences — 13.82 s of CPU against 2.90 s for a run that
  takes 0.16 s of wall time.

ggml's own threadpool spins a **bounded** ~6.5M rounds (`poll = 50`, `ggml.c:8011`) and then sleeps,
so it spins through a graph's barriers and still sleeps between calls. It also takes libgomp out of
the wheels, which matters past speed: loom-py is imported alongside numpy and torch, which bring their
own OpenMP runtimes, and two in one process is a known hazard.

**If the residual 24-thread gap (0.040 against 0.031 s) is ever worth closing**, the follow-up is a
tenth ggml patch adding `ggml_backend_cpu_set_threadpool` to the CPU backend's registry proc-address
table (`ggml-cpu.cpp:648`). It is absent today, so the `GGML_BACKEND_DL` build the wheels ship
(ADR-009) cannot reach it and `poll` is not tunable in-process — the same constraint that shaped
`apply_cpu_threads`.

#### Scope: this is a many-core fix

**A Pi 4 at its 4 cores is unchanged**, measured both ways with the arms interleaved and cooling gaps
between them so each pays the same thermals (the Pi throttles ~33% under sustained load):
`OMP_WAIT_POLICY=active` is 1.00x (1.108 s against 1.109 s), and `GGML_OPENMP=OFF` against `ON`, same
source tree, three rounds, is **1.011 / 1.015 / 0.997**. At 4 threads the whole barrier bill is a
couple of ms of a 1.1 s synthesis.

That "unchanged" is the result the change needed: ggml's threadpool spins its workers where libgomp
slept them, and the worry was that on a 4-core box three spinning workers would starve the main thread
between graph computes. They do not. The fix is worth only ~1.1x at ggml's default of 4, so raising
that default is a separate question — **and it now has its own benchmark, below.**

### What the default thread count should be, now that the curve is monotonic

The default was left at ggml's 4 because raising it "would be a regression on exactly the machines it
looks like it should help". **That objection was about the libgomp collapse and no longer holds.**
Re-measured on the fixed engine, best of 3 per point, `$LOOM_N_THREADS` the only variable.

**Core Ultra 9 285K — 24 physical cores, no SMT:**

| threads | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| TTS (s) | 0.2887 | 0.1541 | 0.0638 | 0.0481 | 0.0411 | 0.0411 | 0.0358 | 0.0346 | **0.0323** |
| LM (tok/s) | 11.29 | 15.38 | 23.24 | 27.37 | **29.59** | 27.93 | 27.21 | 26.93 | 27.45 |
| ASR (s) | 6.654 | 3.973 | 2.446 | 1.631 | 1.430 | 1.249 | 1.153 | 1.070 | **1.014** |

**Ryzen 3 3250U — 2 physical cores, SMT, so 4 logical:**

| threads | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| TTS (s) | 0.6400 | **0.4339** | 0.5725 | 0.5576 |
| LM (tok/s) | 6.04 | **6.71** | 6.67 | 6.26 |
| ASR (s) | 17.52 | **11.29** | 11.54 | 12.30 |

The 2-against-4 comparison was then re-measured on its own, median of 5 interleaved rounds rather than
one sweep point, because this box is a two-core laptop that also runs the session driving it and its
single-point numbers move by up to 10%: **TTS 0.4230 against 0.5038 (1.19x), LM 6.81 against 6.59
(1.03x), ASR 11.21 against 11.17 (1.00x — unchanged).** The sweep row above overstated the ASR and LM
gains; the shape of the answer is the same.

**The answer is the PHYSICAL CORE COUNT, and the two machines only agree once it is put that way.**
The 285K wants 24 — every logical CPU, because it has no SMT. The Ryzen wants **2, not its 4 logical
CPUs**, and its two extra SMT threads buy nothing on any task and cost on two. "Use every CPU" would be right on one machine and a
1.28x TTS regression on the other; "use every physical core" is right on both.

Against today's default of 4: the 285K gains **1.98x on TTS, 2.41x on ASR, 1.18x on the LM**, and the
Ryzen gains **1.19x on TTS and 1.03x on the LM** by going *down* to 2, with ASR unchanged. A Pi 4
(4 physical cores) does not move.

**On the Ryzen this does not move the published ratios, because onnxruntime prefers 2 threads too** —
it gains 1.17x / 1.02x / 1.07x over the same change, so loom's ASR column actually gets *worse*
(0.72x -> 0.67x) for want of a gain onnxruntime does get. **The physical-core rule is a property of
these CPUs rather than of loom**, which is worth saying out loud before it is quoted as an engine win.

Two things this measurement is not:

* **Nothing goes backwards any more, but the LM still has a preference.** It peaks at 8 on the 285K and
  sits on a plateau after — 27.45 tok/s at 24 against 29.59 at 8, **7% off its own optimum rather than
  the 2.7x collapse it used to be**. A single default cannot serve all three tasks perfectly and does
  not need to.
* **Every number is one inference at a time on an idle machine.** A host running several loom instances
  concurrently, or sharing the box, would have each claim every core; ggml's conservative 4 is a
  reasonable default for that world and a bad one for this. Raising the default is a **policy** choice
  about which world loom is for, not a further measurement.

**Low thread counts on a hybrid part are noisy, and the noise is placement.** On the 285K's 8 P-cores +
16 E-cores a 1-, 2- or 4-thread run can be scheduled entirely onto E-cores: one sweep read 0.0926 s at
4 threads where a repeat read 0.0638 s. Hence best-of-3 above, and **a single measurement below ~6
threads on a hybrid CPU should not be trusted.**
### Three tasks, not one — and the convolutional one is the good one

VITS is the model this whole thread optimised, so it is the least representative thing to quote alone.
**Re-measured 2026-08-24 with both engines run back to back on each machine**, F32 on both sides:

| machine | threads | TTS | LM | ASR |
|---|---|---|---|---|
| Core Ultra 9 285K | 4 | **1.03x** | **1.01x** | 0.71x |
| Core Ultra 9 285K | 24 (all) | **1.17x** | 0.96x | **1.29x** |
| Ryzen 3 3250U | 2 (physical) | **1.02x** | **1.05x** | 0.67x |
| Ryzen 3 3250U | 4 (all) | 0.99x | **1.04x** | 0.72x |
| Raspberry Pi 4B | 4 (all) | 0.98x | **1.08x** | 0.57x |

`>1.00x` is loom faster. The full table with its methodology and caveats is in the
[README](../../README.md#performance-against-onnxruntime); the ones that change how it should be read
are that the baseline is the **PyPI** onnxruntime wheel (conda-forge's build of the *same version* is
1.86x faster on VITS, and against it the x86 TTS wins become losses), and that the Pi row is taken
cooled and interleaved because that board drifts 33% when it is not.

**The estimator is the median** of repeated runs on both sides, except the Pi row, which reports the
FASTEST run — on that board thermal drift only ever makes a run slower, so the coolest run is the least
contaminated one. That choice is not cosmetic: loom's LM at 24 threads is bimodal (24.5-28.6 tok/s
across nine runs, against onnxruntime's steady 27.1-27.6), so its best run wins the cell at 1.03x and
its typical run loses it at 0.96x. The table reports the typical one.

**The previous version of this table was wrong in both directions and is worth keeping as a warning.**
It read 1.25x / 1.16x / 0.41x for the 285K at 4 threads. Two independent errors made it so: its
onnxruntime figures could not be reproduced by any harness in this repository, and its loom LM figure
came from a harness that timed ONE COLD generation where the onnxruntime side warmed up — two
different estimators, worth 5-7%. **Ratios carried forward from a previous table cannot be trusted;
re-measure both sides or quote neither.**

**That LM row was the engine calling the wrong function, and it is fixed.** `loom::text::generate`
called `infer` unconditionally. Every generated causal-LM driver exports **two** entry points — `infer`,
one forward over whatever it is handed, returning one token; and `infer_with_past`, which runs the
decode loop itself against the KV cache and returns the whole sequence. Taking `infer`, the host re-fed
a growing prompt and every step recomputed the entire sequence.

`$LOOM_PROFILE` is what showed it, and only because it prints shapes: **112 `MUL_MAT` calls per step at
`ne1` = the current sequence length rather than 1** — 28 layers x 4 projections, over the whole prompt,
every token. A plausible-sounding Lua-marshalling explanation was argued first and was wrong; `MUL_MAT`
was 83-88% of the profile the whole time. That is the fourth time on this thread that reasoning from
code lost to reading a profile.

| Qwen3-0.6B, 65 tokens | | |
|---|---|---|
| `infer`, re-fed prompt (was shipping) | 8.43 s | 7.71 tok/s |
| `infer_with_past` (now selected) | **2.98 s** | **21.78 tok/s** |
| onnxruntime | 3.33 s | 19.21 tok/s |

**So the LM reached parity, and it was never a kernel problem** (ASR did not — see the re-measured table above, where it is still 1.4-1.8x behind at four threads). The tell was the
*shape* of the curve, not its height: onnxruntime's cost per token is flat (46.7 ms at 64 tokens, 45.2
at 128) and loom's was not (146 -> 305 ms), which is what O(n^2) looks like from outside.

Both ASR engines produce the identical transcript, so that 1.18x is comparable work. The LM figures are
**differenced** — the same prompt at 1 and at 65 new tokens, so model load, tokenisation and prefill
cancel and what remains is 64 decode steps; loom's raw wall times include a 2.4 GB GGUF load that
onnxruntime's do not, and quoting them undifferenced would flatter onnxruntime by about a second.

Harnesses: `scripts/bench_vits_loom.cpp` and, for the other two, `loom_cli` timed against a raw
`onnxruntime` decode loop. **Do not use `optimum` for Qwen3** — 2.3.0 infers `head_dim` as
`hidden / n_heads` (64) where Qwen3 decouples it to 128, and the session rejects the KV cache it
builds. Reading the shapes off `session.get_inputs()` avoids assuming anything.

### P4.18: the ASR gap is the ENCODER'S ATTENTION and a loop-invariant copy in the DECODER

*(GELU done 2026-08-24; the `k % KN` GEMM cliff done 2026-08-24 as `ggml-0011`; the encoder/decoder
split below was WRONG in the first version of this section and is corrected here. **`SOFT_MAX` /
`ggml_v_expf` closed 2026-08-25**, on a floor argument, with the 285K's memcpy arm finally run;
**`ggml-0011`'s reach across the ASR models measured 2026-08-25**, and its mechanism turned out to be
the attention head dimension rather than the clip length. **Two items are left: the decoder's
loop-invariant V transpose, and `QK^T` at `k = 64`.**)*

whisper-small is the one task still behind onnxruntime (0.57-0.72x at four threads). The
cross-attention K/V export fix bought 2.4-2.6x and closed most of it; the question is what is left.

**Split it and the answer is not ambiguous.** whisper's onnxruntime export puts the encoder in its own
graph, so the two halves can be timed directly; loom's half comes from `$LOOM_PROFILE` at **one
thread** (see P4.14's floor trap), where `ne1 = 1500` is an encoder op and `ne1 = 1` is a decode step.
Same clip (`jfk.wav`, 11 s), same transcript, Core Ultra 9 285K:

| | loom, as first published | **corrected** | onnxruntime | |
|---|---|---|---|---|
| encoder | 5.91 s (84.8%) | **5.46 s** | 2.38 s | loom **2.29x slower** |
| decode (25 tokens) | 0.65 s (9.3%) | **1.10 s** | 0.97 s | loom **1.14x slower** |
| total | 6.97 s | 6.97 s | 3.34 s | |

**Read the corrected column.** The first version of this table put the whole `CONT 1500 x 64` bucket
(471 ms) in the encoder, on the reasoning that `ne1 = 1500` is an encoder op and `ne1 = 1` is a decode
step. That bucket's `ne1` is **64**, so it is neither, and it was assigned by eye. It is the decoder's:
**454 of those 471 ms are 26 executions of the decode graph**, and only 17 ms are the encoder's own.
See "the layout bucket was the decoder all along" below, which has the node names.

So **the decoder is not ahead, and the claim that it "needs nothing" was wrong** — it is 1.14x behind,
and 41% of it is one removable memcpy. The encoder is still where most of the gap is (3.08 s of 3.63 s)
and is still the place to start, but not to the exclusion of the other graph.

#### Where the encoder's 5.91 s goes

| op | shape | calls | ms | share of run | |
|---|---|---|---|---|---|
| `MUL_MAT` | 768 x 1500 | 84 | 1941 | 29.2% | |
| `MUL_MAT` | 64 x 1500 | 12 | 1062 | 16.0% | |
| `MUL_MAT` | 3072 x 1500 | 12 | 813 | 12.2% | |
| `MUL_MAT` | 1500 x 1500 | 12 | 755 | 11.3% | |
| `CONT` | 1500 x 64 | **324** | 471 | 7.1% | *(not the encoder's — see below)* |
| `SOFT_MAX` | 1500 x 1500 | 12 | 396 | 5.9% | |
| `UNARY` | 3072 x 1500 | 12 | 273 | 4.1% | |

The 84 calls at `768 x 1500` are 12 layers x (Q, K, V, O, fc2) = 60, plus the 24 cross-attention K/V
projections the export now computes once. The 25 `NORM` at `768 x 1500` are 12 layers x 2 + the final
one — **checked, because 25 is also the token count and that coincidence looks exactly like the
cross-KV defect. It is not one; nothing encoder-width runs per decode step.**

#### The cheap test ran first, and it inverted the ordering

The first candidate was *attention is materialised rather than fused* — 31.8% of the run in
`QK^T` + `SOFT_MAX` + `AV`, writing and re-reading a 108 MB score matrix per layer, against
onnxruntime's fused `MultiHeadAttention`. The deciding experiment was a **read** of onnxruntime's own
per-op profile, so it went first.

**It is ruled out. onnxruntime does not fuse attention either** — its encoder is 96 `MatMul`, 12
`Softmax`, 24 `LayerNormalization`, 12 `BiasGelu`, 49 `Transpose`, 48 `Reshape`, and no attention node
of any kind. It materialises the same score matrix loom does. *Fusing attention is not what makes it
faster, so nothing here should start by writing a fused kernel.*

The same profile then attributes the gap directly. onnxruntime shares are apportioned over its
un-profiled 2.376 s encoder (profiling costs it ~1.18x, so only shares are used); loom's 84
`768 x 1500` matmuls include 24 cross-attention K/V projections that onnxruntime computes in its
**decoder's** first pass, so those are stripped to leave 96 against 96:

| | loom | onnxruntime | ratio | gap |
|---|---|---|---|---|
| `MatMul` / `MUL_MAT` (96 calls each) | 4017 ms | 2086 ms | **1.93x** | **1931 ms — 66%** |
| layout (`CONT` 324 vs `Transpose`+`Reshape` 97) | 471 ms | 14 ms | **33x** | 456 ms — 16% |
| `Softmax` | 396 ms | 93 ms | **4.3x** | 303 ms — 10% |
| GELU (+ bias) | 289 ms | 55 ms | **5.3x** | 234 ms — 8% |
| conv frontend | 82 ms | 40 ms | 2.0x | 42 ms — 1% |
| LayerNorm | 21 ms | 64 ms | **0.32x** | **-43 ms — loom is 3x faster here** |

#### Three candidates, re-ordered by what the profile says — and what happened to each

**Read this ordering knowing it was wrong twice.** The first candidate (fused attention) was ruled out
above by reading onnxruntime's own profile. The ordering below then put GELU *third*, as the small
safe one — and it turned out to be the only one of the three that was a kernel defect at all.

1. **GEMM efficiency at the encoder's shapes — "66% of the gap". HALF DONE, and the premise was
   wrong.** The 1.93x is an average over four shapes that do not behave alike, and splitting
   onnxruntime's own `MatMul` time by shape (which the first pass did not do) shows loom's dense GEMMs
   are **within 10-12% of MLAS** and the whole matmul gap is the two batched attention matmuls. One of
   those, `A@V`, turned out never to have entered tinyBLAS at all — see "`k % KN` rejected the whole
   matmul" below, now shipped as `ggml-0011`. `QK^T` at `k = 64` is the piece still open.
2. **Layout churn — "16%". NOT AN ENCODER ITEM.** The 324 `CONT` of `1500 x 64` are 12 encoder nodes
   and **312 executions of the decode graph**. See "the layout bucket was the decoder all along" —
   it is a loop-invariant transform of a constant, and the fix is in the exporter's `cross_kv` phase,
   not in the encoder's layout.
3. **`SOFT_MAX` and GELU — 18% together. GELU IS DONE; SOFT_MAX IS MEASURED OUT.** These were put last
   as the safe pair, on the reasoning that both are "plain elementwise kernels at 4.3x and 5.3x". Only
   half of that reasoning survived contact with a bench.

##### GELU: the only ggml activation with no SIMD path at all — DONE (`ggml-0010`)

`ggml_vec_gelu_erf_f32` was a scalar **`erff()` libm call per element**, on every architecture. That is
easy to miss because the tanh-approximation GELU beside it in `vec.h` has a 128 KB f16 lookup table and
`ggml_v_silu`/`ggml_v_expf` have hand-written SVE, NEON, AVX-512, AVX2, SSE2 and RVV paths — the exact
form *looks* like the rarely-taken one. It is not: it is what PyTorch's default `approximate="none"`
lowers to, and **loom picks it deliberately** (`op_gelu`, `src/ops/primitives_basic.cpp`, and the
exporter distinguishes the two modes in `topology_ops.py`), so it lands in transformer MLPs at full
hidden width.

Replaced with `erf(z) ~ z*P(z^2)/Q(z^2)`, degree 5 over degree 5, `z` clamped to `[-4, 4]`, result
saturated to `[-1, 1]`. **One thread, `3072 x 1500`, median of seven** (`scripts/bench12.cpp`, both
arms ggml's own functions so what is measured is the patch and not a copy of it):

| | Core Ultra 9 285K | Ryzen 3 3250U |
|---|---|---|
| `erff()` libm, per element | 19.0 ms | 121 ms |
| `ggml-0010` rational | **1.32 ms** | **5.5 ms** |
| | **14.3x** | **21.8x** |

**In model, which is the number that counts**, same binary with the kill switch flipped, one thread,
`$LOOM_PROFILE`, the post-cross-KV-fix export (84 calls at `768 x 1500`, so directly comparable to the
table above):

| `UNARY 3072 x 1500`, 12 calls | ms | share of run |
|---|---|---|
| `GGML_CPU_DISABLE_GELU_ERF_SIMD=1` | **209.2** | 3.6% |
| patched | **21.2** | 0.4% |

**9.9x in model** — less than the microbench's 14.3x, which is what a kernel that now streams memory
instead of computing should do. Against onnxruntime's 55 ms for the same 96 activations, 21 ms is
**2.6x faster**: the 8% GELU line closes and reverses. `GGML_CPU_DISABLE_GELU_ERF_SIMD=1` restores the
libm path exactly, without a rebuild.

**Do not quote the whole-transcription delta from that run** (6.115 -> 5.290 s). The two arms were not
interleaved and `MUL_MAT` moved 17% between them, which nothing in this patch touches — that is
thermal drift on a loaded box, and it inflates the end-to-end figure. The defensible statement is
**~190 ms off a ~5.9 s encoder, about 3%**; the per-op line is the evidence, not the wall clock.

**Accuracy was bounded exhaustively, not sampled** — every one of the 2^32 float32 bit patterns against
a double-precision `erf`, with the libm path measured the same way in the same sweep: **2.64e-07
against 1.08e-07 relative to `max(|x|,1)`, i.e. 2.4x the error of the path it replaces**, about two f32
ulps of the value's own scale, worst case at `x = 5.0` rather than in a tail.

**Two things that sweep caught, and a grid would not have.** (a) `P/Q -> 0.053` as `w` grows, so the
approximation *diverges* outside the fitted interval rather than merely degrading — the clamp makes the
function total, it is not a tidiness measure. (b) Without the saturation to `[-1,1]`, the residual of
`1 + z*P/Q` at the clamp is ~1e-7 instead of 0 and then gets multiplied by `|x|`: **at `x = -FLT_MAX`
the error was 2e31 for an exact answer of `-0`.** And one trap for anyone editing it: the branchless
min `0.5*(a+b-|a-b|)` vectorises where a select does not, which is why it is tempting, and it cancels
to zero in f32 once `a+b == a` — `gelu(1e30)` came back as `5e29`.

**What it costs against the real oracle.** `test_e2e_whisper_mil_export` compares the encoder output
with HuggingFace, not with loom's own other path, at `max_abs_diff < 1e-3 * ref_absmax`. Same binary,
kill switch flipped:

| encoder vs HF | libm | patched | limit |
|---|---|---|---|
| `max_abs_diff / ref_absmax` | 3.25e-04 | **4.19e-04** | 1e-3 |
| `mean_abs_diff` | 9.27e-06 | **9.36e-06** | 1e-3 |

So it spends real headroom on the max — **3.1x margin becomes 2.4x** — and essentially none on the
mean (+0.9%). The decoder logits are unchanged (`mean_abs_diff` 6.50e-06 -> 6.47e-06). Worth watching
if a second approximation ever lands in the same graph; not close today.

**And the gate was checked for teeth, per ADR-015.** Setting the clamp to 0.5 and rebuilding takes
`max/absmax` from 4.19e-04 to **0.907** and fails all four checks, encoder and decoder. It is a gate
that can see this kernel and can go red on it — the two arms above differing is the other half of that
evidence.

**Auto-vectorisation was tried first and does not work here.** The loop is branchless arithmetic plus
one clamp, and GCC 12 still refuses it — *"not vectorized: control flow in loop"* — for the clamp
written as `z<C?z:C`, as `fminf`, and as a clamp on `z^2` recovered with `sqrt`, on both x86-64 and
aarch64. (clang vectorises all of them.) Hence explicit intrinsics, in the shape of ggml's own
`ggml_v_expf`.

##### SOFT_MAX: measured out, then RE-OPENED — the floor arm was not a floor

The mechanism looked as clean as GELU's. Per row `ggml_compute_forward_soft_max_f32` makes **five
passes** (copy to scratch, scale scratch, max-reduce, exp+sum, normalise) where three suffice, and its
AVX2 loop does **a full horizontal reduction of the accumulator every 8 elements**. Folding the scale
into the exp, dropping the scratch copy and keeping four accumulators reduced once is a clean 3-pass
row, and on the *dev box* it is worth 1.34x.

**On the machine the profile came from it is worth 1.06x, and two probes say why it cannot be more:**

| `1500 x 1500 x 12 heads`, one thread, Core Ultra 9 285K | one call | vs ggml |
|---|---|---|
| ggml, 5-pass row | 31.4 ms | — |
| 3-pass, scale folded, accumulators reduced once | 29.6 ms | 1.06x |
| *probe:* 3-pass with a **fast bit-trick exp** (far too inaccurate to ship) | 32.3 ms | **0.97x** |
| *probe:* 3-pass with **no exp at all** | 31.6 ms | **0.99x** |

**That was the conclusion, and it was wrong — the two probes are not floors.** Both were written as
plain scalar C and left to the auto-vectoriser, and both were compared against a candidate written in
hand intrinsics, so what they measured was two compilers' output rather than two amounts of work. The
tell was visible in them: on the dev box the "no exp" arm comes out at 47.2 ms against the candidate's
42.6 — *slower than the thing it claims to bound*, which cannot happen if it is a floor.

Re-measured (2026-08-24) with the floor built the same way as the arm it bounds —
`cand_soft_max_row<0>`, the identical function with `ggml_v_expf` switched off at compile time — and
with the arm neither pass ever had, a `memcpy` of the same bytes. One thread, Ryzen 3 3250U,
`scripts/bench12.cpp`:

| `1500 x 18000` rows | one call | x12 calls | |
|---|---|---|---|
| ggml, 5-pass row | 56.8 ms | 682 ms | — |
| 3-pass candidate | 49.2 ms | 590 ms | **1.16x** |
| *floor:* same arm, exp switched off | 34.6 ms | 415 ms | the exp is **1.42x** of the candidate |
| *floor:* `memcpy` of the same bytes | 14.5 ms | 174 ms | 13.9 GB/s |

**ggml's row body is 3.9x the memcpy floor, so `SOFT_MAX` is not bandwidth bound here.** Where its
682 ms goes: 174 ms is the bytes, 241 ms is the pass structure above them, 175 ms is the exp, and
92 ms is ggml's two extra passes. onnxruntime's own per-node profile on the same box puts its 12
`Softmax` at **515 ms** — between the candidate (590) and the no-exp floor (415), i.e. a core doing
less work per element, not one moving fewer bytes.

**So the 285K's 1.06x stands as a number and falls as a reason.** It was measured directly there and
is not in doubt; what was never measured on either machine is the memcpy floor, which is three lines.
The item to re-open is not a soft_max rewrite — it is **`ggml_v_expf`**, which the corrected floor
prices at 1.4-1.6x of a fused row. That is the same shape of finding as `ggml-0010`'s GELU, in the
same file. See [Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md), corrected.

##### `ggml_v_expf`, and the memcpy floor on the 285K — CLOSED (2026-08-25)

The item above was the right one to open and it is now closed. Both halves of what it asked for were
run: **the memcpy arm on the 285K**, which had never been measured on the machine the original
"bandwidth bound" claim was made about, and **a candidate `ggml_v_expf`**.

| 12 calls, `SOFT_MAX 1500 x 1500 x 12`, one thread | Ryzen 3 3250U | Core Ultra 9 285K |
|---|---|---|
| ggml, 5-pass row | 600 ms | 398 ms |
| 3-pass candidate | 508 ms (**1.19x**) | 365 ms (**1.08x**) |
| *floor:* same arm, exp switched off | 372 ms — the exp is **1.37x** | 338 ms — the exp is **1.08x** |
| *floor:* `memcpy` of the same bytes | 168 ms (14.4 GB/s) | **51 ms (47.4 GB/s)** |
| ggml above the memcpy floor | 3.6x | **7.8x** |

**The 285K is FURTHER from bandwidth bound than the dev box, not closer.** That closes the original
reading for good on the machine it was made about. It also explains the 1.19x / 1.08x split without
appealing to memory at all: **the exp is 26% of the op on the dev box and 7% on the 285K**, because the
285K's cores are fast enough at the arithmetic that what remains is the traffic.

**And the exp has no accessible headroom.** `sm_expf` (`scripts/bench12.cpp`) is ggml's own fast path
with the general-case mask, `movemask` and branch — which SOFT_MAX's domain provably cannot reach —
replaced by one `max`: **0.97-1.02x on the dev box, 1.13-1.16x on the 285K, at bit-for-bit the same
worst-case error** (1.847e-07 both arms). Fourteen of ggml's sixteen operations survive the
specialisation, which is the whole story. The floor that bounds every *other* exp candidate is the same
function with the polynomial replaced by `exp(b) ~ 1 + b`: **1.64-1.95x**, so no exp rewrite at any
accuracy is worth more than that, against an exp that is 7-26% of an op that is 5.9% of the encoder.

*Do not re-propose a faster `exp` for `SOFT_MAX`* —
[Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md) carries the takeaway.

And **the 4.3x in the table above needs no suspicion after all.** It was discounted on the grounds
that a memory-bound op cannot be compared per-op; the op is not memory bound, loom is 1.24x
onnxruntime at the bench level, and the difference is work per element. The premise behind the whole
table was separately checked and does hold — onnxruntime's encoder graph alone, same clip, same
machine, is **2.49 s at `intra_op=1`** against 1.29 at 2 and 0.79 at 4, so the 2.376 s it is compared
with is genuinely a one-thread number and the split is like-for-like.

**And note the LayerNorm row: loom is 3x faster there.** Nothing in this table is uniformly behind,
which is the usual sign that the gap is specific kernels rather than a systemic disadvantage.

#### The layout bucket was the decoder all along

`$LOOM_PROFILE` buckets by `(op, ne0, ne1)` and nothing else, so **no bucket in its report says which
graph it came from**. Patching `profile.cpp` to also print `node->name` settles it in one run
(whisper-small, `jfk.wav`, one thread, Ryzen 3 3250U) — **that patch is P4.19, and it should stop
being something each investigation rewrites**:

```
151.3 ms  x27  CONT  xv_0 (reshaped) (permuted) (permuted) (cont)  ne=1500,64,12,1
147.1 ms  x27  CONT  xv_1 (reshaped) (permuted) (permuted) (cont)  ne=1500,64,12,1
   ... twelve of these, one per decoder layer ...
```

`xv_N` is a **decoder graph input** — the cross-attention V that the cross-KV fix already made
constant for the whole utterance. The decoder re-materialises its transpose **every decode step, in
every layer**: 12 x 4.6 MB = **55 MB of memcpy per token**. The chain is nodes 42-44 of the decoder
topology (`RESHAPE(xv_0) -> PERMUTE(0,2,1,3) -> PERMUTE(1,0,2,3) -> CONT`), and the `CONT` is genuinely
required — a double permute leaves `nb[0] != type_size`, which `ggml_mul_mat` will not take.

**The count is exact arithmetic on the published figure, not a re-measurement.** 324 = 12 encoder nodes
+ 12 layers x 26 decoder executions (25 tokens plus the prompt prefill). The run here produced 26
tokens and 336 = 12 + 12 x 27. So on the 285K, **454 of the 471 ms are the decode loop**.

At one thread on the dev box the twelve nodes are **1730 ms of a 19.28 s run — 9.0% of the whole
transcription and 47% of the decode loop**, larger than the LM head (352 ms) and larger than every
per-step projection put together (694 ms).

##### The fix — DONE 2026-08-29 (P4.18 item B), and it was one wrapper AND one rewrite

`_WhisperCrossKvWrapper.forward` now emits the V half already head-split and transposed —
`[1, frames, d_model]` becomes `[1, heads, head_dim, frames]`, the layout `ggml_mul_mat` wants as its A
operand — so the transpose happens **twelve times per utterance instead of 324**. K is untouched:
`Q @ K^T` is `transpose_y=True`, which composes to a bare `MUL_MAT` over the natural layout, and only
`scores @ V` is the `transpose_y=False` case that `topology_ops._op_matmul_x_y` composes as
PERMUTE + CONT.

**Changing the wrapper is only half of it, and the half that does not work alone.** HF's attention
reshapes and transposes whatever `v_proj` returns *whatever shape reaches it*, so the traced decoder
still contains the chain. `whisper_export.hoist_cross_v_transpose` deletes it afterwards — it matches
`RESHAPE(xv_i) -> PERMUTE[0,2,1,3] -> PERMUTE[1,0,2,3] -> CONT` per layer, checks each intermediate has
exactly one consumer, rewires the `MUL_MAT` to read `xv_i` directly, and **raises** if any of that is
not exactly so. It runs from `ExportPhase.topology_rewrite`, a new hook whose whole justification is
that "invariant across the driver's calls" is a fact about the *driver's loop* that no pass over one
traced function can see.

**Measured.** Ryzen 3 3250U, `samples/jfk.wav`:

| | before | after |
|---|---|---|
| `CONT` total, 1 thread | 1583 ms (2007 calls) | **209 ms** (1695 calls) |
| the `xv_N` buckets | 12 buckets x 27 calls | **one bucket x 12 calls**, 63 ms |
| whole transcription, 2 threads | 10.64 s | **9.63 s** |

**1.106x end to end** — 10 paired rounds in both orders (old-first and new-first, because the second
arm of a fixed order inherits the box's thermal ramp), min 1.041, max 1.176. The decoder topology loses
48 nodes; the `cross_kv` topology gains 36.

**Numerically it is a no-op, and that is checked rather than argued.**
`test_e2e_whisper_mil_export` teacher-forces the decoder over the whole prompt against HF's own logits,
and the diffs are **bit-identical** to the previous export's — `mean_abs_diff` 6.29746e-06 and
`max_abs_diff` 3.52859e-05 on the decoder, `max/absmax` 4.34e-04 on the encoder, 67/67 either way. That
test mattering here is the point: whisper's gate compares tensors, so a wrong V permutation fails it,
where the transcript alone would not have (Retro-006).

**Its cost is a release cost.** A whisper re-export and a fixture refresh (ADR-003), so the Hub copy
needs a push — it lands in rc7, and rc6 shipped without it.

**And the general lesson, which is Retro-012's shape again:** a bucket keyed on `(op, ne0, ne1)` cannot
tell you which graph it is in, and `ne1 = 1500 means encoder` only classifies buckets whose `ne1` is
1500 or 1. This one's was 64. The `NORM` coincidence in the same table *was* checked; this one was not.

#### `k % KN` rejected the whole matmul — DONE (`ggml-0011`)

Splitting onnxruntime's encoder profile by input shape — one thread, same clip, same box, its numbers
carrying ~1.18x profiling inflation, loom's from `$LOOM_PROFILE`, the `768 x 1500` bucket apportioned
by FLOPs to strip the 24 `cross_kv` calls so it is 96 against 96:

| Ryzen 3 3250U, 1 thread | loom | onnxruntime | ratio |
|---|---|---|---|
| Q/K/V/O + fc2 (60 calls) | 4866 ms | 4434 ms | **1.10x** |
| fc1 (12) | 2493 ms | 2235 ms | **1.12x** |
| `QK^T` (12) | 2570 ms | 1193 ms | **2.15x** |
| `A@V` (12) | 2638 ms | 1185 ms | **2.23x** |
| `Softmax` | 949 ms | 515 ms | 1.84x |
| GELU | 77 ms | 206 ms | **0.37x** |
| LayerNorm | ~90 ms | 176 ms | **0.51x** |
| conv frontend | ~200 ms | 176 ms | 1.14x |

**The dense GEMMs are not the problem.** P4.15 finished that job: 1.10-1.12x of MLAS. The entire
matmul gap — 2830 ms, **75% of the encoder gap** — is the two batched attention matmuls. The 1.93x in
the earlier table is an average that hides this, and "GEMM efficiency at the encoder's shapes" as an
item was aimed at the wrong half of it.

**The obvious explanation is wrong, and was tested first.** loom hands `QK^T` a *permuted* `src0` (`K`
is `[64, 12, 1500]` contiguous, and `permute(0,2,1,3)` gives `lda = 768` floats while only 64 are
used), where onnxruntime materialises a dense transpose per head. Materialising it in loom too is
worth **4%** (`scripts/bench13.cpp` section 1): 154.3 -> 147.7 ms plus a 1.5 ms `cont`. Not the
mechanism, and the section is kept so nobody spends a day on the transposes again.

**What it is:** `sgemm.cpp`'s `matmul` opened with `if (k % KN != 0) return false;`, which sends the
*whole* matmul to ggml's generic one-output-element-per-call kernel. `A@V` contracts over the encoder's
1500 frames, and **1500 % 8 == 4** on AVX2 (`KN = 8`), **1500 % 16 == 12** on AVX-512. A sweep at that
exact shape, one thread:

| `m=64, n=1500` | k=1496 | k=1497 | k=1500 | k=1502 | k=1504 |
|---|---|---|---|---|---|
| GFLOP/s | 44.5 | 16.4 | **20.8** | 17.0 | 47.0 |

(`scripts/bench13.cpp` section 3 — which is **flat** once `ggml-0011` is in the tree, so reproducing
that column means building a ggml with the patch removed.) `n` divisibility does not matter (46.0
GFLOP/s at n=1496/1500/1504) and `m` was already handled by `ggml-0003`. **`k` is the contraction, so in attention it is a sequence length — a number nothing
rounds**, which is exactly `ggml-0003`'s own argument about `m = 287` on the other axis.

**NEON has `KN = 4`, and 1500 % 4 == 0.** The Pi never sees this. `scripts/bench6.cpp` being
aarch64-only was not an inconvenience in getting to this item — it is *why* the whole of P4.15 ran on
the one ISA where whisper's contraction happens to be aligned.

`ggml-0011` splits the contraction the way `ggml-0003` splits the rows: aligned prefix through the
vector loop, the <= `KN-1` leftovers accumulated as scalars inside the tile before the `hsum`.
Measured, one thread:

| | baseline | patched | |
|---|---|---|---|
| bench, `m=64 n=1500 k=1500` x 12 heads | 176.2 ms | **81.8 ms** | **2.15x** |
| in model, `MUL_MAT 64 x 1500` (12 calls) | 2240 / 2460 ms | **1119 / 1136 ms** | **2.0-2.2x** |
| in model, every other bucket | — | unmoved | |

(two interleaved runs per arm; `MUL_MAT 768 x 1500` 5508 vs 5517 ms across the pair, which is what
"unmoved" means here.) End to end at four threads, interleaved, six launches: 12.74/12.68/12.46 ->
12.44/12.16/12.17 s — a wall that includes a 2.4 GB GGUF load, so the per-op line is the evidence.

**One trap for anyone editing it.** Folding the tail loop into a single epilogue — where it is
*statically empty* whenever `k % KN == 0` — costs the small-`k` shapes **30%**: `QK^T` at `k = 64` went
154 -> 201 ms. The aligned case needs byte-for-byte the epilogue it had, so there are two of them and
the branch is outside the tile.

**Accuracy, and the gate.** Against a double-precision reference over every `k` in [1, 40] plus
63/64/65 and 1496-1504 (`scripts/bench13.cpp --check`), the worst relative error is 2.6e-05, and the
aligned `k` sit in the same place as the unaligned ones — k=1496 2.1e-05 against k=1500 2.3e-05, i.e.
f32 accumulation noise rather than a dropped term. `tests/ci/test_tinyblas_gemm.cpp` now sweeps the
whole `k % KN` residue range the way it already swept `m % 4`, and it was **verified red**: deleting
the tail accumulation fails 16 of its 113 checks, at exactly the unaligned k, by 2e-02 to 1.7e-01 —
four orders of magnitude past its 1e-5 tolerance (ADR-015). `test_e2e_whisper_mil_export` against HuggingFace
moves `max/absmax` **4.19e-04 -> 4.34e-04** against a 1e-3 limit and `mean_abs_diff` 9.36e-06 ->
**9.33e-06** (better), 67/67 either way; the full suite is 150/150.

#### What `ggml-0011` is worth on the OTHER ASR models — DONE (2026-08-25), and the mechanism was misnamed

The patch was found on whisper, where the ragged contraction is the encoder's **1500 frames**. That
number is fixed (whisper pads every clip to 30 s), so the first reading of the accidental Conformer
result — `conformer_ctc_mil` 856 -> 718 ms, 1.19x — was that a Conformer's frame count is *dynamic*,
`276 % 8 == 4`, the subsampling stride is 4, and therefore **"about half of all clips hit this, per
utterance, at run time"**.

**That mechanism is wrong, and the measurement that says so is a clip sweep.** Six clips trimmed from
`jfk.wav` in 160 ms steps — 160 ms of audio is 16 mel frames is exactly **4 encoder frames**, so the
encoder length steps by 4 and its residue mod 8 alternates 0 / 4. Paired arms
(`scripts/paired_arms.py`, baseline = the same tree with `ggml-0011` reverse-applied), 11 rounds each,
one thread, Ryzen 3 3250U:

| clip | encoder length | `% 8` | p10 | **median** | p90 |
|---|---|---|---|---|---|
| 10.20 s | 256 | **0** | 1.199 | **1.229** | 1.293 |
| 10.36 s | 260 | 4 | 1.160 | **1.196** | 1.237 |
| 10.52 s | 264 | **0** | 1.173 | **1.202** | 1.232 |
| 10.68 s | 268 | 4 | 1.183 | **1.236** | 1.260 |
| 10.84 s | 272 | **0** | 1.178 | **1.229** | 1.246 |
| 11.00 s | 276 | 4 | 1.198 | **1.244** | 1.293 |

**Flat.** The aligned clips gain as much as the unaligned ones, every p10 is above 1.16, and the
encoder length's residue does not appear in the result at all.

**What it actually is: the attention HEAD DIMENSION.** `$LOOM_PROFILE` on both arms, same clips, names
the buckets that move — and the two candidate contractions separate perfectly:

| bucket (output `ne0 x ne1`) | contracts over | base | patched | |
|---|---|---|---|---|
| **enc = 256, aligned** | | | | |
| `511 x 256` (`QK^T`, rel-pos) | head dim **44** | 409 ms | 170 ms | **2.41x** |
| `256 x 256` (`QK^T`) | head dim **44** | 210 ms | 92 ms | **2.27x** |
| `44 x 256` (`A@V`) | encoder length **256** | 27.6 ms | 29.6 ms | **0.93x — nothing** |
| **enc = 276, unaligned** | | | | |
| `551 x 276` (`QK^T`, rel-pos) | head dim **44** | 540 ms | 239 ms | **2.26x** |
| `276 x 276` (`QK^T`) | head dim **44** | 279 ms | 124 ms | **2.24x** |
| `44 x 276` (`A@V`) | encoder length **276** | 108 ms | 38 ms | **2.82x** |

The clip-dependent half is real — `A@V` moves 2.82x at 276 and not at all at 256, exactly as the
original story predicted — but it is **69 ms of a 739 ms total delta, under 10%**. The other 90% is
`QK^T`, whose contraction is `d_model / n_heads` = **176 / 4 = 44**, and **44 % 8 == 4 on AVX2 for
every clip this model will ever see.**

**So the rule is about the model, not the utterance,** and it predicts the other three:

| model | head dim | `% 8` | paired, 1 thread, dev box |
|---|---|---|---|
| `conformer_ctc_mil` | **44** | **4** | **1.196-1.244x**, every clip |
| `whisper_mil` | 64 | 0 | **1.059x** (p10 1.033) — from `A@V` over its fixed 1500 frames |
| `gigaam_mil` | 48 | 0 | 1.025x, p10 0.789 — **unresolved** |
| `parakeet_tdt_mil` | 128 | 0 | 0.992x, p10 0.868 — **unresolved** |
| `parakeet_rnnt_mil` | 128 | 0 | 0.995x, p10 0.898 — **unresolved** |

The three that gain nothing all have head dimensions divisible by 8, and their `A@V` buckets are 1-2%
of the run rather than whisper's 16%. **The backlog's "gigaam 0.97x / parakeet-tdt 1.03x are inside
the noise floor — re-run paired" was right, and re-running them paired confirms them as unresolved
rather than converting them into numbers.**

**What belongs in the README's ASR column** is therefore the Conformer number and not a general claim
about dynamic frame counts: a model whose head dimension is not a multiple of 8 pays this on every
utterance, and Conformer-CTC small is one.

#### The same thing on the 285K — DONE (2026-08-25)

Everything above is the 2-core dev box; the 285K is Arrow Lake, so AVX2, so `KN = 8`, so the mechanism
must carry and only the magnitudes were open. Same harness, same clips, same two arms built from the
same tree:

| paired, one thread | p10 | **median** | p90 |
|---|---|---|---|
| conformer, enc 256 (aligned) | 1.149 | **1.192** | 1.222 |
| conformer, enc 276 (unaligned) | 1.173 | **1.222** | 1.229 |
| whisper-small, 11 s | 1.054 | **1.085** | 1.135 |

It carries, to within a couple of percent of the dev box, flat across the residue there too.

**At 24 threads none of it is resolvable, and more rounds do not fix it.** Eleven paired rounds give
conformer 0.930 / 1.064 and whisper 0.993, every one straddling 1.0; **41 and 31 rounds give conformer
0.955 (p10 0.715, p90 1.231) and whisper 0.988 (p10 0.849, p90 1.089)** — a tighter median around the
same place, and a spread that has not moved.

The reason is visible in the raw arms rather than inferred: whisper's baseline spans **1.102-1.328 s
across launches of the same binary**, which is the per-launch thread-placement lottery the README item
below names — and **pairing does not cancel it**, because placement is drawn fresh in each process
rather than drifting slowly across one. Everything a paired test is good for assumes the noise is
shared *within* a pair; this noise is not. A 42 ms Conformer transcription at 24 threads is far under
that spread.

**Flagged, not claimed:** at 24 threads both medians sit slightly *below* 1.0 over 41 and 31 rounds,
and whisper's patched median (1.225 s) is consistently above its baseline (1.119 s) while their minima
are within 1%. That is what a patch that stops helping — or costs a little — at high thread counts
would look like, and it is also what this estimator does to a null. Not resolvable here.
**`ggml-0011` is a ONE-THREAD result on this box**; if a 24-thread number is ever needed, the estimator
has to change (pin the placement, or compare per-op rather than end to end), not just the round count.

#### `ggml-0011` regressed aarch64 for four days — P4.20, FIXED 2026-08-29

Found while re-measuring the README's ASR column, which is the only reason it was found at all: the
Pi's ASR cell had fallen **0.58x → 0.45x** and nothing in the change log predicted it.

Raspberry Pi 4B, Cortex-A72, 4 threads, cooldown to 58 C before **every** measurement, arms alternated
inside each round, one tree, one checkpoint, three builds differing only in which patches
`cmake/patches/` contained:

| Pi 4B, 4 threads | shipped (0001-0011) | without `ggml-0011` | without 0010 and 0011 |
|---|---|---|---|
| whisper-small, `loom_cli` wall | 49.42 / 49.33 s | **42.35 / 39.51** | 40.44 / 39.62 |
| VITS synthesis, 73216 samples both | 1.2545 / 1.2594 s | **1.1142 / 1.0930** | — |
| Qwen3-0.6B, 24 tokens, best of 5 | 16.80 / 16.68 s | 16.66 / 16.58 | — |

**and against onnxruntime on the same board, which is what the README's Pi row now says:**

| Pi 4B, 4 threads | as shipped | without `ggml-0011` | published 2026-08-24 |
|---|---|---|---|
| TTS | **0.84x** | ~0.95x | 0.98x |
| LM | **1.05x** | 1.06x | 1.08x |
| ASR | **0.45x** | ~0.54x | 0.58x |

**The LM is untouched**, and that is the control this finding needed: a decode step's `mul_mat` has
`ne1 = 1` and never reaches the blocked GEMM, so a patch to that kernel should do nothing there. It
does nothing there. The two tasks that do reach it both lose.

(These whisper wall times include the model load, which both arms pay; the harness figure for the same
board and build is ~46 s. **Every measurement on this board cools it to 58 C first** — that is the only
reason any of them repeat, and `~/pi_accept.sh` on the Pi is the acceptance sweep with that protocol in
it. The checkpoint throughout is `~/bench/fixtures/whisper_new.gguf`, **`md5 1deaac83…`, the rc6
export**: this item is about a ggml kernel, so comparing against the published Pi row needs the
published artifact, which is also why the ASR cell below lands at 0.57x rather than at the ~0.63x the
newer V-transpose export would give on the same board. The variant builds used for the bisect were
deleted once the fix landed; rebuilding one is `cmake -B build-no11` with the patch moved aside, about
40 minutes there.)

**`ggml-0011` costs this board 1.17-1.25x on whisper and 1.13-1.15x on VITS.** `ggml-0010` is inside
the noise here (42.35 against 40.44, then 39.51 against 39.62), which is what a 3.6%-of-runtime op can
be and agrees with the 0.9% the README already claims for it on this machine.

**Why it can only cost on this ISA, and why that was predictable from the patch's own analysis.** The
patch exists because tinyBLAS rejected any matmul with `k % KN != 0`. On AVX2 `KN = 8` and whisper's
1500-frame contraction leaves 4, so the patch converts a rejected matmul into an accepted one — 2.15x.
**On NEON `KN = 4`, and 1500, 768, 3072 and 64 all divide 4**, so the tail it adds is *never taken* on
any shape this model has. Everything that remains is its restructuring of the **aligned** path — and
this epic already records that folding that path's epilogue costs the aligned case **30%** on x86,
which is why the patch carries two epilogues with the branch outside the tile. The aarch64 build was
never re-measured after it landed. `ggml-0001` exists precisely because GCC's register allocation for
the NEON tile is fragile (15.6 → 25.1 GFLOP/s from a tile it can actually allocate), so a second
structural change to the same loop is exactly where a NEON regression would be expected to appear.

##### What it actually was, and the fix

**Step 1 reproduced it at the kernel, and made it bigger.** `scripts/bench15.cpp` on the Pi, one
thread, three rounds, both builds — whisper's five encoder GEMMs, **every one of whose `k` divides
NEON's `KN = 4`, so none of them ever takes the tail this patch adds**:

| Pi 4B, m x n x k | with `ggml-0011` | without | |
|---|---|---|---|
| 1500 x 1500 x 64 (`QK^T`) | 4.67 GFLOP/s | 7.62 | **1.63x** |
| 64 x 1500 x 1500 (`A@V`) | 4.70 | 7.84 | **1.67x** |
| 768 x 1500 x 768 (`proj`) | 4.86 | 8.50 | **1.75x** |
| 3072 x 1500 x 768 (`fc1`) | 4.86 | 8.56 | **1.76x** |
| 768 x 1500 x 3072 (`fc2`) | 4.01 | 5.28 | **1.32x** |

That it hits **every** shape rather than the small-`k` ones is what ruled out the epilogue's runtime
cost and pointed at the main loop's code generation.

**Step 2 bisected the patch into its two hunks**, four builds of one `.so`, ABBA:

| variant | `proj` | `QK^T` |
|---|---|---|
| shipped (truncated `kk` + two epilogues) | 4.87 | 4.65 |
| `kk = k`, two epilogues kept | 4.82 | 4.45 |
| truncated `kk`, single epilogue, **tail removed** | **8.56** | **7.67** |
| neither (unpatched) | 8.47 | 7.57 |

**It is not the `kk` arithmetic — it is the presence of the scalar tail loop in the tile function.** A
fifth build folding the tail into one epilogue rather than two is also slow (5.12), so it is not the
number of epilogues either. The tail reaches `Aat`/`Bat` after the main loop, and on GCC/aarch64 that
changes what the register allocator does with the 4x3 NEON tile — **the same fragility `ggml-0001`
exists for, approached from the other side.** The patch's own x86 note says folding the tail into one
epilogue costs *x86* 30%; both ISAs were saying the same thing about this code, and only one had been
asked.

**Step 3: the tail got its own `NOINLINE` function, dispatched on `k % KN` before the tile.** The
aligned path is then instruction-for-instruction what it was before the patch existed, on every ISA —
which is what the x86 trap note demands and what aarch64 needed. Measured on both:

| | x86 (285K, AVX2) | aarch64 (Cortex-A72) |
|---|---|---|
| `A@V` k=1500, **where the tail fires** | 150.7 vs 53.2 unpatched — **2.83x, intact** | n/a (1500 % 4 == 0) |
| `QK^T` k=64, aligned | 119.3 vs 118.8 shipped / 120.7 unpatched | 7.38 vs 7.54 unpatched |
| `proj` k=768, aligned | 162.0 vs 162.2 / 161.9 | 8.43 vs 8.38 |
| whisper end to end | 2.076 s vs 2.086 s — unmoved | 36.4 s vs 46.4 s — **1.27x** |
| `test_tinyblas_gemm` | 113/113 | 113/113 |

**Acceptance — the Pi's three README cells against onnxruntime**, cooled to 58 C before every
measurement, arms alternated:

| Pi 4B, 4 threads | published 2026-08-24 | regressed | **fixed** |
|---|---|---|---|
| TTS | 0.98x | 0.84x | **0.96x** |
| LM | 1.08x | 1.05x | **1.06x** |
| ASR | 0.58x | 0.45x | **0.57x** |

**Neither half of that table was optional.** The x86 column is what says this is a fix rather than a
revert with extra steps; the aarch64 column is what the patch shipped without. `UPSTREAM.md` carries
both now, and that is the standing rule this cost bought
([Retro-019](../retros/retro-019-a-patch-measured-on-one-isa.md)).

#### Where the encoder gap is NOW, measured per shape — P4.18 item A (2026-08-28)

The table above is what motivated `ggml-0011`. **This one is the state after it**, and it exists
because the version of it that P4.18 was steering by had one measured column and one **composed** one
(`A@V` scaled by the microbenchmark's 2.15x, GELU by `ggml-0010`'s), while every remaining decision
rested on it.

Ryzen 3 3250U, **one thread**, `taskset -c 0`, `samples/jfk.wav`. **Five interleaved rounds** — one
loom transcription and one onnxruntime encoder run per round — so the box's thermal drift, 12% across
five runs here, moves both arms together instead of landing on one of them. loom's column is
`$LOOM_PROFILE` with `$LOOM_PROFILE_NODES=1`, split **by node** so the 24 `cross_kv` projections are
removed rather than apportioned by FLOPs; onnxruntime's is its own per-node profile keyed on
**(op, input shape, output shape)** — the output shape is what separates `fc1` from `fc2`, which share
an input shape with the projections — apportioned over a separately measured un-profiled encoder wall
of **9.30 s**. Profiling costs onnxruntime **1.01x** here, not the 1.18x it costs on the Pi.

| encoder piece | calls l/o | loom | onnxruntime | ratio (lo-hi) | share of gap |
|---|---|---|---|---|---|
| **`QK^T`** | 12/12 | 1957 ms | 1027 ms | **1.96x** (1.77-2.08) | **51%** |
| `Softmax` | 12/12 | 780 | 491 | 1.60x (1.54-1.71) | 16% |
| `fc2` | 12/12 | 2302 | 2016 | 1.14x (1.12-1.26) | 16% |
| `fc1` | 12/12 | 2240 | 2034 | 1.09x (1.04-1.22) | 11% |
| Q/K/V/O projections | 48/48 | 2138 | 1970 | 1.11x (1.06-1.37) | 9% |
| `A@V` | 12/12 | 1101 | 1046 | **1.07x** (1.02-1.21) | 3% |
| layout (`CONT` vs `Transpose`) | 152/97 | 124 | 70 | 1.77x (1.63-1.83) | 3% |
| conv frontend | 2/2 | 179 | 155 | 1.15x (1.08-1.22) | 1% |
| norm + bias + residual | 186/108 | 212 | 281 | 0.77x (0.73-0.82) | −4% |
| GELU | 14/14 | 78 | 195 | **0.41x** (0.38-0.43) | −7% |
| **encoder total** | | **11110 ms** | **9284 ms** | **1.20x** | 1826 ms |

**`ggml-0011` did in the model what the microbenchmark predicted.** `A@V` is **2.23x in the table
above and 1.07x here** — the largest single item in the pre-patch split is now the smallest. That is
the check this item existed to run, and it passed: the composed column was not wrong about where the
rest of the gap is, so the item below is still aimed at the right thing.

**`QK^T` is 51% of what is left**, against the composed column's 58%. Closing it entirely would take
the encoder from 1.20x to **1.10x**, and it is **6.0% of a whole one-thread transcription** (15.4 s of
node time) — that is the number to weigh the work below against.

**The dense GEMMs are 1.09-1.14x each AND 36% of the gap.** Both are true and neither replaces the
other: per shape they are inside P4.15's stated 1.10-1.12x of MLAS and there is nothing in them to
win; in aggregate they are 6.7 s of an 11.1 s encoder, so their 12% is 660 ms — more than `Softmax`,
`A@V` and the layout bucket combined. **A gap can be dominated by the path that is already fast.**

**Two rows are loom ahead, and they are this thread's two ggml patches**: GELU at 0.41x (`ggml-0010`)
and the norm/bias/residual group at 0.77x. Together they are −11% of the gap; without them the encoder
would read 1.22x instead of 1.20x.

**What is not in either column.** loom's 24 `cross_kv` projections — 1090 ms at one thread — are a
separate topology with no counterpart in `encoder_model.onnx`, because onnxruntime computes those
tensors on its decoder's first pass. They are real work both engines do; they are simply not encoder
work on either side. The mel frontend is excluded for the same reason: onnxruntime's is `transformers`
in numpy, outside its timer.

**The absolute times here are ~20% below the ones in the table above, and that is the box, not a
regression.** Buckets that nothing touched between the two measurements moved by the same factor as
the ones that did; only the *ratios*, taken inside a round, are comparable across the two tables. It
is the reason this one was taken paired.

#### `QK^T` at `k = 64` — STILL OPEN, and the largest thing left in the encoder

`k = 64` divides 8, so `QK^T` does run through tinyBLAS — at **23.5 GFLOP/s where the same core does
44 at k >= 256**:

| `m=n=1500` | k=64 | k=128 | k=256 | k=512 | k=768 |
|---|---|---|---|---|---|
| GFLOP/s | **23.5** | 36.7 | 41.6 | 43.8 | 44.4 |

(`scripts/bench13.cpp` section 2a, and this one is NOT flattened by `ggml-0011` — every k in it
already divided 8.)

MLAS drops only 9% over the same range (34.8 GFLOP/s at k=64 against 38.3 at k=768, from its own
per-node profile), so it is a property of ggml's kernel and not of the shape. **Measured against
onnxruntime in the model** (the table above): 1957 ms against 1027, a **930 ms gap that is 51% of
everything left in the encoder** and 6.0% of a one-thread transcription. *Do not start by writing a
fused attention kernel* — that is still ruled out above.

**First, the estimator, because it cost two wrong answers here.** The same binary measured this shape
at 27.9 GFLOP/s and, twenty minutes later on a box still warm from a `ctest` run, at 12.6 — a 2.2x
swing. Min-over-interleaved-rounds fixes the worst of it (every shape runs once per round, rounds
repeat, and a `proj`-shaped GEMM re-runs every round as a clock witness) but **it is still not enough
on this machine**: the witness's own worst/best spread is 1.4-2.5x, which is larger than most of the
effects being tested, and a longer table depresses every row in it by ~15% as the laptop heats.

**What does work here is a PAIRED test.** Two arms run back to back inside one round, the *ratio* is
recorded per round, and the report is the median ratio with its p10/p90. Clock drift moves both halves
of a pair together and cancels; two independent minima do not. Everything below is 31 paired rounds.
**Nothing smaller than about 1.2x is resolvable on this box even so**, and a result whose p10 crosses
1.0 should be read as "weak" rather than as a number.

**Second, the ceiling, because 44 GFLOP/s is not "peak".** The dev box is Zen+, which has **128-bit
FPU datapaths**: a 256-bit FMA is two uops retiring one per cycle, so single-core F32 peak is 8 lanes
x 2 flops x ~3.4 GHz = **~54 GFLOP/s**, not the ~112 a lane count suggests. The `proj` witness lands
at 84-88% of that on a cool run — which is the real reason loom's dense GEMM is only 1.10x behind
MLAS: **there is nothing there to win.**

##### The size of the thing, and what is under it

| paired, 31 rounds, ratio of rates (`scripts/bench14.cpp`) | p10 | median | p90 |
|---|---|---|---|
| `QK^T` k=64 against `proj 768x1500x768` | 1.66 | **2.10** | 2.48 |
| `BM=4` (m=1504) against `BM=1` (m=1500), k=64 | 0.95 | **1.16** | 1.45 |
| `BM=2` (m=1496) against `BM=1` (m=1500), k=64 | 0.86 | 1.06 | 1.36 |
| `BM=4` (m=1504) against `BM=1` (m=1492), k=64 | 1.01 | **1.13** | 1.58 |
| *control:* the same BM pair at **k=768** | 0.89 | **1.02** | 1.09 |

So `QK^T` at k=64 runs at about **half** the rate of a projection-shaped GEMM, and **tinyBLAS's row
blocking is a real but partial contributor**: `BM = 4` is worth ~1.15x at k=64 and **nothing at k=768**
(1.02, which is the control behaving exactly as a blocking story predicts). It is not the whole 2.1x,
and its p10 sits on 1.0, so it is weakly resolved.

**And whisper's `QK^T` cannot reach `BM = 4`.** tinyBLAS picks `BM` from `m % 16 / 8 / 4`, and
1500 % 16 == 12, so it gets `BM = 1` — the least blocked of the three, for the same reason `k = 1500`
missed `KN`: a frame count is a number nothing rounds.

##### What was tried, and what each attempt is worth knowing for

| candidate | measured | verdict |
|---|---|---|
| a 16-row prefix so `BM = 4` is reachable for an m that is only `% 4` | 27.2 -> 21.1 GFLOP/s in ggml | **a regression — but on the tail, not the idea** |
| rows inner, columns outer, so a 4-float store finishes a 64-byte line of C | 1.26x once; **no dependence on the size of C** on re-run | **mechanism falsified** |
| a cheaper `hsum` epilogue — 3 `hadd` + one `_mm_storeu_ps` for four reductions | 7.77 -> 7.33 ms standalone | 1.06x, under this box's noise floor — **and right: 1.085x with counters on the 285K, see below** |
| `ggml-0002`'s aarch64 address hoist applied to x86 | 27.3 against 27.2 | **neutral** |

**The 16-row prefix failed on the row tail, not on the idea** — which the paired table above now makes
clear, since `BM = 4` really is worth ~1.15x. `ggml-0003`'s `gemm_tail` is `gemm_bloc<1,1>` per output
— one `hsum` per element instead of one per twelve — so widening the tail from <= 3 rows to <= 15
costs more than the better schedule returns. **The shape of a fix that could work is a cascade**: a
16-row prefix at `BM = 4`, then the next 8 at `BM = 2`, then 4 at `BM = 1`, then the existing <= 3 row
1x1 tail. That needs a row offset threaded through `gemm`'s job partitioning, which is threaded code
this box has two cores to test, for a ceiling of ~1.15x on one op — about 1.7% of a transcription.
Worth doing on the workstation, not here.

**The store-locality one is the cautionary tale.** A standalone 4x3 tile measured 36.4 GFLOP/s with
rows inner against 28.9 with columns inner — same tile function, verified identical C, and the gap
vanishing at k=768 exactly as a store-traffic story predicts. It looked settled. It was not: re-run
with C shrunk until it fits in cache and the flops held equal, the ratio went 1.07 / 1.28 / 0.96 /
1.04 / 0.96 / 0.98 — **no dependence on the size of C at all**, which is the one thing a store-traffic
mechanism has to show. That is a falsified mechanism rather than a noisy number, and it is the useful
half of the result.

##### The counters answered it — P4.18 item C, 2026-08-29

Run on the workstation with `perf`, one thread, pinned to a P-core with `cpu_core/`-prefixed events
(`scripts/bench15.cpp`, which exists so a counter reading covers ONE shape instead of a table of
shapes that behave differently). **Instruction counts are deterministic; they resolve what this
thread's timings could not.**

**1. The FP port is not saturated at `k = 64`, so the time is not arithmetic.**

| m=n=1500, 1 thread, 285K P-core | k=64 | k=768 |
|---|---|---|
| GFLOP/s | 118.8 | 158.4 |
| instructions retired per FMA | **4.16** | **2.08** |
| IPC | **5.40** | 3.50 |
| FMA per cycle (2/cycle is the port) | **1.15** | 1.73 |

At `k = 768` the kernel is at 86% of the FP ports. At `k = 64` it is at 58%, while retiring 5.4
instructions per cycle — it is **retirement-bound on instructions that are not FMAs**.

**2. What those instructions are: a fixed cost per OUTPUT, not per flop.** Sweeping `k` and fitting:

```
instructions per output element = 0.2355 * k + 18.0        (within 1.5% at every k in [32, 768])
```

| k | 32 | 64 | 128 | 256 | 512 | 768 |
|---|---|---|---|---|---|---|
| GFLOP/s | 79.0 | 118.8 | 137.2 | 153.6 | 158.4 | 158.4 |
| insn/output | 25.9 | 33.3 | 48.1 | 77.9 | 138.1 | 199.3 |
| the intercept's share | 70% | **54%** | 37% | 23% | 13% | 9% |

The intercept is **the tile epilogue** — `gemm_bloc` ends by horizontally reducing `RN*RM` vector
accumulators to scalars and storing each one on its own, and that is the only per-output,
`k`-independent work in the function. **At whisper's `k = 64` it is 54% of everything the core
retires.** The whole GFLOP/s curve is this one term going away.

**3. `BM` is NOT the mechanism, and the ~1.15x above is a cache effect that does not travel.** All
three `BM` branches call `mnpack<4, 6, BM>` — the *register tile is 4x6 in every one of them*; `BM` is
the outer block, i.e. cache blocking. Counted at `k = 64` on the 285K:

| m | 1500 (BM=1) | 1496 (BM=2) | 1504 (BM=4) |
|---|---|---|---|
| instructions | 15.06e9 | 15.01e9 | 15.07e9 |
| GFLOP/s | 117.1 | 118.6 | 119.6 |

**Identical instruction counts and rates within 2%.** The 1.15x measured on the Ryzen was blocking
working against a 4 MB L3; the 285K has 36 MB and whisper's operands are 384 KB. **So the "~1.15x is
explained" line above explained it with the wrong mechanism, and the `BM` cascade it recommends is not
worth building.** Nothing was wrong with the Ryzen measurement — it was a machine-specific cache
result read as a code-shape one.

##### Pricing the fix, and why it is NOT shipped

If the epilogue is the mechanism, batching it is the direct test. The four outputs of an `RM = 4` tile
column are contiguous in C, so they can be reduced together: three `vhaddps`, an extract, an add and
one 16-byte store for four outputs, against four independent `hsum`s and four scalar stores. Written as
an overload on `<4, __m256, float>` so no other type, tile width or ISA changes at all.

**It does exactly what the model says, and the model is why it is not enough.**

| | before | after |
|---|---|---|
| instructions per output | `0.2355k + 18.0` | **`0.2357k + 11.5`** |
| — saved, at every k measured | | 6.67 / 6.68 / 6.68 / 6.77 |
| 285K, k=64, 9 ABBA rounds | | **1.085x** (p10 1.021, p90 1.145) |
| 285K, k=768, 9 ABBA rounds | | 1.022x (p10 1.016) |
| Ryzen, k=64, 9 ABBA rounds | | 1.023x — **p10 0.982, unresolved** |
| `tests/ci/test_tinyblas_gemm` | | 113/113 |

A 20% instruction cut at `k = 64` buys 8.5% of the time, so the kernel is not *purely* retirement-bound
either — three dependent `vhaddps` at 6-cycle latency serialise part of what they save. Weighted over
the encoder's real bucket mix that is **~2.8% of the encoder, 1-2% of a transcription**.

**Declined on exactly the grounds `SOFT_MAX`'s pass fusion was declined**: a ggml patch carried forever
for ~1-2% end to end, resolvable on one of the two x86 machines and not the other. It is recorded in
[Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md) with its numbers so it is not
re-proposed as a guess — it is now a measured option, not an idea.

##### Where the rest of it actually is

After the batched epilogue the fixed cost is still **11.5 instructions per output**, which is 42% of
everything retired at `k = 64`; removing all of it is a **1.76x ceiling on the op**, and the measured
gap against onnxruntime at this shape is 1.96x. **The two numbers agreeing is the finding.** MLAS does
not pay a per-output reduction at all: a dot-product kernel whose vector lanes are `k` must reduce
horizontally once per output, and an outer-product kernel whose lanes are `m` or `n` never does — it
accumulates into C directly.

So the remaining gap is **the kernel's formulation, not its schedule**, and closing it means packing an
operand so the contracted axis is not the vector axis. That is a new kernel plus a packing step rather
than a patch to this one, it is what ADR-014 exists to make a deliberate decision about, and **it
should be scoped as its own item rather than started from here.** What is now settled and should not be
re-derived: the FP ports are not the limit, the epilogue is 54% of the work at this shape, `BM` is a
cache knob **at one thread** (at four it is a false-sharing knob worth 2.75x -- P4.22 and
[Retro-020](../retros/retro-020-a-knob-measured-at-one-thread.md); this paragraph is the entry that
lesson is about), and 6.7 of the 18 instructions are available for 1.085x if anyone wants them.

**And the conclusion of the first sentence did not survive its own experiment:** packing an operand so
the contracted axis is not the vector axis WAS built and measured, as P4.21, and it is 1.38x on this
machine against a 1.52x roofline ceiling. See §5.

**Not a gap, but worth knowing:** whisper pads every clip to 30 s, so an 11-second file pays the full
1500-frame encoder on **both** engines. That is not where loom loses, and shortening it is a
model-semantics change (the positional embedding is fixed at 1500), so it is not part of this item.

### Operating notes: benchmarking

**Machines.** The Pi answers to **`ssh pi@rpi4`** (verified 2026-08-29; it resolves over IPv6 — an
earlier note here said the name does not resolve and gave `192.168.1.35`, but **the IPv4 address
moves**, so prefer the name and fall back to `ip -4 addr` on the console). The workstation is
**`192.168.1.100`** (Intel Core Ultra 9 285K, 24 cores, 40 MB L2 / 36 MB L3, Debian, gcc 14.2); it has
no `cmake` on the default PATH (there is a `buildtools` micromamba env) and its `/home` runs at 99%.

`ssh pi@rpi4` — Raspberry Pi 4B rev 1.5, Cortex-A72, 4 cores @ 1.8 GHz, 1 MB shared L2,
32 KB L1D, LPDDR4, Debian aarch64, gcc 14.2 / clang 19. Repeatable to ~1% **when it is cool and nothing
else is on it**, and to about 9% when it is not. The dev box (Ryzen 3 3250U, AVX2, 2 cores, 4 MB L3) is
**thermally noisy** — pin with `taskset -c 0,2` and take medians of seven, or it will lie by 15%.

**Rules that were learned the hard way:**

* **Make both A/B arms the same binary**, switched at run time, and interleave them ABBA in both orders
  over two rounds.
* **`perf` on the workstation, and the four things that make it usable.** It is installed there
  (6.12.105, 2026-08-25) and it is a **hybrid PMU**: events split across `cpu_core` and `cpu_atom`, so
  **pin to a P-core and prefix the events** — `taskset -c 0 perf stat -e cpu_core/cycles/,cpu_core/instructions/`
  — or they are counted on both and the shares mean nothing. Basic counters (cycles, instructions,
  branches, branch-misses) work per-process without privileges at `perf_event_paranoid = 2`; the
  `topdown-*` group does **not** ("Invalid event in per-thread mode"), and needs system-wide collection
  — ask before changing a sysctl or running sudo on someone's machine. There are no named
  microarchitectural events in this build (`perf list` has zero matches for `fp_arith*` or `uops_*`);
  raw encodings would have to come from Intel's perfmon JSON **for Lion Cove**, not from an older core.
* **Count instructions before timing anything.** Instruction counts are deterministic to ~0.4% where
  this project's timings are noisy to 15%, and P4.18 item C was settled entirely on them: a `k` sweep
  fitted `instructions/output = 0.2355k + 18.0` and named the mechanism, where three years of timing
  had produced four falsified guesses. **Point the counter at ONE shape** — `scripts/bench15.cpp` runs
  a single GEMM in a loop for exactly this reason, because `perf` counts a process and every other
  bench here runs a table of shapes that behave differently.
* **Pin any stochastic sampler before quoting a ratio.** VITS's duration predictor is stochastic and
  the reference host does not seed it, so each run synthesises a different number of samples — see
  [Retro-010](../retros/retro-010-an-unpinned-competitor-baseline.md).
* **Normalise to output samples**, and scale a competitor's rows if its pinned length differs.
* **Rank by the machine's peak as well as by the competitor.** A row where both implementations are
  equally bad sorts to the bottom of a ratio table and can still hold the most time.
* **When a ratio is inexplicable by the kernel, count the nodes before profiling them.**
  `scripts/conv_census.py` needs neither a run nor the target hardware.
* **Probe the floor before optimising the middle.** Before writing a faster kernel, run the arm with
  the expensive part *deleted* — a soft_max with no `exp` in it, a GEMM that only streams. It ships
  nothing and it takes ten minutes, and it separates "this kernel is slow" from "this kernel is a
  memcpy with arithmetic attached". P4.18's soft_max was the second, and the probe said so before any
  of it was written (Retro-012).
* **A `$LOOM_PROFILE` bucket does not know which graph it is in.** The key is `(op, ne0, ne1)` and
  nothing else, so "`ne1 = 1500` is the encoder, `ne1 = 1` is a decode step" classifies only the
  buckets whose `ne1` is one of those two. P4.18's largest layout bucket had `ne1 = 64`, was assigned
  to the encoder by eye, and was 93% decoder — which inverted the item built on it AND the claim that
  the decode loop needed nothing. Patching `record()` to print `node->name` settles it in one run and
  costs nothing to keep out of the report.
* **When a kernel's rate depends on a shape, sweep the shape before blaming the kernel.** P4.18's
  attention matmuls looked like a 2x micro-kernel deficit and were a `k % 8` cliff: 44.5 GFLOP/s at
  k=1496, 20.8 at k=1500, 47.0 at k=1504. A ratio against a competitor cannot see that, because the
  competitor does not have the cliff. Three lines of sweep did.
* **Split the competitor's profile by shape, not just by op.** onnxruntime's 96 `MatMul` against
  loom's 96 gave 1.93x, which read as "the GEMM is 2x slow everywhere" and sent P4.18's largest item
  at the dense projections. Split by input shape it is 1.10x on the projections and 2.2x on the two
  attention matmuls — a different item with a different fix. The split is one `input_type_shape` key
  in the profile JSON onnxruntime already writes.
* **An ISA-conditional bug is invisible from the other ISA, and benches inherit that.**
  `scripts/bench6.cpp` is aarch64-only, so every GEMM measurement in P4.15 ran where tinyBLAS's
  `KN` is 4 and whisper's 1500-frame contraction divides it exactly. The same matmul on x86 never
  entered the file. When a bench only builds on one architecture, that is a coverage hole with a
  measurement attached, not a portability chore.
* **A floor arm must be built the same way as the arm it bounds.** Deleting the expensive part is the
  right idea; writing a *simpler program* that does the same job is not the same thing. Retro-012
  closed `SOFT_MAX` on two scalar-C probes measured against a hand-vectorised candidate — and on the
  dev box the "no exp" one comes out SLOWER than the arm it bounds, which is impossible for a floor.
  Use a template parameter or a `#if` on the real arm. And for "it is memory bound", the floor is
  `memcpy` of the actual bytes at the actual working-set size — three lines, never run, and it turned
  out to be 3.9x below where the kernel sat.
* **On a laptop, interleaving is not enough — pair the arms and take the RATIO.** The same binary
  gave 27.9 and 12.6 GFLOP/s for the same shape twenty minutes apart. Min-over-interleaved-rounds
  still compares two independent minima drawn from different parts of a thermal excursion; a per-round
  ratio of two arms run back to back cancels the drift. Keep a fixed reference shape in every round as
  a clock witness and **publish its spread with the result** — on the dev box it is 1.4-2.5x, so
  nothing under ~1.2x is measurable there, and saying so beats a number that will not reproduce. Two
  P4.18 verdicts were written from inside that noise and both were wrong (Retro-012).
* **Know the machine's actual peak before calling a kernel efficient or slow.** Zen+ has 128-bit FPU
  datapaths, so a 256-bit FMA retires one per cycle and single-core F32 peak is ~54 GFLOP/s, not the
  ~112 a lane count suggests. That one number turned "the dense GEMM is 44 GFLOP/s and MLAS does
  better" into "the dense GEMM is at 88% of the machine and there is nothing to win".
* **For a float32 kernel, the error bound is enumerable — do not sample it.** The domain is 2^32
  values; a sweep against a double-precision reference takes about 30 s on 24 cores and gives the
  actual worst case rather than a grid maximum. It is also the only thing that finds the tails: P4.18's
  GELU approximation was 2.6e-07 relative to scale over a grid and **2e31 absolute at `x = -FLT_MAX`**,
  and nothing but enumeration was going to surface that. Sweep once per ISA branch, since each one
  rounds differently — the AVX2 path came out slightly *more* accurate than SSE2 because of FMA.

**Before opening a performance item, read
[Retro-012: Optimizations That Were Measured Out](../retros/retro-012-optimizations-that-were-measured-out.md).**

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-014](../adrs/adr-014-patch-ggml-rather-than-write-kernels.md), [ADR-017](../adrs/adr-017-no-k-quants.md) |
| Retros | [Retro-010](../retros/retro-010-an-unpinned-competitor-baseline.md), [Retro-011](../retros/retro-011-chasing-the-gemm-and-convolution-gap.md), [Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md), [Retro-014](../retros/retro-014-the-text-encoder-was-in-the-graph-twice.md), [Retro-017](../retros/retro-017-libgomp-slept-at-every-graph-node.md), [Retro-018](../retros/retro-018-a-table-of-ratios-nobody-could-re-derive.md), [Retro-019](../retros/retro-019-a-patch-measured-on-one-isa.md), [Retro-020](../retros/retro-020-a-knob-measured-at-one-thread.md) |
| Active tasks | [Backlog → Performance](../backlog/active-index.md#engine--performance) |

## 4. The Record

### P4.14 — `$LOOM_PROFILE`: per-node timing of the graph the engine actually runs — DONE (2026-08-20)


**Why it exists: three plausible answers were wrong.** Chasing why VITS
(`vits-piper-en-gb-miro`) synthesises ~2.2x slower than the same checkpoint under onnxruntime on a
Raspberry Pi 4, three explanations were argued for from the code and each was measured at a few percent
or less — ggml not fusing conv+bias+activation (the whole unfused elementwise + activation chain is
**6.5%**), the C++<->Lua array boundary (**18.7 ms**), and `GGML_LLAMAFILE` being off in the shipped
wheel (it is off, and building ggml both ways on the Pi measured **no difference at all**: 20.5 vs
20.3 GFLOP/s — tinyBLAS's ARM F32 path does run, it just isn't faster for these shapes — though P4.15
later found out WHY not, and fixing it made this the biggest single win in the thread, so do not quote
this parenthesis without that one). The answer was
`MUL_MAT` at **70.4%**. Rebuilding the engine with a hand-rolled hook to find that out is a day; this
makes it five minutes, and it works against a SHIPPED WHEEL, which is where the question actually gets
asked.

**What it is.** `include/loom/core/profile.h` + `src/core/profile.cpp`. `$LOOM_PROFILE=1` (or a path)
makes `GraphBuilder::compute` walk the already-built, already-allocated graph one node at a time,
bucketing time by `(op, ne0, ne1)`; the report is per-bucket then rolled up by op. Two routes, because
the builder has two: the CPU-only path uses `ggml_graph_view` directly, and the hybrid path installs
ggml's own `ggml_backend_sched_set_eval_callback` (which already runs a split node-by-node when a
callback is present) and clears it afterwards.

**The one trap, documented at the top of the header because a user who misses it gets a misleading
answer rather than a noisy one.** One `ggml_backend_graph_compute` per node is one threadpool
synchronisation per node. Measured on the Pi over ~2 990 node executions:

| | sum of node times | un-profiled run | overhead |
|---|---|---|---|
| `threads=1` | 5.939 s | 5.982 s | **0.7%** — exact |
| `threads=4` | 5.53 s | 2.35 s | **~1.4 ms floor per node** |

At four threads that floor is bigger than most nodes' real work, so it lands on whichever op has the
most NODES rather than the most work. **Profile with one thread.** `Totals::floor_seconds` reports the
floor the run observed and the report prints a floor-corrected column, but a corrected four-thread
number is an estimate where a one-thread number is a measurement.

**What it cannot see:** anything outside a graph — graph *building*, the driver script's host-side
loops, marshalling. For VITS those came to 165 ms of a 2.4 s call, so "the profile does not add up to
the wall clock" is expected.

**Notes on the implementation, both of which were bugs first.** The table is a deliberately leaked
`new`'d `std::map`: an earlier draft dumped from a static destructor, which on a shared-library build
ran after the table's own and threw `std::bad_alloc` out of `__cxa_finalize` — a crash at exit in the
tool whose job is to print at exit. And `profile.cpp` is the one TU that includes ggml's private
`ggml-impl.h` (for `ggml_graph_view`, which returns `ggml_cgraph` by value); the CMakeLists adds
`${ggml_SOURCE_DIR}/src` PRIVATE to that target alone. Affordable only because `cmake/GgmlPin.cmake`
pins ggml exactly, so the header cannot shift without a deliberate bump turning this into a compile
error. The alternative — routing the CPU path through `ggml_backend_sched` to borrow its callback —
would have profiled a *different execution path* from the one production uses.

**Tests:** `tests/ci/test_profile.cpp`, registered TWICE against one binary differing only in
`ENVIRONMENT`, because `profile::enabled()` caches on first call and one process can only observe one
route. It pins (a) node-by-node output is **bit-identical** to whole-graph — the claim that matters,
since a profiler that silently corrupts what it observes is worse than none; (b) buckets account for
every node executed; (c) the branch routes in both directions. **Both directions were verified to go
red** by sabotage (branch forced off; walk made to skip the last node) — and the first attempt at this
test could NOT fail, because `LOOM_CHECK` only counts failures and the test ended in `return 0;`
instead of `LOOM_TEST_REPORT_AND_RETURN()`. Exactly the trap CLAUDE.md warns about, caught only by
running the sabotage.

### P4.19 — `$LOOM_PROFILE_NODES`: which GRAPH a bucket is in — DONE (2026-08-25)

**A `(op, ne0, ne1)` bucket cannot say where its time came from, and that is not a hypothetical
limitation.** P4.18's largest layout cost — `CONT 1500 x 64`, 471 ms — was assigned to whisper's
encoder on the reasoning that `ne1 = 1500` is an encoder op and `ne1 = 1` is a decode step. That
bucket's `ne1` is 64, so it is neither, and it was assigned by eye. It is 93% the decode loop's, which
inverted the item built on it *and* the claim that the decoder needed nothing. The same table's `NORM`
coincidence **was** checked; this one was not, and nothing in the report marked the difference.

**`$LOOM_PROFILE_NODES=1` adds a second table keyed on `(op, node name, all four ne)`.** ggml grows a
node's name as it is transformed (`xv_0 (reshaped) (permuted) (permuted) (cont)`), so the name carries
the graph the node came from; the four `ne` separate nodes that agree on the leading two. Same run,
same `$LOOM_PROFILE` destination, printed after the summary so a reader who did not ask for it still
finds the rollup at the top. Off by default: it is a per-node map on the recording path and a report
long enough to bury what it sits under.

It settles the bucket it was written for in one run — whisper-small, `jfk.wav`, one thread, Ryzen 3
3250U, `CONT` at `1500,64,12,1`:

| ms | calls | name |
|---|---|---|
| 202.3 | 27 | `xv_3 (reshaped) (permuted) (permuted) (cont)` |
| ... | ... | *twelve of these, one per decoder layer, 2276 ms together* |
| 88.7 | 12 | ` (reshaped) (permuted) (permuted) (cont)` — the encoder's own, unnamed |

**96.3% of that bucket is the decode loop**, and the arithmetic the epic previously had to do by hand
(324 = 12 encoder nodes + 12 layers x 26 decoder executions) is now a line of the report. The count
here is 336 = 12 + 12 x 27, for the 27-step decode this run produced.

**Tests:** a THIRD registration of `tests/ci/test_profile.cpp`, for the same caching reason as the
other two. It asserts the finer table accounts for *exactly* the node executions the shape table does —
if the two disagree, some execution landed in a shape bucket without landing in a node bucket, and an
attribution built on the report would be reading a partial graph — and that the table is empty when
the variable is unset. **Verified red** (ADR-015) by making `record()` skip `GGML_OP_ADD`: the
accounting check fails, the other two registrations stay green.

### What the profiler then measured, which P4.13 should read

VITS `flow_vocoder` + text encoder + duration predictor, y_length=287, **1 thread, total 5.94 s**:

| op | calls | ms | % |
|---|---|---|---|
| MUL_MAT | 213 | 4181 | **70.4%** |
| IM2COL | 165 | 767 | **12.9%** |
| CONV_TRANSPOSE_1D | 3 | 528 | **8.9%** |
| ADD | 376 | 256 | 4.3% |
| CONT | 311 | 66 | 1.1% |
| everything else | — | ~145 | 2.4% |

Conv path **92%**. At 4 threads the model runs 2.35 s (2.53x), and the big convs individually scale
only **1.4–2.1x**.

Three things follow, all measured rather than reasoned:

* **`CONV_TRANSPOSE_1D` as mul_mat + reshape is 1.11-2.05x faster** than ggml's native op, which is a
  naive scatter loop with a single-threaded memset+permute prologue. Re-measured on an IDLE box after
  the contamination below, where it got stronger rather than weaker (2.05x on the first upsample vs
  1.30x under load), so unlike the fused-conv result this one is real. The three upsamples cost 189 ms
  of a 2.35 s synthesis, so it is worth ~60-90 ms, ~7% of the loom-vs-onnxruntime gap. The benchmark
  skipped the final interleave, so the real win is smaller. **Open** -- and it is the ONE conv-side
  change in this whole investigation that survived measurement.
* **Re-lowering `CONV_1D` as `kw` shifted mul_mats over views of the activation — to avoid im2col's
  ~7x memory blowup — is WORSE, 0.58–0.77x.** The per-tap `cont(transpose(view))` re-materialises the
  activation anyway. Do not re-propose this; it was measured on the real shapes.
* **ggml's own fused conv does NOT help here, measured twice.** `GGML_OP_CONV_2D` /
  `ggml_conv_2d_direct` does im2col into a per-batch working buffer and GEMMs each batch -- the
  blocked/implicit-GEMM structure onnxruntime's MLAS uses, already in ggml, already implemented for the
  CPU. There is no `ggml_conv_1d_direct`, but a 1-D conv IS a 2-D conv with `KH=1, H=1` and every step
  to get there is a reshape, so it is a few lines in `primitives_conv.cpp` to try. Output is
  **bit-identical** (max abs diff 0.0). Over all 60 `flow_vocoder` convs at 4 threads it is **0.98x** --
  i.e. nothing, marginally worse -- and per-shape it ranges 0.88-1.09x with no pattern worth exploiting.
  **Closed, do not re-propose.**

  **Read the retraction below before quoting any earlier number for this.** A first pass measured
  1.14-1.30x per shape and 1.22x overall and was written up here as a real win. It was an artefact: an
  orphaned `prof_main` (a `timeout`-killed ssh whose REMOTE process kept running) had been pegging one
  of the Pi's four cores for 67 minutes, so every "4-thread" benchmark in that window actually had three
  cores. ggml's `mul_mat` partitions work assuming `n_threads` equal workers and degrades badly when one
  is starved; `conv_2d_direct`'s per-batch structure tolerates it. Remove the contention and the
  advantage is gone. The tell was visible and was ignored for one round: the contaminated run put the
  conv total at **2.589 s inside a 2.35 s synthesis**, i.e. 110% of a whole that it is a part of. On an
  idle box the same measurement gives 1.550 s = 66%, which is possible. **A component that does not fit
  inside its own total is not a noisy measurement, it is a wrong one -- stop and find the cause.**

**Two benchmarking traps, both of which produced a wrong answer that survived a full write-up:**

1. **Never read a ggml tensor you have not written.** Untouched anonymous pages all map to the shared
   zero page, so the entire read set sits in L1 — that inflated an isolated conv benchmark by **2.1x**
   (103 ms measured against a real 219 ms).
2. **Check the box is idle, and check that the parts fit inside the whole.** See the orphaned-process
   retraction above. `uptime` before a run costs nothing; a `timeout`-killed ssh does NOT kill the
   remote process, so use `pkill` on the far side after any aborted remote benchmark.


### P4.15 — the F32 GEMM micro-kernel, which is 71% of that gap — DONE (2026-08-21)

**What it turned out to be: not a missing kernel, a spilled one.** Nothing was written from scratch,
nothing in this repo computes a GEMM, and `scripts/bench6.cpp`'s prototype shipped nothing -- it was
the measuring stick. Two build-side changes:

* **`GGML_LLAMAFILE=ON`** (`cmake/Dependencies.cmake`, behind a `LOOM_TINYBLAS` option so the A/B
  stays runnable). A standalone ggml defaults tinyBLAS OFF -- it is llama.cpp that turns it on -- so
  every wheel this project has ever shipped ran ggml's one-output-element-per-call F32 kernel. On
  x86-64 that flag alone is **1.97x** at the eleven vocoder shapes (27.6 -> 54.4 GFLOP/s, AVX2 Ryzen 3
  3250U, median of seven runs pinned to the two physical cores). On the Pi it is worth **nothing**
  (15.0 -> 15.6), which is exactly why it was written off above. It costs 111 KB of `libggml-cpu`
  (979 -> 1090 KB, x86-64 release build).

  **That 1.97x was first written up here as 2.83x**, from one unpinned run of each build on a
  thermally noisy laptop -- 20.6 and 58.3, which are the low and high ends of this box's spread. Seven
  pinned runs put the pair at 27.6 and 54.4. The x86 numbers in this entry are all medians of pinned
  runs for that reason; the Pi's are not pinned because that box is idle, single-socket and repeatable
  to about 1%.
* **`cmake/patches/ggml-0001-tinyblas-neon-gcc-tile.patch`**, which routes NEON-on-GCC to tinyBLAS's
  *16-register* schedule instead of its 32-register one, because GCC spills the 24-accumulator tile
  the 32-register one asks for. **15.6 -> 22.0 GFLOP/s, 1.41x**, from a one-line change to a `#if`.
* **`cmake/patches/ggml-0002-tinyblas-aarch64-address-hoist.patch`**, which writes the operand
  addresses in a form GCC will strength-reduce, because it will not do it through the class members:
  35 instructions per k-iteration where 21 do the same work. **22.0 -> 25.1 GFLOP/s**, and that is
  PAST the hand-written 4x4 kernel this item was scoped around (24.3 in the same process). Found by
  asking why the first patch stopped at 92% of it -- see the section below, and note that the answer
  was worth as much again as the tile.

**The open question left at the bottom of this item -- tinyBLAS accumulates exactly like the fast
prototype yet runs 1.6x slower -- turned out to BE the item. Four measurements answered it.**

1. **At ONE thread the gap gets WIDER**, 1.78x against 1.55x at four (`./bench6-on 1`). That rules out
   ggml's threadpool, its barriers and its chunk scheduling in one command, before reading any of it.
2. **Lifting tinyBLAS's `gemm_bloc` out of ggml and into the prototype's own OpenMP driver reproduces
   the loss with ggml gone** (`scripts/bench7.cpp`). Same buffers, same parallelisation, same file --
   only the tile differs, so the tile IS the difference. All eleven shapes, 4 threads:

   | tile (RM x RN) | live accumulators | q-registers spilled in the block | GFLOP/s |
   |---|---|---|---|
   | 4x3 | 12 | 0 | 24.2 |
   | **4x4** | 16 | 0 | **24.5** |
   | 4x5 | 20 | 8 | 18.4 |
   | **4x6 -- what tinyBLAS picks on ARM** | 24 | 10 | **16.3** |
   | 8x4 | 32 | 32 | 16.3 |
   | 4x8 | 32 | 29 | 14.6 |

   The array-of-vectors form is NOT the cause: tinyBLAS's own `D Cv[RN][RM]` code at 4x4 measures
   24.4, indistinguishable from the prototype's 16 named variables at 24.5. Only the size matters.
3. **The disassembly says it outright.** In the 4x6 block gcc 14.2 (aarch64, `-O3`) emits twelve
   `stp q` per k-iteration -- all 24 accumulators written to the stack on **every** step, against the
   24 `fmla` they exist to accumulate. The A72 has one store pipe, so those stores cost about what the
   arithmetic does. Writing the same tile with 24 named variables instead of an array only reduces it
   (5 spill stores, 22.5 GFLOP/s); it does not remove it.
4. **So the paper ratio was measuring the wrong constraint.** 4x6 issues 10 loads per 24 FMAs (0.42)
   against 4x4's 8 per 16 (0.50) and should therefore win. It loses because 24 accumulators do not fit
   *this compiler's allocator* -- not because they do not fit the register file, where 29 of 32 live
   values is comfortable. A load/FMA argument is only valid downstream of "does the tile stay in
   registers", which is a fact about the compiler and has to be read out of the object file.

**Why the patch reuses the existing 16-register branch instead of adding a 4x4 one.** Through the
dispatcher the two measure 22.2 (RN=3) and 22.3 (RN=4) -- a wash -- and RN=3 is the schedule every
AVX2 x86 build already takes, so the diff is one `#if` line and no new code path. It is scoped to
`defined(__GNUC__) && !defined(__clang__)`: clang's aarch64 allocator was never measured (no clang on
the Pi, the dev box or the workstation that day), and Apple silicon is both clang-built and far wider
than an A72, so a change that helps here could hurt there.

**End-to-end, the same VITS utterance, Pi 4, 4 threads, idle box, steady state over five reps:**

| build | synthesis | vs onnxruntime's 1.024 s |
|---|---|---|
| as shipped (no tinyBLAS) | 2.318 s | 2.26x |
| tinyBLAS on, unpatched | 2.262 s | 2.21x |
| tinyBLAS on + tile patch (0001) | 2.021 s | 1.97x |
| **+ address hoist (0002)** | **1.94 s** | **1.89x** |

The 1-thread profile attributes it where predicted: `MUL_MAT` **4181 -> 3210 -> 2782 ms** over the
three steps, whole run 5.94 -> 4.96 -> 4.55 s, with `IM2COL` and everything else unmoved.

Steady-state reps land at 1.93-1.95 s; the baseline row was measured the same way and lands at
2.318-2.319. Reproduced from a clean checkout on the Pi -- `rsync`, `cmake -B build
-DCMAKE_BUILD_TYPE=Release`, one target -- twice, which is what says the CMake plumbing (patch
application in order, `LOOM_TINYBLAS` default) delivers this and not just a hand-edited `_deps` tree.

**And a trap that cost half an hour, worth knowing before any measurement here:** `cmake -B build`
gives you **RelWithDebInfo**, this repo's default (top-level `CMakeLists.txt`), i.e. `-O2`. The same
tree, same sources, same patch, Release against RelWithDebInfo: **2.019 s against 2.810 s, 1.39x**.
That is bigger than everything P4.15 changed. A benchmark run out of a default build dir is measuring
`-O2` and is not comparable with any number in P4.14 or P4.15, all of which are Release -- which is
also what the wheels ship (`loom-py/CMakeLists.txt` forces it).

**It is 0.37 s against the ~0.47 s predicted above, and what is left of the difference has a name.**
The GEMM itself is done -- ggml's kernel now runs slightly faster than the hand-written one this item
was scoped around, so there is nothing left to win by writing a kernel. What remains is the work
tinyBLAS never sees: **it declines any `mul_mat` whose `m` is not a multiple of 4**, and VITS's
heaviest MUL_MAT bucket is exactly that -- `[287, 384]`, 28 calls, **825 ms, 18.1% of the 1-thread
run**, m = 287 -- so that work still runs on ggml's one-element-at-a-time kernel. A tail path in
tinyBLAS, or padding the duration-predictor shapes on the export side, is now the biggest single lever
left in this thread.

**What is pinned, and how each part fails loudly.**

* `tests/ci/test_tinyblas_gemm.cpp` asserts the CPU backend reports the `LLAMAFILE` feature (through
  the registry's `ggml_backend_get_features`, not `ggml_cpu_has_llamafile()`, which a `GGML_BACKEND_DL`
  build does not link -- the trap `tests/support/cpu_backend.h` documents) and checks `mul_mat` against
  a double-precision reference at shapes on **both** sides of every condition tinyBLAS selects on
  (`n >= 4`, `k % KN == 0`, `m % 16/8/4`), because a tiling change's failure mode is a wrong tail
  block. Both directions were sabotage-verified: a 1.0001x factor on the result makes it red, and a
  `-DLOOM_TINYBLAS=OFF` build correctly reports the feature absent.
* `cmake/GgmlPatches.cmake` re-checks the patches on **every** configure, not on populate, so an
  existing build tree cannot end up silently unpatched; a ggml bump that makes one stop applying is a
  configure FATAL_ERROR naming the choice (delete it if upstream took it, rebase and re-measure if
  not). They are applied in filename order and touch disjoint regions of one file.
* The numerics change is real and inaudible, and was checked as a waveform rather than argued from the
  flag: the same VITS utterance synthesised by an otherwise identical `-DLOOM_TINYBLAS=OFF` build
  differs by **max 3.7e-6 against a 0.17 peak** (2e-5 relative), rel-RMS 7.1e-6, cosine
  **0.99999999998** -- computed in float64, because a float32 dot product over 73472 samples prints
  1.0000000000 for this and would have hidden a difference ten times larger. No existing tolerance gate is anywhere near that -- the ci suite (66/66) and every
  gate with a fixture present on this box (8 of 82, including the kokoro / styletts2 / matcha lua
  drivers, i.e. the conv-bearing ones) stay green. What P4.15 predicted might need re-baselining does
  not, because tinyBLAS is a different summation ORDER and not a different algorithm. The tensor-oracle
  gates whose reference dirs are absent here were not run; if a full-fixture sweep is done later, this
  is the change to attribute a ~1e-5 movement to. Patch 0002 moves nothing at all: the waveform after
  it is **bit-identical** to the waveform before it, as an addressing change should be.

**Still to do upstream.** Both patches are diffs against `v0.19.0` and belong in ggml-org/ggml. The
clang question that was open here is answered -- clang 19/aarch64 holds the 4x6 tile and wants neither
change, which is exactly what the two guards say -- so what a PR still needs is breadth this bench
cannot supply: one wide ARM core (Neoverse V2, an M-series Mac) to confirm that the smaller tile is
right for GCC there too and not just on an A72, and one more x86 part to confirm 0002's guard. Neither
blocks carrying them locally, and both are the kind of thing an upstream reviewer will ask for.


### Where the loom-vs-onnxruntime gap is NOW (idle Pi 4, 4 threads)

P4.14's version of this table put convolution at 71% of the gap. That row came from an isolated
benchmark of the 60 `flow_vocoder` convs; this one comes from `$LOOM_PROFILE` over the whole graph in
both builds, so the before and after are the same measurement rather than two. `-DLOOM_TINYBLAS=OFF`
reproduces the shipped-before state exactly, which is what the option is for. Node times are scaled to
each build's own un-profiled wall clock (profiling costs ~6%, and with prof_main's persistent
threadpool the per-node floor is 0.001 ms rather than P4.14's 1.4 ms -- create/join per node was that
number, not dispatch).

| | before (2.33 s) | 0001+0002 (1.94 s) | +0003 (1.79 s) | +0004 (1.62 s) | **+0005 (1.58 s)** | onnxruntime (1.02 s) | delta | share of gap |
|---|---|---|---|---|---|---|---|---|
| convolution: `MUL_MAT` | 1.30 s | 0.90 s | 0.78 s | — | — | — | | |
| convolution: `IM2COL` | 0.53 s | 0.53 s | 0.51 s | — | — | — | | |
| convolution: `CONV_2D` (+ bias from 0005) | — | — | — | 1.06 s | **1.10 s** | — | | |
| **convolution, total** | **1.81 s** | **1.43 s** | **1.27 s** | **1.07 s** | **1.10 s** | 0.605 s (`Conv`+`FusedConv`) | 0.50 s | **84%** |
| transposed conv | 0.18 s | 0.18 s | 0.18 s | 0.19 s | 0.19 s | 0.162 s (`ConvTranspose`) | 0.03 s | 5% |
| everything else | 0.34 s | 0.33 s | 0.33 s | 0.36 s | 0.29 s | 0.257 s | 0.03 s | 6% |

**loom-vs-onnxruntime is 2.28x -> 1.86x -> 1.75x -> 1.58x -> 1.54x -> 1.41x with the direct kernel,
its phase-major sibling and contiguous block partitioning (ggml-0006), and the gap itself 1.31 -> 0.92
-> 0.76 -> 0.59 -> 0.55 -> 0.42 s: 68% of it closed.**
The x86 box, which had been on a different lowering entirely and had no end-to-end number until this
one, went 1.503 -> 1.169 s on the same utterance.

**The 0005 column's split is arithmetic, not measurement, and this is the one place in this entry where
that is true.** Its total (1.576 s) is measured, against 1.605 s for the identical build with
`GGML_CPU_DISABLE_FUSION=1`. The rows are the +0004 profile with the bias ADD moved out of "everything
else" and into the convolution, less the 0.03 s the fusion actually saves -- because `$LOOM_PROFILE`
CANNOT observe a fused run: it submits one node per graph, and a one-node graph has nothing to fuse
with. If a future item needs the fused split for real, that is the thing to fix first. The last column is one op now, because the convolutions no longer lower to
im2col + mul_mat on this architecture -- see the im2col section below.
Only `MUL_MAT` moved (1.375 -> 0.990 s of profiled node time, 1.39x); `IM2COL`, `CONV_TRANSPOSE_1D`
and everything else are unchanged to within a millisecond, which is the check that the patches did what
they claim and nothing else. onnxruntime re-measured the same day at 1.01-1.08 s raw -- its duration
predictor is unseeded so each run synthesises a different length, and normalising to our 73472 samples
puts it at ~1.04 s, against the 1.024 s the per-op shares above were apportioned over.

**What the new balance says, and it is not what it said before.**

* **The GEMM was no longer the thing to fix -- the work it never saw was. FIXED, patch 0003.**
  `MUL_MAT [287, 384]` was **324 ms before and 324 ms after** the first two patches, unchanged to the
  millisecond, because m = 287 is not a multiple of 4 and tinyBLAS declined it outright -- every tile
  in that file is 4 rows tall, so one leftover row sent all 287 to ggml's generic kernel.
  `ggml-0003-tinyblas-row-tail.patch` runs the aligned prefix through the tiles and finishes the <= 3
  remaining rows in a 1x1-blocked loop, which is what those rows would have got anyway. That bucket is
  **324 -> 190 ms**, all four m = 287 buckets **354 -> 208 ms**, and the synthesis **1.94 -> 1.79 s**.
  It cannot regress anything: the tail rows get exactly the kernel they had, and every other shape is
  unaffected. Architecture-neutral, so x86 gets it too.
* **`IM2COL` was 40% of loom's convolution** (0.51 s of 1.27 s) where onnxruntime has no equivalent
  line at all -- MLAS packs inside `Conv`. P4.14 measured `ggml_conv_2d_direct` at 0.98x and CLOSED it;
  that verdict was reached when the GEMM was 1.7x slower and im2col was proportionally a smaller
  share. **Re-opened and repaired** -- see the section above: convolution 1.27 -> 1.07 s.
* **Transposed conv is at parity** (0.18 vs 0.162 s) and worth 2%. The mul_mat + reshape lowering from
  P4.14's follow-up list is ~60-90 ms, i.e. still real but no longer near the top.


### The hybrid: a direct kernel behind a cache-size heuristic (ggml-0006)

The prototype said a direct convolution wins where the activation is long and the weights are small and
loses badly where they are not. `cmake/patches/ggml-0006-conv1d-direct.patch` is that kernel inside
ggml's `CONV_2D`, with the heuristic that decides. **loom now lowers `CONV_1D` to `CONV_2D` on every
architecture** — the `#if defined(__aarch64__)` is gone, because the direct kernel is what x86 was
missing.

| | before | after | |
|---|---|---|---|
| Pi 4, 4 threads | 1.576 s | **1.440 s** | 1.09x |
| Ryzen 3 3250U, 2 threads pinned | 1.503 s | **1.169 s** | **1.29x** |

**Three conditions, and the third one is the whole lesson.**

1. **Shape and type**: F32, `KH = 1`, one image, stride 1, contiguous, `OC % 4 == 0`. Anything else
   takes the batched im2col, which is also the fallback when the scratch does not fit.
2. **Weights must fit cache** — they are re-read once per position block, and at 1.5 MB against a
   287-position activation that is the entire cost. The budget is `L3/2` where the machine reports it,
   else `L2`, else 512 KB: **2 MB on the x86 box and 512 KB on the Pi**, which is exactly where the two
   machines' measurements say the line is. One rule, two answers, no `#if`.
3. **There must be at least as many position blocks as output-channel tiles.** Without this the
   synthesis went **1.576 -> 1.703 s** — slower than not having the kernel at all. The eleven shapes
   the bench holds are not the shapes the model runs: it also has 77 convolutions of 100 positions and
   192 channels, every one of which passes the weight test and none of which should take this path.
   Adding the condition also improved x86 (1.258 -> 1.169 s), so it was never an ARM quirk.

**Why ARM gets 1.05x where the bench promised 1.6x: dilation.** The bench measured every shape at
dilation 1. The model's HiFi-GAN resblocks run the same shapes at **1, 2, 3, 6 and 12**, and a dilated
convolution's taps are twelve floats apart rather than adjacent. Re-measured at the model's own
dilations, the direct kernel's advantage on those shapes falls from ~1.6x to **1.2-1.4x**, which is
what the end-to-end number reflects. `scripts/bench10.cpp` now carries the real dilations; a bench that
does not is measuring a convolution the model does not have.

**Why x86 gains six times more than ARM.** It was starting from further back: `CONV_2D`'s batched
im2col measured 0.87x there against plain im2col + mul_mat, so x86 had been left on the mul_mat
lowering entirely (P4.15's earlier entry). The direct kernel is 3.5-4.7x on that machine's
long-activation convolutions, which is enough to carry the whole op past the lowering it replaces.

**Numerics.** This one is NOT bit-identical -- it is a different summation order, so the products are
grouped differently. Against a `-DLOOM_TINYBLAS=OFF` build the whole synthesis differs by max
**3.5e-6** on a 0.17 peak, rel-RMS 6.5e-6, cosine 0.99999999998 -- the same order as every other change
in this entry, and every gate passes on both lowerings.


A convolution with dilation d is d independent DENSE convolutions over the subsequences `p = j*d + r`.
It matters for prefetching: at dilation 12 with kw 7 the direct kernel reads, per channel, seven
separate 64-byte runs spread over 352 bytes, so a 128-channel convolution asks the prefetcher to track
896 streams. Phase-major, each channel is one contiguous run.

**The transform, vectorised, is 2x cheaper.** NEON de-interleaves by 2, 3 and 4 in one instruction
(`vld2q`/`vld3q`/`vld4q` and the `vst` mirrors), and 6 and 12 factor into two such passes: splitting by
`a` then `b` lands `p = j*d + r` in phase `r1 + a*r2`, because `p = a*(j*b + r2) + r1`. **Both** passes
have to be vectorised -- doing only the first, the obvious thing, leaves 6 and 12 exactly where they
started. On the largest shape: dilation 3 **28.4 -> 16.2 ms**, 6 **40.0 -> 26.3**, 12 **53.6 -> 27.8**.

**Then the comparison had to be made fair, and that halved the case.** The bench's direct variants still
had the scalar ragged tail that `ggml-0006` had just stopped using. With both on the overlapping tail,
medians of three:

| shape | direct | phase-major | |
|---|---|---|---|
| 128x128 kw7 L2296 **d12** | 35.0 ms | **23.0 ms** | **1.52x** |
| 128x128 kw7 L2296 d3 | 26.1 ms | 23.5 ms | 1.11x |
| 64x64 kw7 L18368 d12 | 52.0 ms | 49.8 ms | 1.04x |
| 64x64 kw7 L18368 d3 | 44.2 ms | 50.4 ms | 0.88x |
| 128x128 kw5 L2296 d6 | 15.7 ms | 17.0 ms | 0.92x |
| 32x32 kw5 L73472 d6 | 26.5 ms | 54.9 ms | 0.48x |

Before the tail was fixed on both sides, the same table showed phase-major winning 2.99x on a shape
where it actually ties (192x384 kw5 L287, 13.2 vs 13.0 ms): the artefact was the whole margin.

**Shipped, behind a window narrow enough to take only what it wins**: `kw >= 7`, `dilation >= 3`,
`IC * kw >= 768`, aarch64 only (without `vld3`/`vst3` the transform is the strided copy it exists to
avoid, and AVX2 has no 3-way de-interleave). The benefit grows with the number of streams (`kw`) and
the dilation that spreads them, and the transform is only affordable when there is enough arithmetic
per element to hide it (`IC * kw`). In this model that window holds **two** convolutions, and measured
in one binary with the path gated on and off: **1.487 -> 1.463 s**.

**One bug found while shipping it, of the kind that would not have shown up here.** The two-pass
transform gives each thread its own de-interleave scratch, and the size check reserved ONE. At these
shapes the buffers in use total 2.9 MB of a 16 MB work buffer, so four threads' worth of overrun still
landed inside the allocation and nothing failed -- the tests passed, the audio was right. It is fixed
(`nth` slices, sized for the longer of the two passes), and it is a reminder that a scratch check which
happens to be slack is not a scratch check.

## The direct convolution, prototyped: right in one regime, four times wrong in the other (SHIPPED, above)

With the GEMM finished, the im2col that feeds it is the largest single item left -- **396 ms of the
1070 ms** this model spends in convolution, phase-timed inside ggml's own op (GEMM 663 ms, gather 396,
staging 11). The obvious next move is the one MLAS makes: don't materialise anything. Hold a tile of
the OUTPUT in registers, sweep the input in place, one lane-broadcast weight per (channel, tap).
`scripts/bench10.cpp` is that kernel, at the same eleven shapes.

**First, what the gather is NOT.** Its inner copy is a scalar element-at-a-time loop with a bounds test
per element; making it branch-free and width-specialised (so the copy inlines instead of calling
`memcpy` ~20 M times a synthesis) is worth **~10%, measured twice**. The cost is not the loop, it is
the 137 M element-writes im2col performs because it writes every input element `kw` times.

**The result, best tile (4 output channels x 16 positions), padded copy included:**

| shape | ggml conv | direct | |
|---|---|---|---|
| 32x32 kw7 L73472 | 59.3 ms | **37.7 ms** | **1.57x** |
| 32x32 kw5 L73472 | 46.6 ms | 30.9 ms | 1.51x |
| 64x64 kw7 L18368 | 56.3 ms | 41.0 ms | 1.37x |
| 128x128 kw7 L2296 | 27.5 ms | 24.7 ms | 1.11x |
| 192x384 kw5 L287 | 10.3 ms | 40.8 ms | **0.25x** |
| 768x768 kw3 L100 | 24.8 ms | 167 ms | **0.15x** |

**Weighted by call count: 0.37x overall.** Take the faster of the two per shape and it is **1.15x**,
worth ~174 ms of this model's convolution time, ~11% of the synthesis.

**The split is not about the kernel and not close.** A direct convolution re-reads the WEIGHTS once per
position block; a GEMM blocks both operands. When the weights are 1.5 MB and the activation is 287
positions, re-reading them seventeen times is the entire cost and no register tuning touches it. When
the activation is 73472 long and the weights are 28 KB, the direct form wins by not materialising 66 MB
nobody needed. That is why MLAS is a library of kernels behind a heuristic rather than one kernel.

**Two things that looked like the answer and were not, each worth about 3x on its own** -- both are the
kind of mistake that would have made this look like a dead end:

* **The loop nesting.** With the position block INSIDE the output-channel loop -- the obvious way to
  write it -- every channel block re-streams the whole input from DRAM, 8 passes over 9.4 MB for the
  first shape. The tell was that the kernel measured the same time whatever `kw` was: it was not doing
  arithmetic, it was waiting for memory. Position block outermost: **0.12x -> 0.36x**.
* **The edges.** Testing "is this block interior" and sending whole blocks to a scalar path costs a
  third of a short convolution. One zero-padded copy of the input makes every block interior, and that
  copy is one pass against im2col's `kw`.

**What it would take to ship, and why it is not done here.** A hybrid: direct kernel when the weights
are small against cache and the activation is long, GEMM otherwise, with weight packing done once
(24 ms for this model, or moved to the exporter). That is a new kernel in ggml -- the thing this whole
item avoided writing -- plus a shape heuristic, plus an x86 counterpart, for ~11%. It is the right next
move for this thread, it is a bigger piece of work than any of the five patches, and the prototype
above is what a decision about it should be made from rather than from the idea.

**Not the same thing as P4.14's closed item.** That one ("lower CONV_1D as `kw` shifted mul_mats",
0.43-0.98x) kept `mul_mat` and re-materialised the activation per tap through `cont(transpose(view))`.
This writes the accumulation loop directly and materialises nothing. The closed verdict does not
transfer -- and neither does it vouch for this one, which is why it was measured.


**The single biggest lever this investigation found, and it is ~60 lines.**

`ggml_compute_forward_mul_mat`'s F32 path computes ONE output element per `ggml_vec_dot_f32` call --
1x1 register blocking. On plain NEON that inner loop (`GGML_F32_STEP` 16, `GGML_F32_ARR` 4) issues
**two 128-bit loads per 128-bit FMA**. A72 is load-issue limited well before FMA peak, so that ratio,
not cache or bandwidth, is what caps it. The 16x16 "block-tiling attempt" above it only helps locality;
it does not amortise anything into registers.

MLAS (**github.com/microsoft/MLAS**, now its own repo, **MIT** -- so license-compatible with this one)
does the opposite: `src/lib/aarch64/SgemmKernelNeon.S` holds a **4 rows x 16 columns** accumulator tile
in v16-v31 and its inner op is a broadcast-lane FMA, `fmla v.4s, v4.4s, vA.s[lane]` -- ~5 loads feed 16
FMAs, **0.31 loads/FMA against ggml's 2.0**. That requires B pre-packed into 16-column panels, which is
what `MlasSgemmCopyPackB` exists for.

**We do not need the packing, because loom's operand layout is friendlier than the one MLAS was
designed for.** A conv mul_mat here has BOTH operands K-contiguous (im2col `[K, M]`, kernel `[K, N]`),
i.e. `C = A * B^T` with both row-major over K. A 4x4 tile of dot-product accumulators then needs no
repacking at all: 8 loads feed 16 FMAs, 0.5 loads/FMA. Measured on the Pi at the eleven real
`flow_vocoder` GEMM shapes (`tests/`-external prototype, ~60 lines of NEON intrinsics):

| | GFLOP/s | vs ggml |
|---|---|---|
| ggml `mul_mat`, `GGML_LLAMAFILE=OFF` | 15.0 | — |
| ggml `mul_mat`, `GGML_LLAMAFILE=ON` | 15.4 | 1.03x |
| **4x4 blocked NEON prototype** | **24.4** | **1.62x** |
| onnxruntime/MLAS, measured in-model | 25.7 | 1.71x |
| A72 fp32 peak (1.8 GHz x 4 cores x 4 lanes x 2) | 57.6 | — |

**A ~60-line kernel reaches 95% of MLAS's rate.** Per shape the win is 1.35-2.13x, concentrated in the
long-activation convs (M = 18368-73472); the two small-M cases (M=288) only get 1.13-1.22x.

**This also re-confirms the `GGML_LLAMAFILE` finding after the zero-page bug invalidated the first
test.** The original ON/OFF comparison ran on unfilled buffers; redone with filled inputs it is 15.4 vs
15.0 GFLOP/s -- still nothing. Curiously tinyBLAS's output is bit-identical to the 4x4 prototype's
(max diff exactly 0.0, where ggml's own path differs by ~3e-7), so it is accumulating the same way and
losing the 1.6x somewhere else. Worth one look before writing a kernel from scratch. **That one look
was the whole item**: it spills the tile it accumulates into. See the top of this entry.

**What it is worth end-to-end.** MUL_MAT is ~84.5% of loom's 1.55 s conv time (from P4.14's 1-thread
profile, MUL_MAT:IM2COL = 4181:767), so 1.56x on it saves ~0.47 s: **2.35 s -> ~1.88 s**, and
loom-vs-onnxruntime **2.30x -> 1.84x**. With the CONV_TRANSPOSE_1D item above, ~1.76x.

**Where the change has to live, in preference order.** Not in loom: loom calls `ggml_mul_mat`, and a
loom-side `ggml_map_custom` kernel would be a C function pointer no GPU can dispatch, forcing scheduler
splits (see `lua_bridge.h`'s own criterion and `graph_builder.h` on hybrid splits) as well as violating
the lean-runtime rule. So: **(1)** fix tinyBLAS's ARM F32 path in ggml's `llamafile/sgemm.cpp` -- it is
already the designated hook and already gated in; **(2)** add a blocked F32 path to
`ggml_compute_forward_mul_mat` and upstream it; **(3)** vendoring MLAS wholesale is license-clean but
7.6 MB of multi-architecture assembly with its own build system, against an engine whose selling point
is size -- read it, do not import it.

**One gate consequence.** The prototype is NOT bit-identical to ggml's current output (~3e-7 relative,
a different summation order). Any byte-identity gate over a conv-bearing model has to become a
tolerance gate, or be re-baselined, before this can land. See CLAUDE.md's tensor-oracle rule.

## Picking this up from a cold start — environment, commands, and the numbers to check yourself against

Everything below was measured on **`rpi4`** (Raspberry Pi 4B rev 1.5, Cortex-A72, 4 cores @ 1.8 GHz,
LPDDR4, Debian aarch64), reachable over the LAN as `ssh pi@rpi4` with key auth. Two engines, one model,
one utterance throughout:

* model `loom-ai-org/vits-piper-en-gb-miro-loom`, HF cache at
  `~/.cache/huggingface/hub/models--loom-ai-org--vits-piper-en-gb-miro-loom/snapshots/*/vits-piper-en-gb-miro.gguf`
  (the same GGUF is at `../hf-models/vits-piper-en-gb-miro/` on the dev box);
* the ONNX original at `/home/pi/pipertts-en-gb-miro/miro_en-GB.onnx`, driven through `phoonnx`;
* venv `/home/pi/test` (has `loom-py-rt` 1.0.0rc4, `onnxruntime` 1.28.0, `phoonnx`);
* text "Hey, can you shutdown the computer, my friend?", phonemes
  `hˈeɪ, kæn juː ʃˈʌtdaʊn ðə kəmpjˈuːtɐ, maɪ fɹˈɛnd?` -> 100 ids -> y_length 287 -> 73472 samples.

**Baselines to reproduce before trusting anything you measure** (idle box, both steady-state after a
warm-up call): loom **2.35 s**, onnxruntime **1.024 s**, ratio **2.3x**. loom `from_file` load 0.19 s
(`from_pretrained` is 1.55 s — that extra 1.36 s is an HF revision check, not loading); onnxruntime
`InferenceSession` construction **5.5 s**. If your loom number is far off 2.35 s, stop and fix the
measurement rather than reasoning from it.

**Commands.**

```sh
## per-op profile of the real graph -- ONE thread, see the trap in P4.14
LOOM_N_THREADS=1 LOOM_PROFILE=1 ./build/tools/loom_cli/loom_cli --model <gguf> --prompt "..." --n-predict 8
LOOM_PROFILE=/tmp/p.txt loom_cli ...            # to a file instead of stderr

## the GEMM prototype vs ggml, at the eleven real flow_vocoder shapes
g++ -O3 -std=c++17 -fopenmp -march=armv8-a -I <ggml>/include -I <ggml>/src \
    scripts/bench6.cpp -o bench6 -L<build>/src -L<build>/src/ggml-cpu \
    -lggml -lggml-base -lggml-cpu -lgomp -lpthread -lm
./bench6 4          # expect ggml ~15.0 GFLOP/s, prototype ~24.4, ratio ~1.62x

## onnxruntime's own per-op profile, for the other side of the comparison
##   ort.SessionOptions(); so.enable_profiling = True; so.intra_op_num_threads = 4
##   ... then session.end_profiling() -> JSON; aggregate events with
##   cat == "Node" and name ending "_kernel_time" by args["op_name"].
##   This build's events carry NO run_index: split runs by ORDER (len(kernels)//n_runs).
##   Profiling costs onnxruntime 1.18x, so use SHARES only and apportion them over an
##   un-profiled wall time measured separately.
```

**Three measurement traps, each of which produced a wrong answer here that survived a full write-up.
Check all three before believing a number.**

1. **Never read a ggml tensor you have not written.** Untouched anonymous pages all map to the shared
   zero page, so the whole read set sits in L1. Inflated an isolated conv benchmark **2.1x** (103 ms
   against a real 219 ms) and silently invalidated the first `GGML_LLAMAFILE` A/B. `scripts/bench6.cpp`
   fills every input; copy that.
2. **Confirm the box is idle.** `uptime` costs nothing. An orphaned `prof_main` held one of four cores
   for 67 minutes and manufactured a 1.22x win for `ggml_conv_2d_direct` that is really 0.98x.
   **A `timeout`-killed ssh does NOT kill the remote process** — `pkill` on the far side after any
   aborted remote run.
3. **Check the parts fit inside the whole.** The contaminated run put the conv total at **2.589 s
   inside a 2.35 s synthesis**. A component larger than its container is not a noisy measurement, it is
   a wrong one; stop and find the cause instead of caveating it.

**Already measured and CLOSED — do not re-propose without new evidence:**

| proposal | result |
|---|---|
| unfused elementwise/activation chain is the gap | **no** — 6.5% of graph time at 1 thread |
| the C++<->Lua array boundary is expensive | **no** — 18.7 ms total |
| `GGML_LLAMAFILE=ON` (tinyBLAS) fixes the GEMM | **on ARM only once GCC's codegen for it is fixed** — 15.4 vs 15.0 as upstream ships it, 25.1 with P4.15's two patches. On x86-64 the flag alone is ~2x |
| lower CONV_1D as `kw` shifted mul_mats to dodge im2col | **worse** — 0.43-0.98x |
| `ggml_conv_2d_direct` (ggml's MLAS-shaped fused conv) | **nothing** — 0.98x, despite bit-identical output |
| im2col materialisation is the gap | **no** — removing it (row above) changes nothing |

**Still open, in the order they are worth doing** — P4.15 is DONE and took 0.75 s off 2.33 s, leaving
a 0.55 s gap that the table at the top of this entry attributes. Convolution is 84% of it, and inside
that convolution the GEMM is finished (23.5 GFLOP/s in-model against 25.1 standalone) while the im2col
that feeds it is 37% of the time:

1. ~~A direct convolution behind a shape heuristic~~ — **DONE, `ggml-0006`**, 1.05x on the Pi and
   1.29x on x86; see the section above for why those two numbers are so far apart and why the bench
   promised more than either. What is left inside the convolution is the dilated shapes, where the
   direct kernel gets 1.2-1.4x rather than the 1.6x it manages at dilation 1. The tap-major traversal that
   would fix it **is shipped** (section above), for the two shapes in this model where it measures a
   win: 1.487 -> 1.463 s. The shapes outside its window -- 32 and 64
   channels -- are **measured out**: the kernel is at 83% of peak in cache and the loss is scaling on a
   shared L2, the tile is already the best of three, and the one thing still on the table there was
   round-robin block partitioning, now fixed for another 10% on the largest shape. What remains needs
   graph-level fusion (keeping a resblock's intermediate in cache), not a better kernel — **done as
   P4.15b below**, for 1.441 -> 1.307 s. Note that P4.15b also MEASURED OUT the "keep the intermediate
   in cache" half of that sentence: it is worth 1.05x, and the win came from the elementwise nodes and
   from a padded copy that was writing everything twice.
2. ~~**`CONV_TRANSPOSE_1D` as mul_mat + reshape**~~ (P4.14's follow-up list, ~60-90 ms) — **DONE,
   P4.15e**, and worth about twice what this line estimated: 195.8 -> 79.1 ms for the op, ~115 ms of
   the synthesis. Half of it was a serial prologue nobody had looked for. "At rough parity with
   onnxruntime's" was true and misleading — both engines were sitting on the same floor, and a ratio
   against a competitor cannot see that. Phase-timing against the machine's own peak can.
3. **"everything else"**, 0.29 s against onnxruntime's 0.257 s — nearly closed, and what is left of it
   is loom's own 165 ms of graph build, Lua driver and marshalling rather than any op.

P4.13 (quantized conv kernels) **conflicts** with the direct-conv lowering — `CONV_2D` takes an F32/F16
kernel and a quantized one falls back to the im2col path, so decide between them rather than starting
both.

> **Resolved when P4.13 shipped (2026-08-30), and the conflict turned out to be a choice, not a
> collision.** The trade is per FILE, not per build: the kernel fold happens only when the export is
> block-quantizing, so an F32 or F16 artifact keeps the direct lowering untouched and byte-identical
> (verified — an unquantized VITS export is byte-for-byte what it was), and only a Q4_0/Q8_0 artifact
> gives it up. It was never giving up anything it could have had: `ggml_conv_2d_direct` takes an
> F32/F16 kernel, so a quantized model could not have reached that path in any layout. See P4.13 in §5.

**This is what the item told its own future reader to do first, and it was the right instruction:**
re-read `scripts/bench6.cpp`'s header, then ggml's `src/ggml-cpu/vec.cpp:ggml_vec_dot_f32` and
`llamafile/sgemm.cpp`'s `mnpack`/`gemm<M,N>`, and settle the bit-identity oddity BEFORE writing a
kernel. Settling it made the kernel unnecessary. `scripts/bench7.cpp` is the bench that settled it:
one OpenMP driver, tinyBLAS's own block lifted verbatim, one tile shape per column.


### P4.15b — graph-level fusion for resblock chains — DONE (2026-08-21)

**What shipped, and what it is worth.** Three changes. `ggml-0007-conv1d-elementwise-fusion.patch`
folds a resblock's `LEAKY_RELU` and its residual `ADD` into the convolution between them; a
one-function change to `ensure_packed` in loom's own op layer stops emitting one `ggml_cont` per
CONSUMER of a non-contiguous tensor; and `ggml-0006`'s padded copy stops zeroing a buffer it is about
to overwrite. On a Pi 4 at 4 threads, steady state on a quiet box, **1.441 -> 1.345 s** —
**96 ms, 6.7%**, and the same delta again from the interleaved four-arm run below. Step 2 turned out
to be worth 1.05x on the chains and was **not built**; what its investigation found instead — a padded
copy writing every element twice — took another **38 ms**, so the item as a whole is
**1.441 -> 1.307 s, 134 ms and 9.3%**. Against onnxruntime **re-measured with its duration predictor
pinned** — 1.044 s raw `session.run` at 72192 samples, 1.063 s normalised to loom's 73472 — the ratio
goes **1.36x -> 1.24x**. (The 1.024 s this thread quoted for years was an UNPINNED run; see the
correction under P4.15's own onnx baseline. Every ratio derived from it is ~4% optimistic.)

That is close to the ~120 ms this item predicted from the traffic table below, which is the first thing
in this thread where the arithmetic and the measurement agreed.

**Why this exists.** P4.15 took the convolution from 2.33 s to 1.44 s and then measured itself out: the
kernel reaches 83% of the machine's peak in cache, it has no spills, its tile is the best of three, and
its remaining loss is scaling on a shared L2 (1 -> 4 threads is 2.35x, not 4x). What is left is not
arithmetic, it is **how many times the vocoder's activations cross the memory bus**. This item is about
removing crossings, which is a graph-level question and not a kernel one.

**The machine's number, which bounds everything below.** One pass over a 9.4 MB activation (the
vocoder's largest) costs **4.7 ms** to read-and-write and **8.3 ms** for a two-read-one-write add, at
**3.4-4.0 GB/s** — and *one* core almost saturates that, so more threads do not help. Elementwise ops
here are not slow code; they are the bus. The only way to make them cheaper is to not do them.

### What the graph actually looks like

From `model.graph_topology.flow_vocoder`, a HiFi-GAN resblock layer is exactly this, repeated (dilations
1, 2, 3, 6, 12 across layers):

```
LEAKY_RELU(x)            -> h          # a full pass, 4.7 ms at the largest scale
CONV_1D(w, h)            -> c
ADD(c, bias_reshaped)    -> xt         # FUSED by ggml-0005
ADD(xt, x)               -> x'         # the residual: a full pass, 8.3 ms
LEAKY_RELU(x')           -> h'         # and round again
```

Per convolution there are **two** elementwise passes the convolution could absorb and one it already
has. Measured on the pre-P4.15b build, 4 threads, `$LOOM_PROFILE` (which does not fuse, so the bias
ADDs in it are not a real cost — subtract them):

| op | profile | of which is real |
|---|---|---|
| `ADD` | 193 ms | ~80-100 ms is the residual; the rest is bias, already fused |
| `LEAKY_RELU` | 41 ms | all of it, and every one feeds a convolution |
| `UNARY` | 36 ms | mostly tanh/sigmoid in the flow, not the resblocks |
| `CONT` | 49 ms | copies the graph asks for — and **three quarters of them were redundant**, below |
| `MUL` | 15 ms | gating |

### What was done, and the four numbers that decompose it

Both changes are switchable at runtime in the measurement build (`LOOM_NO_DEDUP`, `LOOM_NO_ACT`,
`LOOM_NO_RES` — probes, not shipped), so all four arms are the same binary. That matters more than it
sounds: see the measurement note below.

| arm | Pi 4, 4 threads | vs base |
|---|---|---|
| base (patches 1-6, no dedup — matches the 1.441 s baseline binary within noise) | **1.44 s** | — |
| `ensure_packed` dedup alone | 1.43-1.45 s | ~0-20 ms |
| the fusion alone | **1.40 s** | ~50 ms |
| both | **1.34 s** | **~100 ms** |

**The two are strongly superadditive, and that is the whole finding.** The dedup on its own is worth
almost nothing — it removes three copies of a large tensor, and a copy is cheaper inside a graph than
its isolated profile time suggests. What it is really worth is **unblocking the fusion on half the
vocoder's convolutions**, which is 50 ms more on top of the fusion's own 50.

* **The fusion** (`ggml-0007`). `ggml_cpu_conv_2d_bias_add_idx` becomes `ggml_cpu_conv_2d_fusion`,
  matching the chain from either end — from the `LEAKY_RELU` when the convolution is its only consumer,
  or from the convolution — with all of `act`, `bias` and `residual` optional. Fusion runs FORWARD from
  a node, so a unary before the convolution can only be absorbed by starting the match at the unary;
  that is why there are two entry shapes and not one.
* **The dedup** (`src/ops/primitives_*.cpp`). `ensure_packed` called `ggml_cont` unconditionally, so a
  non-contiguous tensor got one packed copy per op that read it. The upsampler's output is a trimmed
  view read by a `LEAKY_RELU` and by three resblock residual adds: **four copies of the same 9.4 MB
  tensor** (`CONT 73472x32, 4 calls` in the profile, now 1). It now reuses one already built in the same
  `ggml_context`, which is per graph build, so there is nothing stale to find.

**Why the dedup unblocks the fusion.** With a `CONT` per consumer, the graph reads
`CONV -> ADD(bias) -> CONT -> ADD(residual)`, and the `CONT` is a real node whose result the residual
add needs — it cannot be skipped, and the detector requires the residual add to be the next node it
would compute. So the three convolutions per upsample stage whose residual is the upsampler output
matched `act=0 bias=1 res=0`, and only the three whose residual is a plain ADD output matched fully.
After the dedup all six match, three of them with the unary as well:

```
3 FUSE conv ne=[73472,1,32] act=1 bias=1 res=1      # the inner convolutions
3 FUSE conv ne=[73472,1,32] act=0 bias=1 res=1      # act=0: their LEAKY_RELU has three consumers
```

The first activation of each resblock layer is shared by three convolutions, so it can never be
absorbed — 3 of every 4 unaries in a layer can be, which is where the leaky half of the win stops.


Phase-timing the shipped kernel (pad+pack against sweep, `$LOOM_PROFILE` cannot see inside an op) put
the padded copy at **77 ms of a 1.345 s synthesis** and the sweep at 428 ms. Reading the copy to find
out why it was so expensive found it doing half again as much work as it needed:

```c
memset(row, 0, LP * sizeof(float));      // the WHOLE row
memcpy(row + pad, x + ic * L, L * ...);  // ... and then all but 2*pad of it again
```

Every element of a 9.4 MB buffer written twice, to zero two strips of 36 floats. Zeroing only the
strips is **1.345 -> 1.307 s**: a runtime A/B inside one binary, ABBA in both orders, puts it at
**26 ms**, and the two shipping builds' steady states 38 ms apart. Call it ~30 ms, 2%.
It is in `ggml-0006` rather than `ggml-0007` because it is `ggml-0006`'s copy, and it makes that
patch's upstream claim ("that copy is one pass against im2col's `kw`") true, which it was not.

What is left of the padded copy is one honest read and one honest write. Removing THOSE is only
possible for the convolutions with no fused unary — the copy is where a fused `LEAKY_RELU` is applied
once, and applying it in the sweep instead costs `kw` times per element, which doubles the inner loop.
That is worth ~25 ms at best and needs the aliasing argument all over again; it is the next item here,
not this one.


**Machines.** `ssh pi@rpi4` — Raspberry Pi 4B, Cortex-A72, 4 cores @ 1.8 GHz, 1 MB shared L2, 32 KB
L1D, gcc 14.2 and clang 19. Repeatable to ~1% **when it is cool and nothing else is on it**, and to
about 9% when it is not — see the measurement note above. The dev box is x86-64 (Ryzen 3 3250U, AVX2,
2 cores, 4 MB L3) and is **thermally noisy** — pin with `taskset -c 0,2` and take medians of seven, or
it will lie to you by 15%.

**The model and the utterance**, used by every number in P4.14/P4.15/P4.15b:
`loom-ai-org/vits-piper-en-gb-miro-loom`, phonemes
`hˈeɪ, kæn juː ʃˈʌtdaʊn ðə kəmpjˈuːtɐ, maɪ fɹˈɛnd?` -> 100 ids -> y_length 287 -> 73472 samples. On the
Pi: `~/.cache/huggingface/hub/models--loom-ai-org--vits-piper-en-gb-miro-loom/snapshots/*/*.gguf` (two
snapshots exist — take `ls -t | head -1`). On the dev box: `../hf-models/vits-piper-en-gb-miro/`.

**Baselines to reproduce before trusting anything.** Pi, 4 threads, idle, steady state: **1.307 s**
after this item, **1.205 s** as of P4.15e, and **1.099 s** as of P4.15f (**1.441 s** before any of
them; onnxruntime does the same utterance in **1.063 s** with its duration predictor pinned, so the
ratio is **1.24x** after P4.15b, 1.13x after P4.15e and **1.03x** now). **P4.15f changed the GGUF**, so
a pre-P4.15f export measures the old number on a new engine -- check the file has a `text` topology
before comparing. x86, two threads pinned: **~1.17 s**. If your first number is far off, fix the measurement,
not the code.

**Scratch trees on the Pi** (all disposable, none of them a git repo):
* `~/loom-p416/loom.cpp` — **the current one (P4.16, 2026-08-30)**: a plain rsync of `45d5db9` with
  `bench_vits_loom` built beside it, `-DCMAKE_BUILD_TYPE=Release`, targets `loom_engine loom_cli` only
  (the test targets are the long tail of that build and nothing here needs them). `~/loom-no12` is the
  same tree with `ggml-0012` removed, for P4.26's A/B. (`~/prof_onnx_shapes.py`, the Pi-only original,
  reads its phoneme ids from `/tmp/ids.json`; `scripts/prof_onnx_conv_shapes.py` carries them inline and
  needs no such file.) **rsync to this board leaves FUTURE mtimes** — its clock runs ~240 s behind
  the dev box — so `find <tree> -type f -exec touch {} +` afterwards, or ninja rebuilds in a loop.
* `~/loom-p415/loom.cpp` — the older one, pre-`ggml-0010`; a full checkout with `prof_main` appended to
  its CMakeLists. Rebuild with `cmake -B build -DCMAKE_BUILD_TYPE=Release` (**Release
  matters: the repo default is RelWithDebInfo and that is 1.39x slower**).
* `~/ggml-bench` — standalone benches with a stale ggml checkout of its own; `bench10` links against
  `~/loom-p415/loom.cpp/build/_deps/ggml-build/src`, so `LD_LIBRARY_PATH` must point there.
* `prof_main <gguf> <phonemes> <reps>` prints per-rep wall time. **It is NOT in this repository** --
  no source, no CMake target (checked 2026-08-29); it exists only as the local addition to the Pi's
  `~/loom-p415/loom.cpp` checkout described one bullet above, so **anywhere else it is not a command
  you can run.** Off the Pi use `scripts/bench_vits_loom.cpp` for VITS wall time (it pins the three
  scales the onnxruntime arm pins) or `tools/loom_cli` with `$LOOM_PROFILE` for per-op.
  **`$LOOM_N_THREADS` sets threads -- not `LOOM_THREADS`, which nothing reads** (`backend.cpp:323`);
  `LOOM_PROFILE=1` or `=<path>` gives the per-node profile.

**Commands.**

```sh
cmake -B build && cmake --build build -j"$(nproc)"     # applies cmake/patches/ automatically
ctest --test-dir build -L ci                            # 68 tests, hermetic
LOOM_FIXTURES=~/Dev/loom-engine-artifacts/v5 ctest --test-dir build -L gate    # 82, real models
cmake -B build -DCMAKE_CXX_FLAGS=-DLOOM_CONV1D_DIRECT=0 # the OTHER conv lowering; both must pass
cmake -B build -DLOOM_TINYBLAS=OFF                      # no tinyBLAS, for GEMM A/Bs
GGML_CPU_DISABLE_FUSION=1 <binary>                      # turns every CPU-backend fusion off
```

**Measuring against onnxruntime — PIN THE DURATION PREDICTOR FIRST.** It lives in the `~/test` venv on
the Pi (onnxruntime 1.28.0, phoonnx), model `~/pipertts-en-gb-miro/miro_en-GB.onnx`. VITS's duration
predictor is stochastic and phoonnx does not seed it, so consecutive runs synthesise 72k-76k samples
and time a different amount of work each; both engines are near-linear in output samples, so an
unpinned comparison is a comparison of two different utterances. It cost this thread a baseline that
stood for weeks — see the correction under P4.15's onnx section.

```python
cfg = SynthesisConfig(noise_scale=0.0, noise_w_scale=0.0)   # phoonnx, or
sess.run(None, {"input": ids, "input_lengths": n, "scales": [0.0, 1.0, 0.0]})  # raw session
```

Then normalise by sample count before quoting a ratio: onnxruntime is 1.044 s at its pinned 72192 and
**1.063 s at loom's 73472**. Drive the session directly for an engine-to-engine number (phoonnx adds
~14 ms of Python); its per-op profile needs `so.enable_profiling = True` and `so.intra_op_num_threads
= 4`, and **shares only** — profiling costs onnxruntime 1.18x, and this build's events carry no
`run_index`, so runs split by ORDER. **Both halves are in `scripts/` as of 2026-08-30** —
`bench_onnx_tasks.py vits` (wall) and `prof_onnx_conv_shapes.py` (per-shape); the Pi's
`~/bench_onnx2.py` and `~/prof_onnx_shapes.py` are the originals they were lifted from.

**The twelve ggml patches** live in `cmake/patches/` and are applied at configure time by
`cmake/GgmlPatches.cmake`, which resets and retries if one no longer applies — so **editing a patch and
re-running cmake just works**, and `git -C build/_deps/ggml-src checkout -- .` is the manual reset.
`cmake/patches/UPSTREAM.md` is the PR write-up for all seven. The benches are in `scripts/`:
`bench6` GEMM, `bench7` register tiles, `bench9` conv lowering, `bench10` direct convolution,
`bench11` the resblock chain. To develop an eighth, snapshot the three
files `cmake/patches/` touches after a clean configure, edit `build/_deps/ggml-src/...` directly, and
`diff -u` the snapshot against the tree — the snapshot IS the "earlier patches applied but not yours"
baseline, and it survives the reconfigure that a `git checkout` would undo.

**Five traps, each of which produced a wrong answer that survived a write-up** (and see the three
above, which are new):

1. **The profiler cannot see fusion.** `$LOOM_PROFILE` submits one node per graph, and a one-node graph
   has nothing to fuse with, so a profiled run is always the unfused one. Measure fusion with
   `GGML_CPU_DISABLE_FUSION=1` against the same binary — or, better, with a runtime switch per fusion,
   which is what P4.15b did — never by profiling.
2. **A node's isolated profile time is an upper bound on its marginal cost, sometimes a loose one.**
   The bias ADDs profiled at 0.20 s and removing them saved 0.03 s; the redundant `CONT`s profiled at
   7 ms and removing them saved under 20.
3. **A graph allocator will hand an op the buffer its own input just vacated.** Safe in the unfused
   order, corruption when fused — it cost a 0.54 max-abs-diff on two gate models. Any new fusion must
   check `ggml_cpu_tensors_overlap` against every tensor it reads while writing. P4.15b adds a second
   case of the same thing: the residual operand is usually the DESTINATION.
4. **An artefact big enough to hide the comparison is worth fixing before reading the comparison** —
   three times here now: a scalar tail, scalar edge blocks, and the accumulator spill above.
5. **The shapes a bench holds are not the shapes a model runs.** A heuristic tuned on eleven shapes made
   the synthesis *slower* until it was checked against all 153 convolutions the model actually issues;
   and a bench without the model's dilations measures a convolution the model does not have.
6. **A ratio against another engine silently assumes both engines are doing the same work.** They were
   not: loom ran the text encoder twice, and P4.16's ORIGINAL table (since re-measured and replaced,
   see P4.16 below) was written, ranked and reasoned from
   before anyone counted the nodes (P4.15d). Counting them cost an afternoon and no measurement rig, and it moved the worst
   row in that table from 2.59x to 1.38x. `scripts/conv_census.py` is that count, for any GGUF.

**And the check that made this item honest: prove the gate can fail before believing it passed.**
Perturbing the fused slope by 5%, and separately the fused residual by 5%, both make
`test_e2e_matcha_mil_lua_driver` fail — so the 82 green tests are green about something. Fused and
unfused VITS output agree to 6.7e-8 max on a 0.17 peak, identically at 2 and 4 threads, which a race
would not do.


### P4.15e — `conv_transpose_1d`: a serial prologue and a dot-product compute — DONE (2026-08-22)

**What it was worth.** Two patches, `ggml-0008` (the prologue) and `ggml-0009` (the compute), on a Pi 4
at 4 threads: **1.314 -> 1.202 s, about 115 ms and 8.5%.** The op itself goes **195.8 -> 79.1 ms,
2.5x**, and against onnxruntime's 166.4 ms for the same three convolutions it ends up **2.1x faster**
where P4.16's original table had it 1.18x slower. Each half was measured on its own with the old path switchable at
runtime inside one binary, ABBA in both orders over two rounds: the prologue 1.314 -> 1.247 by mean
(~70 ms), the GEMM 1.252 -> 1.202 by mean and 1.239 -> 1.186 by min (~50 ms).

### Part 1: the prologue (`ggml-0008`, ~70 ms)

 The op goes from 196 ms to
**126.3 ms** re-profiled, which puts it **below** onnxruntime's 166.4 ms for the same three
convolutions — 1.32x FASTER, where P4.16's original table had it 1.18x slower. Per shape, before -> after:
73476x32 105.7 -> 63.3, 18376x64 53.1 -> 43.0, 2304x128 37.0 -> 19.9 ms. Measured by switching the old prologue back on at runtime inside one binary, ABBA
in both orders over two rounds.

**How it was found, which is the reusable part.** P4.16's original table put `CONV_TRANSPOSE_1D` at 1.18x and +29 ms —
the smallest row in the table, and by that ranking not worth doing. The ranking was wrong because it
compared loom against onnxruntime, and **both engines were slow at it**: 1.43 GFLOP in 196 ms is
**7.3 GFLOP/s**, against 25 GFLOP/s for a GEMM on this machine, and onnxruntime's 166 ms is only
8.6 GFLOP/s. A ratio against a competitor cannot see a floor they are both sitting on. Phase-timing the
op against the machine's own peak could, and did, in about twenty minutes.

**What the prologue was doing**, all of it under `if (ith == 0)`:

```c
memset(params->wdata, 0, params->wsize);   // the whole PLAN's work buffer -- 16 MB here
... transpose the kernel  (K x Cout x Cin) -> (Cin x K x Cout)
... transpose the source  (L x Cin)        -> (Cin x L), element by element
memset(dst->data, 0, ggml_nbytes(dst));    // dst is accumulated into, so this one is needed
```

| OC x OL | calls/synth | prologue (serial) | compute |
|---|---:|---:|---:|
| 128 x 2304 | 1 | 27.2 ms | 16.9 ms |
| 64 x 18376 | 1 | 16.9 ms | 40.1 ms |
| 32 x 73476 | 1 | 56.2 ms | 54.7 ms |
| **total** | 3 | **100.4 ms** | 111.7 ms |

**47% of the op, serial, in the middle of a graph whose every other op is parallel.** And the `wdata`
memset needed to exist at all: the two transposes write every element of the two regions the op uses,
and `wsize` is not this op's requirement but the **maximum over every node in the plan** — 16 MB, sized
by an unrelated convolution, zeroed per call to use about 4.

The fix is the obvious one plus one non-obvious choice: the source transpose is split over **`L` rather
than `Cin`**, the other way round from how it read. A transpose is strided on one side, and strided
READS beat strided writes — each thread now fills whole contiguous `ne11`-wide rows instead of
scattering single floats `ne11` apart across the buffer, which also keeps two threads off one cache
line.

### Part 2: the compute is a GEMM (`ggml-0009`, ~50 ms)

The inner loop was one `ggml_vec_dot_f32` of length `Cin` per `(output channel, input position, tap)` —
1x1 register blocking, the exact shape P4.15 removed from `mul_mat`, and 7.3 GFLOP/s where the machine
does 25.

**The right-hand side does not depend on `s0` at all.** `y[oc][i10*s0 + k] += sum_ic w[oc][k][ic] *
x[i10][ic]` is, over `(oc, k)` and `i10`, exactly `[Cout*K, Cin] x [Cin, L]`; only the SCATTER of the
result knows about the stride. And the prologue's two transposes already leave both operands in the
layout a GEMM wants, contraction over `Cin` fastest on both sides — so this is one
`ggml_call_mul_mat_ldc` (patch 0004's helper) plus an overlap-add, and nothing had to be repacked.

Blocked over input positions, because the whole result is `K/s0` times the size of dst — 18.8 MB
against 9.4 for the largest upsample — and not paying that traffic is the point. A 256 KB block stays
in L2 and the scatter reads it immediately; `ggml_graph_plan`'s work size for the op grows by the same
budget, and both read one constant so they cannot drift.

**The scatter splits by OUTPUT CHANNEL, and that is load-bearing.** Consecutive input positions write
overlapping runs whenever `K > s0` — which is every upsampling convolution, they are built that way —
so a split over positions would race exactly where this op accumulates. Per (channel, position) both
sides are then contiguous over the tap.

Not bit-identical: the GEMM sums over `Cin` in a different order, 8.9e-8 max on a 0.17 peak, and
identical at 1, 2 and 4 threads — which is also the check that the scatter does not race.

This was the "`CONV_TRANSPOSE_1D` as mul_mat + reshape" that P4.14 and P4.15 both listed and neither
did. Per shape, 195.8 ms -> 79.1 ms: 73476x32 105.7 -> 35.1, 18376x64 53.1 -> 27.6, 2304x128
37.0 -> 16.4 ms.


## 5. Planned Work

### P4.21 — `QK^T` at `k = 64`: the vector lanes are on the wrong axis — MEASURED OUT, CLOSED 2026-08-29

**The premise held and the ceiling did not.** An outer-product tile really does delete the per-output
horizontal reduction — instructions per output fall **36.2 -> 19.7**, a 1.84x cut, exactly the
mechanism P4.18 predicted. It buys **1.38x** on the machine the mechanism was found on, because the
instruction count was never a time ceiling: the reductions were filling issue slots the FMA ports were
not using, and once they are gone the kernel is port-bound at **91% of the machine's roofline**. The
gate was 1.5x; it clears on one of three boxes. **Do not re-propose it as scoped.** What the
experiment found instead is worth 2.75x and is P4.22, below.

The measuring stick is `scripts/bench16.cpp`, which is self-contained against a built ggml and prints
the gate verdict itself. Everything below re-derives from it. **The counters this item was scoped from
are not repeated here** — the `perf` table (IPC 5.40, 58% of the FP ports at `k = 64`, the
`0.2355k + 18.0` fit and its `k` sweep) is P4.18 item C's, in §2 above, and it is still correct; what
was wrong was the conclusion drawn about time, below.

#### The gate, on three boxes

whisper-small's own `QK^T` — `m = n = 1500, k = 64, 12 heads` — one thread, ABBA per rep and median
per arm, **pack counted**, best register tile and column block per box (`-DMR_VECS -DNR_TILE
-DNB_COLS`):

| box | ISA | roofline, 1 core | `tinyBLAS` dot | outer product | **ratio** | perfect-kernel ceiling |
|---|---|---:|---:|---:|---:|---:|
| Ryzen 3 3250U | AVX2 | 54.6 GF | 23.9 GF (44%) | 44.3 GF (81%) | **1.85x** | 2.24x |
| Core Ultra 9 285K, P-core | AVX2 | 177.2 GF | 116.6 GF (66%) | 161.3 GF (91%) | **1.38x** | 1.52x |
| Raspberry Pi 4, Cortex-A72 | NEON | 14.3 GF | 7.45 GF (52%) | 9.18 GF (64%) | **1.23x** | 1.93x |

The last column is the number that ends the item: **`roofline / ggml`, what a flawless kernel of any
formulation could be.** On the 285K it is 1.52x, so the 1.5x gate was at the machine's own ceiling
before a line was written, and the 8% between 1.38x and 1.52x is all that any further tuning could
find. The 1.76x from P4.18 was `(0.2355k + 18.0) / 0.2355k`, an INSTRUCTION ratio, and it is only a
time ratio if IPC holds. It does not: **5.02 -> 3.63** across the two arms.

The pack is not what killed it, and this is worth stating because it was the suspected cost: **1.0-1.5%
of the packed arm** at this shape (0.32 ms against 21.5 ms on the 285K, 4.4 MB copied). Panel-major,
not a plain transpose — a `[m, k]` transpose leaves the tile striding 6000 bytes between `k` steps.

#### Why the three boxes disagree, which is the transferable part

**`hsum` costs what the vector is wide.** In `sgemm.cpp`, `hsum(__m256)` is an `extractf128`, an
`addps`, a `movehl`, an `addps`, a `movehdup` and an `addss` — **six instructions**. `hsum(float32x4_t)`
is `vaddvq_f32` — **one**. So P4.18's 18.0-instructions-per-output intercept is an **AVX2 artifact**,
and on NEON there is almost no epilogue to remove. That is the whole of the Pi's 1.23x, and it is why
an aarch64 measurement could never have been extrapolated from the x86 one, or the reverse.

**And the wider the core, the better it hides what is left.** The dot-product kernel reaches 44% of
roofline on a 2019 laptop Zen+, 52% on an A72 and **66%** on a Lion Cove P-core. The reductions are
cheap ALU work that a wide out-of-order core issues in slots the FMA ports leave idle, so the newer
the core, the less there is to win — the trend runs against this item, not with it. Anyone tempted to
re-open it x86-only should read the 1.85x on the Ryzen and the 1.38x on the 285K as two points on that
line, not as two boxes disagreeing.

#### The `k` threshold, since the sweep is cheap and settles the dispatch question

285K, one thread, unblocked, ratio with pack: **k=32 1.77x, k=48 1.72x, k=64 1.38-1.54x, k=96 1.25x,
k=128 1.14x, k=192 1.08x, k=256 1.03x, k=384 1.01x, k=768 0.94x.** It goes under 1.0 by `k = 768`,
where the pack is 3.9% and the epilogue is 9% of the work. So the dispatch condition the design sketch
asked for is real and it is roughly `k <= 128` — but the band where it pays anything is also the band
where it pays least on the boxes that matter.

#### What was tried before it was called, so it is not re-tried

* **Register tile**, both ISAs: `MR_VECS x NR_TILE` of (2,6) (3,4) (2,5) (3,3) (4,3) (2,4) on x86;
  (2,6) (3,4) (4,3) (2,8) (4,4) (3,6) on NEON. Best per box is in the table's config; the spread
  between the best and worst tile is 1.1-1.2x, and no tile changes the verdict.
* **Column blocking.** It is NOT optional on a small cache and it is nearly free on a large one: the
  Ryzen goes **1.50x -> 1.80x** from `NB = 48` alone (94 A-panels each streaming 384 KB of B becomes
  A re-read once per column block), the Pi 1.005x -> 1.23x, and the 285K moves 1.8% because its 3 MB
  L2 already held everything. A naive loop order failing is not the formulation failing — that is why
  the knob exists.
* **The operand-pointer hoist.** Worth ~1.15x on the Ryzen, nothing on the 285K. Same failure
  `ggml-0002` documents on aarch64, on the other ISA.

#### One trap in the harness, recorded because it nearly published a false number

The roofline probe must be checked in the disassembly, twice. Written the obvious way, gcc 14.2
collapsed sixteen independent vector FMA chains into a handful of **scalar** `vfmadd132ss` and reported
585 GFLOP/s on a machine that does 55 — every "% of roofline" on this page would have been a seventh of
the truth. It needs an empty `asm volatile` barrier on the operands. Then, fixed, `ACC = 16`
accumulators plus two operands is 18 live vectors in a 16-register file, and the spill reported 108
GFLOP/s on a machine whose own GEMM was doing 163. **`grep -c vfmadd231ps` on the probe's own loop is
the check**, and `bench16.cpp` carries both fixes.

### P4.22 — `tinyBLAS` false-shares `C` and stops threading at whisper's `m` — DONE 2026-08-29

**Shipped as `cmake/patches/ggml-0012-tinyblas-line-aligned-jobs.patch`.** whisper-small's `QK^T`
bucket is **391.2 -> 185.9 ms in model, 2.10x**, and it is the only bucket of forty that moves by more
than 1.2 ms. The transcription is **4.050 -> 3.858 s at 4 threads (1.050x)**, 1.056x at 8. Output is
**bit-identical**, so no gate baseline moved.

#### The bug, which is one line of arithmetic

`gemm()` gives ONE JOB the rows `[ii, ii + BM*RM)`; `C` is `m`-contiguous; so a job's store to one
column of its range is `BM*RM*4` **bytes** wide. `matmul_aligned` picks `BM` from what divides `m`:

| `m` | branch | `BM` | bytes of `C` per job | 1 thread | 4 threads | scaling |
|---:|---|---:|---:|---:|---:|---:|
| 1496 | `m % 8 == 0` | 2 | 32 (half a line) | 29.4-30.5 ms | 21.2-21.9 ms | 1.40x |
| **1500** | `m % 4 == 0` | **1** | **16 (a quarter)** | 29.6-30.9 ms | 30.0-31.5 ms | **0.98x** |
| 1504 | `m % 16 == 0` | 4 | 64 (**a full line**) | 29.4 ms | 10.6-11.0 ms | **2.75x** |

Monotone in the fraction of a line a job owns, and `perf stat -e task-clock` shows **3.65 CPUs busy**
in the 0.98x row — the threads run, they just pass a line back and forth. **It does not make the
kernel slower; it stops it threading.** `m` is a sequence length in every attention matmul, so this is
the common case: whisper's 1500 frames is `4 mod 16`, the worst residue there is.

#### The fix, and the guard that bounds its blast radius

PR 3's trick on the other end of the same axis: run `m - (m % 16)` at `BM = 4` and finish the 0/4/8/12
leftover rows in a column-split loop that keeps the same `4 x RN` tile (`gemm_rows`, modelled on
`gemm_tail`). **Guarded on `nth > 1`** — false sharing needs a second thread by definition, so at one
thread the schedule is instruction-for-instruction the old one. That guard is not decoration: the dev
box measured 148.7 vs 147.8 ms single-threaded either way, i.e. the guard costs no measured win and
removes a whole class of risk.

| `m = 1500`, 285K, `QK^T` shape | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---:|---:|---:|---:|
| before | 29.38 ms | 30.30 ms | 29.40 ms | 19.90 ms |
| after | 29.38 ms | 24.98 ms | **15.00 ms** | **8.12 ms** |
| | 1.00x | 1.21x | **1.96x** | **2.45x** |

**The `m = 1504` control is flat to within 1% at every thread count**, which is what says the patch
reaches only the branch it is aimed at.

#### Both ISAs, and "no change" is one of the results

* **x86, Core Ultra 9 285K**: the table above, plus the in-model numbers.
* **aarch64, Raspberry Pi 4**: **no change, and there is nothing there to fix** — it threads 3.5x at
  *every* one of the three `m`, so the collapse does not exist on it. 133.10 ms before against 133.65
  after (`m = 1500`, 4 threads, ABBA-interleaved medians of ten). A first pass that sampled base first
  in each pair read 1.9% slower; the box warmed 54 -> 82 C across the run and the ordering attributed
  the ramp to one arm. **Interleave ABBA or do not quote the number.**
* **Ryzen 3 3250U, 2 cores**: cannot resolve it, +-40% spread by the end of a session. Recorded as
  unresolved rather than as neutral — [Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md)'s
  dev-box lesson.

#### Why it stayed invisible, which is the transferable part

**P4.18 measured `BM` at one thread and concluded it was a cache knob, not a mechanism.** That was
TRUE: the instruction counts across 1496/1500/1504 are identical, which is what it measured. At four
threads the same knob is a **false-sharing** knob worth 2.75x. A single-threaded instruction count
cannot see a coherence protocol, and this repo's default profiling advice is *"profile with ONE
thread, or dispatch cost dominates"* — which is right for attributing time to ops and blind to exactly
this class of bug.

And it was found by an experiment aimed at something else: P4.21's standalone gate needed a threaded
arm to compare against, and the incumbent's arm not scaling was visible only because both were in the
same harness. **A measuring stick that only measures the thing you are proposing cannot tell you the
thing you are proposing is not the problem.**

#### Testing

`tests/ci/test_tinyblas_gemm` gains the residue class of the 16-row split — `m =
1488/1492/1496/1500/1501`, with 1488 as the no-remainder control — plus an `n` that does not divide
evenly across the threads, so the ragged column slice inside `gemm_rows` is exercised. The element
check gains a window around the seam, and **that window is load-bearing**: the leftover rows sit in the
MIDDLE of the matrix, not at its edge, so the pre-existing first-and-last-four-rows check had nothing
to say about them. With the window disabled, a sabotage that skips the first leftover tile drops from
8 failing checks to 2.

Verified red three ways before it was believed ([ADR-015](../adrs/adr-015-ci-and-gate-test-classes.md)):
leftover rows never computed (7 checks fail, 1e1-1e2 relative), ragged column slice never finished
(3 fail), first leftover row-tile skipped (8 fail). **137/137 green** with all three removed, on the
dev box and on the workstation. The whole `ci` label is green as well: **70/70**.

**The export gates were NOT re-run, and here is the argument for why they cannot move.** This is a
SCHEDULING change: which thread takes which tile, and how many rows a job owns. Every output element
is still computed by the same `gemm_bloc<RM, RN>` over the same `k` range in the same order, so the
bytes are identical -- verified rather than asserted, with an FNV-1a hash of the whole result buffer
agreeing between the two builds at `m = 1492/1500/1501/1504` x 1/4/8 threads, and at every GEMM shape
whisper's encoder runs (`QK^T`, `A@V`, the three dense projections) plus VITS's `m = 287`. A
byte-identity baseline cannot move under a change that produces identical bytes; if a future edit here
ever does move them, that is the signal to re-record.

#### Still open here, and it is not this patch

`m = 1500` reaches 15.0 ms where `m = 1504` reaches 10.9. The remainder is **`ldc` alignment, not
sharing**: 1500 floats is 6000 bytes, `6000 % 64 = 16`, so a job's 64-byte store still straddles two
lines on odd columns. Closing it means padding `C`, which is an allocator change and a much bigger
proposition than this. Worth ~4 ms of a 3.9 s transcription; **do not open it without re-measuring
first.**

### P4.16 — the convolution table, re-measured — DONE, AND CLOSED: the gap is not in the convolution

**Executed 2026-08-30 on the Pi 4B, against the post-P4.15f export** (`vits_mil.gguf`, md5
`28f0cd01…`, byte-identical to the v5 fixture) **and a fresh build of `45d5db9`** — the tree the old
numbers came from predates `ggml-0010/0011/0012`, which includes P4.20's aarch64 regression fix, so
nothing measured before this is comparable on this machine.

**The verdict, in one paragraph.** Convolution is no longer where the gap is, and there is no longer a
per-shape ranking to attack. At four threads loom is **1.074x** onnxruntime (**+78 ms** of 1137), its
convolution runs at **~22 GFLOP/s against the box's ~25 GFLOP/s GEMM peak** and onnxruntime's at 23.7,
and its largest elementwise ops sit at **98-99% of the machine's measured streaming roofline**. Both
engines are against a roofline in both halves of the model. What the re-measurement *did* find is a
mechanism, and it is not a convolution: **ggml runs `TANH`, `SIGMOID`, `EXP` and `LEAKY_RELU`
single-threaded by construction** (`ggml_get_n_tasks`, `n_tasks = 1`) with scalar `tanhf`/`expf` loops
in `ggml_vec_*`, and VITS's WN gate spends **30.4 ms per synthesis** there — identical at one thread
and at four, and nowhere near any roofline. That is P4.25; this item is closed.

#### The invariant, checked before any ratio was believed

Both engines synthesised **73216 samples** on every run (the duration predictor pinned to
`noise_scale = noise_scale_w = 0`, `length_scale = 1` on both sides), and `scripts/conv_census.py`
and both profiles agree on the graph **one node for one node**:

| | loom | onnxruntime |
|---|---:|---:|
| dense convolutions | 117 | 117 (109 `Conv` + 8 `FusedConv`) |
| depthwise | 12 | 12, weights `[192, 1, 3]` |
| transposed | 3 | 3 |
| per group (L = 100 / 286 / 2288 / 18304 / 73216) | 57 / 41 / 6 / 6 / 7 | 57 / 41 / 6 / 6 / 7 |

#### Wall time, cooled to 60 C before every run and the two arms interleaved

| threads | loom | onnxruntime | ratio |
|---|---:|---:|---:|
| 4 | 1.1384 / 1.1371 / 1.1235 s | 1.0501 / 1.0741 / 1.0588 s | **1.074x**, +78 ms |
| 1 | 2.9903 / 3.0057 s | 2.4545 / 2.5047 s | **1.209x**, +518 ms |

**The gap is thread-count dependent, which is new and is why the old reasoning no longer describes the
shipped configuration**: loom scales 2.64x over four cores, onnxruntime 2.34x, so more than half of
the single-thread gap closes on its own. P4.16 was scoped from a picture that is only true at one
thread.

#### The table, at ONE thread — because at four it cannot be built

| group | n | loom | GF/s | onnx | GF/s | ratio | excess |
|---|---:|---:|---:|---:|---:|---:|---:|
| resblocks 32ch @ L73216 | 7 | 718.4 ms | 6.3 | 560.5 ms | 8.1 | 1.28 | +157.9 |
| resblocks 64ch @ L18304 | 6 | 613.8 ms | 7.3 | 473.8 ms | 9.5 | 1.30 | +140.0 |
| resblocks 128ch @ L2288 | 6 | 332.0 ms | 6.8 | 231.2 ms | 9.7 | 1.44 | +100.9 |
| flow/encoder @ L286 | 41 | 567.5 ms | 7.5 | 487.4 ms | 8.7 | 1.16 | +80.0 |
| text encoder @ L100 | 57 | 214.7 ms | 6.3 | 166.1 ms | 8.2 | 1.29 | +48.6 |
| `CONV_TRANSPOSE_1D` | 3 | 223.1 ms | 6.7 | 269.8 ms | 5.6 | **0.83** | -46.8 |
| depthwise @ L100 | 12 | 7.2 ms | 0.2 | 1.5 ms | 0.9 | 4.72 | +5.7 |
| **convolution** | **132** | **2677 ms** | 6.9 | **2190 ms** | 8.4 | **1.22** | **+486** |
| everything else | | 321 ms | | 289 ms | | 1.11 | +32 |
| **wall** | | **2998 ms** | | **2480 ms** | | **1.209** | **+518** |

Arithmetic from `scripts/conv_census.py --syms n_tokens=100 --syms flow_vocoder:n_tokens=286`
(18.39 GFLOP). loom's column is the difference of two `$LOOM_PROFILE` runs — four calls minus two,
over two — so the cold first call is cancelled rather than averaged in; onnxruntime's is
`scripts/prof_onnx_conv_shapes.py`'s shares over its own un-profiled wall, rescaled to the
interleaved median.
Every row is **within 5% under either overhead model**, so at one thread the table is real.

**Rank it against the machine and the ranking dissolves.** Every dense row is 6.3-7.7 GFLOP/s for loom
and 8.1-9.7 for onnxruntime, against **~6.3 GFLOP/s per core** for a pure GEMM on this box (P4.15's
25.1 GFLOP/s at four threads). loom's convolution is *at* a single core's GEMM rate. onnxruntime is
above it, which is a real 1.2-1.4x, but it is 1.2-1.4x over a roofline-shaped floor, not over an
implementation with an obvious defect — and it is worth +486 ms only at a thread count nothing ships at.

#### Why there is no four-thread version of that table, and this is the reusable part

`$LOOM_PROFILE` computes **every node alone**. At one thread that costs 51 us per node — 108 ms on a
3.0 s wall, 3.6%, small enough to ignore. At four threads it costs **137 us per node, 291 ms on a
1137 ms wall — 3.7x the entire gap being measured.** And how that 291 ms is charged decides the
answer, not marginally but completely:

| 4-thread model | convolution | everything else |
|---|---|---|
| charge it per node executed | 931 ms, **1.20x**, +155 | 207 ms, 0.73x, -76 |
| charge it per millisecond (share x wall) | 757 ms, **0.98x**, -19 | 380 ms, 1.34x, +97 |

**A 230 ms swing on a 78 ms gap.** Neither model is right, and the reason is measurable: **135 ms of
the 291 is fusion the profiler cannot see** — `GGML_CPU_DISABLE_FUSION=1` costs 1.136 -> 1.265 s and
1.127 -> 1.267 s, and with `ggml-0005`/`ggml-0007` a fused convolution *absorbs* its bias and
activation, so under fusion "the convolution" is not a separable term at all. The rest is a
thread-pool barrier per node, which lands only on the ~190 node executions in the buckets that
measurably scale — the convolutions and almost nothing else. **The honest four-thread split** — taking the one-thread breakdown and applying the
measured per-bucket scaling, since the elementwise buckets measure 1.00-1.14x over four cores — is
convolution **~837 ms against onnxruntime's 776** (1.08x, +61) and everything else **~300 against 283**
(+17). Both halves are close, and neither has a defect in it.

#### What the elementwise half is actually bound by — measured, not assumed

`scripts/membw.c` (a `c[i] = a[i] + b[i]` over exactly the `ADD 73216x32` bucket's 8.9 MB tensor):

| threads | 1 | 2 | 4 |
|---|---:|---:|---:|
| GB/s (2r + 1w) | **4.56** | 4.13 | **3.64** |

The bus is saturated by **one** core and gets *worse* with four. loom's two largest `ADD` buckets run
at **98% and 99%** of that four-thread number, and its `LEAKY_RELU` — one read and one write rather
than two and one — moves 4.5 GB/s at both thread counts, which is what the bus gives a single core.
So the fact that they do not scale is the machine, not the engine, and no amount of threading
recovers it.

#### The one thing that is NOT against a roofline — now P4.25 — AND IT WAS A PROFILER ARTIFACT

> **Corrected 2026-08-30 by [P4.27](#p427--the-26-ms-that-arrived-as-5-ms-of-wall--closed-2026-08-30-it-was-never-on-the-table).**
> "30.4 ms at one thread and 30.4 ms at four" is what `$LOOM_PROFILE` reports for **any** op ggml
> declares `n_tasks = 1` for, at any thread count, because the profiler runs each node as a graph of
> its own and ggml plans a thread count per graph. The gate was threaded the whole time; removing that
> threading costs this board 2.5% of the synthesis. The `n_tasks` half of the paragraph below is
> wrong; the scalar-`tanhf` half (no SIMD path where GELU got one in `ggml-0010`) is still true and is
> still untried.

`ggml_get_n_tasks` (`ggml-cpu.c:2271`) hands `n_tasks = 1` to `TANH`, `SIGMOID`, `EXP`, `RELU`,
`LEAKY_RELU` and the rest of the cheap-unary list, while `GELU`, `GELU_ERF` and `SILU` get
`n_threads`; and `ggml_vec_tanh_f32` / `ggml_vec_sigmoid_f32` (`vec.h:909`, `:936`) are scalar
`tanhf` / `1/(1+expf(-x))` loops with no SIMD path — where GELU got one in `ggml-0010`.

VITS's flow has 17 `TANH` and 16 `SIGMOID` nodes on **220 KB** tensors, and they measure
**30.4 ms per synthesis at one thread and 30.4 ms at four** — 26 cycles per element, 0.46 GB/s, a
full order of magnitude clear of the memory roofline and unaffected by three idle cores. It is 2.7% of
the wall and **39% of the 78 ms gap**, with two independent untried levers (thread it; vectorise it).
That is the first named mechanism this thread has had since P4.15f, and it is not a convolution.

#### Also fixed here

`scripts/conv_census.py`'s `macs()` counted a **transposed** convolution over its output length. A
transpose scatters each of `il` inputs across `k` taps, so the count is `il * oc * ic * k`; over `ol`
it overstates by exactly the stride — 8x, 8x and 4x for VITS's three, which put **both** engines above
the machine's roofline on that row (149.9 and 56.8 GFLOP/s on a box that does 25) and would have made
the only row loom *wins* unreadable. The file's total for VITS goes 26.58 -> **18.39 GFLOP**. Node
counts and dense-conv arithmetic are unaffected, so nothing P4.15d concluded moves.

#### Reproducing all of it

**Both halves are in `scripts/` now.** `prof_onnx_conv_shapes.py` is new here — the per-shape half
existed only as `~/prof_onnx_shapes.py` on the Pi, which is precisely
[Retro-018](../retros/retro-018-a-table-of-ratios-nobody-could-re-derive.md)'s failure, and this
table would have been unreproducible for the same reason the last one was.

```sh
# loom, on a fresh Release build (the repo default is RelWithDebInfo and 1.39x slower)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 --target loom_engine
g++ -O3 -std=c++17 -I include -I tests/support -I build/_deps/ggml-src/include \
    -I build/_deps/nlohmann_json-src/single_include scripts/bench_vits_loom.cpp \
    -o bench_vits_loom -L build -lloom_engine -L build/_deps/ggml-build/src -lggml -lggml-base -lpthread
export LD_LIBRARY_PATH=build:build/_deps/ggml-build/src
LOOM_N_THREADS=4 ./bench_vits_loom <dir-with-vits_mil.gguf> 9          # wall
GGML_CPU_DISABLE_FUSION=1 LOOM_N_THREADS=4 ./bench_vits_loom <dir> 9   # what fusion is worth
# per-op: TWO runs at different call counts, so the cold first call can be differenced out
#   (nrun is MEASURED calls; the harness makes nrun+1, so these are 2 and 4)
for n in 1 3; do LOOM_N_THREADS=1 LOOM_PROFILE=prof_$n.txt LOOM_PROFILE_NODES=1 \
    ./bench_vits_loom <dir> $n; done
gcc -O3 -fopenmp -march=native scripts/membw.c -o membw && ./membw 4   # the streaming roofline

# onnxruntime -- SAME estimator on both sides (one warm-up, median of 9) or the ratio is meaningless
python3 scripts/bench_onnx_tasks.py vits <miro_en-GB.onnx> 4 9         # wall
python3 scripts/prof_onnx_conv_shapes.py <miro_en-GB.onnx> 4 5         # per shape
```

**Check the invariant before believing any ratio:** every harness above prints its sample count, and
all of them must say **73216**. A row whose two sides synthesised different utterances is
[Retro-010](../retros/retro-010-an-unpinned-competitor-baseline.md) again.

**Cool to a fixed temperature before every single run and interleave the two arms** — the board
reaches 75 C during one measurement, and two back-to-back runs have measured 33% apart:

```sh
cool() { until vcgencmd measure_temp | tr -dc 0-9. |
             awk -v t="${1:-60}" '{exit !($1 <= t)}'; do sleep 10; done; }
for i in 1 2 3; do cool 60; <loom arm>; cool 60; <onnxruntime arm>; done
```

With that discipline the three loom walls spanned 1.3% and the three onnxruntime walls 2.3%, which is
what makes a 6.9% gap readable at all on this machine. Alternating the arms without cooling is not
enough — one arm systematically gets the cool half of every thermal excursion.


### P4.26 — `ggml-0012` cost a Cortex-A72 2.4% on VITS — FIXED AND CLOSED 2026-08-30

**Found while re-measuring P4.16, from a 3% disagreement with the README's own Pi cell.** It is
[Retro-019](../retros/retro-019-a-patch-measured-on-one-isa.md)'s pattern one patch later — a patch
worth **2.75x** on x86 at the shape it was aimed at, checked on aarch64 **only at that same shape**,
and shipped — and the fix says the axis was never the ISA.
[Retro-022](../retros/retro-022-a-benefit-and-a-cost-on-the-same-axis.md) is the lesson.

**The predicate is `k`.** `ggml-0012` now reads

```c
const bool ragged_prefix = (m % 16 != 0) && params->nth > 1 && k <= 256;
if (m16 > 0 && (m16/16 >= params->nth) && (m % 16 == 0 || ragged_prefix)) { ... }
```

so an `m` that already divides 16 keeps the schedule it has always had at every thread count, and the
ragged prefix — the only thing the patch adds — is taken only where it pays.

#### Why `k`, and why it is not the ISA

The branch removes a **per-output** cost and adds a **per-work** one. Removed: `m*n` contended stores,
one per element of `C`, because at `BM = 1` a job owns 16 bytes of each column and four threads write
four quarters of one line. Added: a job's row block is four times taller, so it holds `RM*BM*k*4` =
`16k` bytes of `A` instead of `4k`, and there are four times fewer jobs to balance. The benefit
therefore decays as `1/k` while the cost does not.

`scripts/bench19.cpp` sweeps exactly that, at `m = 284, n = 384`, 4 threads, **both arms in one
process** through the run-time switch in `scripts/probes/ggml-p426-sgemm-policy-probe.patch`,
ABBA-interleaved, ratio per round (`> 1` means the branch beats the `m % 4` schedule it replaces):

| `k` | 64 | 128 | 192 | 256 | 384 | 512 | 768 | 960 | 1536 | 2304 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Core Ultra 9 285K | **2.035** | **1.317** | 1.046 | 1.030 | 1.005 | 0.996 | 0.987 | 0.983 | 1.000 | 0.974 |
| Cortex-A72 | 1.014 | 1.011 | 1.005 | 0.997 | 0.995 | 0.977 | 0.959 | **0.912** | **0.886** | 0.901 |

Monotone on both to within the witness's own resolution — the two rows that break it, `k = 1536`, are
1.000 and 0.886 against neighbours of 0.983/0.974 and 0.912/0.901 — and they cross 1.0 within a factor
of two of each other. **A `!defined(__aarch64__)` gate would have been wrong in both directions** — it would have kept a loss on x86 at `k >= 512` and
thrown away a win on aarch64 at `k <= 192`. 256 is the largest power of two inside the winning region
on both machines. The Pi's clock witness (a shape neither arm touches) held 0.992-1.027 across the
sweep; the 285K's held 0.995-1.150.

#### Why a patch written for attention reached a TTS vocoder

`m` is a sequence length in an attention matmul and `k` is a head dimension — small, which is the
regime `ggml-0012` was measured in. But `ggml-0004` and `ggml-0009` lower **convolution** and
transposed convolution through the same `sgemm`, and there `k` is `in_channels * kernel_width`.
`GGML_SGEMM_CENSUS=1` on one VITS synthesis (the probe patch prints it; the shape set is
**identical on both ISAs**, being a property of the graph) gives 22 distinct shapes, of which the three
ragged `m` — 100, 196, 284 — carry **10.9 of the synthesis's 14.4 GFLOP, 76% of all `sgemm` work**,
with `k` from 96 to 2304. Nothing in the patch's own reasoning mentions convolution, which is exactly
how this was missed.

#### Per shape, 4 threads, ratio against the pre-`0012` schedule

`scripts/bench19.cpp` mode 0, the VITS census shapes plus whisper's. `1.000` is the clock witness's
resolution on each machine.

| shape (`m`x`n`x`k`) | calls | Cortex-A72 `0012` | A72 **fixed** | 285K `0012` | 285K **fixed** |
|---|---:|---:|---:|---:|---:|
| 284 x 384 x 960 | 32 | **0.896** | 0.999 | 0.973 | 0.994 |
| 100 x 192 x 2304 | 12 | **0.889** | 0.996 | 0.926 | 0.984 |
| 100 x 768 x 576 | 12 | 0.929 | 1.002 | 0.982 | 1.001 |
| 284 x 384 x 192 | 24 | 1.009 | 1.013 | 1.068 | 1.025 |
| 284 x 192 x 192 | 8 | 1.021 | 1.018 | 1.092 | 1.098 |
| 284 x 192 x 96 | 8 | 1.027 | 1.036 | **1.672** | **1.687** |
| 196 x 100 x 96 | 24 | 1.022 | 1.018 | **1.476** | **1.466** |
| 100 x 100 x 96 | 24 | 0.877 | 0.892 | **1.367** | **1.321** |
| 100 x 192 x 192 | 76 | 0.976 | 0.978 | 1.007 | 1.023 |
| 284 x 96 x 192 | 8 | 0.930 | 0.933 | 0.980 | 0.976 |
| **1500 x 1500 x 64** (whisper `QK^T`) | — | 1.001 | 0.993 | **1.809** | **1.941** |

Weighted by the census, `ggml-0012` as shipped cost the Pi **+48 ms of `sgemm` per synthesis**; the
fixed predicate costs it **+1.0 ms**, and keeps every x86 win including the one the patch exists for.
The two shapes still slightly down on the A72 (`284 x 96 x 192`, `100 x 100 x 96`) are together 0.6 ms
and are the job-count effect rather than the footprint one — a job-count guard removes them and costs
x86 the two 1.3-1.5x wins above, which is a worse trade on both machines.

#### End to end

Paired ABBA rounds, both arms one binary switched by `GGML_SGEMM_POLICY`, `scripts/paired_arms.py`
(which grew an `--env` arm mode and a `--between` cooling hook for this), Pi cooled to a fixed 60 C
before **every** arm:

| | rounds | median ratio | p10 | p90 |
|---|---:|---:|---:|---:|
| Pi 4B, VITS, 4 threads — **fixed** / `0012` | **24** | **0.986** | 0.968 | **0.997** |
| Pi 4B, VITS, 4 threads — **fixed** / pre-`0012` | 12 | **1.003** | 0.985 | 1.018 |
| 285K, whisper, 24 threads — **fixed** / pre-`0012` | 15 | 0.956 | 0.829 | 1.108 |
| 285K, whisper, 24 threads — **fixed** / `0012` | 15 | 0.996 | 0.883 | 1.151 |

**The first row is the result and it resolves**: 24 paired rounds put the whole p10-p90 band below
1.0, medians 1.1364 s against 1.1158 s. **The second says parity with pre-`0012` is restored**
(1.1103 s), and it does not resolve, which is the correct shape for a null. The two 285K rows are
unresolved for a reason the README already documents — thread placement on that machine is chosen once
per launch and is bimodal — so they say only that nothing moved there. The case rests on the op-level
sweeps above, where the witness resolves to 1%.

`ctest -L ci` 74/74 and `-L gate` 83/83 with the fixed patch, on x86. **On aarch64 — which compiles the
*other* arm of the `#if`** — the tree builds 267/267 and one synthesis gives `fnv1a=aa320f8a1377a92a`,
**the same digest as every other predicate on that board**, which is what a scheduling-only change has
to show. The reference Pi cannot run `ctest -L ci` at all (its `python3` has no `gguf` module, so the
fixture generators fail and 49 tests never run); that is an environment gap on the board, unrelated to
this change, and it is the reason the digest is the aarch64 correctness evidence here.

#### What is left, and one thing that was not done

* **The Pi's README TTS cell is re-measured and reads 0.95x** (0.96x before the patch, 0.93x with it).
  Four rounds, both engines back to back, cooled to a fixed 60 C before every arm, arm order alternated,
  onnxruntime normalised to loom's sample count: loom 1.1118-1.1193 s against onnxruntime
  1.0552-1.0609 s, per-round ratios 1.002 / 0.950 / 0.948 / 0.947, **median 0.949**. The 1.002 is the
  session's first onnxruntime arm paying the 63 MB model's cold page cache; it is kept, because
  dropping the round that disagrees is how a number stops being reproducible.
  **Two things about that harness were wrong and are fixed** (`scripts/bench_onnx.py`): its model path
  was hardcoded to one Pi directory, and `LOOM_SAMPLES` — which normalises onnxruntime's time to loom's
  output length — still said 73472 where the current export gives **73216**, worth 0.35% in loom's
  favour. Both now come from the environment, and the second should be read off `bench_vits_loom`'s own
  output in the same session.
* **The Pi's LM and ASR cells did not move, and could not have.** whisper's `QK^T` is `k = 64` and
  measures 1.001 on this board with the patch and 0.993 with the fix (both inside the witness's
  resolution), its `A@V` is `m = 64` and takes the same branch either way, and a decode step's
  `mul_mat` has `ne1 = 1`, which `llamafile_sgemm` rejects at its second line — the same control that
  made `ggml-0011`'s diagnosis trustworthy. End to end, fixed against pre-`0012`, 4 paired rounds:
  **median 1.007, p10 0.989, p90 1.016 — unresolved**, on a 97 s transcription with no cooldown
  between arms. **Read the rounds, not the first one**: they run 1.018, 1.011, 1.002, 0.983, and the
  first two would have read as a 1.5% regression to anyone who stopped there. The `fixed` arm's own
  spread is 94.9-98.5 s against pre-`0012`'s 96.1-96.9, which is what an uncooled board looks like —
  the encoder `QK^T` bucket this branch touches is ~1.6 s of those 97, so it cannot produce a 1.5%
  wall difference in either direction. The LM, 6 paired rounds of 32 tokens through
  `infer_with_past`: **median 0.995, p10 0.983, p90 1.003 — unresolved**, 23.35 s against 23.41 s.

### P4.27 — the 26 ms that arrived as 5 ms of wall — CLOSED 2026-08-30: it was never on the table

**The premise was false, and the thing that made it false is loom's own profiler.** P4.27 was opened by
P4.25 to find where 26 ms of op-level saving went. It went nowhere: the VITS gate was threaded before
P4.25's patch, during it and after it, and the two arms of that model A/B ran the same code.
[Retro-023](../retros/retro-023-a-bench-whose-graph-was-the-treatment.md) is the lesson.

#### `ggml_get_n_tasks` does not decide whether a node threads. It decides whether a GRAPH does.

Three lines of ggml v0.19.0, none of which had been read:

* `ggml_get_n_tasks` is called in **exactly one place** — `ggml_graph_plan` — where it sizes the work
  buffer and feeds `max_tasks`, and then `cplan.n_threads = MIN(max_tasks, n_threads)`
  (`ggml-cpu.c:3018`). There is no per-node thread count anywhere in this version.
* `ggml_graph_compute_thread` runs **every node on every thread** with `params.nth = cplan.n_threads`
  (`ggml-cpu.c:3342-3372`).
* An op that must not split therefore says so **itself**: `ggml_compute_forward_sum_f32` and
  `ggml_compute_forward_leaky_relu_f32` open with `if (params->ith != 0) return;`. `apply_unary_op`
  does **not** — it splits over rows through `get_thread_range` (`common.h:74`), whatever `n_tasks`
  says about it.

So `n_tasks` is a graph-level clamp that can only take a whole graph **down** to one thread, when no
node in it declares more. `TANH` sitting beside a `MUL_MAT` has been threaded all along.

`scripts/bench20.cpp` is that in one table — it prints `ggml_graph_plan(...).n_threads` itself, so the
mechanism needs no timing — for a graph of 128 `TANH [286, 192]` nodes, and for the same graph with
**one** `MUL_MAT [32x8]` added, which is 0.03% of the work:

| machine | threads asked | TANH only: planned / us per node | + one MUL_MAT: planned / us per node |
|---|---:|---|---|
| Raspberry Pi 4B | 4 | 1 / 1108.6 | **4 / 281.1** (3.94x) |
| Core Ultra 9 285K | 24 | 1 / 129.3 | **24 / 27.2** (4.76x) |
| Ryzen 3 3250U | 2 | 1 / 942.2 | **2 / 546.2** (1.72x) |

And VITS's real graphs are not clamped: `LOOM_PLAN_PROBE=2`
(`scripts/probes/ggml-p427-graph-plan-probe.patch`) prints `ok: 1469 nodes, asked 24, planned 24` and
`ok: 651 nodes, asked 24, planned 24`. Across the whole gate suite — 83 tests, 13 real
checkpoints, run at ggml's default four threads with `LOOM_PLAN_PROBE=1` — **not one production graph
is clamped**, so the ggml wart costs loom nothing today (the probe's positive control is `bench20`,
which prints a clamp on every round). It is `bench18`'s
graph, and the profiler's, that were clamped.

#### What P4.25 actually measured, and what P4.16 actually saw

* **`bench18`'s 3.92x was its own graph.** 256 `TANH` nodes and nothing else plans one thread without
  the patch and `n_threads` with it, so its "1 thread vs 4 threads" comparison was really "this graph
  cannot thread vs this graph can". The op number is right and means something else than it was read
  to mean.
* **The model A/B was two identical arms.** VITS's graph plans `n_threads` either way, so 1.005x over
  twelve paired rounds is the correct measurement of no change at all. Nothing was absorbed between the
  nodes; the ggml-threadpool-sleeping suspect is not needed and there is no evidence for it.
* **P4.16's "30.4 ms per synthesis at one thread and 30.4 ms at four" is a profiler artifact, and it is
  the origin of the whole chain.** `profile::compute` runs each node as a graph of its own
  (`ggml_graph_view(graph, i, i+1)`, `src/core/profile.cpp:180`) — so under `$LOOM_PROFILE` every node
  whose op declares `n_tasks = 1` is *planned at one thread*, at any thread count. On VITS at 4 threads
  that is 122 `UNARY`, 126 `SUB`, 106 `SCALE`, 42 `SUM_ROWS`, 32 `LEAKY_RELU`, 12 `CLAMP` and 6 `SQRT`
  nodes, each timed on one core inside a report whose other buckets are threaded. "Identical at one
  thread and four" is what that looks like from outside.

This is a **second, different** failure mode from the per-node floor the profiler's header already
warns about. The floor is noise with a known sign; this changes what the code under measurement
*does*. Both are now in `include/loom/core/profile.h`, and `report()` prints the caveat next to the
floor so it cannot be read without it.

#### The measurement that settles it: take the threading away

The honest way to price something that is already happening is to remove it.
`scripts/probes/ggml-p427-graph-plan-probe.patch` adds `LOOM_UNARY_SERIAL=1`, which puts every row of
an `apply_unary_op` on thread 0 — what P4.25 believed ggml was doing. Paired ABBA rounds of the real
model, `scripts/paired_arms.py --env`:

| | rounds | serial / threaded | p10 | p90 |
|---|---:|---:|---:|---:|
| Pi 4B, VITS, 4 threads (cooled to 62 C before every arm) | 12 | **1.025** | 1.016 | 1.037 |
| 285K, VITS, 4 threads | 15 | **1.046** | 1.043 | 1.060 |
| 285K, VITS, 24 threads | 15 | 1.087 | 0.805 | 1.304 (unresolved) |

**Threading the transcendental unaries is worth 2.5% of a VITS synthesis on the reference Pi and 4.6%
on the 285K, and loom has been getting it since before P4.16 was written.** 2.5% of 1115 ms is
**28 ms**, against the **26 ms** P4.25 predicted from `bench18` — the arithmetic was right to within
the resolution of the board. It was a prediction of something already collected, and P4.25's patch
could not add it a second time.

#### What this closes, and what it leaves

* **P4.27 is closed with no patch**, and so is the "ggml's threadpool sleeps between multi-threaded
  nodes" hypothesis it was opened on — there was no gap to explain.
* **`cmake/patches/UPSTREAM.md`'s `nrows = 1` section was wrong in its remedy** and has been rewritten:
  `n_tasks = MIN(n_threads, ggml_nrows(node))` removes no barrier participation, because `n_tasks` does
  not gate barrier participation. The upstream-worthy finding is the graph-level clamp itself, and that
  `ggml_backend_sched`'s own eval-callback path has the same distortion any node-by-node profiler does.
* **A one-row unary still cannot split by rows** — that observation stands — but in a real graph the
  threads that get an empty range simply reach the per-node barrier they were going to reach anyway.
  The 31x Kokoro figure P4.25 recorded was measured through the profiler, i.e. through one-node graphs,
  and is not a production number.

### P4.25 — thread the unary gate — MEASURED OUT, CLOSED. The op is 3.92x; the model is not.

> **Read [P4.27](#p427--the-26-ms-that-arrived-as-5-ms-of-wall--closed-2026-08-30-it-was-never-on-the-table)
> first: this entry's premise is wrong, and P4.27 says why.** ggml has no per-node thread count —
> `n_tasks` clamps a whole GRAPH — so `TANH` was already threaded in every model graph, both arms of
> the model A/B below ran the same code, and `bench18`'s 3.92x is the difference between its own
> homogeneous graph planning one thread and planning four. The numbers below are all real; the two
> sentences that read them as "the op got faster and the model did not" are not. What survives is
> kept, corrected, in P4.27 and in [Retro-023](../retros/retro-023-a-bench-whose-graph-was-the-treatment.md).

**Executed 2026-08-30, and the patch is NOT carried.** The mechanism was real and every prediction it
made about the op held. It did not survive contact with the model, on any of three machines, and the
gap between those two statements is the useful part of this entry.

**What was built.** One patch to `ggml_get_n_tasks`, in two independent halves:

* **the list** — move `TANH`, `SIGMOID`, `EXP`, `ELU`, `SOFTPLUS`, `EXPM1` from `n_tasks = 1` to
  `n_threads`, leaving the arithmetic unaries (`abs`, `sgn`, `neg`, `step`, `relu`, `hardswish`,
  `hardsigmoid`, the rounding family) and `GGML_OP_LEAKY_RELU` alone;
* **the cap** — `n_tasks = MIN(n_threads, ggml_nrows(node))`, applied to every unary including the
  ones upstream already threads.

Both are gone from `cmake/patches/`. `scripts/bench17.c` and `scripts/bench18.cpp` stay, because a
negative result nobody can re-run is a rumour.

#### The op-level case, which is not in doubt

`scripts/bench17.c` (raw loops, both sizes a real model has, four threads against one, Cortex-A72):

| | tanh | sigmoid | exp | elu | expm1 | softplus | relu | neg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 220 KB, L2-resident | 3.98x | 3.98x | 3.98x | 3.96x | 3.99x | 3.98x | 1.23x | 1.31x |
| 9.4 MB, streaming | 3.99x | 4.00x | 4.00x | 3.86x | 3.99x | 3.99x | **0.77x** | **0.77x** |

A clean class split, and it justifies the list exactly: a transcendental unary is a libm call per
element and compute-bound at any size; an arithmetic one is memory-bound and, on this board, threading
it is a 1.3x **loss**, because one core already saturates the bus (`scripts/membw.c`: 4.56 GB/s at one
thread, 3.64 at four). That is also why `LEAKY_RELU` was left alone.

`scripts/bench18.cpp` runs the same question through **ggml's own threadpool**, 256 nodes in one graph
so the pool wakes once the way a real graph wakes it. At VITS's gate shape `[286, 192]`, four threads
against one on the Pi: **1107.8 -> 282.7 us, 3.92x.** The model issues 32 of those per synthesis, so
the arithmetic says **~26 ms of a 1130 ms wall, 2.3%**.

#### The model-level result, which is nothing

Twelve **paired ABBA rounds** — the ratio recorded per round, both arms the same harness, both Release,
same ggml commit and compiler, cooled to a fixed 60 C before every arm:

| board | rounds | median | mean | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| Raspberry Pi 4B, 4 threads | 12 | **1.0048** | 1.0028 | 0.9952 | 1.0099 |
| Ryzen 3 3250U, 4 threads (2 physical) | 6 | **0.9846** | 0.9867 | 0.9667 | 1.0123 |

**0.5% on one board, -1.5% on the other, both p-ranges straddling 1.0.** By this thread's own
standard — [Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md), "a p10 that
crosses 1.0 is weak, not 1.16x" — that is not a result. The precedent that *did* ship on a weak
resolution was P4.22 at **1.16x**; this is 1.005x.

Output is **bit-identical** throughout (`fnv1a` of the audio, now printed by `bench_vits_loom`): the
same digest at 1, 2 and 4 threads and across both builds, on both ISAs. `ctest -L ci` 74/74 and
`-L gate` 83/83 with the patch applied. Nothing here is a correctness question.

#### Two things it found on the way, and they are why the entry is worth reading

**1. ggml's unary ops split over ROWS, so a one-row tensor cannot split.** *(Corrected by P4.27: in a
real graph the threads handed an empty range simply reach the per-node barrier they were going to
reach anyway, and the Kokoro figure below was measured through the node-by-node profiler, i.e. through
one-node graphs, so it is not a production number. `n_tasks` cannot remove barrier participation,
because it does not gate it.)*
`apply_unary_op` takes its slice from `get_thread_range` (`ggml-cpu/common.h`), which divides
`ggml_nrows(src0)` by `nth`. At `nrows = 1`, thread 0 gets everything and threads 1..n-1 get an empty
range. The first cut of the patch had no cap, and **Kokoro — 870 `UNARY [256, 1]` nodes per synthesis,
a per-frame block — went 5.8 -> 179.9 ms on that bucket, 31x.** `bench18`'s row sweep is the shape of
it (`ne0 = 256`, Pi, four threads): `nrows` 1 / 2 / 3 / 4 / 8 / 16 / 64 / 192 gives **1.00 / 1.75 /
2.36 / 2.91 / 3.36 / 3.69 / 3.89 / 3.92x**. One row is the cliff; two rows already win. The cap fixes
it and is provably never worse — a thread with an empty range does no work and still costs its share
of the barrier — but **nothing shipped hits the cliff on an op upstream already threads** (the only
`UNARY` bucket in whisper, Qwen3-0.6B, LFM2 and the NeMo encoders is `[3072, 1500]`), so the cap alone
buys nothing today either. It is a real latent bug in ggml and it is written up in `UPSTREAM.md`.

**2. A wide machine does not rescue it, and adds a second floor.** At 24 threads on a Core Ultra 9
285K (measured through `SILU`, which upstream already threads, so it needs no patch) the same gate
shape gives **5.58x** — a bigger multiplier, and still worthless end to end, because that machine runs
the whole node in 28 us where the Pi takes 1108. The 32 gate nodes are ~0.9 ms of a 65-120 ms
synthesis there: ~1% before threading, ~0.7% after. **And 24 threads impose a hard ~1.9 us floor per
node**, so on that box threading is a LOSS below ~16 K elements — which VITS's own `UNARY [100, 192]`
(19 200) and Matcha's `[40, 256]` and `[20, 256]` (10 240, 5 120) sit at or under. A correct patch
therefore needs a work floor as well as a row cap, and the floor is machine-dependent (1.9 us at 24
threads against ~0.3 us at 4). That is a tuned constant in a carried patch, for ~0.5%.

#### The reusable lesson

**An isolated op measurement is an upper bound on what a model will see, never an estimate of it.**
Every number above the fold is right: the op really is 3.92x, the class split really is what
`bench17` says, the 26 ms really is what 32 nodes at 825 us each comes to. The model still moved
0.5%.

**This paragraph used to end "whatever absorbs it lives between the nodes, and ggml's threadpool
sleeping between two multi-threaded nodes is the named suspect". That was wrong, and P4.27 chased it:**
nothing absorbed it, because it had already been collected. The 26 ms is real and the model was
already banking it — taking the threading away with `LOOM_UNARY_SERIAL=1` costs this same board
**28 ms, 2.5%** — so a patch that switched it on again could only measure zero. The lesson that
survives is narrower and sharper than the one that was written here: **a bench whose graph is not the
model's graph can measure a property of the bench.** See P4.27 and
[Retro-023](../retros/retro-023-a-bench-whose-graph-was-the-treatment.md).

#### What would reopen this

A model where the transcendental unaries are a much larger share of the wall than VITS's 2.7% —
StyleTTS2 and Kokoro have 73 such nodes against VITS's 33, and neither was measured end to end here
because their fixtures are not on the reference board. **Measure the share first** (`$LOOM_PROFILE`,
one thread) and only build if it is worth more than five percent; and if it is built, it needs the row
cap, a work floor, and a number from **every** ISA and thread count it is enabled for — the three
machines here disagreed in sign, not just in magnitude.

### P4.13 — 2-D conv kernels, so a convolutional model can be Q4_0 — DONE 2026-08-30

**VITS exports at Q4_0 to 30.6 MB and still says the sentence.** 81.7 MB → 30.6 MB, coverage 0% →
73%, 114 conv kernels quantized where the previous build quantized none and printed a warning about
it. The audio transcribes through whisper-small as *"Hey, can you shut down the computer, my friend?"*
— which is the utterance, and which is more of it than the F32 arm's own transcript recovers (*"A. Can
you shut down the computer, my friend?"*: whisper mishears the leading "Hey" on the unquantized file).

**And it costs 2.08x in speed on x86-64.** That is the result this item could not predict and the one
that should govern whether anyone turns it on; it is decomposed below.

> **SUPERSEDED BY P4.29 (2026-08-31), and the rest of this section is the reasoning that led there.**
> The 2.08x is gone: a quantized conv model now runs at F32 speed (1.000x per sample on x86-64, 1.029x
> on a Cortex-A72). Everything below is still the right account of *why* it was 2.08x — it is what
> P4.29 was scoped from — but "read this before turning it on" no longer applies. Turn it on.

#### What the problem was, in one sentence

A convolution kernel is stored `[K, IC, OC]`, ggml lays quantization blocks along `ne[0]`, `ne[0]` is
the KERNEL WIDTH (1, 3, 5 …), and no block size divides that — so **no conv kernel was
block-quantizable as stored**, and P4.12's operand swap, which fixed the *op* gate, bought nothing on
its own. Measured on `vits-piper-en-gb-miro`: 117 distinct conv kernels, every one of them eligible by
op, **zero** of them block-alignable, and a Q8_0 export byte-identical to its F32 one.

#### What shipped, in two halves

**The exporter FOLDS an eligible kernel's spatial axes into `ne[0]`.** `[K, IC, OC]` becomes
`[IC*K, OC]`; `[KW, KH, IC, OC]` becomes `[IC*KH*KW, OC]`. `ne[0]` is then a channel count, which is a
multiple of 32 essentially always: **114 of VITS's 117 kernels align, 100.0% of the convolution
bytes** (the three that do not are `[1, 1, 192]` duration-predictor pre-nets holding 768 bytes each).
`LoomGGUFExporter._fold_conv_kernels_for_quantization` is the pass.

The fold moves no bytes. A C-order `(OC, IC, K)` array reshaped to `(OC, IC*K)` is the same buffer read
differently, and it is the *identical* reinterpretation `op_conv_1d` already performed on the kernel on
every call before handing it to the mul_mat. What the fold does is move that reshape from run time to
export time — which is why it is safe to do to a stored tensor and not only to a view of one.

**The engine takes the geometry back as attrs.** The fold erases the shape the convolution *is*, so
the node carries it: `kernel_k`/`kernel_ic` for CONV_1D, `kernel_kw`/`kernel_kh`/`kernel_ic` for
CONV_2D. OC survives as `ne[1]` and is deliberately not restated. `folded_kernel_geometry` in
`src/ops/primitives_conv.cpp` reads them, and **their presence is the signal, not the tensor's rank** —
which cannot tell a folded `[IC*K, OC]` from a declared `[K, IC]` with OC left implicit.

Feasible at all because **`ggml_compute_forward_im2col_f32/f16` touch `src1->data` only**: `src0`, the
kernel, is read purely for `ne[0]`/`ne[1]` (and `ne[2]` when `is_2D`), to size the patch matrix. What
im2col wants from a kernel is a *shape*, so it is given one.

#### Three departures from the sketch, each of which made it smaller

**1. The fold is quantize-only, so no export baseline moves.** The item predicted "the export sweep is
re-recorded: every conv model's tensor shapes change". It does not, because
`_fold_conv_kernels_for_quantization` returns immediately when the block size is 1 — no `--quantize`,
or F16/BF16. **Verified rather than reasoned: an unquantized VITS export is byte-for-byte identical
before and after this change** (`cmp`, on the whole 81.7 MB file). Every gate that runs on an F32
fixture is therefore untouched, and confirmed so — the four conv-heavy MIL driver gates (VITS, Matcha,
Kokoro, StyleTTS2) plus Whisper all pass unchanged.

**2. The shape carrier is the kernel MINUS its OC axis, which is why it is free.** The sketch's carrier
was a `[K, IC, OC]` leaf gallocr has to allocate — ~1.7 MB for VITS's largest kernel, and the item's
one real cost to decide. But im2col never reads `a->ne[3]`, and in the 1-D case never reads `a->ne[2]`
either: **OC is not part of the patch geometry.** So the carrier is `[K, IC]` (1-D) or `[KW, KH, IC]`
(2-D). VITS's largest is 3×512 floats — **6 KB**, and every carrier in the model together is under a
megabyte. The "carry a ggml delta with an explicit-dims `ggml_im2col`" fallback the sketch warned
against was never needed.

**3. The direct-conv conflict was a per-file choice, not a collision.** §2 recorded P4.13 as
*conflicting* with the direct-conv lowering (`ggml_conv_2d_direct`, ggml-0006) and said to pick one.
Both ship: an F32/F16 artifact keeps the direct lowering, a quantized one gives it up. It gives up
nothing it could have had — `ggml_conv_2d_direct` takes an F32/F16 kernel, so a quantized model could
not reach that path in any layout.

#### The fold, isolated from the quantization — and this is the measurement that proves it correct

The fold and the packing always ship together, so a difference in a Q4_0 file's audio cannot say which
of the two caused it. **Force the fold on an otherwise untouched F32 export and the question separates**
— same dtype, same bytes, same everything else, so the resulting waveform measures the fold and nothing
else. Eight lines, and worth re-running against any future change to either side:

```python
from loom_exporter.exporter import LoomGGUFExporter
from loom_exporter.main_export import main_export
_orig = LoomGGUFExporter.write_gguf
def patched(self, driver_script):
    self._fold_conv_kernels_for_quantization(32)   # `quantize` stays None, so nothing is packed
    return _orig(self, driver_script)
LoomGGUFExporter.write_gguf = patched
main_export(CKPT, OUT, task="text-to-speech", model="vits")
```

All 114 kernels fold; the file stays F32. Against the declared-layout export, over the same synthesis:
**identical sample count (73216), max abs diff 3.05e-05, rmse 4.07e-07, correlation 1.00000000.** That
3.05e-05 is one LSB of the 16-bit WAV the two arms were compared through — i.e. the two float waveforms
are the same waveform. **Every difference in a quantized file's output is the quantization, not the
fold**, and that is now a measured statement about a real 117-kernel model rather than an inference
from unit tests.

#### What it is worth, on four models

Coverage is the fraction of F32 weight BYTES the file quantizes — the number that says whether the file
actually moved, which a tensor count does not.

| model | quant | coverage before → after | file |
|---|---|---|---|
| vits-piper-en-gb-miro | Q4_0 | **0% → 73%** | 81.7 → **30.6 MB** |
| matcha-tts-ljspeech | Q8_0 | 21% → **90%** | 128.7 → **44.1 MB** |
| kokoro-82m | Q8_0 | 29% → **74%** | 325.5 → **149.2 MB** |
| styletts2-ljspeech | Q8_0 | 43% → **79%** | 411.0 → **172.4 MB** |

VITS is the extreme case and the reason the item existed: it is the one model where *nothing* was
quantizable before, because it has no MUL_MAT weight that is not a convolution.

#### The 2.08x, decomposed — this is what P4.29 recovered (and it is 2.13x on aarch64)

Ryzen 3 3250U (2 cores / 4 threads), 4 threads, `scripts/bench_vits_loom.cpp` with the three scales
pinned so both arms synthesise the same utterance, arms interleaved A-B-B-A over two rounds on an idle
box, best-of-7 per arm.

| arm | lowering | time | vs declared F32 |
|---|---|---|---|
| declared F32 | `ggml_conv_2d_direct` (+ bias and resblock fusions) | **0.50 s** | 1.00x |
| **folded** F32 | im2col + mul_mat | **0.78 s** | **1.55x slower** |
| **folded Q4_0** | im2col + mul_mat, packed weights | **1.04 s**¹ | **2.08x slower** |

¹ Q4_0 synthesises 67840 samples rather than 73216 — quantizing the duration predictor changes the
predicted durations — so its 0.96 s raw is normalised to the F32 arms' sample count. The two F32 arms
need no normalisation; they produce the same 73216 samples.

**Where the 1.55x goes, and it is not the arithmetic.** A quantized kernel cannot use
`GGML_OP_CONV_2D`, and three shipped patches hang off that op: ggml-0006's direct cache-blocked
convolution (4.7x on this machine's long-activation convs), ggml-0005's bias fusion, and ggml-0007's
resblock LEAKY_RELU + residual-ADD fusion. Folding costs all three at once. The remaining **1.33x** on
top is the quantized dot products themselves.

**So the trade on a convolutional model is ~2x slower for a much smaller file**, and after P4.28 below
that file is **11.7 MB against 81.7 MB, 7.0x**. That is a real choice and not a defect, but it is the
opposite of what quantization bought the LMs — qwen3 at Q8_0 got 1.22x *faster*, because a
transformer's weights were already in the operand ggml can read packed and there was no fused
direct-convolution path to give up. **Do not carry "quantization makes it faster" across from the LM
column.** Where it is worth it is a size-constrained edge target: VITS in 11.7 MB at half the speed is
a different product from VITS in 81.7 MB, and only the deployment knows which one it wants.

**And on aarch64 it is the same, which this entry predicted wrong (2026-08-31).** Raspberry Pi 4,
Cortex-A72, 4 threads, same harness and same two files, arms interleaved A-B-B-A over two rounds with a
20 s settle between them, best-of-5:

| arm | time | normalised to 73216 samples | vs F32 |
|---|---|---|---|
| declared F32 | **1.096 s** (73216 samples) | 1.096 s | 1.00x |
| folded Q4_0 | **2.166 s** (67840 samples) | **2.338 s** | **2.13x slower** |

Against x86-64's 2.08x, i.e. no better. The prediction was that the Pi would lose less because
`ggml_conv_2d_direct` is worth 1.18x there rather than 4.7x — true, and not the whole mechanism. A
Cortex-A72 loses less on the kernel and **more on the memory traffic**, because materialising the
im2col patch matrix (~550 MB written and read back per synthesis, §2) is precisely what that lowering
was adopted on that machine to avoid. The two cancel. **A cross-ISA prediction derived from one of two
opposing mechanisms is not a prediction.**

The run is clean rather than assumed so: the box went 52.5 °C -> 70.1 °C across it and the four F32
arms — one at each end and one in each round — agree within 2.5% (1.0957 / 1.1208 / 1.0976 / 1.1229 s
minima), so no thermal drift is hiding in the ratio, and each arm's `fnv1a` digest is constant across
all four repetitions.


#### What is deliberately not folded

* **Anything, unless the export is block-quantizing.** See departure 1 above.
* **The depthwise forms and SHORT_CONV.** Not a size trade-off, a correctness one: their mul_mat is
  BATCHED per channel over `ne[2]`, so flattening `ne[0..1]` turns a per-channel pairing into a cross
  product — a wrong answer, not a slower one. It would not pay anyway: a depthwise kernel is
  `[K, 1, C]`, so the fold puts K alone on `ne[0]` and K is 3.
* **A kernel any other node reads.** A weight that is also a MUL_MAT operand, or any node's second
  input, must keep the shape that consumer expects. There is no shape that satisfies both, so the fold
  declines rather than reconciles.
* **A kernel still unaligned after folding.** Folding it would cost the geometry and gain nothing.

The export's declined-for-shape line now reports exactly this remainder, and its wording says so.
Keeping the two gates — eligible-by-op and declined-for-shape — reported separately is what made the
original cause findable at all; do not let a future edit re-merge them.

#### What P4.13 left open, and what each of the five turned out to be — ALL CLOSED 2026-08-31

Every one was measured the day after. One changed a number in this entry, one changed a *prediction* in
it, and one became P4.28 below.

**1. The zero padding — FIXED, and it was not a quantization problem. → P4.28.** 62% of the Q4_0 file
was a constant-folded pad. Removing it takes VITS to **11.7 MB at Q4_0 and 62.8 MB at F32**, and the
coverage line from 73% to **95%**. Its own entry is below.

**2. Intelligibility on the other three families — ANSWERED, all three pass, and the correlations were
worthless.** Matcha, Kokoro and StyleTTS2 at Q8_0 each transcribe **word-for-word identically to their
own F32 arm** through whisper-small, on a real phonemized sentence rather than the gates' synthetic
token ids. StyleTTS2 at **Q4_0** passes too. The measurement is in the resolved flag at the end of this
section, and so is the thing it kills: **"deterministic ⇒ high correlation ⇒ fine" does not survive** —
Matcha's CFM is deterministic and its correlation still fell to 0.58 once 90% of its bytes were
quantized.

**3. The speed trade on aarch64 — MEASURED, AND THIS ENTRY PREDICTED IT WRONG.** See the table above,
which now has both ISAs: **2.13x on a Cortex-A72 against x86-64's 2.08x**, i.e. the same, where this
entry predicted the aarch64 arm would shrink. *(And P4.29 has since removed
both: 2.075x recovered on the Pi, 2.101x on x86-64.)*

**4. A folded kernel on a device backend — RUNS, and not vacuously.** Vulkan (Radeon Vega 3, RADV
RAVEN2), Q4_0 VITS: it synthesises, and whisper transcribes it identically to the CPU arm. The check
that makes that mean anything is `LoomLuaBridge::device_report()`, because a scheduler that handed
every quantized convolution back to the CPU would have produced exactly the same correct audio:
**`flow_vocoder` 771 device nodes / 0 fallback, `text` 1577 / 0.** Nothing fell back — not the shape
carriers, not the IM2COLs, not the kernel-first quantized MUL_MATs. (Those node counts are the 1.55x in
another form: the same graphs at F32 are 651 and 1469, and the difference is the im2col path replacing
one fused convolution node.)

**5. The depthwise exclusion — confirmed not worth revisiting**, as recorded above: 0.028 MB.

One thing the device arm exposed, worth keeping. **The Q4_0 duration predictor is accumulation-order
sensitive**: the same file synthesises 67840 samples on this CPU build, 67072 on the Vulkan build's CPU
backend and 67328 on its GPU, while the F32 file gives exactly 73216 on all three. VITS durations are a
`ceil()` of a float, so a 1-ULP difference flips one and changes the sample count. Harmless — every arm
transcribes — but **a quantized VITS has no cross-device bit-identity to assert**, and a gate that
assumed one would be wrong rather than strict.

#### Testing

`tests/ci/test_primitive_registry.cpp` gained three: `test_conv_1d_folded_kernel_matches_declared`
(three arms in one graph — declared F32, folded F32, folded Q8_0),
`test_conv_2d_folded_kernel_matches_declared`, and
`test_conv_1d_folded_kernel_rejects_inconsistent_geometry`. `tests/ci/test_quantize_export.py` gained
six on the exporter side, including the two declines above and the F32-export-does-not-fold claim.

**Verified by sabotage, which is the only reason to believe the exactness arms.** Replacing
`mul_mat_kernel_first`'s transpose with a bare reshape fails all three arms of the 1-D test, the 2-D
test, and P4.12's own `test_conv_1d_quantized_kernel_matches_f32` — the failure it exists to catch is a
right-shaped tensor full of the right numbers in the wrong places, which nothing else notices.

One thing the sabotage changed in the design: **swapping `kernel_k` and `kernel_ic` keeps their product
equal to `ne[0]`, so a kernel-side consistency check passes it** — and what it produced was a raw
`GGML_ASSERT` abort inside `ggml_im2col` naming no kernel, no attr and no model. `folded_kernel_geometry`
therefore also compares the declared IC against the *activation's* own channel count, which is what
turns that into a named `SchemaError`. That check exists because the sabotage found the gap, not
because it was designed in.

### Benchmarks on record (Ryzen 3 3250U, CPU, medians)

| model | quant | coverage | size | time vs F32 |
|---|---|---|---|---|
| qwen3-0.6b | Q8_0 | 100% | 2390 -> 640 MB | **1.22x FASTER** |
| styletts2 | Q8_0 | 43% | 411 -> 281 MB | 1.03x slower |
| matcha | Q8_0 | 17% | 129 -> 109 MB | 1.13x slower |
| vits | F16 | 67% | 81.7 -> 52.0 MB | **1.8x SLOWER**, cosine 0.999895 |

**These coverages and sizes are PRE-P4.13 and are kept only for the timing column** — the fold moved
every convolutional row of this table, and the current numbers are in P4.13's own table above.

Read these together before assuming Q4_0 will be fast: **integer quants and F16 behave oppositely
here.** Q8_0 sped qwen3 up (real integer SIMD vec_dot, activations quantized to match) while F16 lost
badly — this CPU has `f16c` (convert) and no native FP16 arithmetic, so every F16 dot converts to F32
first. That is a property of THIS box; expect it to invert on the RTX 5090 workstation, which is where
the GPU numbers should be taken and have NOT been. The TTS slowdowns also correlate with low coverage
on small compute-bound models — and P4.13's own measurement is what settled where that goes: at HIGH
coverage VITS was 2.08x slower, and the mechanism is the lost direct-convolution lowering rather than
the arithmetic. **P4.29 gave that lowering back to a quantized kernel, so the conv rows of this table
are stale in loom's favour** and want re-measuring before anyone quotes them.

### Do not spend time on K-quants

`Q4_K_M` is not a tensor type at all — it is a llama.cpp mixed-precision RECIPE. The real type `Q4_K`
exists but `gguf.quants` raises `NotImplementedError` for every K-quant (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K), so
this toolchain cannot write one. `main_export.quantize_choices()` (:24) derives the offered list by
probing the writer for exactly this reason. Writable today: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1,
Q8_0, TQ1_0, TQ2_0. K-quants also use block **256**, where — re-measured on the real folded layout now
that P4.13 exists — only **6 of VITS's 117** conv kernels align against block 32's 114, 17.9% of the
convolution bytes against 100.0%. They lose twice over. **Q4_0 is the target.**

### The quantized-TTS intelligibility flag — RESOLVED 2026-08-31, and correlation was never the test

**What it was.** StyleTTS2 at Q8_0 produced audio at correlation **0.015** against its own F32 audio
while transcribing correctly, and Matcha's deterministic CFM stayed at 0.985 — so the standing
hypothesis was that a *stochastic* sampler diverges onto a different-but-valid trajectory and a
deterministic one does not. P4.13 made the question urgent by taking Q8_0 coverage from 21% to 90% on
Matcha, 29% to 74% on Kokoro and 43% to 79% on StyleTTS2, which is to say none of those numbers
described a file the toolchain still produced.

**What the ASR oracle says, on a real sentence.** Each family phonemized with espeak-ng through its own
`tokenizer.ggml.tokens` table (Kokoro with the real `loom.default_style.ref_s` out of its own file, not
a synthetic style), F32 and Q8_0 arms of the same checkpoint, both transcribed by whisper-small. The
whole pipeline is three committed tools, because a verdict nobody can re-run is not one:

```
scripts/tts_ids.py "Hey, can you shut down the computer, my friend?" model.gguf ids.txt
./tts_synth model.gguf <family> ids.txt <rate> out.wav [--ref-s ids.txt.ref_s]   # scripts/tts_synth.cpp
# resample to 16 kHz, then:
./build/tools/loom_cli/loom_cli --model whisper-small.gguf --wav out16k.wav --language en
```

`tts_ids.py` carries the two facts that are easy to get wrong — a `bos`/`eos`/`blank` of **-1 is a
sentinel meaning "none"**, not an id, and Kokoro's `ref_s` is a **[510, 256] voice pack indexed by the
phoneme count**, not a vector. `tts_synth.cpp` carries the per-family `infer` signature and prints
`device_report()`, and its `synthetic:N` mode is the length sweep P4.28 used.


| model | quant | coverage | corr vs its own F32 | transcript |
|---|---|---|---|---|
| matcha | Q8_0 | 90% | **0.58** | *identical to F32* |
| kokoro | Q8_0 | 74% | **0.22** | *identical to F32* |
| styletts2 | Q8_0 | 79% | **0.025** | *identical to F32* |
| styletts2 | Q4_0 | — | — | *identical to F32* |

Every arm says "Hey, can you shut down the computer, my friend?" — the F32 arms included, word for
word. **All three families are fine at Q8_0**, and StyleTTS2 is fine at Q4_0 too.

**The hypothesis is dead, and that is the useful part.** "Deterministic ⇒ high correlation ⇒ safe" does
not survive contact with high coverage: **Matcha's CFM is deterministic and its correlation fell from
0.985 to 0.58** once 90% of its bytes were quantized rather than 21%. The correlation was never
measuring intelligibility; it was measuring how much of the model had been touched. Three families now
sit between 0.025 and 0.58 and all three are perfectly intelligible.

**And the oracle can fail, which is the only reason to believe it passed.** Matcha exported at
**TQ2_0** — ternary, the coarsest type this toolchain can write — synthesises audio whisper-small
transcribes as **"(water splashing)"**. A test that cannot tell a working model from a broken one is
not evidence; this one can, at the same sentence, through the same pipeline.

The standing rule is unchanged and now has a second demonstration behind it: **transcribe, do not
correlate.** [Retro-006](../retros/retro-006-kokoro-shipped-noise.md) is the case where correlation
0.996 shipped noise; this is the case where correlation 0.025 shipped speech.

### The Matcha gate's own number, kept because it is what looked alarming

`test_e2e_matcha_mil_lua_driver` compares against a frozen PyTorch reference waveform at a 0.02 bound,
and Matcha's Q8_0 export moves it from **max abs diff 0.0105 to 0.498**, on a reference whose own peak
is 0.332 and rms 0.044 — an error no longer small against the signal, and 48x the F32 arm's. That is
what raised the question above, and it is **not a failure**: the gate is an exact-fp32 comparison
against a real-module reference, which a quantized file was never expected to pass, and its inputs are
eight synthetic token ids rather than speech. The fold is not the cause either — see the isolated-fold
measurement under P4.13, correlation 1.00000000.

Kept because anyone pointing that gate at a quantized file will meet the number again, and because it
is a clean instance of the rule above: **a large numerical divergence from an F32 reference is not
evidence about intelligibility, in either direction.**

### P4.29 — a quantized direct convolution: the kernel is already repacked, so dequantize it there — DONE 2026-08-31

**Result. A quantized convolutional model now runs at F32 speed.** VITS at Q4_0 is **2.075x faster on a
Cortex-A72 and 2.101x on x86-64** than it was, which is the whole 2.13x / 2.08x P4.13 measured it giving
up — and against its own F32 export it lands at **1.029x / 1.000x per output sample**. **62.8 MB of F32
speed out of an 11.7 MB file.** The F32 path is bit-identical, on both ISAs and in both of its lowerings.

Shipped as `cmake/patches/ggml-0013-conv-quantized-kernel.patch` (+ `UPSTREAM.md` PR 13) and one branch
in `op_conv_1d`. Nothing in the exporter changed: the fold, the attrs and the file are exactly what
P4.28 left.

#### Where the 2.08x actually lived, measured rather than apportioned

Three arms on the same VITS synthesis, Ryzen 3 3250U, 4 threads, interleaved A-B-B-A over two rounds,
best-of-7. `GGML_CPU_DISABLE_CONV_HEURISTICS=1` is ggml-0004's own kill switch and is what separates
the middle arm from the first:

| arm | lowering | time | step |
|---|---|---|---|
| A | `GGML_OP_CONV_2D`, direct sweep + fusions | **0.50 s** | |
| B | `GGML_OP_CONV_2D`, batched im2col + fusions | **0.73 s** | **1.46x** — the direct sweep |
| C | loom's own im2col + mul_mat, no fusions | **0.78 s** | **1.07x** — the fusions and ggml's batching |

1.46 × 1.07 = **1.56**, against the 1.55x measured end-to-end in P4.13. The arithmetic closes, which is
the check that says these are three measurements of one thing rather than three numbers. The same A/B
on a Cortex-A72 (Pi 4B, 4 threads, A-B-B-A, best-of-3, 58 °C): **1.0930 -> 1.4554 s, 1.33x** — less than
x86-64's 1.46x, as the direct lowering's own per-shape table predicts, and still the largest single
piece.

**So the cheap-looking option was worth almost nothing.** Making ggml's *batched* path accept a
quantized kernel is nearly free — it already calls `ggml_compute_forward_mul_mat` through
`ggml_call_mul_mat_ldc`, and mul_mat has handled a quantized first operand forever — and it buys
**1.07x**. **The prize was the direct sweep (ggml-0006)**, which postdates the standing "ggml's own
fused conv does NOT help here, measured twice, closed" result in §4 — that closure was about the
batched path.

#### Why the hard-looking option was not hard: the sweep never reads the kernel as stored

`ggml_conv_1d_direct_run` **already repacks the whole kernel into an F32 scratch buffer** before it
computes anything:

```c
float * xp = (float *) params->wdata;      // [IC, LP], the zero-padded activation copy
float * wp = xp + (size_t) IC * LP;        // [IC, KW, OC], the packed kernel   <-- this
...
for (ic) for (kx) for (oc)
    wp[(ic * KW + kx) * OC + oc] = w[oc * (IC * KW) + ic * KW + kx];
```

The register-tiled kernel (`ggml_conv_1d_direct_tile`) reads `wp`, never `w`. So a quantized kernel
needs no quantized inner loop, no `vec_dot`, and no activation quantized to `vec_dot_type` — it needs
**one dequantize into the buffer that already exists**, and everything below is untouched: the tile
kernel, the bias fusion (ggml-0005), the LEAKY_RELU + residual fusion (ggml-0007), the phase-major
variant, the thread split.

**And it is more accurate, not less.** Dequantize-then-F32-FMA replaces a path that dotted Q4_0 weights
against activations rounded to Q8_0.

#### The design problem that was load-bearing: the fold and the sweep want opposite layouts

P4.13 folds a quantized kernel to `[IC*K, OC]` so its blocks align along `ne[0]`. `ggml_conv_2d_direct`
reads `KW`/`KH`/`IC`/`OC` **off `a->ne`**, and a folded kernel no longer carries them. Reshaping the
folded tensor back is not available: a `[IC*K, OC]` Q4_0 tensor reshaped to `[K, 1, IC, OC]` puts K=3 on
`ne[0]`, which is not a multiple of the block size, and the resulting `nb` is nonsense (the same hazard
the P4.13 shape carrier exists to avoid — and there it was safe only because `ggml_im2col` never reads
`src0->data`, which is not true here).

So ggml is **told** the geometry. `GGML_MAX_OP_PARAMS` is 64 bytes and `CONV_2D` used 24, so there was
room for three more int32s and no new op code, no new dispatch entry, no ABI change:

```c
ggml_conv_2d_direct_packed(ctx, a, b, s0, s1, p0, p1, d0, d1, /* kw */ K, /* kh */ 1, /* ic */ IC);
```

A zero `kw` in `op_params[6]` is what says "declared layout", which `ggml_new_tensor` gives every
existing graph for free.

#### THE PIECE WORTH KEEPING: dequantize at the top and RE-ENTER, rather than threading geometry down

The scoping note called risk 1 "the single most likely way to ship something slower than today":
`ggml_conv_1d_direct_ok` prices the kernel against a cache budget as `IC*OC*KW*sizeof(float)`, and `wp`
is F32 whatever the kernel is. Priced by 8.4 MB of stored Q4_0 rather than 59.5 MB of dequantized F32,
that predicate silently admits the shapes it was tuned to reject.

**It is not fixed by patching the predicate. It is fixed by making the stored size unreachable.**
`ggml_compute_forward_conv_2d_impl` dequantizes off the FRONT of the work buffer and then **calls
itself** with a stack `ggml_tensor` that is an ordinary contiguous F32 kernel in the declared layout,
and a `ggml_compute_params` holding what is left of the buffer:

```c
ggml_tensor kdq = *kernel;                       // F32, [KW, KH, IC, OC], data = the dequantized buffer
ggml_compute_params p = *params;                 // wdata advanced past it, wsize reduced by it
ggml_compute_forward_conv_2d_impl(&p, &kdq, src, dst, GGML_TYPE_F32, ...);
```

Every predicate below prices the dequantized size **because that is the only size it can see**, and the
two users of `wdata` cannot overlap by construction rather than by an accounting agreement (risk 2).
Threading `kw/kh/ic/oc` down through six functions would have left both risks live at every one of them.

*The general shape: when a representation change has to be invisible below some line, convert at the
line and re-enter, rather than teaching everything below to recognise both representations.*

#### The one new cost, measured

`scripts/bench21.c`, over all 59.5 MB of VITS's convolution weights at once — the whole model's
per-synthesis kernel packing, not one convolution:

| | dequantize | the F32 copy it replaces | delta |
|---|---|---|---|
| Q4_0 | **10.3 ms** (5.77 GB/s of output) | 8.5 ms | **+1.8 ms** |
| Q8_0 | **9.3 ms** | 8.4 ms | +0.9 ms |

**+1.8 ms against a 500 ms synthesis: 0.4%**, to recover 160 ms. The pack loop reads *less* memory in
the quantized case (8.4 MB of Q4_0 rather than 59.5 MB of F32) and writes the same 59.5 MB, which is why
the delta is arithmetic rather than traffic.

**The caveat that bounds where this design is legal.** It dequantizes **per call**. That is right for a
convolution that runs once per utterance, which is every convolutional model in this engine (TTS
vocoders, ASR encoders), and wrong for anything inside a decode loop, where it would be paid per token.
`SHORT_CONV` — the one convolution that *is* in a decode loop — is a different op, is not folded, and
must stay that way; see P4.13's "what is deliberately not folded".

#### What it measured

`scripts/paired_arms.py`, interleaved ABBA rounds, both arms one binary switched by `LD_LIBRARY_PATH`
over three shared libraries, the Pi cooled to a fixed 60 C before **every** arm. Both arms are the same
`vits-q4_0-dyn.gguf`; only the engine changes.

| VITS at Q4_0, 4 threads | rounds | before | after | median ratio | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| **Cortex-A72** (Pi 4B) | 7 | 2.197 s | 1.059 s | **2.075x** | 2.060 | 2.126 |
| **x86-64** (Ryzen 3 3250U) | 9 | 1.417 s | 0.681 s | **2.101x** | 1.927 | 2.441 |

The Pi's band is 3% wide and is the one to read; the x86 band is wide because that box is a noisy
2-core laptop part, and its quiet single-shot pair says the same thing (0.968 -> 0.446 s, 2.17x).

**The recovery is TOTAL, which was the claim most likely to be wrong.** The scoping note warned that if
the packed model landed nearer arm B than arm A, the heuristic was declining shapes it accepts for F32 —
risk 1. It lands on arm A. Same harness, the two exports of the same model against each other:

| VITS Q4_0 / its own F32 | rounds | q4 | f32 | ratio | **per output sample** |
|---|---:|---:|---:|---:|---:|
| Cortex-A72 | 7 | 1.060 s | 1.113 s | 0.950 | **1.029x** |
| x86-64 | 9 | 0.500 s | 0.537 s | 0.923 | **1.000x** |

The raw ratio understates because the two exports produce 67584 and 73216 samples — VITS durations are a
`ceil()` of a float and the arithmetic changed — so the normalised column is the honest one. Well inside
the 1.1x the item was accepted against, and **1.000x on x86-64 is arm A plus the dequantize**, which is
exactly what the design predicted.

#### The rest of the acceptance, in the order it was checked

* **The F32 path is bit-identical**, which is what a branch guarded on `ggml_is_quantized` has to show.
  `bench_vits_loom`'s FNV-1a over the whole audio, before and after, **and in both lowerings**:
  `68c3229eab373455` (x86, heuristic on), `bbb3b173ce35fe2a` (x86, `GGML_CPU_DISABLE_CONV_HEURISTICS=1`),
  `aa320f8a1377a92a` (aarch64) — all three unchanged, so no byte-identity baseline moved.
* **The kill switch still falls back cleanly.** `GGML_CPU_DISABLE_CONV_HEURISTICS=1` on the quantized
  model runs, produces the same 67584 samples, and is slower (1.27 s against 0.45), which is the
  escape hatch behaving as an escape hatch.
* **The audio still says the sentence.** whisper-small transcribes the Q4_0 output as *"Hey, can you
  shut down the computer, my friend?"* — **word-identical to the pre-P4.29 Q4_0 arm**, peak 0.156 and
  rms 0.0161 against 0.149 / 0.0160. Correlation against that arm is **0.054**, and that is the expected
  number rather than a warning: a 1-ULP duration difference shifts the whole waveform, and the
  before-arm's correlation against F32 was already **-0.008** when P4.13 shipped it. *Transcribe, do not
  correlate* — see the Retro-006 discussion above.
* **Vulkan is unchanged: 0 fallback nodes**, both topologies, on a folded Q4_0 VITS, and the audio
  transcribes. The loom-side branch asks `backend_can_run` (P4.7e) and keeps the im2col +
  `mul_mat_kernel_first` lowering when the answer is no, which on Vulkan it is — twice over, on a type
  test and on `cout == op->ne[2]`. **CUDA had to be told**: its `supports_op` for `CONV_2D` was
  `ggml_is_contiguous(src0) && ggml_is_contiguous(src1)`, which accepts a folded quantized kernel and
  then reads it as a wrong-shaped F32 one — a silent wrong answer rather than a failure. Now declined.
* **`test_conv_1d_folded_kernel_matches_declared` has a fourth arm**, and runs over **two shape sets**
  because the packed op has two lowerings inside it and the shape picks between them: `OC=5, IL=11`
  lands on the batched im2col, `OC=4, IL=64` on the direct register-tiled sweep (verified by
  instrumenting both). Arm four builds `ggml_conv_2d_direct_packed` directly rather than through the
  primitive — on a CPU the primitive already chooses it, so driving it through the op would only re-test
  arm three, and would stop testing anything the day `backend_can_run` says no. **Verified red by
  sabotage**: swapping `kw` and `kh` in the geometry reader keeps `kw*kh*ic*oc` equal to `nelements`, so
  every assert passes and only the numbers catch it — 4 checks fail. Swapping `kw` and `ic` does not
  reach a number; it trips `c_in == kernel->ne[2]` first.
* `ctest -L ci` **74/74**, `ctest -L gate` **83/83** on x86-64.

#### What is left

* **A second convolutional family at Q8_0 has not been re-run end to end.** P4.13's follow-ups
  transcribed Matcha, Kokoro and StyleTTS2 word-identically at Q8_0 through the *old* lowering; the
  mechanism here is per-node and model-independent, and VITS's 114 folded kernels already span both of
  ggml's lowerings, but a second family is what would say so rather than argue it.
* **`op_conv_2d` (the genuinely 2-D form) is untouched** and still takes im2col + `mul_mat_kernel_first`
  for a folded kernel. Nothing in tree has a quantized 2-D convolution hot enough to notice; the same
  `ggml_conv_2d_direct_packed` would serve it, with `kh > 1`, whenever one does.
* **`cmake/patches` reverse-check.** ggml-0013 edits `ops.cpp` inside ggml-0007's context, so that patch
  joins 0004-0006 in failing `git apply --reverse --check` on an already-patched tree, and every
  `cmake` re-run therefore takes `GgmlPatches.cmake`'s reset-and-retry path (which works, and rebuilds
  ggml). Pre-existing, now one patch worse, and cheap to fix by regenerating the earlier patches with
  more context.

### P4.28 — the relative-position pad: 18.9 MB of a VITS export was zeros — DONE 2026-08-31

**Found while closing P4.13, by asking what the 22.0 MB of F32 left in the quantized file actually
was.** Twelve tensors named `text.padded*`, `ne=[96, 4105, 1]`, **1.576 MB each and 99.78% zeros** —
864 real values out of 394080, contiguous in one band. 18.9 MB of a 30.6 MB Q4_0 file, and the same
18.9 MB of the 81.7 MB F32 one.

| | before | after |
|---|---|---|
| vits F32 | 81.7 MB | **62.8 MB** |
| vits Q4_0 | 30.6 MB | **11.7 MB** |
| Q4_0 coverage | 73% | **95%** |
| maximum utterance | ~2053 tokens | **no limit** |

**7.0x smaller than the F32 file this thread started with**, and the coverage line moves because the
*denominator* shrank: the tables were unquantizable F32 counted against every export's percentage.

**Why quantization could never have reached them, which is why this hid.** They are a VIEW's source,
and only a mul_mat's FIRST operand is eligible to be packed. The export's own coverage line counted
18.9 MB of zeros in its denominator and reported a healthy "73%, no warning" while 62% of what it wrote
was zeros. **A coverage percentage says what fraction of the bytes moved, not whether the bytes should
have been there at all.**

#### What they were

`_get_relative_embeddings` in piper's `MultiHeadAttention`: pad-and-crop of a learned Shaw-style table.
The engine has had a host-side port of it since the bespoke driver
(`src/core/relative_position.cpp::pad_crop_relative_embeddings`); the MIL trace does not use it, and
could not trace the real thing either, because **coremltools' torch frontend refuses a dynamic pad
amount on a rank>2 tensor** — a documented runtime limitation of that converter, not a gap in this
exporter. `vits_export.py` had routed around it the obvious way: pad by a *static* `_REL_EMB_MAX_PAD =
2048` on each side and slice dynamically, which is exact for any length up to the bound. The pad then
constant-folded into the weight.

#### The fix, and the three small pieces it needed

The pad is now a **CONCAT of dynamically-sized zero blocks** — the same trick `_dynamic_zero_pad_last`
already used in that file for a sibling problem, legal for the same reason: the frontend's restriction
is on `pad`, not on `cat` or on slicing.

The awkward part is worth keeping, because it is why those sibling helpers could not simply be called.
They build their zero block from a slice of `x` itself, which needs `x` to be at least as wide as the
block — and here `x` is the learned table, **9 columns**, while the block needed is `length - 5`. So
the zeros have to come from something whose extent already scales with the sequence, and
`_get_relative_embeddings` is handed `length` as a scalar and no tensor at all.
`_install_length_carrier` wraps `MultiHeadAttention.attention` — a **wrapper, not a transcription**, so
the rest of that method stays upstream's — to put `key` (shaped `[b, d, t_s]`) within reach.

Three pieces, each independently useful:

1. **`ValueFacts._scalar_entry` learned `clip`** (`value_facts.py`), resolving it to `sympy.Max`/`Min`
   so the clamp survives into the emitted shape expression instead of defeating the walk. A
   *derivation*, not a guess — unlike the `select` case beside it, which picks a branch on an
   invariant. It also has to ignore a `beta` of the float32 maximum, which is how `torch.clamp(x,
   min=0)` reaches MIL: a bound at the representable limit is not a bound.
2. **`shape_expr.render`/`parse` learned `Max`/`Min`** — two-argument only, because sympy's are n-ary
   and the engine's grammar is not.
3. **`symbol_env.cpp` learned `Max(a, b)` / `Min(a, b)`.** Both capitalisations, since sympy writes
   `Max` and a hand-written attribute would write `max`.

The exported shape is `2*Max(n_tokens - 5, 0) + 9` — literally "padded, but never by a negative
amount", where 5 is `window_size + 1`.

**One `+ 1`, which is not cosmetic.** `Max(length - 5, 0)` is 0 at `length <= window_size`, and a
zero-width block makes a zero-width VIEW, which the engine rejects outright. A single-phoneme utterance
(`[BOS, p, blank, EOS]`, four tokens) had worked and stopped. Padding by one extra row on each side and
letting `start` move with it keeps both of the real code's branches exact — checked against the
original export at lengths 2, 4, 6 and 62 — and costs 96 floats.

#### Verification, and it is exact

* **The audio is bit-identical.** The rewritten export and the original produce byte-for-byte the same
  waveform at T=62 — same 73216 samples, `np.array_equal` true. Not "close": identical.
* **Both branches, against the original file.** `n_tokens` = 2 and 4 (the crop branch, below the
  window) give the same sample counts and peaks as the pre-change export; 6, 22 and 62 likewise.
* **The bound is gone, not moved.** `n_tokens` = 2202 and 5002 now synthesise. On the old file both
  threw — loudly, which was the redeeming feature of the static pad: the engine's own VIEW bounds check
  caught the overrun and named the tensor, the resolved shape, the offset and the parent extent.
* **The exporter refused the intermediate wrong version by itself.** Before `shape_expr` learned `Min`,
  `render` raised `UnsupportedShapeExpression: Min has no equivalent in symbol_env.cpp's grammar`
  rather than emitting an attribute the engine could not read. That guard is why this change could be
  made at all without risking a silent wrong slice — the exact failure `value_facts.py`'s own docstring
  records from the last time this table's slice went wrong (silently ~34x too long at a real T=62).
* **No other model moved.** `clip`, `Max` and `Min` are general additions, so conformer-ctc and matcha
  were re-exported and diffed against the shipped artifacts: topology JSON and every tensor shape
  identical. The additions are inert for a model that does not need them.

#### The same question asked of every other model

Every export in `hf-models/` was swept for F32 tensors above 200 KB that are more than half zeros:

| model | F32 | zero-heavy | share |
|---|---|---|---|
| vits-piper-en-gb-miro | 81.5 MB | **18.92 MB** | **23.2%** |
| supertonic-2 | 266.5 MB | 2.30 MB | 0.9% |
| everything else (15 models) | — | ≤ 1.05 MB each | ≤ 0.1% |

VITS was the only real case. Supertonic's 2.3 MB (`ttl_text_512.emb_*`, 99.1% zeros) is the same
mechanism at a much smaller scale and **is not the same fix**: its text axis is statically sized on
purpose, for two independent reasons documented in `supertonic_export.py` — one of which is that
`GraphBuilder` resolves only one dynamic-length symbol per topology. That belongs with
[Retro-005](../retros/retro-005-supertonic-fixed-text-length.md), not here. Everything else the sweep
found is genuine all-zero bias vectors, which are real weights.

**The transferable part is how it was found**, and it is `scripts/weight_census.py`: bucket the tensors
a quantized file left as F32 by *what reads them and in which operand position*, not by name. The
buckets came out 86.8% "second operand / GET_ROWS / other" — and a bucket named for what it is *not* is
where a thing nobody is looking at hides. Point it at one file for the buckets and the zero-heavy list,
or at `hf-models/*/*.gguf` for the sweep above; on the fixed VITS the zero-heavy line is now 0.00 MB
and the conv kernels are 95% of what remains.

---
