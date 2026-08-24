---
type: epic
status: active
domain: performance
last_updated: 2026-08-24
---

# Epic-05: Edge CPU Performance

## 1. Context and Scope

The engine's stated target is edge devices, so the reference question is: **how does loom compare to
onnxruntime running the same checkpoint on a Raspberry Pi 4?**

The answer went from **2.2x slower** to **1.03x** over this thread, and every step of it was a
measurement. In scope: profiling infrastructure, kernel-level performance in the pinned `ggml`,
graph-level work reduction, and quantization for artifact size.

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

### P4.18: the ASR gap is entirely the ENCODER — SCOPED 2026-08-24, not started

whisper-small is the one task still behind onnxruntime (0.57-0.72x at four threads). The
cross-attention K/V export fix bought 2.4-2.6x and closed most of it; the question is what is left.

**Split it and the answer is not ambiguous.** whisper's onnxruntime export puts the encoder in its own
graph, so the two halves can be timed directly; loom's half comes from `$LOOM_PROFILE` at **one
thread** (see P4.14's floor trap), where `ne1 = 1500` is an encoder op and `ne1 = 1` is a decode step.
Same clip (`jfk.wav`, 11 s), same transcript, Core Ultra 9 285K:

| | loom | onnxruntime | |
|---|---|---|---|
| encoder | **5.91 s** (84.8%) | 2.38 s | loom **2.49x slower** |
| decode (25 tokens) | **0.65 s** (9.3%) | 0.97 s | loom **1.50x faster** |
| total | 6.97 s | 3.34 s | |

**The decoder is already ahead and needs nothing. 100% of the remaining gap is the encoder** — and
more than that, the gap (3.53 s) is *larger* than the encoder's whole runtime on onnxruntime. Anyone
picking this up should stop reading the decoder.

Note what this also says about the cross-KV fix: it did not merely improve the decode loop, it
**overshot it into a win**. The item that follows is a different problem in a different graph.

#### Where the encoder's 5.91 s goes

| op | shape | calls | ms | share of run |
|---|---|---|---|---|
| `MUL_MAT` | 768 x 1500 | 84 | 1941 | 29.2% |
| `MUL_MAT` | 64 x 1500 | 12 | 1062 | 16.0% |
| `MUL_MAT` | 3072 x 1500 | 12 | 813 | 12.2% |
| `MUL_MAT` | 1500 x 1500 | 12 | 755 | 11.3% |
| `CONT` | 1500 x 64 | **324** | 471 | 7.1% |
| `SOFT_MAX` | 1500 x 1500 | 12 | 396 | 5.9% |
| `UNARY` | 3072 x 1500 | 12 | 273 | 4.1% |

The 84 calls at `768 x 1500` are 12 layers x (Q, K, V, O, fc2) = 60, plus the 24 cross-attention K/V
projections the export now computes once. The 25 `NORM` at `768 x 1500` are 12 layers x 2 + the final
one — **checked, because 25 is also the token count and that coincidence looks exactly like the
cross-KV defect. It is not one; nothing encoder-width runs per decode step.**

#### Three candidates, cheapest experiment first

1. **Attention is materialised rather than fused.** `MUL_MAT 1500x1500` + `SOFT_MAX 1500x1500` +
   `MUL_MAT 64x1500` is **2.21 s, 31.8% of the whole run**, and it writes then re-reads a
   `1500 x 1500 x 12` F32 score matrix — **108 MB per layer**, twelve times, none of which fits in the
   36 MB L3. onnxruntime has fused `MultiHeadAttention`/`Attention` kernels that tile this and never
   materialise it. *Test:* take onnxruntime's own per-op profile (the recipe is in this epic's
   operating notes) and check whether those three ops appear at all or collapse into one fused node.
   That is a read of an existing profile, costs nothing, and decides whether this is the mechanism
   before any kernel is written.
2. **Layout churn around attention.** 324 `CONT` of `1500 x 64` is 471 ms of **pure data movement, 27
   per layer**, and copies are the one thing a graph can often be re-shaped to avoid. *Test:* dump the
   encoder topology and find which `permute`/`transpose` each `CONT` is servicing; ask whether the
   exporter can emit Q/K/V already in the layout attention wants. This is an **exporter** change if it
   works, which is where per-model complexity belongs (ADR-003).
3. **GEMM efficiency at the encoder's shapes.** `768 x 1500` alone is 1.94 s. P4.15 measured `ggml`'s
   F32 GEMM against MLAS at *VITS vocoder* shapes and fixed it there; these shapes are much larger and
   were never measured. *Test:* `scripts/bench6.cpp` with the seven shapes above, against the same
   MLAS baseline. **Do not assume P4.15's conclusion carries** — Retro-012 records that tinyBLAS now
   beats a hand-written 4x4 *at the shapes it was measured on*.

**(1) is the likely answer and (2) is the cheapest real change**; (3) is the one that would be
re-treading measured ground, so it should be entered last and only if the profile read in (1) says
attention is not the story.

**Not a gap, but worth knowing:** whisper pads every clip to 30 s, so an 11-second file pays the full
1500-frame encoder on **both** engines. That is not where loom loses, and shortening it is a
model-semantics change (the positional embedding is fixed at 1500), so it is not part of this item.

### Operating notes: benchmarking

**Machines.** The Pi is **`192.168.1.35`** — the `rpi4` name does not resolve. The workstation is
**`192.168.1.100`** (Intel Core Ultra 9 285K, 24 cores, 40 MB L2 / 36 MB L3, Debian, gcc 14.2); it has
no `cmake` on the default PATH (there is a `buildtools` micromamba env) and its `/home` runs at 99%.

`ssh pi@rpi4` — Raspberry Pi 4B rev 1.5, Cortex-A72, 4 cores @ 1.8 GHz, 1 MB shared L2,
32 KB L1D, LPDDR4, Debian aarch64, gcc 14.2 / clang 19. Repeatable to ~1% **when it is cool and nothing
else is on it**, and to about 9% when it is not. The dev box (Ryzen 3 3250U, AVX2, 2 cores, 4 MB L3) is
**thermally noisy** — pin with `taskset -c 0,2` and take medians of seven, or it will lie by 15%.

**Rules that were learned the hard way:**

* **Make both A/B arms the same binary**, switched at run time, and interleave them ABBA in both orders
  over two rounds.
* **Pin any stochastic sampler before quoting a ratio.** VITS's duration predictor is stochastic and
  the reference host does not seed it, so each run synthesises a different number of samples — see
  [Retro-010](../retros/retro-010-an-unpinned-competitor-baseline.md).
* **Normalise to output samples**, and scale a competitor's rows if its pinned length differs.
* **Rank by the machine's peak as well as by the competitor.** A row where both implementations are
  equally bad sorts to the bottom of a ratio table and can still hold the most time.
* **When a ratio is inexplicable by the kernel, count the nodes before profiling them.**
  `scripts/conv_census.py` needs neither a run nor the target hardware.

**Before opening a performance item, read
[Retro-012: Optimizations That Were Measured Out](../retros/retro-012-optimizations-that-were-measured-out.md).**

## 3. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-014](../adrs/adr-014-patch-ggml-rather-than-write-kernels.md), [ADR-017](../adrs/adr-017-no-k-quants.md) |
| Retros | [Retro-010](../retros/retro-010-an-unpinned-competitor-baseline.md), [Retro-011](../retros/retro-011-chasing-the-gemm-and-convolution-gap.md), [Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md), [Retro-014](../retros/retro-014-the-text-encoder-was-in-the-graph-twice.md) |
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
LOOM_THREADS=1 LOOM_PROFILE=1 loom_cli --model <gguf> --prompt "..." --n-predict 8
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
* `~/loom-p415/loom.cpp` — a full checkout with `prof_main` appended to its CMakeLists; this is what
  produces the end-to-end numbers. Rebuild with `cmake -B build -DCMAKE_BUILD_TYPE=Release` (**Release
  matters: the repo default is RelWithDebInfo and that is 1.39x slower**).
* `~/ggml-bench` — standalone benches with a stale ggml checkout of its own; `bench10` links against
  `~/loom-p415/loom.cpp/build/_deps/ggml-build/src`, so `LD_LIBRARY_PATH` must point there.
* `prof_main <gguf> <phonemes> <reps>` prints per-rep wall time; `LOOM_THREADS` sets threads;
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
`run_index`, so runs split by ORDER. `scripts/` has no onnx bench; the two used here are
`~/bench_onnx2.py` (wall) and `~/prof_onnx_shapes.py` (per-shape) on the Pi.

**The seven ggml patches** live in `cmake/patches/` and are applied at configure time by
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
   not: loom ran the text encoder twice, and P4.16's table was written, ranked and reasoned from
   before anyone counted the nodes (P4.15d). Counting them cost an afternoon and no measurement rig, and it moved the worst
   row in P4.16's table from 2.59x to 1.38x. `scripts/conv_census.py` is that count, for any GGUF.

**And the check that made this item honest: prove the gate can fail before believing it passed.**
Perturbing the fused slope by 5%, and separately the fused residual by 5%, both make
`test_e2e_matcha_mil_lua_driver` fail — so the 82 green tests are green about something. Fused and
unfused VITS output agree to 6.7e-8 max on a 0.17 peak, identically at 2 and 4 threads, which a race
would not do.


### P4.15e — `conv_transpose_1d`: a serial prologue and a dot-product compute — DONE (2026-08-22)

**What it was worth.** Two patches, `ggml-0008` (the prologue) and `ggml-0009` (the compute), on a Pi 4
at 4 threads: **1.314 -> 1.202 s, about 115 ms and 8.5%.** The op itself goes **195.8 -> 79.1 ms,
2.5x**, and against onnxruntime's 166.4 ms for the same three convolutions it ends up **2.1x faster**
where P4.16 had it 1.18x slower. Each half was measured on its own with the old path switchable at
runtime inside one binary, ABBA in both orders over two rounds: the prologue 1.314 -> 1.247 by mean
(~70 ms), the GEMM 1.252 -> 1.202 by mean and 1.239 -> 1.186 by min (~50 ms).

### Part 1: the prologue (`ggml-0008`, ~70 ms)

 The op goes from 196 ms to
**126.3 ms** re-profiled, which puts it **below** onnxruntime's 166.4 ms for the same three
convolutions — 1.32x FASTER, where P4.16 had it 1.18x slower. Per shape, before -> after:
73476x32 105.7 -> 63.3, 18376x64 53.1 -> 43.0, 2304x128 37.0 -> 19.9 ms. Measured by switching the old prologue back on at runtime inside one binary, ABBA
in both orders over two rounds.

**How it was found, which is the reusable part.** P4.16 put `CONV_TRANSPOSE_1D` at 1.18x and +29 ms —
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

### P4.16 — the convolution gap, shape by shape against onnxruntime — SCOPED, NOT STARTED

**Why this exists.** After P4.15b, loom is **1.24x** onnxruntime on the reference utterance (1.313 s
against 1.063 s, same boot, predictor pinned) and **everything outside the convolution is at parity or
better**: ~240 ms against onnxruntime's 282 ms. The whole remaining gap is convolution — 1103 ms
against 767 ms, **1.43x, +336 ms** — so this item is the only place left with anything in it.

Both engines profiled per-op on the same box and boot; onnxruntime's shares apportioned over its
un-profiled 1.044 s, loom's from `$LOOM_PROFILE` (which cannot see fusion, but fusion changes a
convolution's NEIGHBOURS, not the convolution, so the conv rows are valid). onnxruntime's activations
are 1.8% shorter because its pinned `y_length` is 282 against loom's 287 — scale its vocoder rows up
by that before splitting hairs; it moves each ratio by ~2%.

| group | loom calls | loom | onnx calls | onnx | ratio | excess |
|---|---:|---:|---:|---:|---:|---:|
| flow/encoder @ L~100 | 93 | 151.1 ms | 69 | 58.3 ms | **2.59x** ¹ | **+92.8** ¹ |
| resblocks 64ch @ L18368 | 6 | 216.8 ms | 6 | 144.9 ms | 1.50x | +71.9 |
| resblocks 32ch @ L73472 | 7 | 226.9 ms | 7 | 166.5 ms | 1.36x | +60.4 |
| flow/encoder @ L~287 | 41 | 203.4 ms | 41 | 155.6 ms | 1.31x | +47.8 |
| ~~`CONV_TRANSPOSE_1D`~~ | 3 | ~~195.8~~ **79.1 ms** | 3 | 166.4 ms | **0.48x** | **-87.3** |
| resblocks 128ch @ L2296 | 6 | 104.4 ms | 6 | 75.7 ms | 1.38x | +28.7 |
| **total** (after P4.15e) | **156** | **981 ms** | **132** | **767 ms** | **1.28x** | **+214** |

¹ **P4.15d took this row apart afterwards and P4.15f removed it: most of the +92.8 ms was one text
encoder run twice, not a kernel.** The group is now 57 calls, not 93 — at one thread it fell
376.7 -> 199.1 ms, **1.89x**, against the 1.91x arithmetic the census predicted, and end to end the
model went 1.196 -> 1.099 s (1.126x -> **1.033x** of onnxruntime). What is left in this row is the
~1.38x throughput gap, in line with every other row. **The whole table needs re-measuring against the
new export**; this is the only row that moves, but it moves from first to last. The discussion below
predates all of that and is kept for the reasoning, not the ranking.

`CONV_TRANSPOSE_1D` is struck through because **P4.15e did it**, and the way it fell is the warning
this table needs. It was ranked LAST here — 1.18x, +29 ms, the smallest row — and it turned out to hold
115 ms, more than any other row has yet given up. The ranking was wrong because it is a ratio against
onnxruntime, and both engines were sitting on the same floor: 7.3 and 8.6 GFLOP/s where the machine
does 25. **Rank by the machine's peak as well as by the competitor**, or a row where both
implementations are equally bad will sort to the bottom.

**The order this says to work in is NOT the order P4.15 worked in.** The vocoder resblocks — three
rows, +161 ms — are the ones P4.15 and P4.15b spent themselves on and measured out at 83% of the
machine's peak in cache. The **short, weight-heavy convolutions of the flow and encoder are +141 ms and
have never been touched**.

**The top row's 2.59x has since been taken apart, and it was not a kernel at all.** P4.15d's census
shows loom running **1.91x the arithmetic** there — the text encoder is in the graph twice — at 1.38x
lower throughput, and 1.91 x 1.38 = 2.64 against the 2.59 this table measured. So ~72 of the +92.8 ms
is duplicated work (**P4.15f** removed it) and ~21 ms is throughput, which puts this row in line with
every other row rather than at the top of the table. **The whole table still needs re-measuring against
the post-P4.15f export**: it is the only row that moves, but it moves from first to last.

The paragraph this replaces argued that 2.59x "cannot be a GEMM-throughput story" from the arithmetic —
onnxruntime runs `[1,768,103] x [192,768,3]` six times in 22.8 ms (546 MFLOP, **24 GFLOP/s**) and
`[1,192,103] x [768,192,3]` six times in 19.6 ms (28 GFLOP/s), against loom's in-model GEMM at **23.5
GFLOP/s** (P4.15), and a 1.1x arithmetic difference cannot produce 2.59x. That reasoning was right and
its conclusion — "the excess is around the GEMM, not in it" — was righter than intended: the excess was
not in this group's convolutions at all. **When a ratio is inexplicable by the kernel, count the nodes
before profiling them.**

**Both leads are now closed**, and with them everything this entry had a mechanism for:

1. ~~**`kw = 1` convolutions**~~ — **CLOSED by P4.15c, which measured it out.** The obvious fix was
   worth nothing, for exactly the reason that entry suspected: for `kw = 1` the im2col is a transpose
   and not a redundant copy, so a hand-written `mul_mat` lowering pays it too — 1.04-1.05x on the Pi,
   2.2 ms per synthesis. What is left is a layout question for the exporter worth at most 7.3 ms, and
   that bound ignores the transposes it would move onto the `kw = 3` convolutions beside them.
2. ~~**loom issues 153 `CONV_1D` where onnxruntime has 129 dense convolution nodes**~~ — **ANSWERED by
   P4.15d and FIXED by P4.15f.** onnxruntime has 117 dense convolutions plus 12 depthwise, loom had
   153, and the 36-node difference was one text encoder run twice. Every other shape matched one for
   one, and loom now issues 117 too.

**What is left of this item after P4.15c/d/f.** The two mechanisms it could name are spent, and the
model is at 1.033x of onnxruntime end to end — so the remaining rows are the three vocoder resblock
groups (+161 ms) that P4.15/P4.15b measured at 83% of the machine's peak in cache, plus the L~287
row at 1.31x. **Nobody has a mechanism for any of them yet**, which is the same state P4.15's warning
describes: onnxruntime being 1.4x faster there is a fact without a cause attached. Re-measure the
table against the post-P4.15f export before opening a new item on it.

**Checked and NOT a lead: depthwise.** The obvious suspicion — a `groups=192` convolution lowered as a
dense 192x192, which would be 192x the arithmetic — is wrong. The exporter emits **12 `CONV_1D_DW`
nodes**, exactly matching onnxruntime's 12 `[1,192,101] x [192,1,3]` calls — dilation for dilation, four
each at d=1, d=3 and d=9, per P4.15d's census — and `op_conv_1d_dw` runs them batched per channel. They cost loom 5.1 ms (the `IM2COL` row in its profile) against
onnxruntime's 1.1 ms — 4.6x, but +4 ms, so it is a rounding error in this item.

`CONV_TRANSPOSE_1D` at 1.18x (+29 ms) is the same item P4.15 already lists as "~60-90 ms, at rough
parity"; it is now the second-largest op in the profile and still lowered as a dense transpose.

**What NOT to do first.** Do not start on the resblock kernel again. P4.15 measured it at 83% of peak
in cache with the best of three tiles and no spills, P4.15b measured out both graph-level ideas that
were supposed to help, and onnxruntime being 1.4x faster there is a fact without a mechanism attached
yet — get one from the rows above, where the mechanism is visible, before spending another item on it.

**Reproducing the table.** The node lists, which are what the shapes below mean, come from
`scripts/conv_census.py` (P4.15d) and need neither a run nor a Pi. The times: loom
`LOOM_PROFILE=<path> LOOM_THREADS=4 ./build/prof_main <gguf> <ipa> 3`,
divide by the rep count. onnxruntime: the recipe and the pinning requirement are in P4.15b's cold
start; the per-shape aggregation is `~/prof_onnx_shapes.py` on the Pi, which groups `Conv`,
`FusedConv` and `ConvTranspose` events by `args.input_type_shape`.


### P4.13 — 2-D conv kernels, so a convolutional model can be Q4_0 — SCOPED, NOT STARTED

**Do this before P5.** Not because P5 depends on it, but because it closes the thread P4.12 opened and
the measurements below are fresh: the eligibility half shipped on
`loom.cpp feature/packed-conv-kernels` + `loom-exporter feature/phoneme-lexicon-model-cards`, and this
is the other half. Left alone, `--quantize Q8_0` on a convolutional model keeps reporting a 0% file
and the reason keeps having to be re-derived.

**One sentence.** A convolution kernel is stored `[K, IC, OC]`, ggml lays quantization blocks along
`ne[0]`, `ne[0]` is the KERNEL WIDTH (1, 3, 5 ...), and no block size divides that — so no conv kernel
is block-quantizable *as stored*, and the fix is to store it `[IC*K, OC]` and give the op the geometry
it loses.

### What already landed, and what it did not solve

Two independent gates decide whether a weight gets quantized, and only the first is fixed:

1. **Op eligibility — FIXED.** Only a MUL_MAT's *first* operand can be non-F32
   (`ggml_compute_forward_mul_mat` asserts `src1->type == GGML_TYPE_F32` for the operand it converts),
   and every conv kernel sat in the second. `primitives_conv.cpp` now branches on the kernel dtype:
   an F16 kernel keeps its slot and the **im2col follows it** (`conv_im2col_type`, :77 — legal because
   `vec_dot_type[F16]` is F16, and cheaper than converting the big operand per call); a block-quantized
   kernel moves to the first operand via `mul_mat_kernel_first` (:85) and pays a transpose back.
   `conv_kernel_is_packed` (:66) is the predicate. F32 runs the identical graph as before.
   The exporter's `PACKED_WEIGHT_FIRST_OPS` (`exporter.py:2438`) offers conv ops accordingly.
2. **Block alignment — OPEN, and this entry.** `exporter.py:2713` declines any tensor whose fastest
   axis is not a multiple of the block size. For a conv kernel that axis is `K`, so **every real conv
   kernel is declined** and Q8_0 changes nothing. F16 slips through only because its block size is 1.

The warning at `exporter.py` now reports the two separately ("N weight(s) WERE eligible by op and were
declined for shape") — do not let a future edit re-merge them, that wording is what made the cause
findable at all.

### The measurement that says it is worth doing

VITS (`vits-piper-en-gb-miro`, 81.5 MB of weights, 132 conv kernels holding 62.2 MB):

| stored as | aligned for block 32 |
|---|---|
| `[K, IC, OC]` (today) | **0 / 132** |
| `[IC*K, OC]` (proposed) | **117 / 132 = 100.0% of conv BYTES** (the 15 stragglers are rounding dust) |

Projected file: **81.5 -> 28.1 MB at Q4_0**, 35.8 MB at Q8_0. The same reshape helps every
convolutional family — conv kernels are 53-92% of the weight bytes in Kokoro/Matcha/StyleTTS2/
Supertonic and 126-303 MB inside the NeMo ASR encoders.

### Why it is feasible: im2col never reads the kernel

`ggml_compute_forward_im2col_f32` (ggml-cpu/ops.cpp) touches **`src1->data` only**; `src0` (the kernel)
is used purely for `ne[0]/ne[1]/ne[2]` to size the patch matrix. So the kernel passed to `ggml_im2col`
needs correct dimensions and nothing else — its contents are never read. That is the whole reason this
is a moderate change rather than a reimplementation of im2col.

### Design sketch

* **Exporter** — store an eligible conv kernel as `[IC*K, OC]` and put `k`/`ic` on the CONV node's
  attrs (the op currently recovers both from the kernel's own shape). Keep the 3-D form for kernels
  that stay F32, or make 2-D unconditional and let the op reshape back — decide by whichever keeps the
  gate baselines readable. `_collect_mul_mat_weight_names` already selects the right tensors.
* **Engine** — in `op_conv_1d`/`op_conv_2d`, when the kernel arrives 2-D, build a **shape carrier** with
  `ne = [K, IC, OC]` to hand `ggml_im2col`, then `mul_mat_kernel_first(kernel_2d, im2col_2d)` exactly as
  today. The kernel is ALREADY the shape the mul_mat wants, so this path gets *simpler*, not harder.
* **The one real cost to decide.** The shape carrier is a graph leaf, so `gallocr` allocates it even
  though nothing reads it — ~1.7 MB for VITS's largest kernel, reused across the graph, so peak is the
  largest single kernel and not the sum. If that is unacceptable on an edge target the alternative is a
  patched `ggml_im2col` taking explicit dims, which means carrying a ggml delta; do not start there.

### Acceptance

* `vits-piper-en-gb-miro` exports at Q4_0 to ~28 MB with a non-zero coverage line and no WARNING.
* Its audio still transcribes correctly through whisper-small (the ASR oracle; correlation alone is not
  enough — see P4.12, and see the StyleTTS2 note below).
* `test_conv_1d_quantized_kernel_matches_f32` (tests/ci/test_primitive_registry.cpp:774) still passes,
  and gains a 2-D-kernel sibling. It is the test that catches a wrong transpose — verified by sabotage:
  replacing the transpose with a bare reshape fails it, and produces a right-shaped tensor with the
  right numbers in the wrong places, which nothing else notices.
* The export sweep is re-recorded: every conv model's tensor shapes change.

### Benchmarks on record (Ryzen 3 3250U, CPU, medians)

| model | quant | coverage | size | time vs F32 |
|---|---|---|---|---|
| qwen3-0.6b | Q8_0 | 100% | 2390 -> 640 MB | **1.22x FASTER** |
| styletts2 | Q8_0 | 43% | 411 -> 281 MB | 1.03x slower |
| matcha | Q8_0 | 17% | 129 -> 109 MB | 1.13x slower |
| vits | F16 | 67% | 81.7 -> 52.0 MB | **1.8x SLOWER**, cosine 0.999895 |

Read these together before assuming Q4_0 will be fast: **integer quants and F16 behave oppositely
here.** Q8_0 sped qwen3 up (real integer SIMD vec_dot, activations quantized to match) while F16 lost
badly — this CPU has `f16c` (convert) and no native FP16 arithmetic, so every F16 dot converts to F32
first. That is a property of THIS box; expect it to invert on the RTX 5090 workstation, which is where
the GPU numbers should be taken and have NOT been. The TTS slowdowns also correlate with low coverage
on small compute-bound models, which is exactly what this entry raises — so Q4_0 on VITS is the first
case where a conv model gets high coverage, and its speed is genuinely unknown rather than predicted.

### Do not spend time on K-quants

`Q4_K_M` is not a tensor type at all — it is a llama.cpp mixed-precision RECIPE. The real type `Q4_K`
exists but `gguf.quants` raises `NotImplementedError` for every K-quant (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K), so
this toolchain cannot write one. `main_export.quantize_choices()` (:24) derives the offered list by
probing the writer for exactly this reason. Writable today: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1,
Q8_0, TQ1_0, TQ2_0. K-quants also use block **256**, where only 9/132 VITS kernels would align even
after the reshape — so they lose twice over. **Q4_0 is the target.**

### Unrelated flag found while benchmarking, worth its own look

StyleTTS2 at Q8_0 produces audio with correlation **0.015** against its F32 audio while transcribing
correctly through whisper-small. The plausible reading is its stochastic style-diffusion sampler
diverging onto a different-but-valid trajectory from small numerical differences (Matcha's
deterministic CFM stayed at 0.985) — but that is a HYPOTHESIS, not a verified result, and P4.12 is the
standing reminder that plausible-sounding TTS reasoning has been wrong before. Verify before shipping a
quantized StyleTTS2.

---

