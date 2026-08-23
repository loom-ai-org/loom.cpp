---
type: retro
date: 2026-08-21
domain: performance
tags: [gemm, convolution, micro-kernel, register-blocking, compiler-behaviour, arm]
---

# Retro-011: Chasing the GEMM and Convolution Gap

## The Issue

On a Raspberry Pi 4, VITS synthesised ~2.2x slower than the same checkpoint under onnxruntime. Three
explanations were argued from the code first, and **all three were wrong** — each measured at a few
percent or less: ggml not fusing conv+bias+activation (6.5%), the C++↔Lua array boundary (18.7 ms),
and `GGML_LLAMAFILE` being off in the shipped build.

## Root Cause Analysis

The real answer, once a per-node profiler existed to ask: **F32 micro-kernel quality**. ggml's generic
`mul_mat` reached 17% of the machine's fp32 peak where MLAS reached 45%, on identical arithmetic. Not a
missing kernel — a *spilled* one. `ggml_compute_forward_mul_mat`'s F32 path computed one output element
per `ggml_vec_dot_f32` call (1x1 register blocking), issuing two 128-bit loads per 128-bit FMA on plain
NEON, and an A72 is load-issue limited well before FMA peak.

## Resolution & Lesson Learned

Nothing in this repo computes a GEMM, and the hand-written 4x4 prototype
(`scripts/bench6.cpp`) **shipped nothing** — it was the measuring stick. What shipped was two patches to
ggml's own tinyBLAS, plus a direct convolution kernel behind a cache-size heuristic. See
[ADR-014](../adrs/adr-014-patch-ggml-rather-than-write-kernels.md) and
[Epic-05](../epics/epic-05-edge-performance.md).

* **Actionable takeaway 1 — reasoning from the code named three causes and got none of them.** Build
  the profiler first. `$LOOM_PROFILE` exists because argument-from-source lost three times in a row.
* **Actionable takeaway 2 — a per-node profile cannot see fusion, and cannot see inside an op.** It
  told us `ADD` cost 12%; it could not tell us that folding the bias in was worth taking. Phase-time
  inside the op when the op is the suspect.
* **Actionable takeaway 3 — GCC stops unrolling the moment the loop body has a branch in it.** Written
  the obvious way, sixteen vector accumulators moved to the stack in *every convolution in the model*
  and the change was a **19% regression** before it was a win. When a register-blocked kernel
  underperforms, check for spills before checking the algorithm.
* **Actionable takeaway 4 — the dev box lies by 15% and the Pi lies by 9% when warm.** Pin with
  `taskset`, take medians of seven, interleave A/B arms ABBA in both orders, and make both arms the
  *same binary* with a runtime switch.

---

## Full record (verbatim from the ledger)

### The 32- and 64-channel shapes: what limits them, and the 10% that was still there


These are the convolutions the phase window excludes and the ones the vocoder spends most of its time
in. Four measurements, in the order that made them answerable:

* **The machine's peak is 56.6 GFLOP/s**, 0.98 FMA per cycle per core, measured with a loop of nothing
  but independent 128-bit FMAs (`scripts/` has it as a comment; it is ten lines). The 57.6 this entry
  had been quoting from the datasheet is right. **Lane-broadcast FMA -- what the kernel uses -- runs at
  the same rate** (1.00/cycle), so the `fmla v, v, w[lane]` form costs nothing.
* **The kernel compiles well**: 22 instructions per 16 `fmla` in the inner loop, one `ldr q` plus
  `ldp`s, and **no spills** -- the trap that patch 0001 exists for does not recur here.
* **In cache, single-threaded, it reaches 83% of peak** (11.9 of 14.4). On the real shape it drops to
  **60%** single-threaded and **35-51%** at four. So what is left is not the loop, it is memory and
  scaling: 1 -> 4 threads is 2.35x, not 4x, on a chip whose L2 is shared.
* **The tile is already the right one.** Against the shipped 4-channels x 16-positions: 2x32 (half the
  weight traffic per FLOP, worse load/FMA ratio) is 20.1 vs 20.2 GFLOP/s at four threads and clearly
  worse at one; 4x8 is 15.6. Both lose at every thread count, so the obvious knob is turned correctly.

**And one thing was not.** The kernel handed out position blocks round-robin -- block b to thread
b % nth -- so every core's working set spanned the whole activation instead of its own share. Handing
out contiguous ranges instead is worth **10% on 32 channels x 73472** (39.3 -> 35.8 ms at four
threads), nothing on the shapes where the activation is short enough not to matter, and it is the same
total work either way. Shipped: **1.465 -> 1.440 s**.

**What is left on these shapes is not a kernel problem.** At 60% of peak single-threaded and 2.35x
scaling, the next lever is reducing how much memory the convolution touches at all -- which for a
resblock means keeping an intermediate activation in cache between two convolutions rather than writing
9.4 MB out and reading it back. That is a graph-level fusion, not a kernel, and nothing in this item's
approach reaches it.


### Fusing the bias into the convolution — and what a per-node profile does NOT tell you

The table above put `ADD` at 0.20 s, 12% of the synthesis, against an onnxruntime that folds bias and
activation into `FusedConv`. `cmake/patches/ggml-0005-conv2d-bias-fusion.patch` does the same thing:
ggml's CPU backend already has a graph-level fusion hook (`ggml_cpu_try_fuse_ops`, used for one pattern,
RMS_NORM + MUL), and this adds `CONV_2D` + per-channel `ADD` to it, adding the bias to each batch of the
result while that batch is still in cache. **The graph is unchanged** -- fusion is a decision the CPU
backend makes at compute time -- so no other backend, and nothing in the exporter, has to know.

**It is worth 1.8%, not 12%, and the difference is the point.** Measured against the same build with
`GGML_CPU_DISABLE_FUSION=1`: **1.605 s -> 1.576 s**. P4.14's profiler runs each node alone, and an
elementwise pass costs less as part of a graph than it does in isolation -- the fused kernel still has
to write the output, so what actually disappears is the ADD's read pass, not the whole node. Nothing
in the profiler is wrong; what is wrong is reading a node's isolated time as the time that would be
saved by removing it. **An op's profile time is an upper bound on its marginal cost, sometimes a very
loose one.** (The profiler also cannot observe fusion at all: it submits one node per graph, and a
one-node graph has nothing to fuse with -- so a profiled run is always the unfused one.)

**Two things had to be true before it worked at all, and the first one was found by the gates.**

* **The destination is usually the convolution's own input.** A graph allocator hands the ADD a block
  the input has just been freed from -- in the unfused order nothing reads that input by the time the
  ADD runs -- and in this vocoder it does that to EVERY large convolution. Writing the result there
  progressively, while later batches still need the input, corrupts it in a way that still sounds like
  speech: **max_abs_diff 0.54** on the Matcha and Kokoro lua-driver gates, which is what caught it. The
  kernel now stages each batch and lands it only after the NEXT batch's im2col has read what it would
  overwrite; batch k reads input from `s_k - pad` upwards, batch k-1's output covers `[s_{k-1}, s_k)`,
  and nothing at or after k ever reads below `s_k - pad`. Where a batch is shorter than the kernel's
  reach the fusion keeps the convolution's own destination and pays for one extra pass instead.
* **The staging buffer has to be channel-major.** Staged the way the existing permute path wants it --
  one row per patch -- landing it reads with a stride of `c_out`, and that gather costs more than the
  entire ADD being removed: **1.70 s, slower than not fusing at all.** Written `[c_out, patch_n]`
  instead, via the `ldc` mul_mat from patch 0004, each channel lands as a contiguous copy out of cache
  and the same code is 1.576 s. Two lines apart, 0.13 s of difference.

An earlier version simply declined whenever the destination overlapped the input. It was correct, and
it fired on nothing that mattered: **0.3%**.

**Pinned by `tests/ci/test_conv_bias_fusion.cpp`**, which asserts the numbers against a double
reference, asserts the two paths agree **bit for bit** (the registrations hand each other their output
through `LOOM_TEST_TMPDIR`), and asserts *that the fusion happened* -- by poisoning the convolution's
own result tensor, which a fused run never writes. It covers the aliased destination explicitly, and
was verified red for each: the detector forced to decline, and the aliasing guard removed (which is the
Matcha bug, reproduced hermetically at 1.3e-1). The whole VITS synthesis is byte-identical fused and
unfused on the Pi.


### Why the tile patch stopped at 92% of a hand-written kernel — and the second patch that answers it

The tile fix left ggml's tinyBLAS at 22.0 GFLOP/s against 24.4 for the standalone 4x4 prototype in
`scripts/bench6.cpp`. Chasing that 8% produced a second patch worth as much as the first, so the
sequence is worth keeping — every step killed a candidate rather than confirming one.

* **It is not per-call overhead.** `scripts/bench6.cpp` now measures the fixed cost of one
  `ggml_backend_graph_compute` (a 4x16x4 node, whose arithmetic is nothing) and subtracts it: **0.009
  ms** at 4 threads. That is 0.03% of these shapes, not 8%. Note this also retires a plausible misuse
  of P4.14's number: the ~1.4 ms floor recorded there is the cost of the PROFILING path, one compute
  per node with its own graph view, and says nothing about a normal compute.
* **It is not ggml's work partitioning.** `scripts/bench7.cpp` grew a driver that copies tinyBLAS's
  own scheme — jobs of `BM*RM` = 16 rows handed out from a shared atomic, instead of OpenMP's static
  split. Same tile, same buffers: **22.7 against 22.9 GFLOP/s**. The scheduling is worth ~1%.
* **At ONE thread the gap is WIDER** — 84% (6.3 vs 7.5 GFLOP/s) against 91% at four. Whatever it is,
  it is in the serial path, which is also what makes the next step cheap: read the object code.
* **It is 14 extra instructions per k-iteration, and they are all address arithmetic.** ggml's
  `gemm<4,3,4>` inner loop is **35 instructions** for 12 `fmla` + 7 `ldr q`; the identical source in
  `bench7.cpp` compiles to **21**. GCC does not form pointer induction variables over `l` when the
  bases and strides are read through `this` — it re-derives all seven operand addresses every
  iteration (8 `add` + 6 `lsl`). **Not the flags:** compiled with ggml's own
  (`-O3 -mcpu=cortex-a72+crc+nodotprod...`, lifted verbatim out of `flags.make`), the standalone copy
  is still 21.
* **Hoisting the bases and strides into locals fixes it**: 35 -> 21 instructions, and 22.0 -> **25.1
  GFLOP/s**, which is past the hand-written kernel (24.3 in the same process). Bit-identical output --
  it changes how an address is computed, not what is loaded or in what order.

**The two patches are not alternatives and neither works alone.** Measured through the dispatcher at
4 threads, each row normalised by the standalone kernel timed in the SAME process (which is how these
numbers survive a laptop-grade noise floor):

| | GCC | clang 19 |
|---|---|---|
| pristine v0.19.0 | 15.6 (0.65) | 23.8 (0.94) |
| tile only (0001) | 22.0 (0.90) | 23.6 (0.98) |
| address hoist only (0002) | 15.5 (0.76) | 24.0 (0.95) |
| **both** | **25.1 (1.03)** | 23.9 (0.95) |

Which settles the guards, both of which are now measurements rather than caution:

* **0001 is GCC-only because clang measurably does not want it.** clang 19 holds the 24-accumulator
  tile with zero spills (checked in the object code) and runs pristine at 23.8; giving it the smaller
  tile costs ~1%. The `!defined(__clang__)` in that patch is what keeps a clang-built wheel — every
  macOS one — on the schedule that is faster for it.
* **0002 is aarch64-only because x86-64 measurably does not want it.** There GCC already forms the
  induction variables, and the extra live values only add pressure: hoisting costs **55.4 -> 53.6
  GFLOP/s** (median of seven runs pinned to two physical cores) and puts 12 more `%rsp` reads in the
  block. With the `#if defined(__aarch64__)` in place, the x86 object code is **register-renaming
  identical to pristine** — which is a better check than re-timing it on a thermally noisy laptop, and
  is how the parity above was confirmed.

**One more thing this round paid for:** clang is now installed on both boxes (`clang-19` on the Pi,
`clang-14` on the dev box), so "we could not measure clang" is no longer a reason for anything.


### Three traps this cost real time to find (P4.15b)

1. **It was a 19% REGRESSION before it was a win, and the residual was not the cause.**
   `ggml_conv_1d_direct_tile`'s accumulators are a 2-D array indexed by the store loop's own counters.
   A compiler keeps such an array in registers only if it unrolls that loop, and GCC stops unrolling
   the moment the body has a branch in it. Written the obvious way — `if (res) …; if (whole) …` inside
   the store — sixteen vector accumulators moved to the stack in **every convolution in the model**,
   including all the ones with nothing fused into them, and the synthesis went **1.44 -> 1.73 s**. Both
   conditions are constant across a whole convolution, so they are template parameters and the call
   site picks one of four specialisations; the "nothing fused" one then compiles to exactly what
   `ggml-0006` shipped. This is P4.15's tile-spill failure in a different function — **when a change to
   a store loop makes the whole model slower, look for the accumulators before you look at the
   feature.**
2. **An idle check catches a hogged core, not a sampled one.** Seven orphaned poll loops from earlier
   sessions -- `until ssh pi@rpi4 'pgrep -f "cmake --build build"'`, whose pattern MATCHES THE POLLER'S
   OWN COMMAND LINE, so the condition can never flip and the loop runs forever -- were firing ssh at
   the Pi a few times a minute throughout this work, each stealing one of four cores for well under a
   second. `uptime` and `pgrep -a prof_main` before a run both read clean every time, because the
   interference is sub-second and intermittent; it took a 5 Hz sample over 100 s to see it at all. It
   plausibly explains the isolated outlier reps here (1.556, 1.667, 1.434 against a 1.31 median) and
   not the sustained thermal drift. Two lessons: **take the min of several reps, never a single one**
   (which is what every number in this entry does), and a poll loop's own pattern must not match
   itself -- `pgrep -f "[c]make --build"` or a pidfile.
3. **A `timeout`-killed background ssh does not kill the remote process, and the second run does not
   look broken — it looks slow.** An orphaned `prof_main` loop from an aborted measurement ran
   alongside a new one and produced 3.3 s where the truth was 1.35. This is already trap #2 in P4.15
   and it was walked into anyway. `pgrep -a prof_main` on the far side before believing any number,
   `pkill -9 -f prof_main` after any aborted remote run.

**And a measurement note that changed the answer.** The Pi is **not** repeatable to 1% under sustained
load: over twenty minutes of back-to-back runs the same arm drifts 1.45 -> 1.58 s, which is larger than
everything being measured. Rebuilding between arms confounds that drift with code layout. Both are
avoided by making every arm a runtime switch inside ONE binary and running them **interleaved, with the
order rotated between rounds and a cooldown before each** — four rounds here, and the ordering of the
four arms was identical in every one.

