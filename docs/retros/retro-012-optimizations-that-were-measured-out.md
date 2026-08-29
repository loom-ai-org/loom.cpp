---
type: retro
date: 2026-08-22
domain: performance
tags: [measure-before-building, negative-results, do-not-re-propose]
---

# Retro-012: Optimizations That Were Measured Out

## The Issue

A performance thread accumulates plausible ideas faster than it can test them, and a plausible idea
that was tried and rejected looks identical to one nobody has tried yet. Without a record, the same
proposals return.

## Root Cause Analysis

Each entry below was argued convincingly from the source before it was measured. The pattern that keeps
recurring: an idea's *mechanism* is real (im2col really does materialise a copy; a fused chain really
does avoid cache traffic) but the quantity it addresses is smaller than assumed, or the fix pays a cost
on the other side that cancels it.

## Resolution & Lesson Learned

**This file is the register. Check it before opening a performance item.**

| idea | verdict |
|---|---|
| a better GEMM micro-kernel | ggml's tinyBLAS now beats a hand-written 4x4; 25.1 GFLOP/s |
| a better direct-conv tile (2x32, 4x8) | both lose at every thread count |
| MLAS-style weight packing | the packing is not the lever; the gather was, and it is gone |
| `kw = 1` lowered as a matmul instead of a convolution | engine half worth 2.2 ms of a 1.099 s synthesis; exporter half at most 7.3 ms, and that bound is optimistic |
| tiling the resblock chain over the sequence | measured out — see below |
| `ggml_conv_2d_direct` in place of im2col + mul_mat | 0.98x on the engine of the day; **re-opened and repaired later**, see below |
| ggml not fusing conv+bias+activation | 6.5% of the whole unfused elementwise + activation chain |
| the C++↔Lua array boundary | 18.7 ms |
| depthwise convolutions lowered as dense | wrong — the exporter emits 12 `CONV_1D_DW` nodes, matching onnxruntime one for one |
| fusing `SOFT_MAX`'s five row passes into three | **CORRECTED 2026-08-24** — 1.06x on the 285K, but the reasoning that closed it was wrong and the dev box gets 1.16x; see below |
| a cheaper `exp` for `SOFT_MAX` | **CORRECTED 2026-08-24, then CLOSED 2026-08-25** — the probe that closed this in the first place was not a floor and the exp really is 1.1-1.4x of the row; but the exp has no accessible headroom, see below |
| specialising `ggml_v_expf` to `SOFT_MAX`'s domain (no overflow/subnormal path) | **1.00x** on the dev box, **1.13-1.16x** on the 285K, at identical accuracy — and the exp is 7-26% of the op. See below |
| an outer-product `QK^T` tile, lanes on `m` instead of on the contraction | **1.38x** on the 285K against a 1.5x gate, and its ceiling there was 1.52x before a line was written; 1.85x on a 2019 Zen+, 1.23x on NEON. See below |
| a cheaper `hsum` epilogue for tinyBLAS at `k = 64` | 1.06x, under the dev box's noise floor (P4.18) |
| rows-inner loop order, so a store finishes a cache line of C | **mechanism falsified**: no dependence on the size of C — see below |
| `ggml-0002`'s aarch64 address hoist applied to x86 | neutral at every k tested, small and large |
| *(NOT measured out)* tinyBLAS's `BM` row blocking at `k = 64` | ~1.15x paired at ONE thread, and 1.02x at k=768. **The "m=1500 cannot reach `BM = 4`" half was the bug, not a constraint** — at four threads that is 2.75x of false sharing, fixed in P4.22; see [Retro-020](retro-020-a-knob-measured-at-one-thread.md) |

### `SOFT_MAX`: closed on the right number for the wrong reason (2026-08-24, P4.18)

*(This section was written as "a real mechanism, on the wrong side of the bottleneck" and is kept
with its original reasoning intact, because the correction below is about the reasoning and a
rewritten section would hide what went wrong.)*

The mechanism was real and easy to confirm by reading `ggml_compute_forward_soft_max_f32`: **five
passes over every row** (copy to scratch, scale scratch, max-reduce, exp+sum, normalise) where three
suffice, and an AVX2 loop that **horizontally reduces its accumulator every 8 elements**. It profiled
at 4.3x onnxruntime. A 3-pass row with the scale folded into the exp and four accumulators reduced
once is an obvious, safe rewrite, and on the dev box it is worth 1.34x.

**On the machine that motivated the item it is worth 1.06x**, and two throwaway probes were read as
saying why before any of it was written as a patch — `1500 x 1500 x 12 heads`, one thread, Core
Ultra 9 285K:

| | one call | vs ggml |
|---|---|---|
| ggml, 5-pass row | 31.4 ms | — |
| 3-pass candidate | 29.6 ms | 1.06x |
| *probe:* 3-pass with a fast bit-trick exp | 32.3 ms | 0.97x |
| *probe:* 3-pass with **no exp at all** | 31.6 ms | 0.99x |

**That was the conclusion, and the two probes it rests on are not floors.** Both were written as
plain scalar C and left to the auto-vectoriser, and both were compared against a candidate written in
hand intrinsics — so what they measured was two compilers' output, not two amounts of work.

#### The correction (2026-08-24, dev box)

Re-run on a Ryzen 3 3250U with the floor arm built the *same way* as the arm it bounds —
`cand_soft_max_row<0>`, the identical function with `ggml_v_expf` switched off at compile time and
nothing else changed — and with the arm the original never had at all, a `memcpy` of the same bytes:

| `1500 x 18000` rows, one thread, Ryzen 3 3250U | one call | x12 calls | |
|---|---|---|---|
| ggml, 5-pass row | 56.8 ms | 682 ms | — |
| 3-pass candidate | 49.2 ms | 590 ms | **1.16x** |
| *floor:* same arm, exp switched off | 34.6 ms | 415 ms | the exp is **1.42x** of the candidate |
| *floor:* `memcpy` of the same bytes | 14.5 ms | 174 ms | 13.9 GB/s |

**ggml's row body is 3.9x the memcpy floor, so this op is not bandwidth bound**, and the tell was
visible in the old probes all along: on this box the "no exp" arm came out at 47.2 ms against the
candidate's 42.6 — *slower than the thing it claimed to bound*, which cannot happen if it is a floor.

The **number** for the 285K stands: the 3-pass candidate really is 1.06x there, measured directly.
What does not stand is the **reason** — "bound by streaming 540 MB through one core" was inferred from
the probes rather than measured, and the memcpy arm that would have tested it costs ten minutes and
was never run. On the machine where it *was* run, the bytes are a quarter of the time and the exp is
another third.

**So: `SOFT_MAX` stays closed on the 285K on its own 1.06x, and is OPEN on any machine where the
fusion is worth more.** The item to re-open is not a soft_max rewrite; it is `ggml_v_expf`, which the
corrected floor prices at 1.4-1.6x of a fused row — the same shape of finding as `ggml-0010`'s GELU,
in the same file.

#### The takeaway, restated correctly

**A floor arm must be built the same way as the arm it bounds.** "Probe the floor before optimising
the middle" was right and is still right; what this retro got wrong is that a probe written in a
different style is not a floor, it is a second candidate. Two rules follow:

* **Delete the expensive part from the REAL arm** — a template parameter, a `#if`, a lambda — rather
  than writing a simpler program that does the same job. If the floor comes out slower than the thing
  it bounds, that is not a surprising result, it is a broken probe.
* **For "it is memory bound", the floor is `memcpy`.** Not a cheaper kernel, not a deleted operation:
  the actual bytes, moved by the actual machine, in the actual working-set size. It is three lines,
  and it is the only arm that can support the claim.

**And the caveat this leaves on P4.18's table:** the 4.3x against onnxruntime was read as suspicious
on the grounds that a memory-bound op cannot be compared. It is not memory bound, and the comparison
is fine — onnxruntime's own per-node profile on the dev box puts its 12 `Softmax` at 515 ms against
loom's 682, which is 1.24x and sits almost exactly on the 3-pass candidate's 590. It was never a rate
"no single core reaches by streaming"; it is a core doing less work per element.

### `ggml_v_expf`: the item the correction opened, and what closed it (2026-08-25, P4.18)

The correction above ended by naming the next item: *"The item to re-open is not a soft_max rewrite; it
is `ggml_v_expf`, which the corrected floor prices at 1.4-1.6x of a fused row — the same shape of
finding as `ggml-0010`'s GELU, in the same file."* That was the right item to open and it is now
closed, on a floor argument rather than on one candidate's number.

**What looked specialisable.** `ggml_v_expf` is ARM's optimized-routines `expf`: correct for every
float, including the ones whose result is subnormal or infinite. It pays for that with a mask
(`|n| > 126`), a `movemask`, and a branch around a slow path that rebuilds the scale in two halves.
SOFT_MAX's argument is `x - max <= 0` and its result is in `(0, 1]`, so **that path can never be
reached** — the argument only needs clamping where the answer stops being a normal float.

`scripts/bench12.cpp` carries `sm_expf`: ggml's own fast path, verbatim, with the mask/movemask/branch
replaced by one `max`. Same range reduction, same degree-5 polynomial, same `2^n` by integer add, so it
isolates the general-case handling and nothing else. One thread, 131072 floats in `[-30, 0]`, ABBA-paired,
both arms writing the same destination:

| | Ryzen 3 3250U | Core Ultra 9 285K |
|---|---|---|
| `ggml_v_expf` | 0.69 ns/elem | 0.17 ns/elem |
| `sm_expf` | 0.71 ns/elem | 0.16 ns/elem |
| **above the load/store floor** | **0.97-1.02x** | **1.13-1.16x** |
| max rel err vs a double `exp` | 1.847e-07 *both arms* | 1.847e-07 *both arms* |

**And the floor that closes it for every other candidate too.** The same function again with the
polynomial replaced by `exp(b) ~ 1 + b` — six of its fourteen operations gone, and an answer wrong in
the second decimal place — runs at **1.64-1.95x**. Nothing that still computes an exp can beat that, so
**1.8x is the ceiling on any exp rewrite at any accuracy**, and a shippable one gets a fraction of it.

Put that against what the exp is worth inside the op, which is where the two machines part company:

| 12 calls of `SOFT_MAX 1500 x 1500 x 12`, one thread | Ryzen 3 3250U | Core Ultra 9 285K |
|---|---|---|
| ggml, 5-pass row | 600 ms | 398 ms |
| 3-pass candidate | 508 ms (1.19x) | 365 ms (1.08x) |
| *floor:* same arm, exp off | 372 ms — **the exp is 1.37x** | 338 ms — **the exp is 1.08x** |
| *floor:* `memcpy` of the same bytes | 168 ms (14.4 GB/s) | **51 ms (47.4 GB/s)** |
| ggml above the memcpy floor | 3.6x | **7.8x** |

**So the exp is 26% of the op on the dev box and 7% on the 285K**, which is the whole reason the 3-pass
fusion is 1.19x on one and 1.08x on the other: the 285K's cores are fast enough at the arithmetic that
what is left is the traffic. Even a *free* exp would be 1.37x / 1.08x on `SOFT_MAX`, which is 5.9% of
whisper's encoder — **at most 2% and 0.5% of a transcription.** The realistic candidate returns 0-16%
of that.

**The memcpy arm the correction asked for has now been run on the 285K, and it agrees.** Retro-012's
original reading — "bound by streaming 540 MB through one core" — is wrong on *both* machines, not just
the one where it was first disproved: ggml's row body is 7.8x the memcpy floor there, further above it
than on the dev box.

#### The takeaway

**A "specialise the general-case kernel" item needs the floor priced before the candidate is written.**
This one had a real mechanism (a mask, a `movemask` and a branch that provably cannot fire), a real
motivating number (the exp is a third of the op), and a working patch — and it is worth nothing,
because the general-case handling was never where the operations were. Fourteen of ggml's sixteen
operations survive the specialisation. **Count the operations you are removing against the operations
that remain, before measuring; a two-of-sixteen removal cannot be a 1.4x.** The `1 + b` floor is what
turns that from an argument into a bound, and it is three characters of `if constexpr`.

*Do not re-propose a faster `exp` for `SOFT_MAX`.* If SOFT_MAX is ever worth attacking again, the
target is the **pass structure** — 287 ms of the 285K's 398 — and even that is capped by the 3-pass
candidate's own 1.08x there.

### The batched tile epilogue: a mechanism found and a patch declined (2026-08-29, P4.18 item C)

**This one is different from everything else in this file: it works, and it is measured, and it is
still not worth carrying.** Recorded so it is re-proposed as a priced option rather than as an idea.

`tinyBLAS`'s `gemm_bloc` ends by horizontally reducing `RN*RM` vector accumulators to scalars and
storing each on its own. Counted with `perf` on a 285K P-core, that epilogue is a **`k`-independent
18.0 instructions per output element** — `instructions/output = 0.2355k + 18.0`, within 1.5% over
`k` in [32, 768] — which is 9% of the work at `k = 768` and **54% at `k = 64`**, the shape of whisper's
`QK^T`.

Batching the reduction over the row dimension (three `vhaddps` + an extract + an add + one 16-byte
store for four contiguous outputs, as an overload on `<4, __m256, float>` so nothing else changes)
removes **exactly 6.7 of those 18 instructions, at every `k` measured**. What that buys:

| | |
|---|---|
| 285K, k=64, 9 ABBA rounds | **1.085x** (p10 1.021) |
| 285K, k=768 | 1.022x (p10 1.016) |
| Ryzen, k=64, 9 ABBA rounds | 1.023x, **p10 0.982 — unresolved** |
| over the encoder's real bucket mix | **~2.8%**, i.e. 1-2% of a transcription |

* **Actionable takeaway — a 20% instruction cut bought 8.5% of the time**, so "retirement-bound" was
  only half true: three dependent `vhaddps` at 6-cycle latency serialise part of what they save. If
  this is ever revisited, an unpack/shuffle reduction tree has the same instruction count with a
  shorter dependency chain, and that is the version to measure.
* **Declined on the same grounds as `SOFT_MAX`'s pass fusion** — a ggml patch carried forever for 1-2%
  end to end, resolvable on one of the two x86 machines and not the other. Two items now closed on
  size rather than on being wrong; that is the bar this thread has settled on.
* **And it re-priced an older verdict.** The line above that reads "a cheaper `hsum` epilogue, 1.06x,
  under the noise floor" was measuring the right thing on a box that could not resolve it. The
  mechanism was real all along; what was missing was a machine with counters.

### The outer-product tile: the mechanism was real, the ceiling was not (2026-08-29, P4.21)

**The cleanest example in this file of an idea that was right about the cause and wrong about the
size**, and the only one so far killed by a number about the MACHINE rather than about either kernel.

P4.18 counted `tinyBLAS`'s per-output horizontal reduction at a `k`-independent **18.0 instructions per
output**, 54% of everything the core retires at whisper's `QK^T` (`k = 64`). An outer-product tile
never reduces per output, so the item scoped a standalone gate: clear **1.5x** at `m = n = 1500,
k = 64, 12 heads`, transpose counted, or stop. `scripts/bench16.cpp` is that gate.

**It did exactly what it was supposed to do mechanically.** Instructions per output on a 285K P-core:
**36.2 -> 19.7**, a 1.84x cut, the intercept gone. **And it bought 1.38x**, because:

| | 285K P-core | Ryzen 3 3250U | Raspberry Pi 4 |
|---|---:|---:|---:|
| one-core FMA roofline | 177.2 GF | 54.6 GF | 14.3 GF |
| `tinyBLAS` dot | 116.6 GF (**66%**) | 23.9 GF (44%) | 7.45 GF (52%) |
| outer product, pack counted | 161.3 GF (**91%**) | 44.3 GF (81%) | 9.18 GF (64%) |
| ratio | **1.38x** | **1.85x** | **1.23x** |
| what ANY perfect kernel could be | **1.52x** | 2.24x | 1.93x |

* **Actionable takeaway — an instruction ratio is not a time ratio, and the way to know in advance is
  to divide the incumbent's throughput by the machine's roofline.** IPC went **5.02 -> 3.63** between
  the two arms: the horizontal reductions were being issued in slots the FMA ports were leaving idle,
  so deleting them freed instructions and not cycles. The 1.76x ceiling in the item's own scope was
  `(0.2355k + 18.0) / 0.2355k`, and on the box it was measured on the real ceiling was **1.52x**. One
  cheap probe — a dependency-free FMA loop — would have priced the item before it was scoped.
* **The intercept was an ISA artifact and nobody had noticed.** `hsum(__m256)` in `sgemm.cpp` is six
  instructions; `hsum(float32x4_t)` is `vaddvq_f32`, **one**. So "every output ends in a horizontal
  reduction" is true everywhere and expensive only where the vector is wide. That is the Pi's 1.23x,
  and it is the second time this thread has been caught extrapolating one ISA's kernel result to the
  other ([Retro-019](retro-019-a-patch-measured-on-one-isa.md)) — this time before shipping rather
  than after.
* **The trend runs against re-opening it.** The dot-product kernel reaches 44% of roofline on a 2019
  Zen+, 52% on a Cortex-A72 and 66% on a Lion Cove P-core. The wider and more out-of-order the core,
  the better it hides the epilogue and the less this is worth. The Ryzen's 1.85x and the 285K's 1.38x
  are two points on that line, not two boxes disagreeing.
* **The pack was NOT the problem**, which is worth saying because it was the suspected cost: 1.0-1.5%
  of the packed arm, panel-major. Nor was the tile shape (six swept per ISA, spread 1.1-1.2x) or the
  loop order — though **column blocking is worth 1.50x -> 1.80x on the Ryzen and 1.005x -> 1.23x on
  the Pi**, so a first-cut kernel with a naive loop order would have failed the gate for the wrong
  reason. Sweep the blocking before believing a negative.
* **And the experiment paid for itself anyway**, which is the argument for cheap standalone gates: it
  found that ggml's F32 GEMM **uses 3.65 CPUs to deliver 1.02x** at this exact shape, because `BM`
  leaves each job writing 16 bytes of a 64-byte line of `C`. Four rows of padding on `m` are worth
  **2.75x**. That is P4.22 in [Epic-05](../epics/epic-05-edge-performance.md), and it is a bigger
  number than the item that found it.

### A dev-box noise floor that ate two verdicts (2026-08-24, P4.18)

Chasing `QK^T` at `k = 64`, the same binary reported **27.9 GFLOP/s and then 12.6** for the same shape
twenty minutes apart, on a 2-core laptop still warm from a `ctest` run. Two verdicts were written from
inside that noise before it was characterised, and both were wrong:

* **"rows-inner loop order is worth 1.26x"** — it appeared once, in one standalone run, with a
  plausible mechanism (a 4-float store finishes a 64-byte line of C) and a control that fitted
  (the gap vanishes at k=768). Re-run with C shrunk until it fits in cache and the flops held equal,
  the ratio went 1.07 / 1.28 / 0.96 / 1.04 / 0.96 / 0.98 — **no dependence on the size of C**, which
  is the one thing that mechanism has to show. Implementing it in ggml measured -6%.
* **"`BM` row blocking has no effect"** — three shapes, one run, 27.2 / 25.7 / 27.7 GFLOP/s. A later
  run of the same three gave 17.8 / 22.4 / 26.5, a 1.49x spread the other way. Neither was the answer.

**What worked was a paired test** (`scripts/bench14.cpp`). Two arms back to back inside one round, the
*ratio* recorded per round, 31 rounds, reported as a median with p10/p90. Clock drift moves both halves of a pair together
and cancels; two independent minima do not. It resolved `BM = 4` at **1.16x (p10 0.95)** with a k=768
control at **1.02x** — a real but partial effect, weakly resolved, which is a different verdict from
either of the first two.

* **Interleaving arms is not enough; pair them and take the ratio.** Min-over-interleaved-rounds still
  compares two independent minima drawn from different parts of a thermal excursion. A per-round
  ratio does not.
* **Publish the noise floor with the result.** The witness spread on this box is 1.4-2.5x. Anything
  under ~1.2x is not measurable here, and saying so is more useful than a number that will not
  reproduce. A p10 that crosses 1.0 is "weak", not "1.16x".
* **A falsified mechanism outlives a noisy number.** The useful half of the loop-order result is not
  "-6%"; it is that C's size does not matter, which closes the store-traffic explanation for good
  however the timings land next week.

* **Actionable takeaway 1 — a verdict has a shelf life.** The im2col item was closed at 0.98x with
  "do not re-propose", and that verdict was *right for the engine it was measured on*. Once the GEMM
  got 1.7x faster, `IM2COL` went from a modest slice to **40% of all convolution time** and the item
  was correctly re-opened. Record what a verdict was measured *against*, not just the number.
* **Actionable takeaway 2 — measure before building is worth its own item.** P4.15c exists entirely to
  say "do not build this", and `scripts/bench9.cpp` keeps the table re-runnable so the next person can
  re-check rather than re-argue.

---

## Full record (verbatim from the ledger)

### P4.15c — `kw = 1` is a matmul, and loom runs it as a convolution — MEASURED OUT, CLOSED


**The answer: the engine half is worth 2.2 ms of a 1.099 s synthesis, and the exporter half is worth
at most 7.3 ms — and that bound is optimistic.** This item did what it told itself to do: it measured
before building, and the measurement says do not build. `scripts/bench9.cpp` grew the table below and
keeps it re-runnable.

**Four lowerings of the same arithmetic**, over every pointwise convolution the post-P4.15f VITS export
actually issues (66 calls, weighted by count from `conv_census.py`), all four **bit-identical**
(max relative difference 0.0e+00, so any of them is a legal substitution):

| | | Pi 4, 4 threads | Pi 4, 1 thread | x86, 2 threads pinned |
|---|---|---:|---:|---:|
| **A** | `ggml_conv_2d_direct`, KH=1 — **what runs today** | **45.74 ms** | 143.91 ms | 61.08 ms |
| **A2** | `ggml_im2col` + one `mul_mat` (`-DLOOM_CONV1D_DIRECT=0`) | 45.82 ms | 147.85 ms | 57.94 ms |
| **B** | `cont(transpose(x))` + `mul_mat` — "it's just a matmul" | 43.58 ms | 137.92 ms | 53.30 ms |
| **C** | bare `mul_mat`, activation already `[IC, L]` | 38.43 ms | 121.11 ms | 50.04 ms |
| | **A/B** (the engine question) | **1.05x** | **1.04x** | 1.15x |
| | **A/C** (the exporter question) | 1.19x | 1.19x | 1.22x |

**A/B is 1.04-1.05x on the target machine, which is inside the 10% this item set as its own threshold
for "this belongs in loom-exporter".** Rewriting `op_conv_1d`'s `kw == 1` case as a transpose plus a
GEMM would buy **2.2 ms per synthesis, 0.2%**. That is the whole engine-side idea, priced.

**Two of the three leads are dead, and the way they died is the useful part:**

1. **"The batched im2col's per-batch bookkeeping and barrier are the cost."** No: the ratio is
   **identical at one thread** (1.04x) as at four (1.05x). A barrier nobody waits at cannot be the
   story, and this is what one-thread runs are for.
2. **"`ggml_conv_1d_direct_ok` declines these shapes, so they fall to a path tuned for `kw >= 3`."**
   True, and it does not matter: **A2 ≈ A** on the Pi (45.82 against 45.74), so which of the two
   lowerings loom picks is worth nothing at `kw = 1`. The declining is correct, not costly.
3. **The layout — bullet three, the only one left — is real but small and not free.** C is 1.19x, which
   is 7.3 ms per synthesis, **0.7% of 1.099 s**. And **that number is an upper bound that ignores where
   the transposes would go**: the pointwise convolutions are interleaved with the encoder's `kw = 3`
   FFN convolutions, which are 12x the arithmetic (88.5 against 7.4 MFLOP at L~100) and want the
   `[L, IC]` layout that C gives up. Flipping the encoder to channel-major does not remove a transpose,
   it moves it onto the more expensive operand. Anyone reviving this has to price THAT, not this table.

**What made the difference between the two machines is worth keeping.** On x86 A/B is 1.15x, mostly one
row: `192 x 384 x 100` costs A **1.932 ms** against A2's 0.882 — ggml's batched convolution has a bad
case at short L with wide OC there, which ARM does not have. If a third machine ever shows 1.3x on this
table, that row is where to look first; it is a ggml issue, not a loom one.

**Reproducing it.** `scripts/bench9.cpp`, second table, built as its header comment says and run as
`./bench9 <threads>`. The counts in it come from `conv_census.py` and are POST-P4.15f: 38 of the
`192 x 192 x 100` calls, not the 62 loom used to issue.


### Step 2: tile the chain over the sequence — MEASURED OUT (P4.15b)

The idea: each convolution still writes its whole output and the next one reads it back, so tile the
chain — compute `[p0, p0+T)` of the second convolution from `[p0 - halo, p0 + T + halo)` of the first
— and the intermediate never leaves cache. Ten passes over full activations per resblock layer become
three, which the traffic table above prices at ~120 ms.

**It is worth 1.05x on the chains, which is under 2% end-to-end. Do not build it.**
`scripts/bench11.cpp` is the prototype and the measurement, on the model's own nine resblock chains
(three per upsample stage: kw 3 d(1,2), kw 5 d(2,6), kw 7 d(3,12)). It is a fair prototype, not a
strawman: the rolling window means the first convolution is computed **exactly once per position**
after a 2*halo prologue per thread, so there is none of the recomputation this item's own sketch
budgeted for. Pi 4, 4 threads, best T per shape:

| chain | two convs | chained | ratio |
|---|---|---|---|
| 32x73472 kw3 d(1,2) | 50.9 ms | 37.2 ms | **1.37x** |
| 32x73472 kw5 d(2,6) | 65.4 ms | 54.8 ms | 1.19x |
| 32x73472 kw7 d(3,12) | 85.4 ms | 77.5 ms | 1.10x |
| 64x18368 kw3 d(1,2) | 47.8 ms | 40.6 ms | 1.18x |
| 64x18368 kw7 d(3,12) | 100.1 ms | 98.2 ms | 1.02x |
| 128x2296 kw3 d(1,2) | 25.4 ms | 25.6 ms | 0.99x |
| 128x2296 kw7 d(3,12) | 75.9 ms | 92.3 ms | **0.85x** |

and with an ORACLE picking T per shape — which no heuristic gets — the nine together are 1.046x. The
resblock convolutions are 505 ms of a 1.34 s synthesis, so that is ~25 ms, for a kernel spanning four
graph nodes and a detector that has to prove aliasing safety across a whole resblock layer.

**Why the traffic model overpredicted by 20x, which is the part worth keeping.** Sweep the tile size
and the optimum is the SMALLEST tile, degrading monotonically to 0.39x at T = 2048 — the intermediate
wants to be in L1, not the L2 this was scoped around, and at 128 channels the chain never wins at any
T. What the chain actually recovers, shape by shape, is close to exactly what that layer's two PADDED
COPIES cost and nothing more: 32x73472 kw3 saves 12.3 ms against 16.4 ms of padded copy, and kw7
d(3,12) saves 5.0 ms against 16.3 ms.

So the sweep's own memory traffic **was never exposed**: its loads are already overlapped with its
FMAs, and removing a read that the arithmetic was hiding removes no time. Only the padded copy, which
is a pure `memcpy` with nothing to hide behind, is on the clock. That is a general correction to how
this thread has been estimating — **count only the passes that have no arithmetic over them.**


### The im2col item, re-opened and repaired (P4.14 `ggml_conv_2d_direct`, closed at 0.98x)

P4.14 measured ggml's own fused convolution against loom's im2col + mul_mat lowering, got **0.98x**,
and closed it: "nothing, marginally worse -- do not re-propose". That verdict was right for the engine
it was measured on and wrong for this one. With the GEMM 1.7x faster, `IM2COL` had gone from a modest
slice to **40% of all convolution time**, and re-running the same comparison (`scripts/bench9.cpp`)
reproduced 0.97x — so the op really was no better, and the question became WHY, given that not
materialising a 66 MB patch matrix ought to be worth something.

**Two answers, both in ggml's implementation rather than in the idea.**

1. **Its batch was 16 MB, so the patches never stayed in cache.** The whole point of doing im2col a
   batch at a time is that each batch is written and immediately consumed by the GEMM without reaching
   DRAM -- which requires the batch to FIT IN CACHE. `GGML_IM2COL_WORK_SIZE` is 16 MB, larger than most
   last-level caches and 16x the Pi's L2, so the op wrote the patches out and read them back exactly as
   a full im2col does, for no saving and with extra structure. Sweeping the budget on the A72: 16 MB
   **0.97x**, 2 MB 1.03x, 1 MB 1.12x, **512 KB 1.16x**, 256 KB 1.07x.
2. **It scattered its GEMM output one element at a time.** The GEMM wrote to scratch and a permute loop
   then copied it into place with a strided scalar store per output element -- 2.35 M of them for one
   vocaler conv. That pass is unnecessary: a mul_mat writes `C[ldc*col + row]`, which with
   `ldc = OW*OH` and the patches in `src0` **is** the [OW, OH, OC] layout the destination already has,
   so the GEMM can write straight into it. Worth the difference between 1.08x and **1.18x**.

Both are `cmake/patches/ggml-0004-conv2d-cache-blocked.patch`, along with a rule not to split a
convolution whose whole patch matrix already fits the budget (without it the two shortest shapes
measured 0.98x and 0.92x; with it, 0.98x and 1.02x). loom then lowers `CONV_1D` to `GGML_OP_CONV_2D`
with KH = 1 (`src/ops/primitives_conv.cpp`), which is the "few lines in primitives_conv.cpp" P4.14
predicted -- it just needed the op underneath to be worth calling.

**On aarch64 only, and that is measured, not cautious.** The same bench on an AVX2 Ryzen 3 3250U says
**0.87x** — best case 0.91x at a 2 MB budget — so on x86-64 the patch matrix is worth materialising.
That machine does in 0.555 s what the Pi takes 1.37 s to do; cache-blocking buys a bandwidth-rich core
much less, and the batching costs it its one big GEMM. A hypothesis that the difference was the
parallel axis (`conv_2d` splits patches, ggml's standalone `IM2COL` splits channels, so each thread
juggles IC read streams) was tested by switching `conv_2d` to the channel split: **worse**, 0.70x. The
`#if` in `primitives_conv.cpp` is therefore a measurement on two machines, and neither of them is a
many-core x86 server -- which is the configuration to re-run `bench9` on before generalising it.

**Testability, because an `#if defined(__aarch64__)` otherwise means the path is never covered by the
gates.** The macro is `LOOM_CONV1D_DIRECT`, defaulting to the architecture but overridable:
`-DLOOM_CONV1D_DIRECT=1` builds the direct lowering anywhere. The whole suite was run that way on
x86 -- **ci 66/66, gate 82/82**, including every conv-bearing gate (kokoro, styletts2, matcha, the
NeMo encoders), none of which exists on a Raspberry Pi.

**The two lowerings agree bit for bit**, on every one of bench9's eleven shapes and on the whole VITS
synthesis: the engine built `-DLOOM_CONV1D_DIRECT=0` and `=1` on the Pi produces byte-identical audio
(`cmp`, 293888 bytes). That is expected rather than lucky -- the batching splits the patch axis and
never the reduction, so every output element is the same sum in the same order -- but it is the
difference between a refactor and a rewrite, and it is worth checking rather than assuming.

**What it bought:** convolution 1.27 -> 1.07 s, synthesis **1.79 -> 1.62 s**, and 153 `CONV_2D` nodes
in place of 165 `IM2COL` + 213 `MUL_MAT`.


### What NOT to re-propose (all measured, in P4.15 and above)

| idea | verdict |
|---|---|
| a better GEMM micro-kernel | ggml's tinyBLAS now beats a hand-written 4x4; 25.1 GFLOP/s |
| a better direct-conv tile (2x32, 4x8) | both lose at every thread count |
| MLAS-style weight packing | the packing is not the lever; the gather was, and it is gone |
| phase-major for the low-channel shapes | 0.48-0.92x; shipped only for `kw>=7, dil>=3, IC*kw>=768` |
| vectorising the im2col gather | ~10% of it, twice measured; the cost is the write amplification |
| `kw` shifted mul_mats to dodge im2col | 0.43-0.98x (P4.14), and the direct kernel supersedes it |
| runtime `if`s in the direct kernel's store | **1.44 -> 1.73 s** — see trap 1 above |
| fusing the resblock layer's FIRST unary | it has three consumers; nothing to skip |
| tiling the two convolutions of a resblock as one chain | **1.05x on the chains, <2% end-to-end** — step 2 above, and `scripts/bench11.cpp` is the prototype that says so |
| scaling a published loom-vs-onnxruntime ratio by a measured engine-side factor instead of re-measuring both sides | 30% out on the first cell checked; the published baseline was itself unreproducible. [Retro-018](retro-018-a-table-of-ratios-nobody-could-re-derive.md) |
| clamping `ggml_get_n_tasks` by work size for small elementwise ops | **not the mechanism** of P4.17's thread-scaling collapse — that was libgomp's wait policy, and nothing was clamped. [Retro-017](retro-017-libgomp-slept-at-every-graph-node.md) |
| a cheaper `ggml_barrier` | a fifth of P4.17 at most: 2520 barriers x 11.4 us = 28.7 ms of a ~124 ms delta. Retro-017 |
| `OMP_PROC_BIND` / `taskset` pinning for the many-core collapse | no effect once threads spin; `bind=close` without spinning is far worse. Retro-017 |
| counting a convolution's memory passes to predict a win | only the passes with NO arithmetic over them are on the clock; the rest are hidden behind the FMAs |
| lowering `kw = 1` as a matmul instead of a convolution | **1.04-1.05x on the Pi, 2.2 ms of 1.099 s** — the im2col at `kw = 1` IS the transpose a `mul_mat` needs, so there is nothing to skip. P4.15c, `scripts/bench9.cpp`'s second table |
| channel-major activations so `kw = 1` needs no transpose at all | 1.19x, 7.3 ms — and that is an upper bound that ignores the transposes it moves onto the `kw = 3` convolutions next to them, which are 12x the arithmetic. P4.15c |

