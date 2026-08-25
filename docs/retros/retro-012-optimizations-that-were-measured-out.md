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
| a cheaper `exp` for `SOFT_MAX` | **CORRECTED 2026-08-24** — the probe that closed this was not a floor; the exp is 1.4-1.6x of the row, not 1.0x |
| a cheaper `hsum` epilogue for tinyBLAS at `k = 64` | 1.06x, under the dev box's noise floor (P4.18) |
| rows-inner loop order, so a store finishes a cache line of C | **mechanism falsified**: no dependence on the size of C — see below |
| `ggml-0002`'s aarch64 address hoist applied to x86 | neutral at every k tested, small and large |
| *(NOT measured out)* tinyBLAS's `BM` row blocking at `k = 64` | ~1.15x paired, and 1.02x at k=768; real but partial, and whisper's m=1500 cannot reach `BM = 4` |

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

