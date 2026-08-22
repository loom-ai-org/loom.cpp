# Upstreaming these patches to ggml-org/ggml

Seven diffs, each independently useful and independently reviewable. They are written against the pin in
`cmake/GgmlPin.cmake` (**v0.19.0**, commit `30bf8685`), so the first step for any of them is a rebase
onto `master` — the files move rarely, but `sgemm.cpp`'s tile-selection block and `ops.cpp`'s
`ggml_compute_forward_conv_2d_impl` are both areas that see occasional churn.

**Order and independence.** 1, 2 and 3 touch `llamafile/sgemm.cpp` and are independent of each other,
though 1 and 2 only pay off together (see below) and are best sent as one PR or two linked ones. 4 and
5 touch the CPU convolution; **5 depends on 4** (it uses the `ldc` mul_mat helper 4 introduces, and its
staging only makes sense once 4 has made the op batch-oriented). A reviewer taking 4 without 5 is fine;
5 without 4 is not. **6 also sits on top of 4** -- it is a second path inside the same op -- and is
independent of 5, though it serves 5's fused bias for free. **7 generalises 5** -- it replaces 5's
entry point with one that also takes a `LEAKY_RELU` on the input and a residual `ADD` on the output --
so it must be sent after 5, or the two folded into one PR.

**Every number below is a Raspberry Pi 4B (Cortex-A72, 4 cores @ 1.8 GHz, 1 MB shared L2, Debian
aarch64, gcc 14.2) unless it says otherwise; x86-64 numbers are a Ryzen 3 3250U (AVX2, 2 cores), median
of seven runs pinned to the physical cores.** The workload is a VITS TTS vocoder — an all-convolutional
generator, F32 throughout, one image per batch, `KH = 1`. The benches are in `scripts/` in this repo
(`bench6.cpp` GEMM, `bench7.cpp` register tiles, `bench9.cpp` conv lowering, `bench10.cpp` direct
convolution) and are self-contained against a built ggml.

**What a reviewer is likely to ask, for all seven: measurements on hardware neither of these two boxes
represents** — a wide ARM core (Neoverse V2, Apple M-series) and a many-core x86 server. Say so up
front rather than being asked.

---

## PR 1 — `sgemm`: don't pick a register tile GCC will spill on aarch64

*(`cmake/patches/ggml-0001-tinyblas-neon-gcc-tile.patch`)*

**Problem.** `tinyBLAS::matmul` treats `__ARM_NEON` as a 32-vector-register target and picks a 4x6 tile,
which keeps 24 accumulators live. That fits the register file on paper — 24 accumulators plus 4 A
vectors and 1 B vector is 29 of 32 — but GCC does not allocate it. At `-O3` it keeps the `Cv[][]`
array's canonical copy in memory and stores all 24 accumulators to the stack **on every k iteration**:
twelve `stp q` against twenty-four `fmla`, visible in the disassembly. The A72 has one store pipe, so
those stores cost about what the arithmetic they protect does.

**Evidence.** Spilling tracks tile size and nothing else. Counting q-register stores to the stack per
block, and GFLOP/s over eleven real vocoder GEMM shapes (K 96–1344, M 288–73472, N 32–384) at 4
threads, with each tile run under an identical OpenMP driver (`scripts/bench7.cpp`):

| tile | 4x3 | 4x4 | 4x5 | **4x6 (current)** | 8x4 | 4x8 |
|---|---|---|---|---|---|---|
| accumulators | 12 | 16 | 20 | 24 | 32 | 32 |
| q-register spill stores | 0 | 0 | 8 | 10 | 32 | 29 |
| GFLOP/s | 24.2 | 24.5 | 18.4 | **16.3** | 16.3 | 14.6 |

Through ggml's own dispatcher on the same shapes: **15.6 GFLOP/s as shipped, 22.0 with the 16-register
schedule**. The tile with the *better* load/FMA ratio on paper (0.42 vs 0.50) loses, because it does
not fit this compiler's allocator — not because it does not fit the register file.

**Fix.** One `#if` line: on aarch64 with GCC, take the existing `VECTOR_REGISTERS == 16` schedule.

**Why it is scoped to GCC.** clang 19 holds the same 4x6 tile with **zero** spills (checked in the
object code) and runs the current schedule at 23.8 GFLOP/s on the same core; giving it the smaller tile
costs it about 1%. So this is a GCC code-generation fix, not a microarchitecture one, and a clang-built
binary — every macOS one — should keep the schedule upstream chose. A reviewer may prefer a different
spelling of that condition; the measurement is what matters, not the `#if`.

**Interaction with PR 2.** Tile alone is 22.0 GFLOP/s, addresses alone (PR 2) are 15.5 — the spill
dominates until it is gone — and both together are **25.1**, which is past a hand-written 4x4 NEON
kernel measured in the same process (24.3). Neither is worth much without the other.

**Testing.** Bit-identical output: the tile shape changes how work is grouped, never the order of the
accumulation, so every output element is the same sum of the same products. Verified over the eleven
shapes and over a whole 73472-sample synthesis (byte-identical `cmp`).

---

## PR 2 — `sgemm`: write the operand addresses so GCC will strength-reduce them on aarch64

*(`cmake/patches/ggml-0002-tinyblas-aarch64-address-hoist.patch`)*

**Problem.** `gemm_bloc` addresses its operands as `A + lda * (ii + i) + l` against the class members.
On aarch64, GCC 14.2 does not form pointer induction variables over `l` from that: it re-derives all
seven operand addresses every iteration — 8 `add` + 6 `lsl` wrapped around 12 `fmla` and 7 `ldr q`, a
**35-instruction loop where 21 do the same work**. The identical source with the bases and strides as
locals compiles to those 21, with the same compiler and the same flags (checked by compiling the
extracted block with ggml's own `flags.make` line).

**Fix.** Hoist the two base pointers and the strides into locals before the loop. No arithmetic change
whatsoever — bit-identical output, and it is an addressing rewrite, not a numerical one.

**Evidence.** With PR 1 also applied: **22.0 → 25.1 GFLOP/s** at the eleven shapes, 4 threads.

**Why it is scoped to aarch64.** x86-64 does not want it: there GCC already forms the induction
variables, and the extra live values only add pressure — **55.4 → 53.6 GFLOP/s** and 12 more `%rsp`
reads in the block. With the `#if defined(__aarch64__)` in place, the x86 object code is
register-renaming-identical to the current one, which is a better check than re-timing it. clang on
aarch64 is neutral (24.0 vs 23.8), since it already did this.

---

## PR 3 — `sgemm`: handle `m % 4 != 0` instead of declining the whole matmul

*(`cmake/patches/ggml-0003-tinyblas-row-tail.patch`)*

**Problem.** Every tile in the file is 4 rows tall, so `matmul` accepts only `m` divisible by 4. When it
is not, it returns false and ggml computes the **entire** matmul with the generic
one-output-element-per-call kernel. One leftover row costs the other thousands their blocking.

This is not a corner case: a VITS vocoder's second-largest GEMM has `m = 287` — a frame count, i.e. a
number nothing rounds. That bucket measured **324 ms before and 324 ms after** two other improvements
to this same file, unmoved to the millisecond, because none of that work ever entered it.

**Fix.** Split the row count: the aligned prefix goes through the tiles as before, the ≤ 3 leftover rows
go through a 1x1-blocked loop split by column across the threads. **Those rows get exactly the kernel
they would have had**, so nothing can regress; the rest of the matrix gets the tiles it was being
denied.

**Evidence.** `MUL_MAT [287, 384]` **324 → 190 ms**; all four `m = 287` buckets 354 → 208 ms; the
synthesis 1.94 → 1.79 s. Architecture-neutral — x86 benefits identically wherever a model has an odd
row count.

**Testing.** A test covering `m % 4` ∈ {1, 2, 3} against a double-precision reference, plus a matrix
with fewer rows than one tile; verified red by truncating the tail loop.

---

## PR 4 — `conv_2d`: size the patch batch for a cache, and let the GEMM write the output directly

*(`cmake/patches/ggml-0004-conv2d-cache-blocked.patch`)*

**Problem, two of them.**

1. **`GGML_IM2COL_WORK_SIZE` is 16 MB**, so a batch of patches is larger than most last-level caches
   and 16x this machine's L2. The whole point of doing im2col a batch at a time is that each batch is
   written and immediately consumed by the GEMM without reaching DRAM — which requires the batch to
   *fit in cache*. At 16 MB the patches are written out and read back exactly as a full im2col matrix
   would be, and the op is then a slower way to do the same thing.
2. **The result is scattered into place one element at a time** — the GEMM writes a scratch buffer and
   a permute loop then copies it out with a strided scalar store per output element, 2.35 M of them for
   a single vocoder convolution. That pass is unnecessary: a mul_mat writes `C[ldc*col + row]`, which
   with `ldc = OW*OH` and the patches in `src0` **is** the `[OW, OH, OC]` layout the destination
   already has.

**Evidence**, against ggml's own `im2col` + `mul_mat` lowering over eleven convolution shapes
(`scripts/bench9.cpp`), 4 threads, weighted by how often each shape appears in one synthesis:

| patch budget | 16 MB (current) | 2 MB | 1 MB | **512 KB** | 256 KB |
|---|---|---|---|---|---|
| vs im2col + mul_mat | 0.97x | 1.03x | 1.12x | **1.16x** | 1.07x |

Half the L2 is the peak because the kernel panel and the output slice have to live there too. Adding
the direct write takes it to **1.18x**; without it, the same cache budget is only 1.08x. A third
element — not splitting a convolution whose whole patch matrix already fits the budget — takes the two
shortest shapes from 0.98x/0.92x to 0.98x/1.02x.

**Bit-identical**: batching splits the patch axis, never the reduction.

**The question a reviewer should push on: the 512 KB constant.** It is right for a 1 MB L2 and it is
almost certainly wrong for something else; 16 MB is equally arbitrary but much further from any real
cache. The honest options are a constant with this measurement next to it (what the patch does), or a
cache-size query, which ggml does not currently have and glibc answers unreliably on ARM.

**Second question: this op only wins on some machines.** On the AVX2 x86 box the same comparison is
0.87x — best case 0.91x at a 2 MB budget — so there the full im2col matrix is worth materialising. The
patch makes the op better everywhere it is used; it does not make it the right lowering everywhere. In
this project the choice of lowering is made per architecture for exactly that reason.

---

## PR 5 — CPU backend: fuse `CONV_2D` with the per-channel bias `ADD` that follows it

*(`cmake/patches/ggml-0005-conv2d-bias-fusion.patch`, depends on PR 4)*

**Problem.** A convolution's bias add is a full read and write of its output. `ggml_cpu_try_fuse_ops`
already exists and already fuses one pattern (`RMS_NORM` + `MUL`); this adds `CONV_2D` +
per-channel `ADD`, applying the bias to each batch of the result while that batch is still in cache.
The graph is unchanged, so fusion stays a compute-time decision of the CPU backend and no other backend
needs to know.

**Evidence.** **1.605 → 1.576 s** on a whole synthesis, against the identical build with
`GGML_CPU_DISABLE_FUSION=1`. Note that this is **1.8%, where the per-node profile attributes 12% to
those ADDs** — an elementwise pass costs less inside a graph than alone, because the fused kernel still
has to write the output; what disappears is the ADD's read pass.

**The part worth a reviewer's attention: the destination is usually the convolution's own input.** A
graph allocator hands the ADD a block the input has just been freed from — in the unfused order nothing
reads that input by the time the ADD runs — and in this vocoder it does that to *every* large
convolution. Writing the result there progressively, while later batches still need the input, corrupts
it in a way that still looks like a convolution (max abs diff 0.54 on a real model). The kernel
therefore **stages each batch and lands it only after the next batch's im2col has read what it would
overwrite**: batch k reads input from `s_k - pad` upwards, batch k-1's output covers `[s_{k-1}, s_k)`,
and nothing at or after k ever reads below `s_k - pad`. Where a batch is shorter than the kernel's
reach, or the overlap is partial rather than exact, the fusion declines or falls back.

An earlier version simply declined on any overlap. It was correct, and it fired on nothing that
mattered: **0.3%**.

**And the staging buffer has to be channel-major.** Staged the way the existing permute path wants it,
landing it reads with a stride of `c_out`, and that gather costs more than the entire ADD being
removed — **1.70 s, slower than not fusing at all**. Written `[c_out, patch_n]` via PR 4's `ldc`
mul_mat, each channel lands as a contiguous copy out of cache: 1.576 s.

**Testing.** A test that asserts the numbers against a double-precision reference, that fused and
unfused agree **bit for bit**, and — the part an output comparison cannot show — **that the fusion
happened**, by poisoning the convolution's own result tensor, which a fused run never writes. It covers
the aliased destination explicitly and was verified red both ways: with the detector forced to decline,
and with the aliasing guard removed, which reproduces the corruption hermetically.

---

## Things to carry into every PR body

* The bench that produced each number, and the fact that they are reproducible from this repo.
* Which claims are bit-identical (1, 2, 3, 4, and 5's fused-vs-unfused) and which change numerics
  (6 and 7 do, by summation order: 7 is 6.7e-8 max on a 0.17 peak, thread-count invariant)
  (none of these do; enabling `GGML_LLAMAFILE` at all does, by ~3e-7 relative, but that is a build
  option, not a patch).
* The two machines, and the explicit request for a third.
* For 4 and 5: that they are CPU-backend-only and leave every other backend's path untouched.

---

## PR 6 — `conv_2d`: a direct 1-D convolution behind a cache-size heuristic

*(`cmake/patches/ggml-0006-conv1d-direct.patch`, sits on top of PR 4; independent of PR 5, though it
serves the fused bias for free)*

**Problem.** im2col turns a convolution into a GEMM, which is the right move when the weights are large
— a GEMM blocks both operands so neither is re-read. It is the wrong move when the activation is long
and the weights are small: it writes every input element `kw` times (137 M element-writes for one TTS
synthesis) to feed a kernel that reads them once. Phase-timed inside the op, that gather is **37% of
all convolution time** (396 ms of 1070 ms), against a GEMM that is already running at 23.5 GFLOP/s.

**Fix.** A direct kernel: hold a tile of the OUTPUT in registers, sweep the activation where it lies,
one broadcast weight per (input channel, tap). Nothing is materialised. The tile is ISA-sized —
aarch64 has 32 vector registers and a lane-broadcast FMA, so 4 channels x 16 positions with one vector
load per four weights; AVX2 has 16 registers and no lane broadcast, so half the accumulators and a
broadcast load per weight. A bias, when the caller has fused one, costs nothing: the accumulators start
at it.

**Evidence**, against the batched im2col, eleven convolution shapes of a VITS vocoder:

| | Cortex-A72, 4 threads | Ryzen 3 3250U (AVX2), 2 threads |
|---|---|---|
| 32x32 kw7 L73472 | 63.2 → 38.5 ms (1.64x) | 70.8 → 15.0 ms (**4.7x**) |
| 64x64 kw7 L18368 | 58.1 → 40.6 ms (1.43x) | 70.7 → 16.0 ms (4.4x) |
| 192x384 kw5 L287 | 10.6 → 40.9 ms (**0.26x**) | 11.0 → 6.5 ms (1.7x) |
| 768x768 kw3 L100 | 24.6 → 164 ms (**0.15x**) | 16.4 → 16.0 ms (1.03x) |

End-to-end on a whole synthesis: **1.576 → 1.486 s** on the Pi and **1.503 → ~1.19 s** on the x86 box.

The ragged tail is worth a note for a reviewer: it runs **one more overlapping tile** ending at the last
position rather than a scalar loop over the remainder. At OL = 2296 that remainder is 8 positions, but
8 x OC x IC x KW scalar multiply-accumulates is ~2 ms of a 20 ms convolution — 22 ms across a whole
synthesis. Recomputing up to P-1 positions is much cheaper than computing any of them slowly.

**And the padded copy really is one pass.** It is worth saying explicitly because the first version of
this patch made it two: it zeroed the whole padded row and then copied over all but the `pad` floats at
each end, writing every element of a 9.4 MB buffer twice. Zeroing only the two strips is **26 ms of a
1.34 s synthesis**, about a third of what the copy costs. A reviewer comparing against `ggml_im2col`
should compare against the fixed version.

**The heuristic is the reviewable part.** Three conditions:

* shape and type (F32, `KH = 1`, one image, stride 1, contiguous, `OC % 4 == 0`), everything else
  falling through to the batched path unchanged;
* **weights must fit cache** — `L3/2` where the machine reports one, else `L2`, else 512 KB. That is
  2 MB on the x86 box and 512 KB on the Pi, which is where each machine's measurements put the line.
  One rule, two answers, no `#if`. `sysconf(_SC_LEVEL*_CACHE_SIZE)` returns 0 on every aarch64 Linux
  box tried, hence the floor;
* **at least as many position blocks as output-channel tiles.** Without it the synthesis was 1.576 →
  **1.703 s**, slower than not having the kernel: the model has 77 convolutions of 100 positions and
  192 channels that pass the weight test and should not take this path.

**Not bit-identical** — a different summation order. Whole-synthesis difference against the previous
lowering: max 3.5e-6 on a 0.17 peak, rel-RMS 6.5e-6.

**A third path, for wide dilated kernels.** The same patch carries a phase-major ("a trous") variant:
a convolution with dilation d is d dense convolutions over the subsequences `p = j*d + r`, and laid out
that way a channel's taps stop being d floats apart -- at dilation 12 with kw 7 the direct kernel reads
seven 64-byte runs spread over 352 bytes per channel, which is 896 prefetch streams for a 128-channel
convolution. NEON de-interleaves by 2, 3 and 4 in one instruction and 6 and 12 factor into two such
passes. It is worth **1.52x** at 128x128 kw7 dilation 12 and **0.48x** at 32x32 kw5 dilation 6, so it
is gated to `kw >= 7 && dilation >= 3 && IC*kw >= 768` and to aarch64, where those instructions exist.
Two convolutions of the model take it, for 1.487 -> 1.463 s.

**What a reviewer should push on.** The 512 KB floor, the `OC % 4` restriction and the phase window's
three constants are all "good enough for what was measured" rather than principled; the tile sizes are tuned on two machines; and there is
no ARM64 counterpart to the AVX2 path for AVX-512 or SVE, which would want their own tile. Also worth
saying: benchmarks of convolution shapes must carry the model's **dilations**. At dilation 1 this
kernel measures ~1.6x on the shapes above; at the model's own dilations (1, 2, 3, 6, 12) the same
shapes give 1.2-1.4x, and that difference is most of the gap between the bench and the end-to-end
number.

---

## PR 7 — CPU backend: fuse the `LEAKY_RELU` before a `CONV_2D` and the residual `ADD` after it

*(`cmake/patches/ggml-0007-conv1d-elementwise-fusion.patch`, generalises PR 5; benefits from PR 6)*

**Problem.** PR 5 removed one of the three elementwise passes a HiFi-GAN resblock layer puts around its
convolution. The layer is exactly:

```
h  = LEAKY_RELU(x)        a full read and a full write of the activation
c  = CONV_1D(w, h)
xt = ADD(c, bias)         PR 5 folded this one in
x' = ADD(xt, x)           the residual: two reads and a write
```

On a Cortex-A72 one pass over this vocoder's largest activation (9.4 MB) costs **4.7 ms** to
read-and-write and **8.3 ms** for the two-read-one-write add, at 3.4–4.0 GB/s — and one core nearly
saturates that, so threads do not help. Against a convolution that is 20–40 ms, the passes are not a
rounding error. They are also not slow code: the only way to make them cheaper is not to do them.

**Fix.** Neither needs a pass of its own. The convolution already copies its input into a padded
buffer, so the unary is applied *as* that copy is made; the accumulators already start at the bias, so
they can start at `bias + residual[p]` — in this kernel the residual is added as each accumulator is
stored, which is the one moment both are in registers. The intermediates are never written. Detection
extends `ggml_cpu_conv_2d_bias_add_idx` into `ggml_cpu_conv_2d_fusion`, which matches the chain from
either end (from the `LEAKY_RELU`, when the convolution is its only consumer, or from the convolution),
and all three of `act`, `bias` and `residual` are optional.

**Evidence.** Whole VITS synthesis, Pi 4B, 4 threads: **1.441 -> 1.345 s**, and the fusion's own share
of that is **~50 ms** (the other half comes from a change in the calling project that stopped putting a
`CONT` between the bias add and the residual add, which this detector cannot look past). Measured by
switching each fusion off at runtime inside ONE binary — the box drifts 1.45 -> 1.58 s over twenty
minutes of continuous load, which is larger than the effect, and rebuilding between arms confounds that
drift with code layout. Four rounds, order rotated between them, cooldown before each; the ordering of
the arms was identical every time.

**Why it stops here.** The obvious next step is to fuse the resblock's TWO convolutions into one
kernel tiled over the sequence, so the activation between them never leaves cache. That was
prototyped and measured (`scripts/bench11.cpp` in the calling project, with the rolling window that
makes it recompute nothing): **1.05x on the nine real chains with an oracle choosing the tile size per
shape, and below 1.0x at 128 channels for any tile size.** The reason is worth carrying into any
review of this patch — a convolution's loads are already overlapped with its FMAs, so removing them
removes no time. The only memory pass on the clock is the padded copy, which is a `memcpy` with no
arithmetic over it. Count passes that way before predicting a win from any of this.

**Two things a reviewer should push on.**

**1. It was a 19% *regression* before it was a win, and the cause is not the residual.** The
accumulators are a 2-D array indexed by the store loop's own counters. A compiler can only keep such an
array in registers if it unrolls that loop, and GCC stops unrolling the moment the body has a branch in
it. Written the obvious way — `if (res) …; if (whole) …` inside the store — sixteen vector accumulators
moved to the stack in **every convolution in the model**, including all the ones with nothing fused
into them, and the synthesis went **1.44 → 1.73 s**. The two conditions are constant across a whole
convolution, so they are template parameters and the call site picks one of four specialisations; the
"nothing fused" one then compiles to exactly what PR 6 shipped. This is the same failure mode as PR 1,
in a different function.

**2. The residual may be the destination, and the tail tile must not double it.** As in PR 5, the graph
allocator hands `x + f(x)` the block `x` just vacated, so the residual operand and the output are
frequently the same memory. That is safe here — every output element reads exactly the residual element
it is about to overwrite — *except* for PR 6's overlapping tail tile, which deliberately recomputes
positions the last full tile already wrote, on a different thread with no barrier between them. Adding
a residual to those a second time is a real corruption, so the tail tile carries a floor position below
which it does not store. The detector separately rejects any *partial* overlap between the destination
and either the input or the residual, and any overlap at all with the weights or the bias.

**Testing.** ci (68) and gate (82) green, fused-vs-unfused agreeing to 6.7e-8 max on a 0.17 peak and
identical at 2 and 4 threads (a race here would not be). The gate was verified able to fail by
perturbing the fused slope by 5% and, separately, the fused residual by 5%: `matcha_mil` catches both.
