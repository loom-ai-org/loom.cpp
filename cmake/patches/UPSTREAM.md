# Upstreaming these patches to ggml-org/ggml

Eleven diffs, each independently useful and independently reviewable. They are written against the pin in
`cmake/GgmlPin.cmake` (**v0.19.0**, commit `30bf8685`), so the first step for any of them is a rebase
onto `master` — the files move rarely, but `sgemm.cpp`'s tile-selection block and `ops.cpp`'s
`ggml_compute_forward_conv_2d_impl` are both areas that see occasional churn.

**Order and independence.** 1, 2, 3 and 11 touch `llamafile/sgemm.cpp`. 1 and 2 only pay off together
(see below) and are best sent as one PR or two linked ones; **11 is the same fix as 3 on the other
axis** and lands on adjacent lines of the same function, so send them together or 3 first. 4 and
5 touch the CPU convolution; **5 depends on 4** (it uses the `ldc` mul_mat helper 4 introduces, and its
staging only makes sense once 4 has made the op batch-oriented). A reviewer taking 4 without 5 is fine;
5 without 4 is not. **6 also sits on top of 4** -- it is a second path inside the same op -- and is
independent of 5, though it serves 5's fused bias for free. **7 generalises 5** -- it replaces 5's
entry point with one that also takes a `LEAKY_RELU` on the input and a residual `ADD` on the output --
so it must be sent after 5, or the two folded into one PR. **10 depends on nothing** -- it touches
`vec.h` alone, is the only one of the ten that is not about convolution or GEMM, and is the most
self-contained thing here to send first if a reviewer wants a small one.

**Submission status (2026-08-23).** PRs 1, 2 and 3 have been **sent upstream**. 4 through 9 have
not, and are deliberately held: each of 5, 6, 7 and 9 depends on 4, and 4 itself reads more naturally
once the `sgemm` three have landed and the review conversation has a shape. Nothing below is blocked on
new measurement — it is blocked on 1-3.

**Every number below is a Raspberry Pi 4B (Cortex-A72, 4 cores @ 1.8 GHz, 1 MB shared L2, Debian
aarch64, gcc 14.2) unless it says otherwise; x86-64 numbers are a Ryzen 3 3250U (AVX2, 2 cores), median
of seven runs pinned to the physical cores.** The workload is a VITS TTS vocoder — an all-convolutional
generator, F32 throughout, one image per batch, `KH = 1`. The benches are in `scripts/` in this repo
(`bench6.cpp` GEMM, `bench7.cpp` register tiles, `bench9.cpp` conv lowering, `bench10.cpp` direct
convolution, `bench12.cpp` GELU and soft_max) and are self-contained against a built ggml. **PR 10 is
the exception to the workload sentence above** -- it is a speech ENCODER, whisper-small at 1500 frames,
and its numbers are one thread on the Core Ultra 9 285K.

**What a reviewer is likely to ask, for all nine: measurements on hardware neither of these two boxes
represents** — a wide ARM core (Neoverse V2, Apple M-series) and a many-core x86 server. Say so up
front rather than being asked.

### The many-core x86 half of that gap is now measured

**Intel Core Ultra 9 285K, 24 cores, 36 MB L3, gcc 14.2, Debian.** This is the class the paragraph
above admits to guessing at, and it was the one that could plausibly have made PRs 4 and 6 regressions:
both are heuristics, and 6's cache budget is `L3/2`, so a 36 MB L3 lets it say yes to shapes a 1 MB L2
never would.

`bench9`'s weighted total is the LOWERING choice — im2col + one big `mul_mat` against ggml's
`CONV_2D` — and the `CONV_2D` arm carries **both** PR 4 and PR 6, because 6 is a second path inside
the same op. Splitting them by flipping the new kill switch, 24 threads:

| `bench9` weighted, Core Ultra 9 285K | im2col+mm | `CONV_2D` | ratio |
|---|---|---|---|
| PRs 4+6 active | 0.046 s | 0.026 s | **1.80x** |
| `GGML_CPU_DISABLE_CONV_HEURISTICS=1` | 0.045 s | 0.045 s | **1.00x** |

**Unpatched, ggml's `CONV_2D` is dead level with im2col + `mul_mat` on this machine; the entire win is
the two patches.** It does not go below 1.00x here, so there is no bad path being dodged — 4 and 6 are
creating the margin, not avoiding a loss.

**On the 0.87x, which is the number most likely to be misread.** It was PR 4 *alone*, on a 2-core AVX2
Ryzen, and it is why this lowering was once aarch64-only. It was not refuted by moving to a bigger
x86 box — it was **superseded by PR 6**, whose direct kernel materialises no patch matrix at all: 4.7x
on that same Ryzen's long-activation convolutions and 1.19x end-to-end there. That is why
`LOOM_CONV1D_DIRECT` now defaults to 1 on **every** architecture rather than being switched by one.

For the Pi and Ryzen columns quoted elsewhere in this file, read 1.18x and 0.87x as PR 4 alone against
the unpatched op; they are not comparable to the 1.80x above, which includes PR 6.

**Read `bench10` on this machine carefully, because its headline is misleading.** Kernel-only and
applied to *every* shape, the direct convolution is **0.84x** — ggml's patched `conv_2d` reaches 489
GFLOP/s against the best direct tile's 412. But per shape it wins 1.2-1.8x on the long activations
(32-128 channels, L 2296-73472) and loses badly on exactly two — `192x384 kw5 L287` and
`768x768 kw3 L100` — which are the two `ggml_conv_1d_direct_ok` already declines, by the position-block
rule. The weighted total is dominated by those two because of their call counts. End to end, with the
predicate doing its job, PRs 4+6 are the 1.75x in the table.

**`bench10` had never compiled on x86 at all** (fixed 2026-08-23). Its AVX2 `conv1d_direct` was never
given the `dil` parameter the aarch64 one grew: nine parameters against three ten-argument call sites,
and a tail loop naming a `dil` not in scope. So "nobody has measured the generic path" was
understated — nobody had measured the *AVX2* path either.

**Both heuristics now have a run-time kill switch.** `GGML_CPU_DISABLE_CONV_HEURISTICS=1` declines PR
4's cache blocking and PR 6's direct path, spelled after ggml's own `GGML_CPU_DISABLE_FUSION`
(`ggml-cpu.c:4141`), which PRs 5 and 7 already sit behind. Before it the only escape was a rebuild,
which is not available to somebody who installed a wheel. The 1.75x above is that switch measured in
both positions.

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

---

## PR 8 — `conv_transpose_1d`: the whole prologue ran on one thread, and zeroed 16 MB it did not use

*(`cmake/patches/ggml-0008-conv-transpose-1d-prologue.patch`, independent of every other patch here; **PR 9 sits on top of it**)*

**Problem.** `ggml_compute_forward_conv_transpose_1d_f32` does four things before it computes anything,
all of them inside `if (ith == 0)`:

```c
memset(params->wdata, 0, params->wsize);   // the whole PLAN's work buffer
... transpose the kernel  (K x Cout x Cin) -> (Cin x K x Cout)
... transpose the source  (L x Cin)        -> (Cin x L), element by element
memset(dst->data, 0, ggml_nbytes(dst));    // dst is accumulated into
```

Phase-timed on a VITS vocoder (Pi 4B, 4 threads, three calls per synthesis) that prologue is **47% of
the op** — 100 ms of a 1.31 s synthesis spent in a serial section in the middle of a graph whose every
other op is parallel.

The `memset` of `params->wdata` is the easiest part to fix because **nothing needs it**: the two
transposes below write every element of the two regions this op uses, `[0, nk)` and
`[nk, nk + ne10*ne11)`. And `wsize` is not this op's requirement, it is the **whole plan's** — the
maximum over every node in the graph. Here that is 16 MB, sized by an unrelated convolution, so the op
zeroes 16 MB per call to use about 4.

The rest is parallel work written serially. Nothing in it has a cross-thread dependency.

**Fix.** Delete the `wdata` memset; split the kernel transpose over `Cout` and the `dst` zeroing over
bytes, both into disjoint ranges. Split the source transpose over **`L` rather than `Cin`**, which is
the other way round from how it read: a transpose is strided on one side, and strided reads beat
strided writes — each thread then fills whole contiguous `ne11`-wide rows instead of scattering single
floats `ne11` apart across the buffer, which also keeps two threads off one cache line.

**Evidence.** Whole synthesis, Pi 4B, 4 threads, switching the old serial prologue back on at runtime
inside one binary, ABBA in both orders over two rounds: **1.314 s -> 1.247 s by mean, 1.307 -> 1.234 by
min. ~70 ms, 5.3%.** The op itself goes from 196 ms to about 126 ms — which puts it *below*
onnxruntime's 166 ms for the same three convolutions, where it had been 1.18x above.

**What a reviewer should push on.** The `dst` zeroing is split by raw byte range rather than by row; it
is correct because this op's `dst` is contiguous, but a row split would be the more conservative shape.
And the source transpose is still element-at-a-time — a blocked transpose would do better, and was not
attempted here because the serial-to-parallel change alone took the prologue off the critical path.

**Not a numerics change**: identical output, bit for bit, at 1, 2 and 4 threads — which is also the
check that the new parallel sections do not race.

---

## PR 9 — `conv_transpose_1d`: the compute is a GEMM, and it was one dot product per output element

*(`cmake/patches/ggml-0009-conv-transpose-1d-gemm.patch`, on top of PR 8, and independent of 1-7 except
that it calls PR 4's `ggml_call_mul_mat_ldc`)*

**Problem.** The inner loop is

```c
for (i1 = ...)                       // output channel
  for (i10 = 0; i10 < ne10; i10++)   // input position
    for (i00 = 0; i00 < ne00; i00++) // kernel tap
        ggml_vec_dot_f32(ne02, &v, ..., wdata_src + i10*ne11, ..., wdata_kernel + i00*ne02, ...);
        dst_data[i10*s0 + i00] += v;
```

— one `ggml_vec_dot_f32` per `(output channel, input position, tap)`, which is 1x1 register blocking
and the same shape PR 1 removed from `sgemm`. On a VITS vocoder that is **1.43 GFLOP in 196 ms, 7.3
GFLOP/s**, on a machine that does 25 on a GEMM. It is not a competitive-gap story either: onnxruntime
runs the same three convolutions at 8.6 GFLOP/s, so both implementations were sitting on the same floor.

**The observation.** The right-hand side does not depend on `s0` at all:

```
y[oc][i10*s0 + k] += sum_ic  w[oc][k][ic] * x[i10][ic]
```

Over `(oc, k)` and `i10` that is exactly `[Cout*K, Cin] x [Cin, L]`. Only the SCATTER of the result
knows about the stride — and the transposes this op already does in its prologue leave both operands in
precisely the layout a GEMM wants, contraction over `Cin` fastest on both sides.

**Fix.** One `ggml_call_mul_mat_ldc` per block of input positions, then an overlap-add. Blocked because
the whole result is `K/s0` times the size of `dst` — 18.8 MB against 9.4 for this model's largest
upsample — and the point is not to pay that traffic; a 256 KB block keeps it in L2 and the scatter
reads it immediately. `ggml_graph_plan`'s work-size for this op grows by the same budget, and the two
share one constant so they cannot drift.

**The scatter is split by output channel, and that is load-bearing.** Consecutive input positions write
overlapping runs whenever `K > s0` — which is every upsampling convolution, they are built that way —
so a split over positions would race exactly where this op accumulates. Per `(channel, position)` both
sides are then contiguous over the tap: `K` floats read, `K` floats added.

**Evidence.** Whole synthesis, Pi 4B, 4 threads, old path switchable at runtime inside one binary, ABBA
in both orders over two rounds: **1.252 -> 1.202 s by mean, 1.239 -> 1.186 by min. ~50 ms, 4%.** The op
itself is **126 -> 79 ms** re-profiled, and with PR 8 in front of it **196 -> 79 ms, 2.5x**. Against
onnxruntime's 166 ms for the same three convolutions it is now 2.1x faster.

**Not bit-identical** — the GEMM sums over `Cin` in a different order. Whole-synthesis difference
against the previous lowering: max 8.9e-8 on a 0.17 peak, and identical at 1, 2 and 4 threads, which is
also the check that the scatter does not race.

**What a reviewer should push on.** The 256 KB block budget is one constant tuned on one machine, and
the fallback to the old loop when the work buffer is too small exists only for a caller that sized a
`cplan` by hand — a reviewer may prefer that to be an assert. The F16 path is untouched and still runs
the old loop.

## PR 10 — `vec.h`: the exact-erf GELU is a scalar `erff()` call per element

`ggml_vec_gelu_erf_f32` is the only activation in `vec.h` with **no SIMD path on any architecture** —
it is a bare `erff()` libm call per element, where the tanh-approximation GELU beside it has a 128 KB
f16 lookup table and `ggml_v_silu`/`ggml_v_expf` have hand-written SVE, NEON, AVX-512, AVX2, SSE2 and
RVV paths. That asymmetry is easy to miss because the exact form looks like the rarely-taken one. It
is not: it is what PyTorch's default `approximate="none"` lowers to, so it is the GELU in a large
fraction of transformer MLPs, at full hidden width.

**What it costs.** A 12-layer speech encoder (whisper-small, 1500 frames, `3072 x 1500` per layer):
**273 ms of a 5.91 s run at one thread**, against **55 ms** for the same 96 activations under
onnxruntime — **5.3x, the single worst per-op ratio in that model**, in a profile where `LayerNorm` is
3x *faster* than onnxruntime's, so this is one kernel and not a systemic gap.

**Fix.** Keep the function — this is not a switch to the tanh approximation, which is a different
function that callers distinguish deliberately — and replace only the libm call:

```
erf(z) ~ z * P(w)/Q(w),   w = z*z,   deg P = deg Q = 5,   z clamped to [-4, 4],   result saturated to [-1, 1]
```

AVX-512, AVX2, SSE2 and NEON paths, plus a scalar tail running the *same* rational so that one tensor
is one function regardless of where the vector body stopped. SVE and RVV fall to that tail — correct,
just not yet accelerated, and the obvious follow-up.

**Both of the guards are load-bearing, and one of them needed an exhaustive sweep to find.**
`P/Q -> P5/Q5 = 0.053` as `w` grows, so `z*P(w)/Q(w)` grows *linearly* where `erf` saturates: outside
the fitted interval the approximation diverges rather than merely degrading, and the clamp is what
makes the function total. The saturation to `[-1, 1]` is subtler — without it, `1 + z*P/Q` at the clamp
leaves a residual of order 1e-7 instead of exactly 0, and `0.5*x*residual` is then scaled by `|x|`:
**at `x = -FLT_MAX` the error is 2e31 for an exact answer of `-0`.** Saturating makes both tails exact
and costs nothing elsewhere, because true `erf(4) = 0.99999998` rounds to `1.0f` anyway.

**Accuracy, exhaustively.** Not a sampled maximum — every one of the 2^32 float32 bit patterns against
a double-precision `erf`, with the libm path measured the same way in the same sweep:

| | max abs err | max err / max(\|x\|,1) |
|---|---|---|
| this path | 1.32e-06 | **2.64e-07** |
| `erff()` reference | 4.47e-07 | **1.08e-07** |

**2.4x the error of the path it replaces** — about two f32 ulps of the value's own scale — with the
worst case at `x = 5.0` rather than in a tail. Worth stating plainly for a reviewer: the fit is *not*
what limits this. The underlying rational is 7.2e-08 in double; f32 rounding in the Horner evaluation
accounts for the rest, so tightening the coefficients would buy nothing.

**Evidence.** One thread, `3072 x 1500`, median of seven, `scripts/bench12.cpp` in the loom repo:
**19.0 -> 1.32 ms on a Core Ultra 9 285K (14.3x)** and **121 -> 5.5 ms on a Ryzen 3 3250U (21.8x)**.
In-model that takes the op from 5.3x slower than onnxruntime to roughly 3x faster.

`GGML_CPU_DISABLE_GELU_ERF_SIMD=1` restores the libm path exactly, for bisecting a numerical
regression without a rebuild — spelled like ggml's own `GGML_CPU_DISABLE_FUSION`.

**What a reviewer should push on.** (1) Whether 2.4x the libm error is acceptable for a default, or
whether it belongs behind a build option — the counter-argument is that the tanh-approximation GELU
already shipping in this file is far less accurate than either. (2) The `f16` entry point
`ggml_vec_gelu_erf_f16` is untouched and still calls `erff` per element. (3) SVE and RVV take the
scalar tail. (4) The coefficients are a least-squares fit refined by reweighting, not a Remez
exchange; a reviewer who wants an equioscillation certificate rather than an exhaustive sweep should
say so, though the sweep is the stronger statement for a fixed input type.

---

## PR 11 — `sgemm`: handle `k % KN != 0` instead of declining the whole matmul

*(`cmake/patches/ggml-0011-tinyblas-k-tail.patch`)*

**Problem.** The same shape as PR 3, on the other axis, and a worse one. `matmul` opens with
`if (k % KN != 0) return false;`, so a contraction that is not a whole number of vectors sends the
**entire** matmul to ggml's generic one-output-element-per-call kernel. `m` is a row count; `k` is the
**contraction**, and in attention the contraction is a sequence length — a number nothing rounds.

whisper's encoder is the clean example. Its `A@V` contracts over 1500 mel frames, and **1500 % 8 == 4**
on AVX2, **1500 % 16 == 12** on AVX-512. That matmul has never entered this file on x86 at all. NEON
has `KN == 4` and 1500 % 4 == 0, which is why it is invisible from aarch64 — and why the whole of
PRs 1–3 was written without anyone noticing.

**Fix.** Split the contraction the way PR 3 splits the rows: the aligned prefix runs the vector loop
unchanged, and the ≤ `KN-1` leftover elements are accumulated as scalars **inside the tile**, into the
same accumulator, before the `hsum`. No extra pass over `C` and no extra memory traffic; the added work
is at most 7 scalar products per output element.

Scoped to the F32 instantiation because the tail is a scalar multiply and the other instantiations
carry `ggml_fp16_t`/`ggml_bf16_t`, which do not convert that way. Those keep the existing rejection.

**One thing a reviewer should not simplify away.** There are two epilogues, and the branch is outside
the tile on purpose. Folding the tail loop into a single epilogue — where it is *statically empty*
whenever `k % KN == 0` — costs the small-`k` shapes **30%**: whisper's `QK^T` at `k = 64` went
154 → 201 ms, because the extra live values change how the accumulators are spilled around the `hsum`.
At `k = 64` the epilogue is a large fraction of eight loop iterations, so the aligned case has to keep
byte-for-byte the code it had.

**Evidence.** One thread, Ryzen 3 3250U (AVX2), at whisper-small's own `A@V` shape (`m=64, n=1500,
k=1500`, 12 heads): **176.2 → 81.8 ms, 2.15x**. A `k` sweep at that shape shows the cliff is the
divisibility and nothing else:

| `m=64, n=1500` | k=1496 | k=1500 | k=1504 |
|---|---|---|---|
| GFLOP/s | 44.5 | **20.8** | 47.0 |

In model (whisper-small, `jfk.wav`, `$LOOM_PROFILE` at one thread, two interleaved runs per arm) the
`MUL_MAT 64 x 1500` bucket goes **2240/2460 → 1119/1136 ms** and no other bucket moves.
Architecture-neutral in the same sense PR 3 is: every ISA whose `KN` does not divide a model's
sequence length benefits, and the wider the vector the more often that is. **It was not, for four
days** — see the aarch64 section below, which is why the tail is dispatched out of line.

**And whisper is not the best case — a HEAD DIMENSION misses `KN` more reliably than a sequence length
does.** whisper's `A@V` contracts over frames; `QK^T` contracts over `d_model / n_heads`, which is a
constant of the architecture. NVIDIA's Conformer-CTC small is `176 / 4 = 44`, and **44 % 8 == 4 on every
utterance that model will ever run**. End to end, one thread, two builds of this file differing only in
this patch, paired over 11 rounds per cell:

| | Ryzen 3 3250U | Core Ultra 9 285K |
|---|---|---|
| conformer-ctc-small, 11 s clip | **1.244x** (p10 1.198) | **1.222x** (p10 1.173) |
| whisper-small, 11 s clip | **1.059x** (p10 1.033) | **1.085x** (p10 1.054) |

Its `QK^T` buckets move 2.24-2.41x. Six clips trimmed so the *encoder length* steps through both
residues mod 8 gain the same 1.20-1.24x either way, which is what says the head dimension rather than
the frame count is doing the work. Models whose head dimension already divides `KN` — GigaAM (48),
Parakeet (128), whisper (64) — gain nothing measurable from `QK^T` and only what their `A@V` shape
happens to give.

**THE TAIL IS DISPATCHED OUT OF LINE, AND THAT PLACEMENT IS LOAD-BEARING (aarch64, 2026-08-29).**
The first version of this patch put the scalar tail loop inside `gemm_bloc` — as a second epilogue,
with the branch outside the tile. On x86 that is the fast arrangement and the note above says why. On a
Cortex-A72 it cost **every f32 GEMM shape measured 1.3-1.75x**, one thread, three rounds, and *none of
those shapes ever takes the tail* — NEON's `KN` is 4 and every one of these `k` divides it:

| Pi 4B, m x n x k | with the tail inlined | with it out of line | unpatched |
|---|---|---|---|
| 1500 x 1500 x 64 (`QK^T`) | 4.67 GFLOP/s | **7.38** | 7.54 |
| 64 x 1500 x 1500 (`A@V`) | 4.70 | **7.80** | 7.88 |
| 768 x 1500 x 768 (`proj`) | 4.86 | **8.43** | 8.38 |
| 3072 x 1500 x 768 (`fc1`) | 4.86 | — | 8.56 |
| 768 x 1500 x 3072 (`fc2`) | 4.01 | **5.25** | 5.30 |

**Bisected to the hunk, not guessed:** a build with `kk` truncated and the tail loop *removed* runs at
full speed (8.56 GFLOP/s on `proj`); a build with `kk = k` and the tail loop *present* runs slow (4.83).
Folding the tail into a single epilogue instead of two does not help either (5.12). **It is the presence
of the scalar tail loop in the tile function**, which reaches `Aat`/`Bat` after the main loop and
changes what GCC's register allocator does with the 4x3 NEON tile — the same fragility PR 1 exists for,
approached from the other side.

So the tail now has its own `NOINLINE` function, dispatched on `k % KN` *before* the tile. The aligned
path is then instruction-for-instruction what it was before this patch on every ISA, which is exactly
what the x86 note above demands, and the tail path is unchanged in what it computes. Verified on both:

| | x86 (Core Ultra 9 285K, AVX2) | aarch64 (Cortex-A72) |
|---|---|---|
| `A@V` k=1500, where the tail FIRES | 150.7 GFLOP/s against 53.2 unpatched — **2.83x, the win intact** | n/a (1500 % 4 == 0) |
| `QK^T` k=64, aligned | 119.3 against 118.8 shipped / 120.7 unpatched | 7.38 against 7.54 unpatched |
| `proj` k=768, aligned | 162.0 against 162.2 / 161.9 | 8.43 against 8.38 |
| whisper-small end to end | 2.076 s against 2.086 s — unmoved | 36.4 s against 46.4 s — **1.27x** |
| `tests/ci/test_tinyblas_gemm` | 113/113 | 113/113 |

**And the rule this patch is the reason for:** a diff in `cmake/patches/` is not done until this file
carries a number from an x86 box **and** one from the Pi, even when one of them is "no change". This
one shipped with only the first, and cost the reference device 1.3-1.75x for four days
([Retro-019](../../docs/retros/retro-019-a-patch-measured-on-one-isa.md)).

**Accuracy.** Against a double-precision reference over every `k` in [1, 40] plus 63/64/65 and
1496–1504, the worst relative error is **2.6e-05**, and the aligned `k` sit in the same place as the
unaligned ones (k=1496 2.1e-05, k=1500 2.3e-05) — f32 accumulation noise, not a dropped term. loom's
whisper gate (encoder against HuggingFace) moves `max/absmax` 4.19e-04 → 4.34e-04 against a 1e-3 limit
and `mean_abs_diff` 9.36e-06 → **9.33e-06**, passing 67/67 either way.

**Testing.** The same shape as PR 3's test on the other axis: `k % KN` over its whole residue range
against a double-precision reference, including `k < KN` (which still declines), and whisper's own
`k = 1500`. Verified red by dropping the tail accumulation — 16 of 113 checks fail, at exactly the
unaligned `k`, by 2e-02 to 1.7e-01 against a 1e-5 tolerance.

**What a reviewer should push on.** (1) Whether the two-epilogue split is worth its duplication —
the measurement above is the argument, and it is compiler-specific enough to want a second data point.
(2) The f16/bf16 instantiations keep the rejection, so a reviewer may want the same treatment there
via `GGML_FP16_TO_FP32`. (3) This lands on top of PR 3; the two are independent in mechanism but touch
adjacent lines of the same function, so they want reviewing together.

## PR 12 — `sgemm`: a job should own a whole cache line of `C`, not a quarter of one

`llamafile_sgemm`'s F32 path picks its row block from what divides `m`:

```c
if (m % 16 == 0 && (m/16 >= params->nth)) mnpack<4, RN, 4>(...);   // 4 tiles of 4 rows
if (m % 8  == 0)                          mnpack<4, RN, 2>(...);
if (m % 4  == 0)                          mnpack<4, RN, 1>(...);   // one tile of 4 rows
```

`gemm()` hands ONE JOB the rows `[ii, ii + BM*RM)`, and `C` is `m`-contiguous — so a job's store to one
column of its range is `BM*RM*4` **bytes** wide. At `BM = 1` that is **16 bytes, a quarter of a cache
line**, and four threads write four quarters of the same line for every column of the whole matmul.

**It does not make the kernel slower. It stops it threading.** Core Ultra 9 285K (24 cores, no SMT),
`m = n = 1500, k = 64`, 12 head-slices — the shape of whisper-small's `QK^T` — against the same shape
padded, so the only difference is which branch above is taken:

| `m` | branch | `BM` | bytes of `C` per job | 1 thread | 4 threads | scaling |
|---:|---|---:|---:|---:|---:|---:|
| 1496 | `m % 8 == 0` | 2 | 32 | 29.4–30.5 ms | 21.2–21.9 ms | 1.40x |
| **1500** | `m % 4 == 0` | **1** | **16** | 29.6–30.9 ms | 30.0–31.5 ms | **0.98x** |
| 1504 | `m % 16 == 0` | 4 | 64 | 29.4 ms | 10.6–11.0 ms | **2.75x** |

Monotone in the fraction of a line a job owns. `perf stat -e task-clock` reports **3.65 CPUs utilised**
in the 0.98x row: the threads are running, they are just passing a line back and forth.

**`m` is a sequence length in every attention matmul** — a number nothing rounds — so this is the
common case rather than a corner. whisper-small's encoder runs 1500 frames, which is `4 mod 16`, the
worst residue there is.

**The fix takes a prefix instead of demanding divisibility**, which is PR 3's trick on the other end of
the same axis: run `m - (m % 16)` at `BM = 4` and finish the 0/4/8/12 leftover rows in a separate
column-split loop that keeps the same `4 x RN` tile. It is guarded on `nth > 1`, because false sharing
needs a second thread by definition and a patch that cannot help at one thread should not be able to
hurt there either — at `nth == 1` the schedule is instruction-for-instruction what it was before.

| `m = 1500`, 285K | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---:|---:|---:|---:|
| before | 29.38 ms | 30.30 ms | 29.40 ms | 19.90 ms |
| after | 29.38 ms | 24.98 ms | **15.00 ms** | **8.12 ms** |
| | 1.00x | 1.21x | **1.96x** | **2.45x** |

and the `m = 1504` control is flat to within 1% at every thread count, which is what says the patch
only reaches the branch it is aimed at.

**In model**, whisper-small on `jfk.wav` at 4 threads, `$LOOM_PROFILE`, four interleaved rounds per arm:
the `MUL_MAT 1500 x 1500` bucket goes **391.2 → 185.9 ms (2.10x)** and it is the **only** bucket that
moves by more than 1.2 ms out of 40 — the six dense GEMM groups, `SOFT_MAX`, `CONT` and the rest are
all within noise. End to end **4.050 → 3.858 s, 1.050x** at 4 threads, 1.056x at 8 (2.570 → 2.433 s)
and 1.011x at 2.

**Bit-identical output, which is the point of it being a SCHEDULING change.** Each output is still one
dot product accumulated over `k` in the same order; only which thread computes it and how many rows a
job owns have moved. FNV-1a over the whole result buffer agrees between the two builds at
`m = 1492/1500/1501/1504` x 1/4/8 threads — so no byte-identity gate baseline needs re-recording, and
the accuracy question PR 11 had to answer does not arise here.

**Both ISAs, per the standing rule this file learned the hard way (PR 11).** aarch64 is **no change
at `m = 1500`** — a Raspberry Pi 4 threads 3.5x at *every* one of the three `m` above, so there is
nothing there to fix: 133.10 ms before against 133.65 after, 4 threads, ABBA-interleaved medians of ten.

> **CORRECTION (2026-08-30, loom P4.26): that is true at `m = 1500` and NOT true in general.** The
> paragraph above is the only aarch64 number this patch was ever given, and `m = 1500` is the one shape
> where the new branch changes nothing. At the small `m` a convolutional model actually runs —
> VITS's `m = 96 / 100 / 199`, every one of which *does* take the new branch — the same board measures
> **1.032x and 1.016x SLOWER** over two ABBA-interleaved rounds, ~27 ms on a 1.1 s synthesis. Attributed
> by profile: `CONV_2D` +22.6 ms and `MUL_MAT` +5.4 ms per synthesis, because loom lowers convolution
> and transposed convolution through this same `sgemm` (`ggml-0004`, `ggml-0009`) — so a change written
> for attention reaches every convolutional model. **An upstream reviewer should be given both numbers**,
> and the open question is whether a predicate exists that keeps the x86 2.75x without the aarch64
> 2.4%; gating the new branch on `!__aarch64__` is the fallback, not the answer. This is a second
> instance of PR 11's own lesson, one level down: a number per ISA is not enough, it has to be a number
> per SHAPE CLASS the branch is enabled for. Four small cores behind a shared 1 MB L2 do not pay for a
contended line the way a 24-core mesh does. A 2-core Ryzen 3 3250U cannot resolve it either (±40%
spread on that box; see the dev-box noise floor in loom's Retro-012).

**Testing.** `tests/ci/test_tinyblas_gemm` gains the whole residue class of the 16-row split —
`m = 1488/1492/1496/1500/1501`, plus an `n` that does not divide evenly across the threads so the
ragged column slice is exercised — and the element check gains a window around the seam. **That window
is load-bearing:** the leftover rows sit in the MIDDLE of the matrix, not at its edge, and with the
window disabled a sabotage that skips the first leftover tile drops from 8 failing checks to 2.
Verified red three ways: leftover rows never computed (7 checks fail, by 1e1–1e2 relative), the ragged
column slice never finished (3 fail), and the first leftover tile skipped (8 fail). 137/137 green with
all three removed, on x86 and on the workstation.

**What a reviewer should push on.** (1) The `nth > 1` guard makes the patch invisible on a
single-threaded benchmark, which is how this went unnoticed — a reviewer may reasonably want it
unconditional, and the argument against is that it buys nothing measurable there. (2) `BM` remains a
cache knob as well as a sharing knob, and the two now disagree about small `m`; the `m16/16 >= nth`
condition is inherited unchanged rather than re-derived. (3) The residual gap — `m = 1500` reaches
15.0 ms where `m = 1504` reaches 10.9 — is `ldc` alignment, not sharing: 1500 floats is 6000 bytes, so
a 64-byte store still straddles two lines on odd columns. Padding `C` would close it and is a bigger
change than this.

## Not a PR here, but upstream should know: `apply_unary_op` splits over rows, so `nrows = 1` is all barrier

**No patch in this directory depends on this.** It was found by loom P4.25, which built a patch to
thread ggml's cheap-unary list, measured it out end to end (Epic-05 §5, Retro-012) and dropped it.
The *finding* survives the patch and is worth carrying upstream on its own, because it is a property
of code that ships today.

`apply_unary_op` (`ggml-cpu/unary-ops.cpp`) takes its slice from `get_thread_range`
(`ggml-cpu/common.h`), which divides `ggml_nrows(src0)` by `nth`:

```c
const int64_t nr = ggml_nrows(src0);
const int64_t dr = (nr + nth - 1)/nth;
const int64_t ir0 = dr*ith;
const int64_t ir1 = MIN(ir0 + dr, nr);
```

At `nr = 1` thread 0 gets the whole tensor and threads `1..nth-1` get an **empty range** — every one
of them still enters the node, synchronises, and does nothing. `GELU`, `GELU_ERF`, `GELU_QUICK`,
`SILU` and `XIELU` are given `n_tasks = n_threads` unconditionally in `ggml_get_n_tasks`, so any
one-row tensor of those ops pays a full barrier for a single-threaded computation today.

**Measured**, `ne0 = 256`, four threads against one, Raspberry Pi 4B, through ggml's own threadpool
with 256 nodes per graph so the pool wakes once (`scripts/bench18.cpp` in loom.cpp):

| `nrows` | 1 | 2 | 3 | 4 | 8 | 16 | 64 | 192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| speedup | **1.00x** | 1.75x | 2.36x | 2.91x | 3.36x | 3.69x | 3.89x | 3.92x |

and on a Ryzen 3 3250U the `nrows = 1` row is **0.63x** — an outright loss. One row is a cliff; two
rows already win.

**The fix is one line and cannot make anything slower**: `n_tasks = MIN(n_threads, ggml_nrows(node))`
for the `GGML_OP_UNARY` branch. A thread handed an empty range does no work, so capping at the row
count removes only barrier participation.

**How much it matters depends entirely on the model, and in an LLM it does not.** Every `UNARY` bucket
in whisper-small, Qwen3-0.6B, LFM2 and the NeMo encoders is many-rowed (the only one is
`[3072, 1500]`). It bites vocoders: Kokoro issues **870 one-row `UNARY [256, 1]` nodes** per
synthesis, and when loom's P4.25 patch threaded those ops without the cap that bucket went
**5.8 -> 179.9 ms**. So this is worth fixing pre-emptively rather than because it is costing anything
in llama.cpp today.

**A second floor a reviewer should ask about**, from the same measurements: at 24 threads on a Core
Ultra 9 285K the per-node cost bottoms out at **~1.9 us**, so threading a unary below ~16K elements is
a loss there regardless of row count. `n_tasks` has no work floor for these ops at all. Any future
change that threads more unary ops needs one, and it is machine-dependent — which is a large part of
why loom did not carry its patch.
