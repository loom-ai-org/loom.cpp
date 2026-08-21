# Upstreaming these patches to ggml-org/ggml

Five diffs, each independently useful and independently reviewable. They are written against the pin in
`cmake/GgmlPin.cmake` (**v0.19.0**, commit `30bf8685`), so the first step for any of them is a rebase
onto `master` — the files move rarely, but `sgemm.cpp`'s tile-selection block and `ops.cpp`'s
`ggml_compute_forward_conv_2d_impl` are both areas that see occasional churn.

**Order and independence.** 1, 2 and 3 touch `llamafile/sgemm.cpp` and are independent of each other,
though 1 and 2 only pay off together (see below) and are best sent as one PR or two linked ones. 4 and
5 touch the CPU convolution; **5 depends on 4** (it uses the `ldc` mul_mat helper 4 introduces, and its
staging only makes sense once 4 has made the op batch-oriented). A reviewer taking 4 without 5 is fine;
5 without 4 is not.

**Every number below is a Raspberry Pi 4B (Cortex-A72, 4 cores @ 1.8 GHz, 1 MB shared L2, Debian
aarch64, gcc 14.2) unless it says otherwise; x86-64 numbers are a Ryzen 3 3250U (AVX2, 2 cores), median
of seven runs pinned to the physical cores.** The workload is a VITS TTS vocoder — an all-convolutional
generator, F32 throughout, one image per batch, `KH = 1`. The benches are in `scripts/` in this repo
(`bench6.cpp` GEMM, `bench7.cpp` register tiles, `bench9.cpp` conv lowering) and are self-contained
against a built ggml.

**What a reviewer is likely to ask, for all five: measurements on hardware neither of these two boxes
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
  (none of these do; enabling `GGML_LLAMAFILE` at all does, by ~3e-7 relative, but that is a build
  option, not a patch).
* The two machines, and the explicit request for a third.
* For 4 and 5: that they are CPU-backend-only and leave every other backend's path untouched.
