# Upstreaming these patches to ggml-org/ggml

Sixteen diffs, each independently useful and independently reviewable. They are written against the pin in
`cmake/GgmlPin.cmake` (**v0.26.0**, commit `d7cb5741`), so the first step for any of them is a rebase
onto `master` — the files move rarely, but `sgemm.cpp`'s tile-selection block and `ops.cpp`'s
`ggml_compute_forward_conv_2d_impl` are both areas that see occasional churn.

**Rebased onto v0.26.0 on 2026-10-08** (from v0.19.0, 544 upstream commits). Two patches left
because upstream now has them: **16** (llama.cpp #29561, `46fc5b3b`, left and circular `PAD` on Metal,
reading the source through `nb00` -- everything 16 did) and **22's backport, `ggml-0025`** (#26708,
`95414e59`). The bump also brings in llama.cpp #28253 (`04d63b27`, type-aligned `GET_ROWS` and aligned
descriptor binding for every binary op on Vulkan), which was the fix for EnCodec/DAC aborting on a
16-byte-aligned NVIDIA device, so it needs no patch here. What changed in the rest:

* **11 still owns F32 on x86**, although upstream (#29806, `ebefd1bb`) now admits a ragged `k` there
  for every dtype and finishes it with a masked vector load. Measured at the bump (Ryzen 3 3250U, AVX2,
  one thread, eight paired rounds): the two tails are equally accurate against f64 (upstream's worst
  relative error is equal or lower at every k tried), equally fast on aligned shapes (0.94-1.00x, inside
  the noise), and upstream's is ~9% faster on a ragged one (k=100). But the summation order differs, and
  CosyVoice3's free-running waveform gate moved 10x on that alone (RMSE 2.7e-3 -> 2.7e-2 against a 2e-2
  bound; every teacher-forced stage still passed). With 11 kept for F32 the engine is bit-identical to
  the v0.19.0 pin on that test, so 11 keeps F32 everywhere and upstream's tail serves F16/BF16 on x86.
  Delete 11 only together with re-deriving the references it was measured on.
* **2**'s loop bound is `k - k % KN` on x86 so upstream's tail finishes a ragged F16/BF16 `k`; F32 never
  reaches it (11 dispatches first), and elsewhere the bound is `k`.
* **13**'s CUDA `supports_op` hunk gives `CONV_2D` its own `case`. It used to share a `return` with
  `IM2COL`/`IM2COL_3D`, where `op_params[6]` is `is_2D` -- so the check declined every 2-D im2col on
  CUDA and sent it to the CPU (slower, not wrong).
* **15**'s function-constant id moved from 1700 to 2400: upstream took 1700 for `FC_NORM`.
* **14, 15, 21** moved into upstream's per-op Metal sources (`kernels/conv.metal`, `kernels/mul_mm.metal`,
  #26561); kernel-to-library routing is derived from `functionNames`, so the new names need no table.
  21's tensor-API carve-out now sits beside `ggml_metal_op_mul_mat_use_mm()`. Upstream's new few-row
  MMA path for F32 uses `simdgroup_float8x8`, so `GGML_PREC_F32` needs nothing there.
* **20** is re-written against upstream's new matmul pipeline map (`vk_matmul_pipeline_key`): a
  `prec_f32` key field, built from the `_fp32` SPIR-V with the same scalar tiles as upstream's BF16
  fallback, and preferred by `ggml_vk_get_mul_mat_mat_pipeline_map` for `GGML_PREC_F32`.
* **23** keeps upstream's new `warp_size` argument to `ggml_cuda_should_use_mmvf`.

**14, 15 and 16 are the Metal set and depend on nothing else here** — different backend, different
files, and independent of each other: 14 fixes `conv_transpose_1d`'s dispatch, 15 rewrites `conv_2d`'s
inner loop, 16 teaches `pad` a leading pad, and a re-profile of any one moves exactly one op's bucket.
Any of them is a reasonable one to send first. **16 is the smallest thing in this file** — four kargs
fields, an index shift, and a `supports_op` that stops declining — and it is the only one whose case
is made by the SCHEDULER rather than by the op: it removes a fallback, and the fallback was dragging
55 of a graph's 56 splits onto the CPU with it.

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

**Submission status (2026-08-23; 14 and 15 added 2026-09-02).** PRs 1, 2 and 3 have been **sent
upstream**. 4 through 9 have not, and are deliberately held: each of 5, 6, 7 and 9 depends on 4, and 4 itself reads more naturally
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

**It has since been found wrong on a second machine, and the way it is wrong is worth a reviewer's
attention.** On an ARM1176 (16 KB L1, no usable L2) the peak is at **64 KB**, worth 1.061x over
512 KB — but the curve has a **floor as well as a ceiling**: 16 KB, the value this rule's own
reasoning asks for on that core, is as bad as 512 KB, because a batch that small leaves the GEMM eight
columns wide. So "half the last level" is not a rule that extrapolates down; the patch now carries
`GGML_CPU_CONV2D_PATCH_BUDGET` (bytes, 0 = no cap) so the constant can be swept per machine without a
rebuild, which is how that curve was found.

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

**An escape hatch for the cache rule, added after the floor was caught being wrong.**
`GGML_CPU_CONV1D_BUDGET` overrides the whole detection with a plain byte count; zero declines the
direct path for every shape. It exists because the floor is not merely approximate on a machine that
reports nothing — it can be off by a factor of thirty-two. An ARM1176 (Raspberry Pi Zero) has a 16 KB
L1 and no L2 the CPU can use, `sysconf` reports 0 for every level, and the 512 KB floor therefore said
yes to every convolution in a TTS model; routing three of its shapes by their real weight footprint
instead is worth **1.223x on a whole synthesis**. The important part for a reviewer is *why a second
switch*: `GGML_CPU_DISABLE_CONV_HEURISTICS` already existed and could not be used to find this,
because it also disables PR 4's patch-batch budget — which turned out to be independently mis-sized on
the same machine and worth another 1.061x. One switch per decision, or neither number exists.

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

**And the two packed panels must not be a whole number of cache ways apart, which by default they
are.** This op packs the kernel at `wdata` and the transposed activation at `wdata + nk`, and the GEMM
then walks four rows of each with `lda = ldb = k`. `nk` is `K*Cout*Cin`, so it is almost always a
round number of KB — on a VITS vocoder's first upsample it is exactly 2 MB — and on any machine whose
L1 way divides it the a-streams and the b-streams land in the same sets and evict each other.
Measured on a Pi Zero W (16 KB 4-way L1, so a 4 KB way) at that node's GEMM shape, m=2048 k=256 n=64:

| | MMAC/s |
|---|---|
| panels adjacent, as this patch first wrote them | 82.1 / 81.7 |
| second panel skewed by 16 floats | **286.4 / 285.6** |
| `k` loop blocked at 128 or 64, no skew | 81.7 / 83.1 |

The last row is the one for a reviewer: blocking the reduction changes nothing, which is what
distinguishes a set conflict from a capacity miss. End to end the op goes **4813 -> 3236 ms, 1.49x**,
and all three of the model's nodes improve rather than only the one whose `nk` is a round 2 MB. The
output is bit-identical, because it is an address. Padding `lda`/`ldb` instead measures the same and
costs a stride parameter threaded through `ggml_call_mul_mat_ldc`; the skew costs one addend and a
matching four bytes in `ggml_graph_plan`.

No measurable effect on an AVX2 x86 box (min-of-7 on the op, 44.0 against 45.1 ms, inside its own
spread), which is what should be expected: the exposure scales with how little associativity there is
to spare.

---

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

**AND THE RAGGED PREFIX IS GATED ON `k <= 256`, WHICH IS THE OTHER HALF OF THIS PATCH.** The `m = 1500`
number above is the only aarch64 measurement this patch had for a week, and `m = 1500` is the one
shape where it changes nothing there. Shipped without a `k` guard it cost a Raspberry Pi 4 **2.4% on
VITS** — 27 ms of a 1.1 s synthesis, of which a per-node profile put 22.6 ms in `CONV_2D`.

What the branch removes is a **per-output** cost — `m*n` contended stores, one per element of `C`,
however long the contraction is. What it adds is **per-work**: a job's row block is four times taller,
so it holds `RM*BM*k*4 = 16k` bytes of `A` instead of `4k`, and there are four times fewer jobs to
balance. **The benefit decays as `1/k`; the cost does not.** Sweeping `k` at `m = 284, n = 384`,
4 threads, both arms in one process behind a run-time switch, ABBA-interleaved, ratio per round:

| `k` | 64 | 128 | 192 | 256 | 384 | 512 | 768 | 960 | 2304 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Core Ultra 9 285K | **2.035** | **1.317** | 1.046 | 1.030 | 1.005 | 0.996 | 0.987 | 0.983 | 0.974 |
| Cortex-A72 | 1.014 | 1.011 | 1.005 | 0.997 | 0.995 | 0.977 | 0.959 | **0.912** | 0.901 |

Monotone on both, crossing 1.0 within a factor of two of each other — **so the predicate is `k`, not
the ISA**, and `!defined(__aarch64__)` would have been wrong in both directions (it keeps the x86
losses at `k >= 512` and throws away the aarch64 wins at `k <= 192`). 256 is the largest power of two
inside the winning region on both machines. `m % 16 == 0` keeps the schedule it always had, at every
thread count and every `k`, so the patch is strictly additive:

```c
const bool ragged_prefix = (m % 16 != 0) && params->nth > 1 && k <= 256;
if (m16 > 0 && (m16/16 >= params->nth) && (m % 16 == 0 || ragged_prefix)) { ... }
```

**Why a reviewer of an attention patch should care about `k`:** `m` is a sequence length and `k` is a
head dimension in attention, which is where this was measured — but the ggml CPU backend lowers 1-D and
2-D convolution through this same `sgemm` (in loom's tree via the conv patches above; upstream via
`ggml_conv_*`'s im2col + `mul_mat`), and there `k` is `in_channels * kernel_width`: 576 to 2304 in one
small TTS vocoder, which issues **10.9 of its 14.4 GFLOP** through matmuls this branch newly accepts.
With the guard, whisper's `QK^T` on the 285K goes from 1.809x to **1.941x** against the unpatched
schedule (the shapes it now declines were dragging the average down) and the Pi returns to parity with
the unpatched tree: 12 paired ABBA rounds, cooled to a fixed 60 C, **1.003** with p10 0.985 and
p90 1.018.

This is PR 11's lesson one level down: a number per ISA is not enough — it has to be a number per
**regime the branch is enabled for**, and the ISA is only one axis. Four small cores behind a shared
1 MB L2 also do not pay for a contended line the way a 24-core mesh does, so there is nothing for this
patch to win on the Pi at any `k`; a 2-core Ryzen 3 3250U cannot resolve it either (±40% spread on that
box).

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

## PR 13 — `conv_2d`: accept a quantized kernel, because the direct path already repacks it

`GGML_OP_CONV_2D` takes an F32 or F16 kernel. A quantized one is not merely unsupported — it is
**unrepresentable**, and for a reason worth stating: quantization blocks are laid along `ne[0]`, and for
a kernel stored the way it is declared, `[KW, KH, IC, OC]`, `ne[0]` is the kernel WIDTH — 1, 3, 5, 7 —
which no block size divides. So a producer that wants a block-quantized convolution has to fold the
spatial and channel axes into `ne[0]` first, giving `[IC*KH*KW, OC]`, and a folded kernel then carries
none of the geometry this op reads off `a->ne`.

The result is that a convolutional model gets nothing from quantizing except a smaller file. Measured on
a VITS vocoder at Q4_0 — 81.7 -> 30.6 MB — one synthesis went **0.50 -> 1.04 s on x86-64 (2.08x) and
1.03 -> 2.18 s on a Cortex-A72 (2.13x)**, because the whole model dropped out of `CONV_2D` into
`im2col` + `mul_mat` and gave up the direct 1-D sweep (PR 6) and the two fusions (PR 5, PR 7) with it.

**The fix is one dequantize, and the reason it is only one is that the direct path never reads the
kernel as stored.** `ggml_conv_1d_direct_run` repacks the whole kernel into an F32 scratch buffer before
it computes anything, and the register-tiled inner kernel reads only that. So a quantized kernel needs
no quantized inner loop, no `vec_dot`, and no activation quantized to `vec_dot_type` — it needs a
`to_float` into a buffer that already exists.

Three pieces:

* **`ggml_conv_2d_direct_packed`** — `ggml_conv_2d_direct` plus `kw`, `kh`, `ic`, stored in `op_params`
  6..8. `GGML_MAX_OP_PARAMS` is 64 bytes and `CONV_2D` used 24, so this needs no new op code, no new
  dispatch entry and no ABI change. A zero `kw` is what says "declared layout", which is every existing
  graph. Reshaping the folded tensor back is not an alternative: a `[IC*K, OC]` Q4_0 tensor reshaped to
  `[K, 1, IC, OC]` puts K=3 on `ne[0]`, which is not a multiple of the block size, and the resulting
  `nb` is nonsense.
* **`ggml_compute_forward_conv_2d_impl`** dequantizes off the FRONT of the work buffer and **re-enters
  itself** with a stack `ggml_tensor` that is an ordinary contiguous F32 kernel in the declared layout,
  and a `ggml_compute_params` holding what is left of the buffer. Nothing below that point knows: the
  direct sweep, the phase-major variant, the batched `im2col` fallback, all three fusions, every thread
  split. The fold is a pure reinterpretation of the stored order — element `(kx, ky, ic, oc)` sits at
  `oc*(IC*KH*KW) + ic*(KH*KW) + ky*KW + kx` either way — so the dequantized buffer read with the
  declared strides IS the declared kernel, with no permutation and no second pass.
* **`ggml_graph_plan`** adds the dequantized kernel's bytes to `GGML_IM2COL_WORK_SIZE` for a quantized
  `src0`, since the buffer now holds both.

**The re-entry is the design decision, not an implementation detail.** The alternative — threading
`kw/kh/ic/oc` through every function below — leaves each predicate free to price the kernel by its
STORED size, and one of them must not: `ggml_conv_1d_direct_ok`'s cache budget is
`IC*OC*KW*sizeof(float)`, and `wp` is F32 whatever the kernel is. Priced by 8.4 MB of Q4_0 rather than
59.5 MB of F32, that predicate silently admits shapes it was tuned to reject, which is the one way this
change could ship something SLOWER than what it replaces. Re-entering with a rebound tensor makes every
predicate below price the dequantized size because that is the only size it can see.

**What the dequantize costs, measured over all 59.5 MB of that model's convolution weights at once** —
i.e. a whole synthesis's worth of kernel packing, not one convolution (Ryzen 3 3250U):

| | dequantize | the F32 pack it replaces | delta |
|---|---:|---:|---:|
| Q4_0 | **10.3 ms** (5.77 GB/s of output) | 8.5 ms | **+1.8 ms** |
| Q8_0 | 9.3 ms | 8.4 ms | +0.9 ms |

0.4% of a 500 ms synthesis, and the pack loop reads LESS memory in the quantized case (8.4 MB rather
than 59.5 MB) while writing the same amount, which is why the delta is arithmetic rather than traffic.

**End to end, and on both ISAs** — `paired_arms.py`, interleaved ABBA rounds, one binary per arm swapped
by `LD_LIBRARY_PATH`, the Pi cooled to a fixed 60 C before every arm:

| VITS at Q4_0, 4 threads | rounds | before | after | median ratio | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| **Cortex-A72** (Pi 4B) | 7 | 2.197 s | 1.059 s | **2.075x** | 2.060 | 2.126 |
| **x86-64** (Ryzen 3 3250U) | 9 | 1.417 s | 0.681 s | **2.101x** | 1.927 | 2.441 |

which is the whole 2.08x / 2.13x back. The x86 band is wide because that box is a noisy 2-core laptop
part; the Pi's band is 3% wide and is the one to read.

**The end state is a quantized model at F32 speed, which is the claim worth checking.** Because the
dequantize happens before either lowering, a quantized convolution then executes the *same instructions*
as an F32 one. Against its own F32 export, same harness, per-round ratios:

| VITS Q4_0 / its own F32 | rounds | ratio | per output sample |
|---|---:|---:|---:|
| Cortex-A72 | 7 | 0.950 | **1.029x** |
| x86-64 | 9 | 0.923 | **1.000x** |

(the two exports produce 67584 and 73216 samples — VITS durations are a `ceil()` of a float and the
arithmetic changed — so the raw ratio understates and the normalised column is the honest one). **F32 in
11.7 MB rather than 62.8.**

**The F32 path is bit-identical**, which is what a branch guarded on `ggml_is_quantized` has to show:
FNV-1a over the whole audio agrees before and after on both machines and in **both** lowerings —
`68c3229eab373455` / `bbb3b173ce35fe2a` (x86, heuristic on / `GGML_CPU_DISABLE_CONV_HEURISTICS=1`) and
`aa320f8a1377a92a` (aarch64) — so no byte-identity baseline needs re-recording.

**It is also more accurate, not less.** Dequantize-then-F32-FMA replaces a path that dotted Q4_0 weights
against activations rounded to Q8_0.

**Backends.** The CPU's `supports_op` gains "or quantized with a `to_float`"; **CUDA's declines a
quantized or folded `src0`**, which today it would accept (`ggml_is_contiguous(src0) &&
ggml_is_contiguous(src1)`) and then read as a wrong-shaped F32 kernel — a silent wrong answer, not a
failure. Vulkan already declines it twice over (a type test, and `cout == op->ne[2]`). Verified: a
folded Q4_0 VITS on Vulkan still runs with **0 fallback nodes** in both topologies, because the caller
asks `supports_op` and keeps its own `im2col` lowering when the answer is no.

**Where this design is NOT legal.** It dequantizes **per call**. That is right for a convolution that
runs once per utterance — every convolutional model in this space is a TTS vocoder or an ASR encoder —
and wrong for one inside a decode loop, where it would be paid per token. `SSM_CONV` is a different op
and does not route here.

**What a reviewer should push on.** (1) `op_params` 6..8 is a side channel; a `GGML_OP_CONV_2D_PACKED`
would be more explicit and would cost a dispatch entry and a `supports_op` case in every backend.
(2) The producer has to fold the kernel itself, and nothing in ggml helps it or checks it beyond the
three asserts in `ggml_conv_2d_direct_packed`. (3) A per-call dequantize is a policy, and there is no
mechanism to cache it across calls if someone puts one of these in a loop.

## PR 14 — `metal`: `conv_transpose_1d` dispatched one thread per threadgroup

`ggml_metal_op_conv_transpose_1d` ends with

```c
ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);
```

`OL * OC` threadgroups, each of **one thread**. Every threadgroup therefore occupies a single lane of a
32-wide SIMD group, and 31 of every 32 lanes on the GPU are idle by construction. `kernel_conv_2d`, 400
lines above it in the same file, already does the opposite — `max_theads_per_threadgroup` capped at 256,
grid divided by the same factor — so this is a gap rather than a design.

**It is invisible from anywhere except a per-op profile.** The op is fully supported, so it produces no
split, no CPU fallback and no warning; and it is a *small number of nodes*, so it is missed by anything
that ranks by node count. In a VITS synthesis it is **three nodes of 1308 and 45% of the run**.

Measured on an Apple M1 Pro (16-core GPU, 5.31 TFLOP/s F32), over those three vocoder upsamples —
`[286,256] K=16 s0=8 -> [2296,128]`, `[2288,128] K=16 s0=8 -> [18312,64]`, `[18304,64] K=8 s0=4 ->
[73220,32]`, **1.50 GFLOP** in total:

| | time | rate | of peak |
|---|---:|---:|---:|
| stock | 224.7 ms | 6.7 GFLOP/s | **0.13%** |
| with this patch | **7.8 ms** | 193 GFLOP/s | 3.6% |

**29x on the op, and 1.77x on the whole model** — an F32 VITS synthesis goes 494.1 -> 278.7 ms, a Q4_0
one 350.2 -> 149.7 ms. For scale, the same three nodes take 21.6 ms on **one** CPU core with PR 8 and
PR 9 applied, so before this the entire GPU was losing 10x to a single Firestorm core.

The change is three edits and no new kernel:

* `ggml_metal_kargs_conv_transpose_1d` gains an `OL` field, used only for the tail bounds check. The
  grid is rounded up and none of the lengths that matter divide evenly.
* `kernel_conv_transpose_1d` computes `j = tgpig[0]*ntg.x + tpitg.x` instead of `j = tgpig[0]`, returns
  early when `j >= OL`, and writes to `j * nb0` instead of `tgpig[0] * nb0`. The `tgpig[1]` / `tgpg[1]`
  channel indexing is untouched, so the arithmetic is byte-for-byte what it was.
* the dispatch takes `nth = min(max_theads_per_threadgroup, 256)` threads on x and `(OL + nth - 1)/nth`
  threadgroups.

**Verification.** `test-backend-ops test -o CONV_TRANSPOSE_1D` is **116/116 on MTL0** against the CPU
reference (`perf` mode registers no case for this op, which is its own small gap). End to end the VITS
waveform digest is bit-identical to stock on both an F32 and a Q4_0 export, and a re-profile is a clean
control: the `CONV_TRANSPOSE_1D` bucket collapses 224.7 -> 7.8 ms while `CONV_2D` sits unchanged at
241 ms.

**What a reviewer should push on.** (1) 256 is copied from `kernel_conv_2d` rather than tuned; the op is
memory-bound enough that it is unlikely to matter, but nobody has swept it. (2) A threadgroup now spans
`nth` consecutive output positions of one channel, which is the coalescing-friendly axis, but the `src0`
weight reads are still per-thread and unstaged — a threadgroup-memory tile for the kernel would be a
further win and a much larger patch. (3) This does not touch `supports_op`, the F16 path or the
`col2im_1d` sibling below it.

**Where the rest of that model's time went, since a reviewer will ask.** After this patch, 86% of the
same synthesis is `GGML_OP_CONV_2D` — 241 ms for 16.89 GFLOP, **70 GFLOP/s, 1.3% of peak**, from a
`kernel_conv_2d` that is one thread per output element with two global loads per FMA and no reuse of
any kind. **That is PR 15, below.** The two are independent: 15 does not touch
`conv_transpose_1d` and 14 does not touch `conv_2d`, and a re-profile of either moves exactly one
bucket.

## PR 15 — `metal`: `conv_2d` is one thread per output element, at 1.3% of peak

`kernel_conv_2d` gives each thread one output element and loops `IC*KH*KW` doing **two global loads
per FMA with no reuse of either operand** — no register tile, no staging, no vectorisation.
Arithmetic intensity 0.25 FLOP/byte. Over the 117 convolutions of one VITS synthesis (16.88 GFLOP)
on an M1 Pro that is **70 GFLOP/s: slower than a single CPU core**, and 1.3% of the part's
5.31 TFLOP/s spec. After PR 14 it is **86% of the whole model**.

**Divide by an achievable number, not the spec.** A pure FMA loop with sixteen independent chains
reaches **2.1 TFLOP/s** on this part, not 5.31, so the stock kernel is at 3.6% of what a kernel can
actually get and the headroom below is quoted against 2.1.

**The change.** A thread now accumulates `OP_CONV_2D_NC = 8` output channels at one output position:
each activation it loads feeds eight FMAs, and the eight weights beside it are at an address every
lane in the threadgroup shares, so one cache line serves the SIMD group. Loads per FMA go 2 ->
1.125. The grid becomes the output's own shape — x over columns, y over blocks of 8 channels, z over
rows and batch — instead of a flat element count, and `KW` becomes a function constant (`FC_CONV_2D`)
so the tap loop unrolls for the widths a convolution actually uses.

**The half of it that is not the tile: 32-bit element indices.** The same tile written with one
`device const TK *` per output channel — sixteen registers of pure address — runs **68.8 ms against
45.6**, and `maxTotalThreadsPerThreadgroup` says why: **704 threads against 896**. On this GPU
occupancy is latency hiding, and the addresses were competing with the accumulators. The fast path
therefore declines, at threadgroup granularity, anything it cannot address in 32 bits (a stride that
is not a whole number of elements, or a tensor past 4G elements) and falls into the original 64-bit
loop, which is kept as the general path and also carries the edges and partial channel tiles.

Measured over the 31 distinct shapes of that synthesis, at the multiplicity the graph issues them:

| | total | rate | |
|---|---:|---:|---:|
| stock | 222.3 ms | 76 GFLOP/s | occupancy 1024 |
| + register tile, 64-bit pointers | 68.8 ms | 245 GFLOP/s | occupancy 704 |
| + 32-bit element indices | 45.6 ms | 370 GFLOP/s | occupancy 896 |
| + `KW` as a function constant | 44.8 ms | 377 GFLOP/s | |
| + 128 threads per threadgroup | **43.6 ms** | **387 GFLOP/s** | **5.10x** |

End to end an F32 VITS synthesis goes **278.1 -> 97.6 ms** — 2.85x on the whole model — whisper-small
goes 0.874 -> 0.752 s from the two convolutions its encoder opens with, and the
`CONV_2D` bucket of a calibrated per-op profile goes 241.3 -> 60.1 ms while every other bucket sits
still.

**Verification.** `test-backend-ops test -o CONV_2D` is **2026/2026 on MTL0** against the CPU
reference, covering F16 and F32 kernels, `KH > 1`, stride, dilation, padding and batch — the fast
path here handles all of them and the shapes that reach the general path are exercised by the same
run. The VITS waveform digest is bit-identical to stock on both an F32 and a Q4_0 export.

**What did NOT work**, so a reviewer does not ask for it. Staging the weight tile in threadgroup
memory is **worse at every activation length** (55.2 ms) — the uniform loads were already being served
by one cache line per SIMD group, and the barriers are not free. More than 32 accumulators per thread
spills to scratch and costs up to 5x (8 positions x 8 channels is 302.7 ms, slower than the kernel it
replaces — and `maxTotalThreadsPerThreadgroup` still reads a healthy 896 for it, so that probe finds
register pressure but not spilling). And giving a thread several output *positions* loses despite a
much better ratio: a 4-position x 4-channel tile has **0.5 loads per FMA against this kernel's
1.125** and runs 69.7 ms, because it gives back more occupancy than the ratio buys.

**What a reviewer should push on.** (1) `OP_CONV_2D_NC = 8` and `nth = 128` are swept on ONE part
(M1 Pro, 16-core) and one workload family — an all-convolutional TTS vocoder, `KH = 1`, batch 1. A
wider Apple part or a real 2-D image stack may want different numbers, and there is no autotune here.
(2) A 2-position tile is worth about 6% on the two longest activations and is deliberately not taken,
because it needs a second pipeline family and a threshold on `OW`. (3) The ceiling is still open:
this backend's own `mul_mat` runs the same convolutions at about 1.49 TFLOP/s through an `im2col`
lowering, so a simdgroup-matrix implicit GEMM would be another 2-3x and a much larger patch.
(4) `supports_op`, the depthwise sibling and the quantized-kernel type test are untouched.

## PR 16 — `metal`: `pad` can only append, so every leading pad falls back to the CPU — now upstream

**Deleted at the v0.26.0 bump.** llama.cpp #29561 (`46fc5b3b`, 2026-09-28) does the same thing and
more (circular padding, a test); the record below is kept for the measurement.


`kernel_pad_impl` writes `dst[i] <- src[i]` inside the source extent and zero outside it. That is a
pad whose four LEADING pads are all zero, and `supports_op` says so honestly:

```c
return (ggml_get_op_params_i32(op, 0) == 0) && (ggml_get_op_params_i32(op, 2) == 0) &&
       (ggml_get_op_params_i32(op, 4) == 0) && (ggml_get_op_params_i32(op, 6) == 0);
```

So `ggml_pad_ext` with any leading pad — which is what a symmetric or left-only pad is — runs on the
CPU. `test-backend-ops` shows the same thing from the other side: of its 21 `PAD` cases, the ones with
a non-zero `lp` were reported `not supported [MTL0]`, so no Metal `PAD` case with a leading pad had
ever been executed.

**And the cost is not the op, it is the graph.** A VITS text encoder issues 21 such nodes. Removing
them removes 72 nodes and 54 splits, because `ggml_backend_sched` assigns a declined node's supported
neighbours to the CPU as well rather than copy a tensor across each boundary:

| Apple M1 Pro, one f32 VITS synthesis | splits | CPU nodes | time | FNV-1a of the audio |
|---|---:|---:|---:|---|
| stock | 56 | 72 | 97.9 ms | `bbc58397d238efde` |
| with a leading pad | **2** | **0** | **88.7 ms** | `bbc58397d238efde` |

**1.104x**, medians of seven, the two arms interleaved A-B-B-A over six rounds by swapping
`libggml-metal.dylib` between runs, and the digest identical in all twelve. On a discrete GPU the
same fallback costs a PCIe round trip per node rather than a shared-memory one, so the ratio there
should be larger, not smaller.

**A number NOT to quote from this.** An earlier throwaway of this same change measured
493.3 → 484.7 ms, 1.8%, on a Metal backend whose convolution kernels PRs 14 and 15 had not yet fixed.
The absolute saving carried across that 5x almost exactly — 8.6 ms then, 9.2 ms now — and the
percentage did not. It is the reason this was filed as "cleanliness, not performance" and then turned
out to be worth shipping: the denominator moved, not the patch.

**The change.** Four `int32_t lp0..lp3` on `ggml_metal_kargs_pad`, filled from `op_params` 0/2/4/6;
`kernel_pad_impl` maps `dst[i] <- src[i - lp]` per dimension, copying when every index is in range and
zero-filling otherwise; `supports_op` returns `true`. The `_4` vectorised variant needs no thought
because `is_c4` is hardcoded `false` upstream ("note: this is slower"), but the two conditions it would
now need — `lp0 % 4 == 0` and a dense `nb[0]` — are written into the commented-out line beside it.

**One bug this found, which is the argument for the `nb00` line in the diff.** The kernel indexed the
source as `T[]`, i.e. assuming its fastest axis is dense. `ggml_pad_ext` accepts a permuted view and
the CPU path honours one, so that was already wrong — but every shape that could expose it has a
non-zero leading pad somewhere, and those were all declined, so the bug was unreachable. Allowing them
made it reachable and `test-backend-ops`' `tfrm=2` case at `ne_a=[11,22,33,44]` failed at
**ERR 1.94** the first time this kernel saw it. Reading through `nb00` fixes it and costs nothing
measurable (the 1.104x above is the fixed kernel).

**What it does not do.** `supports_op` now returns an unconditional `true` for non-circular `PAD`, and
that keeps a pre-existing hole rather than adding one: only `kernel_pad_f32` is instantiated, and the
old predicate did not check the type either. Circular padding is still declined, as before.

**Verified:** `test-backend-ops -b MTL0` is **21/21 on `PAD`** where it was 20/21 with the leading-pad
cases enabled and skipped before that, and **13887/13887 over the whole suite**. loom's `ctest -L ci`
is 75/75 on the same machine. whisper-small on Metal is unchanged, which is the control — it issues no
leading pad, so nothing in it should move, and nothing does.


## PR 17 — `conv_2d`: gather the im2col patches a run at a time, not an element at a time

*(`cmake/patches/ggml-0020-conv2d-gather-runs.patch`, sits on top of PR 4 and PR 7)*

**Problem.** The patch gather computes, per ELEMENT, two strided coordinates, a four-way bounds test
and a three-term address — roughly fifteen instructions to move one float. For a 1-D convolution over
a contiguous source all of that is spent walking a straight line: for a fixed `(ic, kx)` the elements
successive patches contribute are a **contiguous run** of the input, `p + kx*dilation - pad` for
consecutive `p`, and the only thing that varies is where they land, a store of stride `knl_n`.

**Fix.** Take the run at a time when the shape allows it — `knl_h == 1`, one image, stride 1, F32,
contiguous source — and resolve the bounds into a leading zero-fill, a body and a trailing zero-fill,
computed once per run instead of once per element. Everything else falls through to the general loop
unchanged.

**Evidence**, a Pi Zero W over a VITS vocoder's seven convolution shapes (`scripts/bench41.c`):

| | per element | across a synthesis |
|---|---|---|
| element at a time | 33-75 ns | 2.17 s |
| run at a time | 18-32 ns | **0.98 s** |

End to end, ABBA on one board: **74.44 / 75.13 s -> 69.15 / 69.05 s, 1.082x**, with `CONV_2D` as a
whole 62697 -> 58159 ms.

**Bit-identical**: the same values to the same places in a different order. Verified by hashing a whole
synthesis on x86 and across all four arms on the board.

**Worth noting for a reviewer, because it cuts against the usual caution.** The gather measured alone
saves 1.19 s and end to end it saves 5.69 s — the isolated number UNDERSTATED it, which is rare. The
element-at-a-time gather was evicting the cache for the GEMM that consumes its output immediately
afterwards, so the locality fix pays twice. A phase that interleaves with another in a tight loop is
not separable by construction.

**What a reviewer should push on.** The fast path duplicates the general loop's semantics rather than
sharing them, so a change to one has to be mirrored in the other; the guard is the honest way to keep
that bounded but it is still two code paths. And the leaky-ReLU fusion of PR 7 is applied in both.

---


## PR 18 — `conv_2d_dw` (WHCN): vectorise the interior of each output row

*(`cmake/patches/ggml-0021-conv2d-dw-interior-vec.patch`, independent of the others)*

**Problem.** `ggml_compute_forward_conv_2d_dw_whcn` runs every output through a scalar loop with two
bounds tests, an index computation and a kernel-type switch per tap. For a 1-D depthwise conv (H=1,
small K, long rows) that loop is the whole cost.

**Change.** Per row, the outputs whose taps all land inside the source row (`[x_lo, x_hi)`, the same
for every row and channel) are computed as one `ggml_vec_mad_f32` sweep per tap over the contiguous
run; only the edge outputs that reach into the padding keep the original loop. Taps are accumulated in
the same (ky, kx) order from 0, so the result is bit-identical to the scalar loop wherever both
contract to FMA (verified on WakeHuBERT tiny, F32 and F16, AVX2 variant).

**Measured** (WakeHuBERT tiny, 2.5 s, 1 thread, Ryzen 3 3250U, min over 150 runs, median of 7
interleaved launches): F32 13.94 -> 12.58 ms, Q8_0 13.79 -> 12.39 ms, F16 19.05 -> 17.79 ms, on top
of loom lowering `CONV_1D_DW` to `ggml_conv_2d_dw_direct` (17.22 ms with the im2col recipe). Only
x86 AVX2 measured; the aarch64/armv6 paths of `ggml_vec_mad_f32` are not yet measured.


## PR 19 — CPU backend: fuse the zero `PAD` before a `CONV_2D_DW` and the bias `ADD` + `RELU` after it

*(`cmake/patches/ggml-0022-conv2d-dw-fusion.patch`, on top of PR 18)*

**Problem.** A causal temporal block lowers to `PAD(left=(K-1)*d) -> CONV_2D_DW -> ADD(bias) -> RELU`.
PR 5/7's detector covers `CONV_2D` only, so around a depthwise conv every one of those is a full pass,
and the per-channel bias ADD is a broadcast along `ne0` that ggml runs ~6x slower than a same-shape
residual ADD (0.075 against 0.012 ms per node on WakeHuBERT's [125, 256]).

**Change.** `ggml_cpu_conv_2d_dw_fusion` matches the chain from the PAD (when the convolution is its only
consumer) or from the convolution; every neighbour is optional. A zero PAD on `ne0` only folds into a
larger left pad over the PAD's unpadded input -- the kernel's bounds test already reads zeros outside
the source -- and the bias and RELU are applied to each finished output row while it is in L1, in the
unfused order, so the result is bit-identical. An output that is exactly the source is handled by
copying each source row out before writing it; a partial overlap declines.

**Measured** (WakeHuBERT tiny, 2.5 s, 1 thread, Ryzen 3 3250U, min over 150 runs, 5 alternating
launches): 12.67 -> 11.29 ms. Bit-identical to the unfused graph on F32, F16 and Q8_0 files, and at 4
threads. x86 AVX2 only measured.

## PR 20 — `vulkan`: honour `GGML_PREC_F32` for an F32 x F32 `mul_mat`

*(`cmake/patches/ggml-0023-vulkan-f32-matmul-prec.patch`, independent of the others)*

**Problem.** On any fp16-capable device, `matmul_f32_f32` is the `FLOAT_TYPE = float16_t` build. Both
operands pass through half-precision shared memory, and its coopmat paths convert them to f16
outright. A non-contiguous F32 operand is also copied to f16 before the multiply. `GGML_PREC_F32` only
selects an f32 *accumulator*, and only for F16/quantized `src0`; for F32 x F32 it is ignored. So any
value past 65504 becomes inf. SenseVoice-Small's power spectrum (kaldi fbank, peaks near 1e10) made its
mel projection 99.5% NaN on a Radeon Vega 3, and the model transcribed nothing
([Retro-074](../../docs/retros/retro-074-a-gpus-f32-matmul-was-half-precision.md)).

**Change.** A `pipeline_matmul_f32_prec` built from the always-compiled `_fp32` SPIR-V with the scalar
tile sizes, following the same "reusing CREATE_MM from the fp32 path" pattern the BF16 fallback
already uses. When `src0`, `src1` are F32 and `op_params[0] == GGML_PREC_F32`, `ggml_vk_mul_mat_q_f16`
selects it, its conversion type becomes F32 instead of F16 (non-contiguous copies stay F32, the
coopmat2 forced copy is skipped, the integer-dot Q8_1 path is skipped), and the prealloc sizes follow
that type rather than `sizeof(ggml_fp16_t)`. On a device without fp16 the new pipeline *is*
`pipeline_matmul_f32`. Default-precision graphs are untouched.

**Measured.** SenseVoice on `Vulkan0` (Vega 3, RADV): every node within 1e-5 abs-sum of the CPU,
logits to six significant figures; card gate green on 40 models on the Vega 3 (37/40 on the 5090's
Vulkan, the three failures pre-existing and unrelated). The cost is real on the Vega 3: whisper-small
6007 → 7713 ms, Parakeet-TDT 2706 → 3673 ms (F32 FMAs instead of packed fp16); on an RTX 5090 (coopmat2)
whisper-small 122.0 → 141.3 ms. Which is why it is opt-in
per node rather than a change to the default. Verified failing before and fixed after on coopmat2 and
KHR coopmat (RTX 5090) and on plain fp16 (Intel Arc iGPU, Vega 3).

## PR 21 — `metal`: honour `GGML_PREC_F32` for an F32 x F32 `mul_mm`

*(`cmake/patches/ggml-0024-metal-f32-matmul-prec.patch`; applies after PRs 14–16, independent of their logic)*

**Problem.** `kernel_mul_mm_f32_f32` is instantiated with `S0 = S1 = half`: both operands are loaded
into half threadgroup tiles. This has the same overflow as PR 20, with the same SenseVoice symptom on
an M1 Pro. The mat-vec kernels are F32 throughout and are not affected.

**Change.** `kernel_mul_mm_f32_f32_prec`, the same template with float tiles. B's offset in shared
memory becomes `64*32*sizeof(S0)` (4096 for the half kernels, as before), and the host requests
8192 + 4096 bytes for it. `ggml_metal_library_get_pipeline_mul_mm` picks it when `op_params[0] ==
GGML_PREC_F32`. The tensor-API `kernel_mul_mm` has no float variant here, so on those devices
`ggml_metal_op_mul_mat` sends the case to the mat-vec kernel instead.

**Measured** (M1 Pro, median of 10 calls, three alternating rounds): whisper-small 587.9 → 586.8 ms,
SenseVoice 114.6 → 115.6 ms (and correct). Not run: a tensor-API device.

## PR 22 — `metal`: NORM/RMS_NORM with a partial last simdgroup — already upstream

**`ggml-0025` deleted at the v0.26.0 bump**, which includes `95414e59` (#26708).

*(`cmake/patches/ggml-0025-metal-norm-partial-simdgroup.patch`)*

**Not a PR: a backport.** llama.cpp `a194a75b7e` (#26708, 2026-08-07) rounds `ggml_metal_op_norm`'s
thread count up to whole simdgroups. Before it, a row of 66 gave 66 threads, and the cross-simdgroup
reduction in the 2-lane last simdgroup dropped its own partial. Kokoro's AdaIN norms (66 frames) turned
that into a 11x louder, unintelligible waveform on an M1 Pro
([Retro-075](../../docs/retros/retro-075-upstream-had-already-fixed-the-norm.md)). **Delete this patch
when the pin passes that commit.**

## PR 23 — `cuda`: `mul_mat_vec_f` asserts on a `src1` stride it never steps by

*(`cmake/patches/ggml-0026-cuda-mmvf-src1-strides.patch`, independent of the others)*

**Problem.** `ggml_cuda_should_use_mmvf` checks only `src0`'s strides, but
`launch_mul_mat_vec_f_cuda` asserts `stride_col_y % 2 == 0` on `src1` and aborts the process. A
squeeze-excite's pooled `[1, C]`, permuted to one `[C, 1]` column, has `nb[1] == 4`, so Citrinet-1024
died on an RTX 5090 at its first SE block. The kernel reads `y` at `j*stride_col_y` for `j <
ncols_dst`, so with one column the stride is never used.

**Change.** The assert allows any column stride when `ncols_dst == 1` (not for the multi-token-id
form, which does step by it). `ggml_cuda_mul_mat` and the mat-vec fusion check decline mmvf when a
`src1` stride it *would* step by (column, channel or sample, for a dimension larger than one) is not a
whole number of `float2`, so such a product falls through to the other kernels instead of aborting.

**Measured.** Citrinet-1024 on `CUDA0` transcribes identically to the CPU; the SE `MUL_MAT`'s output
matches the CPU's sum exactly. Card gate green on 7 models.

## PR 24 — `vulkan`: `conv_transpose_1d` overruns its fixed shared window

*(`cmake/patches/ggml-0027-vulkan-conv-transpose-1d-shared-window.patch`, applies after PR 20)*

**Problem.** `conv_transpose_1d.comp` accumulates a workgroup's output window, `128*s0 + K` values,
in a fixed `shared D_TYPE tmp[4096]`, and neither the shader nor `supports_op` checks the size. An iSTFT
written as an overlap-add transposed convolution (Soprano: hop `s0 = 512`, `K = 2048`) needs 67584
slots. The writes land outside the array and the output came back about 4000x too quiet, with no
error. Upstream master has the same shader and the same `supports_op` (2026-10-08).

**Change.** `supports_op` declines a `CONV_TRANSPOSE_1D` whose window does not fit, so the scheduler
runs it on the CPU. A shader that tiles the window would keep it on the device; that is the better
upstream fix and is not attempted here.

**Measured.** Soprano-1.1-80m on `Vulkan0` (Vega 3): peak 0.00099 → 0.54464, identical to the CPU.

## PR 25 — `vulkan`: honour `GGML_PREC_F32` for an F32 x F32 `conv_2d`

*(`cmake/patches/ggml-0028-vulkan-conv2d-f32-prec.patch`, applies after PR 20)*

**Problem.** `conv2d_mm.comp`'s coopmat2 and coopmat1 builds stage both operands in `float16_t` shared
memory and accumulate in `float16_t`. They are chosen whenever the device has either kind of
cooperative matrix, so every F32 convolution on such a device runs at half precision. The scalar build
is float throughout. A 1024-tap DFT-basis STFT followed by `log(clamp(x, 1e-5))` (F5-TTS's reference
mel) comes out with an fp16 noise floor in the near-silent bins. On an RTX 5090 that moved the mean
log-mel from -2.23 to -1.50, and the model's speech came out twice as loud and clipped
([Retro-077](../../docs/retros/retro-077-a-gpus-f32-convolution-was-half-precision-too.md)). Upstream
master has the same shader and selection (2026-10-09).

**Change.** `ggml_prec_set_acc` accepts `GGML_OP_CONV_2D` and stores the request in `op_params[9]`
(0..5 are the geometry, 6..8 loom's folded-kernel layout from PR 13; upstream would only need slot 6).
`vk_conv2d_pipeline_state` gains a `prec_f32` bit. A key with it set is built from the scalar SPIR-V
(`_unroll` or plain, as the vendor rule picks) with the scalar tile configuration: float shared memory,
the vendor's float padding, no `Csh` staging, no required subgroup size. The configuration is captured
before the coopmat setup overwrites the shared variables. Default-precision graphs are untouched, and on
a device without a coopmat conv path the new pipeline is the one it already ran.

**Measured** (RTX 5090, Vulkan). F5-TTS: peak 1.2055 → 0.7478 (CPU 0.7565, CUDA 0.7499), correlation
with the CPU waveform 0.08 → 0.950 (CUDA 0.949), the same on coopmat2 and on KHR coopmat
(`GGML_VK_DISABLE_COOPMAT2=1`). Intel Arc iGPU (no coopmat conv path): bit-identical before and after.
Cost, median of 10, interleaved: Supertonic-2 39.8 → 44.8 ms, Citrinet-1024 14.1 → 15.9 ms, F5-TTS
2558 → 2598 ms, whisper-small and Parakeet-TDT unchanged.

## PR 26 — `cpu`: `soft_max` with XNNPACK's exponential, a vector max and a sum that keeps its accuracy

*(`cmake/patches/ggml-0029-soft-max-xnnpack-kernels.patch`)*

**Problem.** A `soft_max` row is four passes: copy, scale, mask; a scalar max; `ggml_vec_soft_max_f32`;
a final scale. Two of them are slow for reasons that have nothing to do with the arithmetic.
`ggml_vec_max_f32` is a dependent scalar compare per element. `ggml_vec_soft_max_f32` reduces each
vector horizontally into a double, a serial add chain the exponentials wait on. And `ggml_v_expf`, accurate
over the whole real line, leaves its fast path for a three-way blend whenever a vector holds an |n| > 126. A
causal or padding mask puts -inf in up to half of every row, so masked rows cost twice as much as dense ones:
49.3 ms dense against 101.9 ms with an F16 causal mask, 12 heads x 1500 x 1500 on one AVX2 thread.
XNNPACK's `softmax_nc_f32` does the dense case in 29.0 ms on the same box.

**Change.** The exponentials are XNNPACK's `f32-raddstoreexpminusmax` kernels with their constants unchanged
(`avx2-rr2-p5`, `neonfma-rr1-p5`, `scalar-rr2-p5`; BSD-3, notice carried in the source). On aarch64 XNNPACK's own config picks `rr1-lut64-p2`, but rr1_p5 measured faster on both the A72 (2%) and the M1 (20%) at the same accuracy, and needs no table. They only take
x - max <= 0 and zero everything below the denormal cutoff with a mask, -inf included, so they never branch. The
max gets four vector accumulators (`f32-rmax`). The sum does NOT follow XNNPACK, whose f32 accumulators
raise |sum - 1| from 1.3e-7 to 7.1e-7 on these rows. Each unrolled step adds its vectors in f32 (four values
per lane, never more than the old loop) and widens that one vector into f64 vector accumulators. The copy, scale
and mask become one pass, and a plain softmax (no mask, scale 1) reads the source row directly. AVX-512, SSE2,
SVE and RVV keep their existing loops. The scalar fallback (ARMv6, 32-bit ARM) moves from libm `expf` to the
scalar kernel.

**Accuracy** (through `ggml_soft_max_ext` against an f64 softmax: plain, scale 0.125, F32 and F16 causal masks,
ALiBi 8, a 77-wide row for the tails). On AVX2 every metric (|sum - 1|, max abs, max rel) is the same or better.
On aarch64 (qemu) nothing gets worse by more than 5%. On ARMv6 (qemu, bit-identical to the board) the old scalar loop was the most accurate of all, because it added every element into a double. The new one does the same (|sum - 1| 6.5e-8 → 6.8e-8), but the 1.96-ulp exponential in place of libm's 0.5 moves max abs 3.6e-8 → 5.0e-8 and max rel 6.4e-7 → 7.2e-7. That is still below every SIMD path's error before the patch. The exponentials alone, over every exponent band of
[-87, 0]: rr2_p5 1.96 ulp max (`ggml_v_expf` 1.92), NEON rr1_p5 4.73 ulp max at -87, where the result is ~1e-38. On a
row, the error comes from the sum, not the exponential.

**Measured**, 12 x 1500 x 1500, best of 5-7, before/after interleaved, two runs each (dense / F16 causal mask):
Ryzen 3 3250U 1t 49.3 → 34.1 ms (1.45x) / 101.9 → 49.7 (2.05x), XNNPACK dense 29.0 ms. Pi 4 (A72) 1t
347.7 → 165.3 (2.10x; XNNPACK's own margin there 2.05x) / 418.5 → 195.2 (2.14x); 4t 0.98x / 1.08x (memory-bound).
M1 Pro 1t 18.07 → 11.74 (1.54x; XNNPACK's 1.29x) / 23.11 → 13.50 (1.71x); 8t 1.24x / 1.60x. Pi Zero (2 heads)
1105 → 756 (1.46x; XNNPACK's 1.97x) / 1289 → 1117 (1.15x). **The scalar kernel must not compute on -inf**: as first
ported (XNNPACK zeroes at the end), a causal mask made the Pi Zero 18x SLOWER (22.4 s), because the ARM1176's VFP
traps inf - inf = NaN, and denormals, to software. An early return fixed that but lost the dense gain (no
interleaving across the unrolled four). The kernel replaces sub-cutoff inputs by 0 before the arithmetic (a
predicated move) and zeroes the result after. Of the x86 dense gain, 6% is skipping the copy for a plain softmax.

**A gate it moves, and why that gate is not evidence.** loom's CosyVoice3 free-running check (waveform rmse < 2e-2
against PyTorch) goes 0.0027 → 0.0157 (cloned voice 0.0069 → 0.0333, which fails). The teacher-forced flow mel is
unchanged (rmse 1.390e-4 → 1.397e-4). A control with this patch's structure but ggml's OWN exponential, which
differs from v0.26 only in the rounding order of the row sum, lands at 0.0343. So a one-ulp perturbation of
attention moves that free-running waveform anywhere in 0.003-0.034: it measures chaos, not this kernel.
On two more sentences (one per voice) every build transcribes to the reference's words (whisper-small), and the
log-mel distance stays inside PyTorch's own f32-vs-f64 spread. loom's bound is now 6e-2, set from that spread.

## PR 27 — `cpu`: tinyBLAS F32 keeps its rate at large K (K chunks and a B block sized from the caches)

*(`cmake/patches/ggml-0030-tinyblas-f32-k-chunks.patch`)*

**Problem.** tinyBLAS streams all of K for every RM x RN tile, and a job reuses one block of BN x RN columns
of B across all of A's row tiles. The A strip (RM x K x 4 bytes) outgrows a 32 KB L1D at K = 2048, and the B
block outgrows its share of L2. On a Pi 4 (Cortex-A72, GCC's 4x3 tile, BN = 24), the F32 `mul_mat`
768 x 1500 x K falls from 8.5 GFLOP/s at K <= 768 to 5.1 at K = 3072 and 4.3 at K = 4096 on one core,
and from 24.2 to 12.7 / 11.1 on four (the four-thread knee is at K = 1024, where four B blocks overflow the
shared 1 MB L2). Whisper's `fc2` and Qwen3's `down_proj` are K = 3072. The M1 (128 KB L1D, 12 MB L2) is flat
through K = 3072.

**Change.** F32 x F32 only, and only when KN divides K. The B block is capped at half of L2. Inside a job, every
tile advances one K chunk before any advances the next, with the chunk sized to keep the A strip at 3/8 of L1D
(768 on a 32 KB L1). Chunking starts at K >= 2 chunks. The tile's vector accumulators are carried between
chunks as raw vectors, and the reduction happens once at the end, so every output is the same sum in the same
order. **The result is bit-identical** to the unchunked path (FNV hash of every output, every K, on AVX2, the
A72 and the M1; and a CosyVoice3 end-to-end gate reproduces the unpatched numbers to every printed digit).
Cache sizes come from sysfs on Linux and `hw.perflevel0.*` on Apple. Where neither answers, nothing changes.
On the M1 the derived chunk is 3072, so these shapes are untouched (forcing a 512 chunk there cost 3%).

**Measured**, Pi 4, 768 x 1500, best of 3, before/after interleaved with a 60 s cool-down, two runs that agree
(GFLOP/s, 1 thread | 4 threads): K = 768 8.5 → 8.5 | 24.1 → 23.8; K = 1536 7.7 → 7.6 | 18.2 → 20.6;
K = 2048 6.4 → 6.9 | 14.9 → 18.4; K = 3072 5.05 → 6.84 (1.35x) | 12.7 → 17.9 (1.41x); K = 4096 4.31 → 6.06
| 11.1 → 16.5. Whisper's `fc1` shape (3072 x 1500 x 768) unchanged. It recovers about two thirds of the drop.
The rest is the A stream past L2 itself.

## PR 28 — `cpu`: RunFast (flush-to-zero, default NaN) on ARMv6 compute threads

*(`cmake/patches/ggml-0031-armv6-runfast-compute-threads.patch`)*

**Problem.** The ARM1176's VFP11 (Raspberry Pi 1 / Zero) runs only the easy cases in hardware. In the
default IEEE mode it bounces to the kernel's VFP support code, at ~1.1 us against 8 ns for a `vmla.f32`:
any denormal operand, any NaN operand, inf - inf, and any effective subtraction between values at or
below 2^-96 (it might underflow). It also does not pipeline back-to-back `vmla` in IEEE mode. A
BatchNorm-folded checkpoint with dead channels (weights ~1e-33) bounces on most of its 1x1 conv MACs,
and that runs Citrinet-256-ls 2.8x slower than it needs to.

**Change.** On `__ARM_ARCH == 6` with VFP only: `ggml_graph_compute_thread` sets FPSCR's `FZ | DN` on
entry, which every compute thread runs (the caller's included), and puts back just those two bits on
exit. The cumulative flags survive, and the caller's own floating point stays in IEEE mode.
`GGML_CPU_RUNFAST=0` turns it off. Read once in `ggml_cpu_init`.

**Numerics.** A denormal input reads as +0 and a denormal result becomes +0, and a NaN result is the
default NaN. Every other result is bit-identical. On a Pi Zero W across eleven models (ASR, TTS, NER,
F32 and quantized): ten outputs bit-identical, one TTS waveform with 24 int16 samples one LSB apart
(same whisper transcript); tensor-level, one CTC model's log-probs within 9.6e-7 of the peak, the rest
bit-identical. glibc's libm agrees with "flush in, flush out" for every function the CPU ops call, except
`logf` of a denormal argument (-103.97, where a flush predicts -inf). Note that qemu-user flushes to a
*signed* zero, so it does not reproduce the board bit for bit in this mode.

**Measured**, Pi Zero W, one inference per launch, two rounds ABBA, IEEE → RunFast: Citrinet-256-ls F32
55.9 → 20.1 s (2.79x); sanoTTS heart 7.4 → 5.5 s (1.33x); piper-style voices 1.14-1.15x; VITS
66.9 → 60.0 s (1.12x); Moonshine tiny F32 1.10x; Q8_0/Q4_0 models 1.02-1.09x. No model is slower.

## PR 29 — `cpu`: 1-D convolutions on the ARM1176's VFP short vectors

*(`cmake/patches/ggml-0032-armv6-shortvec-conv1d.patch`; generated by loom's `scripts/gen_armv6_conv_f32.py`)*

**Problem.** The ARM1176 (Raspberry Pi 1 / Zero) has no float SIMD, and ggml's ARMv6 F32 convolutions run
on scalar `vmla` at 0.25-0.65 GOP/s. The vocoders of the TTS models spend 64-83% of a Pi Zero run there
(1-D convolutions, k = 3-7, dilation 1-5, 12-64 channels, tens of thousands of steps). But the VFP11 has
short vectors: with FPSCR `LEN = 4`, a data-processing op whose destination is outside bank 0 is four ops,
and a bank-0 operand is a scalar. That makes it a 4-wide FMAC with a broadcast, at up to 0.94 GMAC/s.
No compiler emits it, because every compiled function assumes `LEN = 0`.

**Change.** On ARMv6 with VFP only, `ggml_compute_forward_conv_2d_impl` first offers every 1-D, stride-1,
F32 convolution with k = 2..11 to a new path. A quantized kernel arrives here already dequantized. Per
block of 128 output steps and chunk of input channels (sized so the chunk's window rows and one quad's
weights fit 12 KB of the 16 KB L1), the path copies the window out in C, applying zero padding and a
fused `LEAKY_RELU`. Then, for each quad of output channels, a generated asm function runs the block's
4-channel x 4-step tiles. It sets `LEN = 4` once, runs integer loops and VFP code only, calls nothing,
and clears `LEN` before it returns. The first chunk starts the accumulators from the bias, and each later
chunk continues from the partial sums in the output tile. A fused residual is added as each block is
flushed. A result written over the input (ggml-alloc does that to a vocoder's convolutions) is held until
the next block's window is copied. A ragged block computes whole tiles into staging and flushes only its
steps. The weights are packed `[OC/4][IC][k][4]` per call, and a partial last quad is padded with zero
channels. `GGML_CPU_ARMV6_CONV=0` turns the path off.

**Safety.** Linux clears `LEN` and `STRIDE` for a signal handler, restores the interrupted FPSCR on
sigreturn, and saves it per thread. That was checked on a Pi Zero W under ~118k signals that landed inside
a `LEN = 4` kernel, in RunFast and in IEEE mode, with sabotage turning the check red (loom ADR-079 gate 1,
`scripts/armv6_shortvec_gate.c`). qemu-user does NOT clear `LEN` for a handler.

**Numerics.** Each output is bias plus the sum over input channels (outer) and taps (inner), in order, with
no FMA. That is a different summation order from the paths it replaces, so outputs move at rounding level.

**Measured**, Pi Zero W, one thread, RunFast, the seven vocoder shapes of loom's `bench49` as ggml graphs
(`scripts/bench50.cpp`, conv + fused bias): 13.4-13.5 s → 5.03-5.06 s (2.67x), 1.16 GOP/s, which is at or
above the standalone kernel's rate on every shape. End to end, two rounds ABBA, path off → on: piper-style
voices amy 39.8 → 22.9 s and de051 8.41 → 4.84 s, VITS 60.3 → 34.8 s (1.73-1.74x), with whisper transcripts
identical in both arms. Models without such convolutions are bit-identical. The input-channel chunks
(the first version held the whole window and ran the dilated and wide shapes at 0.35-0.58 GOP/s) took the
models' own slow shapes 8.73 → 5.56 s and VITS a further 34.6 → 25.3 s, amy 22.9 → 18.6 s, with
bit-identical outputs.

## PR 30 — `cpu`: a one-column F32 `mul_mat` on ARMv6, 3.5x faster

*(`cmake/patches/ggml-0033-armv6-f32-gemv.patch`)*

**Problem.** On the ARM1176 (Raspberry Pi 1 / Zero), `llamafile_sgemm` declines n = 1, so a decoder step
or an LSTM step reaches `ggml_vec_dot_f32` once per output. With no `GGML_SIMD` on this core, that is the
scalar branch: a float product summed in double, one dependent chain per output. A 288 x 32768 F32
vocabulary projection streams its weights at 92 MB/s, where the board reads 338 MB/s with PLD.

**Change.** On ARMv6 with VFP only, `ggml_compute_forward_mul_mat` first offers an F32 x F32 product with
one column to a GEMV that runs four rows per pass, each a float sum of float products in k order. Rows are
split into contiguous ranges of four per thread. `GGML_CPU_ARMV6_GEMV=0` turns it off.

**Numerics: not bit-identical.** The scalar branch sums in double; this sums in float, as every SIMD build
of `vec_dot_f32` already does (in another order). The first version kept the double sum and was
bit-identical, at 274 MB/s; the float sum was taken for the speed, and gated on the board across loom's
model zoo (see ADR-079 in loom.cpp).

**Measured**, Pi Zero W, one thread, RunFast, tensors in a CPU backend buffer (loom `scripts/bench51.cpp`),
MB/s of weights at row strides of 1152 B / 1 KB / 2 KB / 3 KB / 4 KB:

| variant | 1152 B | 1 KB | 2 KB | 3 KB | 4 KB |
|---|---|---|---|---|---|
| `vec_dot_f32` (stock) | 92 | 92 | 92 | 93 | 92 |
| double sum, 4 rows (bit-identical) | 274 | | 236 | | |
| float sum, 8 rows | 317 | 135 | 147 | 124 | 50 |
| **float sum, 4 rows** | **318-325** | **294-298** | **270** | **267-269** | **108-119** |

Eight row streams thrash the 4-way 16 KB L1 at most strides, so four it is. A PLD per row and cache line
cost 10% (247 on the double sum). Placement matters on this core: the same kernel and bytes ran 216 MB/s
from a 4-byte-aligned context allocation (page offset 340) and 274 from an 8-byte-aligned one.

## PR 31 — `cpu`: F32 GEMM, 1x1 and strided convolutions on ARMv6's VFP short vectors

*(`cmake/patches/ggml-0034-armv6-f32-gemm.patch`, on top of PR 29's kernels)*

**Problem.** On the ARM1176, F32 `mul_mat` with n >= 2 runs ggml-0018's scalar 4x4 tinyBLAS tile at
0.29-0.50 GOP/s, and a 1x1 convolution runs through im2col into the same tile. PR 29's short-vector
kernel already holds a 4x4 GEMM tile at k = 1: per input row, a vector of four positions times four
scalars.

**Change.** (1) PR 29's convolution path also takes k = 1, with a 32-step block, because a tap-less window
row lets a chunk take many more input channels. (2) A graph `MUL_MAT` node with F32 x F32 operands,
n >= 24 and M % 4 == 0 runs the same kernel. The window is a transposed block of the weights, `[K chunk]
[32 rows]`, the scalars are four columns of the activation packed `[N/4][K][4]` once per call, and each
accumulator vector is a contiguous 4-float piece of a `dst` row, written in place. K is chunked to L1 and
each later chunk continues from the partial sums. A partial last quad of columns goes through a scratch.
`ggml_graph_plan` sizes the work buffer for it. **Graph nodes only:** `ggml_call_mul_mat*` (the
convolutions' im2col fallback) calls `mul_mat` with its own patches in `wdata`, which the packing would
overwrite. `GGML_CPU_ARMV6_GEMM=0` turns the GEMM off. (3) PR 29's odd-k kernels take input channels in
pairs instead of copying the double buffers back after each one. The copy cost as many FMAC cycles as the
four `vmla`s at k = 1. Results are bit-identical; the vocoders' shapes got 5-7% faster. (4) Any stride, by
phases: a stride-S convolution is a stride-1 one over IC * S virtual channels (channel c, phase p holding
x[c][t * S + p]) with ceil(k / S) taps each, so the window copy gathers the phases, the weights are
re-indexed once, and the kernels stay the stride-1 ones (dilation must then be 1).

**Numerics.** Each output is a plain float sum over k in order. For K <= 512 that is bit-identical to
tinyBLAS_F32_ARMV6, which sums each K block from zero only above 512.

**Measured**, Pi Zero W, one thread, RunFast (loom `scripts/bench51.cpp gemm`, `scripts/bench50.cpp
citrinet`): Moonshine tiny's prefill GEMMs 0.29-0.38 → 0.76-0.79 GOP/s (2.1-2.6x), conformer-ctc-small's
FFN (n = 75) 0.54 → 0.76, Citrinet-256-ls's 1x1 convolutions 0.51 → 0.77 (1.52x). The n threshold
comes from a sweep: n = 16 0.47 against 0.49, n = 24 0.56 against 0.49. Moonshine tiny's strided stem
(288 -> 576, k = 7, stride 3) 0.235 → 0.82 GOP/s. End to end, two rounds ABBA against PR 29 + 30:
Citrinet-256-ls Q8_0 29.0 → 16.9 s and F32 20.0 → 16.2 s for an 11 s clip, conformer-ctc-small 11.0 → 9.3 s,
Moonshine tiny F32 47.0 → 32.8 s and Q8_0 65.7 → 58.9 s (both with the stem), sanoTTS heart 5.04 → 4.33 s,
same transcripts.

## PR 32 — `cpu`: depthwise 1-D convolutions on ARMv6's VFP short vectors

*(`cmake/patches/ggml-0035-armv6-shortvec-depthwise.patch`; generated by loom's `scripts/gen_armv6_dw_f32.py`)*

**Problem.** On the ARM1176, `CONV_2D_DW` with one row (a depthwise 1-D convolution) runs PR 18's interior
sweep, one `ggml_vec_mad_f32` per tap, which is scalar on a core without float SIMD: 0.041 GMAC/s.
Citrinet-256-ls spends 4.1 s of an 11 s clip there on a Pi Zero W. PR 29's tile does not apply, because a
depthwise convolution has no input channels to reuse a weight across.

**Change.** On ARMv6 with VFP only, `ggml_compute_forward_conv_2d_dw_whcn_impl` sends each row of a
stride-1, dilation-1, F32 depthwise convolution with K >= 2 and at least 8 outputs to a generated kernel.
It is one function per K % 8, with the first 8 taps peeled when K >= 8. Per channel, the row and the
weights are copied into padded scratch (the kernel reads ahead), which also makes an output written over
its own input safe. The kernel holds 8 outputs as short vectors over a ring of 8 input registers, and its
vector operands wrap around the bank, so the window is renamed by the unroll and never moved. A tap is two
`vmla` and two scalar loads. It PLDs the next channel's row, one line per block. The fused bias and RELU of
PR 19 apply to the finished row as before. The row loop counts the channel instead of taking an int64
`%` per row, which is a library call on 32-bit ARM (3% of the path). `GGML_CPU_ARMV6_DW=0` turns it off.

**Numerics.** Each output is two float sums, the even taps and the odd taps, each in k order, then added:
a short-vector `vmla` waits for the previous one on the same destination (~14 cycles), and one sum per
output left the kernel at 14 cycles a tap instead of ~10. That differs from the sweep's single sum at rounding level
(3.2e-7 of peak on Citrinet's schedule).

**Measured**, Pi Zero W, one thread, RunFast (loom `scripts/bench52.cpp`, Citrinet-256-ls's 104 stride-1
depthwise calls, K = 5..41): 4.14 s → 0.38 s through `ggml_backend_graph_compute` (0.041 → 0.45 GMAC/s,
10.9x); the kernel alone 0.48-0.50 GMAC/s, against 0.29 for a C version with the same window renaming.
End to end, two rounds ABBA: Citrinet-256-ls F32 17.9 → 14.0 s and Q8_0 18.1 → 14.3 s for an 11 s clip,
same CTC ids; conformer-ctc-small, sanoTTS and WakeHuBERT 1.01-1.02x.

## PR 33 — CPU backend: a RELU and a time mask in the convolution epilogue

*(`cmake/patches/ggml-0036-conv-epilogue-fusion.patch`, on top of PRs 5, 7, 19, 29 and 32)*

**Problem.** NeMo's convolutional encoders (Citrinet, QuartzNet, Jasper) end every sub-block with
`MUL(RELU(ADD(conv, bias)), mask)`, where `mask[t] = t < length` is one value per position broadcast over
the channels, and multiply every depthwise convolution's output by the same mask. PRs 5 and 7 fold the
bias into the convolution; the RELU and the MUL stayed two more passes over the activation, and on a Pi
Zero W each pass costs ~48 ns an element. PR 5/7's fusion also declined 53 of Citrinet-256-ls's 129
convolutions outright: the graph allocator had placed the bias ADD's result a few hundred floats ahead of
the convolution's input, a PARTIAL overlap no kernel can write while still reading the input.

**Change.** `ggml_cpu_conv_2d_fusion` also takes, after the bias (and residual) ADD, a `RELU` and then a
`MUL` whose other operand is `[OL, 1]` -- one value per output position. `ggml_cpu_conv_2d_dw_fusion`
takes the same `MUL` after a depthwise convolution. A fused result that partially overlaps the input is
computed into the convolution's own tensor (which the allocator never overlaps with its input, since the
convolution runs while the input is live), and one pass applies bias, residual, RELU and mask on the way
to the output. Kernels that cannot apply the epilogue themselves get one pass after them. All of it is
behind `GGML_CPU_CONV_EPILOGUE`, off by default. (It shipped on by default on ARMv6. PR 34 then made the
unfused RELU and MUL passes 2-4x faster there, and this epilogue, which re-reads output rows the
convolution has just stored, became the slower of the two: Citrinet-256-ls 12.53 s off against 13.07 on.
Upstream it is worth taking only with the epilogue applied before the store.)

On ARMv6 (PRs 29 and 32) the epilogue is a few short passes over a block still in L1: the RELU as an
integer test on the bits, keep iff `lo <= u <= 0x7f800000` unsigned (lo = 0x00800000 under RunFast's
flush-to-zero, 1 in IEEE mode), which is exactly `(v > 0) ? v : 0` -- the VFP compare plus `vmrs` it
replaces stalled the pipeline every element -- and the bias add and mask multiply unrolled by four. The
depthwise kernel writes a scratch row first: an ARM1176's L1 does not allocate on a store, so an epilogue
reading back the row just stored to the output missed on every line (43 ns an element, 10 in cache).

**Numerics.** Bit-identical to the unfused nodes: the RELU and the MUL are the same operations in the same
order, and the partial-overlap path adds the bias after the convolution as the unfused ADD does. The
integer RELU was checked against the VFP's compare on the board over 17.1M bit patterns (every class:
denormals, -0, inf, NaN) in both FPSCR modes.

**Measured**, Pi Zero W, one thread, RunFast (loom `scripts/bench53.cpp`, Citrinet's 256-channel shapes):
at T = 551 the RELU + mask after a 1x1 convolution cost +17 ms as two passes and +3.7 ms fused; the mask
after a depthwise convolution +7.0 ms and +1.55 ms. End to end, with loom's exporter also merging the
mask's 239 identical subgraphs into 4 (ggml expands a node's sources depth first, so a mask rebuilt before
each use lands between the RELU and its MUL, and the fusion never sees them adjacent): Citrinet-256-ls
F32 14.0-14.4 → 13.1-13.3 s and Q8_0 14.3-14.5 → 13.5-13.6 s for an 11 s clip (1.06-1.08x), logits byte-identical
with the epilogue on and off; VITS, WakeHuBERT and Silero VAD within noise.

## PR 34 — `cpu`: ADD, SUB, MUL, SCALE, RELU and LEAKY_RELU at memory speed on ARMv6

*(`cmake/patches/ggml-0037-armv6-elementwise.patch`; independent of the others, and FZ-aware for PR 28's RunFast)*

**Problem.** On an ARM1176 (Pi Zero, Pi 1) the elementwise ops are compute-bound, not memory-bound: each
loop is one load, one VFP op and a store that waits on the op's 8-cycle latency, and nothing prefetches.
A `[1, C]` bias over `[T, C]`, or a `[1, 1]` scale, is worse: `apply_binary_op` calls its row function
once per element (`ne10 == 1`). The RELUs pay a `vcmpe` + `vmrs` per element, which drains the VFP
pipeline. Measured over a cold 12.5 MB tensor: ADD 68 ns an element, a bias ADD or scalar MUL 43, LEAKY_RELU
64, against 12 for `memcpy`.

**Change.** A header, `armv6-eltwise.h`, ARMv6 only (`__ARM_FEATURE_SIMD32 && !__ARM_NEON`): add/sub/mul
over f32 rows and their scalar forms eight floats a step through `vldmia`/`vstmia`, eight independent VFP
ops and one `pld` per input line, after a scalar head that brings the input to a 32-byte line (a `vldmia`
that straddles two lines nearly doubles a cold stream on this core: 21 vs 11 ns for a copy). RELU and
LEAKY_RELU compare on the integer bits: `x > 0` is `lo <= u <= 0x7f800000` unsigned, `lo` the smallest
positive normal under FPSCR.FZ (the compare sees a positive denormal as 0) and 1 in IEEE mode; LEAKY_RELU
keeps its own `max(x, 0) + slope * min(x, 0)` arithmetic. `apply_binary_op` (F32 only), the RELU in
`apply_unary_op`, `leaky_relu_f32` and `scale_f32` (one pass `x * s` instead of `memcpy` and an in-place
scale) call it. `GGML_CPU_ARMV6_ELTWISE=0` turns it off.

**Numerics.** Bit-identical by construction (the same IEEE operations on the same operands), and checked:
every op and broadcast form, with ±0, denormals, inf and NaN in the inputs, byte for byte against the stock
loops under RunFast; twelve models' outputs byte-identical with it on and off.

**Measured**, Pi Zero W, one thread: same-shape ADD 68.3 → 36.4 ns an element, bias ADD / scalar MUL 43.3
→ 11.6 (`memcpy` 11.9), LEAKY_RELU 63.8 → 23.0. End to end 1.02-1.07x on ten of twelve models (amy 1.045x,
de051 1.067x, conformer 1.053x, Citrinet Q8_0 1.065x), neutral on the other two.

## PR 35 — `cpu`: CONV_TRANSPOSE_1D as polyphase direct convolutions on ARMv6

*(`cmake/patches/ggml-0038-armv6-conv-transpose-1d-polyphase.patch`, on top of PRs 9 and 29)*

**Problem.** A stride-s transposed convolution runs as a GEMM `[Cout*K, Cin] x [Cin, L]` and an overlap-add
(PR 9). On ARMv6 the GEMM reaches tinyBLAS through `ggml_call_mul_mat_ldc` at ~180 MMAC/s, and the
overlap-add adds K/s partial results into every output. In the TTS vocoders (HiFi-GAN-style upsampling,
K = 2s) these nodes were 13-20% of a synthesis on a Pi Zero W.

**Change.** With K = J·s, output phase p is a J-tap convolution over the input padded by J - 1 zeros:
`y[oc][q·s + p] = Σ_j Σ_ic w[ic][oc][p + (J-1-j)·s] · xpad[ic][q + j]`. On ARMv6 (when K % s == 0 and
J <= 11) the op runs as s such convolutions per quad of output channels on PR 29's register-tiled
kernels, sharing one input window per block and chunk of input channels, into a per-thread scratch that
the last chunk interleaves into dst: every output is written once. The weights are repacked
`[p][Cout/4][Cin][J][4]` each call. `ggml_graph_plan` reserves the larger of the two paths' work buffers.
`GGML_CPU_ARMV6_CONVT=0` turns it off.

**Numerics.** Not bit-identical: the tile's summation order, 5e-7 to 1e-6 of the peak against f64 (the GEMM
path 1.4e-7 to 2.5e-7). Three vocoders' waveforms move by at most one int16 LSB.

**Measured**, Pi Zero W, one thread, against PR 9 plus the bias ADD: 658 → 367, 1379 → 751 and 1870 → 763
ms on a piper vocoder's three upsampling stages (1.79-2.45x); end to end 1.077-1.145x on three vocoders.

## PR 36 — `cpu`: GELU_ERF, SIGMOID, SILU and TANH on ARMv6's VFP short vectors

*(`cmake/patches/ggml-0039-armv6-activations.patch`; the kernels generated by loom's
`scripts/gen_armv6_act_f32.py`, on PR 29's short-vector discipline)*

**Problem.** On an ARM1176 there is no SIMD body for these: GELU_ERF runs PR 10's rational in its scalar
tail, the others call `expf`/`tanhf` per element. 200, 160 and 443 ns an element on a Pi Zero W. Plain C does
not fix it: every VFP-to-core transfer and every `vcmpe` + `vmrs` drains the VFP11's pipeline, and an own
`expf` in C ran slower than glibc's.

**Change.** Generated short-vector kernels (FPSCR LEN = 4), eight elements a pass, with the clamps and the
exponent arithmetic done in integer passes over a small scratch. GELU_ERF performs PR 10's tail's
operations in its order. SIGMOID, SILU and TANH use Cephes' single-precision `expf` (Cody-Waite, a degree-6
polynomial, 2^n added to the exponent field, with asymmetric range ends: 0 once v <= -86.98, inf once
v >= 88.37) and Cephes' small-|x| tanh polynomial. A row's last n % 8 run through the same kernel on a
padded copy. `GGML_CPU_ARMV6_ACT=0` turns it off; `GGML_CPU_DISABLE_GELU_ERF_SIMD` still selects libm.

**Numerics.** GELU_ERF bit-identical to PR 10's scalar tail (under RunFast; in IEEE mode the VFP11 bounces
denormal operands to the kernel's support code). SIGMOID 2.40 ULP and TANH 1.30 ULP against f64 (glibc 2.39
and 2.16); special values as glibc's.

**Measured**, Pi Zero W, one thread: GELU_ERF 200.6 → 80.9 ns an element, SIGMOID / SILU 161 → 76, TANH
443 → 113. End to end 1.01-1.05x on eight of twelve models (sanoTTS heart 1.054x, Moonshine 0.52 s), the
rest neutral; transcripts unchanged.

## PR 37 — `cpu`: a dynamic int8 `mul_mat` for ARMv6, on Q8_0 weights with one scale per row

*(`cmake/patches/ggml-0041-armv6-int8-mul-mat.patch`; loom ADR-079 P4.33, weights from ADR-076's `Q8_0_PC`)*

**Problem.** On an ARM1176 (no NEON) a Q8_0 `mul_mat` runs `vec_dot_q8_0_q8_0` in scalar C, at about half
the speed of PR 30/31's F32 kernels, so a quantized model is smaller and slower. A one-column `mul_mat` (a
decoder step, an LSTM step) is bound by the weight stream, where int8 is a quarter of F32's bytes.

**Change.** An extra buffer type, `CPU_ARMV6_I8`, registered only on ARMv6 (and not with
`GGML_CPU_ARMV6_I8=0`). `set_tensor` checks that every block of a row carries the same F16 scale and
repacks the row quad by quad (`[N/4][K/4][4]` words of int8, then one float scale per row); a tensor whose
rows are not uniform is stored as it came and its `extra` cleared, so ggml's own Q8_0 path runs it.
`compute_forward` quantizes each activation column to int16 (symmetric absmax, the widest range whose K
products cannot overflow SMLAD's int32 sum: 32767 up to K = 516, then 2^31 / (128 K)) into the work buffer,
then each thread streams its own channel quads: SXTB16 + SMLAD in asm, one PLD per 32-byte line, F32 out as
`acc * (sx * sw[n])`. A NaN or inf in a column makes that column's outputs NaN.

**Numerics.** Integer accumulation, exact; the test holds both kernels to that contract bit for bit,
including the overflow bound (every product at -128 x full range, K = 3072).

**Measured**, Pi Zero W, one thread, cold weights (loom `scripts/bench58.c`): 4.29x over PR 30's F32 GEMV on
all seven decode shapes (Moonshine tiny's 288 x 32768 vocabulary projection 126.9 -> 29.4 ms, 320 MB/s of
int8 against the board's 338 MB/s PLD read ceiling). A PLD every 16 bytes instead, two per line, ran at
half that (2.09x).

## PR 38 — `cpu`: int8 GEMM tiles for PR 37 from 16 columns

*(`cmake/patches/ggml-0042-armv6-int8-gemm-tiles.patch`; the tiles generated by loom's
`scripts/gen_armv6_i8_gemm.py`)*

**Problem.** PR 37's kernel computes one channel quad against one column. As a GEMM it is compute-bound
at 0.70-0.76 GOP/s, a tie with PR 31's F32 GEMM at n >= 32 (loom `scripts/bench59.cpp`).

**Change.** From 16 columns, a 2-column x 4-channel tile on int16 operand pairs (8 SMLADs and 6 loads per
K pair, 128 or 16 pairs unrolled, accumulators in memory between K chunks of 512). The weights of a group
of channel quads are unpacked to int16 pairs into an 8 KB block that stays in L1, a block of 16 column
pairs keeps its accumulators in L1, and x streams past both with a PLD 128 bytes ahead. Each of these was
measured (loom `scripts/bench61.c`): on the Pi Zero a line from its L2 costs ~100 cycles, as much as one
from DRAM, so the tile ran 1.8 GOP/s on L1-resident operands and 0.78 with x from L2; the PLD took that to
1.45, and moving the accumulators out of a buffer over all T (which evicted the W block) to 1.0 inside the
driver.

**Numerics.** The same contract as PR 37, to the bit.

**Measured** (loom `scripts/bench60.cpp`, against PR 31's F32 GEMM as the engine runs it, quantize
included): 0.92-1.01 GOP/s, 1.22-1.40x at K <= 288 (Moonshine tiny's encoder, Citrinet's 1x1 convolutions,
WakeHuBERT, sanoTTS heart), 1.09-1.13x at K = 704-1152, 1.66-2.04x at n = 16-64, and 3.8-12.8x on
DistilBERT-sized K = 768-3072 at n = 16, where PR 31 does not apply.

## PR 39 — `cpu`: pointwise convolutions on PR 37's buffer type, and no conv fusion over a packed kernel

*(`cmake/patches/ggml-0043-armv6-int8-pointwise-conv.patch`)*

**Problem.** A convolution whose kernel sits in an extra buffer type cannot take the direct path: the CPU
device declines a node whose source is not in a host buffer, and a caller that asks first (loom does) falls
back to im2col and a `mul_mat`. With every per-channel weight in PR 37's buffer, two convolutional models
ran slower than F32 on a Pi Zero W (Citrinet-256-ls 0.86x, WakeHuBERT tiny 0.89x). Worse, the CPU graph
loop tries its conv fusions before an extra buffer type's `compute_forward`, so a `CONV_2D` + bias `ADD`
over a packed kernel would run the fused kernel on bytes that are not Q8_0.

**Change.** `CPU_ARMV6_I8` accepts `ggml_conv_2d_direct_packed` with a 1x1 kernel and no padding, at any
stride: the same GEMM, its columns the output positions, read across the channels at the input's channel
stride (every s-th position) and written down the output's. Any other convolution over a kernel there is
declined by `supports_op` and aborts in `compute_forward` (no other path can read the kernel). The conv
fusion matcher declines a kernel in a non-host buffer.

**Numerics.** PR 37's contract, to the bit, at stride 1 and 2, with the bias ADD after it.

## PR 40 — `cpu`: PR 37's quantize and epilogue out of the VFP11's and the write buffer's way

*(`cmake/patches/ggml-0044-armv6-int8-quantize-epilogue.patch`)*

**Problem.** Timed inside a model on a Pi Zero W, the activation quantize and the output epilogue around
PR 38's tiles cost ~1.1 s and ~1.0 s of Citrinet-256-ls's 6.7 s of int8 GEMM. Each absmax compare was a
`vcmp` whose `vmrs` drains the VFP11's pipeline. Each `vcvtr` result was moved to a core register (another
drain). `int64_t` loop indices made gcc spill and redo `(c / 2) * K` in 64 bits per element. The output
address went through three software 64-bit divisions per column per channel group. And a channels-first
output was written column by column, 16 rows at a time, which the write buffer cannot merge.

**Change.** The absmax is an integer max over the floats' bit patterns, with the sign bit cleared: the
order of the patterns is the order of the values, and any NaN or inf is a pattern >= 0x7f800000, which makes
the column's scale NaN. Conversions are stored from the VFP to a scratch that integer code packs. The loops
count in `int` and walk pointers, with a PLD per line in the absmax pass. Column addresses are computed
once per op. A channels-first output is written row by row across the column block.

**Numerics.** Unchanged to the bit (PR 37's test).

**Measured** (Pi Zero W, jfk, one run each): Citrinet-256-ls `Q8_0_PC` 10.6 -> 9.7 s, Moonshine tiny 27.6
-> 26.9 s.

## PR 41 — `cpu`: PR 27's K chunks between one chunk and two, when threads share the L2

*(`cmake/patches/ggml-0045-tinyblas-f32-k-halves-on-a-shared-l2.patch`)*

**Problem.** PR 27 chunks K only from two chunks up (K >= 1536 on a 32 KB L1). Below that every tile streams
its two operand strips, (RM + RN) x K floats, whole. On one core that is the right call. On a Pi 4 (Cortex-A72,
four cores, one 1 MB L2) it is not: whisper's encoder `A@V` (m = 64, n = K = 1500, 12 heads) scales only 2.0x
from one thread to four, where its `QK^T` scales 3.4x. Its strips are 42 KB against a 32 KB L1, and four cores
then queue on the shared L2.

**Change.** aarch64, F32 x F32, KN dividing K, as PR 27. For `kc < K < 2 kc`, K is cut in two balanced halves
when `nth > 1` AND the strips pass 7/8 of L1D. One thread, and any K whose strips fit, keep PR 27's schedule.
**The result is bit-identical** (PR 27's carried accumulators). loom's `test_tinyblas_k_halves` holds 1, 2
and 4 threads to the same bytes on both sides of every boundary.

**Measured**, Pi 4, four threads, halves vs whole, per-round ratio over nine rounds, byte-identical outputs:
768 x 1500 x K at K = 960 / 1024 / 1152 / 1344: 0.944 / 0.965 / 1.035 / 1.096, and `A@V` 1.312 (219.7 -> 167.6
ms). The sign flips where the strips pass 7/8 of L1 (between K = 1024 and 1152 at the 4x3 tile), which is the
predicate. Forcing two chunks at one thread costs 2% on `A@V`, so one thread is left alone. End to end:
whisper-small 1.018x (7 paired launches, p10 1.004, p90 1.043); a VITS synthesis is unmoved (0.992, p10 0.979,
p90 1.016).

## PR 42 — `cpu`: a materialised transpose of 4-byte elements in 4x4 tiles

*(`cmake/patches/ggml-0046-cpu-dup-transpose-x32-tiles.patch`)*

**Problem.** `dup_bytes` copies a source that is strided along its first dimension one element at a time,
with a runtime-sized `memcpy` per element and a fresh cache line per read. The common case is a transpose:
the source runs contiguous along its second dimension. whisper's V^T (`[64, 12, 1500]` permuted to
`[1500, 64, 12]`, twice per layer) ran at 0.42 GB/s on a Pi 4 at four threads, 0.35 at one.

**Change.** When `type_size == 4`, `nb01 == 4`, `nb00 != 4` and dst is contiguous: 4x4 tiles transposed in
registers (NEON `trn` on aarch64, after XNNPACK's x32-transposec 4x4 kernels, a plain 4x4 elsewhere), stored
as 16-byte row segments, with the first dimension walked in blocks of 64 so the source lines stay in L1
across a thread's rows. Threads split the second dimension, as before. Enabled on aarch64, x86-64 and ARMv6
(`__ARM_FEATURE_SIMD32` without NEON), the three ISAs measured.

**Numerics.** A copy, and byte-identical (loom's `test_cont_transpose_x32`, ragged shapes, 1-4 threads).

**Measured** (whisper's V^T, both arms in one process, per-round ratio): Pi 4 4.68x at four threads (21.9 ->
4.68 ms), 3.48x at one; Ryzen 3 3250U 1.76x at two, 1.82x at one. A decoder-sized `[32, 12, 50]` is neutral on
the A72 (0.97x) and 4.3x on x86. End to end, whisper-small F32 on the Pi 4: 1.021x (7 paired launches, p10
1.004, p90 1.050). Pi Zero W (ARM1176, one thread), where the plain 4x4 replaces a libc `memcpy` call per
element: 2.57-5.28x over the shapes its models materialise, and end to end Moonshine tiny 1.084x F32 and 1.101x
Q8_0 (5 paired runs, byte-identical), its cross-attention V^T being copied at every decode step.

## PR 43 — `cpu`: an int8 `mul_mat` for AArch64 cores without DOTPROD, on Q8_0 weights with one scale per row

*(`cmake/patches/ggml-0047-neon-int8-mul-mat.patch`)*

**Problem.** tinyBLAS's Q8_0 GEMM is DOTPROD-only, so on a core without `sdot` (the Cortex-A72 of a Raspberry
Pi 4, A53s, A57s) every Q8_0 `mul_mat` runs `ggml_vec_dot_q8_0_q8_0` one output row at a time, widening each
32-element block's sum to float. whisper-small's encoder linears took as long at Q8_0 as at F32 on a Pi 4.
XNNPACK's dynamic int8 GEMM runs 2.4-2.7x faster than that on the same core.

**Change.** An extra buffer type, `CPU_NEON_I8`, registered on AArch64 when `ggml_cpu_has_dotprod()` is false
(`GGML_CPU_NEON_I8=0` switches it off). It takes a 2-D Q8_0 weight whose blocks all share their row's scale,
which is per-channel int8 written as ordinary Q8_0. At `set_tensor` it repacks the weight into 8-column panels
in XNNPACK's `c2s4` layout. A row with mixed scales stays as given, and ggml's own path runs it. `mul_mat`
quantizes each activation row per 32 k, symmetric (ggml's Q8_0 rule), into the work buffer. After the barrier,
XNNPACK's `qd8-f32-qc8w-gemm 2x8c2s4__neon_mlal` inner loop (BSD-3, notice carried in the source) runs on 2-row
x 8-column tiles. Each 32-k block's int32 sum is FMA-scaled into a float accumulator, and the weight's scale
is applied at the end. A few rows are dealt panel by panel round robin across threads, so DRAM sees one stream.
Many rows go in 160 KB blocks of panels, per thread, that stay in L2. `GET_ROWS` reads the packed panels too,
so a tied embedding (the lookup and the output projection) needs no second, plain copy (Qwen3-0.6B: 165 MB).
Weights reach the buffer only by placement, so nothing changes for a caller that does not place them.

**Numerics.** Integer sums plus a fixed float order, held bit for bit to a scalar reference in loom's
`test_mul_mat_neon_i8` (both schedules, 1/2/4 threads, odd shapes, the int16 bound, NaN rows). Against F32 it
has the error of ggml's own Q8_0 path: whisper-small's encoder output 4.806e-2 against 4.814e-2. XNNPACK's own
per-row asymmetric activations measured 9.8e-2 there, because whisper's outlier channels set the whole row's
range. That is why the activations are quantized per 32 k: it costs 14% of the GEMM's speed. `GET_ROWS` is
byte-identical to ggml's Q8_0 `GET_ROWS` (the same `q * d`).

**Measured** on a Pi 4 at four threads, against ggml's Q8_0 path on the same weights. Whisper-small shapes,
1500 rows: 2.41x (768 x 768), 2.44x (768 to 3072), 2.29x (3072 to 768), 59 GOP/s. One-row decoder steps:
1.09-1.12x. End to end, 4 threads, `GGML_CPU_NEON_I8` off against on: whisper-small 1.547x (21.42 -> 13.85 s, 7
paired launches, p10 1.538, p90 1.554, same transcript); Qwen3-0.6B 1.190x (24 tokens, 4.95 -> 4.16 s, same
tokens); Moonshine tiny 1.191x (p10 1.185, p90 1.197; this path transcribed jfk exactly, ggml's Q8_0 path read
"as" for "ask"); Citrinet-256 1.003x (its pointwise convolutions are not served here, only 0.4 MB of weights).

## PR 44 — `cpu`: an outer-product F32 GEMM for AArch64 cores without DOTPROD, under an L2-blocked schedule

*(`cmake/patches/ggml-0048-neon-f32-outer-product-gemm.patch`)*

**Problem.** On a Cortex-A72 (Raspberry Pi 4) tinyBLAS's F32 GEMM runs whisper-small's encoder linears at
~8.3 GFLOP/s on one core (58% of the FMA roofline) and 15.7-24.2 at four, where onnxruntime's MLAS reaches 30-35
in the same model. XNNPACK's kernel for this core, `f32-gemm 4x8 aarch64-neonfma-cortex-a75-prfm` (an outer
product: no horizontal reduction, 12 loads per 32 FMAs), runs at 10.6 on one core. Its own operator walks 4
activation rows against all of the weight, though, and at 2.4-9.4 MB of weight that re-reads DRAM for every 4
rows: 7-10 GF at four threads.

**Change.** In `sgemm.cpp`, for AArch64 when `ggml_cpu_has_dotprod()` is false (`GGML_CPU_NEON_F32_4X8=0` turns
it off): the 4x8 tile in NEON intrinsics (GCC 14 compiles it to the asm kernel's instruction mix, 98-99% of its
speed), under a schedule that packs one block of the weight, `nb x kc` with `nb x kc x 4 = L2 / 4` (256 x 256
on a 1 MB L2), into 8-row k-major panels. The threads pack it together, a barrier follows, and then each walks
its share of the activation rows against the whole block, 4 rows at a time. The sum is carried through C from
one block of k to the next. There is no software prefetch: one `prfm` per step at XNNPACK's spacing covered half
the lines and cost 20%, and two tied with none. It takes F32 x F32 graph `MUL_MAT` nodes with at least 128
activation rows (`dst->op == GGML_OP_MUL_MAT`; ops.cpp's `ggml_call_mul_mat*` keeps its im2col patches in the
work buffer the pack uses). `ggml_graph_plan` reserves the block for every F32 x F32 `MUL_MAT`.

**Numerics.** Each output is one sequential fused sum over k: bit-exact to a scalar `fmaf` loop and the same at
every thread count (loom's `test_mul_mat_neon_f32_4x8`: edge panels, blocks of rows and of k, ragged k and n, a
strided weight, a batch through one work buffer, at 1/2/4 threads). It is not tinyBLAS's order: the error
against a double product is ~2x tinyBLAS's (5.0e-7 vs 2.5e-7 relative at k = 768). The threshold does not
depend on the thread count, so neither does which sum a product gets.

**Measured** on a Pi 4 at four threads, against tinyBLAS, same process, distinct weights per node: whisper
`qkv` 768 x 768 x 1500 1.32x (31.9 GF), `fc1` 1.31x (31.7), `fc2` 3072 x 768 x 1500 1.71x (27.0). Across 27
shapes (whisper, VITS's convolution GEMMs, Moonshine, attention) it wins from 200 activation rows up
(1.11-1.58x) and loses below 64. End to end (`GGML_CPU_NEON_F32_4X8` off/on, paired launches): whisper-small
F32 **1.174x** (24.6 -> 20.9 s, same transcript), Moonshine tiny 1.057x, whisper-small Q8_0 1.082x (its attention
products are F32 x F32). Citrinet, VITS (byte-identical audio)
and Qwen3-0.6B are unmoved: their products are convolutions or under 128 rows.

## PR 45 — `cpu`: the convolutions' im2col GEMMs on PR 44's outer-product path, with a block of their own

*(`cmake/patches/ggml-0049-conv-f32-outer-product-gemm.patch`; depends on PR 44)*

**Problem.** PR 44's path packs into `params->wdata`, so only graph `MUL_MAT` nodes may call it: the batched
im2col in `ggml_compute_forward_conv_2d_impl` and the GEMM in `conv_transpose_1d` call mul_mat with their own
data already in that buffer. Their GEMMs stayed on tinyBLAS.

**Change.** In `ops.cpp`, on AArch64 with `GGML_USE_LLAMAFILE`: a convolution carves PR 44's block
(`llamafile_sgemm_f32_wsize()` bytes) off the end of its work buffer before sizing its patches, and calls
`llamafile_sgemm_f32_packed` with a copy of `params` whose `wdata`/`wsize` are that block. That happens at all
three of `conv_2d_impl`'s GEMM calls (straight into dst, staged and deferred, through the permute) and at
`conv_transpose_1d`'s, each keeping its tinyBLAS call as the fallback. `ggml_graph_plan` adds the block to
`CONV_TRANSPOSE_1D`'s reservation; `CONV_2D`'s 16 MB already holds it. `GGML_CPU_NEON_F32_4X8_CONV=0` switches
this off alone.

**Numerics.** As PR 44's: one sequential fused sum per output, here in im2col order, bit-exact to a scalar `fmaf`
loop and the same at every thread count (loom's `test_conv_neon_f32_4x8`).

**Measured** on a Pi 4 at four threads, off/on, paired launches: VITS (piper) F32 **1.068x**, its Q8_0 1.054x
(the kernels are dequantized to F32 first), Citrinet-256 1.062x / 1.060x, Moonshine tiny 1.034x, whisper-small
1.000x. It does not touch the direct 1-D kernel (`ggml_conv_1d_direct_ok`, a loom patch), which keeps VITS's
vocoder: through im2col and this GEMM those shapes were slower.

## PR 46 — `cpu`: `PAD` and `CONCAT` a row at a time

*(`cmake/patches/ggml-0050-cpu-pad-concat-row-copies.patch`)*

**Problem.** `ggml_compute_forward_pad_f32` and `ggml_compute_forward_concat_f32` move one element per
iteration, forming 64-bit addresses and (for `PAD`) four range tests each time. On a Pi Zero W's ARM1176 that
is ~180 ns an element for `PAD` and ~100 for `CONCAT`: Pocket-TTS's Mimi decoder pads two 2.8M-element
activations before causal convolutions, 1.0 s of an 18 s synthesis.

**Change.** When the source rows are contiguous floats (`nb00 == 4`, and for `CONCAT` both sources and dst):
`PAD` (non-circular) writes each row as a zero head, one `memcpy` and a zero tail, or a whole zero row;
`CONCAT` copies each dst row from its source row(s), both halves of it along dim 0. Threads split rows as the
loops did. Everything else keeps the loops.

**Numerics.** A copy: byte-identical, NaN payloads, `-0` and denormals included (loom's
`test_pad_concat_rows`, contiguous, strided and transposed sources on every axis, F32 and I32).

**Measured** (Pi Zero W, one thread): Mimi's `[115202, 24]` pad 520 -> 54 ms. A `CONCAT` of rows of one float
(RoPE's pair swap, `[1, 64, 960]` twice) is not faster: two `memcpy` calls a row cost what the loop did.

## PR 47 — `cpu`: ARMv6, binary-op rows walked with counters

*(`cmake/patches/ggml-0051-armv6-binary-rows-without-divides.patch`; depends on loom's ggml-0037)*

**Problem.** `apply_binary_op` derives each row's `(i01, i02, i03)` with three 64-bit divides and its
broadcast `src1` row with three 64-bit modulos. On 32-bit ARM each is a libgcc call: ~1 us a row on an
ARM1176. With short rows that is the op: Pocket-TTS negates a strided `[1, 64, 960]` view (one float a row)
eight times, 61 ms each.

**Change.** On ARMv6, where ggml-0037's row kernels are in use: one divide per thread to find its first row,
then increments, the broadcast indices wrapping with them; a row of one float is computed inline.

**Numerics.** The same IEEE operation on the same operands: byte-identical (loom's `test_eltwise_armv6`,
3-D and 4-D broadcasts that wrap every counter, strided one-float rows, edge values, both registrations).

**Measured** (Pi Zero W): the `[1, 64, 960]` multiply by a scalar 61 -> 8.8 ms.

## PR 48 — `cpu`: ARMv6, `ELU` on the VFP's short vectors

*(`cmake/patches/ggml-0052-armv6-elu.patch`; depends on loom's ggml-0039)*

**Problem.** `op_elu` calls glibc's `expm1f` per element: ~150 ns on an ARM1176, 2.0 s of a Pocket-TTS
synthesis (its SEANet decoder has an ELU before every convolution, up to 2.8M elements each).

**Change.** A fifth kernel in ggml-0039's generated set (loom's `scripts/gen_armv6_act_f32.py`), on the same
Cephes exp with v = x: where the reduction leaves n = 0 (|x| < ~0.35) the polynomial gives `expm1` before
its 1 is added, so there is no cancellation; elsewhere `e^x - 1`, which cannot cancel there (`e^x <= 0.71`).
x itself for x >= +0, for |x| < 2^-24 and for a NaN, all decided on the integer bits.

**Numerics.** Not glibc's: 1.24 ULP worst on the test's inputs (glibc 0.70), `-inf` -> -1, `+-0` kept
(loom's `test_activations_armv6`; with the n = 0 branch removed it reads 122 ULP).

**Measured** (Pi Zero W): Mimi's `[115200, 24]` ELU 420 -> 209 ms, 2.0 s -> 0.9 s over the decoder.

## PR 49 — `cpu`: AArch64, the direct 1-D convolution on per-thread staged slices and an 8 x 12 tile

*(`cmake/patches/ggml-0053-conv1d-staged-8x12.patch`; depends on loom's ggml-0006/0007)*

**Problem.** loom's direct 1-D convolution (the long-activation, small-weight case: HiFi-GAN resblocks)
copies the whole activation, zero-padded, into the work buffer before sweeping it: at 32 channels x 73216
positions that is 9.4 MB written and read back from DRAM per convolution, with no arithmetic over it. Its
weights are packed `[ic][kx][oc]`, so a 4-channel tile reads one new cache line per (channel, tap), from
L2 once the weights pass L1. On a Cortex-A72 it ran VITS's resblocks at 14-24 GFLOP/s.

**Change.** On AArch64 (by default on cores without DOTPROD), for OC a multiple of 8: weights packed per
8-channel tile, an 8 x 12 tile (24 named accumulators), and no global copy. Each thread stages its own
positions ~16 KB of activation at a time (through the fused LEAKY_RELU, as the copy did), carries each
slice's halo into the next, and saves its two cross-thread halos before the barrier, so an output written
over its input stays correct. `GGML_CPU_CONV1D_STAGED=0|1` overrides.

**Numerics.** The same order (bias, then each channel's taps as fused multiply-adds, then the residual):
byte-identical to the direct and phase-major kernels (loom's `test_conv_1d_staged`, bit-exact against an
`fmaf` loop at 1-4 threads, in place and not, both registrations).

**Measured** (Pi 4, four threads, off/on, paired launches): the eighteen resblock shapes 1.91x
(`scripts/bench67.cpp`); end to end VITS (piper) F32 **1.301x** (1.041 -> 0.799 s) and its Q8_0 1.284x, the
audio identical; Citrinet-256 1.070x, Pocket-TTS 1.03-1.07x; Moonshine tiny (no direct convolution) 1.002x.

## Not a PR here, but upstream should know: `ggml_get_n_tasks` no longer decides what it looks like it decides

**No patch in this directory depends on this.** It was found by loom P4.25, which built a patch on the
opposite belief, and then by P4.27, which found out why that patch measured nothing. Both write-ups are
in loom's Epic-05 §5 and
[Retro-023](https://github.com/loom-ai-org/loom.cpp/blob/main/docs/retros/retro-023-a-bench-whose-graph-was-the-treatment.md).

`ggml_get_n_tasks` reads like a per-node thread count, and in this version it is not one. It is called
in **exactly one place** — `ggml_graph_plan` — where it sizes the work buffer and feeds `max_tasks`,
and then:

```c
cplan.n_threads = MIN(max_tasks, n_threads);
```

`ggml_graph_compute_thread` runs **every node on every thread** with `params.nth = cplan.n_threads`.
An op that must not split says so itself — `ggml_compute_forward_sum_f32` and
`ggml_compute_forward_leaky_relu_f32` open with `if (params->ith != 0) return;`. So `n_tasks` survives
only as a **graph-level clamp**, which can take a whole graph down to one thread and can do nothing
else.

**Three consequences, and the third is the one that costs people time.**

1. **The table and the ops disagree.** `GGML_OP_UNARY`'s `TANH`, `SIGMOID`, `EXP`, `RELU`, `ELU` are
   declared `n_tasks = 1`, but `apply_unary_op` splits over rows through `get_thread_range` and threads
   3.9-4.8x — measured on a Cortex-A72 at 4 threads, a Core Ultra 9 285K at 24 and a Ryzen at 2. The
   declaration is ignored in every graph that also holds one op declaring more, which is every real
   model graph. Nothing is *wrong* — but a reader who trusts the table gets the opposite of the truth,
   and a patch that "fixes" the table changes nothing.
2. **A homogeneous graph silently runs single-threaded.** Build a graph of nothing but `TANH` nodes and
   it plans one thread however many you asked for, even though every one of those nodes would have
   split. It is easy to write such a graph by accident in a benchmark, and it makes the benchmark
   measure its own plan.
3. **Anything that computes node-by-node changes the thread count of those nodes.**
   `ggml_backend_sched`'s eval-callback path computes one node at a time, and so does any profiler
   built on `ggml_graph_view(graph, i, i + 1)`. Under such a tool every `n_tasks = 1` node is planned
   at **one** thread while the rest of the report is threaded — in one small TTS model at 4 threads
   that is 122 `UNARY`, 126 `SUB`, 106 `SCALE`, 42 `SUM_ROWS` and 32 `LEAKY_RELU` nodes. loom read a
   threaded op as "30.4 ms, identical at one thread and four, three cores idle" for four days on the
   strength of it, and scoped two items on that reading.

**A one-row tensor still cannot split by rows**, which is worth saying separately because it is a real
(if small) limitation rather than a misunderstanding: `get_thread_range` divides `ggml_nrows(src0)` by
`nth`, so at `nr = 1` thread 0 gets everything. Measured `ne0 = 256`, four threads against one, Pi 4B,
256 nodes per graph:

| `nrows` | 1 | 2 | 3 | 4 | 8 | 16 | 64 | 192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| speedup | **1.00x** | 1.75x | 2.36x | 2.91x | 3.36x | 3.69x | 3.89x | 3.92x |

In a real graph this costs nothing — the threads handed an empty range reach the per-node barrier they
were going to reach anyway — so it is a missed opportunity (split on `ne0` when `nrows` is small), not a
regression. **An earlier version of this section proposed `n_tasks = MIN(n_threads, ggml_nrows(node))`
as a one-line fix that "removes only barrier participation". That was wrong**: `n_tasks` does not gate
barrier participation, and the only thing that cap can do is clamp an all-one-row graph to a single
thread.

**A work floor is still missing, and that one is real.** At 24 threads on a Core Ultra 9 285K the
per-node cost bottoms out at **~1.9 us**, so threading a unary below ~16K elements is a loss there
regardless of row count — against ~0.3 us at 4 threads, so the floor is machine-dependent. Any future
change that threads more ops needs one.

## Not a PR here, but upstream should know: the OpenMP wait-policy mitigation covers only Intel's `libomp`

`ggml-cpu.c` (v0.19.0, around line 4114, inside `#ifdef GGML_USE_OPENMP`) sets `KMP_BLOCKTIME` and
nothing else:

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

**`KMP_BLOCKTIME` is Intel/LLVM `libomp`'s knob and GNU `libgomp` ignores it entirely.** So a
gcc-built `GGML_OPENMP=ON` — which is what Debian, a Raspberry Pi, and any manylinux wheel produce —
gets no mitigation at all, and the line that would have covered it is commented out. What that costs,
measured on a **Core Ultra 9 285K, 24 threads, a 2520-node VITS graph**: every thread sleeps on a futex
at each node's `#pragma omp barrier`, **334,609 voluntary context switches over five syntheses against
160** without OpenMP, and the thread-count curve stops being monotonic — more cores make it slower.
`GGML_OPENMP=OFF` is **4.8x** there. At four threads it is 1.0x, so this is a many-core report;
[Epic-05 §2](../../docs/epics/epic-05-edge-performance.md) and
[Retro-017](../../docs/retros/retro-017-libgomp-slept-at-every-graph-node.md) carry the full table.

**And un-commenting the line would not fix it**, which is the part worth saying in the report rather
than leaving a reviewer to find: libgomp reads `OMP_WAIT_POLICY` in its **load-time constructor**, long
before `ggml_init` runs, so a `setenv` from inside the library is always too late. `OMP_WAIT_POLICY=active`
set from the *shell* is the fastest arm and is still the wrong default — it spins forever, including
between inferences (13.82 s of CPU for a run that takes 0.16 s of wall time), where ggml's own
threadpool spins a bounded ~6.5M rounds (`poll = 50`) and then sleeps.

So there is no environment variable that fixes this from where ggml stands. Two things upstream could
do, and neither is a patch we hold:

1. **Say so in the build system.** `GGML_OPENMP` defaults ON, and on the most common toolchain on the
   most common platform it is a large regression at high thread counts. A note at the option, or a
   default that follows the detected OpenMP runtime, would have saved this diagnosis.
2. **Delete the commented-out block, or replace it with the reason it cannot work.** As it stands it
   reads like a mitigation that is merely disabled, which is how loom first read it.

**A third item that was on this list is withdrawn, because it was not true.** Epic-05 §2 proposed
adding `ggml_backend_cpu_set_threadpool` to the CPU backend's registry proc-address table on the
grounds that it was absent and a `GGML_BACKEND_DL` build therefore could not tune `poll` in-process.
It is **present** at the pin — `ggml-cpu.cpp`'s `ggml_backend_cpu_get_proc_address` exports it beside
`ggml_threadpool_new` and `ggml_threadpool_free`, under a `// threadpool - TODO: move to ggml-base`
comment, and no patch of ours put it there. So the residual 24-thread gap (0.040 s against 0.031 s)
has an in-process route already and needs nothing from upstream; whether it is worth taking is a loom
question, not a ggml one. Checked by reading the pinned tree, 2026-09-02.

loom's own fix was the build flag: `LOOM_OPENMP`, default OFF (`cmake/Dependencies.cmake:73`).
