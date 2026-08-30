---
type: retro
date: 2026-08-29
domain: performance
tags: [ggml-patches, aarch64, benchmarking, measurement-hygiene, p4.18]
---

# Retro-019: A Kernel Patch Measured On One ISA, Shipped To Both

## The Issue

`ggml-0011` landed as a **2.15x** win at whisper's `A@V` shape and **1.19-1.24x end to end on
Conformer-CTC**. Every one of those numbers is from an x86 machine. On a Raspberry Pi 4 — the board
this engine names as its reference target — it is a **1.17-1.25x regression on whisper and 1.13-1.15x
on VITS**.

It was found four days later, by accident, while re-measuring the README's ASR column for an unrelated
item: the Pi's ASR cell had fallen 0.58x to 0.45x and nothing in the change log predicted it. Measured
across the whole Pi row against onnxruntime, as shipped against the same tree without the patch:

| Pi 4B, 4 threads | as shipped | without `ggml-0011` | published 2026-08-24 |
|---|---|---|---|
| TTS | **0.84x** | ~0.95x | 0.98x |
| LM | **1.05x** | 1.06x | 1.08x |
| ASR | **0.45x** | ~0.54x | 0.58x |

**The LM row is the control**: a decode step's `mul_mat` has `ne1 = 1` and never reaches the blocked
GEMM this patch touches, and it does not move. Both tasks that do reach it lose. Epic-05's headline
result — 2.2x slower to 1.03x on a Pi 4 — reads 0.84x on the shipped code.

## Root Cause Analysis

The patch's own analysis said why it could not help on aarch64, and that was read as "harmless".

tinyBLAS opened with `if (k % KN != 0) return false;`, handing any matmul whose contraction was not a
whole number of vectors to ggml's generic one-element-per-call kernel. On AVX2 `KN = 8` and whisper's
1500-frame contraction leaves 4, so the patch turns a rejected matmul into an accepted one. **On NEON
`KN = 4`, and 1500, 768, 3072 and 64 all divide 4** — the tail the patch adds is never taken by any
shape the model has.

What remains on that ISA is the patch's restructuring of the **aligned** path, and that path is not
free to touch. The same epic section already records that folding its epilogue costs the aligned case
**30%** on x86, which is why the shipped patch carries two epilogues with the branch outside the tile.
And `ggml-0001` exists only because GCC's register allocation for the NEON tile is fragile — a tile it
can actually allocate is worth 15.6 → 25.1 GFLOP/s. A second structural change to that same loop is
precisely where a NEON regression should have been expected.

**The measurement that would have caught it costs one run.** The Pi was not re-measured after the patch
landed, and the README's Pi row was left carrying a number taken before it.

## Resolution & Lesson Learned

**Fixed 2026-08-29 (P4.20), and the mechanism was narrower than any of the three fix options guessed.**
Reproducing it per shape made it *bigger* — 1.32-1.76x on every one of whisper's five encoder GEMMs,
none of which ever takes the tail — and a four-build bisect put it on one hunk: **not the `kk`
arithmetic, but the presence of the scalar tail loop inside the tile function.** A build with `kk`
truncated and the tail removed runs at full speed; a build with `kk = k` and the tail present runs
slow; folding the tail into one epilogue rather than two does not help. It reaches `Aat`/`Bat` after the
main loop and changes what GCC's aarch64 register allocator does with the 4x3 NEON tile — the same
fragility `ggml-0001` exists for, from the other side.

The tail now has its own `NOINLINE` function, dispatched on `k % KN` *before* the tile, so the aligned
path is instruction-for-instruction what it was before the patch on every ISA. x86 keeps the full 2.83x
where the tail fires and is unmoved where it does not; the Pi's three cells go back to 0.96x / 1.06x /
0.57x. [Epic-05](../epics/epic-05-edge-performance.md) has both halves of that table.

* **Actionable takeaway 1 — a patch to a kernel with per-ISA code paths needs a number on each ISA it
  is enabled for.** "The new branch is inert there" is a statement about the branch, not about the
  diff: the rest of the diff still moved code the compiler had to re-allocate registers for.
* **Actionable takeaway 2 — the symmetry was already on the record and was not read as a warning.**
  `ggml-0011` was *found* because P4.15 ran entirely on aarch64 and x86 was the ISA nobody had measured
  ("`scripts/bench6.cpp` being aarch64-only is *why* the whole of P4.15 ran on the one ISA where
  whisper's contraction happens to be aligned"). The same blind spot then ran in the other direction,
  four days later, in the same thread.
* **Actionable takeaway 3 — a regression on the reference device was invisible because the reference
  device was not in the loop.** Every patch after `ggml-0009` was measured on x86 first. The cheapest
  guard is a rule rather than a tool: **a `cmake/patches/` diff is not done until `UPSTREAM.md` carries
  a number from an x86 box and a number from the Pi**, even when one of them is "no change".
* **Actionable takeaway 4 — it surfaced only because a published table was re-derived from scratch.**
  The cell had been carried forward, and carrying it forward is what hid a 1.2x. That is
  [Retro-018](retro-018-a-table-of-ratios-nobody-could-re-derive.md)'s first lesson, paid a second
  time.

**What NOT to conclude:** that `ggml-0011` should have been reverted. On x86 it is one of the largest
wins in this thread, and the encoder split that P4.21 was aimed by depends on it — `A@V` went from
2.23x behind onnxruntime to 1.07x. The fix was placement, and it needed neither an architecture guard nor a
second code path.

**And one more takeaway, from the fixing rather than the finding.** The three fix options this retro
first listed were all plausible and all wrong about the mechanism; what identified it was **bisecting
the patch into its hunks and building each one**, four `.so` files of one source. A patch small enough
to review is usually small enough to bisect, and that is cheaper than reading the disassembly it was
going to take.

## It happened again, one patch later (2026-08-30, P4.26)

Takeaway 3 above asked for "a number from an x86 box and a number from the Pi". `ggml-0012` has both,
and it still shipped a 2.4% aarch64 regression, because **the number from the Pi was taken at one
shape** — whisper's `QK^T`, `m = 1500`, where the patch is genuinely neutral (133.10 ms against 133.65,
correctly ABBA-interleaved). VITS's matmuls are `m = 96 / 100 / 199`. Every one of them takes the branch
the patch adds, and there it costs **1.032x and 1.016x** over two ABBA rounds — ~27 ms per synthesis,
of which `CONV_2D` is +22.6 ms and `MUL_MAT` only +5.4, because `ggml-0004` and `ggml-0009` lower
convolution through the same `sgemm`.

It surfaced the same way the last one did: a published cell was re-derived from scratch (P4.16) and
disagreed with itself by 3%.

* **Takeaway 3, amended — a number per ISA is not enough; it has to be a number per SHAPE CLASS the
  patch is enabled for.** `ggml-0012`'s whole purpose is to change what happens when `m % 16 != 0`,
  and it was validated on aarch64 at an `m` from a different model. The shapes to run are the ones in
  `scripts/conv_census.py`'s output for the models that ship, not the one the patch was aimed at.
* **A patch to `sgemm` is a patch to the convolution.** Nothing in `ggml-0012`'s own reasoning mentions
  convolution, and convolution is where 80% of its cost landed. Before scoping a GEMM patch by the ops
  that call `ggml_mul_mat`, check which lowerings reach `sgemm` by another door.

P4.26 in [Epic-05](../epics/epic-05-edge-performance.md) is the open item, and it is deliberately *not*
"gate it on `__aarch64__`": the 2.75x it buys on x86 is a bigger number than the 2.4% it costs here.
