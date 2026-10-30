---
type: retro
date: 2026-10-29
domain: backends
tags: [vc4, cma, memory, device-selection, auto, raspberry-pi-zero, measurement]
---

# Retro-100: A Free-Memory Figure That Is a Lower Bound

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (P4.46); the decision is
[ADR-092](../adrs/adr-092-the-qpus-take-the-epilogue-q4-0-and-the-pointwise-convolutions.md).

## Issue

The P4.46 baseline profile on the Pi Zero, same package as P4.45's run 5, put Moonshine tiny Q8_0_PC
and conformer-ctc Q4_0 on the CPU: "'auto' chose CPU: VC4 has 27.1 MB free and the weights need
29.1 MB" (and 36.8 against 44.1). In run 5, an hour earlier, both had run on the QPUs -- Moonshine
1.52x faster. Which device "auto" picked depended on what the board had done before.

## Root cause

ggml-vc4 reported the pool's free memory as the kernel's `CmaFree`, and `CmaFree` is not what an
allocation can get. The kernel lends idle CMA pages to movable allocations -- the page cache, mostly --
and takes them back by migration when a contiguous allocation needs them. With the pool idle,
`CmaFree` read 44.3 MB of 64; after `drop_caches`, 55.7. Worse, within a process a freed buffer never
shows up in `CmaFree` again, though the next allocation reuses it (20 MB held: 43.2 MB free; freed:
43.4; the same 20 MB allocated again: 43.2). A probe allocation to force the migration was no way out:
a 60 MB probe was not handed back until the process exited, and a large first request can fail
(EBUSY) while pages are still migrating, with the next, larger one succeeding.

## Takeaway

**Know what a resource counter counts before deciding with it.** `CmaFree` is free pages in the CMA
region at this instant -- a lower bound on what `cma_alloc` can produce, by however much the kernel has
lent out. A decision made from a lower bound is conservative, and here that was also nondeterministic,
which is worse: an A/B that lands on different devices in different runs measures nothing. The fix is
the figure the backend can know: the pool less what this process holds (counted at every GPU
allocation) less a slack for other users, or `CmaFree` when that is larger. What it cannot know is
another process holding the pool; that case now fails at allocation and says so, instead of quietly
picking the CPU.

## Record

* Board, `/proc/meminfo` and `cma_probe` / `cma_free` (`~/.claude/tmp/p446/board/`): CmaFree 63.9 MB at
  boot, 44.3 MB idle after the run-5 workload, 55.7 after `echo 3 > /proc/sys/vm/drop_caches`.
* A 40 MB dma-heap allocation with 55.7 MB free failed (dmesg `cma_alloc ... ret: -16`); a 56 MB one
  right after it succeeded. `GpuBuffer` now retries a failed allocation once.
