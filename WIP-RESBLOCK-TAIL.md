# KEEP THIS BRANCH — shelved prototype, do not delete, do not merge as is

`wip/p432f-f2-resblock-tail` holds `ggml-0040`, the resblock-tail fusion prototype of ADR-079 P4.32f f2
step 3b, and its CI test (`tests/ci/test_conv_resblock_tail.cpp`). It is the starting point for the hub's
open item **P4.32f f2 step 3c** (`docs/backlog/active-index.md`).

**What it does.** After a resblock convolution's fused bias and residual, it also fuses the stage's tail:
the ADD of the previous resblock's output (or the running sum), the MUL by the 1/3 scalar, and the
LEAKY_RELU feeding the next transposed convolution. It is bit-identical in `tests/ci`, including the case
where a graph allocator puts the sum in place on its dead operand. Three sabotages go red.

**Why it was shelved** (board, Pi Zero W, 2026-10-16, ADR-079 "Step 3b"): amy 1.009x, de051 1.000x,
VITS 0.994x. Its passes re-read output rows just stored, which an ARM1176 does not cache, so each costs
what `ggml-0037`'s separate pass does. A tail that trips an aliasing check also makes the matcher
decline the whole fusion, conv + bias included.

**What step 3c should change.** Do the tail in ONE short-vector pass on the block before it is stored,
and back off to the plain conv + bias fusion when the tail cannot alias.

**Base.** Branched from `perf/p432g-fft` at `0c685ef`, before `ebccf36`, which amended `ggml-0036` to
default its epilogue OFF. This prototype's matcher sits behind the same `GGML_CPU_CONV_EPILOGUE` flag,
so rebase it onto that commit or later, and give the tail its own switch, before measuring again.
