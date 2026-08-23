---
type: retro
date: 2026-08-14
domain: infrastructure
tags: [testing, gates, backend-dl, stale-documentation]
---

# Retro-008: The Prerequisite Was Already Done, and the Gate Was Green for the Wrong Reason

## The Issue

P4.8b closed by naming a blocker: 109 test files called `ggml_backend_cpu_init()` directly, that symbol
lives in the CPU backend, and a `GGML_BACKEND_DL` build dlopens rather than links it — so **the gate
suite could not run against the configuration the wheels ship**. Picking that up as the next task found
it already finished: a commit had converted 112 files through `tests/support/cpu_backend.h`, and the
ledger text was simply never updated to say so.

## Root Cause Analysis

The ledger entry and the tree had diverged. Nothing re-checked the claim because the gate was green —
and the gate was green under the build configuration the developer runs, not the one that ships.

## Resolution & Lesson Learned

* **Actionable takeaway 1 — verify a recorded blocker before scheduling work against it.** A stale
  "still open" costs the same as a stale "done".
* **Actionable takeaway 2 — a gate proves something about the configuration it ran in.** If the shipped
  artifact is built differently, the gate has not tested the artifact. Confirm the gate can go red in
  the shipping configuration; see [ADR-015](../adrs/adr-015-ci-and-gate-test-classes.md).

---

## Full record (verbatim from the ledger)


P4.8b ends by naming a prerequisite: 109 test files call `ggml_backend_cpu_init()` directly, that
symbol lives in the CPU backend, a `GGML_BACKEND_DL` build dlopens rather than links it, so **the gate
suite cannot run against the configuration the wheels ship**. Picking that up as the next task found it
already finished: commit `3cb5723` converted 112 files through `tests/support/cpu_backend.h`, and the
P4.8b text was simply never updated to say so. The link half has been fine for a while.

**The run half had not been checked, and it was broken in the worst available way.** Under
`GGML_BACKEND_DL` the two device-parity gates — the entire evidence that the device layer generalises —
did not run. They exited 77. ctest renders 77 as **Skipped** and the suite as **green**, so a build
with Vulkan compiled in reported success while running neither gate. Every earlier "82/82 on the DL
build" in this ledger, including this session's pin-bump verification, was counting those two as passes
when they had quietly declined to execute.

The cause is narrow: `cpu_backend.h` registers this build's backend directory before main (that is what
makes a DL registry non-empty), and the parity gates never included it — they reach the device layer
through `loom::available_devices()`, not through a CPU backend. So they saw a registry holding only the
CPU and concluded, reasonably and wrongly, that the machine had no GPU.

### Two fixes, because one of them only removes the instance

1. **`test_util.h` now includes `cpu_backend.h`.** All 116 test files include `test_util.h`, so
   registration stops being something a new test can forget. The include is there for a static
   initialiser rather than for anything it names, which is a standing invitation to delete it as unused,
   so the comment says so at the point of the include.
2. **A skip that would be a lie is now a failure.** `tests/CMakeLists.txt` defines
   `LOOM_TEST_EXPECTS_DEVICE` when any device backend is configured, and with it defined the gates
   return 1 instead of 77: "a device backend is compiled in and the registry has no device" is a broken
   deployment, not a machine without a GPU. Fix 1 alone would have left the next such bug silent again.

### Verified, including that the new failure can actually fire

* Both gates now **Passed** rather than Skipped under `ctest` on the DL build (Vulkan + BLAS + CPU).
* **The red case was produced on purpose**: moving `libggml-vulkan.so` and `libggml-blas.so` out of
  `bin/` turns both gates from Passed into `***Failed`, and restoring them turns them back. Without
  that check this entry would be asserting a guard that had never been observed to do anything.
* `ci` 58/58 and `gate` 82/82 on the DL build, and the same on the linked CPU build, where
  `LOOM_TEST_EXPECTS_DEVICE` is undefined and the honest skip still applies.

The lesson is the one `CLAUDE.md` already states and this suite still managed to violate: a gate that
cannot fail proves nothing, and **exit 77 is the most dangerous return code in the suite** because it
is indistinguishable from success in the summary line. Any future skip condition should be read as a
claim about the machine that something ought to be able to contradict.

