---
type: retro
date: 2026-09-12
domain: engine
tags: [engine, graph-builder, output-store, drivers, tts]
---

# Retro-045: The Allocator Handed the Same Address Back

## Issue

Kokoro's second synthesis **in the same process** corrupted the heap. The symptom was
`malloc_consolidate(): invalid chunk size`, aborting inside an unrelated `operator new` several
bindings later, and it appeared only through `scripts/bench_driver.cpp` (which calls `infer` twice: one
warm-up and then the timed runs) with `$LOOM_PROFILE` set. Every test in the suite passed. The same
model, the same driver and the same engine produced a **bit-identical waveform** through `loom_cli` and
the e2e gate tests, both of which call `infer` exactly once per process.

## Root cause

A retained graph ends in one `ggml_cpy` per declared output whose DESTINATION is the module's
`OutputStore` tensor, and `GraphBuilder` serves that graph back whenever the axes repeat. ADR-032 gave
the drivers two bindings that reshape a store **without a build** — `loom.expand_by_duration_and_retain`
and `run_bi_recurrent_and_retain` — so a cached graph could now be pointing at tensors a later call had
freed. That much was anticipated: `build()` was given a guard that compares, on a cache hit, the slots
the cached graph writes to against the slots the store holds now.

**The guard compared POINTERS, and the pointers lied.** `OutputStore::reshape` frees its context and
backend buffer and immediately allocates replacements of a similar size; glibc hands the same addresses
straight back. The comparison therefore reported "unmoved" for tensors that no longer existed, the
cached graph was served, and its `cpy` wrote into a freed chunk. The visible damage was the allocator's;
the silent damage was worse — Kokoro's second call read a `[640,100]` frame-expanded sequence out of a
store whose driver had just rewritten it at `[640,32]`, which the shape check on the next binding
happened to catch only because the two disagreed.

`OutputStore` now carries a `layout_epoch()` — bumped whenever `reshape` actually reallocates — and
`GraphBuilder` keys its cache on that. A counter cannot be recycled.

## Takeaways

* **A pointer is not an identity.** Any check of the form "is this still the object I saw?" that
  compares addresses is answering a different question, because an allocator recycles addresses
  eagerly and by preference. Compare a monotonic counter the owner bumps, and the question becomes the
  one you meant to ask.
* **AddressSanitizer said the run was clean, and was right to.** ASAN's quarantine means a freed block
  is never handed back promptly, so the stale pointer never matched and the guard took the correct
  branch — the sanitizer's own allocator made the bug unreachable. When a corruption reproduces under
  the system allocator and vanishes under ASAN, "ASAN found nothing" is evidence about ALLOCATION
  ORDER, not about the absence of a bug.
* **Every TTS gate test calls `infer` once per process, so the second call is an untested regime.**
  This bug lived entirely in state that only exists between two calls: a cached graph, a retained
  store, a reshaped buffer. The suite could not have caught it at any effort level, and what did catch
  it was a benchmark harness whose warm-up call exists for timing reasons. A second call is now part of
  what `test_e2e_kokoro_mil_lua_driver` asserts.
* **A profiler is a probe for more than time.** `$LOOM_PROFILE` runs a graph node by node, which is a
  different allocation pattern — enough to turn a silent write-after-free into a fatal one. It found
  this the way a sabotage run finds a harness defect.
