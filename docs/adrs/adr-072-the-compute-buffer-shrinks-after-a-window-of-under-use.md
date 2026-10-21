---
type: adr
status: accepted
date: 2026-10-11
tags: [allocation, gallocr, compute-buffer, shrink, page-faults, tts, causal-lm, host-api]
supersedes: []
---

# ADR-072: The Compute Buffer Shrinks After a Window of Under-Use

Serves [Epic-01](../epics/epic-01-inference-engine-core.md) §2 (allocation). It revises P4.0.16 and
came out of the LiteRT-borrowing thread on the hub
([Engine — performance](../backlog/active-index.md#engine--performance)).

## Context

`ggml_gallocr` grows its buffer and never shrinks it. Since P4.0.13 a module's builder lives for the
whole session, so P4.0.16 made the builder give the buffer back itself. A growth of more than 2x armed a
probe, and the next build that needed at most half the buffer dropped it. That was tuned on one
workload: an LM's prefill to decode transition (gemma-3-270m-it, 513.2 MiB held where 1.0 MiB was
needed), which happens once per generation.

A TTS module serving sentences of mixed length meets the rule from the other side. On VITS (en-GB miro,
Ryzen 3 3250U), alternating a 50-token and a 198-token sentence armed and fired the shrink on every
switch. The vocoder's 111 MB compute buffer was freed on every short call and faulted back in on every
long one. A long call took **38 740 minor faults against 1 372** when it repeated the previous shape. At
a measured 1.6 µs per first-touched page that is ~46-60 ms of a ~0.95 s synthesis, and the short call
paid its own 6 764 against 288. A gradual 50 → 100 → 198 cycle never shrank, because no single step
grew more than 2x. Everything a real server does falls somewhere between the two.

The same measurement priced what a graph rebuild costs: **~4-5 ms of host work per call**
(1547-node `text`: 2.1-2.5 ms build + ~0.9 ms allocation; 651-node `flow_vocoder`: under 1 ms). That
is why a multi-slot graph cache was not built instead
([Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md)).

## Options

1. **Keep P4.0.16.** This is right for the LM and pays a 111 MB refault per switch on mixed-length TTS.
2. **Hysteresis: never shrink a module again once it has regrown to a size it shrank from.** This fixes
   the alternation, but one unlucky sequence pins the memory for the builder's whole life, including
   across an hour of short traffic after one long paragraph.
3. **A window of under-use: shrink only after `kShrinkWindow` consecutive calls each needed at most
   1/`kShrinkFactor` of the buffer.** An alternating workload never completes the window, a regime
   change still shrinks once, and a burst followed by quiet gives the memory back after the window.
4. **A statistical or learned predictor of the next call's need.** There is one number per module per
   call. On a bimodal workload a sliding maximum is what such a predictor converges to anyway. A model
   would make memory behaviour history-dependent in a way nobody can debug on a Pi Zero, and it adds
   runtime machinery against [ADR-003](adr-003-per-model-complexity-in-the-exporter.md)'s lean-engine
   rule.
5. **Wall-clock rules ("idle for T seconds").** These make the tests depend on timing. Whether a host
   is idle is something only the host knows.

## Decision

Option 3 as the default, plus two controls for the host. Option 5 is rejected as a builder policy,
and the host gets an explicit call instead.

* **Window:** `kShrinkWindow = 8` calls, `kShrinkFactor = 2`. Factor 2 is P4.0.16's own, for the same
  reason: it is the smallest number that separates a regime change from `n_kv` creeping by one bucket.
* **Every call counts, reuses included.** A reuse's need is the retained graph's, recorded when it was
  built (`cached_need_`). If only rebuilds counted, a decode loop reusing its graph for `kKvBucket`
  steps would hold a prefill buffer for up to 8 x 32 tokens.
* **A shrink due on a reuse becomes a rebuild.** The retained graph's tensors point into the oversized
  buffer, and `gallocr` cannot move them. So the shrink costs one graph build per regime change.
* **One planning pass per rebuild into a live buffer.** The pass runs on a persistent, buffer-less
  probe gallocr, never on a reuse, and never on a fresh allocator (whose size *is* the need). P4.0.16
  avoided per-build probing because, before P4.0.15, every decode step rebuilt. Now the pass lands only
  on calls that already pay a graph build: **0.31 ms** median on VITS, and **6.5 ms across ~190
  generated tokens** on Qwen3-0.6B (~0.03 ms/token against 167 ms/token).
* **Host controls** (`GraphBuilder`, and bridge-wide on `LoomLuaBridge`):
  `hold_compute_buffer(true)` turns the automatic shrink off. It is the dynamic-shape form of LiteRT's
  pre-planned arena: hold whatever the largest request needed, and never fault it back.
  `release_compute_buffer()` drops the retained graph and its buffers now, scheduler included on a
  device build, the way `malloc_trim` does for a process. `compute_buffer_bytes()` reports the total.
  These are C++ accessors, not Lua bindings: when to hold memory is the host's call, not the driver's.
  Releasing a hold also drops the retained graph, because a held builder skips the probe, so the need
  it recorded is the whole buffer.

## Consequences

* **Positive:** mixed-length TTS no longer refaults. The same VITS alternation now takes 1 653 faults on
  a long call against 1 372 for a repeat, with zero shrinks and identical audio digests. An LM still
  gives back its prefill buffer once per generation, 8 decode calls in.
* **Positive:** every case is countable, so the contract is tested deterministically
  (`tests/ci/test_gallocr_shrink.cpp` cases 1-9). Four sabotages went red: a window of 1, reuses not
  counted, a big call not resetting the window, and releasing a hold keeping the graph.
* **Negative:** an LM **server** pays the same per-request refault P4.0.16 always charged. A generation
  is longer than 8 calls, so each one shrinks after its prefill, and the next prompt grows the buffer
  back. For a large prompt that is the 513 MiB case. This is what `hold_compute_buffer(true)` is for,
  and choosing it is the host's job. loom-py does not expose either control yet.
* **Negative:** the decode-loop reuse test now counts one more build (4, not 3) for the 41-call loop.
  That is the shrink's rebuild, and its outputs still match the rebuild-every-step oracle bit for bit.
* **Unchanged:** `reserve()` still wins over the shrink. A hybrid builder still never shrinks on its
  own, because `ggml_backend_sched` has no shrink. Only `release_compute_buffer()` frees its buffers.
