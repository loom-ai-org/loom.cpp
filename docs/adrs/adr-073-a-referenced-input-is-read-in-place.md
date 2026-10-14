---
type: adr
status: accepted
date: 2026-10-11
tags: [retained-outputs, inputs, zero-copy, ggml-views, whisper, t5, cross-attention, lua-bridge]
supersedes: []
---

# ADR-073: A Referenced Input Is Read in Place

Serves [Epic-01](../epics/epic-01-inference-engine-core.md) §2 (retained outputs). It extends
[ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md), which made a
driver edge a reference rather than a Lua value. This ADR makes the reference cost nothing. It came out
of the "direct I/O buffers" sub-item of the LiteRT thread on the hub
([Engine — performance](../backlog/active-index.md#engine--performance)).

## Context

The hub item assumed the cost of feeding inputs was the Lua boundary: masks and position biases built
as Lua tables of doubles, then converted and copied. Per-input timers on four models (Ryzen 3 3250U,
2026-10-11) found that cost to be small everywhere but T5:

| model | input time per call | where |
|---|---|---|
| Qwen3-0.6B, 418-token prefill | 2.4 ms of 10.4 s | the causal mask |
| VITS (en-GB miro) | 1.2 ms of 430 ms | tokens and noise |
| flan-t5-small, 410 tokens | 23 ms of 913 ms | the encoder's `position_bias` table |
| **whisper-small, jfk, 27 decode steps** | **501 ms of 8.86 s** | **retained cross-attention K/V, copied** |

The largest cost never crosses into Lua. A driver passes `{from = 'cross_kv', index = k}`, and
`set_tensor_from_output_ref` copied the retained tensor into the decoder's declared input on every
step: 24 tensors of 1.15 M floats, about 110 MB per step. flan-t5's decoder does the same with its
`xk_*`/`xv_*`, about 45 ms of its call.

## Options

1. **Fill Lua-built inputs in place** (the hub's original idea: a binding that writes a mask straight
   into the declared tensor). This is worth 2.4 ms on the LM. On T5 it would mean moving
   `t5_position_bias`'s bucket arithmetic from the driver into C++, against
   [ADR-003](adr-003-per-model-complexity-in-the-exporter.md). Not now.
2. **Bind the source tensor into the consumer's graph at build time.** No copy and no repointing, but
   the graph would then depend on which module feeds an input. That is a call-time fact the build key
   does not carry, and it would break graph reuse across producers.
3. **Point the declared input at the source's memory for one call, and point it back at the next
   `build()`.** The graph is unchanged, and so are the build key and every validation the copy did. Only
   the bytes stop moving.

## Decision

Option 3 (`GraphBuilder::alias_input`, called from `set_tensor_from_output_ref` before its copy):

* **Refused, not attempted, whenever an alias could differ from the copy.** That covers a different
  type, either side non-contiguous, a source too short, a different buffer type, or a hybrid builder.
  The caller then copies exactly as before. Hybrid is excluded because the scheduler plans its split
  copies from each tensor's `buffer` at allocation time, and on a device the copy is device to device
  anyway. A row range (`row`/`rows`) aliases at its byte offset.
* **Every view of the input moves with it.** ggml fixes a view's `data` at allocation, as its base's
  `data` plus `view_offs`, and never looks at the base again. The first version moved only the input.
  Whisper then read its K/V heads, through a reshape, out of the input's unfilled storage and
  transcribed jfk as `[ [ [`, and flan-t5 emitted `<unk>`. The toy test passed because its consumer used
  the input directly. `ggml_new_tensor_impl` flattens a view of a view to the ultimate base, so one scan
  of the graph's nodes for `view_src == input` finds them all. The scan runs once per built graph.
* **The alias ends at the top of `build()`.** Every path that writes a declared input calls `build()`
  first: the Lua input table, the recurrent sweeps, and the ODE stepper's carried state. So nothing can
  write *through* an alias into the module that owns the source. With the restore removed, the test's
  source module had its retained output overwritten.
* **An input fed by alias is not kept.** Its own storage never received the value. A copy used to leave
  the last value there, so a reused graph could be run without re-supplying it. That case is now
  refused by name ("passed by reference on an earlier call and not passed on this one"), not computed on
  stale data. The gate suite says no driver in the zoo relies on it.

## Consequences

* **Positive:** whisper's input time fell from 501 ms to 13.5 ms per jfk transcription, and flan-t5's
  from 69 ms to 29 ms. Both outputs are identical to before (the transcript and the summary). Paired
  end to end over 6 alternating rounds, whisper comes out **1.038x** mean (median 1.041). That is the
  same size as the 5% the copies were, but the rounds span 0.93-1.12, so on this two-core box it does
  not resolve.
* **Positive:** the cost of passing a retained output by reference no longer grows with its size. That
  matters most for the encoder-decoder families: Whisper, T5, Canary, and every cross-attention cache
  computed once and read per step.
* **Negative:** a declared input's `data` is no longer fixed for the life of a graph. Any future code
  that caches an input's data pointer across a `build()` is wrong. `tests/ci/test_input_alias.cpp`
  covers the five ways this was found to fail: equivalence, the source untouched, the restore, the
  refusal, and views. Five sabotages each turned it red.
* **Not done:** device builds still copy. The Lua-side inputs (T5's position bias) still cross as
  doubles, about 2.5% of a flan-t5 call. That is the one case where option 1 would pay, and it costs a
  per-family binding.
