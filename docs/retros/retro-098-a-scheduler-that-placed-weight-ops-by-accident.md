---
type: retro
date: 2026-10-28
domain: backends
tags: [ggml-backend-sched, weights, buffer-usage, device-selection, vc4, hybrid]
---

# Retro-098: A Scheduler That Placed Weight Ops by Accident

Serves [Epic-04](../epics/epic-04-backends-and-accelerators.md) (P4.45); the decision is
[ADR-091](../adrs/adr-091-the-qpus-take-convolutions-and-per-channel-int8.md) (decision 5).

## Issue

With the QPU convolutions in, VITS's and amy's vocoder convolutions ran on the QPUs, but Moonshine
tiny's encoder convolution -- the same op, the same kind of weight in the same buffer, accepted by
`supports_op` -- ran on the CPU, and the VAD's never even reached the QPU backend's `supports_op`.
Marking the weights fixed Moonshine and broke amy, de051, wake and heart: every F32 convolution went
back to the CPU.

## Root cause

Two, one under the other.

* loom never marked its weight buffers `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`. ggml's scheduler runs an
  op on the backend holding its weights through exactly that mark (pass 1, `1.wgt`), which is how
  llama.cpp places every matmul. Unmarked, an op reading a weight was placed by the expansion passes,
  where the CPU -- which supports everything -- spreads to whatever it reaches first; which ops the
  accelerator got depended on what their neighbours were.
* Marked, the rule still missed every weight read through a VIEW: pass 1 checks `src->buffer`, and a
  view has no buffer until the allocator initialises it, after scheduling. loom reshapes F32 kernels,
  so those convolutions read a weight the scheduler could not see, while their bias ADDs -- direct
  weight readers -- were now pinned to the CPU, and the CPU's expansion swallowed the convolutions
  between them. `ggml-0056` reads the view's root (a view of a weight is a weight).

## Takeaway

**When an accelerator gets some instances of an op and not others, look at how the op was placed, not
at the op.** `supports_op` saying yes is necessary, not sufficient: a scheduler's answer depends on
inputs (buffer usage, view initialisation, neighbours) a backend never sees. The check is cheap --
log every op type the backend is asked about, and every node it computes -- and "never asked" was the
line that pointed here.

## Record

* Run 3 (weights marked, no `ggml-0056`): amy-f32 0 CONV_2Ds (44 before), de051-f32 0 (42), wake-f32 0
  (10), heart-f32 0 (26); conformer-f32 145 F32 MUL_MATs on the QPUs (0 before), mt-q8pc 2 CONV_2Ds.
* Run 4 (both): amy-f32 44, de051-f32 42, wake-f32 10, ct256-f32 129, conformer 145 + 33, vad 4.
* `GGML_VC4_DEBUG=1` on vad-f32 before: `asked about` listed no CONV_2D.
