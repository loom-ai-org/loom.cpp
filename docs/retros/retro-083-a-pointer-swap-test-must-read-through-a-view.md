---
type: retro
date: 2026-10-11
domain: inference-engine
tags: [ggml-views, zero-copy, retained-outputs, test-design, whisper, t5]
---

# Retro-083: A Pointer-Swap Test Must Read Through a View

## The Issue

[ADR-073](../adrs/adr-073-a-referenced-input-is-read-in-place.md) made a consumer read a retained output
in place. It does this by pointing the declared input's `data` at the source for one call. The first
version passed its new hermetic test: equivalence, the source untouched, the restore, and the refusal,
each sabotaged red. It passed the existing retained-output equivalence tests too, and the whole CI
suite. Then on real models, whisper-small transcribed jfk as `[ [ [` and flan-t5-small generated
`<unk> <unk> …`. The input-copy time had dropped from 501 ms to 0.03 ms, which looked like success.

## Root Cause Analysis

ggml fixes a view's `data` once, when the graph is allocated: `ggml_backend_view_init` sets it to the
base's `data` plus `view_offs`, and nothing reads the base pointer again. Whisper's decoder reshapes
each K/V input into heads, so every op that used the input did so through a view. Those views still
pointed at the input's own storage, which the alias had stopped filling. The toy consumer in every test
was `MUL_MAT(output.weight, hidden)`. It read the input tensor itself, so it was the one shape that
cannot show the defect.

The mechanism was understood and the tests were sabotaged. But each sabotage checked a property of the
swap (was it undone, did it happen, was the stale case refused), never whether the graph *read* what
the swap pointed at. That was the property that was false.

## Resolution & Lesson Learned

`GraphBuilder::repoint_views` moves every node with `view_src == input` along with the input, and the
restore moves them back. ggml flattens a view of a view to its ultimate base, so one scan of the nodes
per built graph finds them all. `test_input_alias.cpp` case 5 reads the input only through two
`RESHAPE`s. Sabotage of each half of the fix turns it red.

**Takeaway: when a change moves where a tensor's bytes live, test it through a view of that tensor,
not only the tensor.** In ggml a view is a second pointer to the same bytes, taken once. Any change
that repoints `data` after allocation (an alias, a buffer swap, a cache moved between buffers) has to
move every view too, and a test whose consumer reads the base directly cannot tell. The same check
belongs on anything that rewrites `tensor->data` on a live graph.

The second lesson repeats [the ASR-oracle rule](retro-006-kokoro-shipped-noise.md): a timing that
improves is not evidence of a correct answer. Run the real model and read its output before reading
its clock.
