---
type: retro
date: 2026-09-10
domain: exporter
tags: [model-coverage, family-6, scoping, attention]
---

# Retro-040: The Blocker Was Scoped From the Mechanism, Not From the Seam

## The Issue

Family 6 sat on the hub for a day as "blocked on ONE engine primitive". The entry was careful,
specific and well-researched: it named T5's learned relative attention bias, described
`_relative_position_bucket` accurately, warned that `src/core/relative_position.cpp` is a different
mechanism with a colliding name, and listed three ways out — a new primitive, a Lua computation, or a
per-length constant fold.

None of the three was needed. The bias is not a missing primitive at all: by the time it reaches
attention, `transformers` has already summed it with the causal mask into ONE `[1, n_head, q, k]`
additive tensor, and `ggml_soft_max_ext` has always accepted a mask with a head axis. T5's bias is a
**mask this engine could already take**. What was left was one small exporter change (the `n_kv`
retyping carried the head axis instead of flattening it) and a Lua function.

The whole family — three phases, driver, tests — took an afternoon, against an entry that read like a
week of engine work.

## Root Cause Analysis

The scoping read the MECHANISM and stopped: *how is this number computed?* Bucketing, a table, a
gather, per length. Every one of those observations was correct, and together they describe something
the engine cannot do.

What it did not read is the SEAM: *what shape does this arrive in, at the one node that consumes it?*
That question has a different answer — `add(scores, one 4-D tensor)` — and the engine's side of that
seam was already general enough, in a way visible only by reading `ggml_soft_max_impl`'s asserts:

```c
GGML_ASSERT(mask->ne[0] == a->ne[0]);
GGML_ASSERT(mask->ne[1] >= a->ne[1]);
GGML_ASSERT(a->ne[2]%mask->ne[2] == 0);   // <- the head axis, already allowed
```

Nobody had used that third line, because every mask this tree had ever built was one head deep. An
unused degree of freedom in a dependency is invisible to a scoping pass that reads the model and the
model's own code, which is where the whole day's reading went.

There is a general shape here, and it is the same one
[Retro-024](retro-024-a-blocker-read-from-one-half-of-an-agreement.md) names from the other direction. A blocker
between two components is a claim about an AGREEMENT, and an agreement has two halves. This one was
scoped entirely from the producing half.

## What Changed

* [ADR-028](../adrs/adr-028-the-relative-attention-bias-is-a-mask.md) records the decision and, more
  usefully, the two options it rejected with the reasons — so a longer-context leaf in this family has
  the alternative written down rather than having to re-derive it.
* `_retype_fused_mask_input` now widens the KEY axis and carries everything above it, instead of
  rewriting the whole shape as `[n_kv, root_axis]`. The old spelling was correct for every 2-D mask
  and silently declared a 3-D one as 2-D.

## Prevention

**When scoping a capability gap, write down the interface the thing crosses, and read the OTHER side's
constraints before costing the work.** Not the other side's code in general — the specific asserts,
shapes and dtypes at the one seam. Here it was five lines of ggml.

The tell that this was not done: the hub entry's three options were all about how to PRODUCE the
tensor, and none of them said what the consumer would accept. An option list with only producers in it
has not looked at the seam.
