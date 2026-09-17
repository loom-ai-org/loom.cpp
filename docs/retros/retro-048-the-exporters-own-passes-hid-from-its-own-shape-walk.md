---
type: retro
date: 2026-09-15
domain: exporter
tags: [tracing, coremltools, shapes, family-5, verification]
---

# Retro-048: The Exporter's Own Passes Introduced Ops Its Own Shape Walk Did Not Know

## The Issue

Family 5's first leaf (SenseVoiceSmall) took **four consecutive exports** to produce an artifact whose
declared lengths were right, and every failure was the same failure:
`_infer_dynamic_dim_expr` stopped at a producer it did not recognise, fell back to the root axis, and
the topology declared a tensor with one row per audio SAMPLE where it should have had one per encoder
frame — 176,000 against 187.

[Retro-044](retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md) is this lesson
already, written three days earlier for EnCodec, and its own takeaway predicted the recurrence: *"every
new op the walk does not know is a new way to hit it"*. This retro exists for the part 044 did not
say, because 044's framing turned out to be too narrow.

## What Was Actually Different

044 described the gap as arriving with a new MODEL — `pad` and `elu` entered the walk's blind spot
because EnCodec was the first checkpoint to put them upstream of a shape query. That reading says the
risk grows with the zoo.

**Two of family 5's four gaps were the exporter's OWN dialect ops**, and they had been reachable since
the day the passes that emit them were written:

* **`loom_scale`**, which `lower_reduce_mean` puts into *every* graph containing a `reduce_mean`.
* **`loom_broadcast_to`**, which `insert_explicit_broadcasts` puts in front of *every* mutually
  broadcasting `add`/`mul` — and whose own topology rule then reads the walk's answer back out to build
  the `REPEAT` target shape, so a wrong answer here is emitted directly into the artifact.

Neither arrived with a model. Both were introduced by a rewrite the exporter performs on itself, and
neither was added to the walk when its pass was. The blind spot was not growing with the zoo; it was
already there, in the part of the pipeline nobody thinks of as "a new op".

The other two were narrower:

* **`reduce_sum` had a case that only handled `keep_dims=False`.** Every prior user summed a complex
  pair away, so a kept dimension fell through the bottom of a case that looked complete. A missing op
  can be found by grepping the op set; a case that silently declines half its inputs cannot.
* **`fill` whose shape is `shape(x)`** — what `torch.ones_like` lowers to. `facts.reshape_shape`
  resolved a constant array and a `concat` of per-axis gathers, but not the whole-shape form, and
  `_op_fill`'s own comment *already recorded* the resulting bug against Conformer-CTC's `[T, T]` mask.

## The Fix

Four entries, none of them structural: `loom_scale` into the unary passthrough set, `loom_broadcast_to`
into the elementwise-broadcast case (its per-axis rule is that case's, with `like` in `y`'s place),
`reduce_sum`'s `keep_dims=True` branch, and a `shape(x)` case in `reshape_shape`. Afterwards the
topology carries **one** bare `n_samples` — the waveform's own reshape, where it is correct — and 213
nodes carrying `floor(floor((n_samples - 400)/160)/6) + 5`, which is the encoder's real row count.

## What Found Them

The first three were found by exporting a 939 MB artifact and reading its topology back: **five minutes
per attempt, one gap per attempt.** The fourth attempt was different — a 30-line module with the same
*shape* as the real graph (a strided conv for the dynamic axis, then the same position derivation), run
through `apply_loom_mil_passes` and `LoomGGUFExporter.generate_graph_topology`, with
`_infer_dynamic_dim_expr_uncached` monkeypatched to print every producer it visited and what it
returned. That printed the whole chain in **thirty seconds** and named the exact link:

    walk loom_scale   axis=1 -> n_samples      <- stops here
    walk transpose    axis=2 -> n_samples
    walk cumsum       axis=2 -> n_samples
    ...

## Takeaways

* **A lowering pass has two obligations, and only one of them fails loudly.** Emitting a new dialect op
  without adding it to the shape walk produces no error anywhere — the op map's absence raises, the
  walk's absence does not. Any pass that introduces an op type owes it an entry in
  `_infer_dynamic_dim_expr_uncached`, and the review question for a new pass is "what does the walk do
  when it meets this?"
* **A partial case reads as a complete one.** `reduce_sum`'s handler had existed for months and was
  wrong for half its inputs. Grep finds a missing op type; nothing finds a guard that narrows a case to
  the one call site that motivated it. Prefer handling the whole op or raising on the part you did not.
* **Instrument the walk, do not re-export.** A faithful minimal repro plus five lines of monkeypatched
  tracing is two orders of magnitude faster than reading a 939 MB artifact's topology back, and it
  names the link rather than the symptom. Build it on the SECOND failure, not the fourth.
* **A comment recording a bug is not a fix.** `_op_fill`'s docstring described the Conformer-CTC
  failure this session re-hit, in detail, and the code below it still fell back. A known-wrong path
  with an explanation is still a known-wrong path.
