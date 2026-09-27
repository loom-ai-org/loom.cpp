---
type: retro
date: 2026-09-18
domain: exporter
tags: [tracing, coremltools, shapes, family-9, f5-tts, verification]
---

# Retro-051: A Negative Slice Begin Became Twice the Rows at a Negative Offset

## The Issue

F5-TTS's estimator — 22 DiT blocks, 2,178 nodes — exported clean, wrote clean, loaded clean, and died
on its first evaluation:

```
loom.run_ode_and_retain: node 'MUL' (op=MUL, inputs=[t_5,_280], outputs=[_281]):
  MUL: incompatible shapes a=[64,861,16,1] b=[64,1722,1,1]
```

`a` is the query, `(1, 16, 861, 64)` in torch order. `b` is the rotary table, and it had **1,722 rows
for an 861-frame sequence** — exactly twice as many.

## Root Cause

x_transformers' `apply_rotary_pos_emb` opens with a defensive trim:

```python
freqs = freqs[:, -seq_len:, :]
```

For a table longer than the sequence, that takes the last `seq_len` rows. This export hands it a table
already cut to exactly `seq_len`, so the trim is the identity — and tracing it emitted

```
VIEW shape=['64', '2*n_tokens', '1'] offset='-256*n_tokens'
```

A **negative begin over a dynamic axis** is the shape the derivation walk renders arithmetically
instead of normalising against the source length. `begin = -n` stayed `-n` rather than becoming
`len - n = 0`, and the size came out as `end - begin = n - (-n) = 2n`. At the traced length the two
readings coincide numerically (begin 0, size 64 either way), which is why nothing upstream noticed.

It appeared **44 times**: once per query and per key in each of the 22 blocks. Every other topology in
the file — the mel front end, the text embedding, the vocoder — had zero nodes with a negative literal
in their attributes, so the defect was one op reached by one library function.

## What Found It

Not the export, which was green. Not the load, which was green. Running it — and then a scan of the
emitted topology JSON for negative literals in node attributes, which turned "something is wrong in
2,178 nodes" into "these 44 nodes, all the same op, all from one call site" in one command.

The instrumented-walk recipe ([Retro-048](retro-048-the-exporters-own-passes-hid-from-its-own-shape-walk.md))
was tried first and answered nothing here, because the walk did not *fail* — it produced an expression,
and the expression was wrong. Reading the artifact was what worked.

## Takeaways

* **A dynamic slice with a negative begin is a distinct failure mode from a dynamic slice the walk
  cannot derive.** [Retro-044](retro-044-mil-retires-the-algebra-and-the-walk-substitutes-the-root.md)
  and 048 are both about the walk *stopping*; this one is about it *continuing* with an expression that
  is right at the traced length and wrong everywhere else — the same shape as
  [Retro-047](retro-047-an-inferred-dimension-outlives-the-reshape.md)'s `-1`.
* **Grep the emitted topology for negative literals.** It is one command, it covers every model, and a
  negative offset into a tensor is never something a correct graph asks for.
* **A defensive trim in library code is a slice like any other.** The line exists to be a no-op for
  every caller that has already done the work; that is exactly what makes it invisible while reading
  the model, and it still reaches the tracer.
* **The fix was to remove the op, not to teach the walk.** The trim is provably the identity for a
  single unpadded sequence, which is this project's convention everywhere, so
  `f5_tts_export._apply_rope_precomputed` substitutes a version without it — and takes 44 redundant
  `cos`/`sin` pairs out of the graph on the way past, over a table that changes neither across blocks
  nor across sampling steps. Teaching the walk to normalise a negative begin is still worth doing and
  is [filed in the hub](../backlog/active-index.md#exporter--mil-compiler); nothing else in the zoo
  emits one today.
