---
type: retro
date: 2026-09-03
domain: exporter
tags: [tracing, dynamic-shapes, coremltools, family-10]
---

# Retro-030: A Guard That Could Not Fire, Written to Prove a Dim Was Static

## The Issue

Family 10's handover carried a worked, "verified" fix for the one thing stopping Dia from converting.
`modeling_dia.rotate_half` slices at `x.shape[-1] // 2`, which traces to 48 ×
`aten::Int(aten::floor_divide(...))` in the encoder alone; coremltools' `_int` handler does
`int(x.val)` on it and dies with `TypeError: only 0-dimensional arrays can be converted to Python
scalars`. The recorded fix read:

```python
def rotate_half_static(x):
    half = x.shape[-1] // 2                 # static per module
    if not isinstance(half, int):
        raise TypeError("last dim is not static; the per-module midpoint assumption does not hold")
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
```

It raises. Every time, on every model, the moment it is traced — which is the only context it exists
for. The first attempt to export the encoder through it died on its own guard.

## Root Cause Analysis

**Under `torch.jit.trace`, `tensor.shape[i]` returns a 0-d `Tensor` for *every* dimension — the
static ones included.** Confirmed directly, on a `(1, 5, 128)` input inside a traced module: `shape[1]`
comes back as `tensor(5)` and `shape[-1]` as `tensor(128)`, both `torch.Tensor`, and only outside
tracing do they come back as `int`.

So `isinstance(half, int)` does not distinguish "this dim is static" from "this dim is dynamic". It
distinguishes *tracing* from *not tracing*. The guard was written to protect against a real hazard —
a `//` on a genuinely symbolic dim is how the original bug arises — and the property it tested was not
the property it meant.

The verification recorded in the handover was consistent with all of this and still said nothing about
it: `max|patched - original| = 0` at lengths 7, 32 and 128 was measured **eagerly**, by running the
encoder twice and diffing hidden states. Eagerly, `shape[-1]` *is* an `int`, the guard passes, and the
arithmetic is correct. The one thing the check never did was trace.

## Resolution & Lesson Learned

`torch.chunk` asks for a **count** rather than an index, so it needs no arithmetic over the last
dimension at all:

```python
def rotate_half_chunk(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
```

Bit-identical eagerly at lengths 7, 32 and 128; traces to zero `floor_divide` and zero `aten::Int`;
converts with a fully symbolic axis (`tokens (1, is0)` in, `(1, is0, 1024)` out). It sidesteps the
question the guard was trying to answer instead of answering it.

The one property it does need is an **even** last dim — `chunk(2)` on an odd dim splits ceil/floor and
rotates by the wrong amount **silently**. That is checked at export time against all three of the
checkpoint's head dims, not the one they happen to share.

* **Actionable takeaway 1 — verify a tracing fix by tracing it.** An eager check of a patch whose whole
  purpose is to change what the tracer emits grades the arithmetic and not the patch. The cheapest
  honest check is the op histogram of `traced.inlined_graph`: the fix was for 48 `aten::Int` nodes, so
  count them.
* **Actionable takeaway 2 — under trace, a shape read is a Tensor, so `isinstance(dim, int)` is a
  tracing detector, not a staticness detector.** There is no cheap in-graph way to assert a dim is
  static; assert it against the *config* at export time, where the number is a real Python `int`, and
  keep the graph free of the question.
* **Actionable takeaway 3 — prefer the op that needs no arithmetic over the axis.** `chunk`/`split`
  take a count; slicing takes a bound. Wherever both express the same thing, the one that never
  computes an index is the one that survives a dynamic axis.

A fourth thing this cost nothing to learn and is worth writing down: the decoder's own 36 `aten::Int`
nodes — from `hidden_states.shape[:-1]` reaching `.view()` in every self-attention block — **were
already handled**, by the `_cast` fold `torch_patches.apply_torch_frontend_patches` installs at
`import loom_exporter` time. A standalone `ct.convert` probe that does not import the exporter is not
reproducing the exporter's conversion, and will report blockers that do not exist.

## See Also

* [ADR-021](../adrs/adr-021-dias-decoder-resolves-two-dynamic-axes.md) — the other tracing finding from
  this family
* [Retro-024](retro-024-a-blocker-read-from-one-half-of-an-agreement.md) — the sibling failure: a
  blocker predicted from reading one side of an interface
* [Retro-005](retro-005-supertonic-fixed-text-length.md) — a traced constant is a contract
