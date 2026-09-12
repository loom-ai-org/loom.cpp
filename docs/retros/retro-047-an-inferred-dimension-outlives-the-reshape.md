---
type: retro
date: 2026-09-12
domain: exporter
tags: [shape-algebra, tracing, mask, lua, family-11, qwen3-tts]
---

# Retro-047: An Inferred Dimension Outlives The Reshape That Wrote It

## The Issue

Qwen3-TTS's 12 Hz speech tokenizer — family 11's fourth leaf — traced, converted, exported, wrote a
456 MB GGUF, and **loaded cleanly**, reporting the right contract:

    codec: 16 stream(s) per frame, 12.5 frame(s) per second
    672 code(s) = 42 frame(s)

Then running it failed:

    loom.run_subgraph: node 'VIEW' (op=VIEW, inputs=[attention_mask_1,], outputs=[attention_mask_3]):
    VIEW: non-positive dimension in resolved shape [42,0,1,1], offset=0, parent ne=[42,42,1,1]

The parent is the right shape. The view of it has a zero.

## Root Cause Analysis

The decoder's pre-transformer needs an additive sliding-window causal mask, built **in-graph** from
`torch.arange` so the frame axis stays symbolic — the technique `qwen3_asr_export.WindowedAudioEncoder`
established, copied down to its outer-product-against-ones spelling. Its last line was copied too:

```python
return torch.where(visible, 0.0, neg).view(1, 1, -1, hidden.shape[1])
```

That `-1` reaches the emitted topology **as a literal `-1`**:

```json
{"op": "RESHAPE", "outputs": ["attention_mask_1"], "attrs": {"shape": ["n_codes", -1, "1", "1"]}}
```

which is fine on its own — the engine can infer one dimension. What is not fine is that the very next
op derives its own extent from that shape. `eager_attention_forward` slices the mask to the key
length:

```python
causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
```

and the walk that propagates shapes through the slice had to divide by the inferred dimension it never
resolved, producing

```json
{"op": "VIEW", "attrs": {"shape": ["n_codes", "floor(1/n_codes)", "1", "1"]}}
```

`floor(1/n_codes)` is **1 at `n_codes = 1` and 0 at every other length**. So the expression is not
obviously wrong — it is arithmetic over real symbols, it survives every check the export has, and it
is correct at exactly the one length nobody runs.

## What Fixed It

Build the mask at rank 4 with **no inferred dimension**: two `unsqueeze`s instead of a `view`, and
`where` branches that are TENSORS (`delta * 0.0`, `delta * 0.0 + neg`) rather than Python floats.
The tensor branches are load-bearing — with scalar ones the result's rank is never established and
coremltools' own `slice` handler dies first, with `IndexError: list assignment index out of range`.

## The Lesson

**An inferred dimension is a promise to the consumer that reads it, and a shape-derived slice is a
consumer.** [Retro-044] recorded the same class of failure from the other direction: there MIL minted
a fresh symbol and the walk substituted the root axis. Here the symbol is fine and the *hole* is the
problem. Both are the same underlying rule — a downstream op that computes an extent needs every
dimension of its input to be a real expression, and `-1` is not one.

**The generalisable check is that the failure is invisible until run time.** Trace, convert, write,
load and contract-report all succeeded; only a call failed. So "it exports" is not evidence about
shape algebra, and the first thing a new leaf should do is *run* — which for a family whose output is
audio means `loom_cli --out` or a driver harness, not a second look at the topology.

**Copying a spelling copies its assumptions.** `view(1, 1, -1, T)` is correct in `qwen3_asr_export`
because nothing downstream of that mask derives an extent from its shape. The line is not portable;
what is portable is the outer-product-against-ones construction, and that half transferred perfectly.

## A Second, Smaller One In The Same Component

The driver emitted `#codes // 16`. The engine embeds **LuaJIT, which is Lua 5.1**; `//` arrived in Lua
5.3. Nothing syntax-checks an emitted driver at export time, so this too exported, wrote, and loaded,
then died on first call with `unexpected symbol near '/'`. `driver_ir.BinOp` already had a `floordiv`
op that renders as `math.floor(a / b)`; it now **refuses `"//"` outright** with a message naming the
alternative, because the next person to reach for it will reach for the operator they know.

Same shape as the lesson above: an emitted artifact is not checked by the thing that emits it.
