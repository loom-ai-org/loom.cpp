---
type: adr
status: accepted
date: 2026-10-11
tags: [exporter, t5, relative-position-bias, gather, lua-driver, family-6]
supersedes: [adr-028 (encoder only)]
---

# ADR-074: T5's Encoder Bias Is a Gather Over a Distance Table

Supersedes [ADR-028](adr-028-the-relative-attention-bias-is-a-mask.md) for the encoder; the decoder
still follows it. Serves [Epic-03](../epics/epic-03-model-coverage.md) (family 6, flan-t5-small) and the lean-runtime rule
of [ADR-003](adr-003-per-model-complexity-in-the-exporter.md). It is the T5 residue of
[ADR-073](adr-073-a-referenced-input-is-read-in-place.md)'s input measurement. The code is
`loom-exporter/loom_exporter/t5_export.py` (`relative_bias_by_distance`, `_T5EncoderWrapper`).

## Context

T5 has no positional embedding. Each stack adds a learned relative bias,
`table[bucket(key - query), head]`, to every attention score, where `bucket` is HF's
`_relative_position_bucket`: a log-spaced bucketing that saturates at `max_distance`. Tracing HF's
`compute_bias` bakes `torch.arange(query_length)` in at the dummy length. So the family shipped with
the bias built in the DRIVER (`t5_position_bias` in Lua) and handed to the encoder as an
`n_head x n_src x n_src` input.

Measured 2026-10-11 (Ryzen 3 3250U, flan-t5-small, 410-474-token source): marshalling that table cost
**23 ms of a 913 ms call** (1.35 M doubles), before counting the Lua bucketing itself. It was the one
Lua-built input in the zoo worth moving.

## Options

1. **Keep it in Lua.** This costs ~5% per call, and the bias stays host-side on every backend.
2. **A per-family engine binding** that fills the bias in place. This is C++ for one family, which
   ADR-003 rejects.
3. **Bucket in the graph:** `abs`, `log`, `floor`, `min` and a `select`, all of which the engine has.
   But the truncation sits exactly on an integer at distances 16, 32 and 64 (`log(rel/8)/log(16)*8` is
   2, 4 and 6 in real arithmetic). So whether a pair lands in the right bucket would depend on the
   engine's `log` agreeing with torch's to the last ulp there.
4. **Bucket at export time, gather in the graph.** Every distance at or beyond `max_distance` shares one
   bucket. So the export calls HF's own `_relative_position_bucket` on `[-max_distance, max_distance]`
   and stores `table[bucket(rel)]` per distance, a `[2 * max_distance + 1, n_head]` weight (257 x 6 on
   flan-t5-small). The graph forms `key - query`, clamps it, and gathers.

## Decision

Option 4, for the encoder.

* The encoder takes `positions` (`0..n_src-1`, from `loom.range`) instead of `position_bias`. It is an
  input rather than an arange for the same reason `cache_position` is: a traced arange bakes its
  length.
* `key - query` is ONE matmul with an inner dimension of 2. `pos[None, :] - pos[:, None]` broadcasts
  both operands, which the engine's elementwise ops refuse
  ([Retro-070](../retros/retro-070-a-one-token-decode-step-hid-a-two-sided-broadcast.md)). The first
  export aborted on exactly that, `SUB: a=[474,1] b=[1,474]`. The matmul is exact for integers below
  2^24.
* `relative_bias_by_distance` checks the saturation claim over eight times `max_distance` before it
  trusts it. A config for which clamping changed a bucket would raise at export, rather than read the
  wrong bias for long pairs.

**The decoder keeps the Lua builder.** Its bias is one row per step (6 144 values over a 32-step decode,
0.11 ms), and it is also the fused attention's causal mask, which the engine pads to the KV bucket
(P4.0.15). Moving it would cost a computed mask through that padding contract, for nothing measurable.

## Consequences

* **Positive:** the encoder output matches HF at f64 to **2.5e-7** (474 tokens) and **3.5e-7** (11
  tokens), against HF's own f32-vs-f64 spread of 3.5e-7 and 2.1e-7. Generation is identical to the
  rc17 GGUF and to HF greedy on both prompts. Paired over 10 rounds in one process, a 474-token
  summarisation is **1.057x faster** (median 1.070, p10 1.039, p90 1.101). The bias is computed on
  whichever backend runs the encoder.
* **Positive:** the exporter tests check the distance table against HF's bucketing over the full
  512 x 512 grid, in both directions, and the wrapper against HF's own encoder past the saturation
  distance. Sabotage of the orientation (`query - key`) and of the table offset each turns them red.
* **Negative:** flan-t5-small's GGUF changes (a new input, a new 257 x 6 weight, a shorter driver), so
  the published model must be re-exported, gated and republished. The engine needs nothing new.
* **Not done:** the decoder's bias, for the reason above.
