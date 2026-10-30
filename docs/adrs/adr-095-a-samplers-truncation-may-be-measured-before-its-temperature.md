---
type: adr
status: accepted
date: 2026-10-30
tags: [engine, sampling, lua-bridge, family-10, audio8-tts]
supersedes: []
---

# ADR-095: A Sampler's Truncation May Be Measured Before Its Temperature

## Context

Audio8-TTS (Edge0's Fish-S2-style DualAR, family 10) draws every semantic token and every codebook id
through `ArkttsLegacyTopKTopPLogitsProcessor`, then divides by the temperature:

```
sorted, idx = sort(scores, descending)
cum         = cumsum(softmax(sorted))                       # the RAW row, T = 1
remove      = (cum > top_p) | (position >= top_k);  remove[0] = False
scores      = scores.masked_fill(remove, -inf) / temperature
```

Two things in that are not what `loom.sample_row` did, even with ADR-047's `top_p_mass = "row"`:

1. **The truncation is measured on the UNTEMPERED row.** `transformers` divides by the temperature
   first and truncates the tempered distribution. Top-k picks the same ids either way; top-p's mass
   does not, so at the checkpoint's `temperature = 0.7` the two keep different prefixes.
2. **The id that crosses `top_p` is dropped.** `cum > top_p` removes the id whose inclusive mass
   exceeds the threshold; `transformers` (and the engine) keep the shortest prefix that REACHES it,
   crossing id included.

Greedy decoding is unaffected, and greedy is how the export is gated (810/810 ids against the
reference). But the checkpoint samples by default, so a file that drew with the engine's rule would run
a different distribution from the same three numbers, and nothing would report it.

## Options

**1. Accept the `transformers` rule.** Rejected: a different sampler with the reference's settings, the
hazard ADR-047 already named.

**2. Sample in Lua over the marshalled row.** Rejected for ADR-047's reason: 4097 floats per semantic
step and 4096 per codebook, nine times a frame, across the boundary
([ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md)).

**3. A `fish` mode on `sample_row`.** Rejected: one model family's name for two independent facts. A
future sampler may truncate untempered and keep the crossing id, or the reverse.

**4. Two options on `loom.sample_row`, each one fact.**

## Decision

Option 4, strings for `top_p_mass`'s reason (an unknown value is refused, never read as the default):

* `truncation = "untempered"`: top-k, min-p and top-p are measured on the row at temperature 1, and the
  survivors are re-weighted at the real temperature for the draw. The default, `"tempered"`, is the old
  behaviour.
* `top_p_crossing = "exclude"`: the prefix keeps the ids whose inclusive cumulative mass is at most
  `top_p` (at least the best one). The default, `"include"`, is the old behaviour.

Audio8's driver passes both with `top_p_mass = "row"`. Its repetition-aware redraw (a semantic draw
that repeats one of the last ten is redrawn at `ras_temperature`/`ras_top_p`) is driver Lua, as
CosyVoice3's is.

## Consequences

* **rc19 and earlier ignore both keys** and run Audio8 with the `transformers` truncation. It still
  speaks; it runs a different sampler. The card says it needs rc20.
* The defaults are unchanged, so every shipped driver keeps its sampler (`tests/ci/test_sample_row.cpp`
  §1-13 unchanged; §14 new, and a build with either option disabled fails it).
* The reference draws with an exponential race (`argmax(p / -log(u))`), not an inverse-CDF walk, so its
  draws cannot be replayed through `uniform`. A sampled Audio8 decode is graded by ear and by ASR; the
  graphs are graded greedy.

## Related

* [ADR-047](adr-047-a-samplers-mass-its-bans-and-its-draw-are-the-callers-to-state.md) -- the
  previous sampler knobs, and the same rule for adding one.
* [ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md) -- why the row stays
  in the engine.
* [Epic-03](../epics/epic-03-model-coverage.md) -- Audio8-TTS.
