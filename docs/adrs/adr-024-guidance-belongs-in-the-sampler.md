---
type: adr
status: accepted
date: 2026-09-03
domain: inference-engine
tags: [engine, sampling, guidance, lua-bridge, family-10]
---

# ADR-024: Restricted Sampling and Guidance Are Options, Not New Bindings

## Context

Family 10's driver decoded greedily and without classifier-free guidance, which is not what its
checkpoint asks for: Dia declares `do_sample: true, temperature 1.8, top_k 50, top_p 0.9` and
`guidance_scale 3.0`. Shipping the greedy decode as the model would be
[Retro-006](../retros/retro-006-kokoro-shipped-noise.md) again — a file that matches a reference and
is not the model anyone published.

Two things were missing from the engine, and neither is expressible from Lua:

* **A window.** `loom.sample_row` sampled the WHOLE row. Dia's four highest ids are control tokens and
  `DiaEOSChannelFilterLogitsProcessor` bans them **per channel**, so an unrestricted draw on channel 8
  can emit PAD or BOS. Under an argmax this never happened, because `argmax_row_range` already had a
  window — a sampler without one is not a smaller version of the same thing.
* **Guidance.** It combines the logits of two runs, and logits are precisely what never crosses the
  Lua boundary. Doing it in the driver means marshalling two 9252-float rows per step, reinstating the
  boundary cost every retained reduction here exists to avoid ([Retro-004](../retros/)).

The handover that scoped this predicted a new `sample_row_range` binding, "symmetric with the argmax
pair".

## Decision

**Both are entries in `loom.sample_row`'s existing options table**: `lo`/`hi`, and
`guidance = {module =, scale =, top_k =}`. No new binding.

The argmax pair is two bindings for a **mechanical** reason its own comment states: `argmax_row`'s
module form ends in an optional trailing `generation`, so `(module, row, lo, hi)` and
`(module, row, generation)` cannot be told apart by arity or type. `sample_row` takes a table
precisely so that "the knobs are a set that grows" — its own words — and a table has no such
ambiguity. A second binding would be a second door onto one reduction, which is the thing
`sample_tensor_row` exists to prevent one level down.

**Guidance is in the sampler rather than beside it** for the same reason: turning logits into a token
happens in one place here, deliberately. Two ways to do it that can disagree is a failure this project
keeps removing (P4.0.14).

**The general form is the engine's, the model's centring is the driver's.** `loom.sample_row`
computes `uncond + scale * (cond - uncond)`, which is `ClassifierFreeGuidanceLogitsProcessor`'s.
Dia's own processor centres on the conditional logits, `cond + g * (cond - uncond)` — the same family
one apart, so its driver passes `g + 1`. The GGUF declares the checkpoint's `g` unconverted, so
`model.hparam("sampling.guidance_scale")` and `generation_config.json` agree.

## Consequences

* **The greedy invariant got stronger, not weaker.** `temperature <= 0` and `top_k == 1` still mean
  "the highest-scoring id", but under guidance those logits are in no tensor, so `sample_tensor_row`
  can no longer delegate to `argmax_tensor_row`. Both now run `argmax_of_window` over a window read by
  one shared function — the same invariant, one level lower and harder to break.
* **`guidance.top_k` is a second operation, not a parameter of the first**, and it is why this ADR has
  a retro beside it. It selects a shortlist with the **guided** logits and then draws from the
  **conditional** ones restricted to it, discarding the guided values —
  `DiaClassifierFreeGuidanceLogitsProcessor`'s `guidance_top_k`. It is applied over the whole row and
  the `lo`/`hi` window after it, which is the order `transformers` composes them in and is not
  interchangeable: a control token inside the guided top-k occupies one of the *k* slots and is then
  banned, leaving *k-1* real candidates.
* **`transformers`' warper ORDER is not reproduced, and that is stated rather than hidden.** This
  engine applies temperature → top-k → top-p; Dia's merged processor list reaches top-k and top-p
  before its temperature warper, and top-p's cutoff depends on the temperature it is computed at. It
  does not affect any oracle here, because **the only exact oracle is a greedy one** — two samplers
  running one algorithm from different RNG streams agree on nothing — and under greedy all three
  collapse to the same argmax. What the gate does compare exactly is greedy **with guidance on**,
  which is fully deterministic and exercises the guided path, its shortlist and the channel filter.
* **The two live clauses of `DiaEOSChannelFilterLogitsProcessor` cost no primitive.** "Force EOS when
  it is already the top logit, suppress it when it is not" — both no-ops under an argmax — become:
  ask which of `[0, EOS]` wins at temperature 0, take EOS if it does, otherwise draw from `[0, EOS)`.
  That is the two clauses exactly, without the engine needing to express `-inf`.

## See also

* [ADR-023](adr-023-a-second-stream-is-declared-not-derived.md) — the other half: two decode streams
* [Retro-031](../retros/retro-031-dias-guidance-is-not-the-standard-formula.md) — what reading the
  generic processor instead of the model's own would have cost
* [ADR-003](adr-003-per-model-complexity-in-the-exporter.md) — why these are per-*task* and belong here
