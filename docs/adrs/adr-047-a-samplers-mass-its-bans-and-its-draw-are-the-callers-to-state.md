---
type: adr
status: accepted
date: 2026-09-24
tags: [engine, sampling, lua-bridge, oracle, family-9, cosyvoice3]
supersedes: []
---

# ADR-047: A Sampler's Mass, Its Bans and Its Draw Are the Caller's to State

## Context

CosyVoice3's LM samples every speech token with `ras_sampling` (repetition-aware sampling):

```
top  = nucleus_sampling(scores, top_p=0.8, top_k=25)      # over the WHOLE softmax, sorted
if count(top in last 10 drawn) >= 10 * 0.1:              # i.e. any repeat
    scores[top] = -inf
    top = multinomial(softmax(scores))                   # the full distribution, top banned
```

and before `min_len` tokens `sampling_ids` sets `scores[6561] = -inf`. Three things in that are not
`transformers`' sampler, which is what `loom.sample_row` implemented:

1. **The nucleus's denominator.** `nucleus_sampling` keeps ids while the running sum of the FULL
   softmax is below `top_p` and the count below `top_k`. `transformers` runs top-k first and measures
   top-p against the survivors' renormalised mass. Same two numbers, different prefix: whenever top-k
   has cut mass off, the reference keeps more ids.
2. **Banned ids**, set to `-inf` before anything reads the row.
3. **The draw.** The checkpoint samples, and greedy is not a usable oracle: an argmax decode stops after
   five tokens and Whisper hears "you". Two RNGs agree on nothing, so a sampled decode could only be
   graded by ear.

## Options

**1. A `ras` primitive in the engine.** Rejected: it is one model's sampling policy in C++
([ADR-003](adr-003-per-model-complexity-in-the-exporter.md)). The window, the threshold and the
redraw are the reference's choices, and the next checkpoint's policy would be another primitive.

**2. Sample in Lua over the marshalled row.** Rejected on the boundary: 6761 floats per step cross into
Lua, which is the cost every retained reduction exists to avoid ([Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md)).

**3. `transformers`' top-p, and accept the difference.** Measured and rejected: with the default
sentence's pinned draws the decode diverges from the reference at token 4 of 76.

**4. Three options on `loom.sample_row`, and the RAS loop in the driver.**

## Decision

Option 4:

* `top_p_mass = "row"`: top-p's cumulative mass is a fraction of the whole window's softmax. The
  default, `"candidates"`, is the old behaviour. It is a string so that a third convention does not
  need a second knob, and an unknown value is refused.
* `banned = {ids}`: ids set to `-inf` before the penalty, the greedy branch, top-k and top-p read the
  row. So they are out of the `"row"` mass as well, as they are out of the reference's softmax.
* `uniform = u`: the draw itself, in `[0, 1)`, instead of one from the stream. The stream is not
  advanced.

The redraw, the window and the `min_len` ban are twelve lines of the exporter's Lua
(`cosyvoice3_driver/01_lm.lua`).

`uniform` is what makes a SAMPLED decode gateable. `scripts/cosyvoice3_reference.py` replaces the
reference's two `multinomial` calls per step with the same inverse-CDF walk the engine makes (sorted
descending, first running sum ≥ `u * mass`) over uniforms it records. That is a sampler of the same
distribution, and the engine given the same uniforms draws the same ids: 76/76 on the default
sentence, with the redraw firing 6 times. It is `run_ode`'s `caller_noise` ([ADR-040](adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md))
applied to a token.

## Consequences

* **rc10 ignores all three keys**, the hazard `min_p` already had: an older engine runs a CosyVoice3
  file with the `transformers` nucleus, no bans and its own draws. It still speaks, but it runs a
  different sampler and nothing reports it. rc11 must carry this.
* The defaults are unchanged, so every shipped driver gets the sampler it had
  (`tests/ci/test_sample_row.cpp` §1-10 unchanged, §11-13 new; a build with all three disabled fails
  them).
* A future checkpoint with a different nucleus convention adds a value to `top_p_mass`, not a knob.

## Related

* [ADR-024](adr-024-guidance-belongs-in-the-sampler.md): the sampler as the one place logits become a token
* [ADR-040](adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md): caller noise for the ODE
* [Retro-058](../retros/retro-058-a-size-stated-twice-was-never-compared.md): the abort the CosyVoice3 gate's sabotage arm found
* [Epic-03](../epics/epic-03-model-coverage.md): model coverage
* `loom-exporter/loom_exporter/cosyvoice3_driver/01_lm.lua`, `src/core/lua_bridge.cpp` (`sample_tensor_row`)
