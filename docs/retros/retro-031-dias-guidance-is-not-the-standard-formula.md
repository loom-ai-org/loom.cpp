---
type: retro
date: 2026-09-03
domain: model-coverage
tags: [family-10, guidance, sampling, transformers, reference]
---

# Retro-031: Dia's Guidance Is Not the Standard Formula

## Issue

Classifier-free guidance is a formula everyone knows: `uncond + scale * (cond - uncond)`. It is what
`transformers`' own `ClassifierFreeGuidanceLogitsProcessor` computes, it is what this engine's
`loom.sample_row` implements, and it is what family 10's plan was scoped against — the handover for
that work says CFG "combines LOGITS, `uncond + scale * (cond - uncond)`" and treats the remaining work
as plumbing.

**Dia does neither half of that.** `DiaClassifierFreeGuidanceLogitsProcessor` is:

```python
scores_processed = cond_logits + (cond_logits - uncond_logits) * self.guidance_scale
if self.guidance_top_k is not None:
    _, top_k_indices = torch.topk(scores_processed, k=self.guidance_top_k, dim=-1)
    scores_processed = cond_logits.masked_fill(top_k_mask, -float("inf"))
```

Two differences, and only the first is a difference of degree:

1. **It is centred on the conditional logits**, not the unconditional ones. `cond + g*(cond - uncond)`
   is `uncond + (g+1)*(cond - uncond)`, so the checkpoint's declared `guidance_scale: 3.0` is the
   general form's **4.0**. Implemented as written, every generated code would have been produced at a
   guidance strength the checkpoint never asked for — and it would have sounded like something, which
   is the problem.
2. **`guidance_top_k` selects with one array and scores with another.** The guided logits pick a
   shortlist of *k*; the **conditional** logits, restricted to that shortlist, are what gets drawn
   from. The guided values are discarded. That is not a parameter of the formula, it is a second
   operation — guidance sharpens the *ranking* and the model's own distribution over that shortlist is
   what is sampled. No general CFG primitive expresses it, and `transformers` passes
   `generation_config.top_k` into it, so the same number 50 lands in two unrelated roles.

## Root cause

The formula was taken from the name of the technique rather than from the model's own processor. Both
processors are in `transformers/generation/logits_process.py`, forty lines apart; the generic one is
what a reader who already knows what CFG is will find, and it is right about every model except the
one being exported.

There was no signal that would have caught it. Guidance is deterministic, so a wrong scale produces a
perfectly reproducible generation; the shortlist step changes which ids are candidates, not how many;
and the only reference this family compares against was captured **with guidance off**, so it would
have agreed with the export exactly while the guided path was wrong in two ways.

## Takeaway

**A named technique is a family, and a checkpoint implements one member of it.** Before implementing
anything a paper named — guidance, RoPE scaling, a sampler, a schedule — read the *model's own*
processor, not the generic one that shares the name. The generic one is what the reader already
believes, which is exactly why it does not get checked.

Two concrete rules this produced:

* **The engine implements the general form; the model's convention converts in its driver.** Dia's
  `+1` is one line next to the constant it belongs to, and the GGUF declares the checkpoint's own
  number so `model.hparam("sampling.guidance_scale")` and `generation_config.json` cannot disagree.
* **A reference captured with a feature OFF cannot grade that feature.** The guided path needed its
  own oracle, and it could have one because guidance — unlike sampling — is deterministic: greedy
  with guidance on is exactly comparable against `transformers`, and that is the arm
  `test_e2e_dia_mil_export.cpp` gained.

## Record

The two processors, verbatim, at transformers 4.57.6:

* `ClassifierFreeGuidanceLogitsProcessor` — `scores_processed = uncond + (cond - uncond) * scale`
* `DiaClassifierFreeGuidanceLogitsProcessor` — the block quoted above, plus its own docstring:
  *"we do not keep the logits of the combined CFG output, but the conditioned output only."*

Composition order, from `DiaGenerationMixin._get_logits_processor`: the CFG processor is **inserted at
index 0** — ahead of `DiaEOSChannelFilterLogitsProcessor`, which is why the guided shortlist is taken
over the whole row and the per-channel window applied after it. `DiaEOSDelayPatternLogitsProcessor` is
appended last.

## See also

* [ADR-024](../adrs/adr-024-guidance-belongs-in-the-sampler.md) — where guidance lives in the engine
* [ADR-023](../adrs/adr-023-a-second-stream-is-declared-not-derived.md) — the two streams it needs
* [Retro-006](retro-006-kokoro-shipped-noise.md) — why shipping the greedy decode was not an option
