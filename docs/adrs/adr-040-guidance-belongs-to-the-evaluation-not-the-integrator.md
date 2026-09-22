---
type: adr
status: accepted
date: 2026-09-18
tags: [engine, lua-bridge, flow-matching, family-9, f5-tts, model-coverage]
supersedes: []
---

# ADR-040: Classifier-Free Guidance Belongs to the Evaluation, Not to the Integrator

## Context

`loom.run_ode` integrates `dx/dt = f(x, t)` with the loop and the state on the C++ side
([ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md)): the driver names one graph, a schedule
and a method, and the engine runs the Butcher tableau. Two flow-matching models shipped through it and
both integrate the estimator's output directly.

F5-TTS does not. Its velocity field is not what its DiT returns — it is

```
v = v_cond + scale * (v_cond - v_uncond)
```

two evaluations of the same graph per stage, one on the reference conditioning and one on a dropped
one. That is classifier-free guidance, and it is not optional for this checkpoint: it was trained and
evaluated with `cfg_strength = 2.0`, and `1.0` is a different model.

So the question is where the second evaluation and the combination live.

## Options

**1. In the driver's Lua, as a hand-written loop.** This is what StyleTTS2's ADPM2 sampler does, and
`flow_matching_export.py`'s own docstring says the integration rule is deliberately not generalised.
The cost is the one ADR-031 was written from: F5-TTS's state is a whole mel spectrogram
(`n_frames * 100` floats — 86,800 for a six-second clip), and a Lua loop crosses it *in* and the
velocity *out*, per evaluation. Guided, that is four crossings per step instead of two, 32 steps deep,
for an update that is elementwise and has no decision in it.

**2. As a second integrator method.** `kOdeMethods` is a table of Butcher tableaus; "guided euler"
could be a row in it. Rejected: guidance is orthogonal to the tableau. It would have to be duplicated
into every method, and the two evaluations are not two *stages* — they are one stage evaluated twice.

**3. As a property of the EVALUATION, inside `run_ode`.** `guidance = {inputs = ..., scale = ...}`:
the same module, the same graph, the same cache, called a second time with a different fixed-input
table, combined where `k[stage]` is filled — before any Butcher weight is applied.

## Decision

Option 3.

`compute_and_emit` already takes its inputs table by absolute stack index, so the unconditional run is
the same call with a second index and the whole change is one factored lambda plus a combination loop.
Every method in the table keeps working unchanged, because the integrator sees one velocity field and
nothing about it has to know guidance happened.

**`inputs` is what differs, not `module`** — and that is the asymmetry with `loom.generate`, which
spells its own guidance `{module =, scale =, top_k =}`. There the two runs are two KV-cached histories
that must not see each other, so they must be two modules. Here they are two values of one graph's
inputs, and using two modules would allocate a second copy of a 22-layer graph for nothing.

On the export side this is two declarations on the existing template rather than a bespoke sampler:
`FlowMatchingSpec.guidance` (a bool) and `FlowMatchingSpec.schedule` (`"uniform"` or `"caller"`, because
F5-TTS integrates a sway-reparameterised linspace rather than `k/n_steps`). The loop is unchanged,
which is the test of whether a template still fits.

**`guidance` is a bool rather than a list of inputs.** The two runs supply the *same input names* —
guidance drops a conditioning signal's value, not its existence — so `supplied_inputs` already
describes both tables and the existing `TopologyInput` link checks both by checking one. A declaration
that could name a different set would let the unconditional table drift from the graph with nothing to
catch it.

## Consequences

* **Every shipped flow-matching model is untouched.** Absent the `guidance` key the loop is the
  single-evaluation one, and `tests/ci/test_lua_bridge_ode.cpp` pins that: guidance at scale 0 is
  **bit-identical** to the unguided call, and Euler is still bit-identical to the Lua loop it replaced.
* **The estimator runs twice per stage, so a guided model costs twice the graph work per step.** That
  is the model, not the engine: the reference packs the two runs into one batch-2 forward, which is the
  same arithmetic with the sequence axis doubled instead of the call count.
* **`guidance.scale` is required when `guidance` is present.** A missing scale is named rather than
  defaulted to 1.0 — a guidance weight of 1 is `v_cond + 1*(v_cond - v_uncond)`, not "no guidance",
  and silently picking it would be picking a different model.
* The engine's method table stays the authority on integrators, and the export side stays unable to
  check a method name — unchanged by this ADR, and recorded here because the same `Unchecked` note now
  covers `schedule` for the same reason.
* **A guided flow-matching export cannot be graded without also pinning the initial noise**, which is a
  consequence of this family rather than of guidance: the state is a Gaussian draw and the two RNGs are
  different algorithms, so the same seed is a different sample. `FlowMatchingSpec.caller_noise` is the
  third declaration this template grew, and `loom.run_ode`'s existing `state` option is what it reaches
  — the engine already drew only when no state was supplied, so nothing in the binding changed.

## Related

* [ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md) — why the loop and the state moved into
  the engine at all, and the measurement that decided it.
* [Epic-03 §2](../epics/epic-03-model-coverage.md) — family 9's third leaf, and what it cost.
