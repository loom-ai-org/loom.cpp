---
type: adr
status: accepted
date: 2026-09-24
tags: [engine, lua-bridge, drivers, rng, flow-matching, vocoder, family-9, chatterbox, performance]
supersedes: []
---

# ADR-042: A Sampler's Noise Is the Engine's; a Graph Input's Noise Is the Driver's

## Context

Every random value in the zoo comes from one host stream, the bridge's `rng_` (`std::mt19937` with a
float normal and a float uniform distribution). Seeded runs are reproducible only if the draw ORDER
stays the same. There are two ways a value leaves that stream:

1. **As a sampler's starting state.** `loom.run_ode` / `loom.run_ode_and_retain` take `n_elems` and
   draw the state themselves, at the point in the stream `loom.gaussian_array` used to take
   ([ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md)). A caller may
   pass `state` instead. `FlowMatchingSampler` (`flow_matching_export.py`) emits this for every
   flow-matching model: Matcha and Supertonic draw in the engine, while F5-TTS and Chatterbox's
   estimator declare `caller_noise`, so the engine draws unless `inputs.noise` is given.
2. **As an ordinary graph input.** The driver calls `loom.gaussian_array` / `loom.uniform_array` and
   passes the Lua table, and the caller may override it:
   * the NSF source's per-sample noise in Kokoro, StyleTTS2 and Chatterbox (a phase per harmonic, then
     one Gaussian per harmonic per output sample, in that order);
   * VITS's `z_noise`, and its `z_p` draw, which the driver then does arithmetic on;
   * SNAC's `NoiseBlock`s through the `NOISE` binding in `driver_components.py`;
   * any traced `randn` / `rand`, which `exporter.py` lowers to the same two calls.

StyleTTS2's ADPM2 style sampler is the one sampler outside (1). It is an ancestral SDE whose state is
256 floats, so it stays in Lua, as ADR-031 records.

Chatterbox's vocoder noise raised the question of moving (2) into the engine too: a
`{gaussian = n}` / `{uniform = n}` input spec that the binding fills straight into the tensor from
`rng_`. This would remove the Lua table of 9 × 480 values per frame, about 311k for 1.4 s of audio.

## Measurement

One seeded synthesis (`seed=1`, 72 frames, 1.44 s of audio) on the dev box, with temporary
`steady_clock` timers in `compute_and_emit` and `loom.gaussian_array`:

| where | time | share of 43.2 s |
|---|---|---|
| `estimator`, 20 evaluations (10 Euler steps × CFG) | 29.5 s | 68% |
| `t3_lm` + `t3_lm_uncond`, 37 steps each | 12.1 s | 28% |
| `vocoder` compute | 0.92 s | 2.1% |
| `flow_encoder` | 0.51 s | 1.2% |
| **the noise crossing**: table push 4.0 ms + input fill ≤ 5.9 ms | **~10 ms** | **0.02%** |
| drawing the 311,040 Gaussians | 9.0 ms | paid in both designs |

The fill figure also covers the mel tail and the nine phases, so the noise's own share is smaller.

## Options

* **Engine-side fill for Chatterbox only.** Rejected. It would make Chatterbox the one NSF vocoder
  whose noise follows a different rule from Kokoro's and StyleTTS2's, which draw the same kind of
  noise the same way. That would reduce consistency, and it would buy about 0.02% of synthesis time.
* **Engine-side fill for every graph-input draw.** Rejected for now. It would touch every emitter in
  (2) above, and making the files consistent would mean re-exporting and re-uploading the shipped
  Kokoro, StyleTTS2, SNAC and VITS files. It also would not remove the bytes that matter on an
  accelerator: `rng_` lives on the host, and ggml has no random-number op, so an engine-side fill
  still generates on the host and uploads once, just as the Lua path does. The only thing removed is
  a bulk `push_number_array` and one double-to-float conversion. ADR-031's own measurement says that
  is the cheap kind of crossing: the savings there tracked interpreted per-element Lua loops, not
  bulk pushes.
  It also adds a hazard. `compute_and_emit` walks the inputs table with `lua_next`, whose order is
  undefined, so two engine-drawn inputs in one call would draw in hash order. That needs a
  one-draw-per-call rule, and a rule that `run_ode`'s fixed inputs never contain one, because they
  would be redrawn at every evaluation.
* **Keep the two conventions and write them down.** Chosen.

## Decision

**A sampler's loop-carried noise is drawn by the engine.** A diffusion or flow-matching sampler's
starting state goes through `run_ode`'s `n_elems` and never becomes a Lua table. A caller override is
`state` / `inputs.noise` (`caller_noise`). A new sampler follows this rule unless it is an ancestral
SDE that `run_ode` does not express, as ADPM2 is.

**Noise that a graph consumes as an input is drawn by the driver.** It uses `loom.gaussian_array` /
`loom.uniform_array` and the form `inputs.<name> or <draw>`, so a caller can pass the draw and an oracle
stays exact. Draws happen in the reference's order, from the one shared stream.

## Consequences

* Chatterbox ships as it is: `04_nsf.lua` keeps its two draws, the gate keeps passing the reference's
  draws (2.51e-05), and no binding is added, so there is nothing new for rc11 or loom-py's vendor pin.
* Reopen this if a graph-input draw ever sits inside an interpreted Lua loop, or once the engine has
  an on-device generator. Either one changes the cost this decision was measured on.
* The time in Chatterbox is the estimator's (68%). That is where to look for speed, not the vocoder
  boundary.

Related: [ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md),
[ADR-040](adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md),
[ADR-029](adr-029-a-multi-rate-codec-keeps-one-row-per-coarsest-frame.md) (SNAC's noise, measured at
export as immaterial), [Epic-01](../epics/epic-01-inference-engine-core.md).
