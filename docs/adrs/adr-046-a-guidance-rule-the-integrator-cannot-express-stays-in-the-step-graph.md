---
type: adr
status: accepted
date: 2026-09-24
tags: [exporter, driver, flow-matching, guidance, family-9, voxcpm2]
supersedes: []
---

# ADR-046: A Guidance Rule the Integrator Cannot Express Stays in the Step Graph

## Context

VoxCPM2 generates one PATCH of 4 x 64-d latents per autoregressive step, and each patch is a flow
matching solve: a 12-layer local DiT integrated from a Gaussian draw over 10 Euler steps on a swayed
schedule. It is guided, but not the way F5-TTS and Chatterbox are guided. `UnifiedCFM.solve_euler` uses
**CFG-Zero\***:

```
st = <v_c, v_u> / (|v_u|^2 + 1e-8)          # a dot product over the whole patch
v  = v_u * st + cfg * (v_c - v_u * st)
```

and it skips the first `max(1, int(0.04 * (n + 1)))` steps entirely (their velocity is zero).
`loom.run_ode` already knows guidance ([ADR-040](adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md)),
but only as `v_c + s * (v_c - v_u)`: a fixed linear combination of the two evaluations, applied
elementwise. The projection `st` depends on the two velocities themselves.

## Options

**1. Teach `run_ode` CFG-Zero\*.** A second combination rule inside `compute_and_emit`, selected by an
option. Rejected: it is per-MODEL arithmetic in the engine
([ADR-003](adr-003-per-model-complexity-in-the-exporter.md)). The next checkpoint's guidance rule would
be a third branch there, and the zero-init skip would be a fourth option on an integrator that does not
otherwise know which step it is on.

**2. The driver's Lua evaluates the DiT twice per step and combines the velocities.** Rejected on
precision, not cost. The combination and the Euler update are f32 arithmetic in the reference, and the
driver's LuaJIT has doubles only. The Pocket-TTS lesson was the same: f32 arithmetic the reference does
belongs inside a graph.

**3. One graph per Euler step, `dit_step`, that holds the whole update.** It runs the DiT on the
conditional and the unconditional sequence as one batch of two, then the projection, the guidance and
`x - dt * v`. The driver's loop names the step (`t`, `dt`) and skips the zero-init steps.

## Decision

Option 3.

What makes a hand-written loop acceptable here is the size of the state. ADR-031 and ADR-040 put the
loop in C++ because F5-TTS's state is a whole mel spectrogram (86,800 floats for six seconds), which
would cross the Lua boundary four times per step. A VoxCPM2 patch is **256 floats**. Crossing it per
step costs nothing measurable next to a 12-layer DiT, so there is no boundary cost for `run_ode` to
save.

**The schedule ships as data.** `solve_euler` accumulates `t -= dt` and `dt = t - t_span[k + 1]` in
f32. Doubles cannot reproduce that sequence, so the export computes the default 10-step schedule with
the reference's own ops and ships it as the `euler_schedule` driver weight. Other step counts fall back
to a Lua recomputation that is a few ulps off.

## Consequences

* **No engine change.** The loop, the skip and the schedule are the export's. The combination rule is
  traced from the reference's own arithmetic and checked against `solve_euler` in torch at 2.4e-06 over
  a full 10-step solve (`tests/ci/test_voxcpm2_export.py`).
* **The unconditional run costs a second batch row, not a second call.** Its `mu` rows are zeroes
  (`solve_euler` zeroes `mu_in[b:]`), which the graph builds as `mu * 0`.
* **A guided sampler whose rule IS ADR-040's should still use `run_ode`.** This ADR covers a rule the
  integrator cannot express, on state small enough to cross. If either stops being true, the answer
  changes.

## Related

* [ADR-040](adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md): guidance as a property of the evaluation
* [ADR-031](adr-031-a-driver-edge-is-a-reference-unless-the-host-does-arithmetic.md): why state stays on the C++ side
* [ADR-003](adr-003-per-model-complexity-in-the-exporter.md): per-model complexity in the exporter
* [Epic-03](../epics/epic-03-model-coverage.md): model coverage
* `loom-exporter/loom_exporter/voxcpm2_export.py` (`DiTStepPhase`, `euler_schedule`)
