---
type: retro
date: 2026-09-24
domain: engine
tags: [engine, lua-bridge, ode, robustness, sabotage, family-9, cosyvoice3]
---

# Retro-058: A Size Stated Twice Was Never Compared, and a Wrong One Aborted the Process

## The Issue

CosyVoice3's gate was run with a sabotaged sampler (top-p measured the `transformers` way) to confirm
its LM arm goes red. It did: 75 tokens where the reference drew 76. The free-running arm then took the
test process down with a ggml abort instead of reporting a failure:

```
ggml-backend.cpp:334: GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds")
  #4 LoomLuaBridge::run_ode_impl ... #6 run_ode_impl
```

## Root Cause

The flow-matching template calls `loom.run_ode` with both a caller's initial `state` (the pinned noise)
and `n_elems` (the size of the grid the estimator is sized by). `run_ode_impl` read `state` and, if
it was non-empty, never looked at `n_elems`. The gate's pinned noise is the REFERENCE's grid,
`(87 + 76) x 2 x 80` values. A decode that took a different number of tokens builds a smaller grid, and
the oversized state went straight into `ggml_backend_tensor_set`. That is an assert, not an exception,
so the host dies.

The other direction is worse: a SHORT state is an in-bounds write, so the integration ran over a
partly uninitialised carried tensor and returned a plausible mel.

Every earlier caller (F5-TTS, Chatterbox) passes a state it computed from the same length the driver
passes as `n_elems`, so the two never disagreed in a run that anyone saw.

## The Fix

When both are given, `run_ode` compares them and raises a Lua error naming both sizes
(`tests/ci/test_lua_bridge_ode.cpp` §5b, a state one value short). Under loom-py that is a Python
exception; before, it was a crashed interpreter.

## Takeaway

**When an API takes the same fact twice, compare the two copies at the boundary.** The redundancy is
free evidence, and the bug here was paying for it without reading it. And **run the sabotage arm all
the way through**: the CosyVoice3 gate's own checks were right, and the abort came from a LATER arm
fed a size that only a wrong earlier arm could produce. A gate that aborts on a failing input does not
report which check failed, so it has not been shown to fail cleanly until one has.

## Related

* [ADR-047](../adrs/adr-047-a-samplers-mass-its-bans-and-its-draw-are-the-callers-to-state.md): the sampler options whose sabotage found it
* [ADR-040](../adrs/adr-040-guidance-belongs-to-the-evaluation-not-the-integrator.md): caller noise, where `state` comes from
* [ADR-015](../adrs/adr-015-ci-and-gate-test-classes.md): a gate that cannot fail proves nothing
