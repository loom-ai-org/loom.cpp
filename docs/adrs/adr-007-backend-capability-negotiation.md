---
type: adr
status: accepted
date: 2026-08-13
tags: [ggml, backends, gpu, graph-splits, portability]
---

# ADR-007: A Primitive Asks the Backend What It Can Run

## Context

`ggml` defines ops that not every backend implements, **and the gaps do not line up**. CUDA has
`PAD_REFLECT_1D` but no `POOL_1D`; Vulkan has `POOL_2D` but neither; OpenCL, OpenVINO and Hexagon have
none of the three. An op a backend lacks becomes a node the scheduler must run on the CPU, cutting the
graph — and a cut graph was measured costing more than the op ever saved
([Retro-009](../retros/retro-009-host-callback-count-was-the-wrong-lens.md)).

## Options Considered

1. **Emit the native op and accept the splits.** What the engine did; measured Qwen3 at 0.95x and
   Matcha at 0.84x on a GPU — slower than the CPU.
2. **Decide at export time** — emit only ops the target backend has. But an export is **one GGUF that
   any backend may run**, so this compiles every artifact for the least capable backend in existence.
3. **Decide at run time, in the engine**, where the backend is known.

## Decision

A primitive queries the backend (`backend_can_run`) and emits **either the native op or an exactly
equivalent composition**. `PAD_1D_REFLECT` composes; `POOL_1D` is spelled as the `POOL_2D` every GPU
backend implements, with a one-tall window — but only *where it has to*, so Metal and SYCL, which do
implement `POOL_1D`, and a CPU-only build all keep the native op.

**The topology does not change.** The same Kokoro file builds 1692 `ggml` nodes on a CPU and 1732 on
Vulkan, and its topology says `PAD_1D_REFLECT` either way.

## Consequences

* **Positive:** one artifact stays portable and gets the best available lowering per device. Eleven of
  twelve models run a whole module on the GPU with nothing falling back.
* **Positive:** it is the correct home for the decision, so new backends need no re-export of anything.
* **Negative:** the substitution must be *exactly* equivalent, which is a per-op proof obligation. Where
  no exact composition exists anywhere, the engine needs a separate, explicit decision — see
  [ADR-008](adr-008-atan-approximation.md).
* **Negative:** node counts now vary by backend, so any test asserting a node count must say which
  backend it means.

## Related

* Epic: [Epic-04: Backends and Accelerators](../epics/epic-04-backends-and-accelerators.md)
* Ledger record, verbatim:


P4.7d's support matrix showed P4.7c had solved the right problem in the wrong repo, and this is the
correction — generalized, because the shape of it recurs: **ggml defines ops that not every backend
implements, and the gaps do not line up.** CUDA has `PAD_REFLECT_1D` but no `POOL_1D`; Vulkan has
`POOL_2D` but neither; OpenCL, OpenVINO and Hexagon have none of the three.

**The exporter cannot answer this question and the engine can.** An export is ONE GGUF that any backend
may later run, so composing around a gap there compiles every artifact for the least capable backend
anyone might use — P4.7c had Kokoro shipping a forty-node open-coding of a pad that CUDA, Metal, SYCL
and CANN all run in one node. The engine sees the actual backend, and `ggml_backend_supports_op` will
answer directly.

So `PrimitiveContext` now carries the `Backends`, and a primitive in that position builds the native op,
**asks**, and keeps it or emits an equivalent composition:

```
ggml_tensor* native = ggml_pad_reflect_1d(pc.ctx, a, lp0, rp0);
if (backend_can_run(pc, native) || lp0 + rp0 > kReflectPadComposeLimit) return {native};
return {compose_pad_reflect_1d(pc.ctx, a, lp0, rp0)};
```

### One artifact, two lowerings

The same Kokoro GGUF, whose topology says `PAD_1D_REFLECT` and means it:

    cpu  (CPU     ): 1692 ggml nodes, 0 splits     <- native op, nothing composed
    gpu  (Vulkan0 ): 1732 ggml nodes, 3 splits     <- composed, because Vulkan has no kernel for it

That is the whole point: the decision is made where the backend is known, per run, and the file on disk
says what the model does rather than what one backend could not do. The topology went back to 1373 nodes
(from P4.7c's 1413), and the exporter change is reverted.

The device outcome is unchanged from P4.7c — 3 splits, 1 CPU node (the `ATAN`) — which is the point: the
same result, obtained without teaching every artifact about Vulkan.

### Two rules for anything added this way

Both were learned by nearly getting them wrong, and both are written into `primitive_registry.h` beside
the helper:

* **The fallback must be EXACTLY equivalent, and shown to be.** P4.7d found two spellings of the same
  ggml op that divide by different numbers; a composition that differs at the edges is a wrong answer no
  shape check catches. `tests/ci/test_pad_reflect_lowering.cpp` compares the composition against
  `ggml_pad_reflect_1d` bit-for-bit across seven shapes and widths — **and independently asserts what
  reflect padding IS** (`[a,b,c,d]` with (2,1) → `[c,b,a,b,c,d,c]`), because two implementations can be
  wrong the same way and ggml's convention is not something either of them gets to define.
* **A composition has a width past which it stops being worth it.** `kReflectPadComposeLimit = 32`;
  above it the native op is kept and allowed to fall back. Whisper pads 200 either side, which would be
  800 nodes in a 503-node graph.

### Applied to POOL_1D as well

`op_pool_1d` was the first lowering of this kind and it predated the mechanism, so it substituted
`ggml_pool_2d` unconditionally wherever the two were equivalent. That worked, and it was doing more than
it needed to: **Metal and SYCL implement `POOL_1D`**, and a CPU-only build — the default — implements
everything, so all of them were getting a rewritten graph to work around a gap they did not have.

It now asks first, and the same Whisper GGUF resolves differently per backend:

    cpu  (CPU     ): POOL_1D=1  POOL_2D=0
    gpu  (Vulkan0 ): POOL_1D=0  POOL_2D=1

The order of the two conditions is worth keeping: `backend_can_run` first, `pool_2d_fallback_is_equivalent`
only as the reason to reach for the fallback. Where there is no equivalent fallback — a padded average —
the native op stays and the scheduler sends it to the CPU, because a correct fallback beats a fast wrong
answer. `pool_1d_lowers_to_pool_2d` is renamed `pool_2d_fallback_is_equivalent` to say which of the two
questions it answers.

### What this does not do

`backend_can_run` is unreachable on a CPU-only build -- the CPU implements every op, so the native branch
always wins there and no hermetic test can provoke the other one. The test therefore checks the two
spellings against each other rather than trying to force the branch, so that whichever a device takes, it
takes one of two things already known to be identical. The branch itself is exercised only by a real
device, which is what `tests/gate/test_e2e_device_parity*.cpp` are for.

This mechanism was built for ops with an EXACT composition, and the remaining `ATAN` had none — ggml has
no inverse trig in any backend. **P4.7f takes it anyway, as an approximation**, which is a different
decision with a different bar: an accuracy budget stated and measured rather than a bit-identity claim.
The mechanism turned out to be exactly what made that acceptable, because it confines the approximation
to the backends that cannot do better.

