---
type: adr
status: accepted
date: 2026-08-13
tags: [numerics, approximation, gpu, stft, precision]
---

# ADR-008: `atan` Is the One Accepted Approximation

## Context

`atan` was the last `ggml_map_custom` host callback in the model zoo — one each in Kokoro's and
StyleTTS2's STFT phase. Unlike the reflect pad and the 1-D pool, there was nothing exact to compose
from: **`ggml` has no inverse trigonometry in any backend**, and every closed form for `atan` over the
available real ops routes through `asin`, `acos` or a complex logarithm, none of which exist either.

[ADR-007](adr-007-backend-capability-negotiation.md) requires an *exactly equivalent* composition. This
is the case where that is impossible.

## Options Considered

1. **Leave the host callback.** Keeps exactness, and keeps two graph cuts in two shipped models on
   every accelerator.
2. **Add an `atan` kernel to each backend.** Out of scope: it is upstream `ggml` work across many
   backends.
3. **Accept a bounded approximation**, confined to the backends that cannot dispatch the callback.

## Decision

Range reduction, a **degree-8 minimax polynomial** and a branchless reconstruction, measured at
**1.81 ULP** — and **confined to backends that cannot dispatch the host callback**, so a CPU build
still gets `libm`.

This is the project's first accepted approximation, and it is deliberately narrow: the mechanism that
confines it is the same `backend_can_run` gate from ADR-007, not a global build flag.

## Consequences

* **Positive:** the last host callback is gone; the affected models run whole modules on a GPU.
* **Positive:** the precision cost is bounded, measured, and does not exist on the CPU path the gate
  suite compares against.
* **Negative:** a device result and a CPU result are no longer bit-identical for these two models, so
  cross-backend comparisons need a tolerance rather than an equality.
* **Negative:** it sets a precedent. The confining mechanism is what keeps it from becoming a habit —
  any future approximation must state its ULP bound and its confinement the same way.

## Related

* Epic: [Epic-04: Backends and Accelerators](../epics/epic-04-backends-and-accelerators.md)
* Ledger record, verbatim:


The last `ggml_map_custom` node in the zoo. **ggml has no inverse trigonometry in any backend** — not in
the unary enum, not in CUDA, Metal, Vulkan or any other — so unlike the reflect pad and the 1-D pool
there was nothing exact to compose from: every closed form for `atan` over the available real ops routes
through `asin`, `acos` or a complex logarithm, none of which exist either.

So this one is an **approximation**, which is a first here, and it is confined by the same mechanism
P4.7e built: `op_atan` keeps libm's `std::atan` wherever `backend_can_run` says the backend can dispatch
the callback, so **a CPU-only build — the default — is bit-for-bit what it always was.** Only a device
that would otherwise have split the graph ever sees the polynomial.

### The method, which is what a GPU's own `atanf` does inline

Three stages, no branches — not CORDIC, which is what older fixed-function hardware used:

1. **Range reduction.** `atan(-x) = -atan(x)` strips the sign; `atan(x) = pi/2 - atan(1/x)` folds
   everything above 1 back below it. Written as `t = min(a,1) / max(a,1)`, which needs no reciprocal.
2. **A minimax polynomial in `z = t*t`**, Horner-evaluated so every step is a multiply-add. Not a Taylor
   series: Taylor converges far too slowly at the end of the interval to be affordable at this width.
3. **Branchless reconstruction.** `step(a-1)` is a 0/1 mask and `r + m*(pi/2 - 2r)` is the arithmetic
   form of a select, because a device wants every lane executing the same instructions.

### Degree 8, and why not 9

Fitted on a Chebyshev grid and evaluated **through the exact fp32 op sequence the engine emits**, against
`atan` in double precision:

| degree in z | max ULP |
|---|---|
| 6 | 11.74 |
| 7 | 3.55 |
| **8** | **1.84** |
| 9 | 2.56 |
| 10–14 | 1.88 – 2.18 |

It stops improving at 8 because past there the limit is fp32 rounding in the Horner evaluation itself,
not the polynomial. Further terms cost two graph nodes each and buy nothing. Measured on the real
implementation afterwards: **1.81 ULP**, against a test bound of 2.5. For reference a GPU vendor's own
`atanf` is typically specified at 2–4 ULP; glibc's is under 1, and glibc is what a CPU build still gets.

### One formulation was tried and rejected, by the test

`t = a / max(a,1)^2` saves a clamp by using `min(a,1)*max(a,1) == a`. It is wrong twice over, and the
first way is the kind of wrong that reaches production: `max(a,1)^2` **overflows to infinity** for any
`a` past ~1.8e19 — and `ggml_clamp` caps at `FLT_MAX` rather than at infinity, so `atan(inf)` came out
`inf/inf` = **NaN**. It is also less accurate in the folded branch (2.30 ULP against 1.86), because
squaring throws away bits the divide then cannot recover. The special-value assertions in
`tests/ci/test_atan_lowering.cpp` are what caught it, on the first run.

That test asserts the bound over three sweeps (both reduction branches and the decades from 1e-30 to
1e30), asserts `atan(0)`, `atan(±1)` and `atan(±inf)` **exactly**, and asserts **monotonicity** — a
polynomial can wobble inside an ULP bound and invert two neighbouring inputs, which an error bound alone
would never notice and anything comparing phases would.

### What it bought

| kokoro / styletts2 `decoder_vocoder` | splits | CPU nodes |
|---|---|---|
| before | 3 | 1 |
| after | **1** | **0** |

**Nothing falls back.** The whole vocoder runs on the device.

The wall clock did not move — 1274.8 ms against 1274.1 before, and 4.53x over the CPU either way — and
that is worth stating plainly rather than dressing up. The split it removed crossed a small tensor, and
**this machine's GPU reports `uma: 1`**: it shares memory with the host, so a "device→host→device round
trip" here is a synchronisation and not a transfer. Every split-cost number in P4.7 through P4.7f is
therefore a **lower bound** on what a discrete GPU over PCIe would pay, and the case for removing the
last split is stronger on the hardware this engine is not being measured on.

Also: the Kokoro reference gate test produces `rms=0.865596, max_abs=16.2979` on the Vulkan build both
before and after this change — swapping libm for a 1.8-ULP polynomial moved the vocoder's output by less
than the printed precision.

### Where the zoo now stands

* kokoro, styletts2, matcha, qwen3, lfm2, conformer, gigaam, parakeet ×2, vits, supertonic: **1 split,
  nothing falling back.**
* whisper `encoder`: **2 splits** — the 400-wide reflect pad, which the composition limit correctly
  declines (800 nodes into a 503-node graph) and which CUDA, Metal, SYCL and CANN all run natively
  anyway. The only genuinely Vulkan-shaped hole left, and the answer for it is a shader upstream.

`ATAN2` is still a host callback and deliberately untouched: **no exported model has ever contained one**
(0 occurrences across all thirteen). If one ever does, it is `compose_atan` plus the quadrant correction,
not new mathematics.

