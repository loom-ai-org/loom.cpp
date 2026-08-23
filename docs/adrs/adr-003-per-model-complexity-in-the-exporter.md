---
type: adr
status: accepted
date: 2026-08-07
tags: [lean-runtime, edge-ai, code-retirement, orchestration]
---

# ADR-003: Per-Model Complexity Belongs in the Exporter, Not the Engine

## Context

Even after [ADR-001](adr-001-data-driven-gguf-topologies.md) and
[ADR-002](adr-002-embedded-lua-drivers.md), the engine still held a substantial amount of per-model
C++: transducer decoders, ODE/CFM samplers, style-diffusion samplers, per-family drivers, a
per-model text vectorizer. Each was orchestration — a host-driven sampling or integration loop — and
each already had, or could have, a Lua counterpart on the MIL path.

The engine targets edge devices. Every kilobyte of per-model C++ is carried by every deployment,
including the ones that will never run that model.

## Options Considered

1. **Keep the C++ for anything already written and verified.** Cheapest today; makes the engine's size
   a function of how many models the project has ever supported.
2. **Move orchestration behind engine bindings** (e.g. a `loom.ctc_greedy_decode` call). Smaller than a
   full decoder, but it is still family-specific logic living in an engine that is meant to stay small.
3. **Move orchestration out of the engine entirely**, into exporter-generated Lua and
   exporter-emitted data.

## Decision

**Per-model complexity belongs in the exporter.** C++ in the engine is for what is per-*task* rather
than per-model — tokenizers, CTC decode primitives, the caches, the primitive registry.

Executed as a retirement policy: `tools/convert_*` bespoke converters, `src/core/tdt_decoder.cpp`,
`cfm_euler_sampler`, `ode_stepper`, `style_diffusion_sampler` and the per-model drivers are gone, along
with the tests whose only purpose was to exercise them. A new family should cost a Python change.

**The standing rule for new work:** if a proposed addition to the engine is specific to one model, it
belongs in the exporter. If it is specific to one *task* and shared across models, it may live here.

## Consequences

* **Positive:** the engine stops growing with the model count. Adding a family is an export.
* **Positive:** the same retirement makes the numerical oracle stronger — a decoder in Lua is gated
  against the real upstream implementation rather than a hand-rolled C++ reimplementation of it.
* **Negative:** one known exception is carried deliberately.
  `src/core/supertonic_text_vectorizer.cpp` is per-model C++ kept because it is written and verified —
  and a **second** grapheme TTS model must not add a second class. The generalization is scoped in
  [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md).
* **Negative:** debugging moves across a language and a repository boundary.

## Related

* Enforced by: [ADR-004](adr-004-mil-as-the-single-export-path.md)
* Ledger record of the retirements, verbatim:

### The stranded pre-MIL components — DONE (2026-08-07)


P4.0.8's second follow-up, taken as the decision it was filed as. `cfm_euler_sampler.{h,cpp}`,
`ode_stepper.{h,cpp}` and `style_diffusion_sampler.{h,cpp}` are gone, with the four tests that were
their only consumers (`test_e2e_toy_ode`, `test_style_diffusion_sampler`,
`test_e2e_styletts2_diffusion_sampler`, `test_e2e_supertonic_cfm_sampler`) and the five Python
reference/fixture generators feeding them. Each was orchestration — a host-driven sampling or
integration loop — which is exactly the work `EXPORT-PREPARATION.md` §1.3 records as the exporter's,
and each already had its Lua counterpart on the MIL path.

**The follow-up got one of its four wrong, and the correction is the useful part.** It filed
`bilstm_stepper.h` alongside the others as "a unit test and no product consumer". It has neither: it
has **no unit test at all**, and three real construction sites —
`test_e2e_kokoro_{text_encoder,duration_predictor,f0n}.cpp` build a `BiLstmStepper` to *drive* the
bespoke per-topology check each of those tests exists for. Deleting it deletes those checks, which is
stage C's rule verbatim. Its MIL counterpart (`loom.run_recurrent` + `RecurrentPhase`) has replaced it
in every driver; what keeps it alive is the bespoke conversion path, so it retires with that in P6.
`loom.h` now says so, with the measurement rather than the verdict.

**What the deletions cost, stated rather than glossed.** Three of the four tests were exact numeric
comparisons against Python references, and two of those cannot be reproduced against the Lua
counterpart at all: they replayed fixture noise through an injectable `GaussianSampleFn`, and the
driver draws from `loom.gaussian_array`, which has no such seam. That is a real loss of resolution —
but it is a check *of the code being deleted*, not of anything that ships. Nothing on the MIL path was
covered by them, and StyleTTS2's frozen full-pipeline waveform (`fixtures/legacy_driver_reference/`)
was produced by the retired C++ driver *through* `adpm2_sample`, so the ground truth those tests
established survives one level up, at a looser bound. Supertonic's CFM loop is deterministic given
`z0` and needs no such argument: `test_e2e_supertonic_mil_lua_driver` is a genuine equivalent.

`test_graph_reuse_safety.cpp` is deliberately kept and re-headed. It was written *about* `OdeStepper`,
but what it pins is a property of `ggml_gallocr`, in plain ggml calls, and it is what would catch a
ggml upgrade invalidating `GraphBuilder`'s reason for not needing that discipline.

**Gate:** `ctest` **128/128, 0 failed** (135 before; the seven removed are the four tests plus three
fixture-generator setup steps). Engine size, RelWithDebInfo stripped, same configuration both sides:
**1,248,832 → 1,240,632 bytes, −8,200 (−0.66 %)**; `.text` 1,224,663 → 1,216,923 (−7,740, −0.63 %).
Small, and honestly so — these were four short files, unlike E.4's ~7k lines of driver.


### Parakeet decodes in Lua, and the C++ transducer decoder is gone — DONE (2026-08-07)

P4.0.17 step 2, complete. `parakeet-tdt` and `parakeet-rnnt` now export as one GGUF holding five
topologies (encoder, embed, two prediction cells, joint) plus a driver that decodes. The TDT double
loop — encoder-frame pointer × symbols-per-frame — is a checked Lua fragment, and
`src/core/tdt_decoder.cpp` no longer exists.

**The oracle changed, and that is what makes this more than a port.** The retired path was gated
against `reference_forward_parakeet_*.py`, a hand-rolled PyTorch reimplementation run on a synthetic
waveform — and for RNN-T that fixture decodes to an *empty token list*, so the test could not tell a
working decoder from a broken one. The new gate is NeMo's own `model.transcribe()` on 11 seconds of
real speech.

**That found a defect in the code being deleted.** On `samples/jfk.wav` the C++ decoder emitted **36**
tokens; NeMo emits **38**. It was dropping two `7877`s — the commas in *"And so, my fellow
Americans,"*. The Lua driver reproduces all 38, and RNN-T's 26, exactly. The migration plan in this
very entry originally said the gate was "36 tokens for TDT" — matching the path being removed would
have preserved the bug and called it a pass.

  | | tokens | matches NeMo |
  |---|---|---|
  | retired C++ decoder | 36 | no — two dropped |
  | Lua driver | **38** | **yes, id for id** |
  | Lua driver (RNN-T) | **26** | **yes, id for id** |

The traced phases were confirmed against the independent PyTorch reference first (`[7618, 1815, 7883]`
on the reference waveform, exactly), which is what isolated the jfk difference to the decoder rather
than to the new traces.

**Retired:** `tdt_decoder.{h,cpp}`, `convert_parakeet_tdt.py`, `convert_parakeet_rnnt.py`,
`reference_forward_parakeet_{tdt,rnnt}.py`, the four `{tdt,rnnt}_step` fixture generators, and six
tests — `test_{tdt,rnnt}_decoder`, `test_e2e_parakeet_{tdt,rnnt}`,
`test_e2e_parakeet_{tdt,rnnt}_mil_export` — replaced by one `test_e2e_parakeet_lua_driver`. ctest drops
from 146 to 137 tests while covering strictly more.

**A checker blind spot fell out of this.** `parse_run_subgraph_calls` scanned only for
`loom.run_subgraph(`, so a fragment using `run_subgraph_and_retain` had that call site *invisible* —
the coverage report said the `joint` topology was named by nobody. A blind spot in the checker reads
exactly like a driver that does not call something, which is the failure this parse exists to prevent.
It has scanned both forms since. P4.0.12 added the retaining form and nothing taught D.2's machinery
about it.

