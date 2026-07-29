# Export Process Improvement Proposals

## Context

The MIL-based exporter (`tools/loom_mil_compiler/`) works and, as of this writing, has successfully
retrofitted all 8 target models (Qwen3, Conformer-CTC, Parakeet RNNT/TDT, VITS, Matcha, StyleTTS2,
Kokoro, SupertonicTTS) end to end. But two real pain points surfaced while doing that retrofit work,
and were discussed at length before writing this document (full transcript in
`EXPORT-IMPROVEMENT-MLIR-INTERACTIONS.md`, which records an external-model conversation the author had
about the first pain point):

1. **`exporter.py` (4,440 lines) is a large, growing `if op_type == "..."` dispatcher** mapping MIL ops
   to `ggml` ops/composites, with ad hoc, duplicated shape/const-resolution heuristics scattered through
   it (e.g. `_try_resolve_reshape_shape_input`). Each new model architecture tends to add new branches
   and new special cases rather than composing cleanly with what's there.
2. **The embedded Lua "driver" (orchestration) is derived automatically for exactly one model family**
   (decoder-only causal LMs, via `modular_discovery.py`/`modular_export.py`/`apply_modular_export`)
   and hand-written for everything else (`export_styletts2_mil.py`: 340 lines, `export_kokoro_mil.py`:
   503 lines, vs. `export_qwen3_mil.py`: 28 lines for the generalized family). The goal — inferring the
   orchestration graph (what feeds what, where loops live, where outputs emerge) directly from the
   traced model, ONNX-exporter-style, for virtually any architecture — is not yet realized outside that
   one family.

This document proposes concrete, incremental work for both, and explicitly rejects one tempting
but wrong-direction fix (replacing the MIL frontend with StableHLO/MLIR) so a future agent doesn't
re-litigate it from scratch.

---

## Rejected direction: swapping MIL for StableHLO/torch_xla

An external model proposed ripping out `coremltools`/MIL and adopting `torch.export` +
`torch_xla.stablehlo` to get a "pure math" IR with no implicit semantics, on the theory that this
would eliminate the exporter's heuristic mapping rules. After reviewing the actual exporter code, this
was rejected:

- **MIL is already close to `ggml`'s abstraction level.** Ops like `layer_norm`, `instance_norm`,
  `gelu`, `conv`, `matmul` are each a single MIL op mapping to a single `ggml` op/composite
  (`exporter.py` lines ~3356, 3411, 2193, 3565, 2129). StableHLO is *lower*-level than both MIL and
  `ggml` — it would expand `layer_norm` into `reduce`/`broadcast_in_dim`/`subtract`/`multiply`/`rsqrt`.
  Mapping to `ggml` would then require *re-fusing* those primitives back into the exact op you started
  with — a harder pattern-recognition problem (order/shape variance across XLA versions), not an easier
  one. This relocates the heuristics rather than removing them.
- **The bespoke ops are the hard part, and a rewrite doesn't make them easier.** `vits_spline_op.py`,
  `istft.py`, `group_norm_op.py`, `recurrent.py` are hand-built MIL lowerings for operators that don't
  trace cleanly through any generic frontend. Redoing this under `torch.export` custom-op registration
  + `torch_xla` composite lowering is a full rewrite of already-working infrastructure, not a
  simplification.
- **`torch_xla` is a heavier, more version-brittle dependency** than `coremltools` in practice — built
  and tested primarily for TPU/GPU workflows, with tighter torch-version pinning and rougher
  CPU-only support.
- **Timing/risk**: all 8 target models are done and verified on the current pipeline. Discarding it for
  an unproven alternative, whose main promised benefit doesn't hold up under inspection (see above), is
  not justified.

**Conclusion: keep MIL as the frontend.** The items below improve the current pipeline instead.

---

## Actionable items

### 1. Replace `exporter.py`'s `if/elif` dispatcher with a declarative pattern table

**What**: Refactor the large `if op_type == "..."` chain in `exporter.py` (`transpile_operation` and
friends) into a table of `(mil_op_type, guard_predicate) -> ggml composite builder` entries, looked up
and applied mechanically, instead of nested procedural branches.

**Why**: This captures most of the "declarative rewrite rule" benefit from the original MLIR-style
proposal (pattern → replacement, no manual tree traversal) without touching the IR at all. It also makes
it obvious, for a given MIL op, exactly which guard conditions select which `ggml` mapping — currently
that logic is interleaved with side-effecting shape resolution code, making the decision criteria hard
to audit at a glance.

**Evidence**: `tools/loom_mil_compiler/exporter.py` is 4,440 lines; a `grep` for `if op_type ==`/`if op.op_type ==` turns up ~50+ direct branches plus nested special cases (e.g. lines 3565–3653 for `conv`/`conv_transpose`, 3356–3526 for the norm family).

### 2. Centralize shape/const resolution into one pre-pass

**What**: Pull the scattered shape-inference and constant-resolution helpers (e.g.
`_try_resolve_reshape_shape_input`, the various `producer.op_type == "const"` checks sprinkled through
op handlers) out of individual op handlers and into a single pre-pass that runs once over the MIL
program before op-mapping begins, annotating each value with its resolved shape/const status.

**Why**: Right now this logic is duplicated ad hoc wherever an op handler happens to need it, which is
exactly the kind of heuristic sprawl that made the exporter hard to reason about. Individual op handlers
should be able to assume resolved shapes are already available, rather than re-deriving them
per-callsite.

**Evidence**: symbolic/const-resolution checks appear at `exporter.py` lines ~1496, 1514, 1539, 2003,
2032, in addition to the dedicated `_try_resolve_reshape_shape_input` helper — the same class of problem
solved independently in at least 5 places.

### 3. Capture iterative-refinement loops as explicit control-flow ops before tracing

**What**: For models whose orchestration is a fixed-or-runtime-determined iteration over loop-carried
tensor state — StyleTTS2's diffusion sampler, Matcha's ODE solver steps, VITS/Supertonic's
duration-predictor loop — rewrite or wrap the relevant Python loop using an explicit control-flow
primitive (e.g. a `torch.cond`/`while_loop`-style higher-order op) *before* tracing, instead of letting
`torch.jit.trace` silently unroll a plain Python `for`/`while`.

**Why**: This is the actual blocker on "infer the orchestration graph from the trace" for anything
beyond decoder-LLMs — not a limitation of MIL vs. any other IR. `recurrent.py`'s own docstring already
confirms coremltools' MIL *can* capture genuine loop structure: tracing a plain `torch.nn.GRU` produces
a real MIL `while_loop`/`slice_by_index` sequence, not an unrolled one. `exporter.py` already has
`while_loop`/`cond` cases in `transpile_operation` (lines ~1804, 1815). The gap is that the *source*
models' hand-written Python loops don't emit these ops today because they're plain imperative control
flow, not because MIL can't represent them. Note this is also not a limitation ONNX-style export
tooling magically solves in general — no mainstream exporter infers arbitrary recurrence from an eager
Python loop; they all rely on the loop being expressed through an explicit higher-order/scripted
construct (or, like today's Lua driver, on the host language doing the looping around a traced
single-step graph).

**Evidence**: `tools/loom_mil_compiler/recurrent.py` lines 38–49 (the GRU `while_loop` finding);
`exporter.py` lines 1804/1815 (`while_loop`/`cond` handling already present, currently mostly unused).

### 4. Generalize a second family template for iterative-refinement models

**What**: Alongside `ModularExportSpec` (which generalizes decoder-LLM export to ~3 lines of
declarative boundary spec per model, per `modular_export.py`), build an analogous spec/template for
the "N-step iterative refinement over loop-carried state" family — parameterized by step count,
loop-carried tensor name(s), and the per-step submodule to call. This depends on item 3 above (the loop
needs to be capturable as a structural unit first).

**Why**: StyleTTS2, Matcha, and Supertonic aren't three unrelated bespoke problems — they're one
recognizable pattern with three different implementations. Right now each gets a fully hand-written
driver (`export_styletts2_mil.py`: 340 lines, `export_kokoro_mil.py`: 503 lines). A shared template
would shrink these the same way `ModularExportSpec` shrunk `export_qwen3_mil.py` to 28 lines, while
still conceding true one-offs (custom vocoders, spline ops) as bespoke.

**Evidence**: `tools/loom_mil_compiler/modular_export.py`'s `ModularExportSpec`
(`prefix_attr`/`repeated_attr`/`suffix_attrs`/`aux_attr`) is the existing precedent for this pattern.
This item is also a natural continuation of `BACKLOG.md`'s already-tracked, deliberately-deferred
"Phase 2 (fully automatic prefix/suffix boundary discovery)" note under "Modular-export blueprint" —
worth revisiting together with this proposal rather than as a separate effort.

### 5. (Low priority, optional) Empirically prototype StableHLO on one already-solved model

**What**: If the IR question ever needs to be reopened with harder evidence than architectural
reasoning, prototype the `torch.export` → `torch_xla.stablehlo` path on one simple, already-solved model
(e.g. Conformer-CTC) and literally count the resulting decision points/pattern-matches needed to map
back to `ggml`, compared to the current MIL path for the same model.

**Why**: Turns the item-0 rejection above from an architectural argument into a measured one, in case
priorities change later. Not recommended as near-term work — it's a validation exercise, not a fix for
either of the two real problems above.
