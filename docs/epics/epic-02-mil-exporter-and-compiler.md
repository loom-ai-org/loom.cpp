---
type: epic
status: active
domain: exporter
last_updated: 2026-08-22
---

# Epic-02: The MIL Exporter and Compiler

## 1. Context and Scope

`loom-exporter` turns a PyTorch checkpoint into a GGUF that the engine runs. It is where **all**
per-model complexity lives, by standing rule
([ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md)). Adding a model family should be a
change here and nowhere else.

In scope: tracing, MIL→topology lowering, driver generation, weight packing and quantization, the
export config/registry API, and the verification sweeps that gate all of it.

## 2. Architectural Overview

The pipeline, front to back:

1. **Recognise.** `TaskRegistry` maps a checkpoint to a config via per-family `detect()`, reading the
   checkpoint's own config rather than its path. Entry point: `loom-export` / `main_export()`.
2. **Patch.** `ModelPatcher.prepare_environment()` applies each family's import-order stubs as a named
   hook. Class-level monkeypatches stay module-level, deliberately — they need the real class imported
   first.
3. **Trace.** `coremltools` traces the **real upstream module** into MIL
   ([ADR-004](../adrs/adr-004-mil-as-the-single-export-path.md)). Never a re-implementation.
4. **Lower.** MIL→MIL passes first (`fuse_rms_norm`, `lower_pow`, `normalize_matmul`, …), then a
   **declarative rule table** in `topology_ops.py` keyed on `(mil_op_type, guard_predicate)`. Unclaimed
   ops fall through to a generic `OP_MAP` — a deliberate route, not an accident.
   `python3 -m loom_mil_compiler.topology_ops` prints the whole table.
5. **Shape.** Symbolic shapes are **sympy objects**, not concatenated strings (`shape_expr.py`).
   `render()` is the only thing that turns one into text, and it emits exactly `symbol_env.cpp`'s
   grammar or raises. Compose with `as_expr`/`floor_div`; never build a shape attribute with an
   f-string.

   **Two traps that a change here must not lose.** Shape symbols are built as *positive integers*
   (`shape_expr.symbol`), which is the only reason `floor(512*n_tokens/512)` reduces to `n_tokens` at
   all — a bare `sympy.Symbol("n_tokens")` compares unequal to the interned one and silently stops
   cancelling. And `floor` arguments are recombined with `sympy.together` **before printing**, because
   sympy distributes rational coefficients over sums on construction (`floor((n-512)/160)` →
   `floor(n/160 - 16/5)`) and the engine evaluates in `double`, where the distributed form takes three
   roundings inside a floor instead of one.

   `value_facts.py` answers "what is this Var's compile-time value/shape" in one memoized
   place — the memo is load-bearing, not tidiness: without it the shape walk is exponential in encoder
   depth.
6. **Decompose.** A strategy object, not a mode string: `Flattened`, `Modular(spec)`, `MultiPhase`.
7. **Drive.** `DriverBuilder` + `DriverComponent` over `driver_ir` generate and validate the Lua.
8. **Write.** Weights merged with a content-aware dedup-on-match / hard-fail-on-mismatch check;
   repeated tensors aliased by content hash through `loom.tensor_alias.*` KVs.

### Family templates

Per-family templates, not universal orchestration inference, are the direction that works:

| template | families |
|---|---|
| `causal_lm_export.py` | Qwen3, LFM2, SmolLM2, Gemma 3 |
| `modular_export.py` | decoder-LLMs split per layer |
| `flow_matching_export.py` | Euler-CFM samplers (Matcha, Supertonic) |
| `nemo_asr_export.py` | Conformer-CTC, Parakeet-TDT, Parakeet-RNNT |
| `multi_phase_export.py` | the shared TTS/multi-phase tail |
| composition template | audio encoder + projector + causal LM |

The third family template's own finding: only **three** of the five differing fields predicted were
real — `ASRModel.restore_from` dispatches on the checkpoint's own config target and returns the
identical concrete class, so the restore class dissolves as a discriminator.

## 3. Verification

**Every exporter change is gated on `snapshot_gguf.py`** — snapshot the exports before and after and
require a zero-line `diff -r`. The `.gguf` files in the tree are `.gitignore`d build outputs and are
routinely stale; regenerate the baseline rather than diffing against them.

When a change is *meant* to rewrite shape attributes, use `compare_snapshots.py` instead: it evaluates
both sides of every differing value at concrete lengths and reports anything not numerically equivalent
as structural.

Record the baseline from a `git worktree` at the merge-base with its own `cwd` and `PYTHONPATH` — see
[Retro-015](../retros/retro-015-export-snapshot-sweeps.md) for why, and for what the sweeps caught.

## 4. Related Decisions and Artifacts

| | |
|---|---|
| Decisions | [ADR-004](../adrs/adr-004-mil-as-the-single-export-path.md), [ADR-005](../adrs/adr-005-export-config-and-task-registry.md), [ADR-006](../adrs/adr-006-model-constants-belong-to-the-export.md) |
| Retros | [Retro-013](../retros/retro-013-retrofitting-eight-bespoke-converters.md), [Retro-016](../retros/retro-016-the-profile-field-was-not-inert.md), [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md), [Retro-002](../retros/retro-002-lstm-gate-stack-computed-twice.md) |
| In `loom-exporter` | `docs/BACKEND.md` (read first if you are touching the exporter), `docs/EXPORT-IMPROVEMENT.md`, `docs/EXPORT-PREPARATION.md`, `docs/EXPORT-ROADMAP.md`, `docs/LOOM_MIL_CONVERSION.md`, `docs/DRIVER-COMPONENTS.md` |
| Active tasks | [Backlog → Exporter](../backlog/active-index.md#exporter--mil-compiler) |

## 5. How the Current Shape Was Reached

The implementation order was fixed rather than arbitrary — each phase either shrank the surface the
next had to preserve, or produced something the next would otherwise have had to guess. Two ordering
constraints still apply to any new phase:

* **Named axes must land before a config schema**, because `LoomExportConfig.inputs` *is* the axis
  declaration. Writing the schema first means migrating every config that exists by then.
* **The registry skeleton must land before any new family.** A family written as a script is one more
  script the registry has to delete; written as a registry entry it is the registry's acceptance test.

| phase | what it settled | where its knowledge went |
|---|---|---|
| **P0** | removed `profile="atomic"`; content-hash weight dedup via `loom.tensor_alias.*` | [ADR-004](../adrs/adr-004-mil-as-the-single-export-path.md) |
| **P1** | named dynamic axes (`DynamicAxes`); canonicalizing MIL→MIL passes | this epic, §2 |
| **P2** | multi-output topologies | [Retro-002](../retros/retro-002-lstm-gate-stack-computed-twice.md) |
| **P3** | `LoomExportConfig` hierarchy, `TaskRegistry`, `ModelPatcher`, `loom-export` | [ADR-005](../adrs/adr-005-export-config-and-task-registry.md) |
| **P4.0** | 16 items: spec protocol, `DriverBuilder`/`driver_ir`, the component registry, the legacy-C++ retirement policy, KV cache on the MIL path, module-owned output buffers, graph persistence, index-tensor KV writes, allocator release | [ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md), [ADR-006](../adrs/adr-006-model-constants-belong-to-the-export.md), [ADR-016](../adrs/adr-016-kv-cache-shape.md), [Retro-003](../retros/retro-003-tdt-decoder-recomputed-its-prediction-network.md), [Retro-004](../retros/retro-004-luajit-array-limit-caps-prefill.md) |
| **P4.0.17** | the NeMo ASR family's own driver builder; CTC and transducer decode leave C++ | [ADR-003](../adrs/adr-003-per-model-complexity-in-the-exporter.md) |
| **retrofits** | all eight bespoke converters replaced and numerically verified | [Retro-013](../retros/retro-013-retrofitting-eight-bespoke-converters.md) |

Full per-commit detail, unmaintained:
[`docs/archive/ledger-2026-07-to-08-exporter.md`](../archive/ledger-2026-07-to-08-exporter.md).
