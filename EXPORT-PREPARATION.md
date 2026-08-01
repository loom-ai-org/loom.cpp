# Export Preparation — what to settle before P4

This document is the working record of a review session (2026-07-31) that started from one question:
*could the exporter be reorganized to emulate `optimum-onnx`, so that a single entry point exports any
causal LM, any ASR model, any TTS model?*

The short answer is that the `optimum` port is largely already done — it is what P3 and P4.0 built — and
that the thing standing between the current state and broad model coverage is not the export-config
hierarchy at all. It is **who writes the Lua driver**, and the absence of a common protocol for declaring
and composing one.

Relationship to the other documents:

* [`BACKLOG.md`](BACKLOG.md) — the live backlog. Its "Implementation sequence for the roadmap" table is
  the authority on phase ordering. This document proposes **five items that belong in P4.0** (the phase
  whose stated purpose is already "settle these before the first from-scratch family config") and does
  not change anything already recorded there.
* [`EXPORT-ROADMAP.md`](EXPORT-ROADMAP.md) — the R1–R7 thread. This document closes out one piece of R3's
  residue ("driver templates as first-class artifacts", currently filed under *What P3 deliberately did
  not build*) by specifying it, and argues it should be promoted from residue to the P4 critical path.
* [`BACKEND.md`](BACKEND.md) — the exporter's working log. Its closing conclusion (per-family templates,
  not universal orchestration inference) is assumed here, not revisited.

Everything below that cites a file or a count was checked against the tree during the session; where a
claim is an inference rather than a measurement, it says so.

---

## 1. Findings

### 1.1 The `optimum` analogy is weaker than its reputation, in a way that matters

Measured against `/home/flavio/Dev/optimum-onnx` at the revision reviewed:

* **`ORTModelForTextToSpeech` does not exist.** TTS is the task `text-to-audio`, registered for exactly
  three model types — `musicgen`, `speecht5`, `vits` (`optimum/exporters/onnx/model_configs.py:2055,
  2248, 2362`). There is no ORT runtime class for it; it is reached through a `TextToAudioPipeline`.
* **`ORTModelForSpeechToText` does not exist either.** There is `ORTModelForSpeechSeq2Seq` and
  `ORTModelForCTC`.
* ASR coverage is ~16 model types (counted from `register_tasks_manager_onnx` decorators across
  `automatic-speech-recognition`, `-with-past`, and the CTC/audio families).

CrispASR ships 53 ASR backends and 51 TTS engines. So "emulate `optimum` and get any TTS model" is not a
thing `optimum` does; its TTS coverage is roughly family 7 (VITS/piper) plus SpeechT5, and its speech
handling is *less* principled than `multi_phase_export.py` — `SpeechT5OnnxConfig` hand-declares a
four-subgraph decomposition in a docstring (`model_configs.py:2269`).

**Where its generality does come from**, and this is the part worth copying:

1. **`transformers` is the normalizer, not `optimum`.** `AutoModelForCausalLM` guarantees a uniform
   forward contract and `NormalizedConfig` maps heterogeneous config fields onto
   `num_layers`/`num_heads`/`hidden_size`. That is why `Qwen3OnnxConfig` is three lines and the whole
   `model_configs.py` averages ~17 lines per config across 169 classes.
2. **ONNX carries no orchestration.** The artifact is a graph; the loop lives in `GenerationMixin` and
   the `ORTModelFor*` classes, written once per task. `ORTModelForCTC.forward` is ~40 lines and covers
   every CTC model because the *task* fixes the I/O contract, not because the architectures are alike.
3. **Task = I/O contract**, not architecture.

And where those enablers run out, `optimum` looks exactly like what this project's own scripts used to
look like: `ORTModelForWhisper` and `ORTModelForMoonshine` are per-model runtime subclasses
(`optimum/onnxruntime/modeling_seq2seq.py:1368,1390`), and `_MODEL_PATCHER` is a per-model escape hatch.

**Conclusion.** Copy the object model, not the coverage claim. Enabler 1 has no analogue for the TTS zoo
and never will. Enabler 2 is the one this project deliberately inverted (see §1.3), which is exactly why
the driver is the load-bearing artifact here and why it needs the protocol §2 describes.

### 1.2 The export-side port is substantially done

| `optimum` concept | Loom today | state |
|---|---|---|
| `OnnxConfig` | `LoomExportConfig` (`export_config.py`) | done |
| `TasksManager` | `TaskRegistry` (`registry.py`) | done |
| `optimum-cli export onnx` / `main_export()` | `loom-export` / `main_export()` (`main_export.py`) | done |
| `OnnxConfig.inputs` (named axes) | `ExportPhase.root_axis`/`declared_axes` + `_validate_input_axes` | done, per-phase by decision (P4.0.2) |
| submodel decomposition | `Decomposition` = `Flattened`/`Modular`/`MultiPhase` (`decomposition.py`) | done, and cleaner than `optimum`'s |
| `ModelPatcher.patch_model_for_export` | `ModelPatcher.prepare_environment()` (`patcher.py`) | half — import-order stubs only |
| `NormalizedConfig` | — | absent |
| `DummyInputGenerator` | — | absent (hand-written per phase) |
| `ORTModelFor*` | — | absent by decision; see §1.3 |

So the premise that the exporter is "far less organized and not harmonized" than `optimum` is, on the
export side, no longer true. What is genuinely unharmonized is narrower and is itemized in §1.4–§1.6.

### 1.3 The runtime is deliberately lean; the per-model C++ drivers are legacy

Stated by the author during this session and recorded here because the tree does not currently say it
anywhere: the execution engine stays as small as possible (the target is edge devices, explicitly
positioned against onnxruntime's footprint) and **all complexity is pushed into the exporter, because
adding a new model family is far cheaper in a Python library than in specialized C++.** That is the
project's main selling point. `src/core/{kokoro,vits,matcha,styletts2,supertonic,whisper}_driver.cpp`
predate the Lua drivers becoming the orchestration device and are slated for removal.

The thesis is already demonstrated, not merely intended:

* **The entire engine contract for a MIL-exported model is four headers.** `src/core/lua_bridge.cpp`
  includes only `graph_builder.h`, `kv_cache.h`, `duration_aligner.h`, `relative_position.h`.
* **The Lua-visible surface is ~12 bindings** (`run_subgraph`, `run_recurrent`, `get_weight`,
  `argmax_row`, `causal_mask`, `range`, `zero_mask`, `gaussian_array`, `uniform_array`, `seed_rng`,
  `expand_by_duration`, `pad_crop_relative_embeddings`) plus the `n_tokens`/`n_past` axis keys.
* **The cases predicted to be hardest already moved out of C++.** `cfm_euler_sampler.h`,
  `ode_stepper.h`, `style_diffusion_sampler.h`, `bilstm_stepper.h`, `generation.h`, `ctc_decode.h` and
  `tdt_decoder.h` are unreachable from `lua_bridge.cpp`: the CFM Euler loop, the ADPM2 diffusion
  sampler and BiLSTM stepping all run as Lua today. If ADPM2 did not need C++, essentially no
  orchestration shape will.

Two qualifications worth keeping honest:

* `expand_by_duration` and `pad_crop_relative_embeddings` are **family-specific primitives in the
  bridge** (VITS/Matcha/Kokoro duration expansion; VITS relative-position cropping). By the stated
  criterion these are model adaptation living in C++. They may be defensible as *task* primitives, the
  same category as a tokenizer or CTC decode — but that should be a recorded decision, not an accident.
* Vocoders are already on the exporter side: HiFi-GAN and iSTFT are traced graph phases in families
  7/8/9 and `tools/loom_mil_compiler/istft.py` exists. The "tokenizers and vocoders" exception is
  really just tokenizers.

`include/loom/loom.h` is the artifact keeping the legacy drivers load-bearing: lines 14–24 re-export all
six from the umbrella public header, which is why every test transitively depends on them and why a
naive grep for consumers reports none.

### 1.4 The real blocker: driver authorship has two regimes

| regime | models | hand-written Lua | gets `driver_ir` validation |
|---|---|---|---|
| **synthesized** — `apply_monolithic_export` (`exporter.py:1174`), `apply_modular_export` (`:1236`) | Qwen3, LFM2 ×2, Conformer-CTC, Parakeet ×2 | none | yes |
| **hand-written `.lua` + marker substitution** — `flow_matching_export.render_driver` (`:193`) into `BaseMultiPhaseModelExportConfig.driver_script_path` | Kokoro, VITS, Matcha, Supertonic, StyleTTS2 | all of it | **no** |

Under the §1.3 architecture, that table *is* the coverage answer: "supports any TTS model" means "the
exporter can synthesize the driver". Today it synthesizes two orchestration shapes, and for seven of
eleven models a human writes the orchestration while the exporter performs a string replacement into it.
Those seven also silently forgo the checks the other four get.

This reframes the coverage problem usefully. Across the ~108 real CrispASR converters the distinct
orchestration shapes are roughly:

| shape | status |
|---|---|
| prefill + argmax | synthesized |
| modular submodule chain | synthesized |
| N-phase feed-forward | hand-written (×5) |
| Euler ODE / CFM | codegen'd (`FlowMatchingSpec`) |
| duration expansion + vocoder | hand-written |
| ADPM2 / Karras diffusion | hand-written |
| cross-attention AR decode + KV cache | **not started** — families 2 + 6, ~16 models |
| AR codec-token loop with delay / RQ patterns | **not started** — family 10, ~20 models |

Eight shapes, not 120 models; two unstarted shapes account for ~36 models.

### 1.5 The builder substrate already exists and is under-used

`tools/loom_mil_compiler/driver_ir.py` (456 lines) is a real IR, not a string helper:

* expressions — `Var`, `Lit`, `BinOp`, `UnaryOp`, `Len`, `FieldAccess`, `Index`, `Call`, `TableLit`,
  `RawExpr`
* statements — `Local`, `LocalDecl`, `Assign`, `SubgraphCall`, `Argmax`, `If`, `While`, `Break`,
  `Return`, `RawBlock`
* `validate()` — every symbol defined before read
* `check_subgraph_calls()` — every `run_subgraph` call cross-checked against the target topology's
  declared inputs and its declared output count
* `LuaCodegen` — the only place that knows Lua's concrete syntax

`RawBlock` matters disproportionately for the migration plan (§3.2): it is an escape hatch for Lua the IR
does not model, which means an existing hand-written driver can be adopted wholesale as one block and
then decomposed incrementally.

The component inventory likewise exists — what is missing is a shelf to put it on:

| component | emits | assembled by |
|---|---|---|
| `FlowMatchingSpec` | Euler-CFM sampler function | marker string substitution |
| `EstimatorSpec` | nothing — validation only | same call site |
| `ModularExportSpec` | prefix → aux → layers → suffix chain | `apply_modular_export`, straight to IR |
| prefill prologue/epilogue (`loom.range`, `loom.causal_mask`, `argmax_row`) | driver preamble/epilogue | `apply_monolithic_export`, inline |
| `recurrent.py` | LSTM/GRU cell topology + Lua stepping loop | ad hoc |
| `ExportPhase` | one traced topology | `MultiPhase.export` |

Six components, four assembly mechanisms, no common calling convention. That heterogeneity — not any
missing capability — is what makes adding a family feel bespoke.

### 1.6 Two smaller findings on the registry

* **Recognition is per-model where it could be per-`model_type`.** The causal-LM family is already
  model-agnostic underneath: one wrapper over plain `AutoModelForCausalLM`, `architecture` inferred from
  `config.model_type` (`causal_lm_export.py:98`), tokenizer family and pretokenizer auto-detected inside
  the exporter (`exporter.py:1861`). The only per-model data in `_build_qwen3`/`_build_lfm2_*` is an
  architecture string, an optional `tokenizer_pre` override and a decomposition choice — all with
  working defaults. Yet adding Llama means hand-writing `_is_llama` + `_build_llama`, because `detect()`
  is a closure comparing one literal.
* **The task vocabulary names mechanisms and libraries, not tasks.** Registered today: `causal-lm`,
  `nemo-asr-encoder`, `tts-multi-phase`, `tts-flow-matching`. Two name a decomposition and one names a
  loader library. P3.2 defined "task = the export shape a family builds" — but P4.0.3 then extracted
  decomposition into its own field, which removes the reason for that conflation. `tts-multi-phase` and
  `tts-flow-matching` are now one task whose members differ by a field.

---

## 2. The design principle: a protocol instructed by data

The disagreement worth resolving before writing any of this is *how a family declares itself*.

**The constraint that rules out plain text specs as the foundation.** Every spec in this tree earns its
existence by being checked against the real model, and the checks are predicates over live objects:
`EncoderOutput.validate` raises if a CTC spec is pointed at an RNNT checkpoint, naming the checkpoint's
own `d_model`; `EstimatorSpec.validate_against_topology` raises naming the exact declared-input
mismatch; `ModularExportSpec`'s attribute paths raise `AttributeError` at build time;
`_validate_input_axes` catches an undeclared second dynamic axis. A YAML/JSON file can carry the field
values but not the predicate, so a naive text-spec front-end would put the declaration in one place and
its validation in another — the same split P4.0.3 spent a commit undoing for `profile`.

**The resolution (author's direction, and it is better than either starting position).** The predicate
does not have to live in a *per-spec* method. Each spec class today hand-writes a bespoke `validate()`,
but the things they check are the same handful of relationship kinds over and over. Lift those into a
shared vocabulary, have every spec class declare its fields *against* that vocabulary, and the
validation becomes generic machinery while the model-specific content stays pure data:

> **All spec'ing classes validate and check generally analogous dependencies, links and
> correspondences; the specificities are informed as data.**

The checkable link kinds, generalized from the validators that already exist:

| link kind | means | generalized from |
|---|---|---|
| `TopologyName` | must exist among the exported topologies | `flow_matching_export._check` |
| `TopologyInput(topology)` | must be a declared input of that topology | `EstimatorSpec.validate_against_topology` |
| `TopologyOutputArity(topology)` | must match the topology's declared output count | `driver_ir.check_subgraph_calls` |
| `ModuleAttrPath` | must resolve on the loaded `nn.Module` | `ModularExportSpec` |
| `Axis` | must be in the axis vocabulary and declared by the phase | `axes.py` + `_validate_input_axes` |
| `ConfigDerived(reader, measured)` | a claim about the checkpoint that must equal a measured property | `EncoderOutput.expected_channels` |
| `WeightName` | must exist in the merged weight dict | `merge_phase_weights` |
| `DriverSymbol` | defined before read in the emitted driver | `driver_ir.validate` |

A spec class then declares `field → link kind` once; the protocol walks it. Three consequences:

1. **New families get validation for free**, in the shape reviewers already trust, instead of each
   author reinventing a `validate()` of variable quality.
2. **A text front-end becomes free later rather than duplicative** — the values are data, the predicates
   are library-side, so `asdict`/`fromdict` plus a schema is all a JSON/YAML/TOML path needs. It is
   deliberately deferred until families 2, 6, 10 and 11 are written, so the schema is not frozen against
   four unknown shapes.
3. **It composes with the driver builder (§3.2)** — a component's declared links are checked before it
   emits anything.

**Acceptance test for the protocol, stated up front:** each of the four existing bespoke validators must
be re-expressible as link-kind declarations *with no loss of error-message quality*. This tree's errors
name the offending input, the expected channel count and its config source; a generic checker that
degrades those into "validation failed" is a regression, not a refactor.

**Standing rule to adopt with it:** every spec field must be either checkable against the real
model/topology, or explicitly documented as unchecked. That is what makes a component trustworthy
enough to reuse without reading its implementation, which is the real requirement behind "marketplace".

---

## 3. Proposed items — P4.0.4 … P4.0.8

P4.0's stated purpose is "settle these before the first from-scratch family config", and P4.1 (Whisper),
P4.2 (GigaAM) and P4.3 (composition) are exactly those from-scratch configs. All five items below get
cheaper now and more expensive after three more families exist. Same gates as everything else:
`snapshot_gguf.py` for changes that must not alter output, `compare_snapshots.py` for deliberate shape
rewrites, per-model reference tests for anything numerical.

**The embedded driver source is a GGUF KV (`model.driver_script`), so `snapshot_gguf.py` gates driver
refactors byte-for-byte** — P3.3's own snapshot caught a driver-text change down to one renamed
identifier inside a comment. Every item below is therefore provable rather than argued.

### P4.0.4 — task vocabulary and generic recognition

*Independent, smallest, and it removes surface the later items would have to preserve — the same logic
that put P0 first.*

* Rename tasks to real tasks: `causal-lm` → `text-generation`, `nemo-asr-encoder` →
  `automatic-speech-recognition` (with the loader per recognizer, which is what P4.2/GigaAM forces
  anyway), `tts-multi-phase` + `tts-flow-matching` → `text-to-speech`, plus `audio-codec` reserved
  (decision 3) — declared in the vocabulary, with no family registered against it until family 11 exists.
* Add a generic causal-LM recognizer matching any HF directory with a `model_type`, with a per-`model_type`
  override table for the exceptions, replacing the one-literal closures.
* **Gate:** byte-identical re-export of all 11 models; `loom-export` resolves the same recognizer for
  every real checkpoint on this machine as it does today, plus at least one HF causal LM that no
  recognizer currently claims.

### P4.0.5 — the spec protocol (§2)

* Link-kind vocabulary + the declaration protocol + the generic checker.
* Retrofit the four existing validators onto it (`EncoderOutput`, `EstimatorSpec`/`FlowMatchingSpec`,
  `ModularExportSpec`, `_validate_input_axes`).
* **Gate:** every existing validator's error messages preserved verbatim where they name specifics
  (assert on message content, not just on the raise); byte-identical re-export of all 11 models, since
  no emission path is touched.

### P4.0.6 — `DriverBuilder` + `DriverComponent` over `driver_ir`

The graph side already has this shape; the driver side does not:

```
Decomposition : how the model becomes topologies
DriverBuilder : how those topologies become a driver
```

```python
class DriverComponent(Protocol):
    def links(self) -> list[Link]: ...            # checked by the P4.0.5 protocol
    def emit(self, ctx: DriverContext) -> list[Stmt]: ...

class DriverBuilder:                              # one per family, or per orchestration shape
    def components(self) -> list[DriverComponent]: ...
    def build(self, ctx) -> IRFunction            # check links, emit, validate(), check_subgraph_calls()
```

`DriverContext` is the topologies dict + declared axes + weights — what `MultiPhase.export` already
assembles before calling `render_driver`.

**Migration is incremental, not big-bang.** A family moves onto the builder by wrapping its current
hand-written `.lua` in a single `RawBlock`, immediately gaining `check_subgraph_calls()` on everything
around it, and then peeling blocks out into real components one at a time. Suggested order: Matcha
(smallest with a real generated sampler) → Supertonic (same shape, validates reuse) → VITS → Kokoro →
StyleTTS2 (hardest, ADPM2).

* **Gate:** byte-identical `model.driver_script` at every step, per model.

### P4.0.7 — the component registry ("marketplace")

Extract the six existing components (§1.5) onto the one `DriverComponent` calling convention and
register them by name. Nothing new is written here; this is the item that turns an inventory into a
shelf, and it is what makes P4.1/P4.3 able to *reuse* rather than restate.

* **Gate:** all 11 models re-exported byte-identically through registered components.

### P4.0.8 — legacy C++ driver retirement policy

R6's retirement policy currently covers `tools/convert_*` only. Extend it, with the precondition
spelled out rather than rediscovered:

* Same rule — a driver may be deleted only in the commit that re-points the last test consuming it.
* **The precondition that is not obvious:** the pre-MIL C++ oracle tests (`test_e2e_kokoro_driver.cpp`,
  `test_e2e_styletts2_driver.cpp`, …) are the *numeric ground truth* several MIL/Lua tests were
  validated against. Retiring a driver means its Lua test must first carry its own reference fixture
  instead of comparing to the C++ oracle. That is the real cost and the actual reason all six are still
  alive.
* Split `include/loom/loom.h` into the lean runtime surface and a legacy header, so the boundary is
  auditable and new code stops accreting against the drivers.
* ~~Decide whether `expand_by_duration` / `pad_crop_relative_embeddings` stay~~ — **decided (§5.1): they
  stay, reclassified as generic host-side tensor ops.** This bullet becomes a documentation task:
  `lua_bridge.h` gains the criterion a new binding must meet, with both labelled against it.

*This one trails the others — it is bookkeeping plus test work, and nothing in P4 depends on it.*

---

## 4. Deliberately not in this roadmap

* **`LoomModelFor*` runtime entry points.** BACKLOG.md's existing argument stands: a GGUF is
  self-contained via its embedded driver, so the `ORTModelFor*` half is less load-bearing here than in
  `optimum`, and it should wait for a second consumer besides the tests. Note also that what the
  question "should `LoomModelForCausalLM` export any causal LM?" actually asks for is the *export* side,
  which is `loom-export --task text-generation` and therefore P4.0.4.
* **KV cache in the exported causal-LM graph.** Both synthesized drivers bind `n_past = Lit(0)` and
  `loom.causal_mask(n_tokens, 0)` (`exporter.py:1225,1238,1307,1312`): prefill-only, argmax the last row.
  The engine has a KV cache and the bespoke drivers use it, but the MIL causal-LM path neither declares
  nor uses one, so "export any causal LM" today means one prefill, not a generation loop. `optimum`
  spends most of `TextDecoderOnnxConfig` on exactly this (`use_past` / `decoder_with_past` / merged
  decoder). **This is a real gap and a real capability item — it belongs in P4/P5, not in preparation**,
  and it is recorded here so it is not mistaken for something P4.0 covers.

  **Where the blocker actually is (measured 2026-08-01, and it is not where this bullet implied).**
  Binding a real `n_past` is the *last* step, not the first. The engine's cache is reachable through
  exactly one door — the **`ATTENTION` topology node**. `op_attention` is what reads `n_past`/`n_kv` from
  `SymbolEnv`, appends this step's K/V at cells `[n_past, n_past+n_tokens)` and reads back `[0, n_kv)`
  (`src/ops/primitives_attention.cpp:29-80`; `GraphBuilder` derives `n_kv = n_tokens + n_past` at
  `graph_builder.cpp:129`). There is no other seam: a topology with no `ATTENTION` node cannot touch the
  cache, and one that declares `kv_cache=true` without a cache raises (`primitives_attention.cpp:51`).

  The bespoke converters have that node because **a human writes it**, layer index and all —
  `convert_qwen3.py:93`: `{"op": "ATTENTION", "inputs": ["q","k","v","kq_mask"], "attrs": {"layer":
  "{i}", "scale": "1/sqrt($n_embd_head_k)"}}`. **A MIL-exported causal LM has zero `ATTENTION` nodes.**
  Qwen3's exported topology is `ADD 450, MUL 450, MUL_MAT 254, RESHAPE 228, PERMUTE 142, VIEW 140,
  POW/REDUCE_SUM/SCALE/RSQRT 113 each, CONCAT 57, REPEAT 56` — attention arrives fully expanded. The
  exporter does map `loom_fused_attention` → `ATTENTION` (`exporter.py:131`) and `dialect.py` does
  register a `FuseLoomAttention` pass — but its `_fuse_blocks` body is `pass`, a documented placeholder
  (`dialect.py:259`). The op is registered and never produced.

  So the item is four pieces, in order, and only the fourth is what this bullet described:
  1. **Implement `FuseLoomAttention`** — pattern-match the SDPA subgraph in MIL per layer (post-RoPE
     q/k/v, mask, softmax, the GQA `REPEAT` that `fuse_gqa_repeat_kv` already normalizes), assign the
     `layer` index, emit `loom_fused_attention`. Architecture-sensitive: Qwen3 has qk-norm, LFM2
     interleaves conv layers, Llama is plain GQA. Each needs numeric verification against the expanded
     path, which is what makes this the expensive piece.
  2. **Trace with past-KV semantics**, so a decode step computes K/V for the new token only. This is
     precisely `use_past`/`decoder_with_past`.
  3. **Change the exported input contract** — `tokens` at `n_tokens=1`, `cache_position` =
     `range(n_past, n_past+n_tokens)`, mask `[1,1,n_tokens,n_kv]`. That makes `n_kv` a second dynamic
     axis and lands directly on P4.0.2's `_validate_input_axes`.
  4. **Synthesize a two-phase driver** (prefill, then a decode loop) instead of one prefill+argmax. The
     driver-side machinery for this already exists — `transpile_operation` emits
     `loom.causal_mask(n_tokens, n_past)` with real variables (`exporter.py:1504-1507`), and `driver_ir`
     has `While`/`Break`; it is step 1 that has no code at all.

  Consequence for decision 2 below: the new decomposition inherits this gap, but it cannot start with
  it. **Step 1 is a prerequisite, and it is an R2-shaped MIL pass, not driver work.**
* **A JSON/YAML/TOML spec front-end.** Agreed in principle, deferred by decision (§2, consequence 2)
  until the protocol is stable and families 2/6/10/11 have shown their shapes.
* **`NormalizedConfig` / `DummyInputGenerator` analogues.** Worth having eventually; neither blocks P4,
  and for the TTS zoo there is no upstream normalizer to build a `NormalizedConfig` on top of.

## 5. Decisions (resolved 2026-08-01)

The three open questions above, plus one on verification budget, were answered by the author. Each
changes something concrete in §3, noted with the answer.

1. **`expand_by_duration` / `pad_crop_relative_embeddings` stay in the bridge, reclassified as generic
   host-side tensor ops.** Neither reads a model config; both exist because the operation has a
   data-dependent output length, which cannot live in a static topology. **Consequence:** P4.0.8's third
   bullet becomes documentation rather than code — `lua_bridge.h` gains a written criterion that a new
   binding must be a generic host-side tensor op, not model adaptation, and the two existing ones are
   labelled against it.
2. **The cross-attention AR decode shape gets its own `Decomposition`**, not a `DriverComponent`.
   **Consequence, and it reaches back into P4.0.6:** the driver builder must be *selected by* the
   decomposition rather than owned by the family, so a fourth decomposition can bring its own builder
   without reopening the component API. P4.0.6 therefore adds a `Decomposition.driver_builder(config)`
   hook, and the component API is shaped by the six existing components only — no speculative
   loop-carried-state design. The KV-cache gap in §4 becomes that decomposition's problem, in P4.1 —
   **but see §4's measured note: it is blocked on a MIL attention-fusion pass that does not exist yet,
   and no amount of driver work substitutes for it.**
3. **The TTS task splits now: `text-to-speech` + `audio-codec`.** **Consequence:** `audio-codec` is a
   *reserved* name with no family registered against it yet, which is only meaningful if the vocabulary
   is a real, checked list — so P4.0.4 grows a `tasks.py` declaring the canonical names (and each task's
   base config class), with `TaskRegistry.register()` validating against it.
4. **Verification budget: affected models per commit, full 11-model sweep per completed item.** Each
   step below states which models it can possibly touch; the item-closing sweep is the cross-family net.

---

## 6. Implementation plan

Five stages, in the order they must happen. Each numbered step is one commit. "Touches" names the models
a step can possibly affect, which is its per-commit gate; every stage ends with a full 11-model
`snapshot_gguf.py` sweep before the next begins.

**Two standing practicalities**, both learned the expensive way and recorded in BACKLOG.md P4.0.1: run
the export tests with `TMPDIR=` *and* pytest's `--basetemp=` pointed under `/home/flavio/.claude/tmp/`
(`TMPDIR` alone does not move `tmp_path`, and the real exports fill `/tmp`'s 28 GB partition); and take
the pre-change snapshot baseline from a `git worktree` at the current commit rather than trusting the
`.gguf` files in the tree, which are gitignored build outputs and routinely stale.

### Stage 0 — record the plan (1 commit)

**0.1** Add the P4.0.4–P4.0.8 rows to BACKLOG.md's P4.0 section and a pointer to this document beside the
existing `EXPORT-ROADMAP.md` pointer. Docs only. *Touches: nothing.*

### Stage A — P4.0.4: task vocabulary and generic recognition

*First because it is the only stage that removes surface the others would have to preserve.*

**A.1 — `tasks.py`: the canonical vocabulary.** Four names (`text-generation`,
`automatic-speech-recognition`, `text-to-speech`, `audio-codec`), each with a docstring saying what
export shape it covers and which base `LoomExportConfig` it builds. `TaskRegistry.register()` validates
`entry.task` against the list, raising with the known names.

*Design note this forces, and it is a real improvement:* merging `tts-multi-phase` +
`tts-flow-matching` into one task will trip `register()`'s existing "two families disagree on
config_class" guard, because `TTSFlowMatchingModelExportConfig` is a *subclass* of
`BaseMultiPhaseModelExportConfig`, not the same class. Relax the check from identity to
`issubclass(entry.config_class, task_base_config)` with the base declared in `tasks.py`. That is
strictly better than what is there now: today the first family to register defines the task's class by
accident of import order.

Tests: unknown task name raises naming the vocabulary; a subclass registers cleanly; an unrelated class
still raises. *Touches: nothing (no rename yet).*

**A.2 — rename the four task strings.** `causal-lm` → `text-generation`; `nemo-asr-encoder` →
`automatic-speech-recognition`; `tts-multi-phase` + `tts-flow-matching` → `text-to-speech`. No
backwards-compatible aliases: the task name is a CLI argument, not a stored artifact, and carrying two
spellings would re-create exactly the two-names-for-one-thing problem P4.0.3 removed.

**First action of this step is a check, not an edit:** confirm the task string reaches no GGUF KV. If it
does, this step's gate becomes a snapshot diff instead of a pytest run. *Touches: nothing if the check
holds; all 11 if it does not.*

**A.3 — recognizer specificity.** `ModelRecognizer` gains `fallback: bool = False`; `TaskRegistry.detect`
prefers non-fallback matches and consults fallbacks only when no specific recognizer matched. Without
this, A.4's generic recognizer would make every Qwen3 and LFM2 detection ambiguous.

Tests: a specific match wins over a fallback; two specific matches still raise (LFM2's deliberate
ambiguity is preserved); a fallback-only match resolves. *Touches: nothing.*

**A.4 — the generic causal-LM recognizer.** Matches any HF directory whose `config.json` declares a
`model_type` *and* an `architectures` entry ending in `ForCausalLM` — the architectures check is what
keeps it from claiming every HF directory on disk, including the ASR and TTS ones. Registered as
`fallback=True` under `text-generation`, building an `LMCausalModelExportConfig` with
`architecture=None` (inferred in `load_model`), `tokenizer_pre=None` (auto-detected in the exporter) and
`Flattened()`. A `_MODEL_TYPE_OVERRIDES` table carries the exceptions; `qwen3` and both `lfm2` entries
stay as specific recognizers because LFM2's two decompositions genuinely cannot be inferred.

Tests: synthetic HF fixtures — a Llama-shaped dir resolves to the fallback; a Qwen3 dir still resolves
to `qwen3`; an LFM2 dir still raises the two-way ambiguity; a directory with `model_type` but no
`ForCausalLM` architecture does not match.

*Touches: Qwen3, LFM2 ×2 (byte-identical re-export required). If a non-Qwen3/LFM2 HF causal LM is
available locally, export it end-to-end as the real acceptance test — the whole point of this step is a
model that could not be exported before. If none is available, say so in the commit message rather than
claiming coverage the synthetic fixtures do not give.*

**A.5 — stage gate.** Full 11-model sweep; `loom-export` auto-detection re-run against every real
checkpoint on this machine, resolving to the same recognizer as before the stage.

### Stage B — P4.0.5: the spec protocol

*Before the builder, because the builder's components are the first specs that would otherwise be
written against nothing.*

**B.1 — `spec_protocol.py`.** The eight `Link` kinds from §2, each with a `check(ctx)` and a message
template lifted from the validator it generalizes. `LinkCheckContext` carries the topologies dict, the
loaded model, the merged weights and the declared axes — all optional, because they become available at
different times.

**The one design detail that must not be skipped:** a link whose context is never populated must be
*reported*, not silently skipped. The checker returns a deferred-links list and the export raises at the
end if any remain unchecked. Otherwise "validated" quietly comes to mean "validated where convenient",
which is the failure mode this whole protocol exists to prevent. *Touches: nothing.*

**B.2 — retrofit `EstimatorSpec` / `FlowMatchingSpec`.** Smallest and already declarative; it is the
shape the other three get compared against. *Touches: Matcha, Supertonic.*

**B.3 — retrofit `EncoderOutput`.** The richest messages in the tree (they name the checkpoint's own
`d_model` and the config field it came from), so this is the real test of §2's acceptance criterion.
Tests assert on message *content*, not just that a `ValueError` was raised. *Touches: Conformer-CTC,
Parakeet ×2.*

**B.4 — retrofit `ModularExportSpec`.** `ModuleAttrPath` links replacing the incidental `AttributeError`.
Note this upgrades the behaviour: today a wrong path raises whatever Python raises, at whatever point the
traversal reaches it; a link check raises up front naming the path and the module type. Assert the new
message, and keep the old failure mode's timing documented. *Touches: LFM2-modular.*

**B.5 — retrofit `_validate_input_axes`.** `Axis` links on `ExportPhase.root_axis`/`declared_axes`. The
existing checks in `LoomGGUFExporter` stay where they are (they operate on the traced program, not on a
spec); what moves is the *declaration* side. Keep P4.0.2's two raises and their messages intact.
*Touches: Kokoro, Conformer-CTC, Parakeet ×2 (the non-default-axis models).*

**B.6 — enforce the standing rule.** A test that walks every registered spec class and fails on any
field that is neither link-declared nor explicitly marked unchecked. This is what makes it a protocol
instead of a convention, and it is cheap now and expensive after four more families. *Touches: nothing.*

**B.7 — stage gate.** Full 11-model sweep (no emission path touched, so byte-identical throughout) plus
the message-content assertions from B.2–B.5.

### Stage C — P4.0.6: `DriverBuilder` + `DriverComponent`

**C.1 — `driver_builder.py`.** `DriverContext`, the `DriverComponent` protocol (`links()` + `emit(ctx)`),
and `DriverBuilder.build()` = check links → emit → `driver_ir.validate()` → `check_subgraph_calls()` →
`LuaCodegen`. Plus, per decision 2, a `Decomposition.driver_builder(config)` hook so the builder is
selected by the decomposition. *Touches: nothing.*

**C.2 — retrofit the two synthesized paths.** `apply_monolithic_export` → a `PrefillArgmaxBuilder`;
`apply_modular_export` → a `ModularChainBuilder`. Deliberately first: these already build `IRFunction`s,
so the API is proven against working code before the harder migration, and the gate is unambiguous.
*Touches: Qwen3, LFM2 ×2, Conformer-CTC, Parakeet ×2 — byte-identical driver text required.*

**C.3 — route `MultiPhase` through the builder with one `RawBlock`.** The existing `.lua` (post
`render_driver`) becomes a single raw block inside a built `IRFunction`. No semantic change; the five TTS
drivers gain `check_subgraph_calls()` for the first time.

*Expect this step to find real bugs*, and treat that as success rather than a blocker: five hand-written
drivers have never had their `run_subgraph` calls cross-checked against their topologies' declared
inputs. `LuaCodegen` must emit raw-block text verbatim, with no reindentation, or the gate fails for a
cosmetic reason. *Touches: Kokoro, VITS, Matcha, Supertonic, StyleTTS2 — byte-identical driver text
required.*

**C.4–C.8 — peel one family per commit**, in order: Matcha (smallest with a real generated sampler) →
Supertonic (same shape; proves reuse rather than re-derivation) → VITS → Kokoro → StyleTTS2 (ADPM2,
hardest, and the one most likely to stay partly raw).

**The gate changes here and the plan should not pretend otherwise.** Once a block is emitted from IR
instead of pasted, the driver text will differ — comment placement, spacing, local naming. Byte-identity
is achievable for C.3 and not for C.4–C.8. The gate for each peeling commit is therefore: (a) the model's
existing MIL Lua-driver e2e test passes unchanged — all five have one (`test_e2e_{matcha,supertonic,
vits,kokoro,styletts2}_mil_lua_driver.cpp`); (b) the driver-text diff is read and attached to the commit
message; (c) every topology, weight and non-driver KV is byte-identical. That is the same discipline
`compare_snapshots.py` applies to shape attributes, applied by hand because there is no equivalence
checker for Lua.

**C.9 — stage gate.** Full 11-model sweep; all five TTS e2e Lua-driver tests plus the six synthesized
models' numeric reference tests.

### Stage D — P4.0.7: the component registry

**D.1 — `driver_components/`**: name → component registry, and the six existing components moved onto
the one calling convention. No new capability. *Touches: all 11 — byte-identical driver text required,
since this is a pure re-homing.*

**D.2 — `recurrent.py`'s LSTM/GRU stepping loop becomes a registered component.** It is the one
inventory item that is ad hoc today rather than merely differently-shaped. *Touches: Kokoro, StyleTTS2.*

**D.3 — the catalogue.** One documented table: per component, its links, what it emits, and which models
use it. This is the artifact that makes P4.1/P4.3 able to reuse rather than restate, and it is the
deliverable of this stage more than the code is. *Touches: nothing.*

**D.4 — stage gate.** Full 11-model sweep.

### Stage E — P4.0.8: legacy retirement (trails; nothing in P4 depends on it)

**E.1 — write the bridge criterion.** Per decision 1: `lua_bridge.h` gains the rule that a binding must
be a generic host-side tensor op rather than model adaptation, with `expand_by_duration` and
`pad_crop_relative_embeddings` labelled against it. Docs only. *Touches: nothing.*

**E.2 — split `include/loom/loom.h`** into the lean runtime surface and a `loom_legacy.h`; tests that use
the legacy drivers include it explicitly. Pure include hygiene, and it is what makes the remaining
dependencies visible instead of transitive. *Touches: nothing (C++ only, no export path).*

**E.3 — retire one driver per commit**, each following R6's rule: give the model's Lua test a
self-contained reference fixture first, *then* delete the C++ driver, its oracle test and its
`loom_legacy.h` entry.

**Order and one blocker:** VITS, Matcha, Supertonic, Kokoro, StyleTTS2 are all retirable now. **Whisper
is not** — `whisper_driver.cpp` has no MIL export to replace it; it is blocked on P4.1 and must be
stated as such rather than attempted here.

**E.4 — stage gate.** Full `ctest` green with five drivers deleted, and **record the engine binary size
before and after in the commit message.** Leanness is the stated goal of the architecture; measuring it
is how the goal stops being a slogan.

### Sequencing rationale, in one line each

* **Stage 0 before everything** — so the backlog reflects the plan while it is being worked, not after.
* **A before B** — B's protocol would otherwise have to preserve four task names that are about to
  change, and A.1's `tasks.py` is where a task's base config class gets declared, which B's link checks
  read.
* **B before C** — C's components are the first specs written from scratch; writing them before the
  protocol exists means retrofitting them immediately afterwards.
* **C.2 before C.3** — prove the builder API on code that already builds IR before migrating code that
  does not.
* **C.3 before C.4–C.8** — one mechanical, byte-identical step establishes the route; the semantic work
  is then per-family and independently revertable.
* **D after C** — a registry of components that do not yet share a calling convention is a directory,
  not a shelf.
* **E last** — it is test work and bookkeeping, it blocks nothing, and one of its six targets is blocked
  on P4.1 anyway.
