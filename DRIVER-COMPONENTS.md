# The driver components — what exists, what it checks, who uses it

This is the artifact P4.0.7 exists for (`EXPORT-PREPARATION.md` stage D.3, `BACKLOG.md` P4.0.7). A
driver — the Lua a GGUF embeds as `model.driver_script`, and the only orchestration a Loom model
carries — is assembled from **components**, one per contribution. The point of writing them down is
reuse: P4.1 (Whisper), P4.2 (GigaAM) and P4.3 (composition) are supposed to *reuse* these rather than
restate them, and that only works if a reader can trust a component without reading its implementation.

So the standing rule from `EXPORT-PREPARATION.md` §2 applies to every row below: **every field is either
checkable against the real model/topology, or explicitly documented as unchecked.** The tables say which
is which, and the counts come from the declarations themselves.

**Everything between the generated markers is produced by
`python -m loom_mil_compiler.component_registry` and `lua_library.catalogue()`, from the registry, the
classes' own `__links__`/`__unchecked__` declarations, and the families' real component lists.**
`test_component_registry.py` regenerates and compares, so this file cannot quietly fall out of date.
Do not edit inside the markers; run the generator.

Related reading:

* [`EXPORT-PREPARATION.md`](EXPORT-PREPARATION.md) — why the driver is the load-bearing artifact here
  (§1.3: lean runtime, fat exporter) and the stage plan these components were built under.
* [`BACKLOG.md`](BACKLOG.md) P4.0.5–P4.0.7 — the spec protocol the links belong to, the builder, and
  this registry.
* `tools/loom_mil_compiler/driver_builder.py` — the `DriverComponent`/`DriverBuilder` contract itself.

---

## 1. How a driver is assembled

```
Decomposition : how the model becomes topologies
DriverBuilder : how those topologies become a driver     (selected BY the decomposition)
DriverComponent : one contribution to it
```

`DriverBuilder.build` runs the checks in the one order that makes each meaningful:

```
check links -> emit -> driver_ir.validate() -> check_subgraph_calls() -> DriverSymbol links
```

and, since D.1, looks each component up in the registry as it emits — so a component that ships in a
driver but appears in no catalogue fails the export rather than the review.

Three builders exist:

| builder | shape | models |
|---|---|---|
| `PrefillArgmaxBuilder` | one traced graph, run once over the prompt, argmax the last row | qwen3, lfm2-monolithic, conformer-ctc, parakeet-tdt, parakeet-rnnt, any HF causal LM |
| `ModularChainBuilder` | prefix → [aux] → layer_0..N → suffix_0..M, then the same argmax | lfm2-modular |
| `MultiPhaseDriverBuilder` | a family's own component list (peeled), or one hand-written `.lua` adopted whole | kokoro, matcha, vits, styletts2, supertonic |

## 2. The components

<!-- generated: component catalogue -->

| component | class | emits | links | unchecked | used by |
|---|---|---|---|---|---|
| `driver_inputs` | `DriverInputs` | statements | 0 | 3 | conformer-ctc, hf-causal-lm, lfm2-modular, lfm2-monolithic, qwen3 |
| `monolithic_call` | `MonolithicCall` | statements | 2 | 4 | conformer-ctc, hf-causal-lm, lfm2-monolithic, qwen3 |
| `modular_chain` | `ModularChain` | statements | 0 | 1 | lfm2-modular |
| `prefill_decode_loop` | `PrefillDecodeLoop` | statements | 2 | 7 | hf-causal-lm, lfm2-monolithic, qwen3 |
| `ctc_greedy_epilogue` | `CtcGreedyEpilogue` | statements | 1 | 6 | conformer-ctc |
| `argmax_epilogue` | `ArgmaxEpilogue` | statements | 1 | 3 | hf-causal-lm, lfm2-modular, lfm2-monolithic, qwen3 |
| `export_constants` | `ExportConstants` | statements | 0 | 1 | kokoro, matcha, parakeet-rnnt, parakeet-tdt, supertonic, vits |
| `raw_lua_driver` | `RawLuaDriver` | prelude, statements, postlude | 2 | 2 | *nobody* (see below) |
| `lua_fragment` | `LuaFragment` | prelude, statements | 4 | 3 | kokoro, matcha, parakeet-rnnt, parakeet-tdt, styletts2, supertonic, vits |
| `subgraph_call` | `SubgraphCallComponent` | statements | 2 | 6 | kokoro, matcha, parakeet-rnnt, parakeet-tdt, styletts2, supertonic, vits |
| `flow_matching_sampler` | `FlowMatchingSampler` | prelude, statements | 0 | 6 | matcha, supertonic |
| `driver_return` | `DriverReturn` | statements | 0 | 1 | kokoro, matcha, styletts2, supertonic, vits |
| `lua_library` | `LuaLibrary` | prelude | 1 | 0 | kokoro, matcha, styletts2, vits |

### `driver_inputs` — `DriverInputs`

Binds every name the topologies below are called with: read from the caller's `inputs` table, or computed host-side (`cache_position` via loom.range, `attention_mask` via loom.causal_mask).

*Emits:* statements. *Used by:* conformer-ctc, hf-causal-lm, lfm2-modular, lfm2-monolithic, qwen3.

* nothing — every field is `__unchecked__`, with its reason

### `monolithic_call` — `MonolithicCall`

The single `run_subgraph` call a flattened export's driver makes, capturing the output's shape alongside its data so the epilogue knows the vocab size -- or, for a KV-cached topology, retaining the output engine-side and binding nothing, so the logits never become a Lua table at all.

*Emits:* statements. *Used by:* conformer-ctc, hf-causal-lm, lfm2-monolithic, qwen3.

* `topology` — TopologyName
* `inputs` — TopologyInput(FieldRef(field='topology'), exact=True)

### `modular_chain` — `ModularChain`

Threads one tensor through an independently-traced submodule chain: prefix -> [aux] -> layer_0..N -> suffix_0..M, each stage carrying its own resolved input map.

*Emits:* statements. *Used by:* lfm2-modular.

* `stages` — holds spec(s) with links of their own, checked in DriverBuilder.build, which registers every ChainStage with the export's own checker (DriverComponent.sub_specs) so a failure names the stage that failed rather than the chain it was in

### `prefill_decode_loop` — `PrefillDecodeLoop`

The `infer_with_past` generation loop: prefill, then decode one token at a time against the KV cache until max_new_tokens or eos_token. One loop rather than a prefill plus a decode loop, because a cached ATTENTION node makes the prefill its first iteration. **The `used by` column over-states this one**, and it is the only entry where that is true: it is a field of every flattened causal-LM builder, but the exporter sets it only for a topology whose cross-step state is ENTIRELY the KV cache. LFM2-monolithic's ten ShortConv layers are not, so it carries the field and exports `infer` alone.

*Emits:* statements. *Used by:* hf-causal-lm, lfm2-monolithic, qwen3.

* `topology` — TopologyName
* `inputs` — TopologyInput(FieldRef(field='topology'), exact=True)

### `ctc_greedy_epilogue` — `CtcGreedyEpilogue`

Greedy CTC decode: per-frame argmax over the retained logits, then collapse consecutive duplicates and drop the blank. `argmax_epilogue`'s ASR counterpart -- the same single forward pass, but a reduction over EVERY row returning a sequence, rather than over one row returning a token.

*Emits:* statements. *Used by:* conformer-ctc.

* `retained_module` — TopologyName

### `argmax_epilogue` — `ArgmaxEpilogue`

Returns the next token rather than the raw logits: argmax over the active row, read out of the producing module's retained output by name, or -- for a topology that marshalled its tensor -- over the returned table, guarded for an output that is not an array.

*Emits:* statements. *Used by:* hf-causal-lm, lfm2-modular, lfm2-monolithic, qwen3.

* `retained_module` — WhenSet(TopologyName)

### `export_constants` — `ExportConstants`

Values only the checkpoint knows (a blank id, a duration set, a hidden width), bound as ordinary locals so every read of them is checked by driver_ir.validate -- rather than interpolated into hand-written Lua through a marker, where a misspelled read is a silent nil (BACKLOG.md P4.0.18).

*Emits:* statements. *Used by:* kokoro, matcha, parakeet-rnnt, parakeet-tdt, supertonic, vits.

* nothing — every field is `__unchecked__`, with its reason

### `raw_lua_driver` — `RawLuaDriver`

A hand-written `.lua` adopted whole -- prelude, one verbatim body block, postlude -- with its own `loom.run_subgraph` call sites parsed out and declared. The step every TTS family moved onto the builder through; no family is on it today.

*Emits:* prelude, statements, postlude. *Used by:* **no model** — see below.

* `external` — ConfigDerived(needs=['topologies'])
  <br>*says:* {label} declares topolog(ies) {detail} as coming from outside this export, but this export produces them.
* `_external_is_called` — ConfigDerived(needs=[])
  <br>*says:* {label} declares topolog(ies) {detail} as external, but its driver never calls them.
* `source` — holds spec(s) with links of their own, checked in DriverBuilder.build, via sub_specs() -- every loom.run_subgraph call site in this text with a literal topology name is parsed out and declared as a RunSubgraphCall, which is what gives the five hand-written drivers their first cross-check against the topologies they are actually shipped with

> No model uses it today: the adoption step's component (C.3). All five TTS families passed through it and all five are now peeled, so it ships with no user by design -- it is what the *next* hand-written driver is adopted by, in a commit whose gate is byte-identity. Deleting it would mean the next family's first step has to be written again.

### `lua_fragment` — `LuaFragment`

One hand-written block of a peeled driver, kept as its own `.lua` file, declaring what it reads and defines (and, since D.2, which topologies its computed call sites drive).

*Emits:* prelude, statements. *Used by:* kokoro, matcha, parakeet-rnnt, parakeet-tdt, styletts2, supertonic, vits.

* `drives` — ConfigDerived(needs=[])
  <br>*says:* {label} has computed call site(s) {detail} that no `drives` declaration covers, so the topologies they run are checked by nothing.
* `drives` — ConfigDerived(needs=[])
  <br>*says:* {label} declares call site(s) {detail} that its text does not contain.
* `defines` — ConfigDerived(needs=[])
  <br>*says:* {label} declares that it defines {detail}, but its text never mentions those names.
* `reads` — ConfigDerived(needs=[])
  <br>*says:* {label} declares that it reads {detail}, but its text never mentions those names.

### `subgraph_call` — `SubgraphCallComponent`

One `loom.run_subgraph` as IR rather than text, so `check_subgraph_calls` covers its output arity too -- what a peel buys structurally.

*Emits:* statements. *Used by:* kokoro, matcha, parakeet-rnnt, parakeet-tdt, styletts2, supertonic, vits.

* `topology` — TopologyName
* `inputs` — TopologyInput(FieldRef(field='topology'), exact=True)

### `flow_matching_sampler` — `FlowMatchingSampler`

A `FlowMatchingSpec`'s generated Euler-CFM sampler function, plus the line that calls it.

*Emits:* prelude, statements. *Used by:* matcha, supertonic.

* `spec` — holds spec(s) with links of their own, checked in DriverBuilder.build, via sub_specs() -- the spec's own TopologyName/TopologyOutputArity/TopologyInput links run against the export's real topologies

### `driver_return` — `DriverReturn`

What the entry function hands back to the host.

*Emits:* statements. *Used by:* kokoro, matcha, styletts2, supertonic, vits.

* nothing — every field is `__unchecked__`, with its reason

### `lua_library` — `LuaLibrary`

Emits the `loom_lua` functions a driver declares, and only those -- the transitive closure of `uses`, in definition order.

*Emits:* prelude. *Used by:* kokoro, matcha, styletts2, vits.

* `uses` — ConfigDerived(needs=[])
  <br>*says:* {label} declares loom_lua function(s) {detail}, which do not exist.

<!-- /generated -->

## 3. `loom_lua` — the driver-side standard library

A component emits statements; the hand-written Lua a family still needs lives in `lua/`, one atomic
function per file, and a driver carries only the transitive closure of what it declares. The `drives`
column is D.2: three of these functions call `loom.run_subgraph` with a name they compute from a
namespace their caller passes, and that column is the shape a caller's `HelperCall` expands against.
`<ns>` is the namespace the caller supplies.

<!-- generated: loom_lua catalogue -->

| function | requires | drives | called by |
|---|---|---|---|
| `array_sum` | — | — | kokoro, matcha, vits, styletts2 |
| `array_slice` | — | — | kokoro, styletts2 |
| `array_affine` | — | — | matcha, vits |
| `sigmoid` | — | — | *`predict_durations`* only |
| `round_half_to_even` | — | — | *`predict_durations`* only |
| `to_row_major` | — | — | kokoro, styletts2 |
| `from_row_major` | — | — | kokoro, styletts2 |
| `to_layout_a` | — | — | *`run_proj1x1`, `run_resblk_stack`* only |
| `from_layout_a` | — | — | kokoro, styletts2 |
| `durations_from_logw` | — | — | matcha, vits |
| `pad_last_to_multiple` | — | — | matcha |
| `repeat_by_duration_tfast` | — | — | matcha |
| `predict_durations` | `sigmoid`, `round_half_to_even` | — | kokoro, styletts2 |
| `run_bi_lstm` | — | `<ns>_fwd`, `<ns>_bwd` ← `layer_input`, `h_prev`, `c_prev` | kokoro, styletts2 |
| `run_resblk_stack` | `to_layout_a`, `from_layout_a` | `<ns>_block0`, `<ns>_block1`, `<ns>_block2` ← `x`, `style` | kokoro, styletts2 |
| `run_proj1x1` | `to_layout_a` | `<ns>` ← `x` | kokoro, styletts2 |
| `compute_wsum` | — | — | kokoro, styletts2 |
| `karras_schedule` | — | — | styletts2 |
| `adpm2_step` | — | — | *`adpm2_sample`* only |
| `adpm2_sample` | `adpm2_step` | — | styletts2 |

<!-- /generated -->

## 4. What a failure looks like

Taken from real failing exports rather than from the source — each of these was produced by breaking
one declaration and running the export it belongs to (the negative gate `EXPORT-PREPARATION.md` §6
requires, recorded here so the next reader does not have to re-run them). This is the half of the
catalogue that says the declarations above are load-bearing rather than decorative.

**A component with no registry entry** (D.1 probe: `lua_fragment`'s entry removed, Kokoro export):

```
KeyError: tools.loom_mil_compiler.driver_components.LuaFragment is a driver component but is not in
the component registry, so it would appear in a shipped driver and in no catalogue. Add a
ComponentEntry to component_registry._entries(). Registered: ['argmax_epilogue', 'driver_inputs',
'driver_return', 'flow_matching_sampler', 'lua_library', 'modular_chain', 'monolithic_call',
'raw_lua_driver', 'subgraph_call'].
```

**An entry that claims the wrong emission** (D.1 probe: `subgraph_call`'s `emits` narrowed to
`prelude`, Kokoro export):

```
ValueError: component 'subgraph_call' (SubgraphCallComponent) emitted ['statements'], which its
registry entry does not declare (declares ['prelude']). The catalogue's 'emits' column is generated
from that declaration, so an out-of-date one publishes a false account of what the component
contributes to a driver.
```

**A call site nobody declared** (D.2 probe: one `HelperCall` renamed, Kokoro export):

```
LinkError: LuaFragment('04_f0n.lua') has computed call site(s) ['line 3: "f0n_shared_lstm"'] that no
`drives` declaration covers, so the topologies they run are checked by nothing. Declare them with a
HelperCall (a loom_lua helper that drives topologies) or a ComputedCall (a loom.run_subgraph whose
name this fragment computes).
```

**A declared namespace that was never exported** (D.2 probe, Kokoro export) — the ordinary
`TopologyName` message, which is the point: after D.2 a computed call site fails exactly the way a
mistyped literal does.

```
LinkError: 04_f0n.lua via run_bi_lstm:3 loom.run_subgraph('f0n_shared_lstmm_h_fwd') names topology
'f0n_shared_lstmm_h_fwd', which is not among the exported topologies ['albert_bert_encoder',
'decoder_vocoder', 'duration_adaln_0', ... 'top_lstm_h_fwd'].
```

**A library declaration that disagrees with the topology** (D.2 probe: `run_bi_lstm`'s declared inputs
changed to `c_state`, Kokoro export):

```
LinkError: 02_duration_encoder.lua via run_bi_lstm:12 loom.run_subgraph('duration_lstm_0_h_fwd') does
not match topology 'duration_lstm_0_h_fwd': supplies input(s) it does not declare: ['c_state']; leaves
declared input(s) unsupplied: ['c_prev']; topology declares ['layer_input', 'h_prev', 'c_prev'], spec
supplies ['layer_input', 'h_prev', 'c_state'].
```

## 5. How much of each driver is checked

Printed by every export, and reporting what is left over rather than only what is covered:

| family | as IR | parsed literal | computed sites declared → topologies | uncovered |
|---|---|---|---|---|
| kokoro | 2 | 2 | 9 → 35 | none — every exported topology is named by a call site |
| styletts2 | 2 | 4 | 9 → 35 | none |
| matcha | 3 | 0 | 0 | none |
| vits | 3 | 0 | 0 | none |
| supertonic | 3 | 0 | 0 | none |

"Every exported topology is named by a call site" is *reported*, not enforced: a family may
legitimately export something the host calls directly rather than the driver — P4.1's Whisper encoder
is the obvious candidate — and turning today's coincidence into a rule would prejudge that.
