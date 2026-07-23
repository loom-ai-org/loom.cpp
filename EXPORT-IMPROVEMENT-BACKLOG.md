# Export Generality Improvement Backlog

Follow-up to `EXPORT-BACKLOG.md` (which stabilized LFM2-350M's atomic + monolithic export). That work
proved `tools/loom_mil_compiler/exporter.py` correct for one model; this backlog addresses a different
question raised while comparing it against coremltools' own MIL→backend architecture (source read directly
from a local `coremltools` clone, not from memory): **how much of the exporter's current logic is genuinely
inherent to translating MIL→ggml, versus incidental complexity specific to LFM2 that will break on the next
model.**

Four threads came out of that comparison. Ordered by dependency (each rationale below is the precise
reasoning already worked out in conversation, kept intact rather than re-derived):

| # | Thread | Why this order | Status |
|---|---|---|---|
| 1 | Consolidate torch-frontend patches + generic export driver | Foundational plumbing every other thread's export scripts sit on top of | **Done** (2026-07-23) |
| 2 | Replace scope-based atomic partitioning with a submodule-export blueprint | Biggest complexity/fragility reduction; independent of #3; directly requested | **Done**, first iteration (2026-07-23) |
| 3 | Extract graph rewrites (GQA fusion, linear-bias compose) into real MIL passes | Cleans up the translation step once #2 has changed what that step receives | **Done** (2026-07-23) |
| 4 | Document the MIL op-coverage boundary (opaque-kernel ops, STFT/FFT lowering) | Reference material for the *next* model's op gaps, not a code change to the current exporter | Not started |

---

## 1. Consolidate torch-frontend patches; add a generic HF export driver

**Status: done (2026-07-23).** Implemented as planned, with the plan's own "byte-for-byte-identical
GGUF before/after" bar as the acceptance test:
- `tools/loom_mil_compiler/torch_patches.py` (new) — `apply_torch_frontend_patches()`, idempotent,
  called from `loom_mil_compiler/__init__.py` at import time.
- `tools/loom_mil_compiler/export_hf_causal_lm.py` (new) — the generic `export_causal_lm()` driver /
  CLI: load → trace → `ct.convert` → `LoomGGUFExporter`, with `architecture` auto-inferred from
  `model.config.model_type` when not given.
- `export_lfm2_monolithic.py` / `export_lfm2_atomic.py` shrunk from ~145 lines each to ~20-line thin
  wrappers calling `export_causal_lm(...)` with LFM2's own model_dir/tokenizer_pre/architecture.
  `tools/convert_lfm/make_lfm2_gguf.py`'s duplicated patch block removed, relies on the import-time
  patch instead.
- Verified: re-ran both LFM2 scripts, `cmp`'d output against pre-refactor GGUFs — byte-identical.
  `test_e2e_lfm2_mil_export` (8/8) and `test_e2e_lfm2_tokenizer` (13/13) pass unchanged.

**Rationale.** `_robust_cast`/`_robust_decompose_sdpa` and their `mil_ops._cast = ...` /
`mil_frontend_utils._decompose_scaled_dot_product_attention = ...` monkeypatches are byte-identical across
three files: `export_lfm2_monolithic.py:28-69`, `export_lfm2_atomic.py:28-69`, and
`tools/convert_lfm/make_lfm2_gguf.py:31-72` (confirmed by grep — verbatim duplication, not just
similar-looking code). Neither patch is LFM2-specific: one fixes constant-folding for scalar casts so a
compile-time-constant `.item()`-style cast doesn't produce a dynamic cast node; the other pre-tiles K/V
before SDPA decomposition so grouped-query attention (mismatched Q/K head counts) traces correctly. Any HF
model exported through this pipeline wants both. `loom_mil_compiler/__init__.py` already does exactly this
kind of import-time side effect (`import loom_mil_compiler  # Registers the "loom" backend"`, referenced in
all three scripts) — the patches belong there, not re-pasted per script.

Once the patches are centralized, what's left in the three scripts (aside from the LFM2-specific
`MonolithicModelWrapper`/`LayerSubmodule`-style wrapper classes) is architecture-agnostic: load an HF model,
trace it, `ct.convert(..., convert_to="milinternal")`, hand the `Program` to `LoomGGUFExporter`. That's
reusable as a generic driver for any `AutoModelForCausalLM`-shaped model.

**Plan.**
1. Add `tools/loom_mil_compiler/torch_patches.py` with `_robust_cast`, `_robust_decompose_sdpa`, and an
   `apply_torch_frontend_patches()` that installs both. Call it from `loom_mil_compiler/__init__.py` so any
   `import loom_mil_compiler` gets them automatically (idempotent — guard against double-patching).
2. Delete the duplicated copies from `export_lfm2_monolithic.py`, `export_lfm2_atomic.py`,
   `tools/convert_lfm/make_lfm2_gguf.py`; rely on the import-time patch instead.
3. Add `tools/loom_mil_compiler/export_hf_causal_lm.py`: a generic CLI (`model_dir`, `profile`, `quantize`,
   `tokenizer_dir`, `output_path` — mirroring the env-var/kwarg surface `LoomGGUFExporter.__init__` already
   accepts) that does load→trace→`ct.convert`→`LoomGGUFExporter(...).export()` for a plain HF causal-LM,
   with no LFM2-specific wrapper code. `export_lfm2_monolithic.py` becomes either a thin invocation of it or
   is retired once thread 2 also removes the atomic-specific script.
4. Keep `export_lfm2_atomic.py` (or its replacement from thread 2) as the one place that still needs
   model-aware code, since atomic export inherently needs to know the model's submodule structure.

**Verification.** Re-run `test_e2e_lfm2_mil_export.cpp`/`test_e2e_lfm2_tokenizer.cpp`/`test_e2e_lfm2_q8_0.cpp`
unchanged — this thread is a pure refactor (no behavior change), so byte-for-byte-identical GGUF output
before/after is the acceptance bar.

---

## 2. Replace scope-based atomic partitioning with a submodule-export blueprint

**Status: first iteration done (2026-07-23), landed as a new opt-in path alongside the existing
`apply_atomic_export` (not yet a replacement — see "kept as a separate path" below).**

- `tools/loom_mil_compiler/submodule_discovery.py` (new) — `find_repeated_blocks()` (structural
  `nn.ModuleList`/`Sequential` discovery, exactly as planned) and `capture_calls()` (forward-hook
  capture of a submodule's real call over one eager pass — the "no shape ever guessed" replacement for
  `make_lfm2_gguf.py`'s hand-fabricated dummy tensors).
- `tools/loom_mil_compiler/submodule_export.py` (new) — `SubmoduleExportSpec` (Phase 1's declarative
  `prefix_attr`/`repeated_attr`/`suffix_attrs`, plus one addition the plan didn't anticipate: an
  optional `aux_attr`/`aux_kwarg` — see "found during implementation" below) and `export_submodules()`,
  which traces each submodule standalone via a generic `_ReplayWrapper` (replays a captured
  `(args, kwargs)` call as a flat `forward(*tensors)`) and assembles a multi-`Function` MIL `Program`.
- `export_lfm2_submodule.py` (new) — driver script producing `lfm2_350m_submodule.gguf`, mirroring
  `export_lfm2_atomic.py`'s shape.
- `apply_submodule_export` (new method in `exporter.py`) — consumes the real standalone `Function`s
  directly via `generate_graph_topology(func, name)`, exactly as the plan's "driver-synthesis needs no
  new code" section predicted; no ops_list/inputs_dict reconstruction.

**Found during implementation, not anticipated by the original plan text:**
1. **A shared once-computed tensor problem.** LFM2's rotary-embedding table (`model.pos_emb`) is
   computed ONCE in `Lfm2Model.forward` and passed identically to every decoder layer as
   `position_embeddings=(cos, sin)` — a case the plan's prefix/repeated/suffix boundary doesn't cover
   (it's neither prefix nor per-layer). Fixed by adding a 4th declarative piece, `aux_attr`/`aux_kwarg`:
   a submodule traced once, whose output is threaded into every repeated-block call.
2. **The engine supports exactly one output tensor per subgraph call** (`loom.run_subgraph` always
   returns exactly `(data, shape)`, never more). A submodule that itself returns a tuple (aux's
   `(cos, sin)`) or consumes a tuple-valued kwarg can't be wired as multiple named outputs/inputs the
   way a first attempt assumed. Fixed generically in `submodule_export.py`: any tuple/list-valued
   kwarg or return value made of tensors is concatenated along its last axis into ONE tensor at the
   trace boundary (recording split sizes to reconstruct the tuple inside the wrapper before calling the
   real module) — not LFM2-specific, applies to any future model with the same shape of problem.
3. **Two real bugs in the pre-existing `generate_graph_topology`, both now fixed (and covered by
   regression tests) since they affect the atomic path too, just never triggered by it:**
   - `resolve()` infinite-looped on a self-referential alias, which only happens when a submodule's own
     input is already literally named `"hidden_states"` (true for every independently-traced decoder
     layer here; never true for atomic's SSA-derived slice names).
   - A declared input with no real consumer (e.g. `cache_position`/`position_ids`, dead once
     `past_key_values` is forced to `None` for a stateless standalone trace) got a ggml tensor created
     but never allocated a backend buffer (`ggml_gallocr_alloc_graph` only allocates tensors reachable
     from the declared output) — setting data into it crashed with `GGML_ASSERT ... tensor buffer not
     set`. Fixed by dropping post-pruning-unreferenced inputs from the topology's declared list, so a
     driver that still supplies one now gets a clean `check_subgraph_calls` export-time error instead of
     a runtime crash. Also fixed a related latent bug this crash's error message ran into:
     `src/core/lua_bridge.cpp`'s size-mismatch error used `%zu`, a directive Lua's own `pushfstring`
     (not libc printf) doesn't understand, silently truncating the message before the real numbers ever
     printed — changed to `%d`.
   - Also discovered: which inputs survive pruning can differ **per layer**, not just per model — LFM2's
     conv-type layers never touch `position_embeddings` (only its attention-type layers do), so
     `apply_submodule_export`'s wiring consults each layer's own post-filter declared inputs rather than
     a single shared name list.

**Verification performed:** `test_e2e_lfm2_mil_export` passes against `lfm2_350m_submodule.gguf` — same
expected top-1 tokens (3523 / 2) as atomic/monolithic. Re-ran atomic/monolithic exports after the
`generate_graph_topology`/`lua_bridge.cpp` fixes and confirmed byte-identical output to before (the
fixes are no-ops for those paths). GGUF size: submodule (1.69GB) ≈ atomic (1.69GB) > monolithic
(1.42GB) — reproducing this doc's own predicted tied-embedding duplication cost (one 65536×1024 fp32
matrix, ~268MB), not a new regression.

**Not yet done (still open, deferred out of this iteration):**
- **Kept as a separate opt-in path, not a replacement.** `apply_atomic_export`/`export_lfm2_atomic.py`
  are untouched and still the default "atomic" profile, per this doc's own step 5 ("keep the existing
  fallback available until the new path is verified numerically"). Promoting the submodule blueprint to
  the default (and deleting the scope-partitioning code path) is a follow-up decision, not yet made.
- **A second model in the regression suite.** Everything above was validated on LFM2 alone; the
  generality claim (structurally different attribute names, a non-hybrid homogeneous-layer model) is
  still unproven per this doc's own verification checklist.
- **The content-hash weight dedup** for cross-submodule duplication (rationale part C's "known cost")
  — not attempted; the GGUF-size measurement above is the input to that future decision, not the fix
  itself.
- Phase 2 (fully automatic prefix/suffix boundary discovery via early-exit hooks) — not attempted;
  Phase 1's declarative spec (now 4 fields: prefix/repeated/suffix/aux) was sufficient for LFM2.

**Rationale, part A — the current partitioner is not as heuristic as it looked at first, but its
reconstruction step is.** `apply_atomic_export`'s `torch_scope_key` (`exporter.py:303-312`) reads
`op.scopes[ScopeSource.TORCHSCRIPT_MODULE_NAME]`, which for any model traced via `torch.jit.trace` is exact
ground-truth metadata coremltools' torch frontend attaches per-op (confirmed directly in
`coremltools/converters/mil/frontend/torch/converter.py:1261,1321`) — not a guess. I checked whether
`torch.export` (EXIR) gives richer/safer metadata to switch to instead — it doesn't, yet: coremltools' own
EXIR path uses `ScopeSource.EXIR_STACK_TRACE`, which the source itself marks
`# no serialization for such debug info should be allowed yet` and
`# TODO (rdar://125572392): Support torch.export IO metadata` (`converter.py:1268-1269,1333`). So
`jit.trace` + `TORCHSCRIPT_MODULE_NAME` is already close to the best boundary *signal* coremltools exposes
today. The genuinely fragile parts are narrower than "scope-regex": (1) "a digit segment marks a
`ModuleList`/`Sequential` index" — reliable by construction for that convention, but assumes it; (2)
`name_regex_key`, the total fallback when scope metadata is absent entirely (hand-built MIL, or an
unsupported frontend) — this is the real "exotic naming" risk.

**Rationale, part B — the actual fragility is reconstructing per-slice subgraphs from a flattened trace,
which the current partitioner does at real cost.** Once `apply_atomic_export` has slice boundaries, it
still has to reconstruct each slice's own inputs/outputs from ops embedded in one giant flattened `main`
function: the "replicate consumed constants," `_collect_replica_closure`, and `exposed_ops`
legitimate-external-ref bookkeeping (`exporter.py:390-498`, ~230 lines) exists purely to recover, after the
fact, information that was thrown away by tracing everything as a single function in the first place.
`EXPORT-BACKLOG.md` documents this costing two real, separately-fixed mis-attribution bugs. This is not a
smell to patch harder — it's solving a problem that a different tracing strategy avoids by construction.

**Rationale, part C — you already built and then moved away from the alternative once, and I now think that
was for the wrong reason.** `tools/convert_lfm/make_lfm2_gguf.py:118-160` traces each submodule (embedding,
each decoder layer, output head) **separately** via `ct.convert(..., convert_to="milinternal")` and stitches
the resulting `Function`s into one multi-`Function` `Program`
(`master_prog.functions[f"layer_{i}"] = layer_mil.functions["main"]`). This sidesteps the whole
closure-reconstruction problem: each submodule's MIL graph is self-contained by construction — there is no
cross-slice variable leakage to detect, because nothing was ever flattened into one function to begin with.
Per `EXPORT-BACKLOG.md`, the project moved away from this toward the automatic single-trace partitioner to
avoid its per-model boilerplate: a hand-written wrapper class per submodule type (`EmbeddingSubmodule`,
`LayerSubmodule`, `OutputHeadSubmodule`), a hardcoded `range(16)` loop, hardcoded dummy-input shapes, and a
synthetic `MockOperation`-based orchestration stub built by string-matching op names. That boilerplate is
real, but — and this is the point the user's follow-up question surfaced — **it isn't inherent to the
per-submodule-tracing strategy; it's inherent to doing that strategy by hand.** Every one of those four
pieces of boilerplate can be automated:

- *Which attributes are the repeated layers?* Don't hardcode `range(16)` or a submodule name — walk
  `model.named_modules()` and find `nn.ModuleList`/`nn.Sequential` instances structurally
  (`isinstance` check). This is *more* reliable than either the current scope-digit rule or a
  by-name lookup, because it doesn't need to know what the model's author called the attribute
  (`model.layers`, `transformer.h`, `model.decoder.layers` are all the same shape of thing to this check) —
  it also naturally handles hybrid architectures (LFM2's mix of attention/conv layer classes) correctly,
  since each `nn.ModuleList` child is traced as whatever concrete class it actually is, instead of being
  forced through one shared "layer_N" op-name pattern.
- *What are the dummy input shapes per submodule?* Don't hand-derive them — register forward hooks on the
  real submodules and run **one** ordinary eager forward pass with representative top-level dummy inputs;
  record the actual args/kwargs shapes+dtypes each submodule received. Feed those recorded shapes into the
  standalone `ct.convert()` call for that submodule. No shape is ever guessed or hardcoded.
- *What wraps the "everything before/after the layer loop" computation?* This is the one piece that
  isn't recoverable from module structure alone (a `forward()` method is imperative Python, not a static
  graph), so it needs an explicit boundary. Two options, phased:
  - **Phase 1 (do this first):** a ~3-line declarative spec per model architecture — the qualified attribute
    names for "prefix" (e.g. `model.embed_tokens`), "repeated" (e.g. `model.layers`), and "suffix" (e.g.
    `model.norm`, `lm_head`) — replacing ~230 lines of reconstruction logic with ~3 lines of declared
    structure. This is a fundamentally different kind of "model-specific" than the current heuristic: it's
    a fact stated once and verified at export time (wrong attribute name = immediate `AttributeError`, not
    a silent wrong export), not a pattern inferred from op-name text after the fact.
  - **Phase 2 (stretch goal, do only if Phase 1's per-model 3-line spec still feels like too much):** derive
    prefix/suffix automatically too, using the same "early-exit hook" technique HF's own `accelerate`
    library uses for device-map model splitting — install a hook on the repeated block's first child that
    captures its inputs and raises to stop the forward pass early (yields "prefix" as a traceable partial
    function of the real top-level `forward()`), and separately monkeypatch the repeated `ModuleList` to an
    identity passthrough of the right length to isolate "suffix" (norm + head) by running the *real*
    top-level `forward()` with the loop body neutralized. This needs no per-model subclassing at all, but is
    materially more complex to get right for arbitrary forward-signature conventions — worth attempting only
    after Phase 1 is proven on 2-3 models.
- *How is the orchestration driver generated?* **This part needs no new code at all.** `apply_atomic_export`'s
  own driver-synthesis half (`exporter.py:505-574` — building the `SubgraphCall` sequence from a list of
  named slices) is already generic: it only needs, per slice, a name and a `Function`/inputs/outputs. It
  doesn't care whether that `Function` came from heuristic reconstruction or a standalone `ct.convert` call.
  Better still: `generate_graph_topology(func, func_name)` (`exporter.py:829`) already accepts a `Function`
  directly (the `func` parameter, used today by the "bespoke" workflow) — a submodule's `ct.convert(...)`
  result can be handed to it with **no** `ops_list`/`inputs_dict` reconstruction, since that reconstruction
  is exactly the part being eliminated. Cross-submodule weight-name collisions are already handled too, via
  the existing `f"{func_name}.{weight_name}"` namespacing in `generate_graph_topology`.

**Net effect:** this replaces `apply_atomic_export`'s first ~230 lines (scope partitioning +
`_collect_replica_closure` + `exposed_ops` bookkeeping) with a submodule-discovery step, and reuses its last
~70 lines (driver synthesis) essentially unchanged. It is strictly more general than the current partitioner
(handles hybrid layer types by construction, no ModuleList-naming-convention assumption) and strictly less
boilerplate than `make_lfm2_gguf.py` (no hand-written wrapper classes, no hardcoded shapes, no hardcoded
layer count).

**Known cost, confirmed empirically: larger GGUF output from cross-submodule tensor duplication — deferred
to a second iteration.** Independently observed (not yet root-caused in this codebase, but the mechanism is
clear from how the pipeline is built): each submodule is converted via its own standalone `ct.convert()`
call, so any tensor referenced from *more than one* submodule gets traced, const-folded, and serialized
separately by each call that touches it — there is no shared identity across independent conversions to
catch this. `LoomGGUFExporter.weights` (`exporter.py`) is keyed purely by namespaced name
(`f"{func_name}.{weight_name}"`), with no content-based dedup, so two byte-identical tensors reachable from
two different submodules are written to the GGUF twice under two different names. The most likely large
contributor for causal-LM models specifically: HF's weight-tied embedding/`lm_head` (`config.tie_word_embeddings`)
is the *same* `nn.Parameter` referenced from both the "prefix" (embedding) and "suffix" (output head)
submodules — traced independently, it becomes two full copies of the vocab-sized matrix instead of one.
Smaller shared buffers/precomputed tables referenced by every layer (if any) would duplicate the same way,
once per consuming layer.

Note this isn't fixable by reusing coremltools' own dedup mechanism as-is:
`MILProtoExporter.create_file_value`/`get_weight_path` (`coremltools/converters/mil/backend/mil/load.py:120-170`)
already caches weight blobs by `op.weight_id` specifically to avoid re-writing a tensor shared across
`Function`s — but that cache works by *Python object identity* carried through **one** frontend conversion
producing multiple functions in a single `Program`. The submodule blueprint calls `ct.convert()` once per
submodule, so each call's consts are fresh objects with no shared identity to key off, even when the
underlying data is byte-for-byte identical. A working fix has to dedup on **content**, not identity: hash
each candidate weight array's bytes (+shape+dtype) at `write_gguf` time; if a matching hash was already
written under a different namespaced name, alias the new name to it instead of writing a second copy
(mirroring the *effect* of coremltools' `weight_id` cache, via a different mechanism suited to
independently-traced inputs). Deferred out of this thread's first iteration — get the partitioning
replacement numerically correct first, then measure actual GGUF size overhead on LFM2 (isolating the
tied-embedding case specifically) before deciding whether a full content-hash pass is worth the write-time
cost, or whether skipping re-export of `lm_head`'s weight when `tie_word_embeddings` is set (and aliasing it
to the embedding's own weight name directly) is enough on its own.

**Plan.**
1. `tools/loom_mil_compiler/submodule_discovery.py`:
   - `find_repeated_blocks(model) -> Dict[str, List[nn.Module]]` — `named_modules()` walk, `isinstance`
     check for `nn.ModuleList`/`nn.Sequential` with >1 child.
   - `capture_submodule_io_shapes(model, dummy_inputs, targets) -> Dict[nn.Module, ShapeSpec]` — forward
     hooks + one eager pass.
2. `tools/loom_mil_compiler/submodule_export.py`:
   - `SubmoduleExportSpec` dataclass: `prefix_attr: str`, `repeated_attr: str`, `suffix_attrs: List[str]`
     (Phase 1's declarative boundary).
   - `export_submodules(model, spec, dummy_inputs) -> mil.Program` — runs discovery/shape-capture, calls
     `ct.convert(..., convert_to="milinternal")` per submodule (prefix once, each repeated child once, each
     suffix attr once), assembles the multi-`Function` `Program` (same assembly pattern as
     `make_lfm2_gguf.py:131,148,160`, generalized).
3. Rewire `apply_atomic_export` (or a new `apply_submodule_export`) in `exporter.py` to consume this
   `Program`'s functions directly via `generate_graph_topology(func=..., func_name=...)`, keeping the
   existing driver-synthesis code (`exporter.py:505-574`) with minimal adjustment.
4. `export_lfm2_atomic.py` shrinks to: load model, declare
   `SubmoduleExportSpec("model.embed_tokens", "model.layers", ["model.norm", "lm_head"])`, call the new
   entry point. Delete the now-dead scope-partitioning code path once this is verified.
5. Keep the *existing* `apply_atomic_export` available behind a flag (or in git history) until the new path
   is verified numerically — don't delete the fallback until parity is proven, per item 6 below.

**Verification.**
- New path must reproduce `test_e2e_lfm2_mil_export.cpp`'s existing atomic-profile assertions exactly
  (topology count may differ in naming but token-level output must match real HF top-1 at both tested
  prompt lengths).
- Add a second model to the regression suite specifically *because* this thread's whole point is
  generality — LFM2 alone can't prove the scope-heuristic's replacement is actually more general. A
  same-family but structurally different HF model (different attribute names, ideally a non-hybrid
  homogeneous-layer model to start) is the right next test.
- Confirm hybrid-layer handling (LFM2's mixed attention/conv layers) is not just "not broken" but actually
  simpler under this scheme — each layer is traced as its real class, no shared "layer_N" assumption needed.
- Record GGUF file size (and ideally a per-tensor-name size breakdown) alongside the token-match check, to
  quantify the cross-submodule duplication noted above on LFM2 before deciding whether the content-hash dedup
  fix (second iteration) is worth doing.

---

## 3. Extract graph rewrites into real MIL passes

**Status: done (2026-07-23).** Implemented as planned, with one deliberate scope narrowing (linear-bias
compose was never moved -- the plan's own Rationale already says it shouldn't be, see below) and one
finding not anticipated by the plan text:

- `tools/loom_mil_compiler/passes.py` (new) -- `fuse_gqa_repeat_kv`, a real `AbstractGraphPass` (namespace
  `"loom"`) that pattern-matches the `tile -> reshape` half of HF's `repeat_kv()` idiom directly on the
  pymil graph and replaces it with real `mb.reshape`/`mb.tile`/`mb.reshape` ops, plus
  `apply_loom_mil_passes(prog)` which runs it followed by `common::dead_code_elimination`.
- `exporter.py`'s `export()` calls `apply_loom_mil_passes(self.program)` once, before any of the
  monolithic/atomic/submodule-blueprint workflows walk the program -- **except** the hand-built "bespoke"
  workflow (see "found during implementation" below).
- Deleted from `exporter.py`: `_try_fuse_gqa_repeat_kv` (the old JSON-node-building version), the
  `fused_reshape_op_ids` bookkeeping set, and the inline fusion-detection call from the `"tile"`
  translation branch's body (the branch itself stays -- it's still the generic tile->REPEAT translation,
  now also the thing that mechanically translates the fusion pass's own output).
- `_prune_dead_nodes` was **kept**, not deleted -- see "found during implementation" below.
- `linear`->`MUL_MAT`+`ADD` composition and the `matmul`-transpose composition were **not** touched, exactly
  per the plan's own Rationale ("these are 1:1 schema differences... and correctly stay in the translation
  step") -- the thread-3 table row's "linear-bias compose" phrasing was never meant literally as a
  to-be-extracted item; the detailed plan text already excluded it.

**Found during implementation, not anticipated by the original plan text:**
1. **The plan's own suggestion to delete `_prune_dead_nodes` "once the pass-based equivalents are verified
   to produce identical output" doesn't hold.** `_prune_dead_nodes` was never *only* cleaning up the GQA
   fusion's orphaned dependency chain -- `apply_atomic_export`'s own `_collect_replica_closure` comment
   (exporter.py, "Any resulting now-unused copy left behind in the original 'accidental host' slice is
   harmless: item 3's `_prune_dead_nodes` already drops any node unreachable from that topology's own
   declared output") documents a second, independent, still-live dependency: replicating a producer op into
   every slice that consumes it deliberately leaves an unused copy behind in whichever slice happened to
   host it first. That's a Python-level list-slicing artifact of partitioning one flattened trace after the
   fact -- no MIL-level pass over the pre-partitioned `main` function could ever see or clean it up, since
   the partitioning itself hasn't happened yet at that point. Kept `_prune_dead_nodes`, updated its
   docstring to name both the (now largely moot, MIL-pass-handled) GQA case and the (still load-bearing)
   atomic-partitioning case explicitly.
2. **Scope propagation onto newly-created ops needed to be explicit, not left to coremltools' automatic
   copy-on-replace.** `AbstractGraphPass`'s scope-copy mechanism (`Block._copy_scope_info`, triggered via
   `try_replace_uses_of_var_after_op`) only propagates the replaced op's `TORCHSCRIPT_MODULE_NAME` scope
   onto the *last* new op (the one whose var directly substitutes for the removed op's var) -- the two
   intermediate ops (the inserted-axis reshape and the tile) would otherwise carry only the
   `COREMLTOOLS_GRAPH_PASS` scope from the pass-execution context, not the decoder-layer scope
   `apply_atomic_export`'s scope-based partitioning needs to attribute them to the right layer. Relying on
   them landing in the correct slice purely through positional adjacency (they're inserted exactly where
   the removed `tile`/`reshape` ops sat) would have been exactly the kind of fragile mis-attribution risk
   this backlog's item 2 already documents two real bugs from. Fixed by explicitly wrapping all three new
   ops' construction in `mb.scope(ScopeInfo(source=TORCHSCRIPT_MODULE_NAME, data=...))`, copied from the
   original `tile` op, rather than relying on the automatic mechanism at all.
3. **The pass had to be skipped for the "bespoke" hand-built-Program workflow.** `common::dead_code_elimination`
   run over `test_compiler.py`'s `MockOperation`-based test fixture raised `ValueError: Cannot delete op...
   with active output... used by ops [...]` -- that fixture builds its multi-function `Program` by directly
   splicing Python lists (`main_func.functions["main"].operations = main_operations`) rather than through
   MIL's own block-mutation API, leaving stale `child_ops` edges DCE's post-removal consistency check
   correctly flags. This was never a real regression (nothing exercised `prog.validate()`-adjacent
   machinery on this deliberately-synthetic fixture before), but it meant `apply_loom_mil_passes` needed to
   be scoped to skip exactly the `is_bespoke and self.profile is None` branch -- the one workflow that
   accepts graphs never traced through `ct.convert()` at all, and the only one containing synthetic
   duck-typed ops (like `MockOperation`) a real MIL pass was never meant to see. Also switched
   `apply_loom_mil_passes` to invoke `PASS_REGISTRY[name](prog)` directly rather than going through
   `PassPipelineManager.apply_pipeline` (which additionally calls `prog.validate()` before/after every pass)
   -- narrower and sufficient for what this thread needed, and avoids a second, independent reason the same
   fixture would have failed even after the `is_bespoke` gate above.
4. **One genuine (if cosmetic) output difference surfaced and was fixed as a small, generally-applicable
   fix, not a special case.** The new pass's `mb.tile` call has `reps=1` on every axis except the newly
   inserted one; the *existing*, unmodified generic `"tile"` translation branch's shape derivation
   (`exporter.py`) unconditionally wrapped every axis in a `"(dim * rep_factor)"` expression string even
   when `rep_factor == 1`, producing `"(n_tokens * 1)"` instead of the old hand-crafted composition's bare
   `"n_tokens"` for the unchanged sequence-length axis -- a latent inefficiency in pre-existing code that
   the old JSON-level GQA composition never happened to exercise (it wrote its shape strings directly,
   bypassing this branch entirely) but any future `tile` op with `reps=1` on a dynamic axis would have hit.
   Fixed generically (skip the multiplication wrapper whenever `rep_factor == 1`, regardless of `dim_size`'s
   type), not GQA-specific.

**Verification performed:** `python3 -m pytest test_compiler.py` (4/4). Re-ran all three LFM2 export
scripts (`export_lfm2_monolithic.py`, `export_lfm2_atomic.py`, `export_lfm2_submodule.py`) against the same
`/home/flavio/Dev/models/lfm2-350m` checkpoint used before this thread's changes and `md5sum`'d the output
against the exact pre-change GGUFs: **all three byte-for-byte identical**
(`lfm2_350m_monolithic.gguf`=`bb5ba992...`, `lfm2_350m_atomic.gguf`=`06983624...`,
`lfm2_350m_submodule.gguf`=`92f57be9...`, matching before and after). `ctest -R
"test_e2e_lfm2_mil_export|test_e2e_lfm2_tokenizer"` (the two tests this thread's own Verification section
names, mirroring thread 1's bar) both pass. (`test_e2e_lfm2_lua_driver`/`test_e2e_lfm2_q8_0` were not part
of this check -- they require differently-named fixture files, `lfm2_350m.gguf` and
`lfm2_350m_monolithic_q8_0.gguf`, that neither this thread nor thread 1/2 ever generates.)

**Rationale.** Coremltools' own backend never mixes graph rewriting with serialization: rewrites run as
`PassPipeline` stages over the pymil graph *before* backend translation, keeping the translator itself
mechanical/schema-driven (`MILProtoExporter.translate_generic_op`, `coremltools/converters/mil/backend/mil/load.py:350`).
`generate_graph_topology` currently interleaves both in one walk: the GQA tile+reshape fusion
(`_try_fuse_gqa_repeat_kv`, invoked inline from the `"tile"` branch), the `linear`→`MUL_MAT`+`ADD`
composition, and dead-node pruning (`_prune_dead_nodes`, post-hoc over Loom's own flattened node list) are
all graph-level rewrites, not translation. Pulling them out as real MIL→MIL passes, run once before
`generate_graph_topology` ever walks the graph, has two concrete benefits beyond tidiness: they become
testable directly against pymil graph structure (matching a pattern, replacing ops) instead of against
Loom's derived JSON node list; and dead-node pruning can likely be replaced outright by coremltools' own
`common::dead_code_elimination` (`coremltools/converters/mil/mil/passes/defs/cleanup/dead_code_elimination.py`,
already in the default pipeline "always end with dce") run again after Loom's own fusion passes, instead of
maintaining a hand-rolled backward-reachability walk.

Use coremltools' own pass API rather than inventing one — it already solves op-pattern-matching and
safe replacement:
```python
class my_fusion(AbstractGraphPass):
    def apply(self, prog):
        @block_context_manager
        def apply_block(block):
            for op in list(block.operations):
                if matches_pattern(op):
                    block.remove_ops([...])
                    mb.new_op(...)
        for f in prog.functions.values():
            apply_block(f)
```
(pattern confirmed from `coremltools/converters/mil/mil/passes/defs/optimize_elementwise_binary.py`).

Note `linear`→`MUL_MAT`+`ADD` and the `matmul`-transpose composition are *not* passes — they're 1:1 schema
differences (one MIL op maps to a fixed small sequence of Loom ops, no pattern-matching/search involved) and
correctly stay in the translation step.

**Plan.**
1. `tools/loom_mil_compiler/passes.py`: `fuse_gqa_repeat_kv` as an `AbstractGraphPass` (port
   `_try_fuse_gqa_repeat_kv`'s matching logic, replace its manual node-dict construction with `mb.*` calls
   building real MIL ops — reshape/repeat/reshape — inserted into the pymil graph).
2. Run this pass (plus `common::dead_code_elimination`) via a small `PassPipeline` right after `ct.convert`
   produces the `Program`, before `LoomGGUFExporter` sees it.
3. Delete `_try_fuse_gqa_repeat_kv`, its `fused_reshape_op_ids` bookkeeping, and `_prune_dead_nodes` from
   `exporter.py` once the pass-based equivalents are verified to produce identical output.
4. Leave `_prune_dead_weights` as-is for now (it operates on GGUF weight names post-serialization, not the
   pymil graph — out of scope for this thread) unless DCE at the pymil stage already makes it a no-op, in
   which case delete it too.

**Verification.** Byte-for-byte identical GGUF output on LFM2 before/after (this is a refactor, not a
behavior change) — same regression tests as thread 1.

---

## 4. Document the MIL op-coverage boundary (reference, not a code change)

**Rationale.** Whether "ggml compositions can cover all MIL ops" turned out to be mostly true but not
universally: `OP_MAP` (73 entries) plus the special-cased compositions already cover essentially all of
MIL's activation/elementwise/linear/tensor_transformation/reduction/normalization categories (iOS15 defines
~145 ops across these plus conv/pool; iOS17 adds ~45 more, mostly quantization-related, likely moot since
Loom does its own GGUF-level quantization). Control-flow ops (`cond`, `while_loop`) aren't graph nodes to
translate at all — already correctly handled as Lua control flow, not ggml ops.

The real gap is a narrow but important class: **ops that are opaque single-kernel primitives in pymil with
no MIL-level decomposition exposed to a generic walker.** `complex_stft`/`complex_fft` are the concrete
example already hit: `tools/convert_kokoro/kokoro_stft_common.py` had to hand-derive a DFT-matrix identity
(as CONV_1D kernels, plus a new ATAN2 primitive) entirely outside the compiler, because there is no MIL
subgraph for a generic exporter to walk for these ops as such.

Useful, verified finding for next time: coremltools *does* decompose `complex_stft`/`complex_fft`/etc. into
ordinary matmul/reshape/gather ops via a real pass — `common::lower_complex_dialect_ops`
(`coremltools/converters/mil/mil/passes/defs/lower_complex_dialect_ops.py`), which is in the **default**
pipeline (`pass_pipeline.py:25`) and uses the same DFT-matrix trick
(`_calculate_dft_matrix` in that file) that was independently reinvented by hand for Kokoro. If a submodule
containing `torch.stft`/`torch.istft` is traced through a normal `ct.convert(..., convert_to="milinternal")`
call (which runs the default pipeline, including this pass) rather than converted by hand, the exporter's
existing generic translation path might already receive it pre-decomposed into ops it knows. This is worth a
cheap experiment before assuming any future STFT-using model (Matcha-TTS, VITS, SupertonicTTS) needs a fully
bespoke path the way Kokoro's did — though there may be a reason (numerical precision, complex-dtype
friction) the current Kokoro path avoided it that hasn't been dug into.

Other categories with no ggml equivalent, confirmed to need the same one-off hand-derivation treatment (not
extendable via `OP_MAP`) if a future model needs them: `recurrent` (LSTM/GRU — no native ggml op), `random`,
`image_resizing`/`scatter_nd` (relevant only if a future vision model needs them).

**Plan.** No exporter code changes. Add a short section to this file (or a new `MIL-OP-COVERAGE.md`) listing:
- Confirmed-covered MIL op categories (table above).
- Confirmed-opaque ops requiring hand-derivation, with the STFT precedent as the template for how to do it.
- The `lower_complex_dialect_ops` finding, flagged as "try this first" for the next FFT-family op before
  reaching for a hand-rolled DFT-matrix derivation.

**Verification.** N/A (documentation only). Revisit when the next non-LLM model (Matcha-TTS/VITS per the
procedural-generalization roadmap) hits an op gap.
