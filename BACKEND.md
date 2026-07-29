# Backend / exporter improvement log

Working status and findings for the work items in [`EXPORT-IMPROVEMENT.md`](EXPORT-IMPROVEMENT.md).
One section per item; each records what was done, what was *found* (including things that turned out
differently from the proposal's own expectation), and how it was verified.

Item 5 (empirically prototype StableHLO) is deliberately not started — the proposal itself files it as
"not recommended as near-term work", a validation exercise rather than a fix.

| item | status | new/changed files |
|---|---|---|
| 1 — declarative op table | done, output-identical | `topology_ops.py`, `symbols.py`, `test_topology_rules.py`; `exporter.py` 4,440 → 2,121 lines |
| 2 — centralized value resolution | done, output-identical | `value_facts.py`, `test_value_facts.py` |
| 3 — explicit control-flow capture | **not done as written** — mechanism built and tested, but deliberately not applied to any of the four named models. The "bake the count, re-open it at conversion" workaround was probed too and gets 2 of 3 prerequisites; see the findings | `scripted_loop.py`, `test_scripted_loop.py` |
| 4 — iterative-refinement template | done for Matcha + Supertonic; StyleTTS2 gets the validation half; Kokoro untouched. **Its predicted line-count payoff did not materialize** | `iterative_export.py`, `test_iterative_export.py`; three drivers + export scripts |
| 5 — StableHLO prototype | not started (per the proposal) | — |
| — StyleTTS2 regression (found here) | fixed and numerically re-verified | `topology_ops.py` `reduce_mean` guards |
| — Conformer/Parakeet export blow-up (found here) | bisected to `a29ffe5`, fixed; >2 h (never finished) → 40 s | `value_facts.py` `dim_expr` memo |
| — NeMo ASR encoder template (third family) | done, all three exports byte-identical | `nemo_asr_export.py`, `test_nemo_asr_export.py`; the three `export_*_mil.py` scripts |
| — sympy shape expressions | done; all 4,081 rewritten expressions checked numerically, which found (and got fixed) a pre-existing bug in a heuristic that was keying on an expression's spelling | `shape_expr.py`, `test_shape_expr.py`, `compare_snapshots.py` |

Three bugs surfaced along the way and all are fixed. Two pre-dated this work: StyleTTS2's export was
already broken at `HEAD`, and the Conformer-CTC/Parakeet exports had regressed into an exponential
blow-up that made them effectively unrunnable — both bisected to the commit that introduced them. The
third this refactor introduced itself, and the golden diff caught it (last section).

---

## Verification method (items 1 and 2)

Items 1 and 2 are pure refactors: they must not change a single byte of any exported model. The check
used throughout is a **golden structural snapshot diff** over all 12 export scripts:

1. Re-run every `export_*.py` against the real local checkpoints, from a pristine `git archive HEAD`
   copy of the repo (so the baseline cannot be contaminated by in-progress edits).
2. Snapshot each resulting `.gguf` into diffable text with
   **`tools/loom_mil_compiler/snapshot_gguf.py`** — every metadata KV, each `model.graph_topology.*`
   JSON pretty-printed and key-sorted, the embedded `model.driver_script` Lua, and one line per tensor
   (`name / shape / dtype / sha256-of-data`).
3. Refactor, re-run, re-snapshot, and require a **zero-line diff** (`diff -r` over the two snapshot
   dirs).

That script is in the repo precisely so this is repeatable — its own docstring carries the full recipe.
It is what caught this refactor's one real bug (three silently dropped `LESS` nodes); no existing test
covered it, because the emitted model stayed structurally valid, just wrong.

**Finding — the committed `.gguf` files are stale.** The `.gguf` artifacts sitting in the repo tree do
not match what the current exporter produces (re-running `export_vits_mil.py` alone already differs: the
current exporter emits explicit `REPEAT` broadcast nodes for the mutual-broadcast case that those
artifacts predate). They are `.gitignore`d build outputs, so this is expected, but it means they are
**not** usable as a baseline — the baseline has to be regenerated.

**Finding — StyleTTS2 export was already broken at `HEAD`, before any of this work.** *(Fixed — see
"Repairing the StyleTTS2 regression" below.)* `export_styletts2_mil.py` failed in its `diffusion` phase
with:

```
NotImplementedError: reduce_mean op 'x_full.9': reduction axis size ('(floor(((1) * ((floor(...
must be a static architecture constant
```

Bisected: it exports cleanly at `67a54a9` ("Add MIL-based export of StyleTTS2") and fails at `HEAD`.
The regression came in with `166be64` ("Add MIL-based export of MatchaTTS"), which added a *dedicated*
`reduce_mean` handler guarding that the reduction axis be a static constant. Before that commit,
StyleTTS2's diffusion `x.mean(axis=1)` — a reduction over the **dynamic** token axis — fell through to
the generic `OP_MAP` `MEAN` path instead, which needs no static count (ggml_mean divides by `ne[0]` at
run time). That generic path's own in-code comment still documents StyleTTS2's diffusion as its
motivating case, which is now unreachable.

Before the repair, items 1/2 equivalence for StyleTTS2 was established up to that point only, and it
held: `albert` 550 nodes / 258 weights and `decoder_vocoder` 1373 nodes / 1152 weights before and after,
then the identical failure on the identical op with the identical derived expression.

---

## Item 1 — declarative pattern table for `exporter.py`'s op dispatcher

**Status:** implemented.

`generate_graph_topology`'s 2,000-line `if op_type == "..." / continue` chain is now a table lookup.

- New `tools/loom_mil_compiler/topology_ops.py` holds the table: a `topology_rule(*op_types,
  guard=..., when=...)` decorator registers each handler, `lookup_topology_rule(exporter, op)` returns
  the first rule whose op type matches and whose guard accepts, and anything unclaimed still falls
  through to the generic `OP_MAP` path (unchanged).
- New `TopologyContext` carries the per-topology mutable state. A free-variable scan of the original
  chain established that the handlers collectively touch exactly four outer names — `nodes`, `aliases`,
  `topo_inputs`, `func_name` (plus `resolve`, now a method on the context) — so the context is those and
  nothing more.
- `exporter.py`: **4,440 → 2,119 lines.** The per-op loop is now eight lines.
- New `symbols.py` holds `DYNAMIC_SYMBOL_RE`, shared by both modules without an import cycle.

**Guards.** 37 rules over 37 MIL op types, 7 of them guarded. Most ops have a single unguarded rule; the
four where the guard genuinely selects *which* ggml mapping applies are split into separate entries,
which is the point of the exercise. `python3 -m loom_mil_compiler.topology_ops` prints the whole table,
so the decision criteria are readable without tracing branch bodies:

```
gelu         when mode is EXACT/NONE                                -> _op_gelu_exact
gelu         when mode is TANH_APPROXIMATION/TANH                   -> _op_gelu_tanh_approx
gelu         when any other mode (rejected)                         -> _op_gelu_unsupported
less         when it is a provably-all-true length-validity mask    -> _op_less_always_valid
matmul       when transpose_x=False, transpose_y=True               -> _op_matmul_x_yt
matmul       when transpose_x=False, transpose_y=False              -> _op_matmul_x_y
matmul       when any other transpose combination (rejected)        -> _op_matmul_unsupported
reduce_mean  when the reduced axis has a statically-known size      -> _op_reduce_mean_scaled_sum
reduce_mean  when multi-axis / dynamic count off ne[0] (rejected)   -> _op_reduce_mean_unsupported
```

An unguarded rule acts as its op type's catch-all and is registered last, so the "intentionally
unsupported rather than silently wrong" rejections are now table entries in their own right rather than
`else:` clauses. `less` and `reduce_mean` have *only* guarded rules, so an op neither rule claims matches
nothing and reaches the generic path — a deliberate route in both cases, and the thing whose loss caused
both the bug this refactor introduced and the StyleTTS2 regression it later repaired.

`tools/loom_mil_compiler/test_topology_rules.py` pins the table's invariants: no rule may sit behind an
unguarded catch-all for the same op type (silently unreachable), and each guarded family must select the
composition its `when` text claims.

**Finding — the extraction fixed a latent shadowing hazard.** Inside the old loop, the `concat` branch
and the generic fallback both assigned a local `inputs = [...]`, clobbering the enclosing
`inputs` dict (the topology's declared inputs) for the remainder of the call. Nothing read it after the
loop began, so it was harmless in practice, but the two names are now genuinely separate scopes.

**Verification:** items 1 and 2 were verified together on one golden run — see the results table at the
end of item 2.

## Item 2 — centralized shape/const resolution pre-pass

**Status:** implemented.

New `tools/loom_mil_compiler/value_facts.py` is now the single place any "what is this Var's
compile-time value?" question is answered. It has two layers:

- **Literal statics** — module-level `static_value / static_array / static_scalar / static_ints /
  is_const_producer`. These replace the
  `x.val if x is not None and hasattr(x, "val") and x.val is not None else default` idiom, which was
  written out longhand at **55 call sites** across the op handlers and `_infer_dynamic_dim_expr`, each
  free to get the None-handling subtly differently (and several did — some checked `hasattr` without the
  `is not None`, so a null `.val` yielded `None` where the neighbouring site yielded its default).
- **Derived values** — the five mutually-recursive helpers that used to live on the exporter
  (`_try_derive_gather_shape_value`, `_resolve_scalar_expr`, `_resolve_range_scalar`,
  `_resolve_slice_axis_value`, `_try_resolve_reshape_shape_input`) moved onto `ValueFacts` as
  `gather_shape_value / scalar_expr / range_scalar / slice_axis_value / reshape_shape`, and are now
  **memoized per Var** on `exporter.facts`.

**Finding — memoization is a real fix, not just tidiness.** `scalar_expr` recurses into *both* operands
of every arithmetic op it walks, so on a diamond-shaped expression tree it re-walked shared subtrees
exponentially. Diamonds are ordinary here — the existing in-code comment records VITS's
`end = start + 2*length - 1` reaching the same `length` gather down two paths, and records that an
earlier "visited set" cycle guard turned that into a silent wrong answer. Caching the answer instead of
refusing to revisit it is the correct version of what that guard was reaching for, and it also makes the
answers stable by construction: two call sites asking about the same Var can no longer disagree, which
is precisely the divergence `slice_axis_value`'s docstring records having actually happened.

Caching lives on the exporter rather than per `generate_graph_topology` call: the derivation is pure over
an SSA graph that is immutable by then, so the cache is valid across all of a model's topologies.

**Scope note.** The proposal also cites the `producer.op_type == "const"` checks in
`apply_atomic_export` (old lines ~1493/1511) as part of the same duplication. Those turned out to be a
different question — a *structural* "is this operation a const node" test used for graph partitioning,
operating on `Operation` objects, not a "what is this value" resolution on a `Var`. Only the one that
does operate on a Var (`v.op and v.op.op_type == "const"`) was folded into `is_const_producer`; the other
two are left alone rather than forced into a shared abstraction they don't belong to.

### Results — items 1 and 2

Every export re-run from a pristine `git archive HEAD` baseline and from the refactored tree, then
snapshot-diffed (all metadata KVs, every topology JSON, the embedded driver Lua, and one
`name / shape / dtype / sha256` line per tensor):

| model | result |
|---|---|
| `vits_mil` | byte-identical |
| `kokoro_mil` | byte-identical |
| `qwen3_0.6b_mil_monolithic` | byte-identical |
| `lfm2_350m_monolithic` | byte-identical |
| `lfm2_350m_atomic` | byte-identical |
| `lfm2_350m_submodule` | byte-identical |
| `matcha_mil` | topologies + tensors byte-identical; `driver_script` differs by item 4 only |
| `supertonic_mil` | topologies + tensors byte-identical; `driver_script` differs by item 4 only |
| `styletts2_mil` | baseline could not export at all (regression above); now exports and passes every reference test |
| `conformer_ctc_small_mil_monolithic` | baseline could not export in any bounded time (see below); now exports in 40 s, `max abs diff = 1.6e-04` vs the reference forward |
| `parakeet_tdt_encoder_mil_monolithic` | same; now exports in 92 s, `max abs diff = 5e-06` |
| `parakeet_rnnt_encoder_mil_monolithic` | same; now exports in 86 s, `max abs diff = 1.0e-05` |

---

## The Conformer-CTC exporter blow-up (found and fixed)

Not part of `EXPORT-IMPROVEMENT.md`. It surfaced because the three NeMo exports would not finish, which
blocked items 1/2 verification — and it turned out to be the same bug class as item 2, in the one walk
item 2 hadn't covered.

**Symptom.** A full Conformer-CTC export never completed: >2 h at 96% CPU and 1.9 GB RSS, with both the
pristine `HEAD` baseline and the refactored tree equally stuck, so the refactor neither caused nor cured
it. Both were long past `Running MIL default pipeline: 100%` — coremltools was *done*; the time was all
in `generate_graph_topology`. `gdb` samples showed a shallow Python stack inside `PyObject_Str`.

**Measurement.** Truncating the encoder to N conformer blocks — a throwaway copy of the export script
that slices `model.encoder.layers[:N]` — and timing the exporter phase alone gives the shape of it
directly:

| encoder blocks | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| exporter, at `c64cbbb` | 0.6 s | 1.0 s | 1.4 s | 1.8 s | 2.3 s |
| exporter, at `e03bbac` | 0.2 s | 0.5 s | 1.3 s | 4.7 s | 84.8 s |

Linear (+0.4 s/block) versus roughly **3× per block**. Extrapolated to the real 16-block encoder that is
on the order of 10⁶ seconds — it was never going to finish.

**Bisect.** Timing every commit that touched `exporter.py` after the last Conformer commit, at 5 blocks:

| commit | subject | exporter |
|---|---|---|
| `c64cbbb` | Fix Conformer-CTC-small MIL export | 2.3 s |
| `319c029` | Export VITS via MIL compiler | 2.3 s |
| **`a29ffe5`** | **Add MIL-based export of Kokoro** | **84.8 s** |
| `24cb6a5` / `67a54a9` / `166be64` | Kokoro finish / StyleTTS2 / Matcha | ~87 s |

**Root cause.** `a29ffe5` made two changes to `_infer_dynamic_dim_expr` that are individually reasonable
and together quadratic-to-exponential:

1. it **removed the `id(var)` cycle guard** — correctly, and for a documented reason: the graph is an
   acyclic DAG and the guard was silently returning `None` on ordinary *diamonds* (Kokoro's SineGen
   reaches the same `rad_values` down two paths); and
2. in the same commit it **added a `concat` case that recurses into every operand**, making this the
   first branching walk in the function.

A branching walk over a DAG with no revisit-suppression re-derives every shared ancestor once per path.
That is exactly the failure the same commit message describes fixing in `_resolve_scalar_expr` for
VITS — the guard was removed there too. The difference is that `scalar_expr` got memoized during item 2
and this walk did not, because item 2 read the proposal's "const resolution" framing literally and
covered the *value* half while leaving the *shape* half alone.

**Fix.** `_infer_dynamic_dim_expr` is now a thin wrapper over
`_infer_dynamic_dim_expr_uncached`, memoized by `ValueFacts.dim_expr` on `(id(var), torch_axis)` —
the same treatment, and the same justification, as `scalar_expr`. Safe because the walk is pure in
`(var, torch_axis)`: its `_seen` parameter is still threaded through every recursive call site but has
not been *read* since `a29ffe5` deleted the guard. Caching is the correct form of what that guard was
reaching for — it suppresses the redundant revisit without ever turning a legitimate second visit into a
wrong answer.

**Result.** Exporter time is linear in depth again, and the full models export for the first time:

| encoder blocks | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| exporter, memoized | 0.1 s | 0.2 s | 0.3 s | 0.4 s | 0.5 s |

Full Conformer-CTC: **>2 h (never finished) → 40 s end-to-end**, of which 0.5 s is the exporter.

**Verification.** Output-preserving, and now checked against the real references rather than only
against another export:

| check | result |
|---|---|
| 2-block Conformer, memoized vs. pre-memo | **byte-identical** |
| the other 9 models re-exported | unchanged (6 byte-identical, matcha/supertonic `driver_script`-only, StyleTTS2 as above) |
| `test_e2e_conformer_ctc_mil_export` | 6/6, `max abs diff = 0.000164` vs `reference_forward_conformer.py` |
| `test_e2e_parakeet_tdt_mil_export` | 6/6, `max abs diff = 0.000005` |
| `test_e2e_parakeet_rnnt_mil_export` | 6/6, `max abs diff = 0.000010` |
| every StyleTTS2 / Matcha / Supertonic reference + driver test | unchanged, all passing |

**Worth noting for whoever reads this next:** an earlier version of this document concluded, from the
`PyObject_Str` stack samples and the 1.9 GB RSS, that the cost was in the *size* of the shape expression
strings and that memoization therefore could not help. That was wrong — the strings are large because
the same subexpressions are rebuilt over and over, so suppressing the rebuild fixes both. The stack
sample was real evidence pointing at string construction; the inference from it to "not a memoization
problem" was the error. The remaining idea from that paragraph still stands on its own merits though:
these expressions are algebraically trivial (the StyleTTS2 one reduces to `n_tokens`) and normalizing
them would shrink the emitted shape strings, which memoization does not do. The promising route for that
is to carry **sympy expressions** rather than strings — MIL's own `Symbol` already subclasses
`sympy.Symbol` and sympy is a declared coremltools dependency, so the exporter is currently stringifying
algebra it was handed and then rebuilding it worse by f-string concatenation. See BACKLOG.md for the
full write-up, including the two findings that bound it: MIL propagates compound expressions through
`reshape` but mints a fresh opaque symbol per `conv` (so the walk stays necessary), and
`src/core/symbol_env.cpp` parses only `+ - * /`, parentheses, `floor` and `sqrt` (so emission needs a
restricted printer, not `str(expr)`).

Beyond the golden diff, the Python suite (`test_topology_rules`, `test_value_facts`,
`test_iterative_export`, `test_scripted_loop`, plus the pre-existing `test_compiler`, `test_recurrent`,
`test_stft`, `test_tokenizer_detect`) runs 56 tests, all passing.

*(The three NeMo rows in the table above were originally verified against a 2-block-truncated encoder,
because the full exports would not complete. That workaround is no longer needed — see the blow-up
section immediately below — and the full models are now exported and checked against their real
reference forwards.)*

## Item 3 — explicit control-flow capture before tracing

**Status:** feasibility established empirically; results below.

The proposal's premise is that MIL can represent a real loop and the gap is only that the source models'
Python loops are plain imperative code. That is correct, but the route there is narrower than the
proposal assumes, and the supporting evidence it cites does not actually cover it.

**Finding — the GRU evidence does not generalize.** `recurrent.py`'s docstring notes that tracing a
`torch.nn.GRU` yields a genuine MIL `while_loop`. That comes from coremltools' *dedicated `torch.nn.GRU`
op lowering*, which builds the `while_loop` in MIL directly. It says nothing about whether a
user-written Python loop can reach the same place, because a traced Python loop never becomes a
TorchScript `prim::Loop` at all — `torch.jit.trace` unrolls it before coremltools ever sees it.

**Finding — `torch.jit.script` does work, but only with a compile-time-constant trip count.** Probing
coremltools 9.0 / torch 2.8 with a 4-step loop-carried-state module (throwaway probe; the module is
`RefinementLoop` in `scripted_loop.py`, converted each way and the MIL op histogram compared):

| formulation | result |
|---|---|
| plain Python `for`, `torch.jit.trace` | 30 MIL ops, `linear=4` — **unrolled**, no control flow (status quo) |
| `torch.jit.script`, `while i < self.n` with an `int` counter | **fails** — `cast` to dtype `str` |
| `torch.jit.script`, `for _ in range(self.n)`, `n: int` annotation | **fails** — `less(x: int32, y: str)` |
| `torch.jit.script`, `for _ in range(4)` literal | **19 MIL ops, `linear=1`, `while_loop: 1`** ✅ |
| `torch.jit.script`, `for _ in range(self.n)` with `__constants__ = ["n"]` | **19 MIL ops, `linear=1`, `while_loop: 1`** ✅ |

So a real MIL `while_loop` with loop-carried tensor state is reachable, and the trip count can still be a
module parameter — but it has to be a TorchScript *compile-time constant* (`__constants__`, or a source
literal). Any trip count that stays a runtime-typed `int` hits a coremltools frontend bug that mis-types
the loop bound as `str`.

That constraint is the decisive one, and it cuts deeper than it first looks: the step count has to be
**baked at export time**, and *none* of the affected drivers work that way. Matcha and Supertonic take
`n_steps` as a runtime driver input, StyleTTS2 takes `diffusion_steps`, and VITS/Supertonic's
duration-predictor loops derive their length from predicted durations, which is not knowable at export
time even in principle. Capturing any of these as a MIL `while_loop` today would mean freezing a
parameter callers currently vary.

**Consuming a `while_loop` downstream.** `exporter.py`'s `transpile_operation` already lowers
`while_loop`/`cond` into driver IR (`While` + `Break`, `If`), so the *driver* path can consume one in
principle. `generate_graph_topology` cannot and should not: a static topology is a fixed node list, so a
loop body would have to become its own topology the driver calls per step, and building that split is
real unimplemented work — item 4 deliberately did not go that way (see its own findings below).

### Follow-up: baking the trip count for tracing, then re-opening it at conversion time

A reasonable proposal, since constraint 1 above is the whole blocker: fix the step count as a constant
*only* so the loop survives scripting, then rewrite it back into a runtime argument during MIL → ggml
conversion. Probed directly (throwaway probes; their full output is reproduced below so they need not be rebuilt). The emitted
MIL is:

```
while_loop(loop_vars = [state_iter=0 (i32), state (1,8), t=[0.0]])
  COND:  less(state_iter, y=const 4)   <- the trip count
  BODY:  linear -> tanh -> mul -> add  [the estimator, NOT unrolled]
         add(t, dt) ; add(state_iter, 1)
    => outputs (state_iter+1, state', t')
```

**Two of the three prerequisites hold.** The trip count is a *single* identifiable const (`less.y`) in
the cond block, trivially swappable. And the loop-carried vars are explicit and positionally matched to
the body's outputs — meaning an `IterativeRefinementSpec` could be *inferred* from the trace instead of
declared, which is exactly the item 3 → item 4 chain the proposal envisioned.

**One hazard, solvable.** `dt = 1.0 / self.n_steps` gets constant-folded into the body as a literal
(`mul(y=0.25)`, `add(t, 0.25)` for N=4) in more than one place. Swapping only the loop bound would leave
the integration silently using a step size for the old N, and the folded floats carry no provenance to
find them by. Fix is source-side and clean: take `dt` as a tensor input so it stays symbolic — verified,
after which the only float consts remaining in the body are the estimator's real weights.

**The blocker moves to the body block, and it is structural.** Feeding the body's operations to
`generate_graph_topology` yields a **1-node** topology, not the estimator: a MIL loop body has one output
*per loop-carried var* (`[state_iter_x0_inc, state.7, t.7]`), while the exporter takes `ops_list[-1]`'s
output as *the* topology output — here the counter increment — and `_prune_dead_nodes` then correctly
removes everything unreachable from it, i.e. the whole network. That is not a bug to patch around: the
engine's convention really is one output tensor per topology (`loom.run_subgraph` returns data + shape,
see `submodule_export.py`'s `_flatten_call` comment).

Making it work therefore needs either multi-output topologies in `GraphBuilder`/`run_subgraph` (an engine
change), or a classifier that routes *scalar* loop vars (the counter, `t`) to host Lua and keeps only the
tensor state as the single topology output — plus driver synthesis from `loop_vars` and per-model
numerical re-verification. Note where option two lands: driver owns the counter and the time, topology
computes the one tensor. That is exactly what the hand-written drivers already do, which is the honest
summary of the whole direction — **its payoff is inferring the spec rather than declaring it, not a
better runtime shape.** Weighed against replacing a ~13-line declarative spec that already carries
export-time validation, that is not worth building yet. Revisit if a model turns up whose loop cannot be
expressed host-side at all.

## Item 4 — second family template for iterative-refinement models

**Status:** implemented for the Euler-CFM sampler family (Matcha + Supertonic); StyleTTS2's ADPM2
sampler and the VITS/Kokoro duration loops deliberately left bespoke.

New `tools/loom_mil_compiler/iterative_export.py` provides `IterativeRefinementSpec` +
`render_driver`, the analogue of `SubmoduleExportSpec`. A spec declares the six things that actually
differ between the two models — estimator topology, loop-carried input name, scalar-time input name, the
per-step-constant inputs, and (at the call site) the state's element count and the `n_tokens` to build at
— and the generated Lua sampler replaces a `--@loom:samplers` marker in the hand-written driver.

`export_matcha_mil.py` and `export_supertonic_mil.py` now declare their samplers instead of hand-writing
them; each driver's loop collapses to one call:

```lua
local z = sample_decoder(t_mel, t_mel * n_feats, inputs.n_steps, { mu = mu_y })
local z = sample_vfe(t_lat, t_lat * lat_dim, inputs.n_steps, { txt_emb = txt_emb, stl_emb = inputs.style_ttl })
```

**The proposal's predicted payoff did not materialize, and that is worth stating plainly.** Item 4
expected a shared template to "shrink these the same way `SubmoduleExportSpec` shrunk
`export_qwen3_mil.py` to 28 lines", citing `export_styletts2_mil.py`: 340 lines and
`export_kokoro_mil.py`: 503 lines. Measured before/after:

| file | before | after |
|---|---|---|
| `export_matcha_mil.py` | 432 | 448 |
| `export_supertonic_mil.py` | 237 | 252 |
| `export_styletts2_mil.py` | 340 | 350 |
| `export_kokoro_mil.py` | 503 | 503 (untouched) |
| `matcha_driver_mil.lua` | 101 | 96 |
| `supertonic_driver_mil.lua` | 65 | 55 |
| `styletts2_driver_mil.lua` | 368 | 368 |

Total line count went **up** (+41 in the export scripts, −15 in the drivers). The reason is that the
340/503 lines the proposal points at are *tracing setup* — wrapper modules, dummy inputs, `RangeDim`
declarations, per-phase topology assembly — not loop orchestration. A sampler template cannot touch any
of that. The thing it replaces was ~13 lines of Lua per driver. The `SubmoduleExportSpec` analogy does
not carry: that one subsumed a whole export script because decoder-LLMs share their *entire* structure,
whereas these three share one loop inside otherwise unrelated pipelines.

**What was actually delivered** is `validate_against_topology`: the spec is cross-checked against the
estimator's *real* declared inputs at export time and raises naming the exact mismatch. Supplying an
input the topology never declared, or omitting one it did, is otherwise only caught deep inside the
engine at run time with nothing pointing back at the line that got it wrong. That is a genuine
improvement and it does mirror `SubmoduleExportSpec`'s "a wrong attribute path raises immediately"
property — but it is a different benefit from the one the item predicted, and it is the reason this is
a spec rather than a shared Lua helper function.

**Verification.** Unlike items 1 and 2 this *is* meant to change the output — the embedded driver script
differs. Everything else must not, and the numbers must not move at all:

| check | result |
|---|---|
| `matcha_mil.gguf` diff vs baseline | **only** `model.driver_script`; every topology and tensor byte-identical |
| `supertonic_mil.gguf` diff vs baseline | **only** `model.driver_script`; every topology and tensor byte-identical |
| `test_e2e_matcha_mil_lua_driver` (vs the `loom::MatchaDriver` C++ oracle) | 6/6, `max_abs_diff=0.0104436, rmse=0.000677779` — **identical to the last digit**, before and after |
| `test_e2e_supertonic_mil_lua_driver` (vs the `loom::SupertonicDriver` oracle, 70656-sample waveform) | 10/10, `max_abs_diff=6.35488e-06` — **identical**, before and after |

**Finding — item 4 does not actually want item 3.** The proposal states item 4 "depends on item 3",
i.e. the loop should first become a structural unit in the trace. It turned out to be the wrong
dependency: item 3's `while_loop` route cannot express a *runtime* step count (see its constraint 1),
whereas every one of these drivers takes `n_steps` as a runtime input and always has. Keeping the loop
host-side and declaring it in Python gets the generalization the proposal wanted *and* keeps the runtime
step count; capturing it into MIL would have traded that away for no benefit the engine can currently
use. Item 3's work stands on its own as a documented, tested capability — it is just not this item's
prerequisite.

**Deliberately not generalized: the integration rule.** Both retrofitted models use deterministic
forward Euler with uniform `dt = 1/n_steps`. StyleTTS2's is ADPM2 over a Karras sigma schedule — read
directly from `styletts2_driver_mil.lua` rather than assumed: a second-order sampler with **two** network
evaluations per step, per-step noise injection at `sigma_up`, and real preconditioning math
(`c_skip`/`c_out`/`c_in`/`c_noise`) wrapped around the call. VITS/Kokoro's duration loops are a scatter
over predicted durations, not an ODE at all. Forcing those through one template would produce something
harder to read than the loops it replaced.

**But the validation generalizes further than the codegen does**, which is worth separating rather than
conceding. A bespoke sampler's `run_subgraph` call has the *same* failure mode as a generated one: an
argument name that doesn't match the topology's declared inputs is only caught inside the engine at run
time. So `EstimatorSpec` — the plain "this topology, these inputs" declaration —
is its own type, `IterativeRefinementSpec.estimator_spec()` returns one (single validation
implementation, no drift), and `render_driver(..., estimators=[...])` checks calls that generate nothing.
`export_styletts2_mil.py` now declares its ADPM2 loop's `diffusion` call that way and keeps the loop
hand-written; the emitted GGUF is unchanged, byte for byte, by adding the check.

`tools/loom_mil_compiler/test_iterative_export.py` covers both halves — the emitted Lua's shape, and
every rejection path of the validation.

---

## Repairing the StyleTTS2 regression

Not part of `EXPORT-IMPROVEMENT.md`, but done here because the new table made the fix a one-line change
in shape rather than a rewrite.

**What the op actually is.** Instrumenting the diffusion phase — monkeypatching the `reduce_mean` rule
to print each op's resolved axis before delegating — showed exactly one `reduce_mean` in it, and its
facts decide everything:

```
reduce_mean x_full.9: torch_axis=-1  ne_axis=0  keep_dims=False
                      size='(floor(((1) * (n_tokens) * (512)) / ...))'   <- live, n_tokens-derived
```

`ne_axis=0` is the whole point. `ggml_mean` reduces `ne[0]` *and divides by `ne[0]` at run time*, so it
needs no export-time count for this case — which is why the generic `OP_MAP` `MEAN` lowering had always
handled it, and why the dedicated handler's blanket "must be a static architecture constant" was too
strong rather than wrong in principle.

**The fix.** `reduce_mean` now splits three ways by guard, with the middle case being the restored one:

| condition | route |
|---|---|
| reduced axis has a statically-known size | `_op_reduce_mean_scaled_sum` — `REDUCE_SUM` + `SCALE(1/N)`, works for any ne axis (Matcha's ne[1] LayerNorm) |
| size dynamic **and** the axis is `ne[0]` | *no rule matches* → generic `MEAN`, which supplies its own count |
| size dynamic and the axis is not `ne[0]` | `_op_reduce_mean_unsupported` — genuinely unrepresentable, rejected with a message naming the axis |

A shared `_reduce_mean_plan()` computes `(ne_axis, keep_dims, static_n)` once for both guards and the
handler, so they cannot disagree about which case they are in — the same pattern as `_matmul_transposes`.

**Verification.**

| check | result |
|---|---|
| `export_styletts2_mil.py` | exports cleanly; `diffusion: 153 nodes, 139 weights` — the exact counts the last known-good `67a54a9` export produced |
| diffusion topology JSON vs `67a54a9` | **byte-identical** |
| `test_e2e_styletts2_mil_diffusion_reference` (vs the real checkpoint) | 9/9, `mean_diff=5.35e-07, max_diff=2.86e-06` |
| `test_e2e_styletts2_mil_albert_reference` | 11/11, `max_abs_diff=1.10e-04` |
| `test_e2e_styletts2_mil_decoder_vocoder_reference` | 21/21, `max_abs_diff=2.98e-02` |
| `test_e2e_styletts2_mil_lua_driver` (full pipeline) | 22207/22207 |
| every other model re-exported | unchanged (see the results table above) |

**Why Conformer-CTC and Parakeet cannot be affected**, without needing a re-run: the *old* code raised on
any non-static reduction count, so an export that succeeded is proof that every one of its `reduce_mean`
ops had a static count — and for those, the new guarded rule runs the identical composition on the
identical plan. The change can only alter models that previously failed to export.

## Conclusion of the thread: family templates, not universal inference

The stated goal behind items 3 and 4 was to infer the orchestration graph from the trace
"ONNX-exporter-style, for virtually any architecture". That goal does not reach, and the reasons are
worth stating precisely, because two of the three are not what one would guess up front:

1. **The orchestration usually isn't in the traced module.** For all four models named in item 3, the
   loop lives in demo/inference code, not in any `forward()`. There is nothing in the trace to infer
   *from* unless a scriptable wrapper containing the loop is written first — and writing that wrapper
   costs about what the six-line spec costs. Inference pays off only when the real model's own `forward`
   already contains the orchestration.
2. **A captured loop doesn't map onto a topology anyway.** A MIL loop body has one output per
   loop-carried var; the engine has exactly one output tensor per topology. See the item 3 follow-up
   above.
3. **Roles are not always recoverable from shapes, even in principle.** Matcha's `decoder` declares
   `z` and `mu` with *identical* shapes `[n_tokens, 80, 1]`, and its output has that shape too — so
   nothing about the topology reveals which input is the loop-carried state. Supertonic's `z_t` is
   separable only incidentally (it happens to be the one with a dynamic axis). A `while_loop`'s
   `loop_vars` *does* state it outright, which is the one concrete thing that route buys — worth
   remembering if inference is ever revisited.

**What does generalize is the per-family template**, and there are now two working examples worth
copying the shape of rather than inventing a third style:

| | `SubmoduleExportSpec` (decoder-LLMs) | `IterativeRefinementSpec` (Euler-CFM samplers) |
|---|---|---|
| declares | prefix / repeated / suffix / aux boundaries | estimator, carried input, time input, fixed inputs |
| cross-checked against | `find_repeated_blocks()` re-derives the repeated blocks structurally and rejects a spec that claims a non-qualifying attribute | `validate_against_topology()` re-reads the estimator's real declared inputs and rejects any mismatch |
| concedes | non-causal-LM architectures | ADPM2 / duration-scatter loops (which still declare an `EstimatorSpec` for the check) |

The pattern both share, and the one to reuse for the next family: **declare only what genuinely varies,
re-derive everything else structurally, and make the spec fail loudly at export time when its claim and
the real model disagree.** That is what makes a template worth more than the hand-written code it
replaces — not the line count (see item 4's own measurements, where the count went up).

That candidate next family — NeMo-style ASR encoders — is now built; see the next section. It is the
third worked example of the pattern, and the first one where "re-derive everything else structurally"
*removed* a field the plan had assumed was real.

## Third family template — NeMo ASR encoders (Conformer-CTC, Parakeet-TDT, Parakeet-RNNT)

**Status:** implemented; all three exports byte-identical to their pre-template baselines.

`tools/loom_mil_compiler/nemo_asr_export.py` holds `NeMoASREncoderSpec` + `EncoderOutput` +
`export_nemo_asr_encoder`. The three export scripts are now a docstring, a spec, and a call:

```python
SPEC = NeMoASREncoderSpec(
    checkpoint="/home/flavio/Dev/models/conformer-ctc-small/stt_en_conformer_ctc_small.nemo",
    output=EncoderOutput.CTC_LOG_PROBS,
    architecture="conformer-ctc",
    output_path="conformer_ctc_small_mil_monolithic.gguf",
)
```

**Finding — only three of the five predicted fields were real.** BACKLOG.md's entry listed five things
differing between the scripts. Two dissolved on contact:

* **The restore class is not a variable.** `EncDecCTCModel.restore_from` and `ASRModel.restore_from`
  return the *identical* concrete class for the Conformer-CTC checkpoint —
  `nemo...ctc_bpe_models.EncDecCTCModelBPE`, with identical state-dict keys (checked directly against
  the real checkpoint, not assumed). NeMo's `restore_from` dispatches on the config's own `target`, so
  naming a subclass never selected anything; all three now load through the base `ASRModel`.
* **The wrapper's return value became a claim to be checked rather than an expression to be declared.**
  `EncoderOutput` has two members, each naming a *model family* and carrying why that family's graph
  boundary falls where it does (a CTC decoder is one 1x1 conv and stays in the graph; an RNNT/TDT
  prediction net is an autoregressive LSTM + joint and stays host-side). Each knows its own `forward`
  arity, which axis carries channels, and where the expected channel count is read from
  (`decoder.num_classes_with_blank` vs `cfg.encoder.d_model`), so pointing a `CTC_LOG_PROBS` spec at an
  RNNT checkpoint raises `NeMoASREncoderSpec declares output=CTC_LOG_PROBS, whose family's forward()
  returns 3 values, but EncDecRNNTBPEModel.forward() returned 2` — before the encoder is traced.

Everything else moved into the template as family-wide fact rather than per-script boilerplate: the
sample rate is read from `model.cfg.preprocessor.sample_rate` and the trace length (1 s) and `RangeDim`
bounds (0.1 s .. 20 s) derived from it instead of three copies of `16000`/`1600`/`320000`; the
`transformers.dependency_versions_check` stub and the `TMPDIR` routing (NeMo untars a multi-GB
checkpoint, and `/` has ~2 GB free) are one documented function; and `compute_precision=FLOAT32` is
stated once, as the load-bearing CONV_2D-precision fix it is, rather than three times as a comment.

**Finding — validation has to run on the model's own output, not on the wrapper's.** The first version
validated the tensor the wrapper *returns*, which meant binding it to a local (`selected = ...`). That
alone changed both Parakeet exports: `torch.jit.trace` names a traced value after the Python name it is
bound to, so the topology's declared output went from `var_4640` to `selected` — otherwise byte-identical.
The golden diff caught it. Validation now inspects `outputs[0]` in NeMo's own `(B, D, T)` layout (hence
`channel_axis`), and `select()` is a single expression with no intermediate name.

**Verification.** Re-exported all three from the template — with the exporter itself held at `HEAD`, so
the diff could only be the template's doing — and snapshot-diffed against a pristine `git archive HEAD`
baseline: `conformer_ctc_small_mil_monolithic`, `parakeet_tdt_encoder_mil_monolithic` and
`parakeet_rnnt_encoder_mil_monolithic` are **byte-identical**, every metadata KV, the whole topology JSON
and every tensor digest. The three reference tests still pass at their existing tolerances
(`0.000164` / `0.000005` / `0.000010`). `tools/loom_mil_compiler/test_nemo_asr_export.py` (13 tests)
covers the validation paths without needing NeMo or a checkpoint — a fake model reproduces each mismatch
shape, including the headline "CTC spec pointed at an RNNT checkpoint" case.

## Symbolic shape expressions carry sympy, not strings

**Status:** implemented (the first of BACKLOG.md's three open follow-ups).

New `tools/loom_mil_compiler/shape_expr.py`. The derivation walk in `_infer_dynamic_dim_expr_uncached`
and every derived value in `value_facts.py` now compose **sympy expressions**, and `render()` turns one
into text exactly once, where a JSON attribute is emitted. The size of the improvement is easiest to see
on Conformer-CTC's STFT frame count, which appears in dozens of shape attributes:

```
before  (floor((((floor(((1) * (1) * ((((floor(((1) * (((1)+(((n_tokens) - (1)))))) / ((1) * (1)))))
         + 512))) / ((1))))) + 0 - 512) / 160) + 1)
after   floor(n_tokens/160) + 1
```

Three things make it work, and each is load-bearing rather than stylistic:

* **Shape symbols are declared positive integers** (`shape_expr.symbol`, which also interns them).
  `floor(512*n_tokens/512)` only reduces to `n_tokens` because sympy knows the symbol is an integer; and
  because assumptions participate in sympy's structural equality, a bare `sympy.Symbol("n_tokens")` built
  anywhere else would compare unequal and silently stop cancelling.
* **`render` is a restricted printer, not `str(expr)`.** `src/core/symbol_env.cpp` accepts only
  `+ - * /`, unary minus, parentheses, identifiers, numbers, `floor()` and `sqrt()`, evaluated in
  `double`. Sympy's default printer emits `**`, `ceiling`, `Min`/`Max` and `Piecewise` happily, none of
  which parse — so `render` maps onto that grammar and raises `UnsupportedShapeExpression` naming the
  construct, rather than shipping text that fails at model load. Integer powers become repeated products;
  `sqrt` is the only function besides `floor`.
* **`floor` arguments are recombined with `sympy.together` before printing.** Sympy distributes a
  rational coefficient over a sum on construction, so `floor((n_tokens - 512)/160)` *becomes*
  `floor(n_tokens/160 - 16/5)`. Mathematically identical, but the engine evaluates in floating point,
  where the distributed form takes three roundings inside the floor instead of one — enough to flip it at
  an exact boundary.

`parse()` implements the same grammar (hand-written, mirroring the C++ evaluator rather than using
`sympify`), so `parse(render(e))` round-trips by construction. That is what lets a call site holding an
already-rendered dim string — `slice_by_index`'s `(end - begin)` and byte offsets, `split`'s stride
products, `tile`'s target shape — lift it back into algebra and keep composing instead of concatenating
strings. Those offsets collapse too; from the same Conformer-CTC topology:

```
before  ((0 + (((5000) - ((floor((((floor((((floor((((floor(((1) * (1) * ((((floor(((1) * (((1)+(((n_tokens)
         - (1)))))) / ((1) * (1))))) + 512))) / ((1))))) + 0 - 512) / 160) + 1)) + 2 - 3) / 2) + 1)) + 2 - 3)
         / 2) + 1))) * (1 * 176))) * 4)
after   3519296 - 704*floor(floor(floor(n_tokens/160)/2)/2)
```

**Finding — a heuristic was keying on an expression's *spelling*, and normalization exposed it.**
`slice_by_index`'s derivation treats an `end` that comes back as bare `n_tokens` as "the walk bottomed
out inside the `end` chain, trust `x`'s own extent instead" — correct, and necessary, for the two
Conformer-CTC cases it was written for (an `end` derived from the length input's runtime *value*, which
no shape expression can express). The test for "bottomed out" was a comparison against the literal string
`"n_tokens"`, and it worked only because a *genuinely derived* sequence length happened to come out
spelled `floor((n_tokens + 0 - 1) / 1) + 1` instead. Simplification made the two spellings identical and
the accident stopped working — in two models at once, both silently:

| model | `end` (genuinely derived) | `x`'s own extent, wrongly substituted |
|---|---|---|
| SupertonicTTS VFE | `n_tokens` — a const `text_attn.increments` table of declared shape `(1, 1000, 1)` cropped to `[:, :n_tokens]` | the literal **1000**, baking the trace-time length into 16 downstream shapes |
| VITS `logw`/`stats` | `n_tokens` — the relative-position reshape's own axis | **`n_tokens + 1`**, one element longer than the parent it views |

The fix is to test the answer's **provenance** instead of its spelling: `ValueFacts` now records, next to
each derived scalar, whether it rests on the producer-less-var fallback (the single place a `n_tokens` is
guessed rather than derived) and propagates that flag through the passthrough/select/arithmetic cases, so
`facts.slice_axis_is_guess()` answers the question the heuristic was actually asking. A `gather(shape(x))`
derivation is never a guess, however much its answer looks like one. Both models are back to their
baseline values, and the Conformer-CTC cases still take the override.

This is a pre-existing bug in the heuristic, not a sympy artifact — normalization only made it reachable.
It is also the one that justifies the whole verification approach below: nothing about either export
failed, and both wrong shapes were still *shapes*.

**Verification — the diff is read, not required to be empty.** This change rewrites every symbolic shape
attribute by design, so `diff -r` cannot be the gate. New `tools/loom_mil_compiler/compare_snapshots.py`
walks two snapshot trees in parallel and, for every differing value, evaluates **both** sides at 18
concrete sequence lengths (odd, even, off-by-one neighbours, and the ends of the declared dynamic range)
using the same grammar the engine uses; anything that is not numerically equivalent at every probe — or
that is not an expression pair at all (a renamed node, a changed op, a different node count, a moved
tensor) — is reported as structural. That check is what found both bugs above, in a diff of 4,081
rewritten expressions across 12 models where reading by eye would not have: 28 wrong values, each one a
plausible-looking shape string sitting among hundreds of correct ones.

Final state, all 12 models re-exported from the working tree against the `git archive HEAD` baseline:

| model | expressions rewritten | structural differences |
|---|---|---|
| `conformer_ctc_small_mil_monolithic` | 294 | 0 |
| `parakeet_tdt_encoder_mil_monolithic` / `..._rnnt_...` | 426 / 426 | 0 / 0 |
| `qwen3_0.6b_mil_monolithic` | 448 | 0 |
| `lfm2_350m_monolithic` / `_atomic` / `_submodule` | 176 / 176 / 200 | 0 |
| `vits_mil` | 531 | 0 |
| `kokoro_mil` | 263 | 0 |
| `matcha_mil` | 419 | 0 |
| `supertonic_mil` | 443 | 0 |
| `styletts2_mil` | 279 | 0 |

And numerically, against the real reference fixtures rather than against another export — every MIL
reference test at the tolerance it already had:

| test | result |
|---|---|
| `test_e2e_conformer_ctc_mil_export` | passed, `max abs diff = 0.000164` |
| `test_e2e_parakeet_tdt_mil_export` / `_rnnt_` | passed, `0.000005` / `0.000010` |
| `test_e2e_lfm2_mil_export` (atomic + monolithic) | passed, every greedy top-1 token matched |
| `test_e2e_vits_mil_flow_vocoder_reference` | passed, `max_abs_diff = 5.7e-07` at Tp=194 |
| `test_e2e_matcha_mil_text_encoder` / `_decoder` / `_vocoder` | passed, `1.2e-04` / `4.5e-04` / `4.2e-05` |
| `test_e2e_supertonic_mil_dp` / `_vfe` | passed, `v_max_abs_diff = 5.1e-06` |
| `test_e2e_styletts2_mil_albert_reference` / `_diffusion_` / `_decoder_vocoder_` | passed, `1.1e-04` / — / `2.98e-02` |
| `test_e2e_kokoro_mil_albert_bert_encoder_reference` / `_decoder_vocoder_` | passed |

Full `ctest`: 139 tests, **138 pass**, 34 of them skipping for fixtures this machine does not have. The
single failure, `test_e2e_lfm2_lua_driver`, is unrelated and pre-existing: it hardcodes
`GgufModel::load("lfm2_350m.gguf")` — a *bespoke*-converted artifact (not a MIL export) that is not in
the tree, with no env var and no skip guard, so it aborts rather than skipping. It does not touch the
exporter path.

## Bug found in this work (and fixed)

Worth recording because the mechanical extraction in item 1 looked safe and was not.

Exactly one of the 33 dispatch blocks — `less` — did **not** end by transferring control. Its `continue`
was nested inside an `if bypass_ok:`, so when the bypass did *not* apply, control fell off the end of the
block and reached the generic `OP_MAP` path, which emitted the real `LESS` comparison node. Lifting the
block into a handler silently turned that fall-through into "handled, emit nothing", dropping three
`LESS` nodes from Kokoro's `decoder_vocoder` topology (and three from StyleTTS2's). The golden diff
caught it; nothing else would have.

The fix is the shape the table wanted anyway: the whole bypass derivation became a **guard**
(`_less_is_always_valid_mask`), so when it rejects, *no rule claims the op* and the generic path is
reached by construction rather than by falling off the end of a block. An AST check over the original
`HEAD` confirmed `less` was the only block with this property.

The same extraction also lost a `re` import that only that block used — latent, because the code path
needing it is only reached by Conformer-CTC and Parakeet, which had not yet been re-run at that point.
