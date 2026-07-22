# Export/Compiler Backlog

Follow-up work identified while stabilizing the `add-conversion-via-mil` branch (LFM2-350M atomic +
monolithic export). Ordered by what's most load-bearing first, not necessarily execution order — see each
item's own recommendation.

---

## Status snapshot (items 2/3/5/6, across two sessions)

Quick-reference for what's actually done vs. still open, before diving into each item's own detail below.

| Item | Status |
|---|---|
| 1. Numerical correctness (attention) | RESOLVED (prior session) |
| 2. Driver IR/codegen | **DONE** — `driver_ir.py` landed, all exporters rewired, 2 real bugs caught & fixed |
| 3. Dynamic shapes | **Mechanism substantially hardened, 7 real bugs fixed, 1 deep issue remains** (see item 3) |
| 4. Tokenization | Not started (unchanged from original plan below) |
| 5. MIL primitive review | **Concrete bullets DONE** (incl. a real `op_equal` algebra bug found this round); broader heuristic audit deliberately deferred |
| 6. Export-time quantization | **Code DONE**; LFM2-specific numerical verification still blocked on item 3's last issue |

**What's actually missing right now, in dependency order:**
1. **The one open bug** (item 3): a null-function-pointer crash inside a ggml worker thread, bisected to a
   precise 5-node repro (RoPE's inverse-frequency `MUL_MAT` against a real GGUF-loaded weight). Everything
   below is blocked on this. See item 3's own section for the exact repro, everything already ruled out, and
   the prime remaining suspects — a real debugger (`gdb`/`lldb`, neither available in this environment)
   would very likely resolve it quickly via a core dump backtrace.
2. **A fresh numerical-correctness pass** (item 1's own bisection-against-real-HF technique, at a genuinely
   dynamic, non-128, unpadded length) once (1) stops crashing — a shape/build fix alone only proves the
   graph builds, not that the values are right.
3. **LFM2-specific quantization verification** (item 6): the code is written and mirrors an already-proven
   pattern (Qwen3-0.6B-Base), but has never actually been run against a working LFM2 export + compared,
   because (1)/(2) block producing one.
4. **Item 5's "broader ask"**: audit `primitives_basic.cpp`'s ADD/MUL/MUL_MAT/REPEAT layout-healing
   heuristics for continued necessity now that the exporter emits correct layouts directly for more cases —
   deliberately not started, since these heuristics are shared by every model using these primitives
   (Whisper, Conformer-CTC, VITS, Matcha-TTS, SupertonicTTS, Kokoro), not just LFM2's MIL export path, so
   removing one needs per-model verification.
5. **Item 4 (tokenization)**: not started at all — still just the recommendation/plan below, no code written.

`tests/test_e2e_lfm2_mil_export.cpp` (already committed) is the regression test for (1)+(2) — it'll go from
skipping (no fixture present) to passing once the crash is resolved and values are confirmed; no test
changes needed to make that happen.

---

## 1. Numerical-correctness investigation (attention scores) — RESOLVED

**Status:** root-caused and fixed. Both atomic and monolithic LFM2-350M exports now match a real HF
forward pass exactly (top-10 tokens identical at every checked position; max abs logit diff ~0.003 across
all 65536 vocab entries at seq position 2, consistent with fp16 intermediate rounding, not a correctness
bug).

**Root cause:** `tools/loom_mil_compiler/exporter.py`'s generic `MUL_MAT` handling forwarded MIL's own
`matmul(x, y, transpose_x, transpose_y)` operand order straight into `ggml_mul_mat(x, y)`, silently
ignoring the `transpose_x`/`transpose_y` attributes entirely. `ggml_mul_mat(A, B)` has different semantics
from MIL's `X @ Y` (it always contracts over `ne0` of both operands and returns `ne=[A.ne1, B.ne1, ...]`,
i.e. computes `B_mat @ A_mat^T`, not `A_mat @ B_mat`), so getting the right numerical result requires
choosing the correct operand order — and sometimes an explicit transpose — based on `transpose_x`/
`transpose_y`, not just passing `x, y` through unchanged. Confirmed directly by monkeypatching
`coremltools`'s own `_decompose_scaled_dot_product_attention` to inspect the actual MIL ops: LFM2's SDPA
decomposition emits `matmul(x=query, y=key, transpose_y=True)` for the attention-score matmul and
`matmul(x=softmax, y=value)` (no transpose) for the score×value matmul — both of which the old code-path
mishandled.

**Fix (`tools/loom_mil_compiler/exporter.py`):** added a dedicated `op_type == "matmul"` composition
(alongside the existing dedicated `"linear"`/`"transpose"`/etc. handling) that explicitly derives the
correct `ggml_mul_mat` call from `transpose_x`/`transpose_y`:
- `transpose_x=False, transpose_y=True` (attention scores): emit `MUL_MAT(y, x)` — key-first, matching
  the standard llama.cpp attention convention. Both operands already share `ne0` in their natural layout,
  so no extra transpose node is needed.
- `transpose_x=False, transpose_y=False` (score×value): `y` needs its leading two `ne` axes swapped and
  made contiguous before it can be `ggml_mul_mat`'s first operand — composed as explicit `PERMUTE(axes=
  [1,0,2,3])` + `CONT` nodes ahead of the `MUL_MAT`, rather than relying on C++-side shape guessing.
- Any other `transpose_x`/`transpose_y` combination raises `NotImplementedError` rather than silently
  doing the wrong thing (neither combination occurs in LFM2's SDPA decomposition today).

**Second bug uncovered by the fix (`src/ops/primitives_basic.cpp`, `op_add`):** removed an "axis 0 and 1
swapped" layout-healing heuristic (permute+cont whichever ADD operand looked transposed relative to the
other, judged by `a.ne[0]==b.ne[1] && a.ne[1]==b.ne[0]`). It had been added as a band-aid for this exact
bug in a prior commit. Because the attention-score tensor and the causal-mask constant are both `128×128`
in the common case, that shape check can't distinguish "needs a swap" from "already correct" — so once the
exporter started emitting the right `MUL_MAT` layout directly, this heuristic started **re-corrupting an
already-correct tensor** instead of fixing a broken one (confirmed: removing it was necessary for
`add_0`/`softmax_0` to read correctly after the exporter fix). Left the analogous heuristics in `op_mul`
and `op_repeat` untouched — neither is exercised by the fixed code path, and removing them wasn't
empirically justified the way `op_add`'s was.

**Verification method (reusable for future bisection):** built a throwaway C++ harness (not committed)
using `loom::GraphBuilder` directly against the exported GGUF — no Lua involved — that (a) truncates a
copy of `layer_2`'s topology JSON at a chosen intermediate node name and reads back the raw output tensor,
and (b) chains `embedding` → `layer_0..15` → `model_model_embedding_norm` → `output_head` (atomic profile)
or runs the single monolithic topology directly, comparing final logits against a real HF forward pass.
HF-side ground truth was captured by monkeypatching `transformers.integrations.sdpa_attention
.sdpa_attention_forward` to snapshot its raw `query`/`key`/`value` inputs (pre-GQA-tile, pre-causal-mask),
then recomputing `scale·Q @ Kᵀ + causal_mask` and `softmax(...) @ V` by hand in numpy — necessary because
LFM2's HF path applies causal masking via SDPA's `is_causal=True` flag rather than an explicit additive
mask tensor, so a naive "capture the mask argument" approach silently sees `mask=None`. Key pitfall hit
along the way: `GraphBuilder` owns the `ggml_gallocr` backing the output tensor's data, so the builder
must stay alive (or the output must be copied out) before it goes out of scope — a locally-scoped
`GraphBuilder` returning `BuildResult` by value leaves `result.output` dangling.

**Still open / not addressed by this fix:**
- **Item 5's "broader ask"** (audit the remaining `primitives_basic.cpp` layout-healing heuristics in
  `MUL_MAT`/`MUL`/`REPEAT`) is now more clearly warranted — this investigation directly confirmed that at
  least one such heuristic (in `ADD`) was actively harmful once the exporter emitted correct layouts, not
  just unnecessary. The `MUL_MAT` primitive's own three heuristics (in `op_mul_mat` itself) were not
  re-examined here since the `matmul` fix routes correctly-shaped operands into it directly for the
  attention path; whether they're still load-bearing for other ops/models is unverified.
- Only LFM2-350M's specific SDPA decomposition shape was exercised. A model whose traced graph produces a
  `matmul` with `transpose_x=True` (either alone or combined with `transpose_y=True`) will hit the new
  `NotImplementedError` rather than silently miscompute — by design, but it means that combination still
  needs a real derivation + test case when it's first encountered.

---

Verified against the real HF PyTorch model (`/home/flavio/Dev/models/lfm2-350m`, prompt tokens `[1,2,3]`
padded to 128, comparing at sequence position 2) by bisecting layer-by-layer and then op-by-op inside a
layer:

- **Embedding**: matches exactly.
- **All 10 ShortConv layers** (layers 0,1,3,4,6,7,9,11,13,15): match to fp32 rounding precision, end to
  end, after the fixes below.
- **Attention layers' Q, K, V individually** (post RMSNorm, post RoPE, post GQA head-tiling): all three
  verified to match HF exactly, layer 2 checked in detail.
- **The baked causal mask constant** (`layer_2.mul_0_to_fp16`, ne=`[128,128,1,1]`): verified its values are
  a correct causal pattern (0 for `key<=query`, -30000 for `key>query`) under a `flat[query*128+key]`
  reading.
- **Attention layers' final output does NOT match** despite Q/K/V/mask all checking out individually. The
  bug is confined to the ~5 ops between them: `scale (MUL by 1/sqrt(64)) -> MUL_MAT(query,key) -> ADD
  (mask) -> SOFTMAX -> MUL_MAT(softmax,value)`.

**Leading hypothesis (unconfirmed):** `ggml_mul_mat(A, B)` produces `result->ne = [A->ne[1], B->ne[1],
B->ne[2], B->ne[3]]` (see `ggml.c:ggml_mul_mat`). The exporter's generic `MUL_MAT` handling
(`tools/loom_mil_compiler/exporter.py`, the `elif mapped_op == "MUL_MAT":` branch) just forwards MIL's own
`x`/`y` operand order into `ggml_mul_mat(x, y)` unchanged. For the attention-score matmul, MIL lists query
first (mirroring PyTorch's `Q @ K^T`), which makes the *query* axis land on `ne0` of the result. But
`ggml_soft_max` normalizes along `ne0`, and the mask's own `ne0` is `key` (per the reading above) — so
softmax ends up normalizing over the wrong axis, and/or the mask gets added with query/key swapped. This
is exactly the class of gotcha llama.cpp/whisper.cpp avoid by always calling `ggml_mul_mat(k, q)` (key
first) for attention scores specifically, while leaving `MUL_MAT(weight, x)` (weight first) for ordinary
linear layers, which is coincidentally already what this exporter does correctly today.

**Tried and did NOT immediately fix it:** manually swapping the two `MUL_MAT` operands
(`mul_1_cast_fp16`/`key_1_cast_fp16`) in a standalone copy of the topology JSON changed the output but
didn't make it match HF. Two live possibilities: (a) the mask's own `ne0` assumption above is backwards
and needs re-deriving directly (not inferred from "looks causal"), or (b) there's a second, compounding
issue in the same span (e.g. the scale step, or how `ggml_soft_max` actually treats non-`ne0` batch axes
when `ne2=16` heads are involved).

**How to pick this back up fast:** the bisection technique that got this far — registering a truncated
copy of a real topology's `nodes[:N]` list as a new Lua-bridge module with a chosen intermediate as its
`output`, then reading back via `run_subgraph`'s second return value (the `[ne0,ne1,ne2,ne3]` shape) to
avoid guessing tensor layout — is fast to redo (a few minutes per checkpoint) and should be applied to
`matmul_0_cast_fp16`/`add_0_cast_fp16`/`softmax_0_cast_fp16` specifically, comparing against HF's own
`torch.nn.functional.scaled_dot_product_attention` monkey-patched to capture its inputs (query/key/value
going in) so the *pre*-softmax scores can be diffed directly instead of only the final output.

Recommendation: once root-caused, the fix almost certainly belongs in `generate_graph_topology`'s `linear`/
generic-fallback-adjacent handling — specifically, detect the attention-score `MUL_MAT` (e.g. by shape:
both operands share `ne0` = head_dim and the op is immediately followed by a mask-`ADD` + `SOFTMAX`) and
force key-first ordering, rather than trusting MIL's operand order blindly. Should be paired with a
regression check that dumps intermediate attention scores/softmax output against HF for at least one
attention layer, since this class of bug is silent (same shape, plausible magnitudes, wrong values).

---

## 2. Generalized Lua driver codegen (a real IR, not string concatenation) — DONE

**Status:** implemented as planned. New module `tools/loom_mil_compiler/driver_ir.py`: `Expr` nodes
(`Var`/`Lit`/`RawExpr`/`Len`/`BinOp`/`UnaryOp`/`FieldAccess`/`Index`/`Call`/`TableLit`), `Stmt` nodes
(`Local`/`LocalDecl`/`Assign`/`SubgraphCall`/`Argmax`/`Return`/`If`/`While`/`Break`/`RawBlock`), a
`validate()` pass (linear "every read symbol must be defined by an earlier statement" check) and a
`check_subgraph_calls()` pass (every `loom.run_subgraph()` call's declared inputs must be a subset of the
target topology's own declared inputs), and a `LuaCodegen` class that's the only place that knows Lua
syntax. `apply_monolithic_export`/`apply_atomic_export`/`transpile_to_lua`/`transpile_block`/
`transpile_operation` all now build IR nodes instead of appending raw strings to `self.lua_lines` (removed
entirely). `export()` validates+codegens the IR after whichever path built it — for the `atomic` profile,
validation runs *inside* the same try/except that already falls back to `monolithic` on a partitioning
exception, since an IR that fails validation is the same class of "the heuristic didn't actually work" as
an exception during partitioning itself.

**Two real, previously-silent bugs the new validation pass caught immediately** (both would previously
have produced a runtime crash or silently wrong Lua, never something visible at export time):
- `apply_atomic_export`'s scope-based slice partitioning (item 1's own heuristic) mis-attributes an
  ungoverned op (here, one producing `position_ids`) to whichever slice happens to be "current" in
  iteration order, rather than the slice that actually needs it — surfaces as a `SubgraphCall` reading an
  input no earlier statement ever defined. Not fixed (it's a partitioning-heuristic bug, out of scope for
  this pass) — now caught by `validate()` and safely triggers the existing atomic→monolithic fallback
  instead of producing broken Lua.
- `transpile_operation`'s `cond` (MIL conditional) handling never bound the op's own output(s) to a Lua
  local at all — each branch only ever defined its own internal names, so any later use of the `cond`
  op's result read an undeclared Lua global (`nil`) at runtime. Fixed: the result name is now declared
  *before* the `if`/`else` via `LocalDecl` and plain-assigned (`Assign`, no `local`) from inside each arm,
  since Lua's block scoping means a `local` declared inside an `if`/`else` branch doesn't survive past it.

**Bonus (scoped down from "a Python interpreter"):** implemented as `check_subgraph_calls()` above — a
structural cross-check against the target topology's declared inputs, not a full IR-semantics interpreter
(not worth the cost for what's actually needed here).

**Verified:** all 4 `tools/loom_mil_compiler/test_compiler.py` unit tests pass unchanged (their substring
assertions on the generated Lua — `"function main(inputs)"`, `"if pred then"`,
`"loom.run_subgraph('dense_layer'"`, etc. — pin the exact surface syntax `LuaCodegen` must keep producing).
Full `ctest` suite: 104/105 (the one failure, `test_e2e_lfm2_lua_driver`, is pre-existing and unrelated —
confirmed identical with `git stash` against the unmodified code; see item 3 below for what it needs).

**Problem:** `apply_monolithic_export`/`apply_atomic_export`/`transpile_operation` all build the driver
script by appending raw Lua text to `self.lua_lines` while walking MIL ops in the same pass that decides
*what* the driver should do. Every bug found and fixed this round (wrong index into a flat output array,
a naming collision silently discarding one topology, spurious slice inputs) was a class of error a small
intermediate representation would catch mechanically, before ever emitting a single line of Lua text.

**Plan:**
- Introduce a small driver-IR module (e.g. `tools/loom_mil_compiler/driver_ir.py`) with a handful of node
  types: `SubgraphCall(name, n_tokens_expr, n_past_expr, inputs: dict[str, Expr], outputs: list[str])`,
  `Argmax(tensor, n_vocab_expr, row_expr)`, `Return(exprs)`, `If(cond, then, else)`/`While(cond, body)` for
  the bespoke transpile path, `Local(name, expr)`.
- Exporters (`apply_monolithic_export`, `apply_atomic_export`, `transpile_operation`) build a list of IR
  nodes instead of Lua strings directly.
- Add one validation pass over the IR before codegen: every symbol referenced by a later statement was
  defined by an earlier one (would have caught the `_atomic_final_shape` inputs-table bug class and the
  "referenced a var from a slice that isn't the last op" class immediately, with a clear error at export
  time instead of a runtime crash or silent wrong output).
- A separate `LuaCodegen` class walks the validated IR and emits the actual Lua text. This is also where
  Lua-syntax-specific concerns (variable name sanitization, multi-return handling) should live, cleanly
  separated from "what the driver does."
- Bonus this unlocks cheaply: a Python-side interpreter for the SAME IR, useful for a fast "does this
  driver even make sense" smoke check without spinning up the C++ engine.

---

## 3. Fix dynamic shapes in the Lua driver (both atomic and monolithic) — MECHANISM DONE, LFM2 NUMERICS STILL OPEN

**Status:** `export_lfm2_atomic.py`/`export_lfm2_monolithic.py` now trace with `ct.RangeDim(1, 4096)`
matching `make_lfm2_gguf.py`'s already-proven pattern; the static-padding branches (padded `local x`,
`original_len`, etc.) are deleted entirely from `apply_monolithic_export`/`apply_atomic_export` — both now
always emit the dynamic-`n_tokens` driver path (which also needed a real fix: the monolithic profile's
own dynamic path never actually argmaxed its output, just returned the raw logits array — dead code until
now, since this profile always hit the static/padded branch before).

**Constraint confirmed empirically (see the plan's own note): the engine is single-axis.**
`GraphBuilder`/`SymbolEnv` only ever resolve one dynamic quantity (`n_tokens`) per topology
(`graph_builder.cpp:113-116`). `get_var_info`'s permissive `"is" in dim_str -> "n_tokens"` substitution is
*intentionally* kept (not hardened into a stricter per-topology symbol check, which was tried and reverted
— see below) because CoreML's shape algebra routinely mints several distinct opaque symbol names (`is0`,
`is1`, `is28`, ...) for what is mathematically the *same* one dynamic quantity, whenever it can't simplify
a derivation (e.g. a causal pad+conv+slice that provably preserves length) back to the original input's
symbol — confirmed directly on an LFM2 ShortConv layer (4 distinct symbol names downstream of one `is0`
input) and again on the atomic path's own inter-slice inputs. A stricter "count distinct symbols, raise if
>1" guard was implemented and reverted after it produced a false positive on a real, correct atomic slice
(`layer_2`) for exactly this reason — there is no cheap, reliable way to distinguish "several names, one
true quantity" from "two genuinely independent axes" from the dim strings alone (CoreML doesn't expose
symbol-equality at this level). If a model genuinely needs a second independent dynamic axis, that will
surface as a numerical mismatch against the reference model, not a syntactic error at export time.

**Two real, deep tracing bugs found and fixed** (both `export_lfm2_atomic.py`/`export_lfm2_monolithic.py`,
confirmed by re-running the export and diffing the topology JSON before/after):
- LFM2's `cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])`
  (computed internally by `Lfm2Model.forward` when not passed explicitly) derives its length from a
  Python-level `.shape[1]` query, which `torch.jit.trace` bakes in as a **constant** equal to the tracing
  dummy's length (128) — independent of whatever `ct.RangeDim` is declared on the `tokens` input
  afterward. This was invisible before (the old fixed-128-and-pad driver always called with n_tokens=128
  too, so the baked constant coincidentally always matched); it surfaces as a `RESHAPE` hardcoded to
  `['1','128']` once a real, different length is requested. **Fix:** both wrappers now accept
  `cache_position` as an explicit tensor argument (declared with the *same* `ct.RangeDim` instance as
  `tokens`) and forward it straight through, so `Lfm2Model.forward` never runs its own internal
  (non-symbolic-under-tracing) computation at all.
- The same class of bug recurs one level down: `transformers.masking_utils.create_causal_mask`, when no
  explicit 4D `attention_mask` is given, derives `kv_length` via the same kind of Python-level shape
  query — baking in another hardcoded-128 constant even after `cache_position` itself became genuinely
  dynamic. **Fix:** both wrappers now also accept an explicit, already-built 4D additive causal mask as
  `attention_mask` (same shared `RangeDim`); `_preprocess_mask_arguments` explicitly short-circuits
  ("If the mask is already 4D, simply return as-is") once given one, bypassing the internal
  Python-shape-query path entirely. `apply_monolithic_export`/`apply_atomic_export` auto-generate both of
  these inputs on the driver side now (`cache_position`/`position_ids` via the existing `loom.range(...)`
  binding, `attention_mask` via the existing `loom.causal_mask(...)` binding) rather than expecting the
  caller to know these are implementation details of the traced graph — see `_POSITION_INPUT_NAMES`/
  `_CAUSAL_MASK_INPUT_NAMES` in `exporter.py`.

**One real, general (non-LFM2-specific) C++ engine bug found and fixed while chasing the above**
(`src/ops/primitives_basic.cpp`'s `op_shape`): `ggml_map_custom1`'s output tensor is always shaped
identically to its input (`ggml_dup_tensor` internally — there's no variant that lets a custom op request
a *different* output shape), but `shape_custom_op` only ever writes the first 4 int32 slots (the input's
`ne[0..3]`) regardless of the declared size. Any downstream consumer expecting a genuine small 4-element
shape-vector (e.g. `GET_ROWS` extracting one dim via a gather) instead saw a tensor shaped like the
*original* queried tensor — silently "worked" only when that tensor's own `ne[2]`/`ne[3]` happened to
already be 1, which every previously-exercised fixed-128 shape happened to satisfy. Confirmed as a real,
reproducible crash (`ggml_get_rows`'s own `a->ne[2] == b->ne[1]` assertion) once a genuinely dynamic length
made that coincidence stop holding. **Fix:** take a zero-copy `ggml_view_1d(..., 4, 0)` of the custom-op's
output before casting, giving downstream consumers the actual small shape-vector they expect.

**Update (follow-up session, "solve item 3"): 7 more real bugs found and fixed.** The `op_equal` crash
documented above was chased all the way through; fixing it (and everything it led to) surfaced a chain of
further real, previously-latent bugs. All of the following are now fixed and committed:

1. **`op_equal`'s algebra was simply wrong, independent of any broadcasting concern.** The old formula
   `step(a-b) * step(b-a)` requires `x>0 AND -x>0` simultaneously for the SAME `x`, which is never true —
   it evaluated to 0 (false) for every input, including `a==b`. Fixed to the correct complement formula:
   `d = a-b`; `equal = (1 - step(d)) * (1 - step(-d))`.
2. **The whole comparison-op family had a real broadcast-direction bug**, not just `EQUAL`. `ggml_sub(x, y)`
   requires `x`'s shape to be the *output* shape with `y` merely repeated into it (`ggml_can_repeat(b, a)`
   in `ggml.c`) — it can only broadcast a *smaller* `y` up into a *larger* `x`, never the reverse. `EQUAL`
   computed `b-a` in its second term (with `b` sometimes larger), and `LESS`/`GREATER` computed their one
   `ggml_sub` call with an arbitrary operand order MIL itself doesn't constrain. Confirmed as a real,
   reproducible crash via `ggml_can_repeat` once `cache_position` became a real dynamic input (see below —
   before that, this code path was never live). **Fix:** added a `sub_broadcast(x, y)` helper in
   `primitives_mil.cpp` that always orients the `ggml_sub` call correctly regardless of which operand is
   larger (negating the result if it had to swap), and every comparison op now goes through it.
3. **`RESHAPE`'s target-shape derivation needed a real algorithm, not string substitution** — the earlier
   fix in this file (collapse any symbolic dim to `n_tokens`) turned out to be unsound in a way deeper than
   "multi-symbol expressions are unreliable": a **lone** symbol isn't reliably `n_tokens` either.
   Coremltools reused a fresh, unrelated symbol for a reshape merging `(heads=16, head_dim=64)` into
   `hidden_size=1024` — a genuinely static value reported as one opaque symbol, not `n_tokens` at all — and
   it renumbers symbols even for provably-unchanged pass-through axes (confirmed: a reshape's own input and
   output seq axes were two different symbol objects for the identical, unchanged quantity). There is no
   symbol-identity check that can tell these apart from the output shape alone. **Fix:** two different
   rules depending on whether the reshape changes rank. Rank-reducing merges (fewer output axes than input)
   now derive their target *positionally* from the input's own (reliable) shape for every unchanged axis,
   with a literal `-1` (delegating to `op_reshape`'s already-existing numpy/PyTorch-style inferred-dimension
   support) for the one axis that's actually computed. Rank-preserving/-increasing reshapes keep the
   original "substitute lone symbols, `-1` for genuine multi-symbol products" rule, which is correct for
   that case (splits and pass-throughs don't fabricate new computed quantities the way a merge does).
4. **HF's `repeat_kv()` GQA-tiling idiom (`unsqueeze → tile → reshape-merge`) needed a dedicated fusion.**
   Its `tile` op's own `reps` value isn't a compile-time constant under genuinely dynamic tracing (`reps.val
   is None`, confirmed) — not because `n_rep` is actually dynamic (it's a fixed architectural
   hyperparameter), but because PyTorch's `.expand()` bundles it into one tensor alongside genuinely dynamic
   sizes, poisoning the whole thing. The existing "tile" composition's `if rep_factor is None: rep_factor =
   1` fallback then silently treated the real tiling axis as a no-op. **Fix:** `_try_fuse_gqa_repeat_kv` in
   `exporter.py` detects this exact pattern (via the tile's single consumer) and composes one native
   `REPEAT` directly from reliable information — the tile's own pre-expand input shape for every unchanged
   axis, the downstream reshape's own reliable output shape for the one axis that changes — bypassing the
   poisoned intermediate entirely.
5. **`GraphBuilder` unconditionally builds and *computes* every topology node, whether or not anything
   uses its output** — and that's not just wasted compute. Fusing tile+reshape (above) orphans the
   original tile's own `reps`-computation subgraph (`gather`/`concat`/`equal`/`select`/`div`), and that
   orphaned subgraph, despite having zero real consumers, **crashed on its own** during
   `ggml_backend_graph_compute`. **Fix:** `_prune_dead_nodes` in `exporter.py` does a backward-reachability
   pass from the topology's declared output after all nodes are generated, dropping anything unreachable —
   814 nodes survived out of 1040 for LFM2's monolithic export.
6. **The same problem, for weights.** `generate_graph_topology`'s `const`-handling unconditionally
   serializes every MIL `const` op as a GGUF weight tensor, including incidental attribute-only constants
   (a `matmul` op's own `transpose_x`/`transpose_y` booleans, a permutation-index array, etc.) that the
   exporter's own dedicated compositions already consume directly via `.val` in Python and never reference
   as a real node input. One such orphan (`_80_transpose_x_0`) had GGUF-declared `shape=[]` — a genuine
   zero-rank scalar — sitting immediately before a real, actually-used weight in tensor-declaration order.
   **Fix:** `_prune_dead_weights` in `exporter.py` drops any `self.weights` entry not referenced as an input
   by any surviving topology node (257 tensors survived out of 923). Confirmed correct and worth keeping
   regardless of the next item's outcome — a weight nothing reads should never be written to the GGUF.

**Still open after all of the above — bisected to an extremely small, precise repro.** Even after every
fix above, `test_e2e_lfm2_mil_export` still crashes, now with a different signature: a **null function
pointer dereference inside a ggml worker thread** (`AddressSanitizer: SEGV on unknown address 0x0...pc
0x0`, thread spawned via `GOMP_parallel`), not a clean assertion. Bisected using the same truncated-topology
technique as item 1's own resolution, refined to do real backward-reachability pruning at each truncation
point (not just a naive prefix cut) via a small standalone C++ harness that calls `GraphBuilder`/`GgufModel`
directly (no Lua) — down to a **minimal 5-node repro**:

```json
{"op": "RESHAPE", "inputs": ["cache_position"], "outputs": ["position_ids"], "attrs": {"shape": ["n_tokens", "1"]}}
{"op": "RESHAPE", "inputs": ["position_ids"], "outputs": ["_75"], "attrs": {"shape": ["n_tokens", "1", "1"]}}
{"op": "PERMUTE", "inputs": ["_75"], "outputs": ["_80_cast_fp16_mm_y_perm"], "attrs": {"axes": [1, 0, 2, 3]}}
{"op": "CONT", "inputs": ["_80_cast_fp16_mm_y_perm"], "outputs": ["_80_cast_fp16_mm_y_cont"]}
{"op": "MUL_MAT", "inputs": ["_80_cast_fp16_mm_y_cont", "const_2_to_fp16"], "outputs": ["_80_cast_fp16"]}
```

This is LFM2's RoPE inverse-frequency outer product (`inv_freq_expanded @ position_ids_expanded`, HF's own
rotary embedding code), composed via the exporter's `matmul` `transpose_x=False, transpose_y=False` branch
(item 1's own fix) — `const_2_to_fp16` is the real, GGUF-loaded `[1,32,1]` inverse-frequency table.

**What's been ruled out, each confirmed by an isolated standalone repro (a tiny hand-written `ggml`
program, no `GraphBuilder`/`GgufModel` involved):**
- Plain `ggml_mul_mat` on tensors of the exact same shapes (`[1,3,1,1]` × `[1,32,1,1]`) with fresh
  synthetic data: works fine.
- The full `PERMUTE`→`VIEW`×2→`MUL`(const)→`CONCAT`(self)→`MUL`(broadcast) chain used elsewhere in the same
  RoPE computation, with synthetic data: works fine.
- Self-`CONCAT` (`ggml_concat(x, x, dim)`, the standard `torch.cat([freqs, freqs])` RoPE trick) in
  isolation: works fine.
- Reducing thread count to 1 (`ggml_backend_cpu_set_n_threads(backend, 1)`): crash persists, so it is not
  purely a threading race, despite the worker-thread signature.
- Dead-node/dead-weight pruning (items 5-6 above): both real, valuable fixes, but neither changes this
  crash. Confirmed by testing before AND after landing each.
- This is **not the same bug** as `test_e2e_lfm2_lua_driver`'s own pre-existing failure (the bespoke-path
  test that was already failing before this session started) — confirmed by running that test under
  AddressSanitizer too: it hits a clean `GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0))`
  at a completely different call site, not a null-pointer/thread crash. Two separate, unrelated bugs.

**What this means:** something about the *real* `GraphBuilder`/`GgufModel` construction of this exact
5-node graph — not the raw shapes/values, not the op sequence in isolation, not threading, not the two dead
tensors already found — triggers this. Prime remaining suspects, in order of how cheap they are to check:
GGUF weight buffer alignment/adjacency for `const_2_to_fp16` specifically (it sits immediately after LFM2's
268MB embedding table in tensor-declaration order — worth trying a deliberately reordered/padded GGUF to see
if the crash moves or disappears); `ggml_gallocr`'s buffer-reuse decisions for this specific small-graph
shape (try building with a much larger `compute_meta_bytes`/graph-size margin to see if a reuse decision
changes); and the `ggml_map_custom1`-based custom ops used earlier in the same topology for RMSNorm's
`RSQRT` (`op_rsqrt`, `primitives_basic.cpp`) interacting with gallocr in a way this minimal 5-node repro
doesn't fully isolate (the crash was also reproducible with more RSQRT calls present upstream in a larger
truncation; worth re-confirming it reproduces with *this exact* minimal 5-node topology specifically, since
the bisection's last few steps focused on node count rather than re-verifying the very smallest cut still
crashes standalone). A real debugger (`gdb`/`lldb`, neither available in this environment) would very likely
resolve this quickly via a core dump backtrace — worth installing before spending more time on manual
bisection.

**Explicitly out of scope, still:** items 5's "broader ask" (auditing `primitives_basic.cpp`'s layout-healing
heuristics) and item 4 (tokenization) remain untouched. Item 6's LFM2-specific quantization verification and
a fresh HF numerical-correctness pass (item 1's own bisection technique, at a genuinely dynamic length)
both remain blocked on the crash above — a shape/build fix alone won't confirm values are right, only that
the graph builds and computes at all.

*(The original problem statement and plan for this item — trace with `ct.RangeDim` instead of a fixed
`(1,128)` shape, delete the static-padding stopgap, re-verify GQA tiling under dynamic lengths — is fully
superseded by the "Status" and "Update" sections above: all of it is done, and everything found once it was
done is catalogued there. Not repeated here to avoid the two going stale independently.)*

---

## 4. Tokenization

**Decision (recommended, from prior discussion):** native C++ tokenizers, selected at load time via a
GGUF KV (mirroring llama.cpp's own `tokenizer.ggml.model` convention, which `loom::BpeVocab`/`loom::Vocab`
already partially implements for GPT2-BPE and SentencePiece). **Not** a Lua tokenization script as the
default path — tokenization's core (ranked BPE merges, regex pre-tokenization) is a tight,
performance-sensitive loop with almost no per-architecture branching, unlike model orchestration where
Lua earns its keep. Doing it in Lua would mean inventing a whole new category of string/byte host bindings
(regex split, merge-rank tables, UTF-8 handling) that still has to be implemented in C++ underneath, with
no genericity payoff.

**Plan:**
- Extend the export tooling to detect the source HF tokenizer's class (`tokenizer_config.json`'s
  `tokenizer_class`, or `type(AutoTokenizer.from_pretrained(...))`) and serialize its vocab/merges/config
  into the existing GGUF KV convention.
- Extend `loom::Vocab`/add sibling classes to cover the remaining common tokenizer **families** (there are
  only a handful in practice): WordPiece, SentencePiece Unigram (currently only BPE-style SentencePiece is
  covered), tiktoken-style regex-BPE. This is a bounded, one-time cost per family, not per-model.
- Reserve a Lua-driven tokenization path only as an escape hatch for genuinely exotic/custom tokenizers
  that don't fit any known family — not the default, and not worth building out generically until a real
  model actually needs it.

---

## 5. MIL primitive review and dedup — CONCRETE BULLETS DONE, BROADER ASK STILL OPEN

**Status:** the three concrete, bounded bugs below are fixed. The "broader ask" (last bullet) is
deliberately **not** touched — see its own note.

- **`LESS`/`LESS_EQUAL` and `GREATER`/`GREATER_EQUAL` are implemented identically** — FIXED, but the
  actual diagnosis differs from this bullet's original text once ggml's real kernel was checked
  (`ggml-cpu/unary-ops.cpp`: `op_step(x) = (x > 0) ? 1 : 0`, a **strict** inequality, not `>=`). That makes
  `op_less`/`op_greater` (both already `step(...)`) correct as they stood; the actual boundary bug was in
  `op_less_equal`/`op_greater_equal`, which computed the same strict-inequality formula as their
  counterparts instead of the complement (`a<=b` must be true at `a==b`, but `step` gives 0 there). Fixed
  as `1 - step(...)` via `ggml_scale_bias(..., -1.0f, 1.0f)` (the same "complement" pattern `op_select`
  already used).
- **Dead lowercase `MilDialectRegistrar` registrations** — pruned (the whole `try_alias(...)` block and
  the lowercase half of every direct `reg.register_op("name", ...)` pair). Confirmed safe: every uppercase
  target these aliased stays registered directly elsewhere; two of the aliased names (`"square"`/
  `"relu6"`) turned out to already be dead in *both* directions (registered under no name anywhere in the
  codebase), so pruning them changed nothing observable either way.
- **Missing `OP_MAP` entries** — added (`"abs": "ABS"`, `"neg": "NEG"`, `"sign": "SIGN"`, `"minimum":
  "MINIMUM"`, `"maximum": "MAXIMUM"`, `"reduce_sum": "REDUCE_SUM"`, `"identity": "IDENTITY"`).
- **Broader ask (still needed, larger scope, deliberately out of scope this round):** audit
  `primitives_basic.cpp`'s ADD/MUL/MUL_MAT/REPEAT "dynamically heal transposed/permuted layouts"
  heuristics for correctness and necessity now that the IR/codegen rewrite (item 2) has landed. Not
  touched this round: these heuristics are shared by every model using these primitives (Whisper,
  Conformer-CTC, VITS, Matcha-TTS, SupertonicTTS, Kokoro — not just LFM2's MIL export path), so removing
  one needs per-model verification, not just LFM2's. Full `ctest` was run as a regression check (104/105,
  one pre-existing unrelated failure) to confirm nothing here was touched.

---

## 6. Export-time quantization — IMPLEMENTED, LFM2 NUMERICAL VERIFICATION BLOCKED ON ITEM 3

**Status:** implemented essentially as planned, porting the design already proven by
`tools/quantize/quantize_gguf_q8_0.py` + `tests/test_e2e_qwen3_q8_0.cpp` (see `BACKLOG.md`'s "quantized
weight support" milestone) directly into the exporter instead of as a separate post-conversion pass —
which that POC's own writeup already called out as the natural next step. `LoomGGUFExporter.__init__` now
reads `quantize=`/`LOOM_QUANTIZE`; `write_gguf` identifies quantizable tensors as every `MUL_MAT` node's
*first* input across `self.topologies`' own node lists (this exporter never emits `repeat_for`, unlike the
POC's GGUF-KV-driven version, so no expansion pass is needed) and, for each one that's 2D+, F32, and
block-size-aligned on its last dimension, calls `gguf.quants.quantize(...)` and writes it via
`add_tensor(name, quantized_bytes, raw_dtype=qtype)` (no `raw_shape` — confirmed by the POC that passing
the pre-quantization logical shape there is wrong). Non-aligned/non-MUL_MAT-weight tensors fall back to
the existing F32 path unchanged.

**Not yet independently numerically verified against LFM2** — needs a *working* F32 baseline forward pass
to quantize and compare against, which is exactly what item 3's still-open numerical issue blocks. The
code path itself is a direct port of a technique already proven end-to-end (Qwen3-0.6B-Base, 40/40 `ctest`
passing, real max-abs-logit-diff measured at 0.45–0.79, argmax tokens matching the F32 reference exactly)
using the identical underlying `gguf.quants.quantize`/`add_tensor(raw_dtype=...)` calls — so mechanically
sound — but "the code mirrors a proven pattern" is not the same claim as "verified against LFM2," and
should not be reported as the latter until item 3 unblocks a real quantized LFM2 export + comparison.

**Problem:** `write_gguf` currently always writes weights as f32 (with only dtype coercions for
bool/int64). The original spec (`LOOM_MIL_CONVERSION.md` §2) calls for block-quantized weights
(`Q8_0`/`Q4_K`) as a first-class output, matching llama.cpp's own conversion convention.

**Plan:**
- Add a `quantize=` kwarg to the backend (`LoomGGUFBackend.__call__`/`LoomGGUFExporter.__init__`), e.g.
  `quantize="Q8_0"`.
- In `write_gguf`, run 2D+ weight tensors through the `gguf` Python package's own block-quantization
  helpers before `add_tensor`, mirroring llama.cpp's `convert_hf_to_gguf.py`.
- Skip quantization for 1D tensors (biases, norm weights) — same convention llama.cpp uses, since
  quantizing small per-channel vectors buys negligible size savings at real accuracy cost.
- Verify the C++ side (`GgufModel`/`GraphBuilder`) actually dispatches primitives correctly against
  quantized tensor types end to end, not just that the file loads — this hasn't been exercised at all by
  the MIL exporter path yet.
