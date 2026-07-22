# Export/Compiler Backlog

Follow-up work identified while stabilizing the `add-conversion-via-mil` branch (LFM2-350M atomic +
monolithic export). Ordered by what's most load-bearing first, not necessarily execution order — see each
item's own recommendation.

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

## 2. Generalized Lua driver codegen (a real IR, not string concatenation)

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

## 3. Fix dynamic shapes in the Lua driver (both atomic and monolithic)

**Problem:** `export_lfm2_atomic.py`/`export_lfm2_monolithic.py` trace with a **fixed** `(1, 128)` input
shape rather than `ct.RangeDim`, unlike the bespoke `make_lfm2_gguf.py` (which already correctly uses
`ct.RangeDim(1, 4096)` per submodule). Because of this, every exported slice's declared shape has `128`
baked in as a literal, not the dynamic `n_tokens` symbol the rest of the pipeline (`get_var_info`'s
`"is" in dim_str` check, `GraphBuilder::build(n_tokens, n_past)`) is designed to support. The current
stopgap (added this round, in both `apply_monolithic_export` and `apply_atomic_export`) pads every prompt
to the fixed length and slices out the right row — it works, but wastes compute (full 128-token forward
pass for a 3-token prompt) and is exactly the kind of driver complexity Lua-as-orchestrator was supposed
to make unnecessary in the first place (see LOOM_PROCEDURAL_GENERALIZATION.md's own framing: layers as
subgraphs should let the driver instantiate each call at the *correct* dimension).

**Plan:**
- Switch `export_lfm2_atomic.py`/`export_lfm2_monolithic.py` (and the generic automatic-profile path in
  `exporter.py` generally) to trace with `ct.RangeDim` on the sequence dimension, matching the bespoke
  script's already-working approach.
- Once shapes are genuinely symbolic, delete the static-padding branches in `apply_monolithic_export`/
  `apply_atomic_export` entirely — the existing dynamic-`n_tokens` code path (the `else` branch that's
  already there for when no static length is detected) should just work, calling every subgraph with the
  real prompt length via `#first_input`.
- Do this as part of (or right after) the IR/codegen rewrite in item 2 above — the padding branch is
  exactly the kind of special-casing that gets simpler to delete once driver synthesis goes through a
  validated IR instead of hand-woven strings.
- Re-verify GQA tiling monkeypatch (`_robust_decompose_sdpa` in the export scripts) still works correctly
  under genuinely dynamic/symbolic sequence lengths, not just the fixed-128 trace it's been exercised
  against so far.

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

## 5. MIL primitive review and dedup

Found during this round's review of `src/ops/primitives_mil.cpp` (real bugs, not yet fixed):

- **`LESS`/`LESS_EQUAL` and `GREATER`/`GREATER_EQUAL` are implemented identically** — `op_less` and
  `op_less_equal` both compute `step(b-a)`; same for `op_greater`/`op_greater_equal` computing `step(a-b)`.
  Wrong at the boundary (`a==b`): `less_equal` should be true there, `less` should be false. Currently
  doesn't bite LFM2 because its causal mask is constant-folded at trace time rather than computed via a
  live `band_part`/comparison op at runtime, but will bite any model that does use these dynamically (e.g.
  a non-constant-shape mask, or genuine data-dependent comparisons).
- **The entire block of lowercase `try_alias(...)` registrations (and direct lowercase registrations like
  `"less"`/`"greater"`) in `MilDialectRegistrar` is dead code** given how this exporter actually works:
  `exporter.py`'s own `OP_MAP` always normalizes every MIL op to its uppercase Loom name before ever
  writing topology JSON, so the C++ primitive registry never sees a lowercase op string from this
  exporter's output. Worth pruning for clarity, or repurposing if a future exporter path is added that
  *doesn't* go through `OP_MAP` (unlikely given the current design).
- **`abs`/`neg`/`sign`/`minimum`/`maximum`/`reduce_sum`/`identity` are correctly implemented in C++ but
  currently unreachable** — `exporter.py`'s `OP_MAP` has no entries mapping these MIL op types to their
  (already-correct) uppercase C++ names. A real trace containing e.g. a MIL `neg` (common in RoPE's
  rotate-half, or in any model that doesn't go through this exporter's dedicated inline RoPE composition)
  would currently hit `NotImplementedError` despite the primitive already existing. Cheap, high-value fix:
  add the missing `OP_MAP` entries (`"abs": "ABS"`, `"neg": "NEG"`, `"sign": "SIGN"`, `"minimum":
  "MINIMUM"`, `"maximum": "MAXIMUM"`, `"reduce_sum": "REDUCE_SUM"`, `"identity": "IDENTITY"`).
- **Broader ask (still needed, larger scope):** audit `primitives_basic.cpp`'s ADD/MUL/MUL_MAT/REPEAT
  "dynamically heal transposed/permuted layouts" heuristics (added this round's predecessor commits) for
  correctness and necessity. These guess at a permute based on `ne[]` equality patterns rather than
  deriving the correct layout from the exporter side — risky by construction (silently "fixes" shape
  mismatches that may or may not be the mismatch the heuristic assumes), and already directly implicated
  in one confirmed regression this round (`VIEW`'s wrong stride defaults, see the numerical-correctness
  section above and the fix already applied in `primitives_basic.cpp`). Once the transpose/perm fix and
  IR/codegen rewrite land, re-check whether any of these heuristics are still load-bearing, or whether
  they're now masking bugs rather than fixing them; replace with correct exporter-side layout math where
  feasible instead of runtime guessing.

---

## 6. Export-time quantization

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
