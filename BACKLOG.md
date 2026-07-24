# Backlog

Consolidated from `BACKLOG.md` + `EXPORT-BACKLOG.md` + `EXPORT-IMPROVEMENT-BACKLOG.md` (all three tracked
overlapping/superseding history from different phases of the project — model milestones, the small-TTS-model
roadmap, and the MIL-compiler pivot — and are now merged into this single file). Everything previously
tracked as resolved anywhere in that history has been removed; it lives in git log/commit messages, not
here. Only genuinely open work remains below, grouped by area.

---

## Models

### Roadmap: Qwen3-ASR-0.6B / Qwen3-TTS-0.6B

Not started. Qwen3-0.6B-Base (the base LLM) is done. The ASR and TTS (12Hz-Base) variants of the Qwen3
family remain fully unstarted — no conversion script, no source-level architecture read yet. Qwen3-TTS is
expected to be the most architecturally novel item in this family (needs its own source-level
investigation before scoping).

### F5-TTS

Deferred by explicit user direction (flow-matching TTS, `OdeStepper`-adjacent — likely shares primitives
with Matcha-TTS, which is done). Last of the originally-considered 7-model TTS list still untouched.

### Task #79: permissively-licensed phonemizer

VITS, Kokoro, StyleTTS2, and Matcha-TTS drivers all still take raw token-id/demo text input — none of them
do real text→phoneme conversion. Real `espeak-ng`-based phonemization was confirmed to work numerically
(via the external `piper_phonemize` Python package) but vendoring it was rejected: both stock espeak-ng and
the piper-phonemize fork are GPL-3, incompatible with this repo's permissive licensing (see
`[[loom_engine_licensing_phonemizer]]` memory). Current plan: integrate **phoonnx** (a friend-of-the-user's
project) instead, once its license/API are confirmed — not yet investigated. A `src/text/phonemize.cpp` +
`include/loom/text/phonemize.h` split matching this project's existing driver-code conventions is the
intended shape. SupertonicTTS is the one model in this family that's already fully closed — its
`TextVectorizer` is a license-free unicode codepoint lookup table, no phonemizer needed at all.

---

## Exporter / MIL compiler

### MIL primitive review — broader ask still open

The concrete, bounded bugs originally tracked under this item (`LESS_EQUAL`/`GREATER_EQUAL` boundary bug,
dead lowercase `MilDialectRegistrar` aliases, missing `OP_MAP` entries) are fixed. Not done, deliberately
deferred:

**Audit `primitives_basic.cpp`'s ADD/MUL/MUL_MAT/REPEAT "dynamically heal transposed/permuted layouts"
heuristics for continued necessity**, now that the exporter emits correct layouts directly for more cases.
One such heuristic (in `op_add`) was already found to be actively harmful once the exporter started
emitting correct `MUL_MAT` layouts itself, and was removed — the other two (in `op_mul` and `op_repeat`)
are untouched and unverified. These heuristics are shared by every model using these primitives (Whisper,
Conformer-CTC, VITS, Matcha-TTS, SupertonicTTS, Kokoro), not just LFM2's MIL export path, so removing one
needs per-model verification, not just LFM2's.

### Known gap: `matmul` composition only handles `transpose_x=False`

`tools/loom_mil_compiler/exporter.py`'s dedicated `op_type == "matmul"` composition only derives correct
`ggml_mul_mat` semantics for `transpose_x=False` (either `transpose_y` value — both occur in LFM2's SDPA
decomposition). Any other combination (`transpose_x=True`, alone or with `transpose_y=True`) raises
`NotImplementedError` by design rather than silently miscomputing. Not yet hit by any converted model; a
real derivation + test case is needed the first time one does.

### Retrofit the bespoke `tools/convert_*` scripts onto the MIL exporter

Order being worked (per explicit user direction): Qwen3, Conformer-CTC, Parakeet, VITS, Kokoro, Matcha-TTS,
SupertonicTTS, StyleTTS2.

- **Qwen3-0.6B-Base — DONE.** `export_qwen3_mil.py`, both monolithic and atomic profiles. Needed zero
  bespoke wrapper code (the existing `export_hf_causal_lm.py` driver handled it as-is) plus one real,
  general exporter bug fix: the `concat` translation branch silently dropped an op (no node, no alias)
  when MIL's default pipeline had already folded a concat down to exactly 1 real operand — the shape HF's
  KV-cache update (`torch.cat([past_key_states, key_states], ...)`) takes when traced with an empty/unused
  cache, which every plain-forward-pass export hits. Fixed by aliasing the single operand through (same
  pattern as the `cast` branch). Verified: 8/8 greedy-token match against real HF `generate()` for both
  profiles ("The capital of France is" → " Paris. The capital of Germany is Berlin"). Full `ctest`: zero
  regressions.
- **NeMo Conformer-CTC-small — exports and runs deep into the real model, one open data-flow bug left.**
  `export_conformer_ctc_mil.py` traces the REAL `nemo.collections.asr.models.EncDecCTCModelBPE`
  (preprocessor + `ConformerEncoder` + `ConvASRDecoder`) directly — no hand-reimplemented plain-PyTorch
  module needed (unlike `tools/convert_generic/conformer_ctc_module.py`'s older `aten_to_loom`-oriented
  POC, which needed a custom `loom::rel_pos_attention` torch op; the MIL path has no such requirement, it
  walks whatever real ops the relative-position attention decomposes into under tracing — turned out to be
  ordinary matmul/pad/reshape/gather, no new attention primitive needed at all).

  Getting this far found and fixed a long run of real, general exporter/engine gaps (all committed except
  where noted, full `ctest` clean after every one — zero regressions, `Qwen3` re-verified byte-for-byte
  identical throughout):
  - Missing `logical_not` primitive (new `NOT` op, `1 - x`, mirrors the existing comparison-op complement
    pattern).
  - Missing `stack` translation (MIL's stack-along-a-new-axis, hit by the mel frontend's hand-rolled
    conv-based STFT stacking real/imag parts — composed from existing RESHAPE+CONCAT, no new primitive).
  - Two OP_MAP gaps (`sqrt`/`log` — the ggml primitives already existed, just weren't wired up).
  - A real memory-safety bug in dynamic `fill` handling (`torch.full` sized off a non-constant shape
    pre-allocated `[4096]*rank` — 256 GiB at rank 3 — instead of composing a scalar-constant +
    REPEAT-broadcast, which works at any rank).
  - `slice_by_index` never honored `begin_mask`/`end_mask` (MIL's "ignore this bound, use the full extent"
    convention for e.g. `x[1:]`/`x[:-1]`) and never normalized a negative begin/end against a *symbolic*
    dim size (only the concrete-int case was handled) — silently produced literal negative shapes like
    `-1` instead of `n_tokens - 1` for the mel frontend's pre-emphasis filter.
  - `reduce_sum` always fully-reduced to one scalar (`ggml_sum`), when MIL's own `reduce_sum` here always
    reduces over exactly one real axis (STFT magnitude, CMVN mean/variance) — replaced with a real
    per-axis reduction (permute target axis to `ne[0]`, `ggml_sum_rows`, permute back, reshape away if
    `keep_dims=False`), driven by a new dedicated exporter translation that reads MIL's `axes`/`keep_dims`
    inputs directly instead of dropping them.
  - `RESHAPE`/`VIEW`/`MUL_MAT` all gained real, informative error messages (shape/element-count mismatch
    details) in place of raw uncatchable `GGML_ASSERT` aborts — made every one of the above findable at
    all instead of a bare crash with no context.
  - **The deepest one**: `get_var_info`'s long-standing "collapse every symbolic MIL shape dim to the bare
    string `n_tokens`" heuristic (documented, deliberate, and correct for every model up to LFM2/Qwen3 —
    see EXPORT-BACKLOG's own history) is genuinely wrong here, because Conformer-CTC's mel-frontend
    introduces a **second, derived** dynamic quantity (STFT frame count, `floor(n_tokens/160)+1 ≈ 101` for
    a 1s clip, vs. `n_tokens = 16000` raw samples) that also happens to present as a bare opaque symbol.
    Fixed with a new `_infer_dynamic_dim_expr` — a bounded backward walk from a symbolic dim through its
    producer chain, deriving the real expression instead of guessing, with safe bare-substitution fallback
    for anything it doesn't specifically understand (never regresses a case it used to get right). Handles,
    in order added as each was needed: `cast`/unary shape-preserving ops (`log`/`exp`/`sqrt`/etc.) as pure
    passthrough; `conv` (real stride/pad/kernel formula); `matmul` (per-operand axis correspondence,
    honoring `transpose_x`/`transpose_y`); `tile` (no-op passthrough when `reps==1`, or "resolves to a
    literal 1" when `reps` itself is unreadable — poisoned the same way GQA `repeat_kv`'s `reps` is — but
    the input axis is already a static 1, matching this whole exporter's "batch is always 1" design);
    elementwise binary broadcast ops (comparisons/add/sub/mul/etc. — the real axis comes from whichever
    operand isn't just a size-1 broadcast target); `expand_dims`/`squeeze` (axis-shifted passthrough).

  **The `gather_7`/length-tracking bug above is root-caused and fixed.** The true cause was architectural,
  not a shape-string bug: `op_range_1d` (src/ops/primitives_mil.cpp) can only read a dynamic "end"/"start"
  bound from a Var's own already-computed `.data`, but a `gather(shape(x), axis)` chain's value only
  exists *after* `ggml_backend_graph_compute()` runs — strictly after `GraphBuilder::build()` (which is
  when `op_range_1d` itself executes) finishes. So `in[1]->data` was architecturally always null for this
  exact pattern, silently tripping `op_range_1d`'s own "dynamic sequence length" fallback (a hardcoded
  `n_tokens`) no matter what any shape-string fix did upstream. Fixed with a new
  `_try_derive_gather_shape_value` (detects the `gather(shape(x), axis)` pattern specifically and derives
  its real value via `_infer_dynamic_dim_expr`, correctly flipping a torch-order axis index into the
  shape-vector's own ne-order storage) plus a dedicated `range_1d` translation that emits `start`/`end`/
  `step` as real symbolic **attrs** (which `op_range_1d` already natively supported, evaluated via
  SymbolEnv) instead of data-dependent graph inputs, whenever all three can be resolved this way.

  Chasing this all the way through also found and fixed a real, previously-silent negative-index bug: a
  constant used as `gather`'s own "indices" input can hold genuine Python-style negative indices (MIL's
  own convention, e.g. `x.shape[-1]`), but `GET_ROWS`/`ggml_get_rows` has no such convention — it read
  garbage/wrapped memory instead of raising. Fixed by normalizing at const-bake time, with the ne-order
  reversal a `shape()`-vector gather specifically needs (torch axis `-1` is ne-order index `0`, not `3`).

  `_infer_dynamic_dim_expr` also gained several more cases needed to reach `gather_7`'s real producer
  chain at all: `pow` (added to the elementwise-broadcast set — squaring in the STFT magnitude
  computation), `reduce_sum` (output-axis-to-input-axis remapping when `keep_dims=False` drops an axis,
  mirroring the existing `squeeze` case), `stack` (mirrors `expand_dims` — one new axis from N identically-
  shaped operands), and a `range_1d` pass-through (a range's own output length, reusing the same
  attrs-resolution logic as its node-emission branch) — plus a broadened `tile` heuristic: when `reps` is
  unreadable (poisoned the same way GQA `repeat_kv`'s is) AND the input axis is *already* dynamic (not a
  static 1), treat `reps` as 1 and pass through, reasoning that a genuine multiplicative tile of an
  already-dynamic axis would have been intercepted by `passes.py`'s dedicated `fuse_gqa_repeat_kv` pass
  already, so a bare `tile` op surviving to this point is overwhelmingly unlikely to be that case.

  Also fixed along the way: `op_select` (src/ops/primitives_mil.cpp) called `ggml_mul` directly on
  operands in MIL's own order, but `ggml_mul(a, b)` requires `a` to be the larger/target shape — added a
  `mul_broadcast` helper (mirrors the existing `sub_broadcast`) plus a real error message in place of a
  raw `GGML_ASSERT` abort.

  **The model now traces, exports, and runs the complete forward pass with zero crashes** — the
  `x_std` CMVN broadcast bug mentioned above, and every other shape/data-flow crash found chasing it, are
  fixed. What follows is everything found and fixed getting from "still open, one more data-flow bug" to
  "runs end-to-end" (all committed, full `ctest` clean and Qwen3 re-verified byte-exact after every one):

  - **The `x_std` bug itself**: `_infer_dynamic_dim_expr`'s `expand_dims`/`squeeze` case correctly derived
    a per-axis formula from the input, but the bottom-of-function fallback (for any axis no specific case
    understood) always substituted a bare `"n_tokens"` regardless of which axis was being asked about —
    wrong whenever that axis is genuinely the always-1 batch axis (axis 0), which `x_std`'s own tangled
    select/sub/pow/tile chain (CMVN's masked-mean/variance computation) bottoms out on. Fixed with a new
    final fallback: `torch_axis == 0` on a rank≥2 var resolves to literal `"1"`, matching this exporter's
    standing "batch is always 1" assumption (already used by several other cases) rather than a fresh
    "n_tokens" guess. Also had to fix the SAME assumption inside the existing `conv` case, which had a
    bare `if torch_axis < 2: return None` that returned from the *whole function* (a `return` inside an
    `if` block returns from the enclosing method, not just that branch) — bypassing the new fallback
    entirely for any conv-produced tensor's batch axis.
  - **`RESHAPE`'s own `"shape"` INPUT is now resolved directly** (new `_try_resolve_reshape_shape_input`),
    not just its OUTPUT var's declared shape — needed because a `reshape` (unlike `expand_dims`/`squeeze`)
    has no per-axis correspondence formula to its input at all (elements get freely redistributed), so the
    output var's own symbolic dims are the only place `get_var_info` had to look, and MIL mints a
    *fresh, unrelated* opaque symbol per axis there — collapsing two genuinely different axes (e.g. batch
    and time in a Q/K/V head-split) to the same blind `"n_tokens"`. Resolves via the same `concat`-of-
    gathers pattern `RANGE_1D`'s own start/end resolution already uses, handles a literal constant `.val`
    array directly, and a literal `-1` (PyTorch's own "infer this axis" marker) via the general
    `total_elements(input) / product(other resolved axes)` formula (NOT a same-position guess — confirmed
    wrong on `rel_shift`'s `x.view(b,h,-1,qlen)`, which swaps which physical input axis the `-1` position
    ends up representing).
  - **`slice_by_index`'s own VIEW-composition** (the "real" translation, not just shape-inference) had the
    identical `"n_tokens"`-substitution-only view of the world: it only ever read a literal `.val`
    begin/end array, discarding the WHOLE array (not just the missing axis) whenever begin/end was instead
    a dynamic `concat`, silently turning a real crop into a no-op on every axis. Fixed with a shared
    `_resolve_slice_axis_value` used by both the shape-inference case and the real VIEW-shape/offset
    computation. Also found: the VIEW-shape composition only ever treated ne-axis 0 as "the axis being
    sliced" (copying every other axis straight from the parent) — wrong whenever the real slice lands on a
    non-fastest axis (confirmed on `rel_shift`'s `x[:, :, 1:]`), silently keeping the parent's FULL
    (unsliced) size there; generalized to compute `end - begin` on every axis uniformly.
  - **New op-type cases added to `_infer_dynamic_dim_expr`** while chasing the above through more of the
    encoder than any earlier fix reached: `layer_norm`/`linear` (shape-preserving passthrough on every
    axis but the last), `transpose` (real per-axis correspondence via `perm`, not blind substitution),
    `pad` (passthrough plus the padded axis's real `+lp+rp` formula), `split` (passthrough on every axis
    but the split one), a broadened `matmul` batch-axis case (leading axes beyond the trailing 2 "real"
    matmul axes), and `select`/`softmax`/`logical_not`/`silu` added to the existing unary/elementwise
    passthrough sets.
  - **A genuine off-by-something bug in `_ELEMENTWISE_BROADCAST_OPS`'s operand-selection**: when BOTH
    operands of e.g. a `mul` report a dynamic symbol at the same axis (MIL can't always prove one side is
    a literal 1, even when it genuinely broadcasts from one), the walk picked whichever operand it checked
    *first*, not whichever was actually informative — confirmed wrong on `att_mask = pad_mask_for_att_mask
    * att_mask_3`, where the first operand's own axis resolved to `"1"` (a real broadcast-from-1, MIL just
    couldn't prove it statically) while the second operand held the real formula. Fixed to prefer any
    operand that resolves to something other than literal `"1"`, falling back to `"1"` only if every
    operand bottoms out there.
  - **`op_mul`/`op_add` (`src/ops/primitives_basic.cpp`) gained the same informative-`SchemaError`
    treatment `op_reshape`/`op_view`/`op_mul_mat` already had** in place of raw uncatchable
    `GGML_ASSERT(ggml_can_repeat(...))` aborts.
  - **`op_softmax` and every conv primitive (`CONV_1D`/`CONV_1D_DW`/`CONV_2D`/`CONV_2D_DW`) now `ggml_cont`
    their input if non-contiguous** — needed once a real strided VIEW (a GLU channel-split) started
    feeding straight into `ggml_im2col`/`ggml_soft_max`, both of which assert dense strides internally
    with no context on failure.
  - **Real int32↔float type-mismatch bug across every elementwise arithmetic primitive**: ggml's own
    ADD/SUB/MUL/DIV kernels only support same-family FLOAT type combos (F32/F16/BF16) — there is no
    integer-arithmetic path at all, not even I32-with-I32 (confirmed by reading `ggml_compute_forward_add`
    itself: I32 isn't one of its listed cases). MIL, in contrast, does real int32 arithmetic wherever a
    model computes on the "length" input directly. New shared `promote_i32_to_f32` helper (duplicated
    per-TU, matching this file's own convention) casts any I32 operand up before it reaches
    `op_add`/`op_sub`/`op_mul`/`op_div`/`sub_broadcast`/`mul_broadcast`/`add_broadcast` — this project's
    target models only ever do small, exact-integer arithmetic here, well within F32's exact range.
  - **`floor_div` (PyTorch `//`) was silently mapped to the same plain `"DIV"` primitive as `real_div`**,
    dropping the floor entirely. New dedicated `FLOOR_DIV` primitive (`op_floor_div`, composed as
    `ggml_floor(ggml_div(...))`) plus an OP_MAP fix. This one is load-bearing, not cosmetic: it's how the
    exporter can still recover NeMo's own `calc_length()` formula's real floor semantics for a length
    computed from the *user-supplied* "length" input, given coremltools' own tracing had ALSO already
    eliminated the standalone `torch.floor()` MIL op as a no-op for the specific dummy trace length used —
    confirmed via `grep`, there are zero `FLOOR` ops anywhere in the exported topology.
  - **`GraphBuilder::build_node` now wraps a primitive's own exception in a `SchemaError` naming the
    failing node** (`src/core/graph_builder.cpp`) — every one of the shape-mismatch bugs above was found
    by reading THIS wrapped message rather than a bare, nodeless `GGML_ASSERT` abort.

  **The `calc_length`/`all_paddings` bug above is fixed** (Root cause recap: NeMo's own traced
  `calc_length()` computation, used to build the CNN-subsampling padding/validity mask, has the wrong
  constant baked in for `all_paddings` — the traced graph computed `all_paddings=1` instead of the real
  `2` (2×the conv's own `padding=1`), and coremltools' own optimizer had ALSO already eliminated every
  standalone `torch.floor()` call in this chain as a provable no-op for the specific dummy trace length
  used (confirmed via `grep`, zero raw `FLOOR` ops anywhere in the exported topology) — so this isn't a
  dropped-floor bug fixable by recovering a MIL node, the arithmetic MIL actually traced is wrong).

  Rather than reverse-engineer coremltools' own pass pipeline to find and patch whichever pass causes
  this (considered and rejected — `ct.PassPipeline.EMPTY` confirms passes ARE responsible, since it
  changes the result, but a fully empty pipeline breaks essential decompositions like
  `lower_complex_dialect_ops`/STFT, and surgically removing just the offending pass isn't tractable
  without a much deeper coremltools-internals investigation than this narrow bug warrants — `const_
  elimination` alone runs ten separate times in the default pipeline and is load-bearing for legitimate
  simplifications everywhere else, so a mistargeted removal risks silently breaking Qwen3 or other
  Conformer-CTC paths that currently work), the fix exploits an invariant this WHOLE exporter already
  assumes everywhere else: every model it targets runs with `length` set to the exact real length of
  `waveform` (no actual padding, ever — the same "batch is always 1" guarantee used throughout). Under
  that guarantee, `torch.arange(T) < length`-style length-validity masks (the ONLY thing `calc_length`'s
  buggy value ever feeds) are true for every position by construction, regardless of what NeMo's own
  traced arithmetic computes for the bound.

  New `_traces_to_range_1d`/`_traces_to_length_input` (backward producer-chain walks, mirroring this
  file's other `_traces_*`/`_try_resolve_*` helpers) recognize this exact `arange(T) < length`-derived
  pattern on the MIL `less` op specifically (the only comparison op actually seen doing this — narrow by
  design, matching this file's "not implemented since nothing here has needed it yet" convention) and
  replace its result with a constant all-1.0 tensor of the correct (already-correctly-derived) output
  shape, bypassing the buggy traced arithmetic entirely rather than trying to repair it. Confirmed: all 6
  `less` ops in Conformer-CTC's topology matched and were replaced (`grep`: zero raw `LESS` ops remain in
  the exported topology). Full `ctest` clean, Qwen3 re-verified byte-exact.

  **A separate, previously-masked numerical bug is now visible.** With the masking fix in place, the
  model's logits move meaningfully closer to `reference_forward_conformer.py`'s output but still don't
  match (bisected past the masking layer down to the raw log-mel CMVN-normalized features themselves --
  `x_17_cast_fp16` -- which already differ from an independently-computed `compute_mel_features()`
  reference, e.g. `-0.27` vs. `-1.52` at index 1). This is NOT the same bug: masking is now provably a
  no-op (matches the single-utterance/no-padding invariant exactly), so this is a genuine, distinct
  mismatch somewhere in the STFT-via-CONV_1D / mel-filterbank / CMVN-normalize chain itself, not yet
  investigated.
- **Parakeet, VITS, Kokoro, Matcha-TTS, SupertonicTTS, StyleTTS2 — not started.** VITS/Kokoro/Matcha-TTS/
  SupertonicTTS are plausible but unproven — their two historical showstoppers (STFT/ISTFT, and LSTM) were
  exactly what got generalized into the MIL exporter as follow-up work, and their iterative bits (flow
  reverse-steps, Euler CFM, RNG-fed sampling) are fixed-depth, so should trace as one static graph with
  noise supplied as an input tensor. StyleTTS2 is the one likely to stay bespoke — its diffusion sampler's
  ~3e-3 residual mismatch persisted even with hand-matched float32 host math, and an auto-traced version
  gives less control to chase that kind of thing down.

The real tradeoff: doing this would replace ~10 hand-verified conversion scripts with one generic path, but
trades "verified against hand-derived reference, primitive by primitive" for "trust the trace" — worth it
for showcasing generality, riskier for correctness confidence on the trickiest models.

### Submodule-export blueprint: promote to default, prove generality, dedup weights

The submodule-export blueprint (`tools/loom_mil_compiler/submodule_discovery.py`/`submodule_export.py`,
`apply_submodule_export` in `exporter.py`, `export_lfm2_submodule.py`) landed as a working first iteration,
proven on LFM2 (`test_e2e_lfm2_mil_export` passes against `lfm2_350m_submodule.gguf`, same top-1 tokens as
atomic/monolithic). Four things from that iteration's own plan are still open:

- **Not yet promoted to the default atomic path.** `apply_atomic_export`/`export_lfm2_atomic.py` (the
  older scope-based partitioner) are untouched and still what the "atomic" profile actually uses. Whether
  to make the submodule blueprint the default (and delete the scope-partitioning code path) is a follow-up
  decision, not yet made.
- **Unproven generality — only ever validated on LFM2.** The whole point of this thread was generality
  (no `ModuleList`-naming-convention assumption, structural rather than by-name discovery), but that claim
  is still resting on a single model. Needs a second, structurally different HF model (different attribute
  names, ideally non-hybrid/homogeneous-layer to start) added to the regression suite.
- **Cross-submodule weight duplication is unfixed.** Each submodule is traced independently
  (`ct.convert()` per submodule), so any tensor referenced from more than one submodule — the most likely
  case being HF's tied embedding/`lm_head` weight — gets serialized twice under two different namespaced
  names (confirmed: submodule GGUF is 1.69GB vs. monolithic's 1.42GB on LFM2, consistent with one full
  extra copy of the ~268MB vocab embedding). Planned fix: hash each candidate weight's bytes+shape+dtype at
  `write_gguf` time and alias a repeat hash to the first-written name instead of writing it again — not yet
  implemented. A narrower, cheaper alternative worth considering first: when `tie_word_embeddings` is set,
  just skip re-exporting `lm_head`'s weight and alias it to the embedding's own name directly.
- **Phase 2 (fully automatic prefix/suffix boundary discovery) not attempted.** Today's `SubmoduleExportSpec`
  needs a ~3-line declarative boundary per model (`prefix_attr`/`repeated_attr`/`suffix_attrs`/`aux_attr`).
  The stretch-goal alternative — an early-exit-hook technique mirroring HF `accelerate`'s device-map
  splitting, deriving prefix/suffix without any per-model spec at all — was deliberately not attempted since
  Phase 1's spec was sufficient for LFM2. Worth doing only if Phase 1 starts feeling like real friction
  across 2-3 more models, not speculatively.

---

## Engine

### Performance optimizations designed but not implemented

- **Bucketed KV-cache graph-reuse.** `GraphBuilder::build()` always does a full rebuild + no_alloc pass
  per call. Plan: round `n_kv` up to a bucket boundary (e.g. 32) and skip the rebuild when the bucket
  hasn't changed, reusing the previous `ggml_cgraph*` and just overwriting input tensor data. Safe to
  attempt now that the graph-reuse aliasing bug (`ggml_gallocr` aliasing an input tensor's buffer) is
  root-caused, as long as every declared input is rewritten every decode step — still needs its own
  bit-identical-to-rebuild regression test (same pattern as `test_graph_reuse_safety.cpp`). See
  `GraphBuilder::reserve()`/`GraphBuilder::build()` in `src/core/graph_builder.cpp`.
- **`ggml_backend_sched` / multi-backend.** Not used anywhere — engine talks to a single `ggml_backend_t`
  directly via a plain `ggml_gallocr`. Fine for CPU-only; needed once a second backend (CUDA/Metal) is
  added and graphs need splitting across devices.
- **Flash attention.** `ATTENTION` (`src/ops/primitives_attention.cpp`) always uses the composite
  (`MUL_MAT`→`soft_max_ext`→`MUL_MAT`) path — chosen because `ggml_flash_attn_ext` forces an F16 K/V cast
  that fights exact fp32 verification. A `FLASH_ATTENTION` primitive can be added later as a purely
  additive alternative once a GPU backend makes the perf/precision tradeoff worth it.

### Scope limitations (still true)

- **`KvCache` is single-sequence.** Contiguous append only — no ring buffer, no multi-stream/multi-sequence
  support, no `ggml_set_rows` index-tensor indirection like llama.cpp's `llama_kv_cache`.
- **KV cache storage is always F32.** No quantized cache types (`Q8_0` etc.). Weight quantization is
  handled per-model by the MIL exporter's `quantize=` kwarg (LFM2, Qwen3) — KV-cache quantization is a
  separate, still-untouched runtime concern (different mechanism, different point in the inference
  pipeline; check how the KV cache is currently allocated/typed before assuming it's a trivial extension).
- **Sampling is greedy argmax only** (`Generator::argmax` in `src/core/generation.cpp`). No temperature,
  top-k, top-p, or repetition penalty.
- **Only one level of `repeat_for` nesting** is supported in the JSON graph schema
  (`include/loom/core/graph_topology.h`'s `RepeatBlock::nodes` is a flat `vector<TopologyNode>`, not
  recursive).
- **`GgufModel::hparam_env()` only surfaces numeric scalar KV types** into the `SymbolEnv`; string, bool,
  and array-typed `loom.*` KVs are silently skipped.
- **No chunked/windowed inference for long Conformer-CTC audio** — cost grows O(n²) with length (relative
  position attention over the whole clip at once), per an explicit prior choice to defer this. Would need
  window size/overlap selection and stitching per-window CTC token sequences at the boundaries.
- **Only the small (16-layer, `d_model=176`) Conformer-CTC checkpoint has been verified.** Larger variants
  should work unmodified (topology is generated entirely from `model_config.yaml`) but this is unexercised.
- **Remaining BPE pretokenizer families** beyond the ~40 already in `bpe_vocab.cpp`'s `pre_spec_table()`
  (CJK-script splitters, case-transition/camelCase shapes, `byte_encode=false` SPM-style-BPE families like
  gemma4/sarvam-moe) raise a named error rather than being supported — bounded, add one when a real model
  needs it, per `pre_spec_table()`'s own comment.
- **General multi-scheme quantization tool.** `tools/quantize/quantize_gguf_q8_0.py` is Q8_0-only,
  single-model-shaped. A model-agnostic tool covering more of ggml's quant families, plus a
  per-tensor-role policy (skip norm weights/embeddings) rather than blanket quantization, is unbuilt.
- **Attention-variant primitives beyond `ATTENTION`/`REL_POS_ATTENTION`/`REL_POS_ATTENTION_SHAW`** (e.g.
  a dedicated flash-attention op) — add only when a real model needs one, not speculatively.

### Minor cleanups

- `KvCache::write_k/write_v/read_k/read_v` use `std::vector::at()` for layer-index bounds checking, which
  throws `std::out_of_range` rather than a `loom::Error` subtype. A malformed topology's `"layer"` attr
  could in principle trigger this uncaught-by-`catch (loom::Error&)` path — low risk today since the layer
  index always comes from `repeat_for`'s own loop bound, not arbitrary user input.
