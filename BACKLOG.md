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
- **NeMo Conformer-CTC-small — DONE, fully numerically verified as of 2026-07-24** (see the dedicated
  "CONV_2D bug — ROOT-CAUSED AND FIXED" entry further down this same section for the final two bugs that
  got it there: a silently FP16-rounded constant weight, and a completely dropped conv bias). The
  historical narrative below (data-flow/masking bugs found and fixed getting the model to trace and run at
  all) is kept for the reasoning trail, not because anything in it is still open.
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

  **The separate, previously-masked numerical bug (raw log-mel CMVN-normalized features not matching an
  independently-computed `compute_mel_features()` reference) is now root-caused and fixed.** Confirmed
  by calling the REAL checkpoint's `nemo_asr` preprocessor module directly (bypassing MIL/coremltools
  entirely) and diffing its output against `compute_mel_features()`: real NeMo's own
  `AudioToMelSpectrogramPreprocessor.get_seq_len()` computes `floor(length/hop_length)` valid frames --
  exactly ONE LESS than the true STFT frame count `floor(length/hop_length)+1` its own center-padded STFT
  produces -- even when `length` is the waveform's true full sample count with no real padding at all.
  NeMo's `normalize_batch()` therefore always excludes the LAST STFT frame from CMVN mean/variance (N-1
  denominator, N = valid frames, one fewer than total) and zeroes it outright in the final output. This is
  a genuinely distinct bug from the encoder's own `calc_length`/`all_paddings` masking fixed above (that
  one really is a no-op here) -- it's a separate structural off-by-one specific to the mel frontend's own
  frame-validity convention, present regardless of whether any real padding occurs. Fixed in
  `mel_common.py` (documented), `reference_forward_conformer.py`'s `compute_mel_features()`, and
  `convert_conformer_ctc.py`'s CMVN section (new `valid_frames_expr()` = `t_mel_expr() - 1`; a `VIEW` slices
  off the last frame before both `SUM_ROWS` reductions, and the final normalized output has its last frame
  zeroed via slice+`PAD_1D`). The identical copy-pasted bug was also fixed in `convert_parakeet_rnnt.py`/
  `convert_parakeet_tdt.py` and their reference scripts (same mel frontend, verbatim copy) -- unverified
  against a real checkpoint (none available locally; their e2e tests skip for that reason), but mechanically
  identical to the Conformer-CTC fix, which IS verified: `test_e2e_conformer_ctc`/
  `test_e2e_conformer_ctc_dynamic_length` both pass end-to-end (max logit abs diff <= 1e-3) against
  regenerated fixtures, and the fixed mel features were independently confirmed to match the real
  `nemo_asr` preprocessor's actual output to ~1e-6 (float32 rounding only).
- **Parakeet-TDT (encoder only) — DONE and numerically verified.** `export_parakeet_tdt_mil.py` traces
  the REAL `nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel`'s preprocessor + FastConformer
  `ConformerEncoder` directly (the same MIL-tracing approach as Conformer-CTC-small; the LSTM prediction
  network + joint network + greedy TDT search loop stay as the existing hand-derived small topologies /
  `TdtDecoder` C++ driver — autoregressive host-side control flow, not something a static traced graph can
  express, same split as every other ASR/LLM model in this project). Verified against
  `tools/convert_nemo/reference_forward_parakeet_tdt.py`'s independent hand-rolled reference at the real
  `nvidia/parakeet-tdt-0.6b-v3` checkpoint (`test_e2e_parakeet_tdt_mil_export.cpp`, max abs diff 0.092 at
  the encoder output, tolerance 0.12 — see that test's own tolerance comment for the STFT-precision
  reasoning below). `CONV_2D_DW` (already existed for Conformer-CTC's own depthwise pointwise convs) needed
  zero changes to support FastConformer's real `dw_striding` subsampling; the exporter's existing
  groups>1-detection in its generic `conv` translation already mapped it correctly.

  Finding this real, working end-to-end took one genuine, general exporter bug fix (in
  `tools/loom_mil_compiler/exporter.py`, applies to every model using this exporter, not just Parakeet):

  **The `less`-bypass heuristic (originally added for Conformer-CTC-small's `calc_length`/`all_paddings`
  tracing bug, see above) was too aggressive and silently corrupted a DIFFERENT, semantically-real NeMo
  masking pattern that happens to share the exact same `arange(T) < f(length)` shape.** Real NeMo's mel
  frontend (`nemo/collections/asr/parts/preprocessing/features.py`'s `FilterbankFeatures.forward`/
  `normalize_batch`, traced for REAL here — unlike Conformer-CTC-small's own MIL export, which was never
  numerically verified until this session and so never caught this) deliberately treats the LAST STFT frame
  as invalid (the same "last frame always invalid" convention already root-caused and hand-replicated in
  `convert_conformer_ctc.py`/`convert_parakeet_tdt.py`'s own `valid_frames_expr()`) — this is NOT a tracing
  artifact, it's intentional, and the old bypass logic forced it to always-valid anyway because it couldn't
  tell "these two sides are the same quantity, safe to force-equal" (the genuine `calc_length` tracing bug)
  apart from "these two sides are DELIBERATELY different quantities" (CMVN's `T` vs. `T-1`). Fixed by adding
  a real identity check before bypassing: derive BOTH sides' real symbolic formulas (the range's own length
  via the existing `range_1d` case of `_infer_dynamic_dim_expr`; the length-side's real formula via a new,
  more general `_resolve_scalar_expr` extension — added `select`/`where` handling, taking the "b"/false
  branch per this exporter's own "real audio is never the degenerate edge case" convention, plus a base
  case resolving the bare `"length"` input itself to `"n_tokens"`) and compare them AS STRINGS; only bypass
  when they're identical. Verified end to end: the mel-frontend's own last-frame zeroing is now correctly
  preserved (confirmed via a standalone preprocessor-only debug export: last frame exactly `0.0`, matching
  `compute_mel_features()`; per-frame diff elsewhere ~0.01–0.02, attributed to coremltools' OWN
  `complex_stft` MIL lowering computing the DFT phase matrix in fp32 arithmetic throughout — see
  `lower_complex_dialect_ops.py`'s `_calculate_dft_matrix`, `cos(2*pi*i*j/n_fft)` computed at fp32 with
  angles up to ~3200 radians for `n_fft=512` — vs. `mel_common.py`'s own kernels, built once in float64 and
  only rounded to fp32 at the final GGUF weight write). Full `ctest` clean, zero regressions (LFM2's own
  MIL export re-verified byte-exact).

  **CONV_2D bug — ROOT-CAUSED AND FIXED (two real, general exporter bugs, both applying to every model
  this exporter has ever produced, not just Conformer-CTC).** Re-testing Conformer-CTC-small's own MIL
  export with the masking fix above (previously never numerically verified at all, since no test compared
  its output against a reference) found its logits diverging from `reference_forward_conformer.py` by ~19.
  Isolated via progressively smaller debug wrappers (`preprocessor`-only, `preprocessor+pre_encode`, then
  just the raw `conv[0..3]` Sequential layers) diffed against real NeMo's own EAGER (non-traced) intermediate
  activations (captured via `register_forward_hook`, confirming the exact `unsqueeze(1)`/reshape NeMo
  applies before the first `Conv2d` — ruling out a mistake in the hand reference, not just the export) down
  to the SECOND subsampling `conv[2]`+ReLU stage, where only 23/176 output channels matched exactly.

  1. **`ct.convert()`'s own default (`compute_precision=None`) silently FP16-rounds every constant weight,
     even under `convert_to="milinternal"`.** Confirmed directly: coremltools' own `_converters_entry.py`
     `_need_fp16_cast_pass(None, "milinternal")` returns `True` (`"milinternal" != "neuralnetwork"` is the
     entire condition — only `compute_precision=precision.FLOAT32` explicitly disables it). The exported
     GGUF's own `conv.2.weight` tensor (visible in its MIL var name, literally suffixed `_to_fp16`) matched
     `real_weight.astype(np.float16).astype(np.float32)` bit-for-bit. Harmless for a small kernel (few taps
     to accumulate rounding error over) but this stage sums 176×3×3=1584 taps per output channel, turning
     ~1e-2 per-weight relative rounding error into a real, visible output error. Fixed by adding
     `compute_precision=ct.precision.FLOAT32` to every `ct.convert()` call in the project (all of
     `export_conformer_ctc_mil.py`/`export_parakeet_{tdt,rnnt}_mil.py`/`export_hf_causal_lm.py`
     [Qwen3 + LFM2's own monolithic/atomic exports go through this one] /`submodule_export.py`/
     `tools/convert_lfm/{make_lfm2_gguf,export_profiles_demo}.py`).
  2. **The exporter's `conv` translation (`exporter.py`'s `op_type == "conv"` branch) never read MIL's
     OPTIONAL `"bias"` input at all — every biased conv1d/conv2d in every model this exporter has ever
     produced silently lost its bias term.** Unlike its sibling `conv_transpose` translation just below
     (which explicitly REJECTS a non-zero bias rather than silently dropping it), this branch had no bias
     handling whatsoever — confirmed directly by reading the emitted topology JSON: `CONV_2D` fed straight
     into `RELU` with no `ADD` node in between. This was present even for the FIRST (1-channel) subsampling
     stage (isolated separately, ~3.0 max diff on its own) — the "23/176 channels" signature was really
     "some channels have small enough real biases that dropping them barely shows," not a channel-count-
     specific bug at all. Fixed by composing `CONV_2D`/`CONV_1D` + `RESHAPE`(bias to `[1,1,OC,1]` or
     `[1,OC,1]`, matching this project's own established `convert_conformer_ctc.py` convention) + `ADD`,
     the same pattern the existing `linear` translation already used for ITS OWN optional bias.

  A standalone multi-channel `CONV_2D` unit test (`op("CONV_2D")` directly, IC=4/OC=3/kernel=3×3/stride=2,
  against `F.conv2d`) confirmed the ggml PRIMITIVE itself was correct throughout (diff ~2e-6) — this was
  never a `ggml`/primitive bug, purely an exporter translation gap.

  **Result after both fixes**: Conformer-CTC-small's encoder output matches the reference to ~5e-5 (was
  ~2-30 before), and the full CTC logits match to ~1.6e-4 (was ~19) — tightened
  `test_e2e_conformer_ctc_mil_export.cpp`'s tolerance to `1e-3`, matching the bespoke conversion's own.
  Parakeet-TDT/RNNT's own "STFT fp32 precision ceiling" tolerances from earlier in this session turned out
  to be measuring these SAME two bugs, not real STFT precision noise: re-exporting both with the fix
  dropped their diffs from ~0.09/~1.14 to ~5e-6/~1e-5 — tightened both tests' tolerances to `5e-2` (matching
  their own bespoke-conversion counterparts) accordingly. **Lesson for next time a "toolchain precision
  ceiling" tolerance gets written: verify it's real by fixing the cheap, structural possibilities (a
  disabled precision flag, a dropped bias/scale term) FIRST** — a plausible-sounding "coremltools does fp32
  trig at large angles" story turned out to be almost entirely beside the point.

  **The masking-bypass refinement needed ONE more correction after this fix, not before.** The original
  "does the range/length formula match AS A STRING" identity check (this session's earlier fix) was
  actually too CONSERVATIVE for Conformer-CTC-small specifically: `MaskedConvSequential`'s own per-stage
  masks (`_create_mask`/`apply_channel_mask`, applied after each strided subsampling conv) get fed a
  "length" that already passes through the mel frontend's own `get_seq_len` (T−1) convention, so their OWN
  "range vs. bound" formulas come out structurally unequal too — string-identical to the CMVN case's own
  shape, but NOT the same thing to leave un-bypassed (confirmed by a controlled experiment: force-bypassing
  every "less" match, CMVN included, dropped Conformer-CTC's own encoder diff from 2.09 to 0.13 once the
  fp16/bias bugs above were ALSO fixed — leaving these encoder-level comparisons real was making things
  WORSE). Replaced the string-equality check with a NUMERIC one: evaluate both sides at 8 different concrete
  `n_tokens` probes (a tiny sandboxed `eval()` with only `floor` exposed) and only refuse to bypass when
  `range == length + 1` at EVERY probe (CMVN's actual, provable relationship) — default to bypass otherwise,
  including the encoder-level case (which is off by 1 only some of the time, depending on integer-halving
  parity, and empirically wrong to preserve). Final result: 0.00005 max diff on the encoder, better than
  either the pure-string-check (2.09) or the force-bypass-everything experiment (0.13) alone.

  **Also found and fixed in the process (unrelated, general project hygiene, not an exporter bug):** the
  `ref/` fixtures under `/home/flavio/.claude/tmp/{conformer_ctc,parakeet_tdt}_model/` were stale, generated
  2026-07-18 — BEFORE the 2026-07-24 mel-frontend CMVN off-by-one fix (commit `482a559`) — so comparing a
  freshly-exported model against them was comparing against outdated expected values. Regenerate via
  `tools/convert_nemo/reference_forward_{conformer,parakeet_tdt,parakeet_rnnt}.py` any time a mel-frontend/
  preprocessing fix lands, not just once at fixture-creation time.

- **Parakeet-RNNT (`nvidia/parakeet-rnnt-0.6b`, encoder only) — DONE and numerically verified.**
  `export_parakeet_rnnt_mil.py` (near-identical to `export_parakeet_tdt_mil.py`) exported and verified
  cleanly on the first real run, no new exporter issues, and confirms the CONV_2D bug above never applied
  to either Parakeet checkpoint (both use `dw_striding` — depthwise + 1×1-pointwise subsampling stages
  only, no plain multi-channel conv). `test_e2e_parakeet_rnnt_mil_export.cpp`: max abs diff ~1e-5 (tolerance
  5e-2, matching its own bespoke-conversion counterpart) against `reference_forward_parakeet_rnnt.py` —
  down from ~1.14 before the fp16/bias fixes above (see that section for why the original ~1.14 was
  wrongly attributed to STFT precision amplified by this checkpoint's `xscale=32.0`). Full `ctest` clean,
  zero regressions.
- **VITS (piper) — DONE and numerically verified, `export_vits_mil.py`.** Traces the REAL
  `piper_train.vits.models.SynthesizerTrn` submodules directly (TextEncoder, StochasticDurationPredictor,
  ResidualCouplingBlock, HiFi-GAN Generator) via three wrapper modules mirroring
  `tools/convert_piper_vits/convert_vits.py`'s own phase split (`stats`/`logw`/`flow_vocoder` —
  GraphTopology supports one declared output per topology, and the duration predictor's own output
  determines the total frame count, a genuinely data-dependent value the host must compute between
  phases 1 and 2). Unlike every MIL export before it, this one needed real NEW infrastructure, not just
  new op translations, because two of piper's real modules use patterns plain `torch.jit.trace` cannot
  correctly capture at all (not just "not yet translated" — genuinely mis-traces):

  - **`MultiHeadAttention`'s relative-position shift trick (`_get_relative_embeddings`/
    `_relative_position_to_absolute_position`/`_absolute_position_to_relative_position`) uses `F.pad`
    with a DYNAMIC pad amount on a rank≥3 tensor.** coremltools' own `pad` converter hard-rejects this
    at the frontend level — confirmed via its own source comment: `mb.pad`'s dynamic-padding support is a
    genuine CoreML **runtime** limitation (rank-1 tensors only), not a converter gap. Fixed with two
    per-call-site tricks, both verified bit-identical to the real piper code (across lengths both above
    and below `window_size+1`, i.e. both of the real code's own branches) via a standalone pure-eager
    equivalence check before ever tracing anything: (1) `_get_relative_embeddings` pads its FIXED-size
    learned table by a generous STATIC bound unconditionally, then dynamically SLICES out the real
    window (only dynamic pad *amounts* are the problem; dynamic slicing is already well-supported) — the
    real code's window in the table's own pre-pad coordinates is provably the same formula regardless of
    which of its two branches would have fired, so this is exact, not an approximation; (2) the two
    "shift trick" helpers instead pad via CONCAT (no rank restriction), building the zero block from a
    same-dynamically-sized SLICE of the tensor itself multiplied by 0 rather than constructing a raw
    dynamic-shape `torch.zeros(...)` argument list (whose own frontend conversion has a different,
    unrelated bug: `.narrow` with a dynamic length fails outright; `x[..., :right]` slicing doesn't).
  - **`StochasticDurationPredictor`'s `ConvFlow` uses a boolean-mask-indexed rational-quadratic spline
    transform (`transforms.py::piecewise_rational_quadratic_transform`) — genuinely data-dependent output
    shape, not something any pad/slice rewrite can fix.** Bridged via a new custom op,
    `torch.ops.loom.spline_inverse` (`tools/loom_mil_compiler/vits_spline_op.py`, registered via
    `torch.library.custom_op` — same pattern as the older `aten_to_loom` pipeline's
    `loom::rope_neox`/`loom::attention`, applied here to coremltools' MIL frontend instead via a new
    `@register_torch_op` hook), into MIL's `loom_spline` op. That MIL op turned out to already exist as
    unwired scaffolding in `dialect.py` from an early, abandoned prototype (never imported anywhere,
    and broken against the current coremltools version — `TensorInputType` now requires `type_domain`,
    fixed as part of this work) — composed by a new `exporter.py` translation down to the
    already-independently-verified `RQ_SPLINE_INVERSE` ggml primitive
    (`src/ops/primitives_spline.cpp`, itself already using an elementwise inside/outside-mask blend
    instead of boolean indexing, precisely because it hits the identical problem in C++ terms). Verified
    standalone (a tiny isolated `ConvFlow`-shaped trace, `RQ_SPLINE_INVERSE`'s own C++ output vs. the
    real `piecewise_rational_quadratic_transform`, max abs diff ~1.5e-6 including out-of-tail-bound
    inputs) before integrating into the full model.

  Two real, general (not VITS-specific) exporter bugs found and fixed getting the full model to build
  and run correctly at a real T (62, "Hello world, this is a test.") rather than just at the dummy trace
  T (both previously masked because every earlier MIL model's own dummy/real trace lengths happened not
  to expose them):
  - **`_resolve_scalar_expr`'s cycle guard (`id(v) in _seen`) treated any DAG DIAMOND as a false cycle.**
    Unlike `_infer_dynamic_dim_expr`'s single-input producer-chain walk (where a "cycle" really would be
    unreachable), this function recurses into TWO operands per arithmetic op, so the SAME upstream scalar
    legitimately gets reached via two different paths in an ordinary expression tree — confirmed on
    `end = start + 2*length - 1`, where `start` and the `2*length` term both independently reference the
    same `length`-derived `gather` var. The second reference hit the guard and silently returned `None`,
    so `slice_by_index`'s "end" bound fell back to the axis's full unsliced extent while "begin" (whose
    own resolution never revisits the var a second time) looked completely fine — the sliced
    relative-position table came out ~34x too long at T=62, one axis short of crashing GraphBuilder's own
    RESHAPE element-count check downstream. MIL/SSA graphs are acyclic by construction (an op's inputs
    always name EARLIER-defined vars, never itself), so the guard was simply removed rather than scoped
    more narrowly.
  - **`_infer_dynamic_dim_expr` didn't walk through `leaky_relu` or `conv_transpose`.** HiFi-GAN's 3-stage
    upsample chain interleaves `conv_transpose` (real, biased, `pad_type="custom"` — the FIRST biased
    conv_transpose and the first non-"valid"-padding conv_transpose this exporter has ever hit; composed
    as a full/valid conv_transpose + bias-ADD + crop-VIEW, needing a new `conv_transpose` case in
    `_infer_dynamic_dim_expr` itself for the crop's own dynamic length — real formula `(L_in-1)*stride -
    (pad_before+pad_after) + kernel`) with `leaky_relu` activations between stages. Missing `leaky_relu`
    from the unary-passthrough set broke the recursive dynamic-length derivation exactly at that boundary
    — stage 2 of 3 silently read only stage 1's first 1/8th (64 of 512 real elements), corrupting the
    rest of the vocoder's output. `conv_transpose` also needed adding to `get_var_info`'s own
    "always re-derive, don't blind-substitute" producer set (alongside the existing `reshape`/`fill`
    cases) — MIL's own type inference already reports a COMPOSITE formula for a conv_transpose's output
    (e.g. `"8*is50"` for a stride-8 upsample), and blind per-symbol substitution is provably wrong
    whenever the input is itself an already-derived length (chained upsample stages): `is50` doesn't mean
    "n_tokens", it means "8*n_tokens" one stage up.
  - Also fixed along the way (smaller, mechanical): `torch.flip` (VITS's `Flip` module) had no MIL
    translation at all — composed via the same `GET_ROWS`-with-baked-reversed-index trick
    `convert_vits.py`'s own hand-built `add_flip` already used, restricted to ne_axis==1 (GET_ROWS' own
    native reversal axis) since that's the only pattern needed so far. `leaky_relu` and `F.gelu` (DDSConv)
    had no translation/had a latent bug respectively (`gelu`'s MIL op carries an extra "mode" string
    input the generic OP_MAP fallback would otherwise add as a bogus second ggml node input — rejected
    outright, rather than silently mismatched, if a future model traces gelu's TANH-approximate variant
    instead of the EXACT/erf one ggml's own `GELU` primitive always computes). `op_gelu`
    (`src/ops/primitives_basic.cpp`) gained the same "cont a non-contiguous input first" fix `op_softmax`/
    every conv primitive already had, needed once DDSConv started feeding a real strided intermediate
    straight into it.

  **Numerically verified end-to-end against `reference_forward_vits.py`'s real-checkpoint reference at
  T=62**: `stats` max abs diff ~3.3e-6 (`m_p`) / ~9.5e-7 (`logs_p`), `logw` max abs diff ~7.9e-6,
  `flow_vocoder` waveform max abs diff ~5.4e-8 (small-scale Tp=8 case) / ~5.7e-7 (realistic-scale T=194
  case, see below). Full `ctest` clean throughout (zero regressions).

  **Packaged and wired up the same way every other Lua-ported model is** — NOT into `loom::VitsDriver`
  (`src/core/vits_driver.cpp`, the pre-procedural-generalization C++ driver, now legacy/oracle-only, kept
  only as the reference the bespoke topology's own `vits_driver.lua` was checked against when that
  architecture landed). `export_vits_mil.py` packs all three topologies (`stats`/`logw`/`flow_vocoder`)
  plus a new hand-written orchestration script, `tools/convert_piper_vits/vits_driver_mil.lua`, into one
  combined `vits_mil.gguf` (mirroring `convert_vits_lua_all.py`'s own packing for the bespoke topology).
  The cross-phase host logic (duration-based frame expansion, RNG sampling) is genuine host control flow
  no amount of MIL tracing can produce either way — it was always hand-written, in Lua, regardless of
  whether the topologies underneath are hand-built or machine-traced; `vits_driver_mil.lua`'s own math is
  IDENTICAL to `vits_driver.lua`'s, differing only in the new topologies' own conventions (no host-side
  `emb_rel_k`/`emb_rel_v`/`attn_mask` plumbing needed at all — computed in-graph now; `stats`/`z_p` are
  T-fast, not channel-fast — see the script's own comments for why). One real exporter-side bug surfaced
  building this: each independently-traced phase's own topology serializes small internal constants under
  auto-generated, per-program-local SSA names (e.g. `"_235"`) that trivially collide by coincidence across
  three SEPARATELY traced programs without meaning the same thing — fixed by giving each phase's own
  `generate_graph_topology` call a real per-phase `func_name` (namespacing every weight as
  `f"{phase}.{weight_name}"`) instead of the `"main_topo"`/`profile="monolithic"` combo every other
  single-topology `export_*_mil.py` script uses (which deliberately disables namespacing, correct for "one
  file per topology" but wrong for "three phases sharing one file").

  **Found and root-caused a real, previously-uncaught correctness bug in the BESPOKE topology while
  building the end-to-end driver test — not a bug in the new MIL pipeline.** A first version of
  `test_e2e_vits_mil_lua_driver.cpp` compared the new pipeline's full-synthesis waveform against
  `loom::VitsDriver`'s own (the bespoke topology's oracle) and found a large, real divergence (~0.22
  absolute, against a ~0.01-0.02 rms signal) — NOT the expected ~1e-6-level match. Isolating it (dumping
  the real end-to-end `z_p` from both pipelines — confirmed numerically IDENTICAL to ~5e-6, ruling out the
  new pipeline's own `stats`/`logw`/generate_path/RNG math — then feeding that same real `z_p` into (a)
  the new MIL `flow_vocoder` topology, (b) the bespoke `flow_vocoder` topology, and (c) a real PyTorch
  `ResidualCouplingBlock`+`Generator` forward pass, all three independently) found: **(a) matches (c) to
  ~1.2e-6; (b) diverges from (c) by ~0.22 — the same magnitude as the original end-to-end mismatch.** The
  bespoke topology (`convert_vits.py`'s hand-built `RESIDUAL_COUPLING_LAYER_REVERSE`/HiFi-GAN composition)
  is the one that's wrong, not the new one. Root cause of why this was never caught: the bespoke path's
  own numerical verification (`reference_forward_vits.py`/`test_e2e_vits_flow_vocoder_reference.cpp`) has
  ONLY ever used a small-scale synthetic `z_p` (`torch.randn(1,192,8)*0.5`, so values roughly in
  [-1.9, 2.0]) — real end-to-end `z_p` (duration-expanded, T=194 for this test's real input) has values up
  to ±24 (confirmed: re-verified the new MIL topology against i.i.d. random `z_p` at that SAME wide range,
  still matched to ~5.7e-7 — magnitude alone isn't what triggers it, something about the bespoke
  primitives specifically mishandles it). The exact mechanism inside `primitives_flow.cpp`'s
  `RESIDUAL_COUPLING_LAYER_REVERSE`/the hand-composed HiFi-GAN ops was NOT further chased down (out of
  scope for finishing the MIL export) — this is a known, reproducible, but not-yet-root-caused bug in
  the bespoke path specifically, left as-is since the MIL-traced path is intended to supersede it. A new
  fixture (`reference_forward_vits_widerange.py`, T=194, `z_p` up to ±24) plus
  `test_e2e_vits_mil_flow_vocoder_reference.cpp` (green) captures this as a proper regression test for the
  new topology; the existing small-scale bespoke fixture/test are left untouched (still green, but now
  known not to be a reliable correctness signal at realistic scale).
  `test_e2e_vits_mil_lua_driver.cpp` was rewritten accordingly — it no longer compares against the
  (known-unreliable-at-scale) bespoke oracle, only checks the full Lua-orchestrated synthesis runs
  end-to-end and produces a finite, plausible waveform; the real numerical confidence comes from the
  per-phase reference tests instead.

  **Open, not yet done:** root-causing and fixing the bespoke `flow_vocoder` bug itself (or just retiring
  the bespoke topology/driver in favor of this one, given the correctness gap just found); deciding
  whether to keep the bespoke topology around at all afterward.
- **Kokoro, Matcha-TTS, SupertonicTTS, StyleTTS2 — not started.** VITS's own experience revises the
  earlier optimistic prediction here: "should trace as one static graph with noise supplied as an input
  tensor" undersold the real risk — TWO of piper's real modules needed genuinely new infrastructure
  (a coremltools *runtime* pad limitation, not just a missing translation; a custom op bridge for
  boolean-mask-indexed math no rewrite can avoid), not simply "new ops to wire up". Kokoro/Matcha-TTS/
  SupertonicTTS should still be checked against the SAME two failure modes specifically (dynamic pad on
  rank≥3 tensors anywhere in their own attention/positional machinery; boolean-mask-indexed transforms
  anywhere in their own flow/sampling code) before assuming they'll trace cleanly. StyleTTS2 is still the
  one likely to stay bespoke regardless — its diffusion sampler's ~3e-3 residual mismatch persisted even
  with hand-matched float32 host math, and an auto-traced version gives less control to chase that kind
  of thing down.

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
