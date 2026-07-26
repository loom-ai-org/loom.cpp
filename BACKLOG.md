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
- **Kokoro — DONE (2026-07-26).** Unlike every model MIL-exported so far, Kokoro leans heavily on
  `torch.nn.LSTM` (TextEncoder, DurationEncoder×3+AdaLayerNorm, `predictor.lstm`, F0Ntrain's shared
  LSTM) — ggml has no native LSTM op, and `generate_graph_topology` hard-fails on any traced `lstm`/
  `gru` MIL op (`recurrent.py`'s `build_lstm_cell_topologies` + a per-timestep host stepper is the only
  path, same as the bespoke pipeline's own `BiLstmStepper` already uses) — so full end-to-end MIL
  tracing of those pieces would mean auto-splitting a single traced module into alternating static/
  recurrent segments, real unimplemented infrastructure (`generate_graph_topology`'s own comment calls
  this out explicitly), not attempted here. Deliberate scoping decision instead: MIL-trace only the
  LSTM-free parts (CustomAlbert+bert_encoder — not yet started; the `istftnet.py` decoder/vocoder back
  end — done, see below), and reuse the existing bespoke `duration_predictor`/`text_encoder`/`f0n` GGUFs
  unchanged for the LSTM-bound pieces once a driver exists to wire them together. Revisit "is
  auto-splitting worth building" only if a future model makes LSTM-avoidance infeasible.

  **`decoder_vocoder` phase (Decoder.encode/decode + SineGen + STFT + Generator — replaces FOUR bespoke
  scripts, `convert_kokoro_{decoder_core,sinegen,stft,generator}.py`, with one combined MIL trace) —
  traces, exports, and mostly builds; one open shape-consistency bug remains.** `export_kokoro_mil.py`
  traces the REAL `kokoro.istftnet.Decoder` (`disable_complex=True` checkpoint construction, needed for
  a conv-based rather than complex-dtype STFT — see the STFT rewrite below for why even that path isn't
  used as-is). No LSTM anywhere in this subgraph, confirmed the highest-value MIL-tracing target per the
  prior "check the same two VITS failure modes" prediction — which undersold the REAL scope again: this
  phase hit a long tail of genuinely new infrastructure, none of it a dynamic-pad-rank≥3 or
  boolean-mask-index case at all:

  - **`torch.rsqrt(torch.tensor(2))`** (`AdainResBlk1d.forward`'s own `1/sqrt(2)` residual scale) traces
    the constant as int64, which MIL's `rsqrt` op rejects outright (`dtype[int32]` not in `fp16/fp32`).
    Replaced with the equivalent plain float constant, matching how the bespoke `convert_kokoro_f0n.py`
    already reduces this exact expression to a constant.
  - **`torch.multiply`/broadcast-along-two-different-axes-at-once** (`SineGen`'s
    `f0 * torch.arange(1,dim+1).view(1,1,-1)`, f0 broadcasting on axis 1 while the arange broadcasts on
    axis 2): `torch.multiply` itself isn't implemented by coremltools' torch frontend at all (use `*`);
    once fixed, the resulting MIL `mul` still can't compose to a single `ggml_mul` call regardless of
    operand order — ggml's broadcast model requires ONE operand's shape to be a per-axis divisor of the
    other's, uniformly, and this shape needs axis 1 from one operand and axis 2 from the other
    simultaneously. Composed instead via `dim` (9, static) unrolled per-harmonic SCALE+CONCAT calls —
    the exact fix `convert_kokoro_sinegen.py`'s own bespoke topology already needed for this identical
    shape ("no generic outer-product/broadcast-repeat primitive existed").
  - **`F.interpolate(mode='linear')` on a rank-3 tensor** (SineGen's own phase pre-/post-filtering
    downsample-then-upsample-by-300, and separately `UpSample1d`'s plain 2x nearest shortcut) hard-fails
    coremltools' frontend ("input to torch_upsample_bilinear must have rank 4" — 1D linear/2D bilinear
    share one dialect op requiring rank 4 always). Fixed via an `unsqueeze(2)→interpolate→squeeze`-back
    trick — but chaining two such calls (downsample then upsample) with a squeeze/re-unsqueeze in
    between the two loses the SECOND call's static rank in coremltools' own MIL type inference (a real,
    narrow, reproduced-in-isolation-first coremltools bug, not a rank bug in this code) — fixed by
    keeping the intermediate `cumsum` at rank 4 throughout (dim=3, not transposing back to rank 3
    between the two interpolate calls), verified bit-level equivalent to the original modulo cumsum's
    own floating-point reduction-order noise (~6e-5 abs).
  - **A genuine numpy/coremltools version bug**: `int(np.array([1.0]))` — the exact shape coremltools'
    OWN `_cast` op converter produces internally for a dynamic-size `F.interpolate(recompute_scale_
    factor=True)` call's degenerate (scale=1) axis — raises on numpy>=1.25 (silently worked pre-1.25;
    `_cast`'s own shape check already claims to allow "a length-1 tensor", just never squeezes it before
    calling `int()`/`float()`). Self-contained monkeypatch in `export_kokoro_mil.py` (mirrors the
    existing `transformers`-version-pin stub convention already used for `CustomAlbert`), not a
    site-packages edit.
  - **Instance normalization (`nn.InstanceNorm1d(affine=True)`, `AdaIN1d`'s own norm)** had no ggml
    mapping (`instance_norm` MIL op). New dedicated exporter translation, same LAYER_NORM+MUL+ADD
    composition `layer_norm` already uses — but with gamma/beta's own affine axis (channel, ne[1])
    correctly reshaped to `[1,C,1]` before the MUL/ADD, unlike layer_norm's own gamma/beta (which index
    ne[0], the SAME axis being normalized) — a raw `[C]`-shaped MUL against `[T,C,1]` broadcasts against
    the WRONG axis otherwise (confirmed via a real `ggml_mul` shape-mismatch error before the fix).
  - **Depthwise/grouped `ConvTranspose1d`** (`AdainResBlk1d`'s learned upsample "pool":
    `kernel=3,stride=2,groups=dim_in,padding=1,output_padding=1`) — ggml has no grouped CONV_TRANSPOSE
    primitive at all. New composition in the exporter's `conv_transpose` translation: zero-stuff the
    input (insert `stride-1` zeros after each real sample via a PAD_1D+RESHAPE dummy-axis trick),
    symmetric `kernel_size-1` padding both sides, then an ordinary depthwise CONV_1D_DW with the KERNEL
    FLIPPED (baked as a new namespaced weight at export time, numpy `[:, :, ::-1]`) — the standard
    "conv_transpose = correlation with a flipped kernel over a zero-stuffed signal" identity. Confirmed
    (via `get_var_info` dumps) that MIL always traces a GROUPED conv_transpose with `pad=[0,0]`
    regardless of the real PyTorch padding/output_padding, deferring the real crop to a separate,
    already-independently-handled downstream `slice_by_index` — so this composition only ever needs the
    "valid" formula `(L_in-1)*stride+kernel_size`, never a real nonzero pad. Reuses the exact node
    sequence already verified in `convert_kokoro_f0n.py`'s own `add_depthwise_conv_transpose_upsample`
    (itself checked against `test_primitive_registry.cpp`'s
    `test_depthwise_conv_transpose_1d_via_composition`), generalized to a real traced op's own
    stride/kernel/channel-count and a REAL dynamic-length expression instead of that script's own
    hardcoded `$n_tokens`.
  - **New `ATAN` ggml primitive** (`src/ops/primitives_spline.cpp` sibling to the existing `ATAN2`):
    `torch.atan2`'s own MIL lowering doesn't always stay one opaque `atan2` op — for the STFT phase
    computation it decomposes to a plain `atan` op plus quadrant-correction (select/sign/etc., already
    supported), which had no ggml mapping at all. `ggml_map_custom1` wrapping `std::atan`, same "no
    native op, no viable composition" precedent `ATAN2` itself already established.
  - **New `INTERPOLATE_1D`/`upsample_nearest_neighbor`/`upsample_bilinear` exporter wiring**: the ggml
    primitive itself (`op_interpolate_1d`) already existed (added in anticipation of exactly this model,
    per its own comment), but had no exporter translation. MIL's `upsample_nearest_neighbor`/
    `upsample_bilinear` (core ops, post the "torch_upsample_to_core_upsample" SSA pass) always scale the
    LAST TWO axes (height=-2, width=-1) independently — but which of the two ends up holding the REAL
    (non-1.0) scale factor for a promoted-from-rank-3 1D call is NOT consistent (confirmed: Kokoro's
    `f0_upsamp` puts it at width, `UpSample1d`'s own plain call puts it at height) — the new translation
    picks by VALUE (whichever of height/width is actually non-1.0), not by fixed axis position.
  - **New `CUMSUM` OP_MAP entry**: the ggml primitive (`ggml_cumsum`, already used internally by
    `RQ_SPLINE_INVERSE`) had no exporter-level MIL `cumsum` mapping at all. Direct translation (axis
    must be the trailing/ne[0] one, `exclusive`/`reverse` must be false — the only case
    `ggml_cumsum` implements and the only one this model needs).
  - **A genuine 0-D (scalar) constant serialization bug, general, not Kokoro-specific**: any traced MIL
    constant that folds down to a true scalar (first hit here: SineGen's own `2 * torch.pi` literal
    multiply, baked as its own GGUF weight rather than an inline attr) got written as a shape-`()` numpy
    array — GGUF/ggml has no 0-D tensor representation, and round-trips this as a tensor with a ZERO-
    length dimension (`ne[0]=0`) instead of a proper length-1 scalar, silently failing every downstream
    shape check that consumes it. Fixed in the `const` handling shared by every model this exporter
    produces: reshape to `[1]` before writing, matching ggml's own "scalar = shape `[1]`" convention.
  - **`get_var_info`'s dynamic-dim-derivation gating was itself a real, general bug, not just
    Kokoro-specific missing cases.** The gate ("only call the real `_infer_dynamic_dim_expr` derivation
    for a BARE symbol, or an explicit narrow per-op-type allowlist — reshape/fill/conv_transpose") was
    based on a theory that turned out false: `_infer_dynamic_dim_expr` ITSELF falls back to the exact
    same blind-substitution behavior for any op type/axis it doesn't specifically understand, so calling
    it unconditionally can never produce a worse answer, only sometimes a better one. The gate was real
    technical debt: `pad`, `concat`, and the new `upsample_*` cases each had a correct
    `_infer_dynamic_dim_expr` case that was silently unreachable from `get_var_info` whenever MIL's own
    type inference reported a COMPOSITE formula (not a bare symbol) for that op's output — confirmed on
    the STFT reflect-pad, which needs its own real length (`600*n_tokens`-derived) plus a constant 20,
    but without this fix silently substituted the wrong base quantity, computing `n_tokens + 20` instead
    — a ~600x error that produced a zero-length tensor two ops later. **Simplified: `get_var_info` now
    always attempts real derivation first**, removing the allowlist entirely (confirmed safe: VITS
    re-exported and both its own MIL numerical-reference tests still pass bit-identically after this
    change).
  - **New `_infer_dynamic_dim_expr` cases**: `concat` (any non-concat axis passes through from any one
    operand; the concat axis itself is the SUM of every operand's own real expression, not any single
    operand's — the first BRANCHING case in this function, which also exposed and fixed the same
    "shared `_seen` cycle guard wrongly rejects a legitimate DAG diamond" bug already root-caused once
    for VITS's `_resolve_scalar_expr` — removed here too, for the identical reason: MIL/SSA graphs are
    acyclic by construction); `upsample_nearest_neighbor`/`upsample_bilinear` (scale the matching
    height/width axis by its own real constant factor, passing every other axis through unscaled);
    `cumsum`/`atan`/`sin`/`cos` added to the shape-preserving unary-passthrough set; `mod` added to the
    elementwise-broadcast set (needed for `SineGen`'s own `rad_values % 1`).
  - **`ggml_pad_ext`/`ggml_pad_reflect_1d` (`PAD_1D`/`PAD_1D_REFLECT`) now `ggml_cont` their input if
    non-contiguous**, the same fix `op_softmax`/every conv primitive already needed once a real strided
    VIEW started feeding straight into them — first hit here by the STFT reflect-pad receiving a
    non-contiguous reshape/transpose chain from SineGen's own output.
  - **New `LoomGGUFExporter.symbol_overrides` mechanism**: this topology is the first with more than one
    genuinely INDEPENDENT dynamic axis — `asr` (this phase's own `$n_tokens` root) plus `f0_curve`/
    `n_curve` (2x), `noise_in` (600x), `wsum` (600x+20), none derivable from `asr` by any op (they're
    separately-traced LEAF inputs, fixed multiples only by this wrapper's OWN architecture, not
    recoverable from the graph at all — `get_var_info`'s own docstring already flagged this exact
    "engine's dynamic-shape support is genuinely single-axis" limitation as unaddressed). New
    `symbol_overrides` dict (raw MIL symbol string, e.g. `"is531"` → replacement expression, e.g.
    `"2*n_tokens"`) lets a caller who knows the real ratio (because it's inherent to their own wrapper's
    `forward()` signature) override it — populated in `export_kokoro_mil.py` from the REAL traced
    symbol names (`main_func.inputs[name].shape[axis]`), not guessed.

  **The shape-mismatch bug above IS ROOT-CAUSED AND FIXED — two real, general (not Kokoro-specific)
  bugs, both fixed the same session.** The original symptom (`x = x + x_source` hitting `a=[601,...]`
  vs `b=[31,...]`) was a red herring for what turned out to be TWO layered issues:

  1. **`_infer_dynamic_dim_expr` had no case for MIL's `instance_norm` op** (only `layer_norm` was
     handled). `AdaIN1d`'s own `nn.InstanceNorm1d` sits inside every `AdainResBlk1d` call in Generator's
     `resblocks`/`noise_res` — any length several conv_transpose/upsample hops downstream of an
     instance_norm output fell through to bare-symbol substitution, corrupting the SECOND (`i=1`)
     Generator upsample stage's own input length. Fixed by folding `instance_norm` into the existing
     `layer_norm` case (identical shape-preserving-over-every-axis formula, no new logic needed).
  2. **The deeper bug, only exposed once #1 stopped masking it: `ggml_is_contiguous()` is not a
     sufficient guard before ggml-cpu's own elementwise binary/unary compute kernels.** Reading
     `ggml_is_contiguous_n`'s real implementation (`ggml.c`) directly: it treats `ne[0]==1` as
     VACUOUSLY satisfying the "`nb[0]` == element size" check, regardless of the tensor's actual
     declared stride there — correct for most of ggml's own ne[0]-agnostic consumers, but
     `ggml-cpu/binary-ops.cpp`'s own vectorized compute loop asserts `nb00 == sizeof(src0_t)`
     unconditionally, with no such carve-out (confirmed by direct binary-level GDB inspection of the
     crashing tensor once source-level debugging hit a dead end — no debug symbols in the ggml shared
     libs — computing `ggml_tensor` field offsets from `ggml.h` by hand and reading `dst`/`src[0]`/
     `src[1]` straight out of registers/memory at the exact `ggml_compute_forward_{mul,sub,sin,...}`
     call site). First hit by SineGen's `f0 * float(k)` (this project's own trace-friendly rewrite of a
     real broadcast multiply): `f0` is fresh off a real `.transpose(1,2)` call, producing a genuinely
     PERMUTED tensor with `ne[0]=1` (the trailing torch axis) and a non-unit `nb[0]`.
     `ggml_is_contiguous()` reports it contiguous, every existing guard built on exactly that call let
     it straight through, and the compute-time `GGML_ASSERT` aborted with NO informative message at all
     — a raw crash predating every other informative-error convention this project already established
     for shape mismatches, not even a `SchemaError`. Once `op_mul` was fixed the identical crash
     resurfaced one at a time in `op_sub`/`ggml_compute_forward_sin` and finally in
     `primitives_mil.cpp`'s own `sub_broadcast`/`mul_broadcast`/`add_broadcast` helpers (used by
     `op_greater`/`op_less`/`op_select`/etc. — `_f02uv`'s `f0 > self.voiced_threshold` hits
     `sub_broadcast` directly, which had NO contiguity guard at all, unlike `op_sub`). Fixed with a new
     shared `ensure_packed(ctx, t)` helper (duplicated per-TU, matching this project's own established
     convention — see `promote_i32_to_f32`) that checks `nb[0] == ggml_type_size(t->type)` explicitly,
     not just `ggml_is_contiguous()`, applied broadly across EVERY elementwise unary/binary primitive in
     `primitives_basic.cpp` (add/sub/mul/div/floor_div/sqr/sqrt/rsqrt/log/atan/atan2/pow/sigmoid/tanh/
     exp/sin/cos/floor/silu/relu/leaky_relu/cumsum/softmax/softplus/gelu/swiglu/rms_norm/layer_norm/
     interpolate_1d/pad_1d/pad_1d_reflect) and `primitives_mil.cpp`'s three broadcast helpers plus
     abs/neg/sign — not just the one call site that happened to crash first, since every one of these
     shared the identical latent vulnerability whenever fed a real permuted tensor with a size-1 leading
     axis (the ones with `ggml_map_custom1/2`-based implementations — `rsqrt`/`atan`/`atan2`/`pow` — had
     an even sharper version of this bug: no crash at all, just SILENTLY WRONG values, since their own
     manual flat-index loops assume full packing without any check whatsoever).

  **Numerically NOT yet verified** (no reference exists for this phase), but now confirmed
  STRUCTURALLY correct end-to-end: `tests/test_e2e_kokoro_mil_decoder_vocoder_smoke.cpp` builds AND
  COMPUTES the full topology (not just builds it), producing a finite waveform. VITS's own MIL export
  re-verified bit-identical after every fix in this whole section (`test_e2e_vits_mil_lua_driver`:
  49671/49671 checks; `test_e2e_vits_mil_flow_vocoder_reference`: 7/7 checks) — the `ensure_packed`
  fixes are broad enough that a regression there would have been the most likely place to see one.

  **Update, 2026-07-26 (same day, completed): Kokoro's MIL export is DONE — both phases traced, built,
  numerically verified, and wired into a working end-to-end driver.**

  - **`albert_bert_encoder` phase added** (`AlbertBertEncoderWrapper` in export_kokoro_mil.py): traces
    the REAL `model.bert` (a `transformers.AlbertModel`) + `model.bert_encoder` (`Linear(768,512)`) as
    ONE combined topology, replacing `convert_kokoro_albert.py` + `convert_kokoro_bert_encoder.py`'s two
    hand-built graphs. `attention_mask`/`token_type_ids` are synthesized in-graph from `input_ids`'s own
    shape (`torch.ones_like`/`torch.zeros_like`) rather than declared as separate inputs — real usage is
    always a single, unpadded utterance. Deliberately does NOT apply the real code's own final
    `.transpose(-1,-2)` (the exact live-non-contiguous-view footgun `export_vits_mil.py`'s own
    `StatsWrapper` already found for VITS's `stats` output) — returns the natural (T,512) time-major
    layout instead; `kokoro_driver_mil.lua` converts via `from_row_major`, no transpose needed Lua-side
    either. Verified against `reference_forward_kokoro_albert_bert_encoder_mil.py`
    (`test_e2e_kokoro_mil_albert_bert_encoder_reference.cpp`): ~1.8e-6 mean / ~1.5e-5 max abs diff.
  - **Three real, general (not Kokoro-specific) bugs found and fixed getting this phase to trace/build/
    compute correctly**, none caught by structural verification alone:
    1. HF's "gelu_new"/`NewGELUActivation` (the exact tanh-approximate GELU formula) gets fused by
       coremltools' own `fuse_gelu_tanh_approximation` MIL pass into a `gelu(mode=TANH_APPROXIMATION)`
       op, which ggml's GELU primitive can't compute (always the exact erf formula). Fixed in
       `exporter.py`'s `gelu` handling: decompose TANH-mode `gelu` back into the identical explicit
       SQR/SCALE/ADD/MUL/TANH sequence `convert_kokoro_albert.py`'s own bespoke `gelu_new` helper already
       used for this formula.
    2. A `GET_ROWS` index traced through elementwise arithmetic (HF's `zeros_like(input_ids)` idiom,
       decomposed by coremltools to `input_ids - input_ids` rather than a plain fill) is int-typed at the
       MIL level but NOT at the ggml level: this project's generic elementwise primitives
       (`op_add`/`op_sub`/...) unconditionally `promote_i32_to_f32` both operands, so an all-int32 SUB
       still produces an F32 result fed straight into `ggml_get_rows`, which hard-asserts I32. Fixed in
       `exporter.py`: cast a `GET_ROWS` index to i32 whenever its PRODUCER op is one of the known-
       promoting elementwise ops (checked by producer op_type, not the unreliable declared MIL dtype).
    3. **`op_sub`'s own "scalar subtraction" shortcut was a real, general, silently-wrong bug**, found
       chasing the fix above's own downstream consumer: `SUB(a,b)` with `nelements(a)==1 < nelements(b)`
       computed `ggml_neg(b)` unconditionally — correct ONLY for the `(0.0 - b)` idiom it was named after,
       silently WRONG (dropping `a` entirely) for ANY other nonzero scalar `a`. First hit by HF's
       ubiquitous `1.0 - mask` attention-masking idiom (`get_extended_attention_mask`) — confirmed via a
       from-scratch minimal hand-built topology reproduction (not just observed in the full graph) before
       fixing. Fixed generally in `src/ops/primitives_basic.cpp`: explicitly `ggml_repeat` the smaller
       operand up to the larger's shape first, then subtract — valid for any value, not just zero.
  - **`decoder_vocoder` phase numerically verified** against
    `reference_forward_kokoro_decoder_vocoder_mil.py` (runs `DecoderVocoderWrapper` eagerly on real
    checkpoint weights + concrete non-zero inputs — the correct ground truth for this topology
    specifically, not the original untraced `Decoder.forward`, since every trace-friendly patch is
    documented as bit/mathematically equivalent). `test_e2e_kokoro_mil_decoder_vocoder_reference.cpp`:
    ~5.1e-4 mean / ~2.5e-2 max abs diff (at t_frames=40; a real HiFi-GAN-vocoder amplification ceiling
    from compounding per-phase floating-point noise, same category as StyleTTS2's own documented one —
    bisected per-phase first: decoder_core exact ~1e-6, the SineGen chain ~2e-3, forward STFT ~2e-7
    excluding a handful of atan2-boundary elements, Generator-core-with-exact-har ~3e-3).
  - **Two more real, general bugs found and fixed getting THIS phase numerically tight** (both
    invisible to the earlier structural-only smoke test, which only checked "builds and produces a
    finite waveform"):
    1. `_f02sine_traceable`'s `rad_values = (f0_values / self.sampling_rate) % 1` — `x % 1` on a tensor
       with Python-int divisor 1 — traces (via coremltools' own `remainder` lowering) to a raw MIL
       `sub(x, x)`, ALWAYS EXACTLY 0, discarding the entire fractional/phase signal (confirmed by reading
       the raw traced MIL ops directly: no `mod`/`floor_div` op survives at all). A genuine coremltools
       bug, not this project's — looks like their `x - floor_divide(x,y)*y` decomposition short-circuits
       `floor_divide(x,1)` to plain `x` (valid for `real_div`-by-1, invalid for `floor_divide`, which
       must still floor). Fixed by rewriting the SineGen patch as `x - torch.floor(x)` directly, sidestepping
       the broken lowering entirely (algebraically identical, ggml already has a `FLOOR` primitive).
    2. `torch.atan2(im, re)` in `VerifiedSTFT.transform` traces (for this model) not to MIL's fused
       `atan2` op but to `atan(im/re)` plus a manually-composed quadrant correction that covers the
       x<0 case's y>0/y<0 STRICT branches but omits the y==0,x<0 boundary entirely (real
       `atan2(0.0,-1)==+pi`; the decomposition silently returns 0). `im = im_raw - im_raw*boundary_mask`
       (zeroing the imaginary part at DC/Nyquist bins, matching real `torch.stft`'s own convention)
       produces exactly this trigger at ~5.9% of all phase elements in one real trace, AND (found only
       once a real, not random-noise, F0/asr fixture surfaced a ~40-sample/~17x-amplitude resonance
       burst end-to-end) real non-boundary `im_raw` can ALSO coincidentally land on exactly 0.0 for
       sufficiently periodic/structured content. Fixed by nudging `im` with a physically-negligible
       (~1e-20) positive epsilon UNIFORMLY, not just at the boundary-mask positions — closes both cases,
       confirmed via an isolated forward-STFT probe (6206/105622 boundary-only outliers → 3, the
       remainder being genuine, unavoidable float32 sign-crossing boundary sensitivity).
  - **`kokoro_driver_mil.lua` written**, wiring the two new MIL topologies together with the EXISTING
    bespoke LSTM-bound topologies (`text_encoder_cnn`/`text_encoder_lstm_*`, `duration_lstm_*`/
    `duration_adaln_*`/`top_lstm_*`/`duration_proj`, `f0n_shared_lstm_*`/`f0n_f0_block*`/`f0n_n_block*`/
    `f0n_f0_proj`/`f0n_n_proj`, unchanged from `kokoro_driver.lua` — LSTM-bound pieces stay bespoke, ggml
    has no native LSTM op). `decoder_vocoder` replaces FOUR bespoke calls (decoder_core/sinegen/
    stft_forward/generator) with ONE, taking `asr`/`F0_curve`/`N_curve`/`style`/`rand_ini`/`noise_in`/
    `wsum` directly and returning the finished waveform — no host-side `har` (STFT mag/phase) assembly
    needed anymore, just a `compute_wsum` Lua port of `export_kokoro_mil.py`'s own `compute_wsum_np`.
  - **`test_e2e_kokoro_mil_lua_driver.cpp`**: real end-to-end check, mirroring
    `test_e2e_vits_mil_lua_driver.cpp`'s own "no oracle comparison, per-phase references already give
    real confidence, just confirm the orchestration runs and produces a plausible result" strategy —
    same reasoning applies here even more directly, confirmed empirically: this test's own fixture (a
    synthetic sine-wave `ref_s`, not a real speaker embedding) triggers a genuine resonance burst in
    `loom::KokoroDriver` ITSELF (the trusted bespoke oracle, rms=1.09/max_abs=21.7 at sample 11656) that
    closely matches the MIL/Lua path's own (rms=0.88/max_abs=16.7 at sample 11651) — two fully
    independent implementations agreeing closely enough to confirm this is the real model's own
    out-of-distribution response, not a bug in either. 22207/22207 checks passed with plausibility bounds
    set from the oracle's own observed range.
  - Full regression suite re-run clean after every fix above (only the pre-existing, unrelated
    `test_e2e_lfm2_lua_driver` failure — a missing local fixture file, not a computation regression).
- **StyleTTS2 — DONE, numerically verified (2026-07-26)**, done out of the original stated order (ahead
  of Matcha-TTS/SupertonicTTS) on explicit user direction, precisely BECAUSE it could reuse Kokoro's own
  lessons so directly: `export_styletts2_mil.py` produces three MIL topologies into one combined
  `styletts2_mil.gguf` (`styletts2_driver_mil.lua` orchestrates them alongside the EXISTING bespoke
  LSTM-bound topologies from `convert_styletts2_reused.py` — DurationEncoder/predictor.lstm/duration_proj,
  F0Ntrain, TextEncoder's BiLSTM — unchanged, ggml has no native LSTM op, same scoping exclusion Kokoro's
  own MIL export already established):
  - **"albert"**: CustomAlbert alone (input_ids -> raw bert_dur), NOT fused with bert_encoder the way
    Kokoro's own combined "albert_bert_encoder" is — StyleTTS2's diffusion sampler needs the raw,
    unprojected bert_dur as its own conditioning input, a genuine data-flow difference from Kokoro's own
    pipeline. bert_encoder itself stays on the existing bespoke `kokoro_bert_encoder.gguf` topology (a
    single Linear, zero-risk to hand-build, nothing a trace would improve). Verified to ~1.3e-5
    mean/~1.1e-4 max abs against `reference_forward_styletts2_albert_mil.py` (which reuses
    `tools/convert_kokoro/reference_forward_kokoro_albert.py`'s own already-verified `albert_forward`
    unmodified — StyleTTS2's PL-BERT state dict uses the identical "module."-prefixed key convention).
  - **"decoder_vocoder"**: DIRECT reuse of `export_kokoro_mil.py`'s own `DecoderVocoderWrapper`/
    `build_decoder_vocoder_topology`/`VerifiedSTFT` (including every one of its trace-friendly
    AdainResBlk1d/SineGen/SourceModuleHnNSF/Generator/Decoder monkeypatches) — the real payoff of "using
    Kokoro's lessons": Kokoro's own istftnet.py classes ARE StyleTTS2's own (Kokoro is a fork of this
    exact architecture), so tracing them with StyleTTS2's own checkpoint weights needed zero new code,
    only a different state dict. Verified to ~7.5e-4 mean/~0.030 max abs against
    `reference_forward_styletts2_decoder_vocoder_mil.py` — but ONLY once that reference script was driven
    by a REAL forward pass through the rest of the pipeline (real CustomAlbert -> real style-diffusion
    sampler -> real predictor/F0Ntrain/TextEncoder) rather than arbitrary synthetic asr/F0_curve/N_curve/
    style values: even fairly small-magnitude synthetic noise (matching the SAME distribution Kokoro's own
    reference script safely uses) reliably drove this SPECIFIC checkpoint's Generator into its
    `torch.exp()`-based magnitude-reconstruction blow-up regime (`spec_logit` reaching ~27, i.e.
    `exp(27)~5e11`) for every random seed tried — a real, confirmed property of this trained checkpoint
    (bisected via per-stage std/max instrumentation: encode/decode/upsample stages all stayed bounded,
    std~1-4; the explosion is specifically `Generator.conv_post`'s raw output feeding `exp`), not a
    loading or exporter bug. Real, in-distribution values (a real style vector's natural ~0.13-0.32 std)
    keep the whole pipeline in its trained operating regime, matching Kokoro's own ~2e-3 mean/~0.025 max
    abs tolerance almost exactly (same architecture, same real HiFi-GAN-vocoder amplification ceiling —
    see that test's own comments for the full reasoning).
  - **"diffusion"**: StyleTTS2's own genuinely new piece (no Kokoro equivalent) — a real MIL trace of
    `Modules/diffusion/modules.py`'s `Transformer1d.run()` (embedding_scale=1.0 only, the real demo's own
    basic-synthesis default; the classifier-free-guidance branch traces the same network twice and is out
    of scope, matching `convert_styletts2_diffusion.py`'s own identical scoping decision), superseding that
    file's own hand-derived topology. Getting this to trace/build/compute correctly found and fixed THREE
    real, general bugs (all in `export_styletts2_mil.py`'s own monkeypatches except the last, which is a
    genuine general exporter fix — full `ctest` clean after, only the pre-existing unrelated
    `test_e2e_lfm2_lua_driver` missing-fixture failure):
    - `AttentionBase.forward`'s real `torch.einsum` calls hit a genuine coremltools bug (its generic
      einsum solver's diagonal-einsum pre-pass builds a `perm` sized for the wrong rank, `5 != 4`) —
      replaced with the algebraically identical batched-matmul formulation (`q @ k.transpose(-2,-1)` /
      `attn @ v`), sidestepping the einsum solver entirely.
    - `Transformer1d.run()`'s two `x.expand(-1, embedding.size(1), -1)` calls (broadcasting the noisy-style
      "pseudo-token" and the per-batch `mapping` vector out to the real dynamic token count) trace to a
      MIL `tile` whose own `reps` is a runtime shape query, not a compile-time constant. A general fix
      attempted directly in `exporter.py`'s shared tile-shape-inference heuristic got THIS case right but
      regressed CustomAlbert's own attention-mask head-broadcast (the identical "static-1 axis, unreadable
      reps" shape, needing the OPPOSITE resolution there) — reverted in favor of a narrower, single-model
      fix: replaced `.expand()` with a batched-matmul outer product (`ones_like(embedding[...,:1]) @ x`)
      instead, which MIL's already-well-tested matmul shape inference handles directly.
    - **General exporter bug, confirmed via an isolated minimal repro**: `ggml_mean` (the "MEAN" primitive,
      backing `reduce_mean`/`.mean()`) reduces `ne[0]` assuming a CONTIGUOUS source — fed a `PERMUTE`'s own
      output (a non-contiguous view) directly, as `x.mean(axis=1)` naturally produces once the reduced axis
      is transposed to `ne[0]` first, it silently reads with the WRONG stride and produces a
      plausible-looking but WRONG result (no assert, unlike `CONV_1D`'s im2col lowering, which crashes
      outright on the same "permute feeds an op needing contiguity" shape). Fixed generally in
      `exporter.py`: insert an explicit `CONT` node whenever `MEAN`'s input is produced by a `transpose`
      op — always safe (a `CONT` of an already-contiguous tensor is a harmless no-op), and confirmed
      dead-code for every other currently-exported model (`.mean(`/`reduce_mean` appears in exactly zero
      other `export_*.py` scripts, only in unrelated pure-PyTorch `reference_forward_*.py` ground-truth
      scripts that never go through the MIL exporter at all).
    Verified to ~5.4e-7 mean/~2.9e-6 max abs against the SAME `diff_*.bin` fixtures
    `test_e2e_styletts2_diffusion_net.cpp` already uses (`reference_forward_styletts2_diffusion.py`) — no
    `attn_mask` input needed this time (the real `Transformer1d` has no masking at all; the old bespoke
    topology only declared one as a loom `ATTENTION`-op API formality).
  - `test_e2e_styletts2_mil_lua_driver.cpp`: end-to-end orchestration sanity check (mirrors
    `test_e2e_kokoro_mil_lua_driver.cpp`'s own "no oracle, just confirm it runs and produces a plausible
    result" scope, for the same reason — decoder_vocoder's own precision ceiling plus diffusion's own small
    residual compound through the rest of the bespoke pipeline before ever producing a waveform, so a tight
    diff against the bespoke C++ oracle isn't a meaningful target). 22207/22207 checks passed
    (rms=0.0715, max_abs=0.391 — real speech-scale, bounded).
- **Matcha-TTS — DONE, numerically verified (2026-07-26)**. `export_matcha_mil.py` traces the REAL
  `matcha.models.components.{text_encoder,decoder}`/`matcha.hifigan.models` submodules directly (no
  `ConformerWrapper`/LSTM path involved at all — real config uses `down_block_type="transformer"`
  throughout, so unlike Kokoro/StyleTTS2 this model needed ZERO hybrid MIL/bespoke split) into one
  combined `matcha_mil.gguf` (`encoder_mu`/`encoder_logw`/`decoder`/`vocoder`, wired together by a new
  `tools/convert_matcha/matcha_driver_mil.lua`, mirroring `matcha_driver.lua`'s own Euler-CFM-sampling +
  duration-expansion control flow). Verified against the SAME real-module reference fixtures the bespoke
  conversion's own per-module tests already use (`reference_forward_matcha_{text_encoder,decoder,vocoder}
  .py` — the "eager wrapper vs. real module" simplifications made here are mathematically exact for this
  project's standing single-utterance convention, confirmed via a direct 0.0-diff eager-mode check before
  ever tracing anything): text encoder `mu`/`logw` ~1.2e-4/~6.3e-5 max abs diff, decoder (single Euler
  step) ~4.5e-4, vocoder ~4.2e-5. Full pipeline (MIL Lua driver vs. the existing bespoke `loom::MatchaDriver`
  oracle, `test_e2e_matcha_mil_lua_driver.cpp`) ~0.0104 max abs diff on a 10-Euler-step/10240-sample
  waveform — expected compounding of two independently-derived computation graphs through a nonlinear
  HiFi-GAN vocoder, same category as Kokoro's/StyleTTS2's own documented MIL-vs-bespoke residuals, not a
  bug (each phase already validates tight against real-module ground truth on its own).

  Two real trace-friendliness patches (module-local, not general exporter fixes): (1) `sequence_mask`/
  mask-tensor construction replaced throughout with either a direct arithmetic no-op
  (`x[:,:1,:]*0.0+1.0`) or — once THAT was also found to trip a real exporter bug (below) inside the
  Decoder's own multi-stage U-Net — no mask construction at all (`ResnetBlock1D`/`Block1D` reimplemented
  mask-free, since every mask multiply is an exact no-op under this project's single-utterance
  convention); (2) `RotaryPositionalEmbeddings`'s real mutable `cos_cached`/`sin_cached` state (a
  `torch.jit.trace`-hostile "already built for a long enough sequence, skip" fast path) replaced with an
  unconditional rebuild, PLUS deriving `seq_len` from the pre-rearrange tensor's own `t` axis (torch axis
  2) rather than the post-rearrange ("t b h d") tensor's axis 0 — see below for why axis 0 specifically
  was wrong.

  Five real, general exporter/engine gaps found and fixed getting here (full `ctest` clean after every
  one, only the pre-existing unrelated `test_e2e_lfm2_lua_driver` missing-fixture failure):
  - Two missing OP_MAP entries (`square`→SQR, `softplus`→SOFTPLUS — both already-existing ggml
    primitives, just never wired up; needed by `Block1D`'s Mish activation and `FeedForward`'s SnakeBeta).
  - **`reduce_mean` always reduced ne[0] only** (`ggml_mean`'s own hard limitation, silently wrong
    whenever the real axis isn't ne[0]) — first hit by Matcha's own hand-rolled `text_encoder.py::
    LayerNorm` (glow-tts-derived, NOT `nn.LayerNorm`), which reduces the CHANNEL axis (torch axis 1) on a
    (B,C,T) tensor, landing on ne[1] under this exporter's axis-reversal convention. Fixed with a
    dedicated single-axis `reduce_mean` translation (composed as REDUCE_SUM — already a real, axis-aware
    primitive — + SCALE by 1/N), a strict generalization of every previously-working reduce_mean usage
    (which all happened to reduce ne[0]), confirmed via an isolated standalone LayerNorm trace/compare
    (~3.6e-7 max abs diff) before ever touching the full model.
  - **`nn.GroupNorm` (`Block1D`) traces to a genuine two-axis `reduce_mean(axes=[2,3])`** (per-group
    channel count AND the dynamic time axis, jointly) — a real capability gap, not a translation bug: this
    exporter's per-axis reduction machinery (reduce_sum/the new reduce_mean above) only ever handles ONE
    axis, and here one of the two is genuinely dynamic (needing a runtime-computed divisor, not a static
    SCALE). Rather than build that generality, bridged to the ALREADY-independently-verified native
    `GROUP_NORM` ggml primitive (the exact one `convert_matcha_decoder.py`'s own bespoke topology already
    uses) via a new custom torch/MIL op (`tools/loom_mil_compiler/group_norm_op.py`, mirroring
    `vits_spline_op.py`'s own custom-op-bridge precedent) — `nn.GroupNorm.forward` patched globally to
    call it. Needed its own dynamic-dim-tracking fix too: `_infer_dynamic_dim_expr` had no case for this
    new `loom_group_norm` op type, so the backward walk used to derive a `conv_transpose` crop target (or
    any other downstream dynamic-length consumer) stopped dead at every GroupNorm — added alongside
    `layer_norm`/`instance_norm`'s own existing "shape-preserving over every axis" case (same formula,
    zero per-axis distinction needed).
  - **`conv_transpose`'s translation never inserted a CONT before a non-contiguous (PERMUTE'd) input** —
    `ggml_conv_transpose_1d` (like plain conv's im2col) requires a contiguous source and has no assert to
    catch a wrong stride, just aborts with `nb10 == sizeof(float)`. First hit by `Upsample1D`: the real
    `rearrange(x, "b t c -> b c t")` immediately preceding every real `ConvTranspose1d` call is exactly a
    `transpose` op. Fixed the same way MEAN's own identical danger was fixed for StyleTTS2 (insert an
    always-safe CONT whenever the input is transpose-produced).
  - **`RotaryPositionalEmbeddings`'s own `x.shape[0]` (read AFTER a `rearrange(x,"b h t d -> t b h d")`,
    so axis 0 now means sequence length, not batch) silently resolved to the literal constant `1`** via
    `_try_derive_gather_shape_value`'s existing "torch axis 0 of a rank≥2 tensor is always batch=1"
    shortcut — correct for every other model (where axis 0 genuinely never means anything else) but wrong
    here specifically because of the rearrange. Not a general exporter fix (the "axis 0 = batch" shortcut
    stays valid everywhere else) — worked around at the wrapper level by deriving `seq_len` from the
    ORIGINAL (pre-rearrange) tensor's own `t` axis (torch axis 2) instead, which hits the general
    (correct) dynamic-dim backward walk. Root-caused via a standalone isolated `MultiHeadAttention`
    trace/compare (found a real ~0.32 max abs diff, traced to a `RANGE_1D` node with `end` baked to the
    literal string `'1'` instead of `'n_tokens'` in the raw exported JSON) — the single most subtle bug
    of this whole conversion, silently correct-shaped but numerically wrong (no crash) since a length-1
    position-index range still produces a plausible-looking, just wrong, rotation for every token after
    the first.

  `tools/convert_matcha/matcha_driver_mil.lua`'s own layout differs from the bespoke `matcha_driver.lua`:
  the MIL-traced `encoder_mu`'s own `mu` output preserves the real module's native torch (1,n_feats,T)
  layout untouched (T-fast, matching the Decoder's/vocoder's own convention directly) rather than the
  bespoke topology's C-fast "rows_flat" one — deliberately NOT correcting this with a wrapper-level
  `.transpose()` (a bare transpose as a topology's own final output is a live non-contiguous GGML PERMUTE
  view, silently wrong once compiled — same danger already documented for VITS's own `StatsWrapper`).
  Net effect: no transpose needed bridging TextEncoder→Decoder here (simpler than the bespoke driver),
  but the per-token duration-expansion step is a direct nested-loop repeat in T-fast layout instead of
  reusing `loom.expand_by_duration` (which wants the opposite, C-fast convention).
- **SupertonicTTS — DONE, numerically verified (2026-07-26)**. `export_supertonic_mil.py` traces the REAL
  `supertonic_tts.models.modules.*` submodules directly (no hand-built bespoke topology involved) into one
  combined `supertonic_mil.gguf` (`dp`/`ttl_text`/`vfe`/`decoder`, wired together by a new
  `tools/convert_supertonic/supertonic_driver_mil.lua` mirroring the bespoke `supertonic_driver.lua`'s own
  control flow) — needed ZERO hand-reimplemented primitives (unlike the bespoke conversion's own
  `supertonic_common.py`, invaluable here purely as an independently-derived oracle for cross-checking
  every architectural quirk found while reading source, e.g. `StyleCrossAttention`'s `scale=sqrt(dim)` not
  `sqrt(head_dim)` quirk, confirmed identical before ever trusting the trace). Every `assets/pt/*.pt`
  checkpoint is a FULL pickled `nn.Module` (`torch.save(self, path)`), so `torch.load(...,
  weights_only=False)` hands back an already-built, real-weighted module directly — no hyperparameter
  reconstruction step at all (simpler than Matcha's own `hyper_parameters`-driven `TextEncoder`/`Decoder`
  rebuild). Verified against real-module reference fixtures (reusing the bespoke conversion's own
  `reference_forward_supertonic_{ttl_text,decoder}.py` fixtures directly, T=10 — new
  `reference_forward_supertonic_mil_extra.py` fixtures for `dp`/`vfe` only, needed because those bespoke
  fixtures use a different T than this export's own fixed `T_TEXT_FIXED=10`, see below): `dp`
  ~1.2e-7, `ttl_text` ~2.1e-6, `vfe` ~5.1e-6, `decoder` ~1.0e-6 max abs diff. Full pipeline (MIL Lua driver
  vs. the existing bespoke `loom::SupertonicDriver` oracle, `test_e2e_supertonic_mil_lua_driver.cpp`)
  ~6.4e-6 max abs diff on a 10-step/70656-sample waveform — far tighter than every other MIL-vs-bespoke
  full-pipeline residual in this project (Matcha ~0.01, Kokoro/StyleTTS2 ~2-3e-2), because this model's
  `SpeechDecoder` is a plain deterministic causal-conv stack (no ISTFT/GAN-upsampling nonlinear blowup
  regime to compound through) and its CFM sampler needs no explicit per-token duration-expansion step at
  all (the real `DurationPredictor` outputs one scalar total duration in seconds, not per-token durations
  — genuinely simpler control flow than every other TTS model in this project).

  Text-length scope limitation (REAL, carried forward from the bespoke conversion's own
  `SupertonicConfig::txt_len_fixed`, not newly introduced here, and not merely a `GraphBuilder`
  restriction): `T_TEXT` is fixed at trace/export time (`T_TEXT_FIXED=10`) for every topology touching
  text. Two INDEPENDENT reasons force this: (1) `vfe` needs TWO independently-sized sequences at once
  (the CFM-iterated latent-frame count `T_lat` AND the text length `T_TEXT`) — `GraphBuilder::build`
  resolves only ONE dynamic-length symbol per topology, so `T_lat` gets `$n_tokens` and `T_TEXT` must be
  static; (2) `dp`/`ttl_text` independently CAN'T be traced with a dynamic `T_TEXT` at all regardless of
  (1) — confirmed empirically: `MultiHeadRelativeAttention`'s relative-position-table windowing
  (`components.py`, Shaw et al., reused from VITS) pads by a length-DERIVED amount, which coremltools'
  own torch frontend explicitly refuses once that length is genuinely dynamic (`NotImplementedError:
  Dynamic padding for n-dimensional tensors is not supported`) — a real coremltools/MIL limitation, not a
  gap in this project's own exporter, and the SAME underlying reason the bespoke conversion fixed `T_TEXT`
  in the first place.

  Five real, general exporter/engine gaps found and fixed getting here (full `ctest` clean after every
  one — every genuinely `tools/loom_mil_compiler`-touching test across every prior model re-verified,
  the only failures observed were this session's own mis-typed `LOOM_*` env vars pointing at
  never-generated bespoke single-file fixtures, not real regressions):
  - **`nn.functional.pad(..., mode="replicate")` had no translation at all** — SupertonicTTS's
    `ConvNextBlock` (used by EVERY encoder/decoder in this model) pads this way before its depthwise conv,
    and ggml has no native replicate/edge-pad kernel (unlike `PAD_1D`/`PAD_1D_REFLECT`, which wrap real
    `ggml_pad_ext`/`ggml_pad_reflect_1d`). Composed purely from already-existing primitives instead of a
    new C++ op: `VIEW` out the single boundary column, `REPEAT`-broadcast it to the pad width (the same
    "materialize the broadcast, then `CONCAT`" idiom `REPEAT` was built for — StyleTTS2's diffusion
    sampler), `CONCAT` it onto the right side. Only the right-edge `VIEW` needs a dynamic offset (the left
    edge is always byte 0) — reuses the existing `_infer_dynamic_dim_expr` backward walk, no new
    shape-inference machinery. Verified standalone (both symmetric and causal-only padding) against real
    `F.pad`: 0.0 max abs diff, before ever touching the full model.
  - **`reduce_sum` only ever supported a single reduction axis** — `VFTextCrossAttention`'s fractional-RoPE
    sequence-length derivation (`mask.sum(dim=[1,2])` on a `(B,1,T)` mask) is a genuine 2-axis reduce, but
    axis 1 (the mask's own channel dim) is ALWAYS static size 1 — a provable no-op, unlike GroupNorm's own
    2-axis case (both axes genuinely contribute, one dynamically-sized, bridged to a dedicated
    `loom_group_norm` custom op instead). Fixed by dropping any reduced axis with static size 1 and
    falling through to the existing single-axis path — no new primitive, and GroupNorm's own bridge is
    untouched (a `>1` non-trivial axis still raises the same `NotImplementedError` it always did).
  - **`squeeze`'s rank-reducing case shared (and misused) `reshape`'s own "merge trailing MIL axes"
    formula** — that formula (`target_shape = [-1] + x_shape[merge_count:]`) is correct for a real
    multi-element-axis MERGE (e.g. a multi-head attention output's `heads*head_dim` flatten) but silently
    WRONG for `squeeze`, which only ever drops an already-size-1 axis, never folds two real-sized axes
    together. Confirmed on `SpeechPromptedCrossAttention`'s `torch.cat([o0,o1],dim=-1).squeeze(0)` (a
    `(1,1,T,256)` → `(1,T,256)` squeeze): the shared formula computed `target_shape=[-1,1,1]` (flattening
    everything into one 2560-element blob) instead of correctly dropping just the last ne-order axis,
    producing a real downstream `MUL_MAT` shape-mismatch crash. Fixed with a dedicated `squeeze` branch
    (reads the op's own `axes` input, or defaults to every static size-1 axis) that does an exact
    positional deletion from the input's own shape — no `-1` inference needed at all, since every
    squeezed axis is provably size 1 by squeeze's own contract. A real, general bug affecting every prior
    model's own `squeeze` usage too, not something newly introduced by this conversion — re-verified
    zero-regression across every other model's own MIL tests.
  - **`nn.BatchNorm1d` (eval mode) had no translation** — `SpeechDecoder.final_norm`. Folded to a plain
    per-channel scale+shift at CONVERSION time (`mean`/`variance`/`gamma`/`beta` are all real constant
    Vars once traced), same "fold at conversion time" precedent as weight-norm/Snake's reciprocal
    elsewhere in this project — not a new runtime primitive.
  - **`add`/`mul`'s generic translation assumed ggml's own single-sided broadcast always suffices** — true
    for every prior model, but `VFTextCrossAttention`'s fractional-RoPE angle computation
    (`theta[d] * frac_pos[pos]`, `ne=[32,1,1] * ne=[1,L,1] -> ne=[32,L,1]`) is a genuine "outer product":
    EACH operand is size-1 on a DIFFERENT axis than the other, so neither divides evenly into the other's
    shape and plain `ggml_mul` can't express it at all (confirmed: a real `MUL: incompatible shapes`
    crash). The bespoke conversion's own `supertonic_common.py` independently worked around the identical
    operation via a dedicated `MUL_MAT`-based outer product; fixed here at the general exporter level
    instead — detect when BOTH operands need broadcasting (on necessarily-different axes) and insert an
    explicit `REPEAT` for each up to the real output shape first, reusing the already-existing primitive
    rather than a new one. Only fires for this new "mutual, different-axis" pattern — every prior model's
    own single-sided broadcast case (the overwhelmingly common one) is untouched.

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
