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

> ### ✅ FIXED (2026-08-03): causal LMs were wrong whenever `n_tokens == n_head`
>
> Found while measuring something else — the fused-vs-unfused logit comparison KV-CACHE.md stage 2 asked
> for. **It predated the KV-cache thread entirely**: the pre-session `qwen3_0.6b_mil_monolithic.gguf`
> (2026-07-30) is bit-identical to a fresh unfused export (`max|Δ| = 0.000e+00` over the whole 151k-token
> logit vector) and failed the same way.
>
> **Symptom.** Qwen3-0.6B agreed with HF to `max|Δ| ≈ 2e-5` at every prompt length from 2 to 32 **except
> 8 and 16**, where it was off by 13.7 and 22.9 logits and the argmax changed. Length-dependent, not
> content-dependent — five different 8-token prompts all gave the wrong top-1, and `"A B C D E F G H"`
> predicted 425 instead of `" I"`.
>
> **Root cause.** `op_add`/`op_mul`/`op_mul_mat` (`src/ops/primitives_basic.cpp`) carry "dynamically heal
> transposed layouts" heuristics that infer an operand's intended layout **from its sizes**. Those are
> ambiguous the moment two axes are equal, and a transformer makes that happen for real:
>
> * RoPE multiplies `cos [head_dim, n_tokens, 1]` into `q [head_dim, n_tokens, n_head]`. `op_mul`'s
>   branch `a->ne[2] == b->ne[1] && b->ne[2] == 1` reads `n_head == n_tokens` — true only at the
>   collision — and permuted `cos` to `[head_dim, 1, n_head]`, turning **per-token rotation into
>   per-head rotation**.
> * `op_mul_mat`'s `a->ne[1] == b->ne[2] && a->ne[2] == b->ne[1]` does the same to attention's own
>   `q`/`k`, both `[head_dim, n_tokens, n_head]`.
>
> A GQA model therefore failed at **two** lengths (`n_head` and `n_head_kv`), because k/v carry the
> smaller head count. Confirmed by construction: tiny models with `(n_head_kv, n_head)` of (2,4) and
> (3,6) failed at exactly {2,4} and {3,6}.
>
> **Fix.** None of these heuristics may run when the operands are *already* compatible — at that point
> there is nothing to fix and a guess can only corrupt. Guarded on `ggml_can_repeat`/`can_mul_mat`
> respectively. This is the same reasoning the NOTE in `op_add` already recorded for a sibling branch
> that had been deleted for "re-corrupting already-correct tensors"; the remaining branches had the
> identical flaw. The standing principle is in that note: *the real fix belongs at the exporter, which
> knows the true layout instead of guessing from ambiguous shapes* — these guards make the guessing
> unreachable for correct graphs, but the heuristics themselves are still layout-guessing and should
> eventually go.
>
> **Verification.** Qwen3-0.6B now matches HF at every length 2–32 (`max|Δ| ≈ 2e-5`), fused and unfused;
> all three tiny models have no failing lengths; LFM2's HF-token gate 8/8; Whisper 13/13; `ctest` 142/142.
>
> **Why nothing caught it, and what changed.** `test_e2e_lfm2_mil_export` was the only numeric gate on
> this path and its prompts are 3 and 7 tokens — both lengths that happen to pass.
> `tests/test_broadcast_axis_collision.cpp` now sweeps `n_tokens` *across* the head count at the
> primitive level (no checkpoint, milliseconds), and was confirmed to fail with the guards reverted.
> `tools/debug/compare_logits.cpp` is the harness that found it: it drives `GraphBuilder` directly,
> because the driver's `infer` entry returns an argmax — the resolution at which this bug is invisible.

> ### 🟡 OPEN: who still depends on the layout-"healing" heuristics?
>
> Follow-up to the fixed bug above, and the reason it is not fully closed. The guards added in `6c170a1`
> make the size-guessing *unreachable for graphs that are already correct* — they do not remove the
> guessing. Any graph that genuinely fails the compatibility check still gets a layout inferred from
> ambiguous sizes, and **nobody currently knows which models that is**. The heuristics were added
> empirically, one model at a time, and `op_add`'s own NOTE records the principle they violate: *the real
> fix belongs at the exporter, which knows the true layout instead of guessing from ambiguous shapes.*
>
> **Remaining surface** (`src/ops/primitives_basic.cpp`):
>
> | site | state |
> |---|---|
> | `op_mul_mat` (2 branches) | guarded by `can_mul_mat` |
> | `op_add` (2 branches) | guarded by `ggml_can_repeat` |
> | `op_mul` (3 branches) | guarded by `ggml_can_repeat` |
> | `op_repeat` (2 branches) | **UNGUARDED** |
>
> `op_repeat`'s second branch is the same bug, still live:
>
> ```cpp
> if (a->ne[0] == ne[1] && a->ne[1] == ne[0]) {   // fires on ANY square target, correct or not
>     a = ggml_permute(pc.ctx, a, 1, 0, 2, 3);
> }
> ```
>
> It transposes an already-correct tensor whenever the repeat target happens to be square in its first
> two dims — exactly the `n_tokens == n_head` shape of the fixed bug, in a primitive no causal LM
> exercises heavily. It was left alone in `6c170a1` because the fix there was verified against causal
> LMs, and `op_repeat`'s documented consumer is StyleTTS2's diffusion `Transformer1d` (broadcasting a
> style vector `[channels]` to `[channels, T]`) — a path with no elementwise numeric gate to catch a
> regression. Guarding it needs its own verification, not a copy-paste.
>
> **The work, in order.** (1) Instrument each branch with a counter naming the op and the shapes, run all
> 11 models, and record which branches fire for which model — that converts "some model probably needs
> these" into a list. (2) For each real firing, fix the layout at the *exporter* so the operands arrive
> correct. (3) Delete the branch. A branch that fires for nothing can be deleted immediately, which is
> the cheap half and may well be most of them.
>
> **Why this matters beyond tidiness:** every one of these is a silent-wrong-answer generator with no
> error path, and the fixed bug shows the failure is invisible to argmax-level tests. Until step 1 is
> done, the honest statement about any of them is "we do not know whether it is load-bearing."

> **Read [`BACKEND.md`](BACKEND.md) first if you are touching the exporter.** It is the working record of
> the `EXPORT-IMPROVEMENT.md` thread (commits `42fc5d5`, `ebafa4e`, `640e49f` on `export-improvement`) and
> describes the shape the exporter now has, which is materially different from what older entries in this
> file assume:
>
> - `exporter.py` is 4,440 → ~2,120 lines. Per-op lowering moved to a **declarative rule table** in
>   `topology_ops.py`, keyed on `(mil_op_type, guard_predicate)`. `python3 -m loom_mil_compiler.topology_ops`
>   prints the whole table. An op that no rule claims falls through to the generic `OP_MAP` path — that
>   fall-through is a deliberate route for `less` and `reduce_mean`, not an accident.
> - "What is this Var's compile-time value / shape expression?" is answered in one place,
>   `value_facts.py`, memoized per Var. The memo is load-bearing, not tidiness: without it the shape walk
>   is exponential in encoder depth (see the Conformer entry below).
> - Three family templates now exist — `modular_export.py` (decoder-LLMs), `flow_matching_export.py`
>   (Euler-CFM samplers) and `nemo_asr_export.py` (NeMo ASR encoders) — and BACKEND.md's closing section
>   argues that per-family templates, not universal orchestration inference, are the direction that
>   actually works, with the evidence for why.
> - Symbolic shape expressions are **sympy objects**, not concatenated strings (`shape_expr.py`).
>   `render()` is the only thing that turns one into text, and it emits exactly `symbol_env.cpp`'s
>   grammar or raises. Compose with `as_expr`/`floor_div` and render at the emission site; do not build
>   a shape attribute with an f-string.
> - **Any exporter change should be gated on `tools/loom_mil_compiler/snapshot_gguf.py`** — snapshot the
>   exports before and after and require a zero-line `diff -r`. Its docstring has the recipe. The `.gguf`
>   files in the tree are `.gitignore`d build outputs and are routinely stale; regenerate the baseline
>   rather than diffing against them. When a change is *meant* to rewrite shape attributes, use
>   `tools/loom_mil_compiler/compare_snapshots.py` instead — it evaluates both sides of every differing
>   value at concrete lengths and reports anything not numerically equivalent as structural.

> **The next thread is [`EXPORT-ROADMAP.md`](EXPORT-ROADMAP.md).** Seven items (R1–R7) aimed at making the
> exporter look like `optimum-onnx` from the outside: named dynamic axes instead of one global `n_tokens`
> (R1), MIL→MIL lowering passes instead of emission-time guards (R2), a task/architecture registry with
> `LoomModelFor*` entry points (R3), transparent export with no user-written wrapping (R4), a grouping
> study of the ~120 models CrispASR covers plus a phased template plan (R5), the retirement policy for the
> bespoke `tools/convert_*` converters (R6), and the (approved) removal of `profile="atomic"` (R7). Items
> below that predate it are still valid; R2 and R7 in particular subsume several of them.
> **The implementation order is fixed — see the next section.**

> **[`EXPORT-PREPARATION.md`](EXPORT-PREPARATION.md) is what P4.0 executes.** It reviewed the question
> "could the exporter emulate `optimum-onnx`, so one entry point exports any causal LM / ASR / TTS
> model?" and found the export-side port already substantially done — the blocker is **who writes the
> Lua driver**: seven of eleven models get a hand-written `.lua` with marker substitution and none of
> `driver_ir`'s checks, while four get a synthesized, validated one. It proposes five items
> (**P4.0.4–P4.0.8**, listed under P4.0 below), records the decisions taken on 2026-08-01, and carries a
> five-stage implementation plan at commit granularity. It closes out R3's "driver templates as
> first-class artifacts" residue by specifying it and promoting it to the P4 critical path.

### Implementation sequence for the roadmap (start here)

The order below is not arbitrary; each phase either shrinks the surface the next one has to preserve, or
produces something the next one would otherwise have to guess. Two ordering constraints matter more than
the rest and are the reason this is not simply "R1, R2, R3…":

* **R1 (named axes) must land before R3's config schema.** `LoomExportConfig.inputs` *is* the axis
  declaration. Writing the schema first means migrating every config that exists by then.
* **The registry skeleton (P3) must land before any new family (P4).** Whisper and GigaAM written as
  scripts are two more scripts R4 has to delete; written as registry entries they are the acceptance
  test for the registry.

Everything is gated the same way as the last thread: `tools/loom_mil_compiler/snapshot_gguf.py` for
changes that must not alter output, `tools/loom_mil_compiler/compare_snapshots.py` for changes that
deliberately rewrite shape attributes, and the per-model reference tests for anything numerical.

| # | phase | items | gate | blocked by |
|---|---|---|---|---|
| **P0** | clear the ground — DONE | R7, writer dedup, R5 audit, R6 policy | golden diff (11 models) | — |
| **P1** | exporter internals — DONE | R1, R2a, R2b | `compare_snapshots.py` | P0 |
| **P2** | enable multi-output topologies — DONE | `GraphBuilder`/`run_subgraph` engine support, `generate_graph_topology` + `_prune_dead_nodes` generalization | existing single-output models byte-identical; new multi-output test topology exercised end-to-end | P1 |
| **P3** | the API skeleton — DONE | R3, R4 | byte-identical re-export of all current models | P2 |
| **P4** | flagship coverage — **all three flagships DONE** | P4.0 carry-over from P3 + the five `EXPORT-PREPARATION.md` items, Whisper (P4.1), GigaAM v3 (P4.2), composition template (P4.3) and its second leaf, Granite Speech (P4.3c). P4.0's own remainders are not all closed — P4.0.11(b), the `KvCache` memory redesign, is explicitly deferred | per-model reference tests | P3 |
| **P5** | breadth | families 12, 11, 4, 5, 9/10, 6, 13, 14 | per-model reference tests | P4 |
| **P6** | cleanup | R6 executions, docs | tests green with bespoke converters deleted | trails P4/P5 |

#### P0 — clear the ground (small, independent, no dependencies) — DONE

Do these first because everything later has to preserve whatever exists at the time, and each of these
*removes* something.

- **P0.1 — remove `profile="atomic"` (R7, approved) — DONE.** `apply_atomic_export` (was `exporter.py`
  ~1119–1417) is gone, along with the `if profile == "atomic":` branch and its monolithic fallback.
  Also removed: `export_lfm2_atomic.py`, the `LOOM_LFM2_ATOMIC_GGUF` case in
  `tests/test_e2e_lfm2_mil_export.cpp` + `tests/CMakeLists.txt`, `--profile atomic` from
  `tools/loom_mil_compiler/export_hf_causal_lm.py`'s CLI/docs, `tools/convert_lfm/export_profiles_demo.py`
  (its only purpose was demoing atomic-vs-monolithic; deleted rather than left demoing a profile that no
  longer does anything), and the atomic section of `LOOM_PROCEDURAL_GENERALIZATION.md` (replaced with a
  `ModularExportSpec` section describing the one split mechanism that remains). `_collect_replica_closure`
  was nested inside `apply_atomic_export` itself and went with it — confirmed `apply_modular_export`
  never called it. Scope-based partitioning as an opt-in `ModularExportSpec` discovery aid is still
  tracked further down this file, not implemented.
  **Gate — passed:** `diff -r` of all 11 remaining models' snapshots (`snapshot_gguf.py`) against a
  pre-removal baseline is empty; `test_e2e_lfm2_mil_export` passes end-to-end against real HF logits at
  two prompt lengths.
- **P0.2 — content-address weight payloads in the GGUF writer — DONE.** `exporter.py`'s `write_gguf`
  now hashes each tensor's FINAL on-disk shape+dtype+bytes (post quantization) and, when a later name's
  hash matches an earlier one, writes only a `loom.tensor_alias.names`/`loom.tensor_alias.targets` KV
  pair instead of a second copy; `GgufModel::load` (`src/core/gguf_model.cpp`) resolves both arrays
  straight into `symbols_` after the real tensors load, so `weight()`/`has_weight()` need no special
  casing anywhere else in the engine. Found empirically and worth keeping in mind for any future touch
  of this code: a pure byte+dtype hash is NOT enough — a rank-1 `[1]` scalar and a rank-3 `[1, 1, 1]`
  scalar holding the identical value hash identically on bytes alone but must not be merged (shape is
  now part of the hash), and two names with byte-identical raw weights but different quantization
  eligibility (`name in quantizable`, e.g. a tied embedding used as a MUL_MAT operand in one topology's
  slice but only via GET_ROWS in another's) must not be merged either (hashing the POST-quantization
  bytes, not the raw array, makes this automatic). Deduping turned out not to be LFM2-embedding-specific:
  every one of the 11 models had real duplicate constants (mostly small per-layer scalars, not just tied
  weights) — `lfm2_350m_modular` dropped from 1611 MiB to 1355 MiB (307 logical tensors, 158 real +
  149 aliased), and even the monolithic single-topology exports shrank (`lfm2_350m_monolithic` 257→158
  real tensors, `matcha_mil` 723→473, `kokoro_mil` 561→425, etc.).
  **Gate — passed:** for all 11 models, the alias-resolved LOGICAL tensor set (name → shape/dtype/sha256)
  is unchanged from the pre-dedup baseline (verified independently in Python, not just by C++ passing);
  `test_gguf_model_load` extended with a dedicated alias-only fixture case (a declared alias with NO
  `tensor_info` of its own, proving the C++ read path resolves it to the exact same `ggml_tensor*`, not
  just equal data); full `ctest`/`pytest` clean.
- **P0.3 — confirm the R5 family grouping — DONE.** See `EXPORT-ROADMAP.md`'s R5 section, now marked
  "confirmed, with corrections" — read all 120 `crispasr/models/convert-*.py` docstrings plus CrispASR's
  own README model tables (not just the architecture column). Four corrections to the original
  hypothesis: 12 of the 120 files aren't model converters (voice/reference bakers, non-GGUF format
  utilities, alternate write paths, one duplicate); family 2 ("Whisper-family") is really "audio encoder
  + AR cross-attention decoder" and over half its members use a Conformer encoder, not Whisper's; family
  3 is ~36 models, not ~20, the single largest group; and the 9/10 split conflated two pipeline STAGES
  (AR token LM vs. acoustic decoder) rather than two disjoint model sets, revealing a 4th acoustic-decoder
  shape (mel + HiFi-GAN TTS) filed under "one-offs" in the original table.
- **P0.4 — adopt the R6 policy — DONE.** Written into `BACKEND.md` as a standing section (the doc
  `BACKLOG.md` already directs exporter contributors to read first): a bespoke converter may be deleted
  only in the commit that re-points the last test consuming it.

#### P1 — exporter internals (must precede the API)

- **P1.1 — R1 named axes — DONE.** Axis vocabulary (`axes.py`: `N_SAMPLES`, `N_ENC_FRAMES`, `N_LATENT`,
  `N_CODES`, `BATCH` alongside `N_TOKENS`) + `GraphBuilder::build`/`loom.run_subgraph` replaced with a
  `DynamicAxes` (name → double) map, refactored across ~120 C++ call sites and all 11 hand-written `.lua`
  drivers + `exporter.py`'s `root_axis`/`declared_axes` replacing `symbol_overrides`. Conformer-CTC and
  both Parakeet variants renamed to `n_samples`; Kokoro's (and StyleTTS2's, reusing it) `decoder_vocoder`
  phase renamed to `n_enc_frames`. **Gate:** `compare_snapshots.py` extended with an alias map
  (`n_samples`/`n_enc_frames` → `n_tokens`) — golden-diff-clean across all 11 models, full `pytest`/
  `ctest` green.
- **P1.2 — R2a canonicalizing passes — DONE.** `normalize_matmul` (rewrites `transpose_x=True` into an
  explicit `transpose` + canonical matmul, closing the gap `topology_ops.py`'s rule table used to reject)
  and `insert_explicit_broadcasts` (a new `loom_broadcast_to` dialect op, lowered 1:1 to the same
  `REPEAT` primitive the exporter used to splice in ad hoc at emission time by comparing rendered shape
  strings). Both are real MIL→MIL passes in `passes.py`, alongside `fuse_gqa_repeat_kv`. No model on the
  current roadmap needs `transpose_x=True`, so `normalize_matmul` is a no-op everywhere today (verified
  via a dedicated `test_passes.py`, since no e2e reference model exercises it); `insert_explicit_broadcasts`
  fires on SupertonicTTS's fractional-RoPE angle computation and Matcha's encoder attention mask,
  producing byte-identical `REPEAT` nodes to the old ad hoc logic. Surfaced and fixed a real gap in the
  process: `LoomGGUFExporter.generate_graph_topology` is called directly (bypassing `export()`) by every
  small TTS model's own `_build_topology` helper (Kokoro/VITS/StyleTTS2/Supertonic/Matcha), which never
  ran `apply_loom_mil_passes` at all — fixed by moving that invocation into a new idempotent
  `_ensure_mil_passes_applied`, called from both `export()` and `generate_graph_topology` itself.
- **P1.3 — R2b `annotate_dynamic_shapes` — DONE.** `ValueFacts.annotate_dynamic_shapes` walks every op's
  output Vars in the program once, eagerly forcing `dim_expr` to resolve (and memoize) every dynamic
  axis up front, right after `apply_loom_mil_passes` — turning the existing per-Var memo from an
  incidental lazy cache into a real "resolve once, up front" pass, with no observable behavior change
  (`dim_expr` was already memoized; only the *timing* changed). Precondition for ever auditing the C++
  "heal transposed layouts" heuristics (tracked below) — that audit itself is not part of this item.
  **Gate (P1.2+P1.3 combined):** re-exported all 11 models, snapshot-diffed against a pristine pre-P1.2
  baseline — zero-byte diff everywhere; full `pytest` (121 tests) and `ctest` (139 tests, 0 failed) green,
  including real numeric reference verification for every rewrite site (Conformer-CTC/Parakeet TDT/RNNT,
  Kokoro decoder_vocoder, VITS, Matcha text_encoder/decoder/vocoder, Supertonic vfe/dp).

R2's remaining composite ops — DONE (all four landed together rather than interleaved with P4; they
turned out cheap enough, and the fresh dialect-op-plus-pass pattern from P1.2 made each one quick to
repeat):

- **`loom.replicate_pad`** — `pad(mode="replicate")`'s VIEW/REPEAT/CONCAT composition (SupertonicTTS's
  `ConvNextBlock`), moved into `passes.py`'s `canonicalize_replicate_pad` + a new `loom_replicate_pad`
  dialect op, lowered 1:1 by `topology_ops.py`.
- **`loom.conv_transpose_dw`** — the depthwise `conv_transpose` zero-stuff-then-conv composition
  (Kokoro's `AdainResBlk1d` upsample, reused by StyleTTS2), moved into `canonicalize_conv_transpose_dw` +
  `loom_conv_transpose_dw`.
- **`stack` lowering** — no new op needed: `lower_stack` rewrites `stack` into `expand_dims` + `concat`,
  both already-real MIL ops with their own full `topology_ops.py` rules, so `_op_stack`'s own ~50-line
  composition (a parallel copy of `concat`'s own N-ary chaining) was deleted outright.
- **`loom.mean`** — `reduce_mean`'s 3-way static/dynamic-ne0/unrepresentable split, moved into
  `lower_reduce_mean` + two ops: `loom_mean` (ggml's own run-time-counted ne[0] reduction) and
  `loom_scale` (the `reduce_sum`-then-divide composition). Caught and fixed two real bugs in the
  process: (1) `loom_scale` originally carried the pre-divided `1/n` as an `fp32` const — MIL casts
  every float const to fp32 on construction regardless of declared domain, silently rounding
  `1/192` (`0.005208333333333333` → `0.0052083334885537624`); fixed by carrying the integer `n` instead
  and dividing in plain Python at emission time, exactly like the old ad hoc code did. (2) the
  CONT-before-MEAN fix for a non-contiguous (transposed) input — previously keyed on `mapped_op ==
  "MEAN"` in the generic OP_MAP path, now unreachable since `reduce_mean` never survives to that path —
  had to move into `loom_mean`'s own `topology_ops.py` rule, or StyleTTS2's diffusion sampler would have
  silently regressed to the wrong-stride bug that fix originally closed.
  **Gate:** re-exported all 11 models, snapshot-diffed against the same pre-P1.2 baseline — byte-identical
  everywhere except SupertonicTTS's `vfe` (a pure `stack`-lowering intermediate-node rename, confirmed
  numerically equivalent via `compare_snapshots.py`); full `pytest` (134 tests, 21 new) and `ctest` (139
  tests, 0 failed) green, including real numeric reference verification for every rewrite site
  (SupertonicTTS `vfe`/`dp`, Kokoro/StyleTTS2 `decoder_vocoder`, Matcha `text_encoder`/`decoder`/
  `vocoder`, Conformer-CTC).

#### P2 — enable multi-output topologies — DONE

The engine's one-output-tensor-per-topology convention (`loom.run_subgraph` returns data + shape, see
`modular_export.py`'s `_flatten_call` comment) is a real, deliberate constraint everywhere it's been
hit so far, not an oversight — but it's the one thing standing between the current state and *inferring*
a `FlowMatchingSpec` directly from a scripted-loop trace instead of hand-declaring it (a MIL loop
body has one output per loop-carried var; see BACKEND.md item 3's follow-up, where two of the three real
prerequisites for that already hold). Scheduled here, before P3's config schema settles, so
`LoomExportConfig`'s iterative-refinement shape doesn't have to assume "always hand-declared" only to be
revisited once inference becomes possible. (The `while_loop`-inference *use* of this capability is still
deliberately not pursued — BACKEND.md's own finding was that its payoff is inferring the spec rather than
declaring it, not worth building yet. P2 only had to make the capability exist.)

- **P2.1 — multi-output support in `GraphBuilder`/`run_subgraph` — DONE.** `GraphTopology` gained a
  `std::vector<std::string> outputs` (JSON's plural `"outputs"` array; singular `"output"` still parses,
  wrapped into a one-element vector — `outputs.front()` always equals `output`). `GraphBuilder::BuildResult`
  gained `std::vector<ggml_tensor*> outputs` alongside the existing `output` field (`== outputs.front()`,
  unchanged for every pre-P2 caller). `build()` now `ggml_set_output()`s and `ggml_build_forward_expand()`s
  every declared output before the one `ggml_gallocr_alloc_graph()` call, mirroring the "mark every
  co-equal output before allocating once" pattern `test_primitive_registry.cpp`'s own hand-built
  multi-output test already documented as load-bearing (an output tensor without its own
  `ggml_set_output()` can have its buffer reclaimed by gallocr once nothing else reads it).
- **P2.2 — generalize `generate_graph_topology` and `_prune_dead_nodes` — DONE.** Both now take the full
  list of a function's declared outputs (`func.outputs`, not just `func.outputs[0]`); `_prune_dead_nodes`
  keeps everything reachable from *any* declared output. The emitted topology dict still writes singular
  `"output"` (byte-identical) for the one-output case and only switches to plural `"outputs"` when a
  function genuinely declares more than one.
- **P2.3 — driver-side plumbing — DONE.** `lua_bridge.cpp`'s `l_run_subgraph` returns every output's DATA
  (declared order) followed by every output's SHAPE (same order) — for one output that's exactly the
  `(data, shape)` pair the binding always returned, so no existing driver script's call site needed to
  change. `driver_ir.py`'s `check_subgraph_calls` was extended to validate a `SubgraphCall`'s `outputs`
  count against the target topology's declared output count, and that `extra_outputs` (shape captures)
  only appear once every data output has been captured first (partial capture then a shape would silently
  bind a shape-named local to the next output's DATA instead, since `run_subgraph` returns all data
  before any shape). `transpile_operation`'s existing "D. Submodule Dispatch" case (a nested-Function op
  binds one Lua local per `op.outputs`) already anticipated N-output calls; it needed no change.
  **Gate — passed:** two real models re-exported and `snapshot_gguf.py`-diffed against a pristine
  pre-P2 baseline (`git archive HEAD`) — zero-byte diff for both `lfm2_350m_modular.gguf` (the
  `apply_modular_export` path, ~20 topologies including the `aux` rotary-embedding submodule) and
  `supertonic_mil.gguf` (the `FlowMatchingSpec` template); full `pytest` (143 tests, 9 new) and
  `ctest` (140 tests, 1 pre-existing unrelated failure confirmed present on the unmodified baseline too)
  green. New multi-output coverage at every layer: `test_graph_topology_parse.cpp` (JSON parsing),
  `test_graph_builder_shapes.cpp` (a two-output build verified against two independent single-output
  oracle builds of the same sub-computations — **its numeric half was unsound as written and was fixed
  later; see the correction below**), `test_lua_bridge_run_subgraph.cpp` (the Lua-visible
  data-then-shapes return convention), `test_driver_ir.py` (`check_subgraph_calls`'s new validation, both
  accept and reject cases), `test_compiler.py` (a real coremltools-traced two-output submodule's topology
  correctly emits `"outputs"` and survives pruning).

  **Correction (found via a CI-only failure, after P4.0.3):** P2's own multi-output test compared
  freed memory. Its `run` lambda built each result from a `GraphTopology` and a `GraphBuilder` that
  were both locals, and returned the `BuildResult` — but `BuildResult::ctx` owns only the tensor
  STRUCTS, while the DATA lives in the `GraphBuilder`'s `gallocr` and the builder holds `topo_` by
  reference. So all three results dangled the moment `run` returned, and the comparisons read whatever
  the arena happened to still contain. It passed on every local run and failed on GitHub Actions, which
  is the whole signature of the bug. `MALLOC_PERTURB_=42` reproduces it exactly: the old test scores
  **51/63**, matching CI's count precisely, and the fixed one 63/63. Fixed by keeping both the
  topologies and the builders alive until every read is done, and `graph_builder.h`'s `ctx` comment —
  which said keeping `ctx` alive was enough, and is what invited the mistake — now states both
  ownership facts. Production code was never affected: every `src/core/*_driver.cpp` copies each output
  into its own storage inside the builder's scope. Two CI-reproducibility defects were fixed in the same
  commit, since they are what let this sit undetected and would let the next one do the same: LuaJIT was
  pinned to `v2.1`, a rolling BRANCH (now pinned to commit `faaf6633`), and CI's `pip install gguf
  numpy` was unpinned (now `gguf==0.17.1`, `numpy==2.4.6`) even though those two write the fixtures the
  C++ tests compare against.

  **Finding worth recording:** the empty diff on `lfm2_350m_modular.gguf` (at the time P2 itself landed)
  was not a coincidence — no model on the roadmap had ever actually produced a multi-output MIL
  `Function`. `modular_export.py`'s `_flatten_call`/`_replay` worked around the pre-P2 one-output
  limitation by concatenating a tuple-valued output (LFM2's rotary-embedding table's real `(cos, sin)`)
  into a single tensor on both the producing and consuming side, specifically *because* multi-output
  topologies didn't exist. P2 itself didn't touch that workaround, so there was no live bug for P2's own
  gate to catch.

  **P2.4 — retrofit `modular_export.py` off the concat workaround — DONE (follow-up).** Once P2 landed,
  the workaround became unnecessary rather than merely undesirable, so it was removed in the same
  session: `_flatten_call` now emits one real leaf per tuple element (named `k`/`k_1`/`k_2`/... —
  `apply_modular_export`'s own `is_aux_input`/idx-suffix convention already anticipated exactly this
  naming, unchanged) instead of concatenating, and `_replay` returns a module's own tuple output as-is
  (`torch.jit.trace`+`ct.convert` turn a tuple-of-tensors return into that many real MIL `Function`
  outputs directly — confirmed empirically before relying on it). `exporter.py`'s driver-synthesis side
  needed zero changes — the aux `SubgraphCall`'s `outputs=[f"_mod_aux_{i}" for i in
  range(len(aux_output_names))]` and each layer's positional `aux_out_vars[idx]` wiring were already
  written for N outputs; they'd just never had more than one to work with before.
  **Gate — passed:** re-exported `lfm2_350m_modular.gguf` — `model.graph_topology.aux` now declares
  `"outputs"` with 2 real entries (was a single concatenated `"output"`), and every attention-type
  `layer_i` (not the conv-type ones, which never touch `position_embeddings` at all) now declares two
  real inputs `position_embeddings`/`position_embeddings_1` (was one concatenated tensor). `LOOM_CHECK`'d
  against real HF top-1 predictions at two prompt lengths via `test_e2e_lfm2_mil_export`'s existing
  `run_gguf_case` oracle (extended to also exercise `LOOM_LFM2_MODULAR_GGUF`/`lfm2_350m_modular.gguf`
  alongside the monolithic fixture it already covered) — both prompts match HF exactly, same as the
  monolithic export. Full `pytest` (143/143) and `ctest` (140/140) green.

Not required to land before P3 for any *technical* reason (the API skeleton doesn't depend on multi-output
support existing) — ordered here because it changes what a family template's own spec needs to be able to
declare, and P4.3's composition template in particular is exactly the kind of model (encoder + adapter +
LM, each producing real intermediate outputs) worth checking against a multi-output topology once one
exists, before that template's own shape is locked in.

#### P3 — the API skeleton (R3 + R4)

- **P3.1 — `LoomExportConfig` base class — DONE.** Went further than "re-express the three templates
  one-for-one," per explicit user direction: harmonized them into a small `{Domain}{Function}ExportConfig`
  family hierarchy (`Domain` ∈ `Base`/`TTS`/`LM`/`ASR`, `Function` a structural role or a bare model
  name for a leaf) general enough to plausibly cover the CrispASR families in R5's table later, the way
  `optimum`'s `ORTModelFor*` names a family by task rather than by the first model that needed it. This
  phase built the root (`export_config.py`'s `LoomExportConfig`) and the **causal-LM family**
  (`causal_lm_export.py`): `LMCausalModelExportConfig` (abstract) with two concrete forms,
  `LMMonolithicCausalModelExportConfig` (one flattened trace — Qwen3's shape, and `export_hf_causal_lm.
  export_causal_lm()`'s existing body moved in verbatim as this class's `export()`) and
  `LMModularCausalModelExportConfig` (independently-traced submodules assembled per `ModularExportSpec`
  — LFM2's shape, `export_lfm2_modular.py`'s `main()` body generalized verbatim). `NeMoASREncoderSpec`
  (`nemo_asr_export.py`) renamed `ASRNemoEncoderExportConfig` to fit the convention and now inherits
  `LoomExportConfig` directly (no Monolithic/Modular split needed — Conformer-CTC/Parakeet-TDT/
  Parakeet-RNNT are parameterized instances, not subclasses). All three classes are `@dataclass(kw_only=
  True)` throughout the hierarchy (needed once a subclass adds required fields after an inherited
  defaulted one — e.g. `model_dir` after `profile`).

  **Scope call, confirmed with the user:** Qwen3 is registered as a real `LMMonolithicCausalModelExportConfig`
  user (`export_hf_causal_lm.py` is now a thin shim over it). LFM2 (`export_lfm2_modular.py`,
  `export_lfm2_monolithic.py`) is deliberately **not migrated this pass** — the scripts stay exactly as
  they are, regression-checked rather than replaced, since real LFM2 migration is a later pass (done in
  the "LFM2 migrated onto the causal-LM registry" entry further down, after P3.3). The
  regression check is `test_causal_lm_export.py`: it runs `export_lfm2_modular.py`'s own `main()`
  unmodified, builds an `LMModularCausalModelExportConfig` by hand with the identical
  `ModularExportSpec`/dummy shapes, and snapshot-diffs the two resulting GGUFs byte-for-byte — proof the
  new class genuinely reproduces the shape the script hand-rolls, not just that it looks plausible (same
  test does the equivalent check for `LMMonolithicCausalModelExportConfig` against `export_qwen3_mil.py`).
  `export_lfm2_monolithic.py` needed no such test since it already calls `export_causal_lm()`, which is
  now the same shim — re-running it directly exercises the new class for real.

  **Gate — passed:** all six affected models re-exported and snapshot-diffed
  (`tools/loom_mil_compiler/snapshot_gguf.py`) against a pre-P3.1 baseline — zero-byte diff for
  `qwen3_0.6b_mil_monolithic.gguf`, `conformer_ctc_small_mil_monolithic.gguf`, both Parakeet GGUFs, and
  both `lfm2_350m_modular.gguf`/`lfm2_350m_monolithic.gguf`. Full `pytest` (143/143, matching the P2
  count exactly since this phase added 2 new tests to `test_causal_lm_export.py` while touching nothing
  else) and `ctest` (140/140) green, including real end-to-end numeric verification via
  `test_e2e_lfm2_mil_export` (both profiles) and `test_e2e_parakeet_{tdt,rnnt}_mil_export` against their
  real HF/NeMo oracles.
- **P3.2 — `TaskRegistry` + loaders + `main_export()` + `loom-export` CLI — DONE.** Registry key is the
  **task**, not the model — a correction made during implementation, per explicit user direction: an
  earlier draft keyed the registry per model (`"qwen3"`, `"kokoro"`, ...), which conflates two axes
  `optimum` deliberately keeps separate. `task` names the export shape a `LoomExportConfig` family
  builds (`"causal-lm"`, `"nemo-asr-encoder"`, mirroring `optimum`'s own `"text-generation"`/
  `"automatic-speech-recognition"` vocabulary); *which* model a checkpoint actually is gets resolved
  separately, by a `ModelRecognizer` (real `detect()` structural check + `build_config()`) registered
  under that task. `tools/loom_mil_compiler/registry.py`'s `TaskRegistry`/`TaskRegistryEntry`/
  `ModelRecognizer` implement this; `main_export()` (`main_export.py`) resolves `(task, model)` from a
  path via `registry.detect()` when neither is given, `registry.get()` when both are — a lone `--model`
  without `--task` raises rather than guessing which family to look it up in.

  Two tasks registered this pass, matching P3.1's classes: `causal-lm` (`qwen3`, detected via HF-style
  `config.json`'s `model_type == "qwen3"`) and `nemo-asr-encoder` (`conformer-ctc`/`parakeet-tdt`/
  `parakeet-rnnt`, detected by opening the `.nemo` archive's `model_config.yaml` directly via `tarfile`
  + `yaml.safe_load` — no `ASRModel.restore_from`, no untar-to-tempdir, so detection alone is cheap).
  **Real finding, confirmed by reading all three checkpoints' own configs**: Parakeet-TDT and
  Parakeet-RNNT both restore through the identical `EncDecRNNTBPEModel` `target` (matching
  `nemo_asr_export.py`'s own earlier finding that the restore class never varies for this family), so
  `target` alone cannot tell them apart — the real secondary discriminator is
  `model_defaults.tdt_durations` (present only in TDT's config). Conformer-CTC's `target`
  (`EncDecCTCModelBPE`) is unambiguous on its own.

  `loom-export` (root-level bash launcher, `PYTHONPATH`-based rather than `cd`-based so a relative
  `-o`/model-path argument still resolves against the caller's own cwd) + `main_export.py`'s CLI
  (`--task`/`--model` overrides) are the `python3 -m tools.loom_mil_compiler.export_hf_causal_lm`-style
  entry point BACKLOG.md's own R3 example (`loom-export nvidia/parakeet-tdt-0.6b-v3 -o parakeet.gguf`)
  described. `export_conformer_ctc_mil.py`, both Parakeet scripts and `export_qwen3_mil.py` are deleted;
  `test_nemo_asr_export.py`'s own copy-paste-guard test (previously dynamically loading those three
  scripts) now builds each recognizer's config through the registry instead, against the same three real
  checkpoint paths. New `test_registry.py` covers the registry/recognizers directly with synthetic
  fixtures (a fake HF dir, fake `.nemo` archives with synthetic `model_config.yaml` content) — no real
  checkpoints needed for the detection logic itself.

  **Gate — passed:** `loom-export` (auto-detected, and again with explicit `--task`/`--model`) for all
  four models, snapshot-diffed against the same pre-P3 baseline — zero-byte diff in every case. Full
  `pytest` (161/161: 143 from P3.1 + 18 new in `test_registry.py`) and `ctest` (140/140) green, including
  `test_e2e_parakeet_{tdt,rnnt}_mil_export` run directly against the registry-produced GGUFs.
- **P3.3 — `ModelPatcher` + `BaseMultiPhaseModelExportConfig`/`TTSFlowMatchingModelExportConfig` — DONE.**
  Went beyond the literal four-script acceptance list, per explicit user direction: **Supertonic migrated
  too** (`export_supertonic_mil.py`, not in BACKLOG.md's original P3.3 list), grouped with Matcha under
  one shared `TTSFlowMatchingModelExportConfig` rather than left a "family of one" — the marginal cost
  was small once the shared base existed for Matcha anyway. **LFM2 stayed out of scope** (P3.1's own
  call, unaffected by this phase).

  `multi_phase_export.py`'s `BaseMultiPhaseModelExportConfig` replaces the near-identical
  `_build_topology`/weight-merge/`write_gguf` tail every TTS script hand-rolled: subclasses declare
  `phases()` (a list of `ExportPhase` — wrapper, dummy inputs, MIL input declarations, optional
  `root_axis`/`declared_axes`), and the shared `export()` traces each phase, merges weights with a
  content-aware dedup-on-match/hard-fail-on-mismatch check (generalizing `export_kokoro_mil.py`'s own
  two-phase merge to N phases — confirmed safe for VITS/Matcha's own fully-namespaced weights too, since
  a merge that can never find a real collision behaves identically to a plain dict union), and writes one
  GGUF. `TTSFlowMatchingModelExportConfig` (Matcha, Supertonic) adds `samplers()` — `FlowMatchingSpec`
  (renamed from `IterativeRefinementSpec`, `iterative_export.py` renamed `flow_matching_export.py`: what
  it declares is Euler integration of a vector field, i.e. flow matching specifically, not the vaguer
  "iterative refinement" — a name that would blur why StyleTTS2's real ADPM2 diffusion sampler is
  deliberately NOT part of this family). `BaseMultiPhaseModelExportConfig` also gained `estimators()` —
  plain `EstimatorSpec`s, validated against the real traced topology but generating no codegen — for
  StyleTTS2's own hand-written sampler.

  `kokoro_export.py`'s `build_decoder_vocoder_phase`/`build_albert_bert_encoder_phase` (renamed from
  `..._topology`, since they now return a deferred `ExportPhase` instead of eagerly tracing) stay
  module-level functions rather than methods specifically so `styletts2_export.py` can still call
  `build_decoder_vocoder_phase` directly — the real cross-model dependency found migrating this
  (StyleTTS2 and Kokoro share the identical iSTFTNet decoder/vocoder architecture; Kokoro's own trace-
  friendly monkeypatches apply to the same real `kokoro.istftnet` classes StyleTTS2 traces its own
  checkpoint through). `ModelPatcher.prepare_environment()` documents each family's own import-order
  stubs (Kokoro's numpy `_cast`/`transformers.utils.versions` patches, Matcha's
  `huggingface_hub.cached_download`/`matcha.utils` stand-in, StyleTTS2's `transformers.utils.versions`)
  as a named hook rather than unexplained top-of-file side effects — same timing as before, since the
  class-level monkeypatches these families also need (e.g. `vits_modules.WN.forward = ...`) still require
  the real class already imported, so those stay plain module-level code immediately after the family's
  own imports, not part of this hook.

  **Real mistake caught by the gate, not by review**: the first `matcha_export.py` draft silently
  dropped `from loom_mil_compiler import group_norm_op` (patches `nn.GroupNorm.forward` globally) — the
  export ran and produced a plausible-looking GGUF right up until `apply_loom_mil_passes` raised
  `NotImplementedError: reduce_mean op ... only a single reduction axis is supported`, deep inside a pass
  unrelated to GroupNorm on its face. Found immediately by actually re-running the export end-to-end
  rather than trusting the line-by-line transcription; fixed by restoring the import. A reminder that
  "moved the code, changed nothing" claims from a migration this size need a real re-run per model, not
  just a diff read.

  None of the five TTS checkpoints have a config.json`/`.nemo`-style self-describing manifest this pass's
  recognizers use for detection (`detect()` returns `False` unconditionally, requiring an explicit
  `--task tts-multi-phase|tts-flow-matching --model <name>`) — a real, stated scope limit (matches
  `optimum` itself needing `--task` for sufficiently custom architectures), not a silent gap. Worth
  revisiting per-model later: Kokoro ships its own `config.json`, Matcha's `.ckpt` has recognizable
  Lightning-style `hyper_parameters`/`state_dict` keys, and Supertonic's `.pt` files are distinctively
  fully pickled `nn.Module`s.

  **Registry design correction, made during this phase, not before it (see P3.2's own entry for the
  fuller reasoning)**: `TaskRegistry.register()` originally raised if a task was already registered —
  broken the moment a second TTS family (`vits_export.py`, then `kokoro_export.py`) tried to register
  under the same `"tts-multi-phase"` task `export_vits_mil.py`'s migration had just created. Fixed to
  create-or-extend: a task is created on first registration and extended by every later family that
  agrees on the same `config_class`, raising only if two families disagree about what a shared task name
  builds.

  **Gate — passed:** all five models (Kokoro, VITS, Matcha, Supertonic, StyleTTS2) re-exported through
  their new config classes and snapshot-diffed against a pre-P3.3 baseline (VITS/Kokoro against the
  pre-existing repo-root `.gguf`s; Supertonic against a freshly-generated baseline from the unmodified
  script in a throwaway `git worktree`, since no baseline existed yet) — zero-byte diff for
  `vits_mil.gguf` and `kokoro_mil.gguf`; Matcha/Supertonic/StyleTTS2 diff clean on every tensor and KV
  except the intentional, expected one (the embedded driver script's own comment naming the renamed
  `FlowMatchingSpec`/`flow_matching_export.py` in place of `IterativeRefinementSpec`/
  `iterative_export.py`). Full `pytest` (164/164, including a real registry-vs-direct-construction
  regression test replacing the "diff against `export_qwen3_mil.py`" check P3.2's deletion of that
  script made impossible to run as originally written) and `ctest` (140/140) green.

### LFM2 migrated onto the causal-LM registry (follow-up to P3.1/P3.2, per explicit user direction)

P3.1 deliberately left LFM2 unmigrated (`export_lfm2_modular.py`/`export_lfm2_monolithic.py` stayed as
the canonical path, only regression-checked). This follow-up migrates it for real: `causal_lm_export.py`
registers `lfm2-monolithic` and `lfm2-modular` under the `causal-lm` task, both using the exact same
`LMMonolithicCausalModelExportConfig`/`LMModularCausalModelExportConfig` classes P3.1 already built and
proved equivalent, with LFM2's own real parameters (`architecture="lfm2"`, `tokenizer_pre="llama3"`, and
the modular profile's real `ModularExportSpec`) hardcoded into their `_build_lfm2_*` factories the same
way `_build_qwen3` already hardcodes Qwen3's.

**Both recognizers detect the same way** (`model_type == "lfm2"` in the checkpoint's own `config.json`,
via a new shared `_hf_model_type()` helper `_is_qwen3` was refactored to use too) — genuinely, not a
bug: "monolithic" vs "modular" is a caller CHOICE about how to export the same checkpoint, not a
property of the checkpoint `detect()` could ever read off it. So `TaskRegistry.detect()` correctly finds
both matching and raises asking for `--model lfm2-monolithic`/`--model lfm2-modular` to disambiguate —
the same honest "can't guess, ask" behavior already established for Parakeet-TDT/-RNNT, not a new gap.

`test_causal_lm_export.py` rewritten: with both original scripts now deleted, all three tests in the
causal-LM family (Qwen3 monolithic, LFM2 monolithic, LFM2 modular) follow the same
"registry-built vs directly-constructed, snapshot-diffed" shape via one shared `_assert_registry_matches_direct`
helper, rather than the previous mix of "diff against a dynamically-loaded old script" (no longer
possible for any of them) and one-off duplication.

**Gate — passed:** `loom-export --task causal-lm --model lfm2-modular`/`--model lfm2-monolithic` against
the real LFM2-350M checkpoint, snapshot-diffed against the original P3.1 baseline — zero-byte diff for
both. Full `pytest` (165/165) and `ctest` (140/140) green.

### What P3 deliberately did not build (R3/R4 residue — still open)

P3 is DONE against its own stated gate ("byte-identical re-export of all current models" through the new
API, all 11 models). Three pieces of R3/R4's original description were not built, none of them blocking
P4, all of them still real:

- **`LoomModelFor*` runtime entry points** (R3's last table row: `LoomModelForCTC` /
  `ForSpeechSeq2Seq` / `ForCausalLM` / `ForTextToSpeech` over the Lua drivers + C++ backends). Nothing
  by that name exists. P3 built the *export* half of the `optimum` analogy (`OnnxConfig` →
  `LoomExportConfig`, `TasksManager` → `TaskRegistry`); the *inference* half (`ORTModelFor*`) has no
  counterpart yet. Arguably it should not be built until there is a second consumer besides the tests —
  a GGUF is already self-contained via its embedded driver, which is exactly why this half is less
  load-bearing here than in `optimum`.
- **Driver templates as first-class artifacts.** R3 asked to "generalize [the `--@loom:samplers` marker]
  into a proper template mechanism rather than string replacement." `flow_matching_export.render_driver`
  is still marker-based string replacement into a hand-written `.lua` file
  (`BaseMultiPhaseModelExportConfig.driver_script_path`), now called uniformly for every multi-phase
  family instead of per script. The uniform call site is the part P3 delivered; the template mechanism
  itself is untouched, and a family whose driver needs more than one substitution point will be the
  thing that forces it. **Now specified and promoted out of residue onto the critical path — P4.0.6
  below** (`EXPORT-PREPARATION.md` §1.4/§3): the template mechanism is `driver_ir`, which already
  exists and is under-used.
- **`.inputs` / `.generate_dummy_inputs()` / `.patch_model_for_export()` as named config members.**
  R3's piece table names all three; none exist under those names. What exists instead: axes are declared
  per phase (`ExportPhase.root_axis`/`declared_axes`) or per family field, dummy inputs are built inline
  inside `phases()`, and `ModelPatcher.prepare_environment()` covers only the import-order half of
  `patch_model_for_export()` (class-level monkeypatches stay module-level, deliberately — see P3.3).
  The ordering constraint at the top of this section ("R1 must land before R3's config schema, because
  `LoomExportConfig.inputs` *is* the axis declaration") therefore did not get cashed in: `LoomExportConfig`
  holds only `architecture`/`output_path`/`profile`, and axis declaration stayed where R1 put it.
  **Tracked as P4.0.2 below** — it has to be settled before P4.1/P4.3, whose configs are the first
  written from scratch rather than migrated.

Also stale: `export_config.py`'s module docstring points at a "Target class hierarchy and naming" section
of this file that does not exist (the hierarchy is described in P3.1's entry instead).

#### P4.0 — settle these before the first from-scratch family config

Sixteen items that P3 left in a state P4 would otherwise inherit and harden — three carried over from P3
(P4.0.1–P4.0.3, all DONE), five added by [`EXPORT-PREPARATION.md`](EXPORT-PREPARATION.md)
(P4.0.4–P4.0.8), three added by [`KV-CACHE.md`](KV-CACHE.md) (P4.0.9, scheduled **before** P4.0.7's
remaining registry steps at the author's direction, plus P4.0.10/P4.0.11, the two capability gaps stage 3
measured), three (P4.0.12–P4.0.14) from the review that followed P4.0.11a's marshalling fix, and one
(P4.0.15) that P4.0.13 discovered it could not finish without, plus P4.0.16, which reviewing P4.0.14's
memory cost turned up in the allocator underneath it. None is large; all get cheaper now and more expensive after Whisper/GigaAM/composition
add three more configs written against whatever shape exists at the time. Same gate as everything else:
byte-identical re-export of all 11 models (`snapshot_gguf.py`), since none of these is meant to change
any output — with stated exceptions: P4.0.6's per-family peeling commits, where driver text legitimately
changes and the gate becomes the model's e2e Lua-driver test plus a read diff, and the three KV items,
which add a *capability* and therefore change the topology of the models they touch by construction.

**Verification budget (decision, 2026-08-01):** affected models per commit — each step in
`EXPORT-PREPARATION.md` §6 names which models it can possibly touch — and a full 11-model sweep per
completed item.

- **P4.0.1 — real `detect()` for the self-describing TTS checkpoints — DONE.** P3.3 registered all five
  TTS families with `detect()` returning `False` unconditionally, so `loom-export <path> -o x.gguf` worked
  for the four causal-LM/NeMo models but needed `--task tts-multi-phase --model kokoro` for the five TTS
  ones. **All five now auto-detect** — this item predicted three (Kokoro, Matcha, Supertonic) and filed
  VITS/StyleTTS2 as explicit-only "unless a real discriminator turns up"; probing the real checkpoints
  turned one up for each, so both were implemented too.

  `checkpoint_probe.py` is the shared primitive: `read_json` (a safe sidecar-config read) and
  `probe_torch_checkpoint`, which opens a `torch.save` archive as a plain zip, reads only its `data.pkl`
  member, and walks it with `pickletools.genops` — returning the set of `module.Class` references and the
  set of strings the pickle contains. It **never unpickles**: no `torch.load`, no `weights_only=`
  question, no checkpoint code executed, and no tensor payload read (8–63 ms per real checkpoint). That
  is a hard requirement for detection specifically, which by construction runs against unidentified paths
  — `TaskRegistry.detect()` hands whatever the user typed to every registered recognizer in turn. The
  probe returns raw structure rather than a decoded object so each family's own discriminating claim
  stays in its family module, beside the `build_config` whose requirements it mirrors.

  Each recognizer checks what its own `build_config`/`phases()` will actually open, which is what keeps
  "detected" from ever meaning "detected, then failed to export":
  - **Kokoro** — a directory holding `kokoro-v1_0.pth` beside a `config.json` with Kokoro's key
    signature (`istftnet`/`plbert`/`n_token`/`style_dim`/`vocab`). Both halves needed: StyleTTS2 loads
    that *same* `config.json` (`TTSStyleTTS2ExportConfig.kokoro_config_path` — shared iSTFTNet
    architecture, shared declaration), so the config alone can't tell a Kokoro checkpoint directory from
    a StyleTTS2 export environment.
  - **Matcha** — a directory with `matcha_ljspeech.ckpt` + `generator_v1`, the ckpt carrying
    `pytorch-lightning_version`/`state_dict`/`mel_mean`.
  - **VITS** — a Lightning `.ckpt` *file* (not a directory, matching `checkpoint_path`) with
    `model_g.`-prefixed generator weights.
  - **StyleTTS2** — a `.pth` file with the `net` wrapper `export()` itself indexes through, plus
    `diffusion`.
  - **Supertonic** — the `assets/pt` directory with all four required `.pt` files, one of which names a
    `supertonic_tts.`-rooted class in its pickle (these are `torch.save(module)` outputs, not state
    dicts — the strongest signature of the five, and reading the class reference is not honoring it).

  **Two near-collisions found by probing, both of which killed the discriminator this item originally
  proposed:**
  1. *A Lightning signature is not Matcha-specific.* `pytorch-lightning_version` + `state_dict` is
     exactly what piper-VITS's own `.ckpt` declares too (Matcha 2.0.8, VITS 1.9.5). The two are separated
     by their state-dict key namespaces instead — Matcha's `mel_mean` (stored mel normalization stats,
     which VITS has no equivalent of) vs. VITS's `model_g.`/`model_d.` generator/discriminator split.
  2. *Kokoro and StyleTTS2 checkpoints are the same kind of object* — a dict of component name →
     `OrderedDict`, both leading with identical `bert` → `module.embeddings.word_embeddings.weight`
     ALBERT keys, no version marker, no config, no class reference beyond `collections.OrderedDict`
     (Kokoro is a StyleTTS2 derivative; this repo's own Kokoro/StyleTTS2 sharing of
     `build_decoder_vocoder_phase` is the same fact in code). Every component name in Kokoro's checkpoint
     is also in StyleTTS2's, so the discriminator had to run the other way — on what Kokoro's
     inference-only release *strips*: the `net` wrapper and the training-time components under it
     (`diffusion`, `mpd`, `msd`, `wd`). `diffusion` is the semantically right key to hold, since it is
     exactly why StyleTTS2 stays a plain `BaseMultiPhaseModelExportConfig` with a hand-written ADPM2
     sampler rather than a `TTSFlowMatchingModelExportConfig`.

  **Gate — passed:** every registered recognizer run against every real checkpoint on this machine (the
  five TTS ones, Qwen3, LFM2, Conformer-CTC) — each resolves to exactly one recognizer, except LFM2's
  two profiles, which is the intended documented ambiguity. Two decoys that must NOT match anything also
  don't: Kokoro's bare `kokoro-v1_0.pth` and Matcha's `generator_v1` (a raw non-zip pickle), neither of
  which is a valid `build_config` input. 17 new tests in `test_registry.py` (36 total in that file, full
  `pytest` 182/182 green), covering the probe directly (protocol-2 GLOBAL and protocol-4 STACK_GLOBAL,
  missing path / directory / non-zip / zip-without-`data.pkl` / truncated pickle) and every recognizer as
  a full 5×5 cross-product against synthetic fixtures — `pickle.dumps` of plain dicts inside a hand-built
  zip, since what the probe reads is the opcode stream, so no torch and no real checkpoints are needed.
  That includes `test_causal_lm_export.py`'s 3 real-export tests (Qwen3 + both LFM2 profiles, ~5 min,
  9.8GB of scratch), which pass unchanged.

  **Machine note, cost 20 minutes here:** those export tests write real multi-GB GGUFs into pytest's
  `tmp_path`, which defaults to `/tmp` — a 28GB partition on this machine that they fill to 0 bytes free,
  at which point unrelated commands start failing. Run the suite with both `TMPDIR=` and pytest's own
  `--basetemp=` pointed under `/home/flavio/.claude/tmp/`; `TMPDIR` alone does not move `tmp_path`.
- **P4.0.2 — where a family declares its dynamic axes — DONE. Decision: per-phase, not on the config;
  `.inputs`/`.outputs` struck from R3 (`EXPORT-ROADMAP.md`'s piece table now says so).**

  **Why not hoist.** `OnnxConfig.inputs` is a config-level property because an `OnnxConfig` describes
  exactly one graph. A `LoomExportConfig` frequently does not: 5 of the 11 models are multi-phase
  (Kokoro 2, VITS 3, Matcha 4, Supertonic 4, StyleTTS2 3), each phase with its own input signature, its
  own dynamic axes, and — for Kokoro — its own `root_axis`. A config-level `.inputs` covering those is
  necessarily `{phase: {input: {axis: name}}}`, i.e. `ExportPhase` with one more level of nesting and no
  more information. R3's row assumed one graph per model, which is true of `optimum`'s ONNX exports and
  false here for nearly half the models.

  **The "three unrelated places" this item was written about turned out to be one place plus an idiom.**
  Counting the real declarations rather than the call sites: `declared_axes` is used by exactly ONE
  phase in the whole tree (Kokoro's `decoder_vocoder`, 4 entries), and a non-default `root_axis` by
  exactly two places (that phase, and NeMo ASR's `n_samples`). Every other phase of every other model
  declares its dynamic axis by sharing one `ct.RangeDim` INSTANCE across the inputs that move together —
  coremltools then gives them one symbol, which the exporter maps to the default `n_tokens`. That idiom
  is load-bearing and was nowhere written down. A schema hoisted onto the config would have had 9 of 11
  models restating what the shared `RangeDim` already says.

  **What was actually wrong was that nothing checked it**, which is what got built instead — two silent
  failure modes, both now export-time errors in `LoomGGUFExporter`:
  1. *An undeclared second dynamic axis silently collapses onto the root.* `_sub_symbol` rewrites any
     MIL symbol it has no override for into `root_axis`, so two genuinely independent dynamic quantities
     both render as e.g. `n_enc_frames` and the emitted shape expressions are wrong — not malformed,
     just wrong, so neither `snapshot_gguf.py` nor a numeric reference test necessarily catches it.
     `_validate_input_axes` now raises naming every input and axis position in each group. Had Kokoro's
     `decoder_vocoder` phase been written without its `declared_axes` table, this is the error it would
     have gotten instead of four wrong shape attributes.
  2. *A declaration naming a static axis does nothing at all.* `_resolve_declared_axes` keys overrides
     on `str(input_var.shape[axis])`; for a static dim that's a literal like `"4000"`, a valid dict key
     no MIL symbol will ever match. Now raises, telling the caller to either make the axis a
     `ct.RangeDim` or drop the entry.

  **Known limit, stated rather than papered over:** the modular-blueprint Program has one Function per
  submodule and no `"main"`, so it gets no axis validation. That is defensible — `apply_modular_export`
  synthesizes its own leaf inputs and their axes rather than accepting them from a caller — and it is
  recorded in `_input_axis_symbols`' own docstring. The write-only exporter (`multi_phase_export.export()`
  constructs one with `program=None` purely to merge already-generated topologies) is skipped for the
  same structural reason.

  **Gate — passed:** 5 new tests in `test_compiler.py` (`TestInputAxisValidation`), covering the shared-
  symbol idiom, both raises, the declared-second-axis case that Kokoro really is, and the no-`main`
  skip. Kokoro/VITS/StyleTTS2/Matcha/Supertonic/Conformer-CTC exported through `loom-export` and
  compared by sha256 against the same six exported from a `git worktree` at the pre-P4.0.2 commit —
  **byte-identical, all six** (the check is pass-or-raise and touches no emission path, so this
  confirms rather than discovers). Qwen3 and both LFM2 profiles are covered by
  `test_causal_lm_export.py`'s own registry-vs-direct snapshot diff. Parakeet-TDT/-RNNT were not
  re-exported: they are the same `ASRNemoEncoderExportConfig` code path Conformer-CTC exercises, with
  the same `root_axis="n_samples"` and the same single dynamic axis. Full `pytest` 187/187 green.

  The six exports also confirm the validation has no false positives on the two shapes that matter:
  Kokoro's `decoder_vocoder` (five distinct dynamic symbols, four declared, one root) and NeMo's
  non-default `root_axis`.
- **P4.0.3 — monolithic/modular is an option again, not a class — DONE (see the next section).**

The five items below come from [`EXPORT-PREPARATION.md`](EXPORT-PREPARATION.md), which carries the
findings, the resolved decisions and the commit-level plan (§6, stages 0/A/B/C/D/E). Ordering rationale
in one line each: **A** first because it is the only stage that *removes* surface the others would have
to preserve; **B** before **C** because C's components are the first specs that would otherwise be
written against nothing; **D** after **C** because a registry of components with no shared calling
convention is a directory, not a shelf; **E** last because it is test work and bookkeeping that blocks
nothing.

- **P4.0.4 — task vocabulary and generic recognition — DONE.** The registered task names were `causal-lm`,
  `nemo-asr-encoder`, `tts-multi-phase`, `tts-flow-matching`: two name a decomposition and one names a
  loader library. Since P4.0.3 made decomposition its own field, `tts-multi-phase`/`tts-flow-matching`
  are one task whose members differ by a field. Rename to real tasks — `text-generation`,
  `automatic-speech-recognition`, `text-to-speech` — plus `audio-codec` **reserved** with no family
  against it until family 11 exists (decision 3), which is only meaningful if the vocabulary is a real
  checked list: hence `tasks.py`, declaring the canonical names and each task's base config class, with
  `TaskRegistry.register()` validating against it. No backwards-compatible aliases; the task name is a
  CLI argument, not a stored artifact. Second half: a **generic causal-LM recognizer** (any HF dir with
  a `model_type` and a `*ForCausalLM` architecture, `fallback=True`), so adding Llama stops meaning
  hand-writing `_is_llama` + `_build_llama` — the family is already model-agnostic underneath, and the
  only per-model data in `_build_qwen3`/`_build_lfm2_*` is an architecture string, an optional
  `tokenizer_pre` and a decomposition, all with working defaults. Requires `ModelRecognizer.fallback`
  and a specific-beats-fallback `detect`, or every Qwen3/LFM2 detection becomes ambiguous.

  **What was built,** four commits (`tasks.py` → rename → `fallback` → the generic recognizer):

  1. `tasks.py` — the four canonical names, each with what export shape it covers and the base config
     class it builds, resolved lazily by `module:QualName` string because every family module imports
     `registry`, which imports `tasks`. `register()` validates the name and checks `config_class` with
     `issubclass` against the declared base rather than identity against whichever class the first
     family to import happened to pass — which is what lets `TTSFlowMatchingModelExportConfig` (a
     *subclass* of `BaseMultiPhaseModelExportConfig`) share one task with the plain multi-phase families.
  2. The rename, with no aliases. **The check this step opened with**, per the plan, is that the task
     string reaches no GGUF KV — it holds two ways: `build_config(path, output_path)` is handed no task
     at all, and all 11 exported GGUFs contain zero occurrences of the four old names. So the gate stayed
     a pytest run rather than a snapshot diff. `--task` became an argparse `choices=` list, which then
     forced a real distinction: a name outside the vocabulary raises "unknown task", while a canonical
     but unclaimed one (`audio-codec`) raises "declared but no family is registered against it yet".
  3. `ModelRecognizer.fallback` + tiering in `detect`. Within a tier the rules are unchanged, so LFM2's
     deliberate two-way ambiguity survives and a fallback can never break a tie between two specifics.
  4. `hf-causal-lm`. Both halves of its guard are load-bearing: `model_type` alone claims every HF
     directory on disk, and Whisper, Parakeet and GigaAM all sit beside the causal LMs here — claiming
     any of them would break three other families, since `detect()` runs every recognizer against every
     path. `_MODEL_TYPE_OVERRIDES` is **empty, and that is a finding rather than an omission**: the
     exporter's tokenizer auto-detection resolves LFM2 to `llama3` and Qwen3 to `qwen2`, exactly what the
     two specific recognizers hardcode (now asserted, not argued).

  **Gate — passed.** All 11 models exported from a `git worktree` at the pre-stage commit and from the
  working tree, snapshotted and compared: **byte-identical, all 11**, `diff -r` over the two snapshot
  roots empty. Detection re-run against every real checkpoint on this machine in both trees; the diff is
  exactly two lines, both intended — LFM2's ambiguity now reported under `text-generation/` instead of
  `causal-lm/`, and SmolLM2-360M going from "no match" to `hf-causal-lm`. 212 pytest green (24 new).

  **Acceptance — two models that could not be exported before, both of which run.** SmolLM2-360M-Instruct
  and Llama-3.2-1B (`model_type: llama`, `LlamaForCausalLM`) each exported end to end through
  `loom-export` with no recognizer, no config and no flags, then **generated correct text through
  `loom_cli`** on the same prompt:

  | model | `tokenizer.ggml.pre` | `"The capital of France is"` → |
  |---|---|---|
  | SmolLM2-360M-Instruct | `starcoder` | `" Paris.\n\nParis is the capital"` |
  | Llama-3.2-1B | `llama3` | `" Paris. The capital of Germany is Berlin"` |

  Every inference the generic path makes landed: `loom.architecture` from `model_type`, and the
  pretokenizer from the tokenizer's own hash — **two different pre-types, neither of them the `qwen2`
  default**, which is what a hardcoded value would have gotten wrong. That the sampled text is right is
  a real end-to-end check of the whole chain (inferred architecture → traced topology → synthesized
  driver → engine → detokenization), not just of the export completing.

  *Still not claimed:* no numeric comparison against an HF forward pass at the logit level — that needs a
  reference fixture neither model has, and correct greedy text is weaker evidence than the ~0.003 max
  abs logit agreement the flagship models are held to.

  **One methodology note worth keeping, cost ~25 minutes here:** the first run of the 11-model sweep was
  vacuous. `loom-export` sets `PYTHONPATH` and runs `python3 -m tools.loom_mil_compiler.main_export`, but
  `python -m` puts the *current working directory* ahead of `PYTHONPATH` on `sys.path` — so invoking the
  baseline worktree's `loom-export` from the working tree silently imported the working tree's modules
  and compared new against new. It surfaced only because the baseline rejected `--task causal-lm` with
  the *new* argparse choices. **`cd` into the tree being measured; setting `PYTHONPATH` is not enough.**
- **P4.0.5 — the spec protocol — DONE.** Every spec in the tree earns its existence by being checked against
  the real model, and the checks are predicates over live objects (`EncoderOutput.validate`,
  `EstimatorSpec.validate_against_topology`, `ModularExportSpec`'s attribute paths, `_validate_input_axes`)
  — which is why a plain YAML/JSON front-end cannot be the foundation: it carries the field values but
  not the predicate, re-creating the declaration/validation split P4.0.3 spent a commit undoing.
  Resolution: the predicate does not have to live in a *per-spec* method. The four bespoke validators
  check the same handful of relationship kinds, so lift those into a shared vocabulary of `Link` kinds
  (`TopologyName`, `TopologyInput`, `TopologyOutputArity`, `ModuleAttrPath`, `Axis`, `ConfigDerived`,
  `WeightName`, `DriverSymbol`), have each spec declare `field → link kind`, and the checking becomes
  generic machinery while the model-specific content stays data. **The rule this establishes:** every
  spec field is either checkable against the real model/topology or explicitly documented as unchecked.
  **Acceptance test, stated up front:** all four existing validators re-expressible with *no loss of
  error-message quality* — this tree's errors name the offending input, the expected channel count and
  its config source, and degrading those to "validation failed" is a regression, not a refactor. A link
  whose context is never populated must be *reported*, not silently skipped, or "validated" quietly comes
  to mean "validated where convenient".

  **What was built,** six commits (`spec_protocol.py` → four retrofits → the enforcing test):

  1. `spec_protocol.py` — the eight link kinds, plus `WhenSet`/`EachOf` combinators and `FieldRef` for a
     link whose subject is a sibling field. **Three of the eight had no call site**
     (`TopologyOutputArity`, `WeightName`, `DriverSymbol`) and were unit-tested directly rather than
     through a family: they are the checks P4.0.6/P4.0.7 components need, since a spec that *generates*
     a `run_subgraph` call knows its arity before any driver text exists, while `driver_ir`'s own checks
     run on a finished `Function`. `TopologyOutputArity` got a real first user one commit later.
  2. **Message fidelity shaped the API rather than being checked after it.** `ConfigDerived` takes a
     `str.format` template with `{spec.<attr>}` access instead of formatting a canned sentence, which is
     what lets `EncoderOutput`'s three messages survive verbatim; `TopologyInput` reproduces
     `EstimatorSpec`'s bidirectional missing/unsupplied wording rather than reporting the first offender
     it finds. Every retrofit's tests assert whole strings, not `match=` fragments.
  3. **Deferral is the design detail that mattered.** Context arrives at different times — the model
     after `load_model`, topologies after tracing, weights after the merge, the driver last — so
     `LinkChecker` retries deferrals as `provide()` brings each slot and `finish()` raises listing
     whatever never became checkable. All three decompositions now own a checker and call `finish()`
     before writing. Without it a skipped check and a passing check are indistinguishable from outside.
  4. Four declaration kinds emerged, not one: a real `Link`, `Unchecked(reason)`, `CoveredBy(field)` and
     `NestedSpec(where)`. The last two are not bookkeeping. `CoveredBy` exists because
     `FlowMatchingSpec`'s `carried_input`/`time_input`/`fixed_inputs` only mean anything as the one
     argument table they compose into — three per-field links would report one offender at a time and
     lose the half of the message saying what is *missing*. `NestedSpec` deliberately does **not**
     auto-recurse: `EncoderOutput`'s links need the traced forward's return value, which exists for one
     instant inside the wrapper's `forward` and nowhere else, so `where` records that site in prose
     instead of pretending the outer checker covers it. Declarations also merge along the MRO, so
     `architecture`/`output_path`/`decomposition` are declared once on `LoomExportConfig`.

  **Four checks that did not exist before, all of them silent-wrong-output gaps rather than restatements:**

  * `FlowMatchingSpec` now requires a **single-output estimator**. `render_sampler` emits
    `local v = loom.run_subgraph(...)` and indexes `v[i]`; against a two-output topology `v` binds the
    first output's *data* and the loop integrates the wrong tensor — valid Lua, plausible shapes, wrong
    audio. Both real estimators are single-output, so this is a guard, not a fix.
  * **A typo'd axis name** is a perfectly good dict key: `_sub_symbol` substitutes it happily and the
    phase emits shape expressions over a symbol nothing else in the model uses. `declared_axes`
    expressions additionally go through `shape_expr.parse`, which is exactly `symbol_env.cpp`'s grammar,
    so a declaration that passes is one the engine can read back.
  * **`declared_axes` keys** must name inputs the phase declares — the same class of error
    `_resolve_declared_axes` raises, but before the trace instead of after it.
  * **`aux_kwarg`** must be a parameter of the repeated block's `forward`. Verified against the real
    LFM2-350m checkpoint, not only a fake.

  **The behavioural upgrade in `ModularExportSpec` is the timing, and the old failure mode is worth
  recording:** `get_by_path` raised a bare `AttributeError` from wherever its traversal reached, which
  for `suffix_attrs` was *after* the prefix and aux submodules had already been traced. A misspelled
  attribute cost a full trace to discover and reported only the missing attribute, not which declaration
  named it. `repeated_attr` is deliberately **not** a `ModuleAttrPath`: `find_repeated_blocks` re-derives
  the qualifying blocks independently, which is both the stronger property and what preserves the
  existing message listing what was discovered — a path check would also have accepted
  `model.embedding_norm`.

  **What `_validate_input_axes` kept, and why the split is not arbitrary.** `LoomGGUFExporter`'s two
  P4.0.2 raises stay where they are: whether two genuinely independent dynamic axes would collapse onto
  one symbol is only answerable once coremltools has assigned real MIL symbols, and no spec can see
  that. Only the half answerable from the declaration alone moved — and that half was not being asked at
  all. `ASRNemoEncoderExportConfig.root_axis` became a field for the same reason: `backend_kwargs()`
  returned the literal `"n_samples"`, so the family's R1 claim was a string in a method body rather than
  a declaration anything could check.

  **The standing rule is enforced by discovery, not by a list.** `test_spec_protocol.py` scans every
  dataclass in the package, so a new spec class in a family module fails until it declares and a new
  field on an existing one fails the same way (verified by adding a field to `FlowMatchingSpec` and
  watching it fail). Exemptions are five infrastructure *modules* and two classes, each mapping to prose
  rather than a bare name — "not a spec" and "nobody got around to it" are different statements. Three
  guards against passing vacuously: the eleven classes the scan must reach are named, the registry's own
  `config_class` entries are cross-checked, and an unimportable module fails, since any spec inside one
  escapes the scan silently.

  Closing the rule found the last seven undeclared classes, and the *reasons* are the deliverable. Each
  TTS config's path field is already established by the recognizer's `detect()`, which probes pickle
  opcodes without unpickling rather than trusting a filename — StyleTTS2's is the sharpest case, since a
  path link would happily accept the Kokoro checkpoint, which is exactly the near-collision `detect()`
  exists to resolve. `Modular.dummy_seq_len` is the one field where a link would be actively misleading:
  its correctness condition is a *non-collision* with the model's own static dims, and a wrong value does
  not fail — it marks a static axis dynamic and exports something plausible. The per-model reference test
  is the real guard, and saying so is worth more than a check that looks like one.

  **Gate — passed.** All 11 models exported from a `git worktree` at the pre-stage commit (`b9e110c`) and
  from the working tree, snapshotted and compared: **byte-identical, all 11**, `diff -r` over the two
  snapshot roots empty — including `model_driver_script.txt`, so the embedded Lua is compared
  character-for-character, which is the part `render_driver`'s rewrite could most easily have disturbed.
  284 pytest green, 67 new tests across the six commits, including whole-string assertions on every
  message the four retrofitted validators produce.

  **One gate the plan did not ask for, and it is the one worth keeping.** Byte-identity cannot show the
  new checks are *wired in*: a check that never runs also leaves output unchanged, which is exactly the
  failure mode `finish()` exists to prevent, applied one level up. So two declarations were deliberately
  broken and Matcha exported for real. Both failed the export with the link's own message —
  `FlowMatchingSpec('sample_decoder') does not match topology 'decoder': supplies input(s) it does not
  declare: ['z_wrong']; leaves declared input(s) unsupplied: ['z']; ...` and `... names topology
  'decodr', which is not among the exported topologies ['decoder', 'encoder_logw', 'encoder_mu',
  'vocoder'].` The same argument applies to every future retrofit stage: prove the check runs, not only
  that the output did not move.
- **P4.0.6 — `DriverBuilder` + `DriverComponent` over `driver_ir`.** The graph side has
  `Decomposition` (how the model becomes topologies); the driver side has nothing (how those topologies
  become a driver). `driver_ir.py` is already a real IR with `validate()` and `check_subgraph_calls()`,
  and its `RawBlock` is what makes migration incremental rather than big-bang: a family moves onto the
  builder by wrapping its current hand-written `.lua` in one raw block — immediately gaining
  `check_subgraph_calls()` on everything around it — then peels blocks into real components one at a
  time. Order: the two synthesized paths first (they already build `IRFunction`s, so the API is proven
  against working code), then Matcha → Supertonic → VITS → Kokoro → StyleTTS2. Per decision 2 the
  builder is **selected by the decomposition** (`Decomposition.driver_builder(config)`), not owned by
  the family, so the cross-attention AR decode shape can arrive as a fourth `Decomposition` bringing its
  own builder without reopening the component API.
  **Gate:** byte-identical driver text through the wrap-in-`RawBlock` step. It stops being achievable
  once a block is emitted from IR instead of pasted — comment placement, spacing, local naming all move
  — so each peeling commit's gate is instead: the model's existing `test_e2e_*_mil_lua_driver.cpp`
  passes unchanged, the driver-text diff is read and attached to the commit message, and every topology,
  weight and non-driver KV is byte-identical.

  **DONE**, eight commits (`driver_builder.py` → the two synthesized paths → adopt the five TTS drivers
  → peel them one at a time).

  **`DriverScript` is prelude lines + an entry function + postlude, not one `IRFunction`.** A real
  driver is a Lua *module*: a preamble, zero or more top-level helper functions, and the entry point the
  host resolves as a global. Modelling that as one function would have made every generated sampler a
  nested closure — a semantic change dressed as a refactor. Lines rather than chunks, joined by a single
  newline, so a component owns the blank lines around its own contribution; that is what made adopting
  an existing driver byte-exact, trailing newline included.

  **The order the checks run in is the content of `build()`**, not an implementation detail: check links
  → emit → `validate()` → `check_subgraph_calls()` → `provide(driver=…)`. The last step is why
  `DriverSymbol` was written in stage B with no call site, and it is what let
  `FlowMatchingSpec.func_name` stop being `Unchecked` — its own note had said "checkable as a
  DriverSymbol only once the driver is IR rather than text", and it now is for both driver shapes,
  because the link resolves against the built *script* rather than the entry function alone.

  **Wrapping a driver in a `RawBlock` would, on its own, have checked nothing** — and the plan's claim
  that the five TTS drivers "gain `check_subgraph_calls()` for the first time" would have been false,
  since that walks `SubgraphCall` nodes and raw text has none. The adoption therefore *parses* its own
  `loom.run_subgraph` call sites and declares each through the P4.0.5 protocol, which is also what gets
  `TopologyInput`'s bidirectional message for free. Coverage is printed per export in **two** numbers,
  because "checked" covers two amounts: a call passing a table literal has its full input set compared,
  one passing a prepared variable only has its topology name checked.

  **The gate found a real, undeclared property of two exports.** Kokoro's and StyleTTS2's drivers call
  topologies their MIL export does not produce — they are *partial* exports whose drivers run against a
  mix of MIL topologies and pre-MIL ones loaded from the bespoke `.gguf` alongside, which the C++ e2e
  tests do from two `GgufModel`s and nothing on the export side said. `external_topologies()` is that
  finding as a declaration, checked in both directions so it cannot rot: a name it lists that this
  export *does* produce is stale, one no call site references is dead. Declaring beats the alternative
  (skip any call naming an unexported topology), which would make a typo and a cross-GGUF dependency
  indistinguishable.

  **Where peeled Lua lives — `.lua` fragments, not Python strings** (author's decision). Each family is
  a directory of small fragments plus a component list that orders them and declares each one's
  `reads`/`defines`. The alternative puts the hand-written half of every TTS model behind a quoting
  layer, and the point of the exercise is to make these drivers easier to reason about. Section spacing
  lives in the fragments too, as data, rather than as a rule the builder would guess.

  **The peels are honestly uneven, and the boundary is the same one BACKEND.md already drew.** Matcha
  and Supertonic peel almost completely; Supertonic introduced *no new component class*, which is the
  reuse claim tested rather than asserted. VITS needed no sampler at all. Kokoro and StyleTTS2 are thin:
  of eleven and thirteen `run_subgraph` calls, two each become IR, while the rest name their topology
  with a computed expression, sit inside a Lua `for` loop, or — StyleTTS2's `diffusion` — inside a
  closure the ADPM2 sampler invokes twice per step. Forcing those into components would mean modelling
  Lua control flow in the IR. **`LuaFragment` parses its own call sites for exactly this reason: a peel
  must never *reduce* checking**, and without that, moving a block into a fragment would take its calls
  out of the parser's reach.

  **Two mistakes the new checks caught, both mine, both worth recording.** A `defines` list copied from
  Kokoro into StyleTTS2 claimed a local that family never binds — the export refused, naming the field
  and the fragment, before any tracing. And the first Matcha component list put the sampler at the top
  to match where its function appears in the output; that reads `t_mel` before the fragment binding it,
  and `validate()` rejected it. A component's prelude is collected separately from its statements, so a
  sampler belongs at its *call* site and its function still comes out on top.

  **Gate — passed.** C.2 and C.3: byte-identical for all eleven, `diff -r` empty including
  `model_driver_script.txt`. C.4–C.8: each family's `test_e2e_*_mil_lua_driver` passing with
  numerically identical output (Matcha max_abs_diff=0.0104421 unchanged; VITS/Kokoro/StyleTTS2 per-sample
  against their bespoke oracles, 49671/22207/22207 checks), every topology, weight and non-driver KV
  byte-identical, and the driver-text diff read and attached to each commit — which is how the
  Layout A/B slip in one rewritten comment was found, since no test covers a comment. Negative gates on
  both builders and all five families, each failing a real export with the link's own message.
- **P4.0.7 — the component registry ("marketplace") — DONE (7 commits).** Six components exist (`FlowMatchingSpec`,
  `EstimatorSpec`, `ModularExportSpec`, the prefill prologue/epilogue, `recurrent.py`'s stepping loop,
  `ExportPhase`) and are assembled four different ways — marker substitution, direct-to-IR, inline, ad
  hoc. That heterogeneity, not any missing capability, is what makes adding a family feel bespoke.
  Extract all six onto the one `DriverComponent` calling convention and register them by name; nothing
  new is written. The deliverable is as much the **catalogue** — per component, its links, what it emits,
  which models use it — as the code, since that is what lets P4.1/P4.3 reuse rather than restate.
  **Gate:** all 11 re-exported byte-identically through registered components.

  **The first three commits were none of them the registry itself.** The author's review of stage C is
  what redirected this item, and the critique was correct on both counts: peeling into `.lua` fragments
  named the blocks but left them heterogeneous, and the export had no business emitting two GGUFs per
  model. Measured before acting: **11 functions totalling 112 lines were shipped byte-identical in
  Kokoro's and StyleTTS2's fragments**, with their own comments saying so ("identical to
  kokoro_driver.lua's own"). The duplication was documented rather than removed.

  1. **`loom_lua` — the driver-side standard library.** Twenty atomic Lua functions in
     `tools/loom_mil_compiler/lua/`, one per file, with each family declaring what it uses and the
     builder emitting only the transitive closure (so Matcha's GGUF does not carry StyleTTS2's ADPM2
     sampler). The 11 duplicates now have one definition each; six array primitives the inventory turned
     up as repeated inline loops (`array_sum` ×4 families, `array_slice`, `array_affine`,
     `durations_from_logw`, `pad_last_to_multiple`, `repeat_by_duration_tfast`) joined them. Dependency
     declarations are checked **both** ways, and that check paid for itself on its first run: one real
     missing dependency (`predict_durations` calls `sigmoid`) and one false positive from matching
     comment prose, since `round_half_to_even`'s docstring *names* `predict_durations`, which calls it —
     believing the comment would have inverted the dependency.

     The boundary this found, rather than assumed: VITS's frame expansion fuses Gaussian
     reparameterisation into its repeat loop, and Kokoro's/StyleTTS2's duration-encoder loops interleave
     a subgraph call with per-timestep row surgery. Generalising either means a callback per inner
     statement — worse to read than the loop, which is the same argument `flow_matching_export.py` makes
     about ADPM2. **The rule: a library function names one operation; a family's own control flow stays
     in the family.**

  2. **`RecurrentPhase` — `recurrent.py` finally wired in.** `build_lstm_cell_topologies` had been
     verified against a real bidirectional `nn.LSTM` to 1e-4 since it was written and had **no caller**;
     `generate_graph_topology` raised on an `lstm` op and named it as the fix. The maths was never the
     missing half. It emits `{name}_h_fwd`/`_c_fwd`/`_h_bwd`/`_c_bwd` — exactly what `loom_lua`'s
     `run_bi_lstm` composes — so no driver changed.

  3. **Kokoro and StyleTTS2 are now self-contained**: 39 and 41 topologies in one GGUF each, and
     `external_topologies()` returns `{}` for both. **This is the item that should not have needed
     doing.** The one-GGUF-per-model convention already existed and the *bespoke* converters already met
     it — `convert_kokoro_lua_all.py` produces a single 43-topology `kokoro.gguf`. The MIL export was the
     regression, and stage C's `external_topologies()` documented it rather than fixing it. 21 phases per
     family from one shared builder (`build_prosody_phases`), reused between them for the same reason
     `build_decoder_vocoder_phase` already was: Kokoro is a StyleTTS2 derivative, so these are the same
     classes with different weights.

  **The gate this needed and did not have.** `test_e2e_*_mil_lua_driver` has no oracle waveform by
  design; its per-sample checks are `isfinite` plus an rms range, so a wrong topology producing a finite,
  plausibly-loud waveform passes it. `test_e2e_kokoro_mil_topology_equivalence` compares each transferred
  topology against **the thing it replaces** — same random inputs into both files' versions — with the
  list *derived* from the intersection, so a phase is covered the moment it is exported. It covers both
  families: **75 topologies, 234 checks**; the 24 LSTM cells at ~1e-7, `duration_proj`/`adaln`/`proj1x1`/
  `bert_encoder` at exactly 0. Two names are excluded by an explicit list rather than by "skip any
  difference" — StyleTTS2's `albert` and `diffusion` deliberately redefined their interface and the
  driver was rewritten to match; every other declared-input mismatch is a real finding.

  **Two general exporter bugs, both found by that gate, neither family-specific:**

  * `_infer_dynamic_dim_expr` had no `gather` case, so the walk gave up at an **embedding lookup** — the
    most ordinary way to start a topology and simply never hit before. The dynamic axis fell out of a
    downstream RESHAPE as a literal and the topology failed to build. Same shape as the
    `leaky_relu`/`conv_transpose` gaps `vits_export.py` already records.
  * **A declared output produced by `PERMUTE` was left as a live view.** `ggml_backend_tensor_get` does a
    raw contiguous byte copy, so reading one back returns pre-transpose data; torch's `.contiguous()`
    cannot prevent it, because MIL has no notion of contiguity and drops the call. The hazard was known
    and until now only ever *avoided* — `matcha_export.py`'s docstring and `vits_export.StatsWrapper`
    both record deliberately not returning a transposed output, and every hand-built converter writes
    `PERMUTE + CONT` by hand. StyleTTS2's `bert_encoder` forced the fix because its driver *requires* the
    transposed layout. Caught as mean_abs_diff=0.717 against a reference reaching 2.23 — what a transpose
    looks like when nothing crashes.

  **Sweep — the general fixes are no-ops where they do not apply.** All eleven exported from a worktree
  at the previous commit and from the working tree: the nine that need neither fix are byte-identical in
  topologies, weights, non-driver KV *and* driver text; Kokoro and StyleTTS2 differ by exactly the
  topologies they gained. 141/141 ctest, 373 pytest.

  **What remained for P4.0.7 proper — DONE (D.1–D.4, four commits).** The registry, the computed-name
  declarations, and the catalogue. The generalisable lesson from the three commits above held for all
  three: *a name is not a mechanism*, so nothing below is a table someone maintains — every entry is
  checked from both sides and every rendered document is generated from the declarations it describes.

  1. **D.1 — `component_registry.py`, the shelf.** Ten components, each with what it emits, and three
     checks that make the entry load-bearing rather than descriptive: a shipped `DriverComponent`
     subclass with no entry **fails the export** (`DriverBuilder.build` looks each one up as it emits;
     `unregistered_component_classes()` asks the same statically, by discovery over the package); an
     entry whose `emits` is narrower than what the component really contributed fails the export, since
     the catalogue's emission column is generated from it; and an entry no model uses must carry the
     reason it is still registered — `raw_lua_driver` is the only one, and it is the adoption step's
     component, which every TTS family passed through and none is on now.

     `usage()` derives which models use what: the TTS half by building each registered family's real
     `driver_components()` (no checkpoint needed — a peeled family's list is paths and IR expressions),
     the synthesized half off the two builders' own dataclass fields. What keeps that non-circular is
     one line in the exporter: `apply_monolithic_export`/`apply_modular_export` construct through
     `driver_components.SYNTHESIZED_BUILDERS`, the same table the attribution reads.

     **A module, not the `driver_components/` directory the plan wrote.** The `/` was shorthand for the
     shelf. A package would additionally have weakened the standing rule: `test_spec_protocol`'s scan
     walks `pkgutil.iter_modules(package.__path__)`, so a dataclass in `driver_components/foo.py` is
     reached by neither the scan nor its unimportable-module report — a real check traded for a
     cosmetic one.

  2. **D.2 — the computed-name call sites, declared as data.** This is the gap the paragraph above
     carried in, and it was bigger than "those call sites cannot be link-checked" suggests: **2
     computed `loom.run_subgraph` sites and 16 helper call sites**, all in Kokoro and StyleTTS2, driving
     35 of each family's ~40 topologies. The helper sites were not merely unresolved — they are inside
     the `loom_lua` function, a level below the fragment, so no fragment parse could ever have seen
     them.

     The declaration splits along what each side knows. `lua_library.DrivenTopologies` declares the
     *shape* a library function's body hard-codes (the four BiLSTM cell suffixes, the three block
     suffixes, and the input table each call supplies); the family declares the *namespaces*, which
     exist only at run time. `HelperCall`/`ComputedCall` expand the two into ordinary
     `RunSubgraphCall`s, so these sites now fail with the same `TopologyName`/`TopologyInput` messages
     a mistyped literal always has. **After D.2 there is no second class of call site with weaker
     checking.**

     Checked in both directions, three ways: a call site no declaration covers fails the export (the
     completeness half — without it, declaring nine of ten sites would read as coverage); a declaration
     whose call site the Lua no longer contains fails the export; and `drives_mismatches()` compares
     each library declaration against the body that hard-codes it, suffixes and input table alike,
     including the case that would bring the gap back — a function that calls `loom.run_subgraph` while
     declaring no `drives`.

     Peeled drivers now print their coverage, and report what is left over rather than only what is
     covered: Kokoro 2 as IR / 2 parsed literal / 9 computed sites → 35 topologies, StyleTTS2 2 / 4 / 9
     → 35, the other three 3 as IR and nothing computed. For all five, **every exported topology is
     named by a call site** — reported, not enforced, since P4.1's Whisper encoder may legitimately be
     called by the host rather than the driver.

  3. **D.3 — [`DRIVER-COMPONENTS.md`](DRIVER-COMPONENTS.md), generated.** Per component: what it emits,
     what it declares, what each declaration *says* when it fails, and which models use it — rewritten
     in place by `python -m loom_mil_compiler.component_registry`, with a test that regenerates and
     compares. Two renderings had to be got right or the document would misreport its own subject: a
     declaration-only field (`ModularChain.stages`, `FlowMatchingSampler.spec`) is not an unchecked one
     and now renders with the `NestedSpec`'s own prose, and `declared_links` silently returned nothing
     when handed a class (`type(cls)` is its metaclass) — `declared_links_for` is the class-side entry
     point. §4 carries all five negative-gate probe messages verbatim from real failing exports, which
     is what the plan meant by taking them from the probe rather than from the source.

  **Gate — passed.** All 11 models exported from a worktree at `32b2271` and from the working tree,
  snapshotted and `diff -r`'d: **byte-identical, all 11**, `model_driver_script.txt` included — every
  one of these commits adds checks and declarations, and none of them emits Lua. Five negative-gate
  probes, each breaking one declaration and failing a real export with that check's own message (two in
  D.1, three in D.2), recorded in their commits and in the catalogue's §4. A sixth check fired unasked
  during the first probe — one class registered under two names — which is `registry()`'s duplicate
  guard.
- **P4.0.8 — legacy C++ driver retirement policy — DONE (8 commits, stage E).** R6's policy covers `tools/convert_*` only; extend it
  to `src/core/{kokoro,vits,matcha,styletts2,supertonic,whisper}_driver.cpp`, which predate the Lua
  drivers becoming the orchestration device. Same rule — a driver may be deleted only in the commit that
  re-points the last test consuming it — plus **the precondition that is not obvious**: the pre-MIL C++
  oracle tests are the *numeric ground truth* several MIL/Lua tests were validated against, so each Lua
  test must first carry its own reference fixture. That is the real cost, and the actual reason all six
  are still alive. `include/loom/loom.h` re-exports all six from the umbrella public header (lines
  14–24), which is why every test transitively depends on them and a naive grep reports no consumers —
  split it into the lean runtime surface and a `loom_legacy.h` so the boundary is auditable. VITS,
  Matcha, Supertonic, Kokoro and StyleTTS2 are retirable now; **Whisper is not** — `whisper_driver.cpp`
  has no MIL export to replace it and is blocked on P4.1. Per decision 1, `expand_by_duration` and
  `pad_crop_relative_embeddings` **stay** in the bridge, reclassified as generic host-side tensor ops
  (neither reads a model config; both exist because the operation has a data-dependent output length,
  which cannot live in a static topology) — so that bullet is documentation: `lua_bridge.h` gains the
  criterion a new binding must meet, with both labelled against it.
  **Gate:** full `ctest` green with five drivers deleted, and the engine binary size recorded before and
  after — leanness is the stated goal of the architecture, and measuring it is how the goal stops being
  a slogan. *Trails the others; nothing in P4 depends on it.*

  **Done (E.1–E.4, eight commits).** `lua_bridge.h` carries the binding criterion — *a generic
  host-side tensor op, not model adaptation* — with three tests for the first half and two
  disqualifiers for the second, and both existing bindings labelled against it with the argument rather
  than the verdict (E.1). `loom.h` split into the lean runtime surface plus `loom_legacy.h`, whose own
  negative check came for free: the first build after the split failed with `'KokoroConfig' is not a
  member of 'loom'`, so the boundary is real (E.2). Five drivers retired, one per commit, after a
  preparatory commit froze their waveforms into `tests/fixtures/legacy_driver_reference/` (E.3).
  **Gate passed:** 137/137 ctest, 0 failed, **98 actually run** (58 at the stage D gate — every TTS
  reference and Lua-driver test was given its model this time). Engine size, RelWithDebInfo stripped,
  same configuration both sides: **1,400,440 → 1,219,952 bytes, −180,488 (−12.9 %)**; `.text`
  1,379,658 → 1,198,924 (−13.1 %). ~7k lines of hand-written C++ orchestration gone, replaced by
  nothing, because the exported Lua driver was already doing the job.

  **Three things this item did not predict**, written up at length under stage E in
  `EXPORT-PREPARATION.md`:
  * **the drivers' *data* outlived their code.** All nine surviving tests default-constructed a
    `VitsConfig`/`MatchaConfig`/… purely to read hyperparameters out of it, so deleting the header
    removed a data structure and not only an implementation. It landed in `tests/tts_driver_inputs.h`
    — honest, but not where it belongs; see the follow-up below.
  * **only two of the seven oracle consumers were MIL tests.** VITS, Kokoro and StyleTTS2's MIL tests
    deliberately do not compare against the bespoke oracle and say why. The bulk of the fixture work
    was the five *bespoke-Lua* tests, which this item does not mention.
  * **a frozen fixture narrows what can be checked, and Supertonic shows where.** Its style vectors are
    a driver *input*, so one waveform is valid for one voice; a different `voice_styles/*.json` now
    skips rather than compares against the wrong reference, and no new style can ever get a fixture.

  **Two follow-ups this stage opened, neither in scope for it:**
  * **the driver-input hyperparameters belong in the GGUF.** `tests/tts_driver_inputs.h` is a test
    holding `n_feats`, `mel_mean`, `style_dim`, `sigma_data`… — properties of the model, which a
    self-contained GGUF should declare and a host should read. Exactly the argument KV-CACHE.md 1.1/1.3
    made for cache geometry, where `test_e2e_whisper_lua_driver.cpp` was sizing a cache from a
    hardcoded C++ struct. It is export-side work and stage E touches no export path, so it was left
    here rather than smuggled in.

    > **DONE (2026-08-07) — six commits, all five TTS families.** See "TTS driver constants moved to
    > the export side" below. The split that made it tractable: a number the **driver** needs is an
    > `ExportConstants` IR local, a number the **host** needs is a `loom.*` GGUF hparam, and which one
    > it is is decided by who reads it. 30 numbers left the `infer` signature; two became hparams
    > (`loom.style_dim`, `loom.txt_len`). `tts_driver_inputs.h` survives for the five *bespoke* Lua
    > tests only, and retires with them in P6.
  * **three components are now C++ with a unit test and no product consumer:** `cfm_euler_sampler.h`,
    `style_diffusion_sampler.h`, `bilstm_stepper.h` (and arguably `ode_stepper.h`). Each existed to
    serve a driver; each has a Lua counterpart the MIL path uses instead — `loom.run_recurrent` +
    `RecurrentPhase`, the `FlowMatchingSampler` component, StyleTTS2's ADPM2 fragment. They were kept
    in `loom.h` rather than `loom_legacy.h` because their remaining consumers are tests *of the
    component*, and deleting them is beyond what this item asks. Whether they follow the drivers is a
    real decision, not an oversight.

    > **DONE (2026-08-07) — three of the four retired, and the fourth was misfiled.** See "The
    > stranded pre-MIL components" below. `cfm_euler_sampler`, `ode_stepper` and
    > `style_diffusion_sampler` are gone with their four tests; **`bilstm_stepper` stays**, because the
    > premise above is wrong for it — its consumers are not tests *of* it but three bespoke Kokoro
    > per-topology tests that construct one to drive the check they exist for. It retires with the
    > bespoke path in P6.

  **Whisper is the one that remains**, and not because it is harder: `whisper_driver.cpp` has no MIL
  export to replace it. `loom_legacy.h` empties out in P4.1, and its docstring says so.

  > **DONE (2026-08-08), and `loom_legacy.h` is gone rather than empty.** P4.1 gave Whisper its MIL
  > export, and the R6 precondition — *the last test consuming it is re-pointed in the same commit* —
  > was then satisfiable: `test_e2e_whisper_mil_export` carries all four retired tests' coverage
  > check for check, against a stronger oracle (HuggingFace) than the two they used (this engine's own
  > other implementation). `src/core/whisper_driver.cpp`, `include/loom/core/whisper_driver.h`,
  > `tools/convert_whisper/` and the four `test_e2e_whisper_*` tests are deleted. All six per-model
  > C++ drivers are now retired, so the header that carried the policy has nothing left to carry and
  > `loom.h` IS the surface.
- **P4.0.9 — KV cache on the MIL path — DONE (stages N/1/2/3).** Specified in [`KV-CACHE.md`](KV-CACHE.md); the one item here
  that adds a *capability* rather than hardening one, which is why its gate differs. `EXPORT-PREPARATION
  .md` §4 filed this for P4/P5 and correctly named `FuseLoomAttention` as the blocker — its
  `_fuse_blocks` body is `pass` (`dialect.py:268`), so `loom_fused_attention → ATTENTION`
  (`exporter.py:125`) is registered and never produced, and a MIL-exported Qwen3 has **28 `SOFTMAX` and
  zero `ATTENTION` nodes**. One measured correction shrinks the work: **`use_past` tracing is not
  needed** — once the SDPA subgraph is an `ATTENTION` node the engine supplies the past itself, so a
  decode step is a call at `n_tokens=1`, not a second traced graph. Four stages: rename every driver
  entry point to `infer` and `main_topo` to `main_topology` (N); declare cache geometry as
  `loom.kv_cache.*` KVs so a host stops needing a per-model C++ struct — `test_e2e_whisper_lua_driver
  .cpp:141` still hardcodes `WhisperConfig` (1); the fusion pass, opt-in so the five TTS families are
  untouched (2); `infer_with_past`, a prefill→decode loop owning its own generation, plus the one input
  that genuinely must be retyped, `attention_mask` → `["n_kv", "n_tokens"]` (3).
  **Gate:** byte-identity for the seven non-causal-LM models; for the four causal ones the topology
  changes by construction, so the gate is their numeric reference tests plus `infer_with_past` agreeing
  token-for-token with iterated `infer`.

  **Done. Gate passed, twelve models swept** (the eleven plus SmolLM2-360M, which reaches the family
  through P4.0.4's generic `hf-causal-lm` fallback and is the smallest fused causal LM on this machine).
  Exported from a worktree at `6170be8` and from the working tree, snapshotted and `diff -r`'d:

  * **nine byte-identical** — conformer-ctc, parakeet-tdt, parakeet-rnnt, kokoro, matcha, vits,
    styletts2, supertonic, and lfm2-**modular** (unfused, so untouched);
  * **three differ, all fused causal LMs, and only where they must.** No weight changed in any of them.
    The topology diff is exactly the retyped mask input plus **one `VIEW` removed per attention block**
    — Qwen3 2094→2066 nodes (−28), SmolLM2 1942→1910 (−32), LFM2-monolithic 836→830 (−6, its real
    attention-block count) — and `attention_mask` going `["n_tokens","n_tokens","1","1"]` →
    `["n_kv","n_tokens"]`, with every other declared input unchanged. Qwen3 and SmolLM2 also gain the
    `infer_with_past` entry; LFM2-monolithic does not, which is the derived-eligibility rule working.

  Numerically: `test_e2e_lfm2_mil_export` still asserts the real HF top-1 tokens 8/8 for **both** the
  fused monolithic and the unfused modular export; `infer_with_past` agrees token-for-token with
  iterated `infer` on Qwen3-0.6B and SmolLM2-360M (22/22 checks each, including `max_new_tokens`,
  `eos_token` early-stop and a prefill issued after generation). 445 python tests, 138/138 ctest.

  **Three things the plan did not predict**, written up under "What stage 3 found" in `KV-CACHE.md`:
  3.1's second half was not implementable as written (the axis cannot reach `_validate_input_axes` at
  all) but a different silent trap in `declared_axes` was, and got closed; §2's soundness argument for
  retyping the mask was **false as measured** — 32 `slice_by_index` ops sat between the input and the
  fused nodes, and their extents were baked at trace time; and **a hybrid architecture cannot decode
  incrementally at all**, which is how LFM2 ended up exporting `infer` alone.

- **P4.0.10 — a state cache for the conv/SSM family, so a hybrid can decode incrementally — DONE
  (4 commits).** Direct
  follow-up to P4.0.9's third unpredicted finding (`KV-CACHE.md`, "What stage 3 found"). LFM2-350M is 6
  attention blocks and **10 ShortConv** ones; it fuses, it gets a cache, it prefills — and the exporter
  then *declines* to emit `infer_with_past` and prints why (`exporter.py:1327-1334`), because a causal
  depthwise convolution is stateful across steps and the KV cache holds K/V and nothing else. So the one
  hybrid in the tree generates at prefill cost per token, and Mamba/RWKV would hit the same rule.

  **The blocklist is designed to shrink, and that is the shape of this item.**
  `_NON_CACHED_SEQUENCE_STATE_OPS` (`exporter.py:1357`) is nine op types by *exclusion* — the safe set is
  the open one — and the last commit here is deleting entries from it, one per op that gains a slot.

  **This is not another attention primitive.** The seam `op_attention` uses is right and should be reused
  verbatim: storage in its own `ggml_context` outside the compute graph (`kv_cache.h:15`), addressed by a
  `layer` attr, writes routed through `PrimitiveContext::side_effects` because they have no
  data-dependency edge to the read, reads as plain views, bound per registered module so no address
  crosses the Lua boundary. What is wrong-shaped is the *storage*: `KvCache` holds a growing prefix
  `[0, n_kv)` indexed by `n_past`, and a causal conv wants a fixed-size rolling window of the last
  `kernel-1` input columns per layer. Generalize it into a per-layer store with two slot families.

  **The first decision, and it is not obvious.** `ATTENTION` owns its cache internally, but
  `SSM_CONV`/`SSM_SCAN`/`RWKV_WKV6`/`7` already exist in the engine (`primitives_recurrent.cpp:32-52`)
  and take their state as an ordinary *graph input* — so either (a) each grows an ATTENTION-style
  internal slot, which `CONV_1D_DW` needs anyway since it has no state parameter at all, or (b) the
  engine gains a general "this declared input is backed by a persistent slot across calls" binding, which
  fits the four recurrent ops as they stand and is the more reusable of the two. Settle it before writing
  code; (b) is the recommendation.

  Exporter side is small once the engine has the storage: read the geometry off the fused nodes the way
  `_kv_cache_geometry()` does (`exporter.py:2064`), emit it as `loom.*` hparams, and lower LFM2's
  ShortConv to the stateful op instead of `CONV_1D_DW`.

  **One requirement worth stating because the oracle depends on it:** a state op must treat `n_past = 0`
  as "no history", exactly as `op_attention` does. That is what keeps iterated `infer` a valid reference
  (`KV-CACHE.md` 3.4 — each call is a full recompute that overwrites the prefix it reads) and what lets
  both paths share one cache in the test. *Touches: LFM2-monolithic only, of the models in the tree.
  **Gate:** LFM2-monolithic gains `infer_with_past` and agrees token-for-token with iterated `infer`,
  under the same 22-check harness Qwen3 and SmolLM2 already pass; every other model byte-identical.*

  **Done, gate passed.** `ConvStateCache` + `SHORT_CONV` (engine), `loom_short_conv` +
  `fuse_loom_short_conv` + the topology rule + `_conv_state_geometry()` (exporter), conv-state
  allocation in the three hosts that allocate a KvCache, and one bug fix without which none of it ran.
  Measured on the real checkpoint: **10 `SHORT_CONV`, dense layers 0–9, 0 `CONV_1D_DW`**, 6 `ATTENTION`
  unchanged, `loom.n_conv_layer=10 / n_conv_state=2 / n_embd_conv=1024`, 830 → 820 nodes (the absorbed
  trim, one per conv block). **LFM2's `infer_with_past` agrees with iterated `infer` 22/22**, its HF
  top-1 reference tokens are unchanged, and Qwen3 is untouched (2066 nodes, 28 `ATTENTION`, 0
  `SHORT_CONV`, no conv keys, 22/22 on its own gate).

  **Sweep — 12 models, 11 byte-identical, 1 differs and only where it must.** Exported from a `git
  worktree` at `4689f79` and from the working tree, snapshotted and `diff -r`'d. Byte-identical:
  conformer-ctc, parakeet-tdt, parakeet-rnnt, kokoro, matcha, supertonic, vits, styletts2,
  lfm2-**modular** (unfused), **qwen3 and smollm2** — the last two matter most, since they take the same `fuse_conv=True`
  path and simply match nothing. LFM2-monolithic differs in exactly three files: `CONV_1D_DW` 10 → 0,
  `SHORT_CONV` 0 → 10, `VIEW` 64 → 54 (the ten absorbed trims), 830 → 820 nodes; three added
  `loom.n_conv_*` hparams; and a driver script that gains `infer_with_past`. **Declared inputs
  unchanged and `tensors.txt` identical — no weight moved.**

  StyleTTS2 is byte-identical too, and getting there fixed a latent breakage: its config declared
  `kokoro_config_path` (a genuinely separate dependency from its own weights) as
  `/home/flavio/.claude/tmp/kokoro_model/config.json`, a path that stopped existing when the
  checkpoints moved to `/home/flavio/Dev/models`. Now pointed there, matching where every other
  hardcoded checkpoint path in the tree already points, and re-verified byte-identical against the
  baseline afterwards.

  **The first run of this sweep was a false pass, and the reason is worth keeping.** It reported all 11
  identical, including LFM2-monolithic, which must differ by construction. `loom-export` runs
  `python -m`, and `-m` puts the caller's **cwd** at `sys.path[0]` ahead of the `PYTHONPATH` the script
  sets to its own repo root — so driving the baseline worktree's `loom-export` from the working tree
  imported the working tree's exporter and measured it twice. This is exactly what §6's "`cd` into the
  tree being measured" is for. A byte-identity gate that cannot fail proves nothing, and the only thing
  that caught it was knowing in advance which model had to differ.

  **The design decision this row asked to settle went the OTHER way, and doing it is what settled it.**
  The recommendation above is (b), a general persistent-slot input binding. `op_attention`'s write path
  is what changed it: a state write-back must be *ordered* against the read, and only an op owning both
  ends can guarantee that. `op_short_conv` gets that ordering more strongly than `ATTENTION` does — its
  write copies a view of the *concatenated* buffer, which reads the slot, so a real data-dependency edge
  exists and ggml cannot schedule the clobber first. (b) would have needed an input binding, an output
  binding, and an ordering guarantee between them that nothing in the graph expresses.

  **And the `VIEW` error this thread started from had two causes, not one.** `KV-CACHE.md`'s third
  stage-3 finding recorded `VIEW: resolved shape [1,1024,1,] ... needs 16380 bytes but parent has 12288`
  and attributed it to the missing conv state. The conv state was real — and after fixing it that
  identical error still fired at `n_tokens = 1`, one layer earlier, on the in_proj channel-split.
  `op_view`'s bounds check spelled "one element" as `parent->nb[0]`, which is only the element size on a
  densely-packed tensor; LFM2's in_proj output is a `PERMUTE` with `ne=[1,3072]`, `nb=[12288,4]`, and
  **`ggml_is_contiguous` reports it as contiguous** because its stride test is skipped whenever
  `ne[0] == blck_size` (`ggml.c:1467`), so the `cont` never fired. The view was always correct; the
  check was not. Using `ggml_type_size` makes it identical to `ggml_nbytes`' own formula (`ggml.c:1299`)
  — the quantity it compares against. `ensure_packed` in the same file already existed for that same
  ggml carve-out, which makes this the third consumer to hit it and worth naming as a recurring trap
  rather than a one-off.

- **P4.0.11 — sliding-window attention. Two items of very different size, and only the small one should
  be done first.** Not on the roadmap and no checkpoint in the tree needs it, but modern hybrids
  (Gemma 3-style interleaved local/global, 5:1) are unreachable without it, so it is filed rather than
  discovered later.

  **(a) Correctness — small.** `ggml_soft_max_ext` already takes an arbitrary `[n_kv, n_tokens]` mask, so
  a banded mask is a `window` argument on `loom.causal_mask` (`lua_bridge.cpp:292`) plus the driver
  builders passing it. Interleaving needs two mask inputs with each `ATTENTION` node routed to its own,
  and **that plumbing already tolerates it**: `_retype_fused_mask_input` (`exporter.py:1936`) iterates the
  *set* of mask names its cached nodes reference and retypes each independently, checking the
  only-consumer property per name — and HF's Gemma trace passes both masks as separate inputs. The real
  work is that the window becomes a per-node fact: the fusion pass must record which mask each block
  consumed (it already carries `mask_var` per block) and the driver must know each mask's window.

  **(b) The memory win — a real `KvCache` redesign, and out of scope for (a).** The header states the
  constraint it would break: single sequence, contiguous append, no ring buffer. (It used to name the
  missing `ggml_set_rows` indirection here too; P4.0.15 added that, so a `pos % window` write is now
  merely a different `fill_cell_index` rather than a different write path.) A window cache wants
  `pos % window` writes, which is exactly what stops "a plain view over `[0, n_kv)` suffices for reads"
  from holding. It also needs
  per-layer capacity, where `KvCache` takes one `kv_size` for every layer and `loom.kv_cache_size` is one
  scalar. This is adjacent to the multi-sequence generalization `SPECIFICATION.md` §8 defers and should
  be done with it, not before it.

  **Why the split is the whole point:** a windowed model is *correct* with a full cache and merely spends
  `n_ctx` memory where it could spend `window`. (a) makes such a model run; (b) makes it cheap. *Gate for
  (a): the first windowed checkpoint's numeric reference test, and a banded-mask unit test at the
  `loom.causal_mask` level; every model in the tree byte-identical, since none declares a window.*

  **(a) is DONE, gated against a real windowed checkpoint (gemma-3-270m-it).**
  `loom.causal_mask(n_tokens, n_past [, window])` bands the mask, with `luaL_optnumber` so every
  existing two-argument call site keeps its exact output — three checks pin the banding, a window wider
  than `n_kv` reproducing the full-causal mask, and `window <= 0` doing the same.

  The routing is `_route_windowed_masks`, and the design was set by a measurement that contradicts the
  plan above: **the window is not in the traced graph at all.** Interleaved models build two masks
  internally only when they build them themselves; this family passes `attention_mask` explicitly (so
  the length stays dynamic under trace), and transformers then uses that one tensor verbatim for both
  mask types — all 18 of Gemma's layers slice the same input. So the fusion pass cannot record a
  per-block window, and `_attention_windows` reads `layer_types`/`sliding_window` off the config
  instead: the one place this exporter prefers a config fact to a graph fact. The exporter then
  *synthesizes* a second declared input (no MIL var behind it), one per distinct window, repoints the
  sliding blocks at it, and the driver fills both in. Keeping the window in the MASK means the engine
  needs no new primitive, attr or branch.

  **Gate: 49977 == HF's top-1 at position 599, 88 tokens past the window; 236881 with the window forced
  off.** Getting there needed a forced token-by-token decode, because two things rule out the obvious
  tests — `infer` cannot prefill past ~512 tokens for this vocab (see the marshalling item below), and
  greedy generation collapses into a repeating `107, 2717` a wrong window would reproduce.

- **P4.0.12 — module-owned output buffers, and retrieval addressed by module name — DONE (2026-08-05).**
  The forward pass and
  the reduction that follows it are currently fused (`loom.run_subgraph_argmax`), because splitting them
  appeared to require handing Lua an opaque tensor handle — and `BuildResult` is only readable while the
  `GraphBuilder` that produced it is alive, so such a handle would dangle the moment the call returned.
  **The author's framing dissolves that:** the KV cache is already persistent state addressed *by module
  name*, with no address ever crossing the scripting boundary (`KV-CACHE.md` §1.1), and an output buffer
  can work exactly the same way. `loom.argmax_row('main_topology', -1)` names a module, which is what
  every Lua call already does.

  Give each declared output a persistent, module-owned allocation — its own `ggml_context` and backend
  buffer, precisely `KvCache`'s shape — with the graph ending in a copy into it, routed through
  `side_effects` the way cache writes already are. The buffer's address is then stable *regardless* of
  whether the graph was rebuilt, which matters because the output is `[n_vocab, n_tokens]` and
  `n_tokens` differs between prefill and decode: a buffer holding only what retrieval needs (often the
  last row) survives every shape change, while "whatever the last build produced" would not.

  **The motivating case is inter-module data flow, NOT the large vocab.** A Lua driver that chains module
  A into module B today reads A's output into a Lua table and writes it straight back as B's input. On
  CPU that is two copies of an intermediate nobody looks at; on a GPU backend it is a device→host→device
  round trip **per edge, per step**. The engine is single-backend CPU today (no `ggml_backend_sched` — see
  the performance section), so the cost is latent rather than paid, but every multi-module model is
  already shaped to pay it the moment a second backend lands: Kokoro, StyleTTS2, VITS, Matcha,
  Supertonic, LFM2-modular's per-layer chain, Parakeet's TDT/RNNT loops.

  **This corrects an earlier reading of the same question, recorded because the correction is the useful
  part.** The first pass at "who should get a buffer" concluded that whole-output consumers (TTS/ASR)
  must keep marshalling and only causal LMs could use one. That is backwards. Those models benefit
  *most*, because their outputs are **intermediates that never need to reach Lua at all**. The rule is
  not "reduce to a scalar"; it is **marshal only when a value is genuinely host-side** — a final result,
  a control decision, or host math the driver actually performs.

  Two things to settle rather than discover. **Staleness:** retrieval reads the module's current buffer,
  so a second run on that module overwrites it and a late read silently returns newer data — wants a
  generation counter that raises, and ideally a static adjacency rule, for which
  `driver_ir.check_subgraph_calls` is already the right home. **Memory:** retaining per-module state
  raises steady-state footprint for many-topology models like Kokoro, though a decode loop *gains*, since
  today every `run_subgraph` allocates and frees a compute buffer.

  *Gate: byte-identity is not it — driver text changes by construction. Per-model e2e Lua-driver tests
  plus a read diff, the same exception P4.0.6's peeling commits take.*

  **What shipped.** `OutputStore` (`include/loom/core/output_store.h`) is the third member of the
  persistent-state family after `KvCache` and `ConvStateCache`, built to the identical seam: its own
  `ggml_context` and backend buffer outside the compute graph, the write returned as a `ggml_cpy` the
  builder routes through `side_effects`, and no address ever crossing the scripting boundary. It is
  owned by the *bridge* rather than lent by the host, which is the one place it had to differ — a
  cache's geometry comes from declared hparams, an output's does not, so only the run that fills it can
  size it. `reshape()` therefore reallocates when the geometry moves, which for a decode loop is once,
  at the prefill→decode transition; retrieval looks the buffer up by name at read time, so it can never
  hold a pointer the store has since replaced.

  Lua surface: `loom.run_subgraph_and_retain(module, axes, inputs)` returns only a generation number,
  and a retained value is read back in exactly one of three ways — which is the "is this genuinely
  host-side?" question made syntactic. `loom.get_output(module, index)` for a final result,
  `loom.argmax_row(module, row)` for a control decision, and `{from = 'module'}` as another module's
  input for the case this exists for, an intermediate the driver merely threads onward. The reference
  form is a table with named fields rather than a new binding: it cannot collide with a data array, it
  is self-describing where a driver is read, and it costs no leanness. `argmax_row`'s module form is an
  *overload* rather than a second binding for a related reason — `n_vocab` is only a parameter of the
  array form because a flat Lua array has lost the shape the tensor still carries.

  **Staleness got both halves.** The runtime one is the generation counter: `check_generation` raises
  naming the module, and every read (`get_output`, `argmax_row`, an `{from = ..., gen = g}` reference)
  can pin itself. The static one landed where the item predicted — `driver_ir.check_subgraph_calls`,
  because an `OutputRef` names a module and `validate()` knows only about symbols, so the ordering
  question had to move to the checker that knows what a module is. It is conservative on purpose:
  retention tracked in statement order, nested `If`/`While` bodies inheriting a copy that does not
  escape, so a producer on one arm of a branch is rejected rather than assumed. Synthesized drivers
  therefore need no `gen` argument and none is emitted; the runtime counter is what covers hand-written
  Lua the static rule cannot see.

  **Adopted on the modular chain only, and deliberately not one stage further.** Every edge of
  lfm2-modular's 20-stage chain is an intermediate — 19 of them now stay engine-side. The *last* stage
  still marshals, because its output is the logits the epilogue argmaxes, and moving that engine-side
  is P4.0.14's own item: doing it here would have added a second reducing path to the modular builder
  while `run_subgraph_argmax` still exists, which is the "two ways to get a token out of a forward
  pass" this project keeps removing. The marshalling cap on that path is therefore still open, exactly
  as P4.0.14 states. *(Closed by P4.0.14 on 2026-08-06: the 20th stage retains too, and
  `run_subgraph_argmax` is gone.)*

  **Gate, measured.** `tests/test_lua_bridge_retained_outputs.cpp` (16 checks) runs every case against
  a marshalled oracle — the retained chain, the pinned chain, `get_output`, `index = 2`,
  `argmax_row` by name and a store reshaped between a 3-token and a 1-token run all reproduce what the
  Lua-table path produces, and the five failure modes each raise an error naming the real problem.
  Re-exported qwen3 (flattened), matcha (multi-phase) and lfm2-modular from a baseline worktree and
  from this tree: `diff -r` over `snapshot_gguf.py` output is empty for the first two — every topology
  JSON and every tensor hash — and lfm2-modular differs in exactly one place, `model.driver_script`
  (and the `kv.txt` line carrying it). The gate could fail and did, for the one model that must move.
  `test_e2e_lfm2_mil_export` against the re-exported modular GGUF: HF's own top-1 at both prompt
  lengths, 3523 at 3 tokens and 2 at 7. Full ctest 142/142, exporter suite 453/453.

- **P4.0.13 — persist the graph itself, after P4.0.12 — DONE (2026-08-05).** The bucketed graph-reuse
  item already described
  under "Performance optimizations designed but not implemented", scheduled here and in this order for a
  reason: once the bridge retains per-module state for P4.0.12, the `GraphBuilder` is already being kept
  alive, which is the part reuse needs.

  **Correction, measured after P4.0.12 shipped (2026-08-05): that last sentence was wrong, and the work
  it promised is still all here.** What P4.0.12 made per-module and persistent is the *output store*, on
  `LoomLuaBridge::Module` beside `kv_cache`/`conv_state` — the builder is untouched. `compute_and_emit`
  still constructs a `GraphBuilder` per call and destroys it on return, so `reserve()` is still dead
  weight on this path and every `run_subgraph`/`run_subgraph_and_retain` still pays a full rebuild plus a
  compute-buffer allocation it throws away. What P4.0.12 genuinely bought this item is smaller but real:
  it established that per-module persistent state on the bridge is the right home for it (three classes
  now use that seam), and it removed the reason a builder's lifetime was entangled with a value's — an
  output no longer has to outlive the builder that produced it, so keeping builders alive can be decided
  on reuse grounds alone. Today `compute_and_emit` constructs a fresh builder per call and
  destroys it on return — so **`GraphBuilder::reserve()` is dead weight on this path**, called only by
  the legacy `Generator` (`src/core/generation.cpp:20`) and never by the Lua bridge, meaning every
  `run_subgraph` pays a full rebuild plus a compute-buffer allocation it then throws away.

  Keeps the hazard this item has always carried, and it is the one place idea 12 does *not* help: the
  `ggml_gallocr` input-aliasing bug is root-caused but reuse is only safe while **every declared input is
  rewritten every decode step**, and it needs its own bit-identical-to-rebuild regression test on the
  `test_graph_reuse_safety.cpp` pattern. P4.0.12 does not go near this, which is exactly why it should
  land first.

  **What shipped.** A `GraphBuilder` retains the last graph it built — its `ggml_context`, its
  `ggml_cgraph`, its gallocr-assigned compute buffer, its declared-input tensors — and `build()` returns
  that same graph unchanged when called again with the same axes. The builder is now the unit of "one
  live graph" rather than a factory producing a new one per call, so `build()` returns
  `const BuildResult&`: the header already said a result is readable only while its builder is alive, and
  a reference is what stops that from being merely documented (~100 call sites, all mechanical). On the
  Lua path the builder moved onto `LoomLuaBridge::Module` beside `kv_cache`/`conv_state`/`outputs` — the
  fourth member of that seam, exactly where the correction above said it belonged — constructed on first
  use, so a many-topology model pays a retained compute buffer only for the modules its driver actually
  runs. That laziness is the answer to the same footprint trade P4.0.12 named, and it lands harder here:
  a per-call builder held one compute buffer at a time, these hold one per live module.

  **The hazard is gone rather than disciplined, and that is the part worth recording.** The plan was
  reuse plus a rule — safe only while every declared input is rewritten every step, because
  `ggml_gallocr` may alias a computed tensor's buffer onto one of the graph's own declared inputs. But
  that rule is only needed because the inputs sit in gallocr's pool at all. They now get the builder's
  own `ggml_context` and backend buffer, outside it — the same seam `KvCache`/`ConvStateCache`/
  `OutputStore` use — and gallocr skips any tensor whose data is already set, exactly as it does a weight
  or a cache view (`ggml_gallocr_is_allocated`, `ggml-alloc.c`). Nothing gallocr places can land on an
  input, so a retained graph cannot be corrupted by an input that was not rewritten.
  `tests/test_graph_reuse_safety.cpp` still holds and still documents the raw-ggml behaviour; what
  changed is that `GraphBuilder` no longer exposes it. `OdeStepper` keeps rewriting all three inputs
  every step because it is the clearest way to write the loop, not because it is load-bearing any more.

  **Exactly one graph is retained, deliberately not an LRU keyed by shape.** A retained `OutputStore` is
  reshaped by the build that fills it, so only the most recent build's `ggml_cpy` destinations are
  guaranteed to still be the store's current tensors; a shape-keyed cache could hand back a graph whose
  copies point into a buffer `reshape()` has since replaced. Going back to an earlier shape rebuilds.
  The key is the axes map plus the `OutputStore*`, because whether a run ends in a copy into a store is
  a property of the call (`run_subgraph` vs `run_subgraph_and_retain`), not of the module.

  **What this does NOT do — a second correction, to this item's own plan this time.** The *bucketed*
  variant is still not implemented, and bucketing alone would never have delivered it. `n_past` is baked
  into the graph independently of `n_kv`: `KvCache::write_k/write_v` build a `ggml_view_2d` at byte
  offset `n_past * nb[1]`, so two consecutive decode steps have different graphs even at an identical
  rounded-up `n_kv`. Making a decode loop reuse its graph therefore needs the KV *write destination* to
  become data — llama.cpp's `ggml_set_rows` index-tensor indirection, which the scope limitations below
  already list as absent — and that is a change to `KvCache`, to `ATTENTION`, and to a synthesized
  input. **Filed as P4.0.15** rather than smuggled in here, and done there on 2026-08-07 — including a
  correction to this sentence's last clause, which also predicted a change to "every causal-LM driver's
  text" and re-gating of every cached model: the synthesized input turned out to belong to the engine,
  and no driver or export moved. What P4.0.13 does cover is every loop whose axes *don't* move, which is
  most of the zoo: `loom.run_recurrent` (one build per direction instead of one per timestep — the
  StyleTTS2/Kokoro BiLSTMs), the CFM Euler and ADPM2 sampler loops, `TdtDecoder`'s per-layer LSTM and
  joint calls, and every module in a chain that is called at a fixed shape. Modules called once still
  build once, as before, but now keep their compute buffer instead of allocating and freeing it per call.

  *Gate: the bit-identical-to-rebuild regression test this item asked for, plus the existing e2e drivers
  unchanged — driver text does not move, so no model needs re-exporting for this.*

  **Gate, measured.** `tests/test_graph_builder_reuse.cpp` (35 checks) runs the toy LLM — a real
  topology with a KV cache, a `repeat_for` block, RoPE and an f32 mask input — four ways. A five-step
  fixed-shape loop through one retained graph is **bit-identical** (`memcmp`, not `==`) to the same five
  steps through a builder that has only ever built once, at `builds()==1, reuses()==4`; the comparison
  can fail, since consecutive steps really do produce different logits and the repeated step reproduces
  the first exactly. A prefill+decode sequence, where `n_past` moves every step and nothing is reused,
  is bit-identical to the same sequence driven through a builder thrown away between every call — the
  check that moving the inputs out of the gallocr pool changed no numerics. The single-entry rule is
  asserted by graph-pointer identity and by eviction. And every declared input is confirmed to carry its
  own backend buffer and to share an address with no node in its own graph, which is the assertion that
  fails first if a future ggml changes what made reuse safe. Full ctest 143/143. Re-exported matcha,
  styletts2, kokoro and lfm2-modular from this tree and ran their Lua-driver e2e tests green (the
  checked-in root GGUFs are from before the 2026-08-02 `infer` rename and abort for that reason on
  `main` too, unrelated to this item).

  **And the wall-clock win is small, which is worth writing down because the item implies otherwise.**
  Measured by toggling only the cache-hit branch in the same binary — everything else, including the
  inputs' move out of the gallocr pool, held constant — over two runs each of the re-exported drivers:

  | driver | rebuild every call | retained graph |
  |---|---|---|
  | kokoro | 26.58s, 21.77s | 19.35s, 19.47s |
  | matcha | 13.72s, 11.65s | 12.26s, 11.79s |
  | styletts2 | 21.45s, 21.57s | 20.73s, 21.69s |
  | lfm2-modular | 4.30s | 4.28s |

  Only Kokoro shows a gain outside the noise, and even there it is ~15%, not a multiple. The reading is
  that on a single CPU backend the rebuild is simply not what these drivers spend their time on — the
  compute is — and the same is true of the compute-buffer allocation the old per-call builder threw
  away. That does not make the item wrong, it locates it: what it removes is per-call *structure*, and
  the structure it removes is what a second backend would make expensive, exactly as with P4.0.12's
  retained outputs. Worth knowing before anyone budgets the bucketed decode-loop follow-up on the
  strength of an expected speedup.

- **P4.0.14 — the same marshalling ceiling still stood on the modular path, and is fixed by P4.0.12's
  mechanism — DONE (2026-08-06).**

  P4.0.12 shipped `loom.run_subgraph_and_retain` plus `loom.argmax_row(module, row)`: the fused call
  said as two facts. The modular chain had adopted the first half — 19 of lfm2-modular's 20 stages
  retained — and deliberately not the second, because its last stage's output is the logits the epilogue
  argmaxes and adding a second reducing path while `run_subgraph_argmax` still existed would have left
  two ways to get a token out of a forward pass. This item is the other half, and the retirement.

  Against LuaJIT's ~2^27 array limit, each checkpoint's own `vocab_size` gave:

  | model | vocab | prefill ceiling | reduced engine-side before this item |
  |---|---|---|---|
  | gemma-3-270m | 262144 | ~512 tok | yes, via `run_subgraph_argmax` |
  | qwen3-0.6b | 151936 | ~883 tok | yes, via `run_subgraph_argmax` |
  | smollm2-360m | 49152 | ~2730 tok | yes, via `run_subgraph_argmax` |
  | **lfm2-350m modular** | 65536 | **~2048 tok** | **no** |

  **What shipped.** Both synthesized builders now retain and reduce by name, in the same mode:

  * `ChainStage`'s last stage retains like every other one, and `apply_modular_export` no longer has a
    step 7 that makes it different. The chain binds no Lua local at all.
  * `MonolithicCall` grew `retained`, set for a KV-cached topology, replacing `argmax_row`.
    `PrefillDecodeLoop` emits the retain and the reduction as two statements instead of one fused call.
  * `ArgmaxEpilogue.already_reduced` (a bool meaning "somebody else already did the argmax") became
    `retained_module` (a topology name meaning "reduce that module's retained output"), which is a
    strictly better field: it is **link-declared** — `WhenSet(TopologyName())` — where the bool could
    only ever be `Unchecked`.
  * `driver_ir` gained `RetainedArgmax`, `OutputRef`'s sibling for the one read of a retained value that
    is a control decision rather than an edge, and `check_subgraph_calls` now enforces the same
    adjacency rule for it. That closes the gap the change would otherwise have opened: an epilogue
    naming a module whose producing call still marshals is invisible to `validate()`, and would have
    failed at runtime rather than at export.
  * `loom.run_subgraph_argmax` is gone — binding, trampoline, declaration, and the IR field and codegen
    branch behind it. The Lua surface is 14 bindings, not 15.

  **What it costs, which the item did not predict.** Retention copies each declared output into the
  module's `OutputStore`, so a prefill now holds the logits tensor twice — once in the compute buffer,
  once retained. At Gemma 3's 262144-wide vocab and a 512-token prompt that is an extra ~512 MB, freed
  down to `[n_vocab, 1]` at the first decode step (`reshape` reallocates only when the geometry moves).
  The fused call read the row straight out of the graph result and kept nothing. That is a real trade and
  it is the same one P4.0.12 named under "Memory" — worth knowing before anyone points this at a
  long-context prefill, and the reason a future item that retains only the row retrieval asks for would
  have something to fix.

  **Read P4.0.16 next, which corrects the proportions here.** Reviewing this paragraph turned up a
  larger and *permanent* retention underneath it — the gallocr compute buffer, which never shrinks — so
  the duplicate above is the smaller half of the memory story and the only transient one.

  **Gate — measured.** Byte-identity is not the gate; driver text changes by construction, so the gate
  is which models change and which must not. All 13 exported from a `git worktree` at `4bc83a5` and from
  this tree, `snapshot_gguf.py` both, `diff -r`:

  * **Eight byte-identical** — conformer-ctc, parakeet-tdt, parakeet-rnnt, kokoro, matcha, supertonic,
    vits, styletts2. No ASR or TTS driver text moved, as intended: those topologies have no KV cache, so
    `MonolithicCall` still marshals and `ArgmaxEpilogue` keeps its `type(...) == 'table'` branch.
  * **Five differ, in `model_driver_script` and the `kv.txt` line carrying its sha, and nowhere else** —
    qwen3, smollm2, gemma-3-270m-it, lfm2-monolithic (4 lines each: `run_subgraph_argmax` becoming
    retain-plus-reduce in both `infer` and `infer_with_past`) and lfm2-modular (6 lines becoming 2: the
    final `run_subgraph` and the whole `type(...) == 'table'` guard collapsing into
    `loom.run_subgraph_and_retain('suffix_1', ...)` + `return loom.argmax_row('suffix_1', ...)`). Every
    topology JSON and every tensor hash identical for all five.

  Numerically, on re-exported artifacts: `test_e2e_lfm2_mil_export` 8/8 — both LFM2 forms reproduce HF's
  own top-1 at both prompt lengths (3523 at 3 tokens, 2 at 7). `test_e2e_causal_lm_infer_with_past` 22/22
  on qwen3 and 22/22 on lfm2-monolithic: the cached decode loop still generates exactly what iterated
  `infer` does.

  **And the capability itself, which is what the item is actually about:**
  `tests/test_e2e_prefill_past_marshalling_ceiling.cpp` prefills a prompt whose logits tensor is larger
  than LuaJIT can hold — the length computed from the file's own vocab — and asserts the call completes
  with a token id in range. There can be no marshalled oracle for it, which is the point: the marshalled
  path does not reach the input at all. lfm2-modular at **2064 tokens** (ceiling 2048) returns 61238,
  lfm2-monolithic the same, qwen3 at **899** (ceiling 883) returns 100.

  **The gate can fail, and does.** The same binary against a **baseline-exported** lfm2-modular reports
  `prefill of 2064 tokens FAILED: ... table overflow`, naming the 135266304 logits the old driver tried
  to marshal. That is the check worth having: a capability test that passes on the tree that lacks the
  capability would prove nothing, and this one does not.

  Full ctest 144/144, exporter suite 463/463.

- **P4.0.15 — index-tensor KV writes, so a decode loop can reuse its graph — DONE (2026-08-07).**
  Opened by P4.0.13, which could not finish without it. P4.0.13 made `GraphBuilder` retain and reuse its
  graph, and that covers every loop whose axes don't move. It did not cover the one this whole thread
  started from — an autoregressive decode — and the plan it inherited ("round `n_kv` up to a bucket
  boundary and skip the rebuild while the bucket holds") would not have covered it either. **`n_past`
  was baked into the graph independently of `n_kv`:** `KvCache::write_k/write_v` built a `ggml_view_2d`
  at byte offset `n_past * nb[1]`, so step N and step N+1 had different graphs whatever `n_kv` rounded
  to. Bucketing is necessary and not sufficient; the write destination had to stop being a build-time
  constant first.

  **What shipped.** `KvCache::write_k/write_v` take a cell-index tensor instead of an `n_past` and
  scatter through `ggml_set_rows` — the indirection `kv_cache.h`'s own comment used to name as absent
  and `llama_kv_cache` already has. `GraphBuilder` then rounds `n_kv` up to `kKvBucket` (32, llama.cpp's
  own non-flash `n_pad`), capped at the cache's capacity, and keys its retained graph on the axes
  reduced to what the structure actually depends on: `n_past` dropped, `n_kv` replaced by the padded
  value. A prefill plus a 40-step decode is **three graphs for 41 calls** — the prefill shape, the
  decode shape in the first bucket, the decode shape in the second.

  **The cell-index tensor is engine-synthesized, not topology-declared, and that is a deliberate
  departure from this entry's own plan.** The plan above said "a declared cell-index input" and "every
  causal-LM driver's text", which would have meant a fifth input on every `ATTENTION` node, a line in
  every synthesized driver, and — the part that decided it — **re-exporting every cached GGUF, with
  every previously exported one becoming unloadable**. Three things argued the other way and won:

  * Its value is `[n_past, n_past + n_tokens)`, a pure function of two axes the caller already binds.
    That is exactly the argument by which `n_kv` is already derived in `GraphBuilder::build` rather than
    passed — "so every caller of an attention-bearing topology gets it without having to compute it".
    A driver supplying it could only restate what the engine already knows.
  * The **bucket is engine policy over the engine's own cache**, and the mask has to be padded to it
    regardless. Having the driver name the cells while the engine silently decides the mask's width
    would split one decision across two authorities.
  * "Fat exporter, lean runtime" is about *per-model* complexity. Nothing here is per-model.

  So `PrimitiveContext` gained a `kv_cells`, `GraphBuilder` allocates it beside the declared inputs
  (outside the gallocr pool — that seam is now load-bearing for correctness, not just for reuse safety)
  and rewrites it **on a reuse as well as on a build**. `tools/` is untouched; the exporter suite passes
  474/474 unchanged, and no model needed re-exporting.

  **Padding the mask, and where that lands.** A bucketed `n_kv` widens the mask input, so somebody has
  to fill the tail with `-inf`. `loom.causal_mask` cannot: it is not told which module its result feeds,
  and an unbucketed topology (MIL-exported Qwen3 declares `["n_tokens", "n_tokens"]`) would break if it
  padded unconditionally. The width is known at the *write*, so that is where it happens —
  `BuildResult` names the declared inputs whose leading dim is `n_kv` plus the un-padded length, the
  Lua bridge places a real-width array into the padded tensor, and the two C++ drivers
  (`Generator`, `WhisperDriver`) simply read the width off the tensor, at which point their existing
  `j <= query_pos` rule writes the `-inf` tail for free. **No driver script changed**, which is what the
  entry's "no driver learns what a bucket is" was really asking for.

  **Padded cells contribute exactly zero, verified rather than assumed.**
  `test_padded_cells_contribute_nothing` primes cells `[n_used, capacity)` with K = 1000 and V = -1000
  in every layer — through the new index-tensor write, which is the first use of it for something other
  than an append — and requires the whole prefill+decode sequence to come out bit-identical to the same
  run against an untouched cache. A zeroed cell reached through a finite mask would also produce zero,
  so a clean cache could not have told the two apart; this can.

  **Gate.** Full ctest **135/135**, exporter suite **474/474**. `tests/test_graph_builder_reuse.cpp`
  gained the two tests above and had its decode-sequence assertion inverted — it read
  `builds() == 4 && reuses() == 0`, the behaviour this item exists to remove, and now reads
  `builds() == 2 && reuses() == 2`. On real checkpoints, every env-gated cached path:
  `test_e2e_sliding_window_attention` against **gemma-3-270m-it** (600 tokens past a 512 window, forced
  decode *and* prefill, both matching HF's own top-1 — the hardest case, with two padded masks and 18
  cached layers), `test_e2e_causal_lm_infer_with_past` against gemma-3-270m-it and against
  **LFM2-350M** (a hybrid, so `ConvStateCache` and `KvCache` advance together),
  `test_e2e_prefill_past_marshalling_ceiling`, `test_e2e_lfm2_mil_export`, and all four whisper tests
  against whisper-tiny — whisper being the last consumer of the bespoke `["$n_kv", "$n_tokens"]` mask
  spelling and of the C++ `WhisperDriver`, and so the only place both non-Lua mask writers are
  exercised at all.

  **No speedup is claimed, and none was measured**, exactly as this entry asked. P4.0.13 measured the
  retained-graph win at ~15% on Kokoro and inside the noise everywhere else on a single CPU backend; the
  rebuild is not where these drivers spend their time. The case is P4.0.12's: it removes per-call
  structure that a second backend, not this one, makes expensive. What it also does is unblock the ring
  buffer and multi-sequence support listed under "Scope limitations" — both wanted this indirection, and
  `KvCache::fill_cell_index` is now the single place a second addressing policy would go.

  **What this does not do.** The bucket is a constant, not adaptive: a 4096-token context still rebuilds
  every 32 steps, and the last bucket of a full cache is ragged (capped at capacity) rather than a
  boundary. `mentions_symbol("n_past")` is a substring test, so a topology with an `n_past`-derived
  shape falls back to per-step rebuilds rather than being handled — safe, and no model does it.

- **P4.0.16 — give the compute buffer back when a build stops needing it — DONE (2026-08-06).** Found
  while reviewing P4.0.14's memory cost at the author's prompting, and it turned out the item I had
  flagged there was the *smaller* of two retentions.

  **gallocr grows and never shrinks.** `ggml_gallocr_reserve_n_impl` reallocates a chunk only when
  `new_chunk_size > cur_chunk_size` (ggml-alloc.c) — the right default for a caller who reserves a worst
  case, the wrong one for a prefill followed by a decode loop. Since P4.0.13 the builder that ran the
  prefill *is* the builder that serves every decode step, so the prefill's buffer was held for the whole
  generation. **Measured on gemma-3-270m-it at a 512-token prefill: 513.2 MiB held where 1.0 MiB is
  needed**, for every step, for the lifetime of the bridge. P4.0.14's `OutputStore` duplicate is the same
  order of magnitude but genuinely transient — `reshape()` drops it to `[n_vocab, 1]` at the first decode
  step. This one never came back.

  `GraphBuilder::build` now drops the gallocr when a scratch plan says this graph needs less than half
  of what the buffer holds, and the next alloc sizes a fresh one. Three things are load-bearing:

  * **The plan runs on a scratch allocator, never the live one.** `ggml_gallocr_reserve_n_size` runs the
    real planner with `no_alloc=true`, which frees the live buffers in the *growing* case — exactly when
    they are about to be needed.
  * **It is armed by a preceding growth, not run per build.** The plan is a second full pass over the
    graph on top of the one `alloc_graph` already does. Running it unconditionally measured slower on a
    1742-node graph, and arming it on *any* growth was no better — a cached LM grows `n_kv` by a token
    per step, so the buffer creeps and re-arms constantly. Arming on a **doubling** separates "a
    different regime is running" from "n_kv grew by one", and the same factor gates the shrink itself so
    the two ends cannot disagree. A 100-step generation now reports `shrinks() == 1, builds() == 101`.
  * **`reserve()` suppresses it entirely.** The two are opposite policies over one buffer — "hold the
    worst case" vs "give back what this shape does not need" — and a builder cannot honour both. Only
    the legacy `Generator` reserves; the Lua bridge never does, so every driven model gets the shrink and
    `test_gallocr_reserve_reuse`'s contract is untouched.

  *On the timing.* Wall-clock deltas on this machine sat inside its own ~1 ms/step run-to-run variance,
  and repeated A/B runs crossed over — so no speed figure is claimed, and the code comment says so. The
  design rests on the counted property (one probe per regime change), which is exact, not on a timing.

  **Gate.** `tests/test_gallocr_shrink.cpp`, deliberately written as the sibling of
  `test_gallocr_reserve_reuse.cpp` and stating the opposite contract, with the `reserve()` case as the
  thing that keeps the pair consistent: shrink happens, costs exactly one probe over 33 builds, never
  fires for a fixed-shape loop (which also still reuses its graph 7 times out of 8 calls — P4.0.13
  undisturbed), and never fires after `reserve()`. On real models: gemma-3-270m-it drops 513.2 → 1.0 MiB
  at the first decode step, and `test_e2e_causal_lm_infer_with_past` still passes 22/22 on it — the
  allocator is recreated mid-generation with a live `KvCache`, which is safe for the reason the cache
  exists outside the pool at all. Full ctest 145/145.

  **What this does not do** is remove the peak. During a prefill the compute buffer and the retained
  output are both live by construction — that is what the `ggml_cpy` into the store *is*. Removing it
  means building into the store rather than copying into it: pre-set the declared output's `data` to the
  store slot so gallocr skips it, exactly as `build()` already does for declared inputs. Needs a fallback
  for an output that is a view (its `data` is its parent's), and it costs the pool the ability to recycle
  that tensor. Not attempted here.

- **P4.0.17 — the NeMo ASR family has no Lua entry point, and needs its own driver builder.**

  Every other family reaches its model through `infer` in the embedded driver. The three NeMo ASR
  encoders do not, and the gap is wider than "not migrated yet": **their MIL exports are currently
  unreachable by anything but their own test.**

  * The synthesized `infer` they *do* carry is the causal-LM one and would raise if called. It argmaxes
    row `#waveform - 1` — one less than the **sample** count — of a `[num_classes, n_frames]` CTC
    tensor, and `l_argmax_row`'s array form bounds-checks that. Known since the exports landed;
    `test_e2e_conformer_ctc_mil_export.cpp` says so in its header.
  * `loom_cli --wav` cannot load them either: it reads the **bare** `model.graph_topology`, which the
    bespoke `tools/convert_nemo/` converters write and the MIL exporter never does (it always writes
    named `model.graph_topology.<name>`).

  So three checkpoints are traced, numerically verified against `reference_forward_conformer.py`, and
  runnable only from `GraphBuilder` in C++. That is the actual defect, not the tidiness of having two
  paths.

  **Root cause: the builder is selected by the *decomposition*, and for ASR the decomposition is not the
  orchestration.** `SYNTHESIZED_BUILDERS["Flattened"]` is `PrefillArgmaxBuilder` — prefill, argmax the
  last row, one token — and the ASR encoders share `Flattened` with the causal LMs while sharing none of
  their host-side shape. `DriverBuilder`'s own premise ("selected by the decomposition, not owned by the
  family", `EXPORT-PREPARATION.md` §5 decision 2) holds for every other family and breaks here.

  **Nothing about this needs new engine C++, which is the point.** Greedy CTC decode is a per-frame
  argmax, then collapse consecutive duplicates and drop the blank (`src/core/ctc_decode.cpp`, 30 lines)
  — all of it expressible in the existing Lua vocabulary over a retained output. TDT/RNNT is the same
  answer one level up: per-layer LSTM cell topologies threaded through a Lua loop plus the joint network
  and duration jumps, which is the shape `whisper_driver.lua` already runs and the conclusion P4.0.6
  reached about `BiLstmStepper`. Both decoders leave the runtime rather than moving behind a binding —
  a `loom.ctc_greedy_decode` would be family-specific logic in an engine that is supposed to stay small.

  Sequence, in dependency order:

  1. **Conformer-CTC** (done — see below). One reduction binding, a CTC epilogue component, and a
     builder the ASR family selects.
  2. **Parakeet TDT/RNNT — route chosen: TRACE the prediction network and joint, and retire
     `convert_parakeet_tdt.py` with them** (author's call, 2026-08-06). Not the same shape as step 1:
     the parakeet MIL export is *encoder-only* (`nemo_asr_export.py`'s `ENCODER_BT_D` says the
     prediction LSTM and joint "are NOT traced ... driven autoregressively by the C++ TdtDecoder"), so
     there is nothing in the artifact for a driver to orchestrate yet. The plan, with the checkpoint's
     real shapes read off `parakeet-tdt-0.6b-v3` rather than assumed:

     * `embed` — `nn.Embedding(8193, 640)`. Its own small traced phase: the driver hands it
       `last_label` and gets the cell's `layer_input`.
     * `pred_lstm` — `decoder.prediction.dec_rnn`, an `nn.LSTM(640, 640, num_layers=2)`. **One
       `RecurrentPhase`** (done below): a stack traces to one `lstm` op per layer, so the phase emits
       `pred_lstm_l0_fwd`/`pred_lstm_l1_fwd`.
     * `joint` — `enc` Linear(1024→640), `pred` Linear(640→640), then ReLU and Linear(640→**8198**),
       which is 8193 token classes plus the 5 TDT durations. Emit the two heads as separate declared
       outputs so the driver can `argmax_row` the tokens without marshalling them and read only the
       five duration logits with `get_output` — no new binding needed. Plain RNN-T has no duration head
       and the second output simply is not there.
     * `encoder` — the existing trace, moved from `main_topology` into a named phase.

     That makes parakeet a **`MultiPhase`** export, whose driver is a checked hand-written Lua fragment
     (the shape all five TTS families use) rather than a synthesized builder — the TDT double loop is
     orchestration, and `MultiPhaseDriverBuilder` is what already exists for orchestration a family
     owns. The loop itself belongs in `loom_lua` beside `run_bi_lstm`.

     Two things already in hand before it starts: the decoder's redundant per-frame prediction recompute
     is gone (see below), and the A/B harness for it — both checkpoints decoded over `samples/jfk.wav`,
     36 and 26 tokens — is the equivalence gate the new driver must reproduce. It matters because the
     `parakeet-rnnt` *reference fixture decodes to an empty token list*, so the existing e2e test cannot
     tell a working decoder from a broken one.

  3. **Retire the bespoke `tools/convert_nemo/` converters** once all three MIL exports are reachable,
     which also removes the bare-vs-named topology split that keeps `loom_cli --wav` on the old files.

  Gate is the one those models already have: `reference_forward_conformer.py` and the existing
  `test_e2e_*_mil_export` fixtures, plus byte-identity for every non-ASR model.

- **Retiring `loom::Generator` — blocked, and on something worth knowing.** It is the pre-Lua host loop
  (`src/core/generation.cpp`), and the natural companion question to P4.0.17. It cannot go yet:

  * **Its users are pre-MIL GGUFs with no `model.driver_script` at all.** Every `Generator` call site —
    `test_e2e_toy_llm{,_generic}`, `test_e2e_gqa`, `test_e2e_qwen3{,_generic,_q8_0}`,
    `test_generation_smoke` — parses the *bare* `model.graph_topology` of a hand-built or bespoke-
    converted fixture. There is no Lua entry to call instead; retiring Generator means re-basing those
    fixtures onto MIL exports or deleting the tests.
  * **`GenerationConfig::on_token` hands back the whole `n_vocab` logits row per step**, and that is
    what the strongest numerical tests in the tree are built on (`expected_logits_step*.bin` compared
    against HF at ~1e-6). The Lua path deliberately no longer marshals logits at all — P4.0.14 removed
    the last way they cross the boundary in a synthesized driver — so those tests would have to be
    re-expressed through `loom.get_output` first.

  The payoff is real when it comes: `GraphBuilder::reserve()` exists *only* for Generator, and P4.0.16
  made `reserve()` the switch that suppresses the compute-buffer shrink — so retiring Generator deletes
  `reserve()`, `reserved_`, and the shrink's only special case along with it. Worth doing, and not as a
  rider on anything else.

- **P4.0.18 — no exporter function should build a driver by interpolating text into a marker. Delete
  `render_driver`'s substitution — DONE (2026-08-07).** (Author's direction, 2026-08-06: "No function
  should be interpolating scripts with marks.")

  `SAMPLER_MARKER`, `_substitute`, `render_driver` and `_TextDriver` are gone from
  `flow_matching_export.py`, which is now the declaration (`FlowMatchingSpec`, `EstimatorSpec`) and the
  codegen (`render_sampler`) and nothing else — a pure `spec -> str` with no opinion about the file its
  output lands in. Where it lands is `driver_components.FlowMatchingSampler`'s business: the function as
  its `prelude`, the line calling it as IR.

  **The item's own prediction was right about the samplers and wrong about the estimators, and the
  correction is the finding.** It said the link checks `render_driver` also ran "are already duplicated
  [on the component path] for peeled families, which is worth confirming rather than assuming".
  Confirmed for `samplers()`: Matcha's and Supertonic's `driver_components()` read `self.samplers()` and
  hand the spec to `FlowMatchingSampler`, whose `sub_specs()` registers that same object with the
  export's checker. One spec, two readers, no copy.

  `estimators()` was **not** duplicated — and had not been checked at all since StyleTTS2 was peeled.
  It was the peeled path that skipped it: `render_driver` ran only on the unpeeled branch, and StyleTTS2
  is the only family that ever implemented `estimators()`. What covers that call today is something
  better than a rehomed declaration, which is why nothing was rehomed: **`LuaFragment` parses it out of
  the fragment's own text.** `02_style_diffusion.lua` contains a literal
  `loom.run_subgraph("diffusion", ..., {x_in = ..., time = ..., embedding = ...})`, so its fragment's
  `sub_specs()` yields a `RunSubgraphCall` with the same topology, the same input set and the same two
  links (`TopologyName`, `TopologyInput(exact=True)`) — plus the file and the line on the label. A
  closure is no obstacle to it: the parse reads Lua source, not entry-function structure. So
  `estimators()` is deleted from both the base config and StyleTTS2 rather than moved, on the standing
  argument that a declaration nobody reads is worse than none — here it was a *second* copy of a check
  that a parse of the real text cannot go stale against.

  `driver_script_path` is **kept and re-documented**. The item allowed retiring it "if nothing needs it
  afterwards"; every peeled family needs it, as the *directory* its `.lua` fragments are read from. Its
  `Unchecked` note said it was "the hand-written Lua the export substitutes generated samplers into",
  which stopped being true at C.4 and was never corrected.

  `driver_components()` no longer defaults to `None`; it raises `NotImplementedError` like `phases()`.
  That default was the switch that selected `RawLuaDriver` around a whole hand-written `.lua`, which is
  what kept the substitution reachable at all. `RawLuaDriver` itself **stays** — its registry entry has
  argued since D.1 that it is how the *next* hand-written driver is adopted, in a commit whose gate is
  byte-identity — but it now has no construction site, which is the honest state: an unused component is
  fine, a live branch selecting an unused component is a route a new family gets taken down by accident.

  One piece of residue the item did not name went with it: `Decomposition.driver_builder(config,
  **context)`. The `**context` was documented as "whatever the specific decomposition needs beyond the
  config", and in the whole tree exactly one thing was ever passed through it — `MultiPhase`'s
  `source=`, the post-substitution driver text. With no text to hand over, the parameter is an
  extension point nobody had asked for twice, so the hook is now `driver_builder(config)`.

  **Gates.** All five TTS families re-exported from a baseline worktree and from the working tree,
  `snapshot_gguf.py` both, `diff -r`: **empty** — every KV, every topology JSON, every tensor hash and
  all five `model.driver_script` texts (1,087 lines) identical. That gate cannot fail by construction
  here, which is exactly the trap `BACKLOG.md` §6 warns about, so the real evidence is the negative one:
  breaking `02_style_diffusion.lua`'s call (`embedding` → `attn_mask`) makes the StyleTTS2 export
  **refuse**, with

      02_style_diffusion.lua:13 loom.run_subgraph('diffusion') does not match topology 'diffusion':
      supplies input(s) it does not declare: ['attn_mask']; leaves declared input(s) unsupplied:
      ['embedding']; topology declares ['x_in', 'time', 'embedding'], spec supplies ['x_in', 'time',
      'attn_mask'].

  — which is the check `estimators()` claimed to provide, still running after `estimators()` is gone,
  naming a line rather than a spec. The removed default is pinned by
  `test_multi_phase_export.TestTheDriverHookIsRequired`. Exporter suite **479/479, 0 failed** (480
  before: −2 for `render_driver`'s two marker-substitution tests, which tested a function that no longer
  exists, +1 for the hook test — every *validation* test `render_driver` hosted was rewritten onto
  `spec_protocol.check_links` and kept, including the four that assert an error message verbatim).
  `DRIVER-COMPONENTS.md` regenerated: one line moves, `raw_lua_driver`'s "no model uses it" note, and no
  component's *used by* column changes — which is itself a check that no family's component list moved.

### TTS driver constants moved to the export side — DONE (2026-08-07)

P4.0.8's first follow-up. Thirty numbers that a host had to pass into `infer` — Matcha's `n_feats`/
`mel_mean`/`mel_std`, Supertonic's five, VITS's four, Kokoro's seven, StyleTTS2's eleven — are
properties of the model, and the only host that ever supplied them was a test. They now come out of
the export.

**The split, and it is the whole design.** A driver reads no GGUF metadata; `loom` gives it topologies
and host math, not hparams. So:

* a number the **driver** needs is an `ExportConstants` value (P4.0.17's answer, unchanged), bound as
  an IR local, so every read of it goes through `driver_ir.validate` instead of being a runtime `nil`;
* a number the **host** needs — to size an input it must build before `infer` can be called at all —
  is a `loom.<key>` GGUF KV, declared by the new `LoomExportConfig.hparams()` and read back with
  `GgufModel::hparam_u32`/`hparam_f32`. Same namespace `loom::make_kv_cache` already reads its five
  geometry facts from; no new engine code.

Which half a number belongs in is decided by who reads it. Exactly two are host-facing:
`loom.style_dim` (Kokoro — a caller cannot build `ref_s` without knowing how long each half is) and
`loom.txt_len` (Supertonic — every text-touching topology was traced at a fixed length, so any other
count is a model that cannot run). StyleTTS2 deliberately declares **nothing**: it samples its style
vector inside the driver, so a KV there would be one nobody reads.

**Where the numbers actually come from, which turned out to be four different answers:**

| source | examples |
|---|---|
| read off the restored **module** | Supertonic's `lat_dim`/`compression_factor`/`base_chunk_size` (the SpeechDecoder's own attributes); Kokoro's and StyleTTS2's `style_dim`/`d_model`/`hidden_per_dir` (`prosody_dims`, one derivation with two readers — `build_prosody_phases` traces against the same values) |
| read off a **config file** | Matcha's `n_feats` + the state dict's own `mel_mean`/`mel_std`; StyleTTS2's `sigma_data`, out of `config.yml`'s `model_params.diffusion.dist.sigma_data` |
| **baked into the trace**, so declared and cross-checked | Kokoro/StyleTTS2's `gen_istft_n_fft`/`gen_istft_hop`/`upsample_scale`; Supertonic's `T_TEXT`; VITS's `inter_channels` |
| genuinely **not in the checkpoint** | Supertonic's 44100 Hz (a `supertonic_tts.lightning` default, not shipped); piper's three synthesis scales; StyleTTS2's Karras `sigma_min`/`sigma_max`/`rho` |

**Two things this item did not predict.**

* **One family's numbers are knobs, not facts.** VITS's `noise_scale`/`noise_scale_w`/`length_scale`
  are piper's synthesis defaults and `length_scale` is *speaking rate*. Binding them hard would have
  made the driver strictly less capable than before, so they are bound as **defaults the caller may
  override** (`inputs.length_scale or LENGTH_SCALE`) — the model declares them, the host still
  decides. StyleTTS2's Karras parameters got the opposite treatment on the same test: its repo's own
  inference entry point exposes `diffusion_steps` and not those, so making them overridable would be
  inventing an interface rather than preserving one.
* **`sigma_data` forced a new file dependency, and correctly.** It is in neither checkpoint nor
  Kokoro's `config.json`; it is in StyleTTS2's own `config.yml`, which also says
  `estimate_sigma_data: True` — a statistic of the training data, not a constant of the architecture.
  So the export reads that file and raises naming it rather than substituting the LJSpeech value for a
  checkpoint that does not state it.

**A check fell out that had nothing to do with the follow-up.** `_STFT_N_FFT`/`_STFT_HOP`/
`_UPSAMPLE_SCALE` are baked into the traced `decoder_vocoder` graph *and* into the driver's host-side
`compute_wsum`, and nothing verified the checkpoint agreed. `check_istftnet_geometry()` now compares
them against `config.json`'s own istftnet section for both families and raises naming both sides.

**Gates.** Per family: the constants the export emits are compared value-for-value against the
literals `tts_driver_inputs.h` supplied, and the model is re-exported and run through its MIL Lua
driver test with nothing passed for them. Matcha reproduced its frozen waveform exactly
(`max_abs_diff` 0.0104421, rmse 0.000678027 — the numbers in its own header); Supertonic 2.0843e-06;
VITS's two calls (defaults vs. explicit) are **bit-identical**; Kokoro 22208/22208; StyleTTS2
22207/22207. Negative gates, each breaking one thing and watching a real export or test fail:
Matcha's `mel_mean` +1.0 → 0.860849 against a 0.02 bound; Supertonic's `base_chunk_size` halved →
0.197135 against 1e-2; VITS's `LENGTH_SCALE` 1.5 → 0.264348; Kokoro's `_STFT_HOP` 4 → the export
refuses before tracing, naming both sides; StyleTTS2's `config.yml` missing → the export refuses,
naming the file and the key.

**Honest about one limit:** Kokoro's and StyleTTS2's MIL tests have deliberately loose bounds (no
frozen oracle — see their headers), so a perturbed constant would not reliably trip them. For those
two the exact claim is the constant comparison, not a numeric probe, and the commits say so.

Full `ctest` **128/128** with all five MIL GGUFs wired in; exporter suite **480/480**. The four
non-TTS families carry `hparams` through `backend_kwargs()` too and declare nothing, so their exports
are byte-identical by construction — `test_export_hparams.py` walks the registry to check the channel
is actually there, because an override that quietly dropped it would disable the hook for one family
with no other symptom.

### The stranded pre-MIL components — DONE (2026-08-07)

P4.0.8's second follow-up, taken as the decision it was filed as. `cfm_euler_sampler.{h,cpp}`,
`ode_stepper.{h,cpp}` and `style_diffusion_sampler.{h,cpp}` are gone, with the four tests that were
their only consumers (`test_e2e_toy_ode`, `test_style_diffusion_sampler`,
`test_e2e_styletts2_diffusion_sampler`, `test_e2e_supertonic_cfm_sampler`) and the five Python
reference/fixture generators feeding them. Each was orchestration — a host-driven sampling or
integration loop — which is exactly the work `EXPORT-PREPARATION.md` §1.3 records as the exporter's,
and each already had its Lua counterpart on the MIL path.

**The follow-up got one of its four wrong, and the correction is the useful part.** It filed
`bilstm_stepper.h` alongside the others as "a unit test and no product consumer". It has neither: it
has **no unit test at all**, and three real construction sites —
`test_e2e_kokoro_{text_encoder,duration_predictor,f0n}.cpp` build a `BiLstmStepper` to *drive* the
bespoke per-topology check each of those tests exists for. Deleting it deletes those checks, which is
stage C's rule verbatim. Its MIL counterpart (`loom.run_recurrent` + `RecurrentPhase`) has replaced it
in every driver; what keeps it alive is the bespoke conversion path, so it retires with that in P6.
`loom.h` now says so, with the measurement rather than the verdict.

**What the deletions cost, stated rather than glossed.** Three of the four tests were exact numeric
comparisons against Python references, and two of those cannot be reproduced against the Lua
counterpart at all: they replayed fixture noise through an injectable `GaussianSampleFn`, and the
driver draws from `loom.gaussian_array`, which has no such seam. That is a real loss of resolution —
but it is a check *of the code being deleted*, not of anything that ships. Nothing on the MIL path was
covered by them, and StyleTTS2's frozen full-pipeline waveform (`fixtures/legacy_driver_reference/`)
was produced by the retired C++ driver *through* `adpm2_sample`, so the ground truth those tests
established survives one level up, at a looser bound. Supertonic's CFM loop is deterministic given
`z0` and needs no such argument: `test_e2e_supertonic_mil_lua_driver` is a genuine equivalent.

`test_graph_reuse_safety.cpp` is deliberately kept and re-headed. It was written *about* `OdeStepper`,
but what it pins is a property of `ggml_gallocr`, in plain ggml calls, and it is what would catch a
ggml upgrade invalidating `GraphBuilder`'s reason for not needing that discipline.

**Gate:** `ctest` **128/128, 0 failed** (135 before; the seven removed are the four tests plus three
fixture-generator setup steps). Engine size, RelWithDebInfo stripped, same configuration both sides:
**1,248,832 → 1,240,632 bytes, −8,200 (−0.66 %)**; `.text` 1,224,663 → 1,216,923 (−7,740, −0.63 %).
Small, and honestly so — these were four short files, unlike E.4's ~7k lines of driver.

### The bespoke NeMo converters are gone — DONE (2026-08-07)

P4.0.17 step 3, and the end of `tools/convert_nemo/` as a converter directory: only `mel_common.py`,
`nemo_common.py` (both still imported by `convert_generic`/`convert_whisper`) and
`reference_forward_conformer.py` (a real numerical oracle for the MIL encoder) remain.

**A prerequisite surfaced that the plan had not named: the MIL artifact carried no tokenizer.** The
bespoke converters wrote the checkpoint's SentencePiece vocab into their GGUF; the MIL export did not,
so its artifact could not be detokenized. That — not the decode loop — was the last thing keeping the
old converters alive. `extract_nemo_tokenizer_dir` unpacks the archive's `<hash>_tokenizer.model` into
a temp dir and hands it to the exporter's existing `sentencepiece_proto` family, so there is no new
writer and no second vocab schema. Conformer now embeds 1024 tokens, Parakeet 8192, and
`loom_cli --model parakeet-tdt.gguf --wav samples/jfk.wav` prints *"And so, my fellow Americans, ask
not what your country can do for you, ask what you can do for your country."* from one file.

Two things had to move for that:

  * **`MultiPhase.export` never forwarded `backend_kwargs()` to the output exporter**, only to the
    per-phase ones — so a multi-phase family had no way to say anything about its own GGUF. It does now,
    which is what lets Parakeet carry a vocab at all.
  * **`tokenizer_common.py` moved to `loom_mil_compiler/spm_tokenizer_export.py`**, beside the other
    vocab writers. `exporter.py` had been importing it as `convert_nemo.tokenizer_common`, which only
    resolved when `tools/` happened to be on `sys.path` as a package root — it failed the moment the
    export actually tried to use it.

**`loom_cli --wav` is now model-agnostic.** It read the *bare* `model.graph_topology`, computed the
relative-position table host-side and called `loom::ctc_greedy_decode` — all three properties of the
bespoke artifact. It now registers whatever topologies the file declares, calls the driver the file
ships, and detokenizes with the vocab the file embeds. One path for Conformer-CTC, Parakeet-TDT and
Parakeet-RNN-T; `compute_pos_emb` went with it.

**Retired:** `convert_conformer_ctc.py`, `test_e2e_conformer_ctc` and
`test_e2e_conformer_ctc_dynamic_length`. The dynamic-length property they proved is not lost — it moved
to `test_e2e_conformer_ctc_lua_driver`, which already runs 10240, 32000 and 176000-sample inputs through
one artifact, which is the same claim on a wider spread. `test_vocab` is re-based onto the MIL GGUF and
still asserts the same 1024-token vocab, unk id and round trips.

`ctc_decode.{h,cpp}` STAYS, and deliberately: it is not a converter and not a per-model driver, it has
its own unit test, and it is the independent oracle `test_e2e_conformer_ctc_lua_driver` compares the
driver's Lua decode against. Deleting it would remove a check, not dead code.

### Parakeet decodes in Lua, and the C++ transducer decoder is gone — DONE (2026-08-07)

P4.0.17 step 2, complete. `parakeet-tdt` and `parakeet-rnnt` now export as one GGUF holding five
topologies (encoder, embed, two prediction cells, joint) plus a driver that decodes. The TDT double
loop — encoder-frame pointer × symbols-per-frame — is a checked Lua fragment, and
`src/core/tdt_decoder.cpp` no longer exists.

**The oracle changed, and that is what makes this more than a port.** The retired path was gated
against `reference_forward_parakeet_*.py`, a hand-rolled PyTorch reimplementation run on a synthetic
waveform — and for RNN-T that fixture decodes to an *empty token list*, so the test could not tell a
working decoder from a broken one. The new gate is NeMo's own `model.transcribe()` on 11 seconds of
real speech.

**That found a defect in the code being deleted.** On `samples/jfk.wav` the C++ decoder emitted **36**
tokens; NeMo emits **38**. It was dropping two `7877`s — the commas in *"And so, my fellow
Americans,"*. The Lua driver reproduces all 38, and RNN-T's 26, exactly. The migration plan in this
very entry originally said the gate was "36 tokens for TDT" — matching the path being removed would
have preserved the bug and called it a pass.

  | | tokens | matches NeMo |
  |---|---|---|
  | retired C++ decoder | 36 | no — two dropped |
  | Lua driver | **38** | **yes, id for id** |
  | Lua driver (RNN-T) | **26** | **yes, id for id** |

The traced phases were confirmed against the independent PyTorch reference first (`[7618, 1815, 7883]`
on the reference waveform, exactly), which is what isolated the jfk difference to the decoder rather
than to the new traces.

**Retired:** `tdt_decoder.{h,cpp}`, `convert_parakeet_tdt.py`, `convert_parakeet_rnnt.py`,
`reference_forward_parakeet_{tdt,rnnt}.py`, the four `{tdt,rnnt}_step` fixture generators, and six
tests — `test_{tdt,rnnt}_decoder`, `test_e2e_parakeet_{tdt,rnnt}`,
`test_e2e_parakeet_{tdt,rnnt}_mil_export` — replaced by one `test_e2e_parakeet_lua_driver`. ctest drops
from 146 to 137 tests while covering strictly more.

**A checker blind spot fell out of this.** `parse_run_subgraph_calls` scanned only for
`loom.run_subgraph(`, so a fragment using `run_subgraph_and_retain` had that call site *invisible* —
the coverage report said the `joint` topology was named by nobody. A blind spot in the checker reads
exactly like a driver that does not call something, which is the failure this parse exists to prevent.
It has scanned both forms since. P4.0.12 added the retaining form and nothing taught D.2's machinery
about it.

### Parakeet's four traced phases — DONE (2026-08-06); the driver is what remains

P4.0.17 step 2, first half. `parakeet_export.ASRParakeetExportConfig` is a `MultiPhase` config that
traces the whole model rather than just its encoder, which is what makes a Lua driver possible at all.
Verified against the real `parakeet-tdt-0.6b-v3`:

| phase | traced result |
|---|---|
| `encoder` | the existing trace, moved into a named phase (`n_samples`) |
| `embed` | 2 nodes, `last_label` → `[640]` |
| `pred_lstm` | `pred_lstm_l0_fwd`, `pred_lstm_l1_fwd`, 6 weights |
| `joint` | 10 nodes, 2 inputs, **2 declared outputs** |

`blank_id` 8192, `pred_hidden` 640, 2 layers — all read off the checkpoint, and 8192 is the same blank
`test_e2e_parakeet_tdt`'s own `kBlankId` hardcodes.

**A cross-check caught a real error, and it is the reason that check exists.**
`joint.num_classes_with_blank` reads like the token count and is not: for a TDT joint NeMo sets it to
`num_classes + 1 + num_extra_outputs`, so it already counts the durations — **8198, not 8193**. Deriving
the blank from it put it five classes too high and would have split the joint head in the wrong place,
token logits running into the duration ones. No shape check would have caught that, because the widths
still add up. The token count now comes off the embedding, the joint's own width is compared against
tokens + durations, and `test_parakeet_export.py` pins the whole thing.

**What remains before the converter can go**, and one of them is a real design question rather than
typing:

  1. **The driver's constants — SETTLED, and shipped: `ExportConstants`, with the family peeling**
     (author's direction). `blank_id`, the duration set, `pred_hidden` and the layer count are read
     from the checkpoint at export time, and Lua cannot read GGUF hparams. They are bound as ordinary
     IR `Local`s, so every read goes through `driver_ir.validate`. The marker alternative was a
     `str.replace` whose injected text is opaque to every checker: a misspelled read is a runtime `nil`,
     and in Lua `id ~= nil` is quietly true — a TDT decoder emitting every blank as a token, first
     visible as a garbage transcript. Rejecting that shape generally is P4.0.18.
  2. **The TDT loop itself**, as a checked fragment beside `run_bi_lstm`, with the prediction-output
     cache the C++ decoder now has.
  3. **Registry wiring** so `--model parakeet-tdt` selects this config, and re-basing
     `test_e2e_parakeet_{tdt,rnnt}` onto the one-GGUF artifact.
  4. **Deletions**: `convert_parakeet_tdt.py`, `convert_parakeet_rnnt.py`, `tdt_decoder.{h,cpp}` and
     their fixtures.

The gate for all of it is already established and does not depend on the reference fixtures (the RNN-T
one decodes to an empty token list): both checkpoints over `samples/jfk.wav` must reproduce **36 tokens
for TDT and 26 for RNN-T**, ids and frame indices, which is what the current C++ path produces.

### `RecurrentPhase` handles a stacked LSTM — DONE (2026-08-06)

The first unknown in P4.0.17 step 2's traced route, settled by tracing rather than reasoning: **a
`num_layers=2` `nn.LSTM` traces to TWO MIL `lstm` ops**, one per layer, each with its own `[4H, I]`
`weight_ih` — not one op carrying both. `RecurrentPhase` required exactly one and raised otherwise, so
Parakeet's two-layer prediction network could not have been a phase at all.

A stack is now one phase emitting one cell per layer, `<name>_l0_fwd`, `<name>_l1_fwd`, .... A
single-layer module keeps its unsuffixed `<name>_fwd`/`_bwd`, so Kokoro's six BiLSTMs and every
`run_bi_lstm("<phase>", ...)` caller are untouched by stacks existing. One phase rather than N because
the layers share a module, a checkpoint and a name — splitting them would make the caller reassemble
what the module already states.

The test that asserted the old rejection is replaced by one asserting the behaviour, including the half
that would pass silently if the phase emitted two names for one op: each layer carries its own weights,
and layer 1's `weight_ih` is the hidden width beneath it rather than the module's input width.

### Every LSTM computed its gate stack twice — FIXED (2026-08-06)

Found while scoping P4.0.17 step 2, and it turned out not to be a parakeet problem at all.

A cell step produces `h_new` and `c_new` from one gate stack. `GraphTopology` allowed a single declared
output when the pattern was established, so the precedent
(`convert_kokoro_duration_predictor.py::build_lstm_cell_topology`) was to emit the **identical node
list twice** and vary only which output it declared — and every caller then ran both. The gate matmuls,
the four gate VIEWs and the six elementwise ops were computed a second time purely to read the other
half of the same result. P2 added multi-output topologies; nothing went back to collect this.

It is not marginal, and it is not parakeet's: **Kokoro and StyleTTS2 each drive six BiLSTMs over a whole
sequence**, forward and backward, one cell call per timestep per direction. All of it was doubled.

`recurrent.py::_lstm_cell_topology` now declares `["h_new", "c_new"]`, and the output ORDER is the
contract every consumer reads by. Retrofitted across all of them:

* `RecurrentPhase` registers `<phase>_fwd`/`<phase>_bwd` instead of four `_h_*`/`_c_*` names.
* `loom_lua`'s `run_bi_lstm` captures both from one call per timestep per direction; its
  `DrivenTopologies` declaration follows, and `lua_library.drives_mismatches` checks the two agree.
* `loom.run_recurrent` takes ONE module name instead of `(h_module, c_module)`, and reads both outputs
  off one compute. It also gained an error for a topology that declares fewer than two, since the
  ordering is now load-bearing.
* The `export_lstm_test_fixture` GGUF and `test_e2e_lstm_recurrent`'s script.

**Not touched, deliberately:** the bespoke converters (`convert_parakeet_tdt.py`,
`convert_kokoro_duration_predictor.py`, `tools/fixture_gen/tdt_step_common.py`) and the hand-written
`kokoro_driver.lua`/`styletts2_driver.lua`, which carry their own copies of the four-topology
convention. They feed the legacy C++ path (`BiLstmStepper`, `TdtDecoder`) whose GGUFs must stay
byte-identical for their own tests, and they retire wholesale rather than being modernised.

**Gate.** `test_e2e_lstm_recurrent` still matches a real `torch.nn.LSTM(bidirectional=True)` to
**5e-8**. Kokoro and StyleTTS2 re-exported and re-run through their MIL Lua drivers: **22207/22207
checks each**, every waveform sample unchanged. Their topology counts fall exactly as predicted —
Kokoro 39 → 27, StyleTTS2 41 → 29, which is 6 BiLSTMs x 2 fewer apiece — and the driver still names
every one of them (`TestPeeledDriverCoverage`). ctest 146/146, exporter suite 466/466.

### The TDT decoder recomputed its prediction network on every blank frame — FIXED (2026-08-06)

The second finding from the same scoping pass, and independent of the Lua migration, so it lands on
its own.

`TdtDecoder::decode_greedy` ran the whole LSTM stack once per inner iteration. Its output is a pure
function of `(last_label, h, c)`, and all three change **only when a token is emitted** — on a blank the
loop discarded `h_new`/`c_new`, advanced the frame, and recomputed bit-identical values from identical
inputs next time round. Most frames of real audio are blank, so that was the bulk of what the decoder
did. Caching it is what NeMo's own implementation does; the equivalence argument is that the discarded
recompute could not have differed.

The state now commits where it is produced rather than on emission, which is the same condition said
the other way round: reaching the run at all means the state is about to become current.

**Measured on 11s of real speech (`samples/jfk.wav`), parakeet-tdt-0.6b: the prediction stack runs 37
times instead of ~140** — once per emitted token plus the initial one, against once per frame. Decode
wall-clock moves less than that ratio suggests, median **~1.25s → ~1.04s over five runs each**, because
the joint network is the widest matmul in the loop (1024 → 8197) and still runs every frame. This
machine's run-to-run spread is wide enough that the timing is worth little; the call-count is exact.

**Gate — an A/B on the real models, because the existing tests cannot reach the branch.** The
`parakeet-rnnt` reference fixture decodes to an *empty* token list, so `test_e2e_parakeet_rnnt` would
have compared empty against empty and passed either way. Instead, the same binary ran both checkpoints
over `samples/jfk.wav` before and after the change:

  * parakeet-tdt: 36 tokens, identical ids and identical frame indices.
  * parakeet-rnnt: 26 tokens, identical — and this is the branch that matters most, since plain RNN-T
    forces `skip = 0` on emission and therefore re-enters the inner loop, the one path where the cache
    must invalidate rather than persist.

`test_e2e_parakeet_tdt` also still reproduces `reference_forward_parakeet_tdt.py`'s own tokens and
frame indices exactly (16/16, "Yeah.").

### Conformer-CTC now has a Lua entry point — DONE (2026-08-06)

Step 1 of P4.0.17. `conformer-ctc`'s driver is no longer the causal-LM template applied to a CTC model;
it is a real `infer` that returns decoded token ids:

```lua
loom.run_subgraph_and_retain('main_topology', {n_samples = #waveform, n_past = 0}, {...})
local _ctc_frames = loom.argmax_rows('main_topology')
... collapse duplicates, drop 1024 ...
return _ctc_out
```

**The driver no longer needs `src/core/ctc_decode.cpp`** — which is the shape every remaining ASR step
takes. The reduction is engine-side because the logits are; the collapse is in the driver because
blank-and-duplicate handling is this family's convention. A `loom.ctc_greedy_decode` binding would have
kept the same C++ behind a new name, in an engine whose claim is that a family costs Python.

The file itself is still in the tree, and honestly so: `loom_cli --wav` decodes bespoke-converted GGUFs
with it, and this step's own gate uses it as the oracle. It becomes deletable at step 3, when the
bespoke converters go and nothing but the oracle is left.

**One binding, and the argument for it.** `loom.argmax_rows(module)` is `argmax_row`'s plural: one class
id per row, one crossing, logits never marshalled. The singular cannot express this — a frame-wise
classifier has no single interesting row, so a driver would first have to learn `n_frames`, which it can
only do by marshalling the tensor it is avoiding. Passes the P4.0.8 criterion: reads no model config,
and two unrelated families could use it unchanged.

**The builder is now NAMED by the family, not inferred from the decomposition** — the root cause above,
fixed rather than worked around. `ASRNemoEncoderExportConfig.synthesized_builder_key()` returns
`"CtcGreedy"`, and *both* readers take it from there: the exporter (via `backend_kwargs`) and
`component_registry.usage()`. The first attempt keyed off the presence of a `ctc_blank_id` kwarg
instead, and P4.0.7's registry caught it immediately — with selection invisible to `usage()`, the
catalogue still credited conformer-ctc with `argmax_epilogue`, a component it no longer uses. That is
the registry doing exactly what it was built for, on the first change that could have drifted.

**Gate.** `test_e2e_conformer_ctc_lua_driver.cpp` runs the driver's `infer` against
`loom::ctc_greedy_decode` over the same model and asserts token-for-token agreement — an equivalence
against the implementation being replaced, the same shape as P4.0.12's oracle tests. 6/6.

*Honest about the fixtures.* There is no speech recording in this tree, and a trained CTC model decodes
synthetic audio to blank: the reference waveform yields 0 tokens and the best synthetic signal found (a
chirp) yields 1. Neither case is vacuous — an empty transcript is a real check of the blank id, since a
wrong one keeps every frame and returns `n_frames` tokens against the oracle's none — but **the
deduplication rule has no behavioural test**, because that needs a token spanning consecutive frames.
It is pinned instead as emitted Lua text in `test_driver_components.py`, the way every component is.
Closing that properly wants a short speech fixture; worth doing when one exists.

Byte-identity elsewhere: the two Parakeet encoders and every non-ASR model are unchanged (they keep
`argmax_epilogue`; the RNNT pair is step 2). ctest 146/146, exporter suite 466/466.

### Driver logits marshalling caps prefill length for large-vocab models — FIXED

Found while gating P4.0.11a. `MonolithicCall` returns the topology's whole `[n_vocab, n_tokens]` logits
tensor across the Lua boundary so `ArgmaxEpilogue` can argmax a row — and LuaJIT's array part tops out
near 2^27 elements. For Gemma 3's 262144-wide vocab that is **~512 prompt tokens**, past which `infer`
raises `table overflow`; a 600-token prefill is 157M doubles. Qwen3 (151936) caps near 880, LFM2
(65536) near 2048, so nothing on the roadmap has hit it before.

**Fixed on the flattened path** by `loom.run_subgraph_argmax(module, axes, inputs, row)`: the module ran
identically and returned one number, the argmax of the requested row, read from the tensor with `nb[1]`
as the row stride so the other rows were never touched. Nothing crossed the boundary but the answer —
the Lua boundary stays a *per-step* boundary rather than a per-logit one, the same reasoning
`KV-CACHE.md` §1.1 gives for not driving attention from Lua.

Gated on KV-cached topologies only, so the blast radius was the causal LMs: the vocab is what makes the
cap reachable and only that family has one, so no ASR/TTS driver text moved. Gemma's 600-token prefill,
which raised `table overflow`, returned HF's own top-1; Qwen3 and LFM2 agreed with iterated `infer`
22/22 each.

**Then fixed on the modular path too, and the fused call retired — P4.0.14 (2026-08-06).** That entry is
where this ends: the modular chain's last stage retains like every other one, both builders reduce with
`loom.argmax_row(module, row)`, and `run_subgraph_argmax` no longer exists. The one-decision-in-two-
components property survives the move in a stronger form — `MonolithicCall.retained` /
`ArgmaxEpilogue.retained_module` are checked by `driver_ir.check_subgraph_calls`, which knows what a
module is, rather than by `validate` noticing an absent local. The cap is now gated by a test that
prefills past it (`test_e2e_prefill_past_marshalling_ceiling`) rather than by a number in this table.

### SentencePiece-style byte-fallback BPE — DONE

`BpeShape::kSpmByteFallback`, added so Gemma 3 tokenizes correctly. `pre_spec_table()`'s own comment had
already scoped it ("needs a different symbol-initialization step in `BpeVocab::encode()` itself") and
`tokenizer_detect.py` raised a named `NotImplementedError` rather than mis-tokenizing — which is what
made this a bounded job instead of a mystery.

Four structural differences from every other shape, each measured against the real tokenizer: no regex
pretokenization (one chunk); no GPT-2 byte-level mapping, so initial symbols are characters and the
vocabulary holds literal UTF-8; a space→U+2581 normalizer with no dummy prefix (`"Hello world"` →
`['Hello', '▁world']`), and no NFC, because the HF normalizer is that substitution and nothing else; and
`<0xNN>` byte fallback for characters with no entry. `decode` mirrors all of it.

Gated by `test_e2e_spm_byte_fallback_tokenizer` — nine cases, every expectation `AutoTokenizer.encode`
verbatim, all encoding exactly and round-tripping. Gemma now exports with no `--tokenizer-pre` override.
The remaining unimplemented families in `_LLAMA_PRE_TO_LOOM_PRE_TYPE` (CJK-script splitters,
case-transition shapes, cascading-whitespace shapes) are still `None` and still raise by name.

### Sweep after the window routing, the marshalling fix and the tokenizer

12 models from a `git worktree` at `abd6b0a` against the working tree, snapshotted and `diff -r`'d.
**Nine byte-identical** — conformer-ctc, parakeet-tdt, parakeet-rnnt, kokoro, matcha, supertonic, vits,
styletts2, and lfm2-**modular** (unfused, so it has no cache and takes no reducing call).

**Three differ, all fused causal LMs, and in one file each**: qwen3, smollm2 and lfm2-monolithic each
show 11 changed driver lines — `run_subgraph`+`argmax_row` collapsing into one `run_subgraph_argmax` in
both `infer` and `infer_with_past` — plus the `kv.txt` line that records the driver script's own sha.
**`model_graph_topology_main_topology.json` and `tensors.txt` are identical for all three**, which is
the claim worth having: reducing engine-side is a driver change and touches no graph and no weight.

Gemma-3-270m-it is new coverage rather than a diff (it could not export at the baseline, whose
tokenizer support predates the SPM family): 1742 nodes, 18 `ATTENTION`, inputs
`[tokens, cache_position, attention_mask, attention_mask_sw512]`, `tokenizer.ggml.pre =
granite-embed-multi-311m` auto-detected.

### `decomposition`: what `profile` was meant to be, and what `profile` actually does

**The premise this item was written on was wrong, and the correction is the most useful thing in it.**
The original entry claimed `profile` was inert, on the evidence that `LoomGGUFExporter` reads
`self.profile` in one place (`exporter.py`'s bespoke-path dispatch) and only against `None`. That grep
covered `exporter.py` and `register.py` and missed the real users: **`topology_ops.py` reads
`self.profile == "monolithic"` in EIGHT places** (`self` inside a topology rule *is* the exporter), each
gating whether a weight gets a `{func_name}.` namespace prefix — `namespaced_name`,
`gelu_tanh_approx.one`, and six more. Deleting the field, as this item originally proposed, would have
renamed weights in every exported GGUF.

What survives the correction, verified rather than assumed:

1. **The monolithic/modular *dispatch* really is on `modular_layout`**, never on `profile`
   (`exporter.py`'s `export()`). Since P0.1 retired `profile="atomic"`, no dispatch anywhere
   distinguishes `"monolithic"` from `"modular"` by value.
2. **The eight naming reads are all shadowed today.** Every one is `func_name == "main_topo" or
   self.profile == "monolithic"`, and the monolithic path emits exactly one topology, always named
   `main_topo` (`exporter.py:1166`). So for every current caller the profile half never decides
   anything — but it is a live guard, not dead code: a multi-topology export whose exporter was handed
   `profile="monolithic"` would flatten the namespace, and the modular path's deliberate *omission* of
   `profile` is what keeps its per-submodule prefixes.
3. **So `profile` is a real switch wearing the wrong name.** It does not name a profile; it names
   "flatten the weight namespace", which correlates with monolithic-ness without being it.

**What was built.** `decomposition.py` — `Decomposition` with `Flattened`, `Modular(spec, dummy_seq_len)`
and `MultiPhase`, each owning the trace-and-assemble mechanics that used to live in a family's own
`export()`. `LoomExportConfig` gains a `decomposition` field and a single `export()` that delegates to
it; no family overrides `export()` any more. The two causal-LM classes collapse into one
`LMCausalModelExportConfig`, so exporting LFM2 both ways is one type with a field set differently
instead of two types — the concrete thing this item existed to fix. `profile` is gone from
`LoomExportConfig` and now appears only where it is meant: inside `Flattened`-shaped families'
`backend_kwargs()`, with a comment at each site saying it controls weight namespacing.

**Why a strategy object rather than a mode string.** The three forms need genuinely different data
(`Modular` a spec and a non-colliding dummy length; `Flattened` a trace length and quantize mode;
`MultiPhase` the phase list). One config carrying every field with a string selecting which subset is
live makes invalid states representable and pushes checking into `export()`.

**The field is universal; the choice is not.** Only causal-LM currently accepts either decomposition,
because only LFM2 exports both ways from one checkpoint — a caller decision, which is why both its
recognizers deliberately `detect()` the same directory. Kokoro cannot be exported flattened and Qwen3
has no phases: for those families the decomposition is a structural fact, declared once via a
`default_factory` rather than chosen per run. `decomposition.py`'s module docstring states this, so the
next family does not read the uniform field as a uniform menu.

**Follow-up, landed separately: `profile` → `flat_namespace`.** Kept out of the restructure commit so a
rename and a restructure wouldn't share one diff. The flag is now a plain `bool` named for its effect,
read in the same eight `topology_ops.py` rules as `func_name == "main_topo" or self.flat_namespace`, and
passed only by the two `Flattened`-shaped families' `backend_kwargs()`. Its second, unrelated meaning is
gone: the bespoke hand-built-Program path is now decided by `is_bespoke` alone (`len(functions) > 1 and
"main" in functions`), since no caller ever passed a profile to suppress it — both call sites in
`export()`/`_ensure_mil_passes_applied` were checked, and the only two tests that passed
`profile="monolithic"` use single-function programs where `is_bespoke` is already `False`. That is the
one behavioral difference in the rename: a hypothetical caller handing the exporter a multi-function
`main` Program *and* a flat-namespace request would now take the bespoke path. None exists.
`LOOM_PROFILE`, the env override on the old field, had no readers anywhere in the tree and is gone.

#### P4 — flagship coverage

- **P4.1 — Whisper — DONE (2026-08-08).** The only flagship with no MIL export, the project's own
  reference model, and a prerequisite for ~10 models in family 3. First family born inside the registry.

  `tools/loom_mil_compiler/whisper_export.py` + `whisper_driver/00_header.lua`. Two phases —
  `encoder` (457 nodes) and `decoder` (740 nodes, 12 cached `ATTENTION` blocks) — one GGUF, and a
  generated driver that is the encoder once followed by `PrefillDecodeLoop`. Registered as
  `automatic-speech-recognition/whisper` against a real `model_type == "whisper"` check, so
  `loom-export <dir> -o whisper_mil.gguf` needs no `--task`/`--model`.

  **The reusable half, which is what the roadmap actually asked for** (`EXPORT-ROADMAP.md`: "deliver the
  cross-attention decoder loop as the reusable half: it is also all family 6 needs"). It is
  `PrefillDecodeLoop.bound` — `{input name: IR expression}` for a step input that is neither
  host-computed from `n_tokens`/`n_past` nor the step's own tokens. That single field is the whole
  difference between a causal-LM decode loop and a cross-attention one, because nothing else about the
  loop changes: a Whisper step is the same cached call at `n_tokens = 1`, with `xa` bound to the
  encoder's single run. A family-6 text encoder-decoder needs the same field and no new component.

  **Finding — decision 2's fourth `Decomposition` is not needed, and building it is what showed why.**
  `EXPORT-PREPARATION.md` §5 decision 2 reserved a `Decomposition` of its own for this shape, reasoning
  that a new orchestration needs a driver builder the family cannot supply. The orchestration turned out
  to be the one `MultiPhase` already has: N independently traced phases, a component list, and
  `MultiPhaseDriverBuilder`. What genuinely differs is **two facts**, and each is now a field on the
  piece that owns it — `ExportPhase.fuse_attention` (per PHASE, because this decoder must be cached and
  this encoder, from the same checkpoint in the same GGUF, must not) and `PrefillDecodeLoop.bound`. A
  `Decomposition` subclass would have restated `MultiPhase.export` verbatim around those two, which is
  the "declare only what genuinely varies" rule the other three templates are built on.

  **Finding — cross-attention is deliberately NOT fused, and that is correct rather than a miss.**
  `fuse_loom_attention` anchors on the `add(scores, mask)` only a masked block has, and Whisper's
  cross-attention has no mask at all. So the pass fuses exactly the 12 self-attention blocks and leaves
  12 `SOFTMAX`-shaped cross-attention blocks expanded. Fusing them would be wrong twice over: the K/V
  would be the encoder's, identical at every step, and `layer` indices are assigned in dense occurrence
  order, so cached cross-attention blocks would consume the slots the self-attention blocks address.
  What it costs is that cross-attention K/V are recomputed per step — the same thing
  `convert_whisper_decoder.py` did with `kv_cache=false`, so this is parity and not a regression.
  A `cross_kv_cache` that projects once from `xa` is the real optimization, and it needs an engine-side
  cache kind that is filled once rather than appended per step; filed under Engine, not done here.

  **Four holes this item found and closed, each one a place a multi-phase export could not say
  something a single-graph export could:**

  1. **A multi-phase GGUF wrote no KV-cache geometry at all.** `_kv_cache_geometry` walks
     `self.program`, and `MultiPhase`'s output exporter has none — each phase was converted by its own
     exporter and only the finished topologies were handed over. The artifact was therefore unloadable:
     `ATTENTION` nodes with `kv_cache=true` and no `loom.n_layer`/`n_head_kv`/`kv_cache_size` for
     `make_kv_cache` to read. `LoomGGUFExporter.phase_programs` + `_fused_ops()` is one enumeration both
     geometries (KV and conv-state) now read from, and the capacity travels from the phase that declared
     it rather than from a second `backend_kwargs()` entry that could disagree.
  2. **`MultiPhaseDriverBuilder.called_topologies` was keyed on `isinstance`**, so a `PrefillDecodeLoop`
     — which declares `topology` as a `TopologyName` exactly like `SubgraphCallComponent` — was invisible
     to it, and the export log reported Whisper's decoder as "a topology no call site names". It now
     reads the components' own declared links, so a new component is counted the day it declares one.
  3. **The encoder output would have crossed the Lua boundary.** 1.15M floats for whisper-small, read
     once per generated token. `SubgraphCallComponent.retain` + an `OutputRef` binding makes it a
     backend-side tensor copy, which is exactly what P4.0.12 built `OutputStore` for; the driver now
     marshals nothing but token ids.
  4. **Two MIL ops had no ggml mapping**, both from the mel frontend: `clip` (MIL's spelling of
     `torch.clamp`, whose bounds are `alpha`/`beta` INPUTS, so the generic path would have dropped them
     — `OP_MAP`'s existing `"clamp"` entry names an op the torch frontend never emits) and `reduce_max`.
     The latter is composed as `RESHAPE` to one row + `POOL_1D(max)` spanning it, the same reduction
     `convert_whisper_encoder.py` hand-wrote, and only for a *global* maximum over statically-sized axes
     — a per-axis maximum is rejected by name rather than approximated.

  **The mel frontend is in the exported graph.** HF's `WhisperEncoder` starts at `input_features`;
  `WhisperMelFrontend` reimplements `WhisperFeatureExtractor._torch_extract_fbank_features` as a
  traceable module (bit-identical to the extractor — `max abs diff 0.0` — with the checkpoint's own
  `mel_filters`), so the exported model takes a **waveform**. Two things a change here must not lose:
  the input is `(1, n_samples)` because a 1-D one makes `torch.stft` trace an `aten::size`/`aten::Int`
  chain, and the magnitude is taken **before** the final-frame slice, because coremltools' complex
  dialect has no `slice_by_index` over a complex tensor.

  **Finding — this task's declared base config class was already false, for Parakeet, and nothing could
  tell.** `tasks.py` declared `automatic-speech-recognition` as building
  `ASRNemoEncoderExportConfig`, but `_build_parakeet_tdt`/`_build_parakeet_rnnt` have returned
  `ASRParakeetExportConfig` (a `BaseMultiPhaseModelExportConfig`) since P4.0.17 step 2. `register()`
  checks the *entry's* `config_class` and nothing checks what a recognizer's `build_config` actually
  returns, so the declaration was never contradicted. Whisper is a third shape again, so the task's base
  is now `LoomExportConfig` — the honest answer for an I/O-contract task whose families genuinely do not
  share an export shape. **Making the check bite again means moving `config_class` onto the
  `ModelRecognizer`**, where the build happens; not done here, because it touches every family and is
  not Whisper's job.

  **Verification.** `tests/test_e2e_whisper_mil_export.cpp` against HF's own
  `WhisperForConditionalGeneration` (`tools/fixture_gen/reference_forward_whisper_mil.py`), on 30 s of
  deterministic noise — harder than speech, which collapses greedy decoding onto a few high-confidence
  tokens where a wrong logit rarely moves the argmax:

  | check | result |
  |---|---|
  | `encoder` topology (mel frontend included) vs HF's encoder | `max_abs_diff = 0.0132` against a reference whose own absmax is 29.86 — 4.4e-4 relative; `mean_abs_diff = 9.3e-06` |
  | the whole driver — encoder, prefill, cached decode loop, greedy argmax — vs HF | **exact**: `522 8002 45981 8 50257`, including stopping on eos |
  | nothing in the test names a per-model C++ struct | `uses_kv_cache()`, `make_kv_cache(*model)` and `loom.n_samples` are all read from the artifact |
  | qwen3-0.6b-base, lfm2-350m-monolithic re-exported after the `fuse_loom_attention` fix | byte-identical by snapshot diff |
  | the Python suite | 481 passed |

  **Follow-up, done 2026-08-08: the driver owns its prompt, and `loom-cli --wav` runs Whisper.** The two
  open host decisions were put to the author, who answered with a rule rather than a case: *an explicit
  argument wins; absent it, autodetect if the model can; otherwise a documented default* — and that this
  should hold for anything a model can infer about its own input, not just Whisper's language. That
  answer decides **where** the logic goes, which is the more important half: whether detection is
  possible at all is a checkpoint fact, so it belongs in the driver, not the host.

  * **`whisper_export.decoder_prompt_constants`** reads five ids off the checkpoint's own generation
    config (`SOT`, the `LANG_LO`/`LANG_HI` window, `TRANSCRIBE`/`TRANSLATE`, `NO_TIMESTAMPS`) and binds
    them as `ExportConstants`. Which of them *exist* is the capability statement: an English-only
    checkpoint has no language or task tokens, so it gets `0`s and the driver correctly builds the short
    prefix and never tries to detect. Read off the config rather than `forced_decoder_ids`, which leaves
    its language slot `None` on a multilingual checkpoint because HF fills it in from detection.
  * **`whisper_driver/01_prompt.lua`** builds the prompt: given language wins, else one decoder step from
    `SOT` alone with the argmax restricted to the language block, else nothing. It costs no extra encoder
    pass (`xa` is already retained) and writes only KV cell 0, which the prefill then overwrites with the
    identical K/V. So `PrefillDecodeLoop` gained `prompt`, an expression for the loop to start from
    instead of `inputs.tokens` — the third and last field this family added to it.
  * **`loom.argmax_row_range(module, row, lo, hi)`** is the new engine binding it needs. The 99 language
    tokens sit *inside* the 51865-wide transcript vocabulary, so an unrestricted argmax answers with a
    word. It is the same `argmax_tensor_row` with an id window (so the two cannot disagree about how a
    maximum is found) and reads only the window off the backend. A separate binding rather than an
    overload, which is the opposite of `argmax_row`'s own documented choice, for a mechanical reason:
    `argmax_row`'s module form already ends in an optional `generation`, so `(module, row, lo, hi)` and
    `(module, row, generation)` cannot be told apart.
  * **`loom-cli`** grew `--language` (resolved to the model's own `<|xx|>` id by TEXT via the new
    `BpeVocab::piece_to_id`, so the CLI carries no per-checkpoint number) and 30 s chunking. The author
    chose sequential windows over rejecting long audio; the seam is real and documented at the loop —
    each window is decoded independently, where Whisper's own long-form algorithm conditions each on the
    previous window's text and cuts at emitted timestamps.

  Three real bugs surfaced doing it: `Vocab::load` **throws** on a `gpt2` schema while `BpeVocab::load`
  politely returns nullptr, so the SentencePiece loader has to be asked second or it kills the run; the
  driver returns the eos token it stopped on, which decodes literally as `<|endoftext|>` in a
  transcript; and — in the test, not the engine — binding `const auto&` into
  `std::get<...>(bridge.call(...))` reads the vector after the returned variant is destroyed. That last
  one is worth remembering because of how it *presents*: the first one or two elements come back as
  freed-heap garbage and the rest are correct, so it reads exactly like a driver computing a wrong
  prompt. It cost a full bisect to find, and the thing that found it was making the driver print each
  token it computed — which were all correct.

  The driver already accepts `task` and `timestamps` on the same terms; only `--language` is exposed on
  the CLI, since that is the one the author decided.

  **R6 retirement — DONE (2026-08-08), once the blocker was removed.** The four `test_e2e_whisper_*`
  tests were fixtured on `whisper-tiny` in **OpenAI `.pt`** form, which this family cannot load; the
  author downloaded the HF conversion (the `.pt` moved to `openai-whisper-tiny/`), and the retirement
  followed. `src/core/whisper_driver.cpp`, its header, `tools/convert_whisper/` (8 files) and the four
  tests are gone, and so is `include/loom/loom_legacy.h` — all six per-model C++ drivers are now retired,
  so the header that carried the policy has nothing left to carry.

  **The coverage was moved, not dropped**, which is the whole content of the R6 rule. The retired tests
  checked four things; `test_e2e_whisper_mil_export` now checks all four, and against a better oracle —
  `test_e2e_whisper_driver` and `test_e2e_whisper_lua_driver` compared two of *this engine's* own
  implementations with each other, while every check here compares against HuggingFace:

  | retired test | what carries it now |
  |---|---|
  | `test_e2e_whisper_encoder_reference` | check 1: the `encoder` topology vs HF's encoder, mel frontend included |
  | `test_e2e_whisper_decoder_reference` | check 1b, added for this: the `decoder` teacher-forced over the whole prompt at `n_past=0`, logits **and** per-row argmax |
  | `test_e2e_whisper_driver` | check 2: the full loop, vs HF's greedy token sequence |
  | `test_e2e_whisper_lua_driver` | check 2, which *is* a Lua driver run — now compared against HF rather than against the C++ driver it was ported from |

  Check 1b runs through a hand-written Lua script rather than a bare `GraphBuilder`, deliberately: the
  mask and positions then come from `loom.causal_mask`/`loom.range`, the same host math a driver uses,
  instead of a second implementation of Whisper's mask living in a test.

  **`whisper-tiny` also made the gate cheap.** 4 layers at `d_model=384` against small's 12 at 768:
  the same test runs in **38 s** instead of ~7 minutes, so the flagship's end-to-end check is now
  something you can run while working rather than once at the end.

  **Measured, since leanness is the stated goal and P4.0.8's own gate asks for it:** `libloom_engine.so`
  12,846,600 → 12,529,328 bytes, −309 KB, with `whisper_driver.cpp` the only translation unit removed.
  8 Python files, 4 C++ tests and 2 headers went with it.

  **Timestamps: interpreted, and driving the chunking (2026-08-08).** `--timestamps` now prints
  `[hh:mm:ss.mmm --> hh:mm:ss.mmm] text` segments, and the long-form loop advances by the model's own
  segment boundaries instead of a fixed 30 s stride — the seam the naive version documented is closed.

  **The finding that made it work at all: omitting `<|notimestamps|>` from the prompt does not get you
  timestamps.** Left to itself the model *emits* that token as its first output — it outscores every
  alternative on most audio — and then produces none. Confirmed against HF directly, which is also where
  the fix came from: Whisper's rule is that the token after the task **must** be a timestamp, enforced in
  HF by a logits processor. The driver does the same thing with machinery it already had — one prefill,
  then `loom.argmax_row_range` over the timestamp block, which cannot answer with `<|notimestamps|>` or
  with a word. On the same clip that produced no timestamps at all, it then produces
  `<|0.00|> [Motor] <|10.00|>`, and the model has correctly noticed the real audio ends at 10 s.

  That token is fed back in *and* returned, which is why `PrefillDecodeLoop` grew a third field,
  `generated_prefix`: the model chose it, so it belongs in the loop's output even though it is also part
  of the prompt. The host needs no per-model constants for any of this — `<|0.00|>`'s id comes from
  `piece_to_id`, and seconds-per-timestamp is `(n_samples / sample_rate) / n_audio_ctx` off the file's
  own KVs, which is what `sample_rate` was added to `hparams()` for.

  **Two bugs worth recording, both of the same shape — a check that could not fail.**

  1. `TS_HI` shipped as **0**, because `vocab_size` lives on the model config and I read it from the
     *generation* config. The export-time cross-check that should have caught it was written as
     `if ts_lo and ts_hi and ...`, so a zero bound skipped its own verification and the driver silently
     stopped forcing timestamps. It is unconditional now: once a checkpoint has a `<|notimestamps|>` at
     all, the block's width is verified against the encoder's frame count or the export fails.
  2. The chunking loop special-cased the final window — advance a full stride once `avail < clip`,
     "because the rest is padding". But `avail < clip` only means the window is not *full*; the audio in
     it is real. A model that closes at 23 s of a 20..45 s window has transcribed three seconds and
     stopped, and the guard threw the other twenty-two away. Removing it turns one window into three
     (`20 → 23 → 26 → 45`) and the transcript covers the whole file contiguously. What replaces it is a
     one-second floor on the advance, which is a progress guarantee rather than a tuning knob.

  **Context conditioning across window cuts — DONE (2026-08-08), completing the long-form algorithm.**
  Timestamps fixed *where* a window is cut; this is what gives the model context *across* the cut. The
  previous window's text goes in front of `<|startoftranscript|>` behind a `<|startofprev|>` marker, so
  the model reads it as context and then starts its real prompt — at most `MAX_PREV` tokens of it
  (`n_text_ctx // 2 - 1`, half the context minus the marker, Whisper's own budget), keeping the most
  recent, since the words just before a window are the ones that predict its first.

  Two constants (`PREV_SOT`, `MAX_PREV`) and one driver branch; the host accumulates what each window
  generated and hands it over **raw**. Which of those ids count as *text* is the driver's question, not
  the host's, because the driver is what has the constants — it drops timestamps, `<|notimestamps|>` and
  eos, all of which are the model's decisions about the previous window rather than words it said.
  `--no-condition-on-previous` turns it off, on by default as in Whisper's own CLI and switchable for
  the same reason it is there: carried context is what makes a sentence survive a boundary, and it is
  also what lets a repetition loop persist across one, with no temperature fallback here to break out.

  **The check nearly could not fail, and fixing that is the interesting part.** The obvious fixture is
  "condition on this run's own output" — which is what the CLI actually does window to window, so it
  looked right. It changes nothing: the greedy path is already consistent with its own output, so HF's
  conditioned and unconditioned runs were **token for token identical**, and a driver that ignored
  `prev_tokens` entirely would have passed. The fixture now uses an ordinary sentence instead, which
  moves the output completely (` (sad music)` → ` The weather is cold and rainy in the north.`), and the
  test asserts up front that the two references differ. The context handed to the driver also has a
  timestamp, a `<|notimestamps|>` and an eos appended that HF's oracle never saw, so forgetting to
  filter fails too. Both halves verified: 52/52.

  **The CLI's remaining two flags landed here too**, on the same terms as `--language`:
  `--task transcribe|translate` (resolved by text, so a bogus one names the two Whisper has and says an
  English-only checkpoint has neither) and `--timestamps`, which omits `<|notimestamps|>` from the
  prompt so the model may emit timestamp markers. That flag found one more leak: with the token no
  longer forced, the model *chooses* it and emits `<|notimestamps|>` as its first output, which decoded
  literally into the transcript. Control tokens are now dropped before detokenizing — eos and
  `<|notimestamps|>`, both resolved by text — while timestamp markers are kept, since asking for them
  is the point of the flag. `--task translate` is visibly real: the same German audio transcribes as
  `(Lockere Musik)` and translates as `[Water sizzling]`.
- **P4.2 — GigaAM v3 — DONE (2026-08-09).** Graph is family 1 (already exported); the point was the
  *second loader* (`AutoModel.from_pretrained(..., trust_remote_code=True)` instead of
  `ASRModel.restore_from`), which is what proves P3.2's loader/template split.

  `tools/loom_mil_compiler/gigaam_export.py` (~200 lines, over half of it prose) + the transducer
  template it reuses. `v3_e2e_rnnt` exports as four phases — `encoder` (1455 nodes), `embed`,
  `pred_lstm_l0_fwd`, `joint` — in one 851 MiB GGUF, registered as
  `automatic-speech-recognition/gigaam-rnnt` against a real check on the checkpoint's own
  `cfg.model.cfg.model_class`, so `loom-export <dir> -o gigaam.gguf` needs no `--task`/`--model`.
  `test_e2e_gigaam_lua_driver.cpp` decodes `samples/jfk.wav` to the same **80 tokens** the checkpoint's
  own `RNNTGreedyDecoding` emits, and the exported encoder matches PyTorch's to 3.2e-4 over 275 frames.

  **The split the roadmap asked for, drawn where the evidence put it.** `parakeet_export.py` was the
  only transducer, so everything in it read as Parakeet's. Peeling GigaAM out of it found the boundary
  is not between the two models at all — it is between *how a checkpoint is loaded and where it keeps
  its modules* and *everything else*:

  * `transducer_export.py` (new) holds `BaseTransducerExportConfig`: the four phases, their shapes, the
    blank-id derivation, the joint/duration cross-check and the whole component list. Both leaves.
  * `parakeet_export.py` shrank to ~70 lines: `ASRModel.restore_from`, `decoder.prediction`/`joint`, and
    two recognizers.
  * `gigaam_export.py`: `AutoModel.from_pretrained(...).model`, `head.decoder`/`head.joint`, one
    recognizer — plus two workarounds the remote code needs, below.
  * `nemo_asr_export.py`'s `EncoderOutput`, `build_trace` and (renamed) `ASREncoderWrapper` turned out
    to be family-1's encoder template with no NeMo in them at all. GigaAM imports all three unchanged.
    **The entire remaining difference is two strings**: `GigaAM.forward` names its inputs
    `features`/`feature_lengths` where NeMo names them `input_signal`/`input_signal_length`, which is
    now `ASREncoderWrapper.input_names`. The module keeps its name because NeMo is still the only
    loader with a config of its own in it; a third loader is when the encoder half earns its own module.

  `transducer_driver/` (was `parakeet_driver/`) is shared verbatim, and Parakeet's own driver text
  changes by exactly the five header-comment lines that named it — the one intended difference in the
  re-export gate below. `RecurrentPhase` gained `number_layers`, because GigaAM's prediction network is
  **one** layer where Parakeet's is two, and the driver loop addresses the cells by index: the phase
  says "number the layers whatever the depth" rather than the Lua restating `topologies()`' rule.

  **The remote code needed two workarounds, as EXPORT-ROADMAP.md R4 predicted, and neither is the one
  it predicted.** No `.item()`, no Python control flow:

  1. The rotary positional table is a `persistent=False` buffer built lazily on the first forward.
     Built during the trace it becomes graph ops; built under `torch.inference_mode()` it becomes an
     *inference tensor* and the trace dies in autograd. `load_model` builds it eagerly, outside
     inference mode.
  2. **torchaudio's `MelSpectrogram` cannot be converted, for a reason that is not about mel.**
     `torchaudio.functional.spectrogram` reshapes the STFT result back to the input's batch shape, and
     reading `.shape` on a *complex* tensor emits a `complex_shape` op coremltools' own
     `lower_complex_dialect_ops` pass cannot lower. `_TraceableMelSpectrogram` is the same arithmetic
     without that reshape — the same shape of rewrite as `WhisperMelFrontend`, for the neighbouring
     reason. It is not asserted equivalent: every export runs both on a chirp and compares, and they
     agree **bit for bit**.

  **Two real exporter bugs, both found by this model and both older than it.** Neither is GigaAM-shaped;
  GigaAM is just the first checkpoint whose frontend and attention are spelled the way that reaches them.

  * **`square` and `clip` were missing from the shape walk's unary-passthrough set**
    (`exporter._UNARY_PASSTHROUGH_OPS`). `spec.abs()` on a complex tensor lowers to
    `sqrt(square(re) + square(im))`, so an `.abs()`-based magnitude puts a `square` immediately
    downstream of the STFT conv whose frame-count formula that walk exists to derive; and
    `torch.clamp` converts to MIL's `clip`, so the `clamp` entry never matched anything. NeMo spells the
    same magnitude `view_as_real(...).pow(2).sum(-1)` and calls neither, which is why this survived four
    ASR exports. The failure was **not an error**: the walk fell back to the bare root axis, the encoder
    frame count came out as the subsampling formula with the STFT's own `/160` simply missing, and the
    export succeeded. It failed at run time as a rotary-table VIEW asking for 44000 rows of a 10000-row
    constant.
  * **`gather_shape_value` assumed axis 0 is the batch axis.** `x.shape[0]` on a rank≥2 activation
    short-circuited to a hardcoded `1` — a claim about the model's *layout*, and GigaAM's
    `RotaryPositionMultiHeadAttention` is the counterexample: it transposes to `(T, B, H, D)` *before*
    applying the embedding, so `q.shape[0]` there is the sequence length. Read as a batch size it made
    the rotary cos/sin crop `pe[0:1]` — one position wide, which ggml then broadcasts over every frame,
    so **every position got position zero's rotation**. The rule is now about the derivation's
    provenance, the same distinction `scalar_expr_is_guess` draws: trust a real answer, fall back to the
    batch reading only when the walk had nothing better than the bare root axis. A genuine batch axis
    traces as a literal `1` and never reaches this code at all.

  **The second bug is the one worth remembering, because nothing structural could have caught it.**
  The graph built, ran, produced the right output shape, and decoded 71 of the first 80 tokens
  correctly — a plausible transcript that was simply wrong. Only dumping the exported encoder's own
  output and comparing it against PyTorch's found it (max abs diff 2.35 on a scale of 2.29), and only
  bisecting the encoder into prefixes — static vs. dynamic axis, mel / subsampling / attention /
  convolution — localized it to one slice in one op. **Token-level agreement is not a substitute for a
  tensor-level oracle**; a transducer decode is robust enough to absorb a badly wrong encoder and still
  look about right.

  **Not claimed:** GigaAM v3 ships five variants and only `e2e_rnnt` is on this machine. The recognizer
  requires `model_class == "rnnt"` rather than claiming every `model_type == "gigaam"` directory, so a
  `ctc`/`e2e_ctc` checkpoint fails detection with the candidate list instead of being exported by a path
  nothing here has run. Those are family 1's `Flattened` shape (`ASRNemoEncoderExportConfig` with
  `CTC_LOG_PROBS`) plus this loader — a small addition, and an untested one until a checkpoint exists.
- **P4.3 — composition template — DONE (2026-08-09).** The audio-encoder + projector + causal-LM family
  (`EXPORT-ROADMAP.md` R5's family 3), the largest group on the roadmap: ~19 converters, ~36 models.

  `tools/loom_mil_compiler/speech_lm_export.py` (the template) + `qwen3_asr_export.py` (~180 lines, the
  loader) + `speech_lm_driver/00_header.lua`. Four phases — `encoder` (793 nodes: mel frontend, chunked
  conv stem, window attention, projector), `embed`, `decoder` (2286 nodes, 28 cached `ATTENTION`
  blocks) and `lm_head` — in one 3.1 GiB GGUF, registered as `automatic-speech-recognition/qwen3-asr`,
  so `loom-export <dir> -o qwen3_asr.gguf` needs no `--task`/`--model`.

  **Acceptance: `test_e2e_qwen3_asr_mil_export.cpp` against HF's own `Qwen3ASRForConditionalGeneration`
  on `samples/jfk.wav`.**

  | check | result |
  |---|---|
  | `encoder` topology (mel frontend + projector included) vs HF's `get_audio_features` | `max_abs_diff = 2.2e-05` against a reference whose absmax is 0.128 — 1.7e-4 relative; `mean_abs_diff = 3.7e-07` |
  | the whole driver — encoder, segmented prompt, cached decode loop — vs HF's greedy sequence | **exact**: all 30 tokens, including stopping on `<|im_end|>` |
  | nothing in the test names a per-model C++ struct | `uses_kv_cache()`, `make_kv_cache(*model)`, `loom.samples_per_chunk`/`frames_per_chunk` all read from the artifact |
  | whisper-tiny, conformer-ctc-small, qwen3-0.6b-base re-exported after the shared-exporter fixes | byte-identical by sha256 |
  | the Python suite | 503 passed |

  **The finding that made this cheap: the prompt needs no concatenation anywhere.** "Inject audio
  embeddings into the prompt" reads as though something must build one `inputs_embeds` out of text
  embeddings and audio embeddings, which would need a backend-side concatenation of two retained
  tensors — an engine op that does not exist and one `OutputStore` has no shape for. It is not needed.
  Attention is causal and the decoder is KV-cached, so a call at `n_past = k` over `n` rows writes cells
  `[k, k+n)` and attends over `[0, k+n)`: feeding a prompt as N successive cached calls is *the same
  arithmetic* as feeding it concatenated. Measured against HF before any component was written —
  segmented and concatenated prefill agree to 2.3e-04 on hidden states whose absmax is 95.7, and pick
  the same first token. `PromptSegments` is that walk, and it is the whole of the "embedding-injection
  driver" the roadmap asked for. It stops one segment short and leaves the running `n_past` for
  `PrefillDecodeLoop.initial_n_past`, so the final text segment is the loop's own first iteration —
  exactly as a plain causal LM's prefill is.

  **`PrefillDecodeLoop` grew three fields and no new component**, the same outcome P4.1 reached:
  `embed_topology` (the step's tokens reach an `inputs_embeds`-traced decoder through the embedding
  graph, bound by `OutputRef`, so a token id is still all that crosses the Lua boundary),
  `head_topology`, and `initial_n_past`. All default to None/0, so every earlier family emits
  byte-identical driver text — which the re-export gate above checks rather than assumes.

  **The head is split off for a cost reason.** A family-3 prompt is dominated by audio rows — 143 of
  this one's 158 — and a head inside the decoder graph would project every one through a 151936-wide
  vocabulary that nothing reads (22 GFLOP). `lm_head` is its own phase, run only where a token is
  needed.

  **The encoder had to be rewritten, and the rewrite is bit-identical.** HF's `Qwen3ASREncoder.forward`
  is untraceable twice over: it packs valid frames with `valid_mask.flatten().nonzero()` (a
  data-dependent output shape) and runs attention per window via `torch.split(q, lengths.tolist())`
  (Python-level lengths that `torch.jit.trace` bakes in). Both dissolve into one observation —
  `get_audio_cu_seqlens` cuts full `block`-sized windows plus a shorter final one, and attention runs
  independently inside each, which *is* a block-diagonal additive mask. With every frame valid the
  packing step is the identity. Verified in torch before anything was exported: `max abs diff 0.000e+00`
  against HF's eager path (6.3e-05 against sdpa, which is that fused kernel's accumulation order).

  **The contract this imposes**: the waveform is a whole number of chunks, all valid — one chunk is
  `hop_length * n_window * 2` = 16000 samples, one second — so a host pads up to the next second. That
  is what makes "every frame valid" true, and it also makes the checkpoint's own feature extractor an
  exact oracle, since its mel-axis right-pad becomes a no-op on such a waveform. The cost is that up to
  one second of trailing silence becomes real audio embeddings the LM reads, where HF would have masked
  them out. Trimming them needs a way to feed a *prefix* of a retained tensor: **P4.3d** below, open,
  and shared with the second leaf.

  **Five bugs in shared exporter machinery, four of which fail silently.** None is Qwen3-ASR-shaped;
  it is the first checkpoint whose encoder is spelled the way that reaches them.

  1. **A conv's batch axis was hardcoded to `1`** in the shape walk — the same correction P4.2 made to
     `gather_shape_value`, one layer down. This encoder folds the CHUNK COUNT into the batch axis so the
     conv stem sees a fixed 100-frame window per chunk, and reading that as 1 did not fail: it made the
     post-stem sequence length 13 instead of 13 *per chunk*, surfacing hundreds of ops later as a mask
     whose two sides had different lengths. Conv preserves batch, so it now recurses; a genuine
     batch-of-1 still resolves to a literal 1 through the recursion, which is why all three re-exports
     are byte-identical.
  2. **`slice_by_index` was missing from the shape walk.** Whisper's frontend drops the final STFT frame
     the same way (`(stft.abs() ** 2)[..., :-1]`) and never needed it, because its clip is always 30 s
     and every dim downstream is a literal. Without it the walk returned `-1` — not an error anywhere,
     just a wrong number that reached a `POOL_1D` span as `-128`.
  3. **`reduce_max` rejected a dynamic global maximum.** `POOL_1D`'s `k0`/`s0` are already read through
     `resolve_attr_int(..., pc.symbols)` on the engine side, so a span known only once the axes are
     bound is as good as a literal one; this was exporter-side only. Whisper's clip is static and folded
     to a constant, and a variable-length mel frontend is what needs the symbolic form.
  4. **`MultiPhaseDriverBuilder.called_topologies` could not see through `WhenSet`.** It reads
     `TopologyName` off the declarations precisely so a new component is counted the day it declares its
     link — but an OPTIONAL topology field is `WhenSet(TopologyName())`, whose wrapper is not a
     `TopologyName`, so the field was checked and yet invisible. It reported `embed` and `lm_head` as
     "topologies no call site names" while the loop was calling both every step. Same failure P4.1 fixed
     for `PrefillDecodeLoop`, one wrapper deeper.
  5. **`driver_ir.Len` only accepted a local's name**, so `#inputs.waveform` had no spelling. It now
     takes either and delegates `reads()`, which is what keeps `validate()` honest: the string form
     reports the local, the expression form reports what the expression reads, rather than a dotted path
     that is not a symbol at all.

  **The two that cost the most were both silent successes, and both are worth remembering.**

  * **Forcing `attn_implementation="eager"` in the loader disabled the KV cache.** It looks harmless —
    `WindowedAudioEncoder` reimplements the tower's attention and never calls the tower's own — but it
    also reaches the LANGUAGE model, and `fuse_loom_attention` matches the sdpa shape. Under eager the
    decoder converted with **zero** fused `ATTENTION` nodes: no cache, no `n_kv` mask retyping, and all
    four phases still exported "successfully". It surfaced only at run time, as a mask sized 143×143
    where the cache wanted 143×152. Eager belongs on the *reference*, which is where it now is.
  * **`cache_position` was pruned from the decoder graph.** Handed `inputs_embeds` and an already-built
    4D mask, this model consumes it nowhere — it exists to derive position ids and to build a mask, and
    both jobs were done — so the trace dropped it, folding the rotary embedding to the eight positions
    the trace ran at. Every call at a different `n_past` would have rotated by the wrong angle. Caught
    by the exporter's own "supplies an input it does not declare" link; the fix is to pass the positions
    the model actually indexes with (`position_ids`).

  **Two ggml-shaped constraints the graph had to be written around**, both in the window mask:

  * ggml's elementwise ops repeat `b` into `a` and cannot do a **two-way broadcast**, so the natural
    `window.unsqueeze(1) == window.unsqueeze(0)` aborts in `ggml_sub` inside `equal`. `.expand(T, T)` on
    both operands is the obvious repair and does **not** work — MIL's `equal` broadcasts natively, so
    coremltools folds the expands away and re-emits the same broadcast. An **outer product against a
    vector of ones** survives, because a `matmul` is not a broadcast and nothing rewrites it into one.
  * That ones vector must be built by arithmetic (`window * 0 + 1`), not `torch.ones_like`, which
    converts to a MIL `fill` whose length the exporter resolves through the fill's own shape input — and
    it resolved to a *different expression for the same quantity*, so the two sides of the comparison
    disagreed about T.

  **The registry now skips a family whose optional dependency is absent**, loudly and only on
  `ImportError`. Families carry mutually incompatible optional dependencies — `nemo_toolkit` pins
  `transformers~=4.53`, and `qwen3_asr` first ships in **5.13** — so there is no single environment that
  imports all of them, and an eager import of every module made the registry only as usable as its least
  installable member: exporting Qwen3-ASR failed on `No module named 'kokoro'`, from a family the caller
  had not asked for. A failed *detection* now names the families that were not loaded, so "unrecognized"
  and "unloadable" stay distinguishable.

  **Not claimed:** only `Qwen3-ASR-0.6B` is on this machine and only the **`-hf`** repo works. Qwen ships
  the same weights twice, and the native `qwen-asr` layout (`thinker.*` weights, sub-configs under
  `thinker_config`) declares the identical `model_type` while transformers reads class defaults off it
  *without raising* — a 1024-wide/24-layer encoder instead of this checkpoint's 896/18. `detect()`
  requires `audio_config` and `text_config` at the top level for that reason, so the native layout fails
  detection with the candidate list rather than being exported as a plausible wrong model.

  **P4.3b — Voxtral-Mini-3B as a second leaf — DEFERRED, and the reason is this machine rather than the
  template.** It was the obvious second leaf (a Whisper encoder, a 4-frame-stack projector, a Llama LM)
  and it is the one member that does not fit here. Measured, so a machine with more memory can pick it
  up without re-deriving any of it:

  | measurement | value |
  |---|---|
  | parameters | 4.68B (`model.safetensors.index.json`'s own `total_size / 2`) |
  | F32 artifact it would write | **18.7 GB** |
  | fp32 checkpoint, resident | **18.7 GB** (peak RSS 20.6 GB; the excess is mmap'd checkpoint pages) |
  | fp16 checkpoint, resident | **9.36 GB** (peak RSS 18.8 GB — RSS double-counts the 9.35 GB of file pages, which are evictable) |
  | `torch.jit.trace` of the 3.6B LM phase | **free** — 18.8 GB before and after, at fp16 |
  | `ct.convert` on the LM phase | ~14.4 GB inferred from the parameter count, **not measured** — every probe died at or before this call |
  | this machine | 28 GB available, **no swap** |

  `MultiPhase.export` holds the torch model, every phase's converted MIL program *and* the merged
  weights simultaneously, so the peak is a sum rather than a max: ~35 GB against 28.

  **Two routes were measured and closed.**

  * **Half precision does not work, and not for a memory reason.** Tracing a Llama-3B at fp16 on CPU
    succeeds — the risk that half kernels would be missing did not materialise — but `ct.convert`
    refuses the resulting graph. With fp16 *inputs* it wants a `minimum_deployment_target >= iOS16`;
    supplying one, and separately declaring fp32 inputs over fp16 weights, both fail with an internal
    `TypeError: only 0-dimensional arrays can be converted to Python scalars`. coremltools 9.0 with
    torch 2.8 will not take a half-precision traced graph through its torch frontend.
  * **The `quantize=` argument cannot help**, though the mechanism is real and proven
    (`test_e2e_qwen3_q8_0`, `test_e2e_lfm2_q8_0`). It is a *serialization*-time transform: the loop in
    `write_gguf` iterates `self.weights`, which by then holds every phase's f32 array, and it
    `astype(np.float32)`s each one before deciding whether to quantize. It shrinks the artifact —
    Voxtral would write ~9.6 GB at F16 or ~5 GB at Q8_0 — and changes peak memory not at all. Disk was
    never the binding constraint.

  **What would actually move it** is P5's per-phase process isolation below. Even then the floor is
  ~29 GB (the LM phase alone needs its 14.4 GB of fp32 weights and their 14.4 GB of MIL constants to
  coexist), so a bigger machine is the honest answer, not exporter surgery.

  **Granite Speech 4.0.1b is the second leaf instead**, and it was a better test of the template than
  Voxtral would have been: a conformer encoder over 160-bin features and a **Q-Former** projector
  (`num_queries = window_size // downsample_rate`), where Voxtral is a Whisper encoder and a linear
  stack — so it varies both halves the template claims to abstract, not one. Done, in P4.3c below.

  **P4.3c — Granite Speech 4.0.1b, family 3's second leaf — DONE (2026-08-09).** A 16-layer conformer
  over 160-bin features (Shaw relative attention in blocks of 200 frames), a **BLIP-2 Q-Former** that
  turns each 15-frame window into three query rows, and a 40-layer Granite causal LM (2048-wide, GQA
  16/4) — 2.31B parameters in one 8.75 GB artifact, registered as
  `automatic-speech-recognition/granite-speech`.

  `tools/loom_mil_compiler/granite_speech_export.py` (~330 lines, the loader plus the one rewritten
  encoder) + `tests/test_e2e_granite_speech_mil_export.cpp` +
  `tools/fixture_gen/reference_forward_granite_speech_mil.py`. The template gained **one hook with a
  default** (`mel_frontend`) and **one shared helper** (`split_prompt_on_audio`), and nothing else.

  **Acceptance: `test_e2e_granite_speech_mil_export.cpp` against HF's own
  `GraniteSpeechForConditionalGeneration` on `samples/jfk.wav`.**

  | check | result |
  |---|---|
  | the 160-bin features `LogMelFrontend` + the pair-stack produce, vs the checkpoint's own extractor | **`max_abs_diff = 0.000e+00`** — bit-identical, in torch, before anything was exported |
  | `ConformerQFormerEncoder` vs HF's `get_audio_features`, in torch | `max abs diff 5.8e-06` against a reference whose absmax is 0.999 |
  | the exported `encoder` topology vs the same tensor, through ggml | `max_abs_diff = 7.3e-04` against a reference whose absmax is 0.999 — 7.3e-4 relative; `mean_abs_diff = 3.7e-06` |
  | the whole driver — encoder, segmented prompt, cached decode loop — vs HF's greedy sequence | **exact**: all 24 tokens, stopping on `<\|end_of_text\|>`; `"and so my fellow americans ask not what your country can do for you ask what you can do for your country"` |
  | the driver, the phase list, the component list | **unchanged from Qwen3-ASR's** — no new component, no new field |
  | kokoro, whisper-small, lfm2-monolithic re-exported against a `git archive HEAD` baseline | byte-identical, every snapshot file |
  | qwen3-asr re-exported | differs in `model.driver_script` **and nothing else** — every topology and every tensor identical. It is the model that makes this gate able to fail, and the difference is exactly the comment header rewritten now that `speech_lm_driver/00_header.lua` serves two leaves |
  | the Python suite | 503 passed |

  **The claim this leaf exists to test, and it held.** The two leaves share *no* encoder and *no*
  projector: a Qwen3-Omni window-attention stack over one-second chunks of 128-bin mel and a two-layer
  linear projector against a conformer over twelve-second chunks of 160-bin features and a Q-Former.
  What they share is the log-mel frontend and `(samples_per_chunk, frames_per_chunk)` — the contract
  P4.3 wrote down before there was a second member to check it against — and that turned out to be
  enough for `PromptSegments` and `PrefillDecodeLoop` to serve both unmodified.

  **`audio_geometry()` returns `(192000, 120)`, and both numbers are forced rather than chosen.**
  Encoder frames must be a multiple of `context_size` (200, the conformer's blocks) **and** of
  `window_size` (15, the Q-Former's), so the chunk is `lcm(200, 15) = 600` encoder frames = 1200 mel
  frames = **192000 samples, 12 s**; the rows are `600/15 × 3 = 120`. Twelve seconds is coarse — a host
  pads up to it — and it is a property of this checkpoint's two block sizes, not of the family, which
  is exactly why the template publishes the pair rather than a duration. `phases()`' existing
  cross-check verified it against the traced encoder rather than trusting the arithmetic.

  **The mel frontend needed two constructor arguments and no new class.** Granite's extractor is a
  torchaudio `MelSpectrogram` (n_fft 512, **win_length 400**, hop 160, 80 mels) followed by
  `clip_(1e-10).log10_()`, a global `amax`, `maximum(x, mx - 8)` and `.div_(4).add_(1)` — and
  `x/4 + 1` *is* Whisper's `(x + 4)/4`. So `LogMelFrontend` needed `win_length` decoupled from `n_fft`
  and a filterbank read off a `MelScale` instead of an extractor attribute; `mel_frontend()` is that
  three-line override, and it has a default because every other family-3 extractor spells it Whisper's
  way. **The final-frame drop turned out to be Granite's too**: HF computes `L // hop + 1` frames and
  drops the last only when the count is odd, and under the chunk contract that count is `1200k + 1`,
  always odd — so Whisper's unconditional `[..., :-1]` removes the identical frame. That is why the
  features are bit-identical rather than merely close, and it is what makes the checkpoint's own
  extractor an exact oracle.

  **The encoder rewrite is of two `math.ceil`s, and two ggml shapes.** `num_blocks =
  math.ceil(num_features / context_size)` with a `remainder`-driven right-pad
  (`GraniteSpeechConformerAttention`) and `nblocks = math.ceil(seq_len / window_size)`
  (`GraniteSpeechEncoderProjector`) are Python-level and bake into the trace — the same wall Qwen3-ASR's
  encoder hit, and the same fix: require the chunk contract so every remainder is zero, and spell the
  block count `reshape(-1, block, ...)`. Two more things had to be written around the backend:

  * **Shaw's relative-position term is a 5-D einsum, and ggml tensors are 4-D.**
    `einsum("b m h c d, c r d -> b m h c r")` becomes a batched matmul over the *query position* axis,
    which contracts the same index with every operand 3-D or 4-D.
  * **The Q-Former's learnable query is `(1, num_queries, hidden)` broadcast against N windows** — a
    two-way broadcast ggml's elementwise ops cannot do. An **outer product against a ones column**
    materializes one copy per window, the same trick and the same reason as `WindowedAudioEncoder`'s
    window mask. The Q-Former's two attention masks are dropped entirely rather than built: HF makes
    them with `torch.ones(...)` over a dynamic batch and then `(1 - mask) * -10000`, which is
    identically zero, so building them would put a MIL `fill` over a dynamic extent in the graph for no
    arithmetic at all.

  **The logits are Granite's, not `lm_head`'s.** `GraniteForCausalLM.forward` ends with
  `logits / logits_scaling` (8.0) *outside* the head, so exporting `lm_head` alone drops it — and the
  driver would never notice, because it takes an argmax and a positive divisor cannot move one. A host
  that sampled would. `_ScaledLMHead` puts it back; the phase is 3 nodes instead of 2.

  **The export did not fit, and fixing that is P5.0's first item** — recorded there in full. Short
  version: peak RSS **30.4 GB**, OOM-killed in the `lm_head` phase, on a 33 GB machine; now **22.9 GB**.
  The finding was that the torch module and the converted MIL program are the *same* arrays, so
  releasing either alone frees nothing (0.2 GB, measured) and `MultiPhase.export` has to release the
  traced module, the wrapper and the program together.

  **Not claimed:** only `granite-speech-4.0.1b`. `detect()` requires `model_type == "granite_speech"`
  **and** `has_lora_adapter == false`, because Granite Speech ships variants whose language model is
  only correct with a LoRA adapter merged in for audio — this exporter traces base weights, so on such
  a checkpoint it would produce a model that runs and transcribes badly, and a candidate list is a
  better answer than that. The two other Granite Speech checkpoints on this machine declare
  `granite_speech_plus` and `granite_speech_nar`, neither of which transformers 4.57.6 has a module
  for at all; `-plus` has the identical block geometry and is the obvious third leaf the day its
  module ships.

  **P4.3d — a prompt segment can be a PREFIX of a retained tensor — DONE (2026-08-09).** Both encoders
  require the waveform to be a whole number of chunks with every frame valid — that is what makes HF's
  `nonzero()`-packing and its `ceil`-padding collapse into identities, which is what makes them
  traceable at all. So a host zero-pads up to the chunk boundary, and the encoder emits **real**
  embedding rows for that padding, which the LM read as speech. HF masks them out
  (`input_features_mask`); the driver had no way to, because the encoder's output deliberately never
  crosses into Lua (P4.0.12) and there was no spelling for "these rows of it".

  Three pieces, one per layer:

  1. **`{from = 'm', rows = N}`** in `set_tensor_from_output_ref` — `copy_row_prefix` builds a
     `ggml_view_2d` over the retained tensor and hands it to `ggml_backend_tensor_copy`, so the prefix
     reuses ggml's own copy strategy rather than restating its three branches. A view needs
     `ggml_backend_view_init` before it has a buffer at all; without that the copy dereferences null.
  2. **`OutputRef.rows`**, an *expression* rather than an int, with `reads()` delegating to it — the
     `Len` lesson from P4.3: a reference this class did not report is a read `validate()` never
     resolved.
  3. **`PromptSegments.audio_rows`** replaces the `(samples_per_chunk, audio_rows_per_chunk)` pair, and
     the component feeds the SAME local as the segment's `n_tokens` axis and as `rows`. That is what
     makes the two impossible to disagree about: a mismatch is a shape error the engine raises, where a
     segment that merely asked for the wrong length would be a silently short prompt.

  **The host says how long its real audio was, or says nothing.** `inputs.audio_samples` falls back to
  `#inputs.waveform`, so a caller that omits it gets exactly the behaviour this family had before — and
  `rows` then equals the whole retained tensor, which check 5b of
  `test_lua_bridge_retained_outputs.cpp` pins as bit-identical to the untrimmed copy.

  | check | result |
  |---|---|
  | a prefix of a retained output vs the same rows sliced out of the marshalled table | identical, at two row counts; and *not* equal to the same number of rows taken from the tail |
  | `rows` = everything retained vs no `rows` at all | identical |
  | the four ways to get it wrong (past the end, zero rows, disagreeing with the consumer's axis, non-2-D) | all raise, naming the real problem — 10 error cases now, was 7 |
  | Granite's rendered Lua row formula, evaluated in **luajit**, vs `_get_num_audio_features` | identical at all sixteen lengths |
  | the driver with `audio_samples` = the whole 192000-sample chunk | the untrimmed sequence, token for token |
  | the driver with `audio_samples` = 32000 (2 s of the 12 s chunk) | `"and so my fellow americans ask"` + eos — the first two seconds and nothing else, from 21 of the retained 120 rows |
  | kokoro, whisper-small, lfm2-monolithic re-exported | byte-identical, every snapshot file |
  | qwen3-asr re-exported | two lines of driver text, both in the audio segment: `inputs.audio_samples or #inputs.waveform`, and `rows = _seg_tokens` on the `{from = 'encoder'}` reference. Every topology and every tensor identical, and the row count is the same number it always computed — this leaf keeps the padded default |
  | the Python suite | 503 passed |

  **The row count is a per-checkpoint formula, and the template's default is the padded one.**
  Granite Speech's is `ceil((L // hop + 1) // 2 / window) * num_queries`, i.e.
  `_get_num_audio_features` line for line, **checked against the extractor at sixteen lengths**
  including both sides of a chunk boundary and lengths shorter than one hop. Qwen3-ASR's is
  deliberately **not** overridden: its count comes from three stride-2 convs over a final partial chunk
  whose valid mel-frame count the extractor derives from a mask with its own padding rules, and a
  closed form over the sample count disagreed with it at 5 of 12 probe lengths. Its chunk is one second
  (≤13 rows), so it keeps the padded default rather than a formula that is wrong at the edges.

  **What this does NOT fix, measured rather than assumed.** Trimming makes the LM stop reading the
  silence rows; it does not make the retained rows equal HF's. On a 13 s clip padded to 24 s, the first
  132 rows of our padded encoder differ from HF's own 132 rows by **max 3.9e-01** against a reference
  whose absmax is 0.992 — because HF's conformer crops back to the real frame count after every
  attention block and masks its final partial block, where ours runs the whole padded sequence
  unmasked. Greedy decoding was unmoved (all three of HF-unpadded, HF-padded and ours-trimmed produced
  the identical transcript), which is why this was worth doing as a prompt-level fix and is not yet
  worth doing as an encoder rewrite. That rewrite is **P4.3e**.

  **P4.3e — the encoder's own padding handling — NOT STARTED.** The other half of P4.3d, and the
  larger one numerically. To make a family-3 encoder agree with HF on audio that does not fill its last
  chunk, the encoder phase needs the valid length as an input and has to build its own validity masks
  in-graph — Granite's per-block conformer mask and its post-attention crop, Qwen3-ASR's frame packing.
  Both are the things the chunk contract exists to avoid, so this is a real project, not a follow-up
  edit. The measurement above (3.9e-01 on rows the model does read, no token difference) is the
  argument for filing it rather than doing it now.
- **P4.4 — KV cache in MIL-exported causal LMs — DONE, as P4.0.9.** Kept as a stub because
  `EXPORT-ROADMAP.md:129` points here. This row's full text — the measurement that `FuseLoomAttention`
  was the blocker, and a four-step plan — is superseded by [`KV-CACHE.md`](KV-CACHE.md) and P4.0.9's
  entry, which record what was actually built and where the plan was wrong (step 2, `use_past` tracing,
  was **not needed**: once the SDPA subgraph is an `ATTENTION` node the engine supplies the past
  itself). Two remainders are live items of their own, P4.0.10 and P4.0.11.

  **What survives that P4.0.9 does not say.** `EXPORT-PREPARATION.md` decision 2 routes this gap to the
  cross-attention AR decode `Decomposition` in P4.1, and that is still the right home — but the
  prerequisite reading has inverted. It was "that decomposition cannot start here, the fusion pass comes
  first"; the pass now exists, `infer_with_past` exists, and `whisper_driver.lua` has been doing exactly
  this orchestration by hand since the Lua port (`KV-CACHE.md` §1.2). P4.1's decomposition is now the
  *reuse* of a solved shape, not a blocked one.

#### P5 — breadth

Ordered by coverage-per-effort, subject to P0.3's corrections: family 12 (BERT token classifiers —
smallest possible template, proves the registry on a non-audio task) → 11 (codec decoders, unlocks
family 10's back half) → 4 (CNN+CTC) and 5 (SANM), both family-1-shaped once the encoder template is
generalized past NeMo → 9/10 (remaining TTS) → 6 (text enc-dec) → 13 (small classifiers) → 14 (music).

- **P5.0 — per-phase process isolation for the conversion step.** Not a family: the thing that decides
  which models can be exported *at all* on a given machine.

  `MultiPhase.export` makes peak memory a **sum** where it should be a **max**. It loads the whole
  checkpoint once, then for every phase held the converted MIL program (`traced_programs`, kept so
  `_fused_ops()` could read the KV geometry back), and `merge_phase_weights` assembles every phase's
  weights before `write_gguf` starts. A four-phase model therefore held four programs, one torch model
  and one merged weight dict at the same moment.

  **Change 1 is DONE (P4.3c), and doing it taught the accounting.** `MultiPhase.export` now extracts
  each phase's fused-node geometry as it converts (`LoomGGUFExporter.fused_geometry()` → the output
  exporter's `phase_geometry`) and then drops **three things together**: the traced module, the phase's
  `wrapper`, and the converted MIL program. Measured on Granite-Speech-4.0.1b (2.31B parameters, four
  phases): peak RSS **30.4 GB → 22.9 GB**, which is the difference between OOM-killed in the `lm_head`
  phase and a clean export on this 33 GB machine.

  **The three had to go together, and that is the finding.** Dropping only the torch half — the
  obvious first try, since a phase's parameters are plainly dead once its topology exists — moved the
  peak by **0.2 GB** (30.25 → 30.41), i.e. not at all. A converted MIL program's constants are the same
  arrays as the module's own tensors, so the program pinned every phase's weights for the sake of the
  few integers `_kv_cache_geometry` reads. `MultiPhase` no longer keeps programs at all, and
  `_fused_ops` is back to meaning "this exporter's own program".

  What remains resident is `exporter.weights` — the real per-phase copies the writer needs — which is
  what changes 2 and 3 below address. The per-phase numbers now print beside the node counts
  (`decomposition._rss_note`), so the next attempt starts from a measurement rather than an estimate:
  encoder 19.1 → 16.8, embed 17.5 → 17.5, decoder 20.9 → **16.4**, lm_head 17.2 → 16.4 GiB.

  Two changes remain, in increasing order of cost and of payoff:

  2. **Quantize (or at least `astype`) each phase's weights as it converts, rather than at write time.**
     `write_gguf`'s loop currently upcasts everything to f32 first, so the `quantize=` argument shrinks
     the artifact and nothing else. Doing it per phase makes the accumulated term small.
  3. **Convert each phase in its own process, loading only that phase's submodule.** The real fix, and
     the expensive one: it needs partial checkpoint loading (instantiate the submodule and load only
     its keys) and a merge step that reads each child's weights back off disk. Peak becomes
     `max_phase(submodule + its MIL)`.

  **Gate:** the existing byte-identity sweep — every model re-exports identical — plus a measured peak
  RSS for one large model before and after. Motivated by P4.3b's Voxtral measurements; note there that
  even all three changes leave Voxtral at ~29 GB against 28, so this is not a fix for that model, and
  should not be scheduled as if it were.

#### P6 — cleanup

R6 executions (one commit per model: re-point the last test, delete the bespoke converter), then the
`tools/convert_*` directories themselves (~14,000 lines across 10 directories), then the docs pass.

**If only three things get done:** P0.1 + P0.2 (a smaller, honest baseline), P3.1 + P3.2 (the registry,
which is what makes every later family cheap), and P4.3 (the composition template, which is where the
model count actually lives). *All three are now done; the next largest lever is family 11's codec
decoders, which pay twice (P5).*

### Third family template: NeMo ASR encoders (Conformer-CTC, Parakeet-TDT, Parakeet-RNNT) — DONE

`tools/loom_mil_compiler/nemo_asr_export.py`; the three export scripts are now a docstring plus a
`NeMoASREncoderSpec`. Two findings worth keeping (both recorded in BACKEND.md with the evidence): only
**three** of the five differing fields this entry predicted were real (the restore class dissolves —
`ASRModel.restore_from` dispatches on the checkpoint's own config target and returns the identical
concrete class), and the wrapper's return value became a validated `EncoderOutput` claim rather than a
free-form expression. Verified byte-identical against a `git archive HEAD` baseline for all three models.

The three end-to-end tests remain the gate for any further change to this family (each takes the
exported GGUF plus a reference fixture): `test_e2e_conformer_ctc_mil_export`
(`LOOM_CONFORMER_CTC_DIR` + `LOOM_CONFORMER_CTC_MIL_GGUF`), `test_e2e_parakeet_tdt_mil_export`
(`LOOM_PARAKEET_TDT_DIR` + `LOOM_PARAKEET_TDT_MIL_GGUF`), `test_e2e_parakeet_rnnt_mil_export`
(`LOOM_PARAKEET_RNNT_DIR` + `LOOM_PARAKEET_RNNT_MIL_GGUF`).

### Open follow-ups from the exporter-improvement thread

All are recorded in full in BACKEND.md; this is the index.

- **Symbolic shape expressions carry sympy objects instead of strings — DONE.**
  `tools/loom_mil_compiler/shape_expr.py` (+ `test_shape_expr.py`). The derivation walk composes algebra
  and renders once at emission through a printer restricted to `symbol_env.cpp`'s grammar, which raises
  on anything it cannot express rather than shipping unparseable text. Conformer-CTC's frame count went
  from `(floor((((floor(((1) * (1) * ((((floor(((1) * (((1)+(((n_tokens) - (1)))))) / ((1) * (1))))) +
  512))) / ((1))))) + 0 - 512) / 160) + 1)` to `floor(n_tokens/160) + 1`. Diffs were read rather than
  required empty, via the new `tools/loom_mil_compiler/compare_snapshots.py` (evaluates every changed
  attribute at 18 concrete lengths and reports anything not numerically equivalent as structural).

  Two things a future change here must not lose:
  - **The assumptions are load-bearing.** Shape symbols are built as positive integers (`shape_expr.symbol`),
    which is the only reason `floor(512*n_tokens/512)` reduces to `n_tokens` at all. A bare
    `sympy.Symbol("n_tokens")` compares unequal to the interned one and silently stops cancelling.
  - **`floor` arguments are recombined with `sympy.together` before printing.** Sympy distributes
    rational coefficients over sums on construction (`floor((n-512)/160)` → `floor(n/160 - 16/5)`), and
    the engine evaluates in `double`, where the distributed form takes three roundings inside a floor
    instead of one.
- **Multi-output topologies in `GraphBuilder`/`run_subgraph` — DONE, see P2** in the implementation
  sequence above. The capability now exists; *using* it to infer a `FlowMatchingSpec` from a
  scripted-loop trace (a MIL loop body has one output per loop-carried var) is still deliberately not
  pursued — BACKEND.md's item 3 follow-up found that inferring the spec isn't worth building yet compared
  to the ~13-line declarative spec it would replace.
- **Item 5 of `EXPORT-IMPROVEMENT.md` (prototype StableHLO on one solved model)** remains deliberately
  not started; the proposal itself files it as a validation exercise rather than a fix.

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

### Known bug: `tools/convert_lfm/make_lfm2_gguf.py`'s per-layer zero-RoPE placeholder produces NaN

`test_e2e_lfm2_lua_driver`'s only previously-recorded failure mode, across every entry in this file that
mentions it, was a *missing local fixture* (`lfm2_350m.gguf` was never actually present, so the bespoke
Lua-driver path itself was never really exercised end-to-end). Regenerating the fixture from the bespoke
converter (`~/.venvs/piper/bin/python3 tools/convert_lfm/make_lfm2_gguf.py /home/flavio/Dev/models/lfm2-350m
lfm2_350m.gguf`) and actually running the test surfaces a real, separate bug: a NaN reaches a SILU
activation and trips ggml's own `assert(!isnan(x))` (`ggml-cpu/ops.cpp`) — a hard `SIGABRT` the test
process cannot catch, unlike a C++ exception.

Root cause (not yet fixed): `make_lfm2_gguf.py`'s `LayerSubmodule.forward` traces each of the 16 decoder
layers *independently*, each time passing RoPE `position_embeddings` as `torch.zeros(1, 1, 64)` — the
script's own comment calls these "Placeholders that we will swap", but they never got swapped. This
predates the exporter-improvement thread entirely (last touched 3 commits before it, unrelated to MIL
compiler work) and is orthogonal to the real MIL export path: `export_lfm2_monolithic.py`'s
`lfm2_350m_monolithic.gguf` (real RoPE, traced as one flattened graph) matches real HF logits exactly via
`test_e2e_lfm2_mil_export`.

**Not fixed.** `test_e2e_lfm2_lua_driver` now skips unconditionally (return code 77, matching the
project's skip convention) with a comment pointing here, rather than attempting a run that would abort
the whole test binary. Fixing this for real means threading each layer's actual token-position-dependent
RoPE table through the independently-traced submodules (the same kind of problem `ModularExportSpec`
already solves correctly for the MIL path) — worth doing only if this bespoke script's own coverage still
matters; the MIL path is otherwise a strict improvement over it.

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
     [Qwen3 + LFM2's own monolithic/atomic exports go through this one] /`modular_export.py`/
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
  plus a new hand-written orchestration script, `tools/convert_piper_vits/vits_driver/`, into one
  combined `vits_mil.gguf` (mirroring `convert_vits_lua_all.py`'s own packing for the bespoke topology).
  The cross-phase host logic (duration-based frame expansion, RNG sampling) is genuine host control flow
  no amount of MIL tracing can produce either way — it was always hand-written, in Lua, regardless of
  whether the topologies underneath are hand-built or machine-traced; `vits_driver/`'s own math is
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
    layout instead; `kokoro_driver/` converts via `from_row_major`, no transpose needed Lua-side
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
  - **`kokoro_driver/` written**, wiring the two new MIL topologies together with the EXISTING
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
  `styletts2_mil.gguf` (`styletts2_driver/` orchestrates them alongside the EXISTING bespoke
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
  `tools/convert_matcha/matcha_driver/`, mirroring `matcha_driver.lua`'s own Euler-CFM-sampling +
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

  `tools/convert_matcha/matcha_driver/`'s own layout differs from the bespoke `matcha_driver.lua`:
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
  `tools/convert_supertonic/supertonic_driver/` mirroring the bespoke `supertonic_driver.lua`'s own
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

### Modular-export blueprint: promote to default, prove generality, dedup weights

The modular-export blueprint (`tools/loom_mil_compiler/modular_discovery.py`/`modular_export.py`,
`apply_modular_export` in `exporter.py`, `export_lfm2_modular.py`) landed as a working first iteration,
proven on LFM2 (`test_e2e_lfm2_mil_export` passes against `lfm2_350m_modular.gguf`, same top-1 tokens as
atomic/monolithic). Four things from that iteration's own plan were originally open; two are now done:

- **Promoted to the default split mechanism — DONE (P0.1).** `apply_atomic_export`/`export_lfm2_atomic.py`
  (the older scope-based partitioner) are gone entirely; `profile="atomic"` was retired (R7, approved) and
  the modular blueprint is now the only split mechanism this exporter has.
- **Unproven generality — only ever validated on LFM2.** The whole point of this thread was generality
  (no `ModuleList`-naming-convention assumption, structural rather than by-name discovery), but that claim
  is still resting on a single model. Needs a second, structurally different HF model (different attribute
  names, ideally non-hybrid/homogeneous-layer to start) added to the regression suite.
- **Cross-submodule weight duplication — DONE (P0.2).** Each submodule is traced independently
  (`ct.convert()` per submodule), so any tensor referenced from more than one submodule — the most likely
  case being HF's tied embedding/`lm_head` weight — used to get serialized twice under two different
  namespaced names (confirmed: modular GGUF was 1.69GB vs. monolithic's 1.42GB on LFM2, consistent with one
  full extra copy of the ~268MB vocab embedding). Fixed exactly as planned here: `write_gguf` now hashes
  each candidate weight's bytes+shape+dtype and aliases a repeat hash to the first-written name via a
  `loom.tensor_alias.*` KV pair, resolved back into `GgufModel::load`'s `symbols_` map at load time — see
  P0.2 above for the full writeup (`lfm2_350m_modular` dropped from 1611 MiB to 1355 MiB).
- **Phase 2 (fully automatic prefix/suffix boundary discovery) not attempted.** Today's `ModularExportSpec`
  needs a ~3-line declarative boundary per model (`prefix_attr`/`repeated_attr`/`suffix_attrs`/`aux_attr`).
  The stretch-goal alternative — an early-exit-hook technique mirroring HF `accelerate`'s device-map
  splitting, deriving prefix/suffix without any per-model spec at all — was deliberately not attempted since
  Phase 1's spec was sufficient for LFM2. Worth doing only if Phase 1 starts feeling like real friction
  across 2-3 more models, not speculatively.

---

## Engine

### Performance optimizations designed but not implemented

- **Bucketed KV-cache graph-reuse — DONE, in two halves: P4.0.13 and P4.0.15.** `GraphBuilder::build()`
  no longer rebuilds per call: it retains the last graph and returns it unchanged when the axes repeat,
  with the declared inputs moved out of the gallocr pool so the aliasing hazard cannot apply (P4.0.13,
  `tests/test_graph_builder_reuse.cpp`). That covered every fixed-shape loop but not an autoregressive
  decode, and the original plan here — round `n_kv` up to a bucket boundary (e.g. 32) and skip the
  rebuild while the bucket holds — would not have either: `n_past` was baked into each layer's
  `ggml_view_2d` write offset (`KvCache::write_k/write_v`), so consecutive decode steps differed in the
  graph regardless of what `n_kv` rounded to. P4.0.15 made the write destination data (`ggml_set_rows`
  plus a cell-index tensor, the indirection listed as absent under "Scope limitations" below), bucketed
  `n_kv` at 32, and padded the mask host-side — a prefill plus 40 decode steps is now three graphs.
  Read P4.0.15 for where the padding actually happens, which is not where this paragraph guessed.
- **`ggml_backend_sched` / multi-backend.** Not used anywhere — engine talks to a single `ggml_backend_t`
  directly via a plain `ggml_gallocr`. Fine for CPU-only; needed once a second backend (CUDA/Metal) is
  added and graphs need splitting across devices. This is what makes P4.0.12's retained outputs a
  latent win rather than a paid one: a marshalled inter-module edge is two host copies today and a
  device→host→device round trip per edge per step the moment this lands.
- **Flash attention.** `ATTENTION` (`src/ops/primitives_attention.cpp`) always uses the composite
  (`MUL_MAT`→`soft_max_ext`→`MUL_MAT`) path — chosen because `ggml_flash_attn_ext` forces an F16 K/V cast
  that fights exact fp32 verification. A `FLASH_ATTENTION` primitive can be added later as a purely
  additive alternative once a GPU backend makes the perf/precision tradeoff worth it.

### Scope limitations (still true)

- **`KvCache` is single-sequence.** No ring buffer, no multi-stream/multi-sequence support. The
  `ggml_set_rows` index-tensor indirection this entry used to list alongside them **exists since
  P4.0.15** — writes are addressed by a cell-index tensor, and `KvCache::fill_cell_index` is the single
  place a second addressing policy would go. What is still missing is a *policy* that uses it: only the
  contiguous append `[n_past, n_past + n_tokens)` is ever written, reads are still a plain view over
  `[0, n_kv)`, and there is one `kv_size` for every layer.
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
