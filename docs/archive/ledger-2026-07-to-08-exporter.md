---
type: archive
status: closed
domain: exporter
covers: 2026-07-24 – 2026-08-09
last_updated: 2026-08-22
---

# Archive: Exporter Execution Detail, July–August 2026

> **This file is not maintained.** It is the verbatim per-commit execution record of exporter work that
> completed more than a fortnight before the ledger was refactored. Its **decisions** live in the
> [ADRs](../adrs/), its **lessons** in the [retros](../retros/), and its **architecture** in
> [Epic-02](../epics/epic-02-mil-exporter-and-compiler.md). Nothing here is open work.
>
> Kept because the measurements are real and the reasoning trail is worth more than the disk it costs.
> Do not add to it; do not cite it as current.

## P0 — clear the ground


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


## P1 — exporter internals

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


## P2 — enable multi-output topologies

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


## P4.0 — the sixteen items settled before the first from-scratch family config

Sixteen items that P3 left in a state P4 would otherwise inherit and harden — three carried over from P3
(P4.0.1–P4.0.3, all DONE), five added by `loom-exporter/docs/EXPORT-PREPARATION.md`
(P4.0.4–P4.0.8), three added by [`KV-CACHE.md`](../KV-CACHE.md) (P4.0.9, scheduled **before** P4.0.7's
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

The five items below come from `loom-exporter/docs/EXPORT-PREPARATION.md`, which carries the
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

  3. **D.3 — `loom-exporter/docs/DRIVER-COMPONENTS.md`, generated.** Per component: what it emits,
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
- **P4.0.9 — KV cache on the MIL path — DONE (stages N/1/2/3).** Specified in [`KV-CACHE.md`](../KV-CACHE.md); the one item here
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


## RecurrentPhase handles a stacked LSTM

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


## Parakeet's four traced phases

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


## Conformer-CTC gains a Lua entry point

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


## SentencePiece-style byte-fallback BPE

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


## Third family template: NeMo ASR encoders (Conformer-CTC, Parakeet-TDT, Parakeet-RNNT)

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


## Symbolic shape expressions carry sympy objects instead of strings
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

