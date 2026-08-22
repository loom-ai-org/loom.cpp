---
type: adr
status: accepted
date: 2026-08-02
tags: [exporter, api-design, optimum, registry, family-templates]
---

# ADR-005: An `optimum`-Shaped Export API — `LoomExportConfig` and `TaskRegistry`

## Context

Once [ADR-004](adr-004-mil-as-the-single-export-path.md) made MIL the only path, each model still
reached it through its own top-level script (`export_qwen3_mil.py`, `export_conformer_ctc_mil.py`, …).
Every script hand-rolled the same tail — build topology, merge weights, `write_gguf` — and there was no
way for a caller to say "export this checkpoint" without knowing which script to run.

`optimum-onnx` had already solved this shape for a different backend: `OnnxConfig` describes how a
family exports, `TasksManager` maps a checkpoint to one.

## Options Considered

1. **Re-express the existing templates one-for-one as classes.** The literal minimum; leaves the naming
   keyed to whichever model needed a template first.
2. **A single universal config with per-family conditionals.** Fewer types, more invalid states.
3. **A small family hierarchy named by task**, plus a registry that recognises a checkpoint.

## Decision

A `{Domain}{Function}ExportConfig` hierarchy — `Domain` ∈ `Base`/`TTS`/`LM`/`ASR`, `Function` a
structural role or a bare model name for a leaf — rooted at `LoomExportConfig`, the way `optimum`'s
`ORTModelFor*` names a family by **task** rather than by the first model that needed it.

* **`TaskRegistry`** maps `(task, model)` to a config, with per-family `detect()` recognisers that read
  the checkpoint's own config rather than its directory name.
* **`main_export()` + the `loom-export` CLI** are the single entry point:
  `loom-export nvidia/parakeet-tdt-0.6b-v3 -o parakeet.gguf`.
* **`ModelPatcher.prepare_environment()`** makes each family's import-order stubs a named hook instead
  of unexplained top-of-file side effects.
* **Shared bases carry the tail**: `BaseMultiPhaseModelExportConfig` (declare `phases()`, get tracing,
  a content-aware weight merge and one GGUF), `TTSFlowMatchingModelExportConfig` (adds `samplers()`).

**Names are chosen to keep families distinguishable.** `FlowMatchingSpec` is *not* called
"IterativeRefinementSpec", because what it declares is Euler integration of a vector field — a vaguer
name would blur why StyleTTS2's real ADPM2 diffusion sampler is deliberately not part of that family.

## Consequences

* **Positive:** a new family is a config class plus a recogniser. Four top-level export scripts were
  deleted outright.
* **Positive:** a detection bug is testable without checkpoints — `test_registry.py` uses synthetic
  fake HF dirs and `.nemo` archives.
* **Negative:** recognisers are only as good as what the checkpoint declares. Parakeet-TDT and
  Parakeet-RNNT restore through the *identical* `EncDecRNNTBPEModel` target, so `target` alone cannot
  tell them apart; the real discriminator is `model_defaults.tdt_durations`.
* **Negative:** three pieces of the original `optimum` analogy were deliberately not built —
  `LoomModelFor*` runtime entry points, driver templates as first-class artifacts, and
  `.inputs`/`.generate_dummy_inputs()`/`.patch_model_for_export()` as named config members. Still open;
  see [the backlog](../backlog/active-index.md#exporter--mil-compiler).
* **Note on process:** a draft silently dropped a global monkeypatch import and produced a
  plausible-looking GGUF; the byte-identity gate caught it, review did not.

## Related

* Epic: [Epic-02: MIL Exporter and Compiler](../epics/epic-02-mil-exporter-and-compiler.md)
* Ledger record, verbatim:


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

## LFM2 migrated onto the causal-LM registry (follow-up to P3.1/P3.2, per explicit user direction)

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

