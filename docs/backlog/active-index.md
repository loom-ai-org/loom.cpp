---
type: index
category: backlog
last_updated: 2026-08-23
---

# Active Ledger — Open Work Across All Three Repos

Only **open** work lives here, one line each. Closed work is not tracked: its decisions are
[ADRs](../adrs/), its lessons are [retros](../retros/), its architecture is in the
[epics](../epics/), and its execution detail is in git history or [`archive/`](../archive/).

**Item IDs are the existing `P`-numbers.** Code comments reference them (`P4.3e`, `P4.15b`), so they
are not renumbered. New items continue the scheme.

---

## Now — the next things to pick up

| item | why now |
|---|---|
| **P4.10 — macOS wheels** | multiplies every model family added afterwards; a family is unreachable from a Mac until this lands → [Epic-08 §4](../epics/epic-08-packaging-and-release.md) |
| **Task #79 part 1 — export the phoneme symbol table** | no licence question, no C++, and it gives four TTS models a real text door → [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md) |
| **P4.13 — 2-D conv kernels** | closes the thread P4.12 opened while its measurements are fresh → [Epic-05 §5](../epics/epic-05-edge-performance.md) |
| **P5 family 12 — BERT token classifiers** | smallest possible template, and the first non-audio task → [Epic-03 §3](../epics/epic-03-model-coverage.md) |

---

## Models

* [ ] **Qwen3-ASR-0.6B / Qwen3-TTS-0.6B variants** — not started. No conversion script, no source-level
  architecture read. Qwen3-TTS is expected to be the most architecturally novel item in the family and
  needs its own read before scoping. *Context: [Epic-03](../epics/epic-03-model-coverage.md)*
* [ ] **F5-TTS** — deferred by explicit direction. Flow-matching, `OdeStepper`-adjacent, likely shares
  primitives with Matcha-TTS. Last of the original 7-model TTS list still untouched.
* [ ] **P5 breadth**, in coverage-per-effort order: family 12 (BERT token classifiers) → 11 (codec
  decoders) → 4 (CNN+CTC) and 5 (SANM) → 9/10 (remaining TTS) → 6 (text enc-dec) → 13 (small
  classifiers) → 14 (music).
* [ ] **P5.0 — per-phase process isolation for conversion.** Decides which models are exportable at all
  on a given machine. Change 1 done (30.4 → 22.9 GB peak on Granite-Speech). Two remain:
  * [ ] quantize/`astype` each phase's weights as it converts, rather than at write time
  * [ ] convert each phase in its own process, loading only that phase's submodule — needs partial
    checkpoint loading and a merge that reads children back off disk
  * *Note: even all three leave Voxtral at ~29 GB against 28. Not a fix for that model.*

## Exporter / MIL compiler

* [ ] **Modular-export generality is unproven** — the blueprint's whole point was structural rather than
  by-name discovery, and that claim still rests on LFM2 alone. Needs a second, structurally different HF
  model in the regression suite. *Context:
  [ADR-004](../adrs/adr-004-mil-as-the-single-export-path.md)*
* [ ] **Modular phase 2 — automatic prefix/suffix boundary discovery.** Deliberately not attempted;
  today's `ModularExportSpec` needs a ~3-line declarative boundary per model. Worth doing only if that
  starts feeling like real friction across 2–3 more models.
* [ ] **`LoomModelFor*` runtime entry points** — the *inference* half of the `optimum` analogy has no
  counterpart. Arguably should wait for a second consumer besides the tests. *Context:
  [ADR-005](../adrs/adr-005-export-config-and-task-registry.md)*
* [ ] **Driver templates as first-class artifacts** — `render_driver` is still marker-based string
  replacement into a hand-written `.lua`. The mechanism is `driver_ir`, which exists and is under-used.
  A family whose driver needs more than one substitution point will force it.
* [ ] **`.inputs` / `.generate_dummy_inputs()` / `.patch_model_for_export()` as named config members** —
  none exist under those names; axes are declared per phase and dummy inputs built inline.
* [ ] **Known gap: `matmul` composition only handles `transpose_x=False`.** Any other combination raises
  `NotImplementedError` by design rather than miscomputing. Not yet hit by any converted model; needs a
  real derivation and test case the first time one is.
* [ ] **StableHLO prototype on one solved model** — deliberately not started; filed as a validation
  exercise rather than a fix.
* [ ] **P6 cleanup** — delete the `tools/convert_*` directories (~14,000 lines across 10), then the docs
  pass.

## Engine — correctness

* [ ] **Who still depends on the layout-"healing" heuristics?** The guards make size-guessing
  *unreachable for already-correct graphs*; they do not remove it, and `op_repeat`'s two branches are
  **unguarded**. Every one is a silent-wrong-answer generator with no error path.
  *Work, in order:* (1) instrument each branch with a counter naming op and shapes, run all models, and
  record which fire — that converts "some model probably needs these" into a list; (2) fix each real
  firing at the **exporter**; (3) delete the branch. A branch that fires for nothing can go immediately.
  *Context: [Retro-001](../retros/retro-001-layout-healing-heuristics.md)*
* [ ] **Known bug: `make_lfm2_gguf.py`'s per-layer zero-RoPE placeholder produces NaN.** Each of the 16
  decoder layers is traced independently with `position_embeddings` as `torch.zeros(1,1,64)` — the
  script's own comment calls them "placeholders that we will swap", and they never were. A NaN reaches
  SILU and trips `assert(!isnan(x))`, a hard `SIGABRT`. `test_e2e_lfm2_lua_driver` skips (77)
  unconditionally. **Not fixed**, and worth fixing only if that bespoke script's coverage still matters —
  the MIL path is otherwise a strict improvement over it.

## Engine — performance

* [ ] **LFM2 is the only causal LM still on the O(n^2) decode path**, because its ShortConv blocks
  carry history no KV cache holds and its export therefore has no `infer_with_past`. Every other causal
  LM now takes the driver's own cached loop. Giving LFM2 a cached entry point means giving the engine
  somewhere to put ShortConv state, which is a real design question and not a one-line change.
  → [Epic-05 §2](../epics/epic-05-edge-performance.md)
* [ ] **P4.16 — the convolution gap, shape by shape.** Both named mechanisms are spent (P4.15c measured
  one out, P4.15d/f fixed the other) and the model is at 1.033x of onnxruntime end to end.
  **Re-measure the table against the post-P4.15f export before opening anything new on it.**
  *Do not start on the resblock kernel again* — measured at 83% of the machine's peak in cache.
  → [Epic-05 §5](../epics/epic-05-edge-performance.md)
* [ ] **P4.13 — 2-D conv kernels, so a convolutional model can be Q4_0.** Op eligibility is fixed;
  layout alignment is not (0 of 132 VITS kernels align for block 32 as stored). Acceptance: VITS exports
  at Q4_0 to ~28 MB with a non-zero coverage line, and its audio still transcribes through whisper-small.
  → [Epic-05 §5](../epics/epic-05-edge-performance.md)
* [ ] **`FLASH_ATTENTION`.** Unbuilt. The blocker is **the gate suite's exact-fp32 comparisons, not the
  hardware** — `ggml_flash_attn_ext` forces an F16 K/V cast. A GPU exists now and the trade still has not
  been made; it is a decision about verification. *Context:
  [ADR-016](../adrs/adr-016-kv-cache-shape.md)*
* [ ] **KV-cache addressing policies beyond contiguous append.** The `ggml_set_rows` indirection exists;
  `KvCache::fill_cell_index` is the single place a second policy would go. What is missing is a policy
  that uses it.
* [ ] **Quantized KV cache.** Storage is always F32. Different mechanism and different pipeline point
  from weight quantization — check how the cache is allocated and typed before assuming a trivial
  extension.
* [ ] **General multi-scheme quantization tool.** `quantize_gguf_q8_0.py` is Q8_0-only and
  single-model-shaped. A model-agnostic tool with a per-tensor-role policy (skip norm weights and
  embeddings) is unbuilt. *Scope note: [ADR-017](../adrs/adr-017-no-k-quants.md)*
* [ ] **Flag, not yet a bug: StyleTTS2 at Q8_0** produces audio at correlation **0.015** against its F32
  audio while transcribing correctly. The plausible reading is its stochastic style-diffusion sampler
  diverging onto a different-but-valid trajectory (Matcha's deterministic CFM stayed at 0.985) — but
  that is a **hypothesis**. Verify before shipping a quantized StyleTTS2.

## Backends & accelerators

* [ ] **NPUs.** Open, and the shape of the problem is known: no NPU registers as a `ggml` device type
  the engine can resolve, so `"npu"` throws by design. CoreML (the Neural Engine, which Metal is not)
  and RKNPU2 are out of tree and cost more, licence check included. *Context:
  [ADR-010](../adrs/adr-010-device-selection-by-kind.md), [Epic-04](../epics/epic-04-backends-and-accelerators.md)*
* [ ] **P4.11 — Metal.** Scoped; strictly after P4.10, because there is no macOS base wheel for a backend
  package to attach to and P4.10's `.so`-vs-`.dylib` loader blocker is the *same code path* that would
  discover `libggml-metal`. Does **not** block new model families. → [Epic-04 §5](../epics/epic-04-backends-and-accelerators.md)
* [ ] **`device_report()` still buckets every node as either device or CPU**, deliberately — it does not
  say *why* a node fell back.
* [ ] **Whisper's 400-wide reflect pad** is cheaper to fall back on than to compose. CUDA, Metal and SYCL
  run it natively regardless.

## Text front-ends

* [ ] **Task #79 part 1 — export the phoneme symbol table** as a vocabulary family. The data already
  sits in each checkpoint. No licence question, no C++. Gives VITS, Kokoro, StyleTTS2 and Matcha a real
  `model.tokenizer` and a working `synthesize(phonemes=...)`.
* [ ] **Task #79 part 2 — the C++ `orthography2ipa` port.** `src/text/phonemize.cpp` +
  `include/loom/text/phonemize.h`, vendored as an Apache-2.0 submodule, verified against the Python
  door as its oracle. **Measure both risks first:** the fold-down into each checkpoint's fixed symbol→id
  table, and pinning the beam search's tie-break.
  *Context: [ADR-012](../adrs/adr-012-permissive-phonemizer.md)*
* [ ] **Generalize the grapheme front end out of C++** — when a real second grapheme TTS model exists.
  Qwen3-TTS is not one. *Context: [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md)*
* [ ] **Remaining BPE pretokenizer families** beyond the ~40 in `pre_spec_table()` (CJK-script splitters,
  case-transition shapes, `byte_encode=false` SPM-style families). Each raises a named error rather than
  mis-tokenizing — bounded; add one when a real model needs it.

## Host API

* [ ] **Sampling is greedy argmax only.** No temperature, top-k, top-p or repetition penalty.
* [ ] **`GgufModel::hparam_env()` surfaces only numeric scalar KVs** into the `SymbolEnv`; string, bool
  and array-typed `loom.*` KVs are silently skipped.

## Packaging & release

* [ ] **P4.10 — macOS wheels**, Apple Silicon and Apple Intel. Four blockers scoped; blocker 4 (the DL
  loader searching for `.so` where CMake wrote `.dylib`) is shared with P4.11.
  → [Epic-08 §4](../epics/epic-08-packaging-and-release.md)
* [ ] **Linux on ARM builds** — the platform that matters most for an engine whose stated target is edge
  devices.
* [ ] **The HF push of the outstanding re-exports.**

## Standing scope limitations

Deliberate boundaries rather than tasks, each naming what would have to change. Kept in
[Epic-01 §4](../epics/epic-01-inference-engine-core.md#4-standing-scope-limitations): single-sequence KV
cache, F32 cache storage, one level of `repeat_for` nesting, no chunked/windowed inference for long
Conformer-CTC audio, only the small Conformer-CTC checkpoint verified, and the attention-variant
primitive set.

## Minor cleanups

* [ ] `KvCache::write_k/write_v/read_k/read_v` use `std::vector::at()`, which throws `std::out_of_range`
  rather than a `loom::Error` subtype. A malformed topology's `"layer"` attr could in principle reach
  this uncaught-by-`catch (loom::Error&)` path — low risk today, since the index always comes from
  `repeat_for`'s own loop bound.
* [ ] `export_config.py`'s module docstring points at a ledger section that no longer exists.

---

## Knowledge Hub

| | |
|---|---|
| **Domains** | [Epics](../epics/) — what each area is and how it works |
| **Decisions** | [ADRs](../adrs/) — why a technical choice was made |
| **Lessons** | [Retros](../retros/) — what broke, why, and the takeaway |
| **Specs** | [SPECIFICATION](../SPECIFICATION.md) · [KV-CACHE](../KV-CACHE.md) · [HIGH-LEVEL-API](../HIGH-LEVEL-API.md) · [PROCEDURAL-GENERALIZATION](../LOOM_PROCEDURAL_GENERALIZATION.md) |
| **Closed detail** | [archive/](../archive/) — unmaintained, kept for the reasoning trail |

**Before opening a performance item**, read
[Retro-012: Optimizations That Were Measured Out](../retros/retro-012-optimizations-that-were-measured-out.md).

**Before trusting a green gate**, read
[ADR-015](../adrs/adr-015-ci-and-gate-test-classes.md) and
[Retro-008](../retros/retro-008-a-gate-that-was-green-for-the-wrong-reason.md).
