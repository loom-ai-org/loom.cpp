---
type: index
category: backlog
last_updated: 2026-09-03
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
| **P5 family 11 — codec decoders** | DAC done and verified; the second leaf is scoped, not started → [Epic-03 §2](../epics/epic-03-model-coverage.md) |
| **P5 family 10 — AR LM + codec TTS** | Gated end to end and the ASR oracle passes; the card and the loom-py bump are what is left → [Epic-03 §2](../epics/epic-03-model-coverage.md) |

---

## Models

* [ ] **Qwen3-ASR-0.6B / Qwen3-TTS-0.6B variants** — not started. No conversion script, no source-level
  architecture read. Qwen3-TTS is expected to be the most architecturally novel item in the family and
  needs its own read before scoping. *Context: [Epic-03](../epics/epic-03-model-coverage.md)*
* [ ] **F5-TTS** — deferred by explicit direction. Flow-matching, `OdeStepper`-adjacent, likely shares
  primitives with Matcha-TTS. Last of the original 7-model TTS list still untouched.
* [ ] **P5 breadth**, in coverage-per-effort order. Family 12 is DONE (2026-09-03) — the remainder:
  11 (codec decoders) → 4 (CNN+CTC) and 5 (SANM) → 9/10 (remaining TTS) → 6 (text enc-dec) → 13 (small
  classifiers) → 14 (music). *Context: [ADR-019](../adrs/adr-019-family-12-needs-no-attention-mask.md)
  for what family 12 cost, which is the estimate the rest of this list should be read against.*
* [ ] **P5 family 10 — Dia-1.6B: one thing left.** The export, the driver, the sampler, the
  classifier-free guidance and the composition with DAC are all done and gated. It ships as TWO files
  chained by the host ([ADR-022](../adrs/adr-022-dia-and-its-codec-stay-two-files.md)), its sampler
  cost the engine two additions
  ([ADR-023](../adrs/adr-023-a-second-stream-is-declared-not-derived.md),
  [ADR-024](../adrs/adr-024-guidance-belongs-in-the-sampler.md)), and Dia's guidance turned out not to
  be the standard formula
  ([Retro-031](../retros/retro-031-dias-guidance-is-not-the-standard-formula.md)). What remains:
  * [ ] **loom-py needs the submodule bump and a rebuild, then its card gate can run.**
    `src/binding.cpp` now calls `loom::register_topologies`, without which a guided generation through
    the Python door runs both decode streams into one KV cache and returns plausible, wrong codes. The
    binding change is written and compiles against the new headers; the bump waits on loom.cpp being
    merged. Until then `test_model_cards.py`'s new `text2codes` arm — which chains the card through
    `dac-44khz` and transcribes the result — has never executed against the shipped file.
  * [ ] **The Hub upload.** `hf-models/dia-1.6b/` is built — a 6.4 GB F32 GGUF and its card, F32 like
    every other entry after the Q8_0 version was tried and reverted for consistency. Not pushed.

* [ ] **EnCodec 32 kHz — two named blockers, both scoped.** MusicGen's codec, and the second family-11
  leaf. (1) coremltools refuses its length-derived convolution padding on a dynamic axis — the
  Supertonic wall — though the pad is provably 0 for the stride-1 decode path and should patch to a
  constant, per stage. (2) Its decoder has a 2-layer LSTM over the time axis, so it needs the
  `ScriptedLoop`/`run_recurrent` path rather than `Flattened`. Its `decode` signature and config
  spellings are already written and tested in `CodecFamily`; the recognizer detects it and raises
  naming both reasons. *Context: [Epic-03 §2](../epics/epic-03-model-coverage.md).*
* [ ] **SNAC** — the other family-11 candidate, and a different axis of difficulty from EnCodec:
  `vq_strides [4, 2, 1]` puts its codebooks at DIFFERENT frame rates, which is what tests whether
  "codes in, frame-major" survives a multi-rate codec. Needs the `snac` package (not in transformers).
* [ ] **A family-12 checkpoint that is not WordPiece.** Two are verified — `dslim/bert-base-NER` and
  `dslim/distilbert-NER`, structurally different encoders — and both are WordPiece with a CoNLL-03
  head, so what is still untested is the TOKENIZER half rather than the graph half. `fullstop-punc` is
  XLM-R (SentencePiece), which is the natural third and the one the roadmap actually names.
* [ ] **P5.0 — per-phase process isolation for conversion.** Decides which models are exportable at all
  on a given machine. Change 1 done (30.4 → 22.9 GB peak on Granite-Speech). Two remain:
  * [ ] quantize/`astype` each phase's weights as it converts, rather than at write time
  * [ ] convert each phase in its own process, loading only that phase's submodule — needs partial
    checkpoint loading and a merge that reads children back off disk
  * *Note: even all three leave Voxtral at ~29 GB against 28. Not a fix for that model.*

## Exporter / MIL compiler

* [ ] **Supertonic carries 2.3 MB of the same zero padding VITS just lost.** `ttl_text_512.emb_*`,
  99.1% zeros — 0.9% of that model, against VITS's 23.2%. **Not the same fix**: P4.28 made VITS's pad
  dynamic, and Supertonic's text axis is statically sized on purpose for two independent reasons its
  own export docstring gives, one of which is that `GraphBuilder` resolves only one dynamic-length
  symbol per topology. It belongs to that limitation, not to the pad.
  *Context: [Retro-005](../retros/retro-005-supertonic-fixed-text-length.md),
  [Epic-05 §5](../epics/epic-05-edge-performance.md) P4.28.*
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

* [ ] **Write down that loom-exporter's tests run under `~/.venvs/piper`.** `python3` on the dev box
  resolves to **`~/.venvs/ovos`** — transformers 5.14.1, **no `sentencepiece`** — which is the
  Qwen3-ASR-only env, and `tests/ci` under it is `4 failed, 568 passed`. **All four are the
  environment and nothing else: under `~/.venvs/piper` the same four are green** (38 passed, verified
  2026-08-29). `test_spec_protocol` fails on ovos because `spm_tokenizer_export` cannot import without
  `sentencepiece`; the three `test_causal_lm_export` registry tests fail with
  `LinkError: MonolithicCall … supplies input(s) it does not declare: ['cache_position']`, a
  transformers-version difference in what the trace sees. **`piper` (transformers 4.57.6,
  sentencepiece 0.2.1) is the env for everything except Qwen3-ASR** and must not be upgraded — NeMo
  pins `~=4.53`. It is written nowhere in the repo, so the next person meets four red tests with no
  way to tell. Put it in loom-exporter's README, and note that piper is ~3x slower to run them
  (4m12s against 1m21s for the same two files).
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

## Backends & accelerators

* [ ] **NPUs.** Open, and the shape of the problem is known: no NPU registers as a `ggml` device type
  the engine can resolve, so `"npu"` throws by design. CoreML (the Neural Engine, which Metal is not)
  and RKNPU2 are out of tree and cost more, licence check included. *Context:
  [ADR-010](../adrs/adr-010-device-selection-by-kind.md), [Epic-04](../epics/epic-04-backends-and-accelerators.md)*
* [ ] **The device hierarchy ranks a GPU above the CPU on unified memory, where the proxy does not
  hold.** The rule stands for "has its own fast memory", which an Apple Silicon GPU does not — it has
  the CPU's. Measured at HEAD with `ggml-0016`: `device=""` picks `MTL0` and is **1.76x faster on
  whisper-small and 1.63x SLOWER on VITS** (2.75x at Q4_0). It is why Metal ships as an extra rather
  than in the base macOS wheel; if this changes, folding it in is worth revisiting. **P4.30a, P4.30d and
  P4.30c step 5 between them took the VITS ratio from 8.98x to 1.63x and did NOT rescue the rule** — a
  GPU that is 1.6x slower on one model and 1.8x faster on another still cannot be ranked above the
  CPU by a rule that reads neither. This stays open on its own terms, with a much smaller number.
  → [Epic-04 §5.8](../epics/epic-04-backends-and-accelerators.md),
  [ADR-010](../adrs/adr-010-device-selection-by-kind.md)

* [ ] **On Metal, a Q4_0 convolutional model is now SLOWER than the same model at f32** — 141.7 ms
  against 88.7 on VITS, an inversion `ggml-0015` created and did not exist before it (149.7 against
  278.7); `ggml-0016` helps both arms and so leaves the ratio slightly WIDER, at 1.60x. The mechanism is known and is not the quantization: ggml-metal declines loom's folded
  block-quantized convolution kernel on its type test ([Epic-04 §5.2](../epics/epic-04-backends-and-accelerators.md)),
  so a Q4_0 export lowers through `im2col` + `mul_mat` and never reaches the new kernel — whose fast
  path is F32/F16-only by the same test. The choice is between teaching that type test about the
  folded kernel (P4.13's format) and dequantizing into the new kernel's fast path the way `ggml-0013`
  does on the CPU — and **P4.30c step 3 has since done the CPU-side equivalent of the second option
  for the 2-D form** (`op_conv_2d` hands a folded kernel to `ggml_conv_2d_direct_packed`, which
  dequantizes and re-enters), so the shape of the fix is now demonstrated on one backend. **Neither is on any critical path** — Metal is an extra — but "quantizing costs
  1.5x on this backend" is a surprising thing to leave undocumented in the model cards.
  → [Epic-04 §5.8](../epics/epic-04-backends-and-accelerators.md),
  [ADR-017](../adrs/adr-017-no-k-quants.md)
* [ ] **`device_report()` still buckets every node as either device or CPU**, deliberately — it does not
  say *why* a node fell back.
* [ ] **Whisper's 400-wide reflect pad** is cheaper to fall back on than to compose. CUDA, Metal and SYCL
  run it natively regardless.

## Text front-ends

* [ ] **Task #79 part 2 — the C++ `orthography2ipa` port.** `src/text/phonemize.cpp` +
  `include/loom/text/phonemize.h`, vendored as an Apache-2.0 submodule, verified against the Python
  door as its oracle. Part 1 is closed — every phoneme-input TTS GGUF carries its symbol table and both
  hosts read it ([Retro-029](../retros/retro-029-a-vocabulary-only-two-hosts-could-read.md)).
  **Measure both risks first:** the fold-down into each checkpoint's fixed symbol→id table, and
  pinning the beam search's tie-break.
  *Context: [ADR-012](../adrs/adr-012-permissive-phonemizer.md)*
* [ ] **Generalize the grapheme front end out of C++** — when a real second grapheme TTS model exists.
  Qwen3-TTS is not one. *Context: [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md)*
* [ ] **Remaining BPE pretokenizer families** beyond the ~40 in `pre_spec_table()` (CJK-script splitters,
  case-transition shapes, `byte_encode=false` SPM-style families). Each raises a named error rather than
  mis-tokenizing — bounded; add one when a real model needs it. **Distinct from the added-token pre-pass** P4.23
  shipped (`<|im_start|>` and friends, which `encode` could not emit at all) rather than pretokenizer
  regexes — but both land in `bpe_vocab.cpp`, so read
  [Epic-07 §4](../epics/epic-07-text-frontends-and-tokenizers.md) before opening this one.

## Host API

* [ ] **`GgufModel::hparam_env()` surfaces only numeric scalar KVs** into the `SymbolEnv`; string, bool
  and array-typed `loom.*` KVs are silently skipped.

## Packaging & release

* [ ] **`distilbert-ner-loom` and `dia-1.6b-loom` are staged but not pushed.**
  `build_model_cards.py` produces both — but until `upload_all.py --create` is run the Hub lists
  seventeen models, and `model.text2class` and `model.text2codes` are doors with no downloadable model
  behind them (loom-py's README documents calls against both). Two `--create` uploads, then the Hub
  count in that README and in [Epic-03 §2](../epics/epic-03-model-coverage.md) goes to nineteen.
  **`dia-1.6b-loom`'s card loads `dac-44khz-loom` too**, so the pair has to be published together or
  its snippet is a broken link. At 6.4 GB it is also by far the largest thing in the collection —
  see [Epic-03 §2](../epics/epic-03-model-coverage.md) for why it is not quantized.
* [ ] **`nlohmann/json` is fetched as a full ~290 MB clone** for a header-only library, and it failed
  twice over a slow link during the macOS work. `GIT_SHALLOW TRUE` on that `FetchContent_Declare`
  (it is pinned to a tag, so shallow works) would remove the largest download in a cold build.
* [ ] **P7 — 32-bit ARM Linux (`armv7l`), after P5.** Scoped 2026-09-03 from a source read, not
  started, and the first task is to make the estimate falsifiable with a QEMU build. **The Pi Zero 2 W
  is not in it** — that board is ARMv8 and already a supported target on a 64-bit OS, costing one
  `QEMU_CPU=cortex-a53` gate row. The port itself is one `CMAKE_SIZEOF_VOID_P` guard on
  `GGML_CPU_ALL_VARIANTS`, a LuaJIT armv7 build, and a runner-policy decision for the wheel; **ARMv6
  (Pi Zero / Zero W / Pi 1) is a declared non-goal**. *Context:
  [Epic-08 §6](../epics/epic-08-packaging-and-release.md)*

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
* [ ] **`GgmlPatches.cmake` asks "already applied?" the wrong way, so every `cmake` re-run rebuilds
  ggml from scratch** (~30 min on the Pi). It reverse-applies **each patch individually against the
  final tree**, which only holds while no later patch rewrites lines an earlier one added —
  `ggml-0004`..`0007` all fail it today. **Not a context-width problem**: regenerating one at `-U3`,
  `-U1` and `-U0` all still fail, because the added lines themselves are gone. The fix is a stamp file
  holding the applied set's names and hashes, skipped when it matches. Build-time cost only; the
  reset-and-retry path it falls into is correct.

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
