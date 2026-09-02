---
type: index
category: backlog
last_updated: 2026-09-02
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
| **Task #79 part 1 — export the phoneme symbol table** | no licence question, no C++, and it gives four TTS models a real text door → [Epic-07](../epics/epic-07-text-frontends-and-tokenizers.md) |
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

* [ ] **P4.17 — the libgomp wait-policy fix is SHIPPED (`bb95996`); only its two remainders are open,
  and both now sit in P4.30.** Root cause: loom built ggml with `GGML_OPENMP=ON`, so `ggml_barrier` was
  `#pragma omp barrier` and every thread slept on a futex at each of VITS's 2520 graph nodes — 334,609
  voluntary context switches over 5 syntheses at 24 threads against 160 without. `GGML_OPENMP=OFF` (now
  `LOOM_OPENMP`, default OFF, `cmake/Dependencies.cmake:73`) is **4.8x at 24 threads** and restores a
  monotonic curve. The full measurement, and why the fix is the build flag rather than
  `OMP_WAIT_POLICY=active` (a library cannot set that variable), is in
  [Epic-05 §2](../epics/epic-05-edge-performance.md) and
  [Retro-017](../retros/retro-017-libgomp-slept-at-every-graph-node.md). The default thread count is
  **closed** (P4.30b, 2026-09-02: it is the physical core count). Open: the upstream note
  (**P4.30c step 6**).

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

## P4.30 — the tails of P4.10–P4.29

Every P4.10–P4.29 item is closed as a headline, and the release carrying them (**1.0.0-rc7**, PyPI and
the Hub, 2026-08-31) is complete. What survives is a set of remainders, gathered here so they stop
being scattered across four sections. **None blocks a release and none is on the critical path to
P5.** Each line names the number that sizes it, and the epic that holds the evidence.

Three tasks: **a** and **b** are one task each, **c** is one sequential pass whose steps share setup.
**P4.30b is CLOSED (2026-09-02)** — the default thread count is the physical core count, and the x86
TTS and LM columns are re-sampled per launch. Two remain.

* [ ] **P4.30a — why Metal costs ~5x the CPU per unit of work on a convolutional graph.** Where VITS's
  5.24x actually lives. Not the `PAD` fallback (P4.30c step 5, worth 1.8%) and not dispatch overhead:
  cost per output sample is flat within 8% across a 5.8x change in utterance length, so it is
  compute-bound. **Starts with a per-op profile on Metal**, not with more reasoning. Its answer also
  decides the device-hierarchy item under Backends.
  → [Epic-04 §5.4](../epics/epic-04-backends-and-accelerators.md)

* [ ] **P4.30c — the small tails, one sequential pass.** Steps 1–3 are the same three exports, so the
  setup is paid once.
  1. **Matcha, Kokoro and StyleTTS2 through the post-P4.29 lowering** — one export and one whisper
     pass each. P4.29's entire verification is VITS; these three were last transcribed through the old
     lowering. Transcribe, do not correlate —
     [Retro-006](../retros/retro-006-kokoro-shipped-noise.md).
  2. **Re-measure the direct-conv budget on those same exports, on macOS.** `ggml-0006`'s new
     `__APPLE__` arm reads a 12 MB L2 where every Mac previously took the 512 KB floor; swept on VITS
     it changes nothing, because VITS's largest conv weight already fits under the floor. The shapes
     where a 24x budget change binds are in exactly these families.
  3. **Decide `op_conv_2d` on the evidence step 1 produces.** The 2-D form still takes im2col for a
     folded kernel; P4.29 gave only the 1-D form. Nothing in tree has a quantized 2-D convolution hot
     enough to notice — step 1's profiles say whether that holds. **If it does, close it unfixed.**
  4. **Re-measure `ldc` alignment, then decide.** `C`'s leading dimension is not a multiple of the
     cache line, so a job's full-line store straddles two lines on odd columns — *alignment*, not the
     false sharing P4.22 fixed. Worth **~4 ms of a 3.9 s transcription**, and closing it means padding
     `C`, an allocator change rather than a kernel one. If the 4 ms holds, that ratio is the argument
     for closing it unfixed.
  5. **A Metal `PAD` kernel that accepts a leading pad** — scoped and prototyped: **27/56 CPU splits
     to 0/2, identical digest, 493.3 → 484.7 ms.** Do not pick it up expecting a speedup; it is worth
     doing for cleanliness, upstreamability, and discrete GPUs where a round trip crosses PCIe. Same
     shape as Vulkan's missing `PAD_REFLECT_1D` (P4.7d).
  6. **Send the `OMP_WAIT_POLICY` / `KMP_BLOCKTIME` gap upstream.** ggml mitigates P4.17's mechanism
     for Intel's libomp only; `cmake/patches/UPSTREAM.md` is where it goes. Last, because it depends
     on nothing here.
  → [Epic-05 §5](../epics/epic-05-edge-performance.md),
  [Epic-04 §5.4](../epics/epic-04-backends-and-accelerators.md)

## Backends & accelerators

* [ ] **NPUs.** Open, and the shape of the problem is known: no NPU registers as a `ggml` device type
  the engine can resolve, so `"npu"` throws by design. CoreML (the Neural Engine, which Metal is not)
  and RKNPU2 are out of tree and cost more, licence check included. *Context:
  [ADR-010](../adrs/adr-010-device-selection-by-kind.md), [Epic-04](../epics/epic-04-backends-and-accelerators.md)*
* [ ] **The device hierarchy ranks a GPU above the CPU on unified memory, where the proxy does not
  hold.** The rule stands for "has its own fast memory", which an Apple Silicon GPU does not — it has
  the CPU's. Measured: `device=""` picks `MTL0` and is 2.69x faster on whisper-small and 5.24x SLOWER
  on VITS. It is why Metal ships as an extra rather than in the base macOS wheel; if this changes,
  folding it in is worth revisiting. **Downstream of P4.30a**, not independent of it: the 5.24x is
  what the rule is being judged against, and P4.30a is what explains it. → [Epic-04 §5.4](../epics/epic-04-backends-and-accelerators.md),
  [ADR-010](../adrs/adr-010-device-selection-by-kind.md)
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
  mis-tokenizing — bounded; add one when a real model needs it. **Distinct from the added-token pre-pass** P4.23
  shipped (`<|im_start|>` and friends, which `encode` could not emit at all) rather than pretokenizer
  regexes — but both land in `bpe_vocab.cpp`, so read
  [Epic-07 §4](../epics/epic-07-text-frontends-and-tokenizers.md) before opening this one.

## Host API

* [ ] **`GgufModel::hparam_env()` surfaces only numeric scalar KVs** into the `SymbolEnv`; string, bool
  and array-typed `loom.*` KVs are silently skipped.

## Packaging & release

* [ ] **`nlohmann/json` is fetched as a full ~290 MB clone** for a header-only library, and it failed
  twice over a slow link during the macOS work. `GIT_SHALLOW TRUE` on that `FetchContent_Declare`
  (it is pinned to a tag, so shallow works) would remove the largest download in a cold build.
* [ ] **Linux on ARM builds** — the platform that matters most for an engine whose stated target is edge
  devices.

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
