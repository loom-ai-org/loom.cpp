---
type: index
category: backlog
last_updated: 2026-08-30
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

* [ ] **P4.17 — loom stops scaling at 8 threads and then goes backwards. ROOT CAUSE FOUND and FIXED
  2026-08-23 (`bb95996`); what is left here is measurement, listed at the end.** It was **libgomp's
  default wait policy**. loom built ggml with `GGML_OPENMP=ON` (ggml's default, which loom's CMake had
  never chosen; it is now `LOOM_OPENMP`, default OFF, `cmake/Dependencies.cmake:73`),
  so `ggml_barrier` was `#pragma omp barrier` and ggml runs one after every non-empty graph node —
  2520 per VITS synthesis — and **every thread sleeps on a futex at every one**: 334,609 voluntary
  context switches over 5 syntheses at 24 threads, against 160 without OpenMP.
  **`GGML_OPENMP=OFF` is worth 4.8x at 24 threads** (0.189 -> 0.040 s) and restores a monotonic curve;
  the LM goes 11.0 -> 19.1 tok/s. Nothing regresses at any thread count, and a Pi 4 at 4 threads is
  unchanged — **this is a many-core fix, not a Pi one**.
  Of the three candidates only **(3) wake-up latency** was right; **(2) `n_tasks` clamping is not the
  mechanism and needs no ggml patch**, and (1) barrier cost is only a fifth of the delta.
  [Epic-05 §2](../epics/epic-05-edge-performance.md) has the full measurement and why the fix is the
  build flag rather than `OMP_WAIT_POLICY=active` (a library cannot set that variable).
  *Still open:* (1) **the README table has been re-measured** (2026-08-24) with both engines run back
  to back on each machine, and the previous one could not be reproduced from either side — see
  [Retro-018](../retros/retro-018-a-table-of-ratios-nobody-could-re-derive.md). (2) **The default thread count is now benchmarked and the answer is the
  PHYSICAL CORE COUNT** — 24 on the 285K (1.98x TTS / 2.41x ASR / 1.18x LM against today's 4), but
  **2 on a 2-core SMT Ryzen**, where the two extra SMT threads buy nothing on any task and cost on two
  (TTS 1.19x, LM 1.03x, ASR unchanged). "Every CPU" would be a TTS regression there; "every physical
  core" is right on both. **On the Ryzen it barely moves the published ratios, because onnxruntime
  prefers 2 threads too** — the rule is a property of the CPU, not of loom. Not implemented: every figure is
  one inference on an idle machine, and a host running several loom instances concurrently is exactly
  what ggml's conservative 4 suits, so this is a policy call rather than a further measurement.
  [Epic-05 §2](../epics/epic-05-edge-performance.md) has both sweeps. (3) Consider sending the `OMP_WAIT_POLICY`/`KMP_BLOCKTIME` gap upstream —
  ggml mitigates this for Intel's libomp only, and `cmake/patches/UPSTREAM.md` is where that would go.
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
* [ ] **The rc7 Hub push — now nine models, not four.** The refreshed whisper GGUF is not on the Hub
  (the decoder's loop-invariant V transpose, 2026-08-29), alongside the three already-stale models
  (whisper, vits, matcha) — and **P4.23 re-exported all five causal LMs**, whose published artifacts
  now tokenize a marker as seven literal ids and carry no chat template. The five cards are already
  regenerated in `../hf-models/` and show `text2text.chat(...)`; the GGUFs beside them are not.
  One push, and it is the only release chore left: *everything else in P4.18 is closed* — the
  per-shape encoder re-measure and the README's ASR column (item A), the V transpose (item B,
  **1.106x**), and `QK^T`'s mechanism (item C — the counters found it; see P4.21).
  [Epic-05 §2](../epics/epic-05-edge-performance.md) has all three.
* [ ] **`ldc` alignment is the last of the `QK^T` thread, and it is small.** After P4.22, `m = 1500`
  reaches 15.0 ms where a padded `m = 1504` reaches 10.9 — 1500 floats is 6000 bytes and `6000 % 64 =
  16`, so a job's full-line store still straddles two lines on odd columns. Closing it means padding
  `C` (an allocator change, not a kernel one) for ~4 ms of a 3.9 s transcription. **Re-measure before
  opening it.** *Context: [Epic-05 §5](../epics/epic-05-edge-performance.md), P4.22.*
* [ ] **The README's TTS and LM columns need the per-launch sampling the ASR column just got.** On the
  Core Ultra 9 285K (8 P-cores + 16 E-cores, no SMT) thread placement is chosen **once per process**
  and then sticks, so every run inside one process inherits the same luck and a within-process median
  does not average it out. Measured on ASR: onnxruntime at `intra_op=4` is bimodal across launches,
  **~1.07 s or ~1.57 s, a 1.48x spread**, and loom at 24 threads spans 0.994-1.291 s over 16 launches.
  The ASR cells were re-sampled as medians over separate launches (2026-08-24) and moved further than
  `ggml-0010` explains; **the TTS and LM cells on both 285K rows were not, and are still single-process
  medians.** Not a new measurement idea — it is [Retro-018](../retros/retro-018-a-table-of-ratios-nobody-could-re-derive.md)'s
  problem with a named mechanism. Pinning is NOT the fix: it constrains onnxruntime more than loom
  (pinned to four P-cores it runs *slower* than its lucky unpinned launches).
  → [Epic-05 §2](../epics/epic-05-edge-performance.md)
* [ ] **P4.26 — `ggml-0012` costs a Cortex-A72 2.4% on VITS.** [Retro-019](../retros/retro-019-a-patch-measured-on-one-isa.md)'s
  pattern a second time: worth 2.75x on x86 at whisper's `m = 1500`, checked on aarch64 **only at that
  same shape**, and every VITS matmul takes the branch it changed. ABBA-interleaved, two rounds:
  **1.032x and 1.016x**, ~27 ms per synthesis, attributed by profile to `CONV_2D` (+22.6 ms) and
  `MUL_MAT` (+5.4 ms) — the convolution, because `ggml-0004`/`ggml-0009` lower through the same
  `sgemm`. **Do not just gate it on `__aarch64__` without re-measuring x86**, and run whatever lands at
  VITS's small `m` *and* `m = 1500` on both ISAs. It moved the README's Pi TTS cell 0.96x -> 0.93x; the
  Pi's LM and ASR cells have **not** been re-measured and may carry it too.
  → [Epic-05 §5](../epics/epic-05-edge-performance.md)
* [ ] **P4.27 — 26 ms of op-level saving arrived as 5 ms of wall, and nobody knows where it went.**
  Opened by P4.25's negative result, and worth more than P4.25 was. VITS's 32 gate nodes measure
  **825 us each faster** when threaded (`scripts/bench18.cpp`, ggml's own threadpool, 3.92x at the
  exact shape) and the model moves **0.5%**, twelve paired ABBA rounds. Every number on both sides is
  careful; the loss is *between* the nodes. **Named suspect: ggml's threadpool sleeping between two
  multi-threaded nodes** — [Retro-017](../retros/retro-017-libgomp-slept-at-every-graph-node.md) is
  that mechanism from the libgomp side. If it is that, it is a tax on **every** threaded node in every
  model, not on 32 of them in one. Measure before building: instrument the gap between nodes, or A/B a
  graph of N threaded nodes against the same N interleaved with single-threaded ones.
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
  mis-tokenizing — bounded; add one when a real model needs it. **Distinct from the added-token pre-pass** P4.23
  shipped (`<|im_start|>` and friends, which `encode` could not emit at all) rather than pretokenizer
  regexes — but both land in `bpe_vocab.cpp`, so read
  [Epic-07 §4](../epics/epic-07-text-frontends-and-tokenizers.md) before opening this one.

## Host API

* [ ] **`GgufModel::hparam_env()` surfaces only numeric scalar KVs** into the `SymbolEnv`; string, bool
  and array-typed `loom.*` KVs are silently skipped.

## Packaging & release

* [ ] **P4.10 — macOS wheels**, Apple Silicon and Apple Intel. Four blockers scoped; blocker 4 (the DL
  loader searching for `.so` where CMake wrote `.dylib`) is shared with P4.11.
  → [Epic-08 §4](../epics/epic-08-packaging-and-release.md)
* [ ] **Linux on ARM builds** — the platform that matters most for an engine whose stated target is edge
  devices. 
* [ ] **The rc7 Hub push** — tracked once, under Engine — performance, with the list of what is stale
  and why.

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
