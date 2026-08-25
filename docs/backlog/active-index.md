---
type: index
category: backlog
last_updated: 2026-08-24
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

* [ ] **P4.17 — loom stops scaling at 8 threads and then goes backwards. ROOT CAUSE FOUND
  2026-08-23; the fix is applied to the working tree and green, not yet committed.** It is **libgomp's default wait
  policy**. loom builds ggml with `GGML_OPENMP=ON` (ggml's default, which loom's CMake never chose),
  so `ggml_barrier` is `#pragma omp barrier` and ggml runs one after every non-empty graph node —
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
* [ ] **P4.19 — `$LOOM_PROFILE` should be able to say WHICH GRAPH a bucket is in.** It keys on
  `(op, ne0, ne1)` and nothing else, so "`ne1 = 1500` is the encoder, `ne1 = 1` is a decode step"
  classifies only the buckets whose `ne1` is one of those two — and **a bucket that is neither gets
  assigned by eye**. That is exactly how P4.18's largest layout cost sat in the encoder's column while
  being 93% decoder, which inverted the item built on it AND the claim that the decode loop needed
  nothing. **The fix is about ten lines in `record()` (`src/core/profile.cpp`)**: behind
  `$LOOM_PROFILE_NODES`, print `node->name` and the full four-element `ne` per execution, so one run
  attributes every bucket. It was written as a scratch patch during P4.18 and reverted; Epic-05
  currently *tells the reader to write it again*, which is the tell that it should be a feature.
  Cheap, and it is the tool item (1) of P4.18 gets verified with.
  → [Epic-05 §2](../epics/epic-05-edge-performance.md)
* [ ] **P4.18 — the ASR gap is the encoder's ATTENTION, plus a loop-invariant copy in the decoder.**
  whisper-small is the one task still behind (0.57-0.72x at four threads).
  **Two of the first version's conclusions were wrong and are corrected in the epic.**
  (a) The `CONT 1500 x 64` bucket was assigned to the encoder by eye; `$LOOM_PROFILE` keys on
  `(op, ne0, ne1)` and cannot say which graph a bucket is in. Node names say it is **the decoder**:
  454 of its 471 ms are 26 executions of the decode graph. The corrected split is **encoder 5.46 s vs
  2.38 (2.29x), decode 1.10 s vs 0.97 (1.14x SLOWER)** — so "the decoder is ahead and needs nothing"
  was wrong. (b) Splitting onnxruntime's own `MatMul` time by shape shows loom's **dense GEMMs are
  within 10-12% of MLAS**; the whole 1.93x is the two batched attention matmuls.
  **Fused attention stays ruled out**, **GELU is DONE** (`ggml-0010`), **`SOFT_MAX` is RE-OPENED**
  (item 3 below — the probes that closed it were not floors), and **`k % KN` is DONE**
  (`ggml-0011`): tinyBLAS rejected every matmul whose CONTRACTION
  was not a whole number of vectors, and whisper's A@V contracts over 1500 frames (1500 % 8 == 4 on
  AVX2, % 16 == 12 on AVX-512, **% 4 == 0 on NEON, which is why an aarch64-only bench6 could not find
  it**). **2.15x at that shape; in model the `MUL_MAT 64 x 1500` bucket goes 2240 -> 1119 ms with no
  other bucket moving.** `scripts/bench13.cpp` is the x86 bench that was missing, `--check` is its
  correctness sweep, and `tests/ci/test_tinyblas_gemm.cpp` now covers the `k` residues (verified red).
  *Five things left; (1) is by far the largest:*
  (1) **the decoder's cross-attention V is re-transposed every step** — 12 nodes x 4.6 MB per token,
  **9.0% of the whole transcription and 47% of the decode loop** at one thread. It is a function of a
  graph input that is constant for the utterance. The fix is `_WhisperCrossKvWrapper.forward`
  (`loom-exporter/whisper_export.py:147`) emitting V already head-split and transposed — a whisper
  re-export and a fixture refresh, so not free at release time (ADR-003).
  (2) **`QK^T` at `k = 64` runs at about HALF the rate of a projection-shaped GEMM** — 2.10x paired
  (p10 1.66, p90 2.48), against a witness that is itself at 84-88% of this box's ~54 GFLOP/s
  single-core peak (Zen+ has 128-bit FPU datapaths, so a 256-bit FMA is one per cycle). ~1.1 s of the
  1-thread encoder. **Partly explained: tinyBLAS's `BM` row blocking is worth ~1.15x at k=64 and 1.02x
  at k=768, and whisper's `m = 1500` cannot reach `BM = 4` because 1500 % 16 == 12** — the same "a
  frame count is a number nothing rounds" as the `k % KN` finding. Reaching it needs a **cascade**
  (16-row prefix at BM=4, then 8, then 4, then the existing <= 3 row 1x1 tail); the naive version — a
  16-row prefix with a 15-row `gemm_bloc<1,1>` tail — is a regression, because that tail is one `hsum`
  per element. Ceiling ~1.15x on one op, needs a row offset threaded through `gemm`'s job
  partitioning: **do it on the workstation, not the 2-core dev box.** The remaining ~1.8x has no
  identified mechanism; the epilogue, the store pattern (mechanism falsified — no dependence on the
  size of C) and the address arithmetic are all ruled out. **Start the next attempt with a hardware
  profiler**, not a fifth guess. *Do not write a fused attention kernel.*
  → [Retro-012](../retros/retro-012-optimizations-that-were-measured-out.md)
  (3) **`SOFT_MAX` is RE-OPENED, and Retro-012's reason for closing it was wrong.** Its two "floor"
  probes were plain scalar C measured against a hand-vectorised candidate, so they compared compilers,
  not work — on the dev box the "no exp" arm comes out SLOWER than the arm it bounds. Rebuilt as the
  same function with the exp switched off, plus the `memcpy` arm nobody ever ran: **ggml is 3.9x the
  memcpy floor, so it is not bandwidth bound**, the exp is **1.42x** of a fused row, and the pass
  fusion is 1.16x. Of its 682 ms over 12 calls, 174 is the bytes, 241 the pass structure, 175 the exp,
  92 ggml's two extra passes; onnxruntime does the same 12 calls in 515. **The item to open is
  `ggml_v_expf`, not a soft_max rewrite** — the same shape of finding as `ggml-0010`'s GELU, in the
  same file. The 285K's 1.06x for the fusion stands as a number and falls as a reason.
  (4) **`ggml-0011` is worth MORE on the other ASR models than on whisper, and nobody has measured
  it.** whisper's 1500 frames are FIXED, so it missed `KN` on every clip; a Conformer's frame count is
  DYNAMIC. Found by accident on `conformer_ctc_mil` (jfk.wav, 1 thread, dev box): **856 -> 718 ms end
  to end, 1.19x**, from `MUL_MAT 551x276` 160.9 -> 82.6 (1.95x), `276x276` 84.8 -> 46.5 (1.82x) and
  `44x276` 35.2 -> 12.9 (2.73x). **276 % 8 == 4**, and the subsampling stride is 4, so an encoder
  length is always a multiple of 4 and its residue mod 8 is 0 or 4 — **about half of all clips hit
  this, per utterance, at run time.** That is a better claim than whisper's and it belongs in the
  README's ASR column. *gigaam 0.97x and parakeet-tdt 1.03x are INSIDE the dev box's ~1.2x noise floor
  — do not report them; re-run paired (`scripts/bench14.cpp`) or on the workstation.* The baseline arm
  is a ggml built without `ggml-0011`, not an older commit. Mind the sweep recipe's
  `cd`-into-the-measured-tree trap and Retro-018's per-launch sampling.
  (5) **Re-measure all of this on the 285K.** Everything above is the 2-core dev box. The corrected
  encoder/decoder split is *arithmetic* on the 285K's own published call counts (324 = 12 encoder
  nodes + 12 layers x 26 decoder executions, exactly) rather than a re-measurement, and the k tail's
  **in-model** effect there has never been run. The mechanism carries — the 285K is Arrow Lake, so
  AVX2, so `KN = 8`, so `1500 % 8 == 4` misses there too — only the magnitudes are open. While on that
  box, re-run the corrected soft_max floors: Retro-012's 1.06x came from there and the `memcpy` arm
  has still never been run on it.
  [Epic-05 §2](../epics/epic-05-edge-performance.md). *GELU + k-tail done, five items open.*
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
