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
- **NeMo Conformer-CTC-small — exports and runs deep into the real model, one open data-flow bug left.**
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

  **Still open**: even with all of the above, one real data-flow bug remains, found via bisection (not
  yet root-caused to a specific fix). The length-validity mask's `RANGE_1D` bound (`gather_7`, read from
  `shape(real_div(...log(...matmul(mel_filterbank, power_spectrum)...)...))` — i.e. the real, correctly
  *shaped* log-mel feature tensor) evaluates to `16000` (raw sample count) instead of the expected `~101`
  (frame count) at actual runtime, despite every op in that chain now having correct shape-string
  derivation and the upstream STFT `CONV_1D`'s own real ggml-computed output already being correctly
  101-sized (confirmed via a truncated-topology bisection). The remaining gap is somewhere in either (a)
  how `SHAPE`/`GATHER` read back a real tensor's dimensions at this specific point in the graph, or (b) a
  node between the STFT conv and this `gather` that's silently producing a wrongly-sized tensor despite
  correct shape *attributes* elsewhere (the C++ ops here don't all need declared JSON shapes — ADD/SUB/DIV/
  LOG/MATMUL derive their output shape automatically from real operand tensors at build time, so a
  shape-string fix doesn't necessarily touch them). Needs a fresh bisection session picking up from
  `/tmp/truncated_conv0.json`'s technique (register a topology truncated to `nodes[:N]` with a chosen
  intermediate as `"output"`, read back via `run_subgraph`'s shape return) — confirm the STFT conv's real
  output is 101-framed (done), then walk forward node-by-node through the `real_div`/`log`/`matmul`/
  `shape`/`gather` chain checking each one's real computed `ne` shape until the exact node that turns 101
  into 16000 is found.
- **Parakeet, VITS, Kokoro, Matcha-TTS, SupertonicTTS, StyleTTS2 — not started.** VITS/Kokoro/Matcha-TTS/
  SupertonicTTS are plausible but unproven — their two historical showstoppers (STFT/ISTFT, and LSTM) were
  exactly what got generalized into the MIL exporter as follow-up work, and their iterative bits (flow
  reverse-steps, Euler CFM, RNG-fed sampling) are fixed-depth, so should trace as one static graph with
  noise supplied as an input tensor. StyleTTS2 is the one likely to stay bespoke — its diffusion sampler's
  ~3e-3 residual mismatch persisted even with hand-matched float32 host math, and an auto-traced version
  gives less control to chase that kind of thing down.

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
