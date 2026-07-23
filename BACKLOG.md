# Backlog

Everything previously tracked here as resolved (Milestones 1-8, the small-TTS-model roadmap covering
VITS/StyleTTS2/Kokoro/Parakeet/FastConformer-RNN-T, the `executorch-ggml` investigation, quantized-weight
POC, and all six TTS/ASR/LLM model ports through the Lua+MIL-compiler pivot) has been removed from this
file. That history lives in git log/commit messages, not here. Only genuinely open work remains below.

---

## Roadmap: Qwen3-ASR-0.6B / Qwen3-TTS-0.6B

Not started. Qwen3-0.6B-Base (the base LLM) is done. The ASR and TTS (12Hz-Base) variants of the Qwen3
family remain fully unstarted — no conversion script, no source-level architecture read yet. Qwen3-TTS is
expected to be the most architecturally novel item in this family (needs its own source-level
investigation before scoping).

## F5-TTS

Deferred by explicit user direction (flow-matching TTS, `OdeStepper`-adjacent — likely shares primitives
with Matcha-TTS, which is done). Last of the originally-considered 7-model TTS list still untouched.

## Task #79: permissively-licensed phonemizer

VITS, Kokoro, StyleTTS2, and Matcha-TTS drivers all still take raw token-id/demo text input — none of them
do real text→phoneme conversion. Real `espeak-ng`-based phonemization was confirmed to work numerically
(via the external `piper_phonemize` Python package) but vendoring it was rejected: both stock espeak-ng and
the piper-phonemize fork are GPL-3, incompatible with this repo's permissive licensing (see
`[[loom_engine_licensing_phonemizer]]` memory). Current plan: integrate **phoonnx** (a friend-of-the-user's
project) instead, once its license/API are confirmed — not yet investigated. A `src/text/phonemize.cpp` +
`include/loom/text/phonemize.h` split matching this project's existing driver-code conventions is the
intended shape. SupertonicTTS is the one model in this family that's already fully closed — its
`TextVectorizer` is a license-free unicode codepoint lookup table, no phonemizer needed at all.

## Performance optimizations designed but not implemented

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

## Scope limitations (still true)

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

## Minor cleanups

- `KvCache::write_k/write_v/read_k/read_v` use `std::vector::at()` for layer-index bounds checking, which
  throws `std::out_of_range` rather than a `loom::Error` subtype. A malformed topology's `"layer"` attr
  could in principle trigger this uncaught-by-`catch (loom::Error&)` path — low risk today since the layer
  index always comes from `repeat_for`'s own loop bound, not arbitrary user input.
