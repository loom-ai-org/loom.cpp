# Backlog

Deliberate deferrals and known gaps from Milestone 1 (small decoder-only LLM) and Milestone 2 (ASR/vision
encoder primitives), and what's still needed per `SPECIFICATION.md`. None of these are oversights — each
was a scoping decision made explicit in code comments and/or the implementation plan
(`/home/flavio/.claude/plans/mighty-orbiting-quail.md`) at the time.

## Resolved in Milestone 2 (ASR/vision encoder primitives)

- **`CONV_1D`/`CONV_2D`/`POOL_1D`/`POOL_2D`/`GELU` primitives** added (`src/ops/primitives_conv.cpp`,
  `src/ops/primitives_basic.cpp`). `CONV_1D`/`CONV_2D` deliberately do **not** call `ggml_conv_1d`/
  `ggml_conv_2d` directly — those force an F16 im2col cast internally even for F32 inputs, which would
  fight exact-vs-numpy verification (same class of issue as flash attention below). Our primitives
  replicate ggml's own im2col→reshape→mul_mat(→permute→cont) recipe with `ggml_im2col(...,
  GGML_TYPE_F32)` instead, confirmed supported on the CPU backend. Worth remembering if any *other* ggml
  convenience wrapper is added later — check whether it silently downcasts before assuming exactness.
- **`ATTENTION` gained an optional `"kv_cache"` bool attr (default `true`)** rather than a separate
  no-cache op — same composite math, skips `KvCache` read/write and derives `n_kv` from `k`'s own shape
  when `false`. Used by both toy encoders for non-causal (fully unmasked) self-attention. Default `true`
  keeps every Milestone-1 LLM topology's behavior unchanged.
- Validated via `tests/test_e2e_toy_vision.cpp` (`CONV_2D` patch-embed ViT-*shaped* encoder) and
  `tests/test_e2e_toy_asr.cpp` (`CONV_1D`-subsampled Conformer/Zipformer-*shaped* encoder), each checked
  against an independent numpy reference (`tools/fixture_gen/reference_forward_{vision,asr}.py`).

## Resolved in Milestone 3 (ODE-stepper driver + VAE decoder support)

- **`OdeStepper`** (`include/loom/core/ode_stepper.h`, `src/core/ode_stepper.cpp`) drives flow-matching's
  forward-Euler ODE integration loop (SPECIFICATION.md §4), building its graph **once** and genuinely
  reusing it across every integration step (see the root-caused finding below for why this is safe).
  **`CONV_TRANSPOSE_1D`/`CONV_TRANSPOSE_2D`** primitives added (`src/ops/primitives_conv.cpp`), calling
  `ggml_conv_transpose_1d`/`ggml_conv_transpose_2d_p0` directly — unlike `CONV_1D`/`CONV_2D`, these
  dispatch purely on the kernel's own dtype (confirmed in
  `ggml_compute_forward_conv_transpose_1d/2d`: F32 kernel → full F32 compute, no forced F16 cast), so no
  im2col workaround was needed.
- Validated via `tests/test_e2e_toy_ode.cpp` (a small `CONV_1D`+`GELU`+`ADD` vector-field network,
  Euler-integrated and compared step-for-step against `reference_forward_ode.py`) and
  `tests/test_e2e_toy_vae.cpp` (`CONV_TRANSPOSE_1D` upsample → `GELU` → `CONV_1D` refine, a one-shot
  forward pass, no control flow needed).
- **Timestep is a plain scalar broadcast across channels, not a sinusoidal/learned embedding** — the
  topology declares `"timestep"` as `[1, n_embd]` and `OdeStepper` fills it with the same scalar `t`
  every element. Validates the stepping *mechanism* (graph-reuse, correct Euler update, in-place tensor
  overwrite), not a specific embedding scheme; a real flow-matching model computes its timestep embedding
  upstream of the graph the same way a real pipeline already does.

### Root-caused: `ggml_gallocr` may alias a computed tensor's buffer with one of the graph's own declared *input* tensors

`OdeStepper` was originally implemented exactly as designed in the plan — build the graph once, then loop
`cfg.n_steps` times only rewriting `"latent"`/`"timestep"` and recomputing on the same `ggml_cgraph*`,
leaving `"conditioning"` (logically constant across steps) written once before the loop. **This produced
numerically wrong results starting from the second compute call** — step 1 matched the numpy reference to
6 decimal places, step 2 diverged completely.

Isolated outside `GraphBuilder`/`OdeStepper` entirely in `tests/test_graph_reuse_safety.cpp` (plain ggml
calls only, no im2col/`CONV_1D` needed to reproduce it — a bare `ggml_add(a, b)` graph shows the same
thing): after `ggml_gallocr_alloc_graph`, the ADD node's *output* tensor's `->data` pointer was the exact
same address as one of its own *input* tensors (`ggml_set_input()`-marked). This directly contradicts
`ggml-alloc.h`'s documented contract that input tensors get "non-overlapping addresses" — or at least,
that contract only protects inputs from being used as **scratch** by other intermediate nodes during a
single pass, not from being chosen as the **final output buffer** of the one op that consumes them last.
Within a single compute pass this is invisible and harmless; but if the graph is **reused** for a second
compute and that particular input is *not* rewritten (because the caller assumes its logical value hasn't
changed), it now silently holds the *previous pass's output* instead.

**The fix: rewrite every declared input tensor before every `ggml_backend_graph_compute` call on a reused
graph — including ones that never logically change.** `OdeStepper::integrate()` now does exactly this
(rewrites `"latent"`, `"timestep"`, *and* `"conditioning"` every step) and was verified bit-identical to a
from-scratch rebuild at every step. `tests/test_graph_reuse_safety.cpp` pins down all three pieces as
permanent regression tests: (1) reuse with full input refresh matches a fresh rebuild, (2) the unsafe
"skip a logically-constant input" pattern is confirmed to actually corrupt results (so if a future ggml
upgrade changes this aliasing behavior, that assertion will fail and this finding should be revisited),
and (3) the full vector-field-shaped topology (matching `OdeStepper`'s actual graph) reuses correctly
under the same discipline.

**This also means the bucketed KV-cache graph-reuse optimization below is no longer blocked** — the same
"rewrite every declared input every call" discipline should make it safe, though it hasn't been
implemented or verified there yet (a different topology/primitive set, and llama.cpp's own bucketing
scheme has more moving parts than `OdeStepper`'s fixed-shape loop).

#### Corollary bug found later: `test_full_topology_reuse_with_full_refresh_matches_fresh_rebuild` itself violated its own discipline

Discovered 2026-07-17 while investigating a consistent (not flaky) `gelu_erf` `NaN`/`Inf` assertion
failure in this same test. The test's `kernel1`/`kernel2` tensors (standing in for real model weights)
were created in the *same* `GgmlScratch` no_alloc context as the true per-step inputs
(`latent`/`timestep`/`conditioning`) and marked `ggml_set_input()` like them — but, matching what a real
model's *weights* should behave like, were only ever written **once**, before the first `compute()`, never
rewritten before the second. That's exactly the unsafe pattern `test_unrefreshed_input_gets_silently_aliased`
demonstrates two tests above it: a diagnostic readback confirmed `gallocr` had aliased `kernel1`'s buffer
with an intermediate tensor's output during the first `compute()` call, silently corrupting it before the
second call ran and producing the `NaN`s that tripped the assertion (`kernel2` happened not to get
aliased, consistent with `gallocr`'s aliasing being real but not guaranteed for every input).

This was a bug in the *test's own setup*, not a production regression: real weights (`GgufModel::load()`,
`gguf_model.cpp`) and `KvCache`'s persistent K/V storage (`kv_cache.cpp`) both live in their own
`ggml_backend_alloc_ctx_tensors`-backed buffer, entirely separate from the ephemeral no_alloc context
`GraphBuilder`/`gallocr` manages per `build()` call, and are never marked `ggml_set_input()` — so they
were never actually at risk of this aliasing in real usage, only in this test's flawed approximation of
"a weight that doesn't change." **Fixed by giving `kernel1`/`kernel2` their own persistent
context+buffer**, mirroring `KvCache`'s exact allocation pattern, matching how the test already claimed to
model real weight behavior. Verified stable across 5 repeated runs after the fix (was 100% reproducible
before it).

**Lesson, in the same spirit as the length-dependent-constant lesson below**: a test that *simulates* a
production invariant (here: "weights are immune to the reused-graph aliasing risk") must actually
reproduce the mechanism that makes the invariant true (a separate allocation buffer), not just approximate
the *symptom* (an input that happens not to get rewritten) — the two aren't equivalent once the underlying
allocator is involved, and only the real mechanism is guaranteed safe.

## Resolved in Milestone 4 (real model: NVIDIA `stt_en_conformer_ctc_small` Conformer-CTC)

First test against an actual published checkpoint rather than a toy fixture. Converted via
`tools/convert_nemo/convert_conformer_ctc.py` (a plain-PyTorch reader of the `.nemo` tar archive, no
`nemo_toolkit` dependency) and verified against `tools/convert_nemo/reference_forward_conformer.py` (an
independent plain-PyTorch reimplementation of NeMo's published forward pass) in
`tests/test_e2e_conformer_ctc.cpp`, within the usual `1e-3` absolute tolerance.

- **New primitives**: `LAYER_NORM`, `SIGMOID`, `GLU` (sigmoid-gated, `src/ops/primitives_basic.cpp`),
  `CONV_1D_DW` (`src/ops/primitives_conv.cpp`), `REL_POS_ATTENTION` + its `rel_shift` sub-step
  (`src/ops/primitives_attention.cpp`, exposed standalone as `REL_SHIFT` purely for unit testing). All
  have isolated hand-computed tests in `tests/test_primitive_registry.cpp` before being trusted in the
  real model.
- **`ggml_conv_1d_dw` reimplemented rather than called directly** — ggml's own header flags it "very
  likely wrong for some cases! - needs more testing". `op_conv_1d_dw` replicates its im2col+mul_mat recipe
  manually with an F32 im2col (same rationale as the Milestone-2 `CONV_1D`/`CONV_2D` finding: avoid a
  forced-F16 cast fighting exact verification), confirmed against a hand-computed 2-channel test first.
- **`rel_shift` (Transformer-XL relative-position trick)** is a pure flat-memory reinterpretation
  (asymmetric left-pad + two reshapes + one view-slice), not a transpose — confirmed by tracing ggml's
  op sequence against real PyTorch execution on a small hand-picked example before writing any C++.
- **`RESHAPE` gained numpy/PyTorch-style `-1` dimension inference** (`op_reshape`) — needed because the
  post-subsampling frame count isn't a named `GraphBuilder` symbol, only known from the actual input
  tensor's element count at build time.
- **Found and fixed a real bug during first conversion run**: the conv-bias-broadcast helper
  (`broadcast_bias_reshape` in `convert_conformer_ctc.py`) reshaped every 1D bias to `[1, channels, 1]`,
  which is correct for `CONV_1D`'s `ne=[T, channels, N]` output (channels at `ne[1]`) but wrong for
  `CONV_2D`'s `ne=[OW, OH, OC, N]` output (channels at `ne[2]`) — the two subsampling conv2d biases need
  `[1, 1, channels, 1]` instead. Manifested as a `GGML_ASSERT(ggml_can_repeat(b, a))` abort the first time
  the real graph was built; added a separate `broadcast_bias_reshape_2d` helper and confirmed the full
  encoder+decoder forward pass matches the PyTorch reference after the fix. Worth remembering for any
  future architecture mixing `CONV_1D` and `CONV_2D` biases in the same topology: **the channel axis
  position depends on which conv primitive produced the tensor being added to, not just the channel
  count.**
- **Constants folded at conversion time, no new primitives needed**: BatchNorm (eval-mode, folded to
  per-channel scale+shift, same recipe as Milestone 3's VAE), the Conformer's half-step (0.5×) feed-forward
  residual, and `xscale` (`sqrt(d_model)`, applied once post-subsampling) — both scale factors commute
  exactly through the preceding Linear layer's weight/bias.
- **New test-fixture pattern: real-model tests are not regenerated by `ctest`.** Every prior e2e test
  procedurally generates its own toy GGUF + reference at `ctest` time (no network, no PyTorch needed to
  run the suite). `test_e2e_conformer_ctc.cpp` can't do that — it needs the actual ~49MB `.nemo` checkpoint
  and a PyTorch environment to produce its fixture. It instead looks for a pre-built directory
  (`$LOOM_CONFORMER_CTC_DIR`, default `/tmp/nemo_model`, containing `conformer_ctc.gguf` + a `ref/`
  subdirectory of `.bin` dumps) and skips cleanly (exit 77, wired to ctest's `SKIP_RETURN_CODE`) if it's
  absent — `ctest` stays 100% green on a fresh checkout with no real model prepared. If more real-model
  tests are added later, reuse this exact skip convention rather than inventing a new one.

### Still out of scope after Milestone 4

- **No tokenizer/detokenization wired in** — CTC decoding (greedy argmax + collapse-repeats + drop-blank)
  is pure host-side logic, not yet implemented anywhere, and the SentencePiece vocab files aren't consumed.
  Same limitation already tracked above for the LLM path.
- **Only the small (16-layer, `d_model=176`) checkpoint has been tried.** Larger Conformer-CTC variants
  (medium/large, more heads/layers/different `conv_kernel_size`) should work unmodified since the topology
  is generated entirely from `model_config.yaml`, but this hasn't been exercised.

## Resolved in Milestone 5 (mel-spectrogram extraction moved into the graph)

Mel-spectrogram extraction (preemphasis → STFT → power → mel filterbank → log → per-feature CMVN
normalize) is now ordinary graph nodes, generated by `convert_conformer_ctc.py` ahead of the Conformer
subsampling front-end, instead of a caller-supplied precomputed tensor — the declared runtime input is
now `"waveform"` (raw PCM samples), not `"mel_input"`. Verified against real `torch.stft` (not just a
restatement of the same formula) in `reference_forward_conformer.py`, end-to-end through the full encoder
+ decoder, within the usual `1e-3` tolerance.

- **STFT expressed as two `CONV_1D` calls against precomputed DFT-basis kernels** (`mel_common.py`
  computes `cos_kernel`/`sin_kernel`, each shaped `(n_freq, 1, n_fft)`, baked as GGUF constant weights) —
  the same trick used by ONNX-exportable audio frontends since `torch.stft` itself isn't graph-friendly.
  Cross-correlating a framed+windowed signal against a `window[n]·cos(2πkn/N)` kernel is exactly a length-N
  DFT sum, and `nn.Conv1d`/ggml's `CONV_1D` both compute cross-correlation (no kernel flip), so no
  transform beyond precomputing the kernel is needed. Power spectrum is `re²+im²` (`SQR`+`SQR`+`ADD`) —
  algebraically exact at inference since NeMo's own `sqrt`-then-`pow(2.0)` guard is `0` when
  `use_grads=False`.
- **`ggml`'s own zero-padding is exactly what NeMo needs, no reflect-pad primitive required** — confirmed
  from NeMo's source that `AudioToMelSpectrogramPreprocessor`'s default is `pad_mode="constant"` (plain
  zero padding) with `center=True`, and `CONV_1D`'s own `p0` (im2col zero-pad) with `p0 = n_fft // 2`
  reproduces that exactly. (`ggml_pad_reflect_1d` exists and was investigated first, assuming NeMo used
  reflect padding like plain `librosa.stft`'s default — it doesn't; worth double-checking a model's actual
  `pad_mode` before reaching for that primitive on a future architecture.)
- **New primitives** (`src/ops/primitives_basic.cpp`, all one-line `ggml_*` wraps, same pattern as
  `SIGMOID`/`RELU`): `SUB`, `DIV`, `SCALE` (attrs: `"s"`), `SQR`, `SQRT`, `LOG`, `SUM_ROWS` (reduces along
  `ne[0]`; needs a `PERMUTE`+`CONT` first for any other axis, same discipline as every other
  axis-sensitive primitive here), `PAD_1D` (attrs: `"lp0"`/`"rp0"`, wraps `ggml_pad_ext` with every other
  dimension's pad fixed at `0`). All have isolated hand-computed tests in `tests/test_primitive_registry.cpp`.
- **Preemphasis's one boundary sample (`x[0]` unchanged, no synthetic "previous sample") is done via a
  1-sample left `PAD_1D` + two `VIEW`s** rather than a dedicated concat/shift primitive: zero-padding by
  one sample and then computing `x[n] - 0.97·padded[n]` for the *shifted* view reproduces
  `x[0] - 0.97·0 = x[0]` at the boundary for free, avoiding a `CONCAT` primitive that would otherwise only
  exist for this one case.
- **Per-feature (per-mel-bin) CMVN normalize needed reductions along the *time* axis, not `ggml_sum_rows`'s
  native `ne[0]`** — `log_mel`'s layout is `[n_mels, T_mel, 1]` (mel-fastest, matching the rest of the
  topology's convention), so both the mean and unbiased-variance (`N-1`) reductions `PERMUTE`+`CONT` to
  `[T_mel, n_mels, 1]` first, reduce, then `RESHAPE` the `[1, n_mels, 1]` result back to `[n_mels, 1, 1]`
  to broadcast against the original layout — the same "transpose, reduce, transpose back" pattern any
  future axis-1/2/3 reduction in this engine will need until/unless a `SUM`-along-arbitrary-axis primitive
  is added directly.
- **`mel_common.py`** is shared by both the converter and the reference script specifically so the mel
  filterbank (`librosa.filters.mel(..., norm="slaney")`) and DFT-kernel construction can't silently diverge
  between them — the reference deliberately does *not* reuse the converter's conv-based DFT trick itself
  (it calls real `torch.stft`), so passing end-to-end is a genuine check that the trick is correct, not
  just self-consistent.
- **`waveform`'s declared length (`n_tokens=10240` samples, 0.64s @ 16kHz) is hardcoded at conversion
  time**, same "fixed shape, `--n-samples` override, documented limitation" precedent as the
  pre-subsampling/subsampled frame counts already were before this milestone (see below) — mel-frame count
  and subsampled-frame count are now *both* derived from it in Python rather than independently hardcoded.

### Still out of scope after Milestone 5

- **Raw waveform sample count (`n_tokens=10240`) is hardcoded at conversion time**, not dynamically
  derived — same underlying limitation as Milestone 4's pre-subsampling/subsampled frame counts, just
  pushed one level earlier (to raw samples instead of mel frames): `pos_emb_raw`/`kq_mask`'s shapes must
  be fixed when the topology JSON is written, and `GraphBuilder` exposes only one runtime symbol
  (`n_tokens`). Supporting arbitrary-length audio in one converted file would need either a mechanism for
  a topology to declare a *derived* runtime symbol, or a small per-length family of pre-baked topologies.
  `convert_conformer_ctc.py` accepts a `--n-samples` override for regenerating at a different fixed length.
- **Dither (train-time-only Gaussian noise) is correctly omitted** since inference never applies it
  (`self.training` is always `False` here) — not a gap, just noting it was checked, not assumed.

## Resolved in Milestone 6 (tokenizer: SentencePiece unigram, llama.cpp GGUF schema)

CTC output can now be turned into real text, and text can be turned into real token ids — both ends of
the "no tokenizer" gap tracked since Milestone 1 are closed for the SentencePiece-unigram case. Follows
llama.cpp's own design closely, per an explicit user request: the GGUF KV schema
(`tokenizer.ggml.model/tokens/scores/token_type/unknown_token_id/add_space_prefix/
remove_extra_whitespaces/precompiled_charsmap`), its `llama_token_type` enum (numerically identical to
SentencePiece's own protobuf `Type` enum — copied straight through, no remapping), and — the highest-risk
piece — its exact XCDA (XOR-compressed compact double array) `precompiled_charsmap` normalizer and
unigram Viterbi segmentation algorithm, both confirmed verbatim from `llama.cpp`'s
`llm_tokenizer_ugm`/`llm_tokenizer_ugm_session` (`src/llama-vocab.cpp`) before writing any C++.

- **`GgufModel` gained generic KV accessors** (`has_kv`, `kv_str`, `kv_bool`, `kv_i32`, `kv_arr_str`,
  `kv_arr_f32`, `kv_arr_i32`, `kv_arr_u8`) — previously only scalar `loom.*`-namespaced hparams were
  readable; nothing read array- or bool-typed KVs, confirmed by grep before building on top of ggml's own
  (previously unused in this codebase) `gguf_get_arr_type/_n/_data/_str`. These take a full key, unlike
  `hparam_*`, since `tokenizer.ggml.*` is a different namespace than `loom.*`.
- **`Vocab`** (`include/loom/core/vocab.h`, `src/core/vocab.cpp`): loads a SentencePiece-unigram vocab
  from `tokenizer.ggml.*` KVs. `encode()` mirrors `llm_tokenizer_ugm_session::tokenize` closely — a
  `naive_trie`-style token matcher (`std::map<char, TrieNode>`, matching llama.cpp's own structure choice
  rather than `unordered_map`) feeding a Viterbi best-path DP over piece scores, with an
  unknown-codepoint fallback (`min_score - 10.0`) guaranteeing the DP always has *some* path.
  `normalize()`/`normalize_prefix()` mirror the XCDA bit-unpacking (`get_base`/`get_lcheck`/`get_leaf`/
  `get_value`) and prefix-replacement-table walk exactly. **Verified bit-exact against the real
  `sentencepiece` Python library on the same model file** on every test case on the first real attempt,
  including the tricky parts: lowercasing (via the charsmap, not a special-cased rule), multi-space
  collapsing, and non-ASCII input genuinely walking the XCDA (not just an identity no-op) — see
  `tests/test_vocab.cpp`.
- **`decode()`'s one intentional divergence from real `sentencepiece`**: a decoded `<unk>` renders as the
  literal piece text `"<unk>"`, not `sentencepiece`'s cosmetic `" ⁇ "` glyph substitution. Not needed for
  the CTC-decode use case (a well-trained CTC acoustic model essentially never predicts the `<unk>`
  class at inference — it's a training-time artifact of text tokenization, not something the model
  itself emits), and not part of what `encode()`'s correctness (checked via exact id-sequence match,
  independent of how any resulting `<unk>` is later rendered) depends on.
- **`ctc_greedy_decode`** (`include/loom/core/ctc_decode.h`, `src/core/ctc_decode.cpp`): pure host-side
  per-frame-argmax + collapse-repeats + drop-blank, same "host logic, not a graph primitive" precedent as
  `Generator::argmax`. Isolated hand-computed test in `tests/test_ctc_decode.cpp`.
- **`tools/convert_nemo/tokenizer_common.py`** (new, shared like `mel_common.py`): parses the `.model`
  protobuf directly via `sentencepiece.sentencepiece_model_pb2.ModelProto` (not the
  `SentencePieceProcessor` wrapper, which doesn't expose `precompiled_charsmap` or the normalizer flags)
  and writes `tokenizer.ggml.*` via `GGUFWriter`'s existing named helpers. `nemo_common.load_nemo()` now
  also returns the tokenizer `.model` file's raw bytes.
- **New real-model hparams**: `loom.n_samples`, `loom.n_subsampled`, `loom.n_pos`, `loom.num_classes` are
  now stored as first-class KVs (previously only baked into the topology JSON), so `loom_cli` and tests
  can read the model's fixed shapes back instead of hardcoding them — mirrors llama.cpp's own practice of
  storing real hyperparameters as KVs rather than burying them in an opaque blob.
- **`loom_cli --wav`**: a from-scratch minimal 16-bit-PCM-mono WAV reader (`tools/loom_cli/wav_file.*`,
  CLI-only, not part of the engine library) plus a C++ port of the already-verified sinusoidal
  positional-embedding formula (needed since `pos_emb_raw` is a required input, not zero-fillable) runs
  the full waveform→mel-frontend→encoder→CTC-decode→detokenize pipeline standalone. Verified against a
  real 16kHz speech recording (whisper.cpp's bundled `samples/jfk.wav`) — correctly transcribed the start
  of "And so, my fellow Americans..." (truncated to the model's fixed 0.64s input length) as `"andnce"`,
  confirming the whole real-audio pipeline (not just synthetic-noise fixtures) works end to end.

### Still out of scope after Milestone 6

- **Only the UGM (SentencePiece unigram) vocab type is implemented.** SPM (llama.cpp's own
  byte-level-BPE-with-scores convention), BPE (GPT-2-style, byte-to-"Ġ"), and WPM (BERT-style,
  "##"-continuation) all encode/decode differently and aren't implemented — `Vocab::load` throws if
  `tokenizer.ggml.model` isn't `"t5"`. No model converted so far needs them.
- **No LLM checkpoint has been converted with a vocab yet** — `loom_cli --prompt` still takes
  whitespace-separated integer token ids, not text, even though `Vocab::encode`/`decode` are now general
  enough to support it. Wiring this up needs an actual LLM `.gguf` conversion script that also writes
  `tokenizer.ggml.*` KVs (none exists in this repo yet — only the Conformer-CTC converter does).

## Resolved in Milestone 7 (genuinely dynamic sequence length)

`loom_cli --wav` no longer pads/truncates real audio to a fixed length — any waveform length now
"just works" against the *same* converted GGUF, matching `SPECIFICATION.md` §4's original design
("rebuilding the compute graph from scratch for every forward pass... injecting the exact
dimensions"). Per an explicit user scope choice: **dynamic shapes only** — a very long recording still
works correctly, it just scales with relative-position attention's usual O(n²) cost, same as real
NeMo's own full-attention encoder; chunked/windowed inference for long-audio performance stays a
documented future optimization (see below), not part of this pass.

- **The gap wasn't a missing mechanism, it was two unwired hardcoded numbers.** `GraphTopology`'s
  declared-input `shape` entries were *already* full `SymbolEnv` expressions evaluated fresh on every
  `GraphBuilder::build()` call (`graph_builder.cpp`'s `env.eval(dim)`) — the same mechanism attrs like
  `"1/sqrt($head_dim)"` already used, and `"waveform"`'s shape was already `["n_tokens", "1", "1"]`.
  Only `pos_emb_raw`/`kq_mask` — whose sizes depend on `n_subsampled`, a non-trivial function of
  `n_tokens` via the mel-frontend's STFT-conv stride then the Conformer's own two subsampling convs —
  had that function evaluated once in Python instead of expressed as a formula.
- **`SymbolEnv` gained `floor(...)`** (`src/core/symbol_env.cpp`, next to the existing `sqrt(...)`
  handling) — a real, previously-latent correctness gap, not just a missing nicety: the conv
  output-length formula (`floor((in + 2·pad − kernel)/stride) + 1`) needs floor, but
  `GraphBuilder::build()` rounds every evaluated shape dimension via `std::llround()`, which rounds
  halfway values *away from zero*. For an even-length input this would have silently produced an
  off-by-one shape (e.g. `llround(31.5) == 32` is fine, but `llround(32.5) == 33` where the correct
  floored value is `32`) had the expression relied on `llround` alone instead of calling `floor()`
  explicitly inside the expression string.
- **`convert_conformer_ctc.py`** now builds `pos_emb_raw`/`kq_mask`'s shapes as nested `$n_tokens`
  expression strings (`conv_stride_out_expr`/`n_subsampled_expr`/`n_pos_expr`), composed the same way
  the equivalent Python numbers were computed before — same formula, now expressed as text the engine
  evaluates per call instead of a number baked in once. `loom.n_samples`/`n_subsampled`/`n_pos` hparam
  KVs are kept, but now document only the *default* length used to regenerate the bundled test fixture,
  not a hard constraint on real usage.
- **`loom_cli --wav`** now calls `build()` with the real WAV's actual sample count and reads
  `n_subsampled`/`n_pos` back from the tensors `GraphBuilder` just allocated
  (`kq_mask->ne[0]`/`pos_emb_raw->ne[1]`), not from the conversion-time hparams.

### Found and fixed during verification: the mel-frontend's CMVN normalize also had a hardcoded length

The dynamic-shape fix above (`pos_emb_raw`/`kq_mask`) alone was NOT sufficient — first-pass testing at a
second length (1 second / 16000 samples) produced logits diverging from the independent PyTorch
reference by >10 (vs. the usual `1e-3` bar), and the full untruncated `jfk.wav` (11s) produced an
**empty** transcript (every frame's argmax was the blank class) despite no `NaN`s and no crash — a
classic silent-wrong-shape symptom, not a numerics blowup, so worth root-causing rather than shrugging
off as "the model just isn't confident."

Root cause: the mel-frontend's per-feature CMVN normalize (Milestone 5) divides by the mel-frame count
`T_mel` twice (once for the mean, once for the unbiased variance), and both divisors were written as
**fixed Python numbers** (`1.0 / hp["t_mel"]`, `1.0 / (hp["t_mel"] - 1)`) computed from the
conversion-time default length (`t_mel=65`) — exactly the same class of bug the shape fix above was
written to eliminate, just in a different pair of `SCALE` node attrs that got missed on the first pass.
For any waveform length other than the original default, this silently normalized by the *wrong*
denominator (e.g. at 16000 samples, `T_mel=101` but the graph divided by the old fixed `65`/`64`),
increasingly corrupting the encoder's input the further the actual length diverged from the default —
consistent with what was observed: exact match at the default 10240 samples, small divergence at
16000, and complete breakdown (all-blank) at 176000.

Fixed by using the *same* `t_mel_expr(hp)` symbol-expression helper already written for the shape
formulas above, so both `SCALE` nodes' `"s"` attrs are now `$n_tokens` expressions too (`SCALE`'s `"s"`
attr already supported this — `resolve_attr_number()` evaluates via `SymbolEnv` same as any other attr,
so no primitive changes were needed, just fixing the two call sites). Re-verified: bit-exact again at
16000 samples (`test_e2e_conformer_ctc_dynamic_length`), and `loom_cli --wav` on the full untruncated
`jfk.wav` now produces a **word-for-word correct transcript** — `"and so my fellow americans ask not
what your country can do for you ask what you can do for your country"` — matching the actual audio
content exactly, not just "some plausible words."

**Lesson for any future length-dependent constant in this codebase**: grep for every Python-computed
number derived from a length hyperparameter (`t_mel`, `n_subsampled`, `n_pos`, or similar) that ends up
baked into a topology node's `attrs`, not just into a declared input's `shape` — both are equally
capable of hiding a "only correct at the default length" bug, and only one of the two is obvious to
audit by inspecting `"inputs"` alone.

### Still out of scope after Milestone 7

- **No chunked/windowed inference for long audio** — a multi-minute recording works correctly but its
  cost grows O(n²) with length (relative-position attention over the whole clip at once), per the
  user's explicit choice to defer this. Would need window size/overlap selection and stitching
  together per-window CTC token sequences at the boundaries.
- Everything else from Milestone 6's "still out of scope" list (vocab types beyond UGM, no LLM
  checkpoint with a vocab yet) is unchanged.

## Resolved in Milestone 8 (real model: `Qwen/Qwen3-0.6B-Base`, byte-level BPE tokenizer)

Closes the "Qwen3-0.6B (base LLM)" roadmap item below — a real, published 0.6B-parameter checkpoint now
runs end to end through loom-engine's *generic* topology-interpretation path, with no bespoke C++ needed
for this architecture. Verified against an independent numpy/PyTorch reference
(`tools/convert_qwen3/reference_forward_qwen3.py`) and manually smoke-tested via `loom_cli --prompt "The
capital of France is"`, which produced the coherent, semantically correct continuation `" Paris. The
capital of Germany is Berlin. The capital of"`.

**Every piece composes existing primitives — nothing new was added to `PrimitiveRegistry`:**

- **QK-norm**: an existing `RMS_NORM` node on `q`/`k` (reshaped to per-head `[head_dim, n_head(_kv),
  n_tokens]`) inserted before `ROPE`, weight shape `[head_dim]` applied via the existing `MUL` broadcast
  pattern. Confirmed directly against the real checkpoint's safetensors header before converting anything:
  `self_attn.q_norm.weight`/`k_norm.weight` are genuinely `[128]` (== `head_dim`, not `[n_head*head_dim]`),
  exactly the per-head design assumed.
- **GQA**: `ATTENTION`'s existing `ggml_mul_mat(kp, qp)` broadcast (`n_head_kv -> n_head`, requires
  `n_head % n_head_kv == 0`) needed no changes — but per this project's "verify before trusting an
  existing mechanism in a new configuration" discipline, it had never actually been exercised with
  `n_head_kv < n_head` before this milestone (the Milestone-1 toy LLM uses `n_head == n_head_kv == 2`).
  **New regression test `tests/test_e2e_gqa.cpp`** (fixture: `tools/fixture_gen/gqa_test_common.py`, 4
  query / 2 KV heads) proves GQA *and* QK-norm's two-extra-broadcast-dim `MUL` together, against a numpy
  reference, *before* the real 16-query/8-KV-head checkpoint depended on either — this caught nothing
  wrong (both worked first try), but was written and run before conversion, not after, per that
  discipline.
- **Tied input/output embeddings**: needed no engine change at all, confirmed by directly inspecting the
  checkpoint (`tie_word_embeddings: true` in `config.json`, and indeed no separate `lm_head.weight` tensor
  in `model.safetensors`) — `convert_qwen3.py`'s topology just references `"token_embd.weight"` by name
  from both the initial `GET_ROWS` and the final logits `MUL_MAT`; `GraphBuilder`'s symbol table already
  resolves a name to the same tensor wherever referenced.
- **`head_dim` is an independent hparam, not `n_embd/n_head`**: confirmed from the real safetensors shapes
  before writing any conversion code (per the plan's explicit first step) — `q_proj.weight` is
  `[2048, 1024]` (16 heads × 128 `head_dim`, projecting *up* from `hidden_size=1024`), `k_proj`/`v_proj`
  are `[1024, 1024]` (8 KV heads × 128), `o_proj` is `[1024, 2048]` (projecting back down). Already fully
  expressible by the existing topology grammar (every `RESHAPE`/`ROPE` dimension is a named symbol, never
  assumed derived from `n_embd`/`n_head`), so this needed no engine change either — just correct hparam
  KVs (`n_embd_head_k`/`n_embd_head_v` = 128, distinct from `n_embd` = 1024).

**A real byte-level BPE tokenizer, with full Unicode fidelity (per explicit user decision, not an
ASCII-only approximation)** — the one genuine new engine subsystem this milestone added, closing the gap
tracked since Milestone 6 ("Only the UGM... vocab type is implemented") and finally letting `loom_cli
--prompt` take real text for an LLM instead of raw token ids:

- **`tools/codegen/gen_unicode_tables.py`** (one-off, NOT part of the CMake build, output checked in as
  `include/loom/core/unicode_data.h`): derives `\p{L}`/`\p{N}` category range tables, a canonical
  decomposition map, a combining-class table, and a composition-exclusion set from Python's stdlib
  `unicodedata` (Unicode 14.0.0) plus the Unicode Character Database's own `CompositionExclusions.txt`
  (fetched once — NOT derivable from `unicodedata` alone, and skipping it would make NFC composition
  silently *wrong*, not just incomplete, for the characters it lists). The C++ engine itself has no
  runtime Python dependency; only this generator does, and it's never invoked at build or convert time.
- **`include/loom/core/unicode.{h,cpp}`**: hand-implements the UAX #15 NFC algorithm (canonical
  decomposition — including Hangul, which is deliberately *absent* from the generated table since
  UnicodeData.txt specifies it algorithmically instead of listing ~11172 mappings — canonical ordering,
  then canonical composition against the generated tables) plus `is_letter`/`is_number`. Unit-tested in
  isolation (`tests/test_unicode.cpp`) against known recomposition cases before `BpeVocab` ever depended
  on it.
- **`include/loom/core/bpe_vocab.{h,cpp}`**: new `BpeVocab` class (the sibling `vocab.h`'s doc comment
  already reserved: "BPE's byte-to-'Ġ' convention... decode/encode differently"), reading llama.cpp's own
  real `tokenizer.ggml.model="gpt2"` GGUF schema (confirmed against the installed `gguf` package's
  `GGUFWriter` methods, not assumed). `encode()` NFC-normalizes, then runs a **hand-written scanner**
  reproducing the real Qwen2/Qwen3 tokenizer.json's fixed pretokenizer regex exactly (confirmed by
  fetching the actual `tokenizer.json` and reading its `pre_tokenizer` field, not assumed from general
  GPT2 knowledge) — same "hardcode the one known fixed pattern as a manual scanner" approach llama.cpp
  itself uses for GPT2-family regexes, since `\p{L}`/`\p{N}` aren't expressible via `std::regex`. Then
  GPT2's standard byte↔unicode-codepoint mapping, then greedy BPE merge per pretokenizer chunk.
  `tests/test_bpe_vocab.cpp` hand-traces exact expected token ids for plain-ASCII cases (a word fully
  reduced by merges, digit-by-digit number splitting — `\p{N}` has no quantifier in the real regex, so
  `"12"` never merges into one piece even though nothing else prevents it, a deliberate easy-to-get-wrong
  detail) against a small synthetic fixture, plus round-trip checks for CJK and NFC-recomposition cases
  the tiny fixture can't hand-trace ids for.
- **Deliberately narrower than full tiktoken/HF-tokenizers generality**: this pretokenizer scanner only
  implements the *specific* fixed regex Qwen2/Qwen3's `tokenizer.json` declares — it is not a general BPE
  pretokenizer-configuration interpreter. The `\s` whitespace class used by the scanner is ASCII whitespace
  only (space/tab/CR/LF/VT/FF), not the full Unicode `White_Space` property some rare Unicode space
  characters have — a narrower scope than the `\p{L}`/`\p{N}`/NFC fidelity elsewhere in this subsystem,
  chosen pragmatically since it only affects extremely uncommon input.

**Conversion tooling** (`tools/convert_qwen3/`, mirrors `tools/convert_nemo/`'s established layout):
`convert_qwen3.py` reads `config.json` + `model.safetensors` + `tokenizer.json` directly — deliberately
avoiding the `transformers` library entirely (same precedent as `tools/convert_nemo/`, which hand-parses a
`.nemo` archive instead of depending on the NeMo toolkit), using the `safetensors` package's own torch
loader only to cast BF16 → F32 (numpy has no native bfloat16). `qwen3_tokenizer.py` parses
`tokenizer.json`'s `model.vocab`/`model.merges` directly via plain `json` (no `tokenizers` library
needed) — confirmed the real vocab (151643 entries, ids contiguous 0..151642) plus 22 `added_tokens`
(151643..151664, no overlap) don't fill the checkpoint's full `vocab_size` (151936); the remaining ids are
unused/reserved embedding-matrix rows, padded with empty-string placeholders so `BpeVocab::id_to_piece`
never throws for the full valid id range.

**Target is specifically Qwen3-0.6B-**Base** (not the instruct/reasoning checkpoint)** — no chat template
(ChatML/Jinja) rendering and no `<think>` reasoning-block special-token handling were needed or attempted;
`loom_cli --prompt` is plain raw-text continuation, matching the Milestone-1 toy LLM's demo shape exactly.
Picking up the instruct/reasoning variant is separate, not-yet-started future work.

### Still out of scope after Milestone 8

- Everything the "Deliberately narrower..." bullet above already covers (ASCII-only `\s`, this specific
  fixed pretokenizer regex only).
- The instruct/reasoning Qwen3-0.6B checkpoint (chat template, `<think>` handling) — see above.
- Qwen3-ASR-0.6B and Qwen3-TTS-0.6B (12Hz-Base) remain fully unstarted roadmap items — see below. Nothing
  about this milestone's audio/TTS-integration gaps changed; this milestone only touched the base LLM.

## Performance optimizations designed but not implemented

- **Bucketed KV-cache graph-reuse.** `GraphBuilder::build()` always does a full rebuild + no_alloc pass
  per call (the SPECIFICATION.md §4 correctness baseline). The plan designed a llama.cpp-style
  optimization — round `n_kv` up to a `kv_pad` bucket boundary (e.g. 32) and skip the rebuild when the
  bucket hasn't changed, reusing the previous `ggml_cgraph*` and just overwriting input tensor data — but
  it was judged unnecessary for milestone-1 correctness and not built. **Now that the graph-reuse finding
  above is root-caused, this is safe to attempt** as long as every declared input (`tokens`, `positions`,
  `kq_mask` — not just whichever ones logically changed) is rewritten every decode step; still needs its
  own bit-identical-to-rebuild regression test (same pattern as `test_graph_reuse_safety.cpp`) before
  trusting it in the generation loop. See `GraphBuilder::reserve()`/`GraphBuilder::build()` in
  `src/core/graph_builder.cpp`.
- **`ggml_backend_sched`.** Not used anywhere — the engine talks to a single `ggml_backend_t` directly via
  a plain `ggml_gallocr`. Fine for CPU-only; becomes necessary once a second backend (CUDA/Metal) is
  added and graphs need splitting across devices.
- **Flash attention.** `ATTENTION` (`src/ops/primitives_attention.cpp`) always uses the composite
  (`MUL_MAT`→`soft_max_ext`→`MUL_MAT`) path, chosen for milestone 1 because `ggml_flash_attn_ext` forces
  an F16 K/V cast that fights exact fp32 verification against the numpy reference. A `FLASH_ATTENTION`
  primitive can be registered later as a purely additive alternative once a GPU backend makes the
  perf/precision tradeoff worth it.

## Scope limitations (single-sequence, milestone-1 LLM only)

- **`KvCache` is single-sequence.** Contiguous append only — no ring buffer, no multi-stream/multi-sequence
  support, no `ggml_set_rows` index-tensor indirection like llama.cpp's `llama_kv_cache`. Needed before
  this engine can serve concurrent/batched sequences.
- **KV cache storage is always F32.** No quantized cache types (`Q8_0` etc.) like llama.cpp supports for
  memory savings on long contexts.
- **Sampling is greedy argmax only** (`Generator::argmax` in `src/core/generation.cpp`). No temperature,
  top-k, top-p, or repetition penalty.
- **No tokenizer.** `loom_cli --prompt` takes whitespace-separated integer token ids, not text. Any
  real-model demo needs a tokenizer wired in (likely vendored from an existing BPE/SentencePiece impl
  rather than hand-rolled).
- **Only one level of `repeat_for` nesting** is supported in the JSON graph schema
  (`include/loom/core/graph_topology.h`'s `RepeatBlock::nodes` is a flat `vector<TopologyNode>`, not
  recursive). Sufficient for a flat per-transformer-layer block; would need restructuring into a proper
  recursive `TopologyItem` variant if a future architecture needs nested repetition.
- **`GgufModel::hparam_env()` only surfaces numeric scalar KV types** (u8/i8/.../f64) into the `SymbolEnv`;
  string, bool, and array-typed `loom.*` KVs are silently skipped since they aren't expression-evaluable.
  Fine today since no schema needs a non-numeric hparam at build time.

## Scope limitations from Milestone 2 (generic encoder, not a faithful replica)

Per an explicit scoping decision, Milestone 2 validated the *primitives* Zipformer/ViT need via a generic
conv-subsampled transformer encoder pattern, not either paper's actual architecture. Still missing if a
faithful reproduction is ever wanted:

- **No positional encoding in the toy encoders at all** — no RoPE, no learned/absolute position
  embeddings. Fine for validating conv+attention wiring (self-attention doesn't require position info);
  a real ViT needs learned absolute position embeddings added post-patch-embed, and a real
  Zipformer/Conformer needs its own relative positional attention variant (neither is standard RoPE).
- **No Zipformer-specific structure**: no multi-branch parallel downsampling, no bypass/scale modules, no
  custom per-branch normalization. **No ViT-specific structure**: no class token, no interpolatable
  position-embedding grid.
- **Only tested with batch N=1** and a single conv layer (one subsampling/patch-embed step, not a stack).
  `CONV_1D`/`CONV_2D`'s N-dimension handling is implemented generically (mirrors ggml's own broadcasting)
  but not exercised at N>1 by any test.
- **`POOL_1D`/`POOL_2D` are registered and unit-tested in isolation** (`tests/test_primitive_registry.cpp`)
  but not exercised by either toy encoder fixture — neither needed pooling given the chosen architectures.

## Scope limitations from Milestone 3 (generic vector field/decoder, not a faithful replica)

Same "generic, not faithful" precedent as Milestone 2:

- **No real timestep embedding** (see above) and **no learned conditioning projection** — `"conditioning"`
  is a fixed `[1, n_embd]` additive bias, not e.g. cross-attention over a text/phoneme encoder's output.
- **No published flow-matching architecture** (Matcha-TTS/VoiceBox/F5-TTS-style U-Net or DiT blocks,
  classifier-free guidance, ODE solvers beyond first-order Euler) and **no published VAE architecture**
  (no residual blocks, no attention, no multi-scale upsampling stack) — both toy networks are two-conv-
  layer minimal examples that exercise the primitives/control-flow, not realistic depth.
- **`CONV_TRANSPOSE_2D` is registered and unit-tested in isolation** (`tests/test_primitive_registry.cpp`)
  but not exercised by the toy VAE fixture — TTS output (audio/mel) is 1D, so only `CONV_TRANSPOSE_1D` was
  needed there; same pattern as `POOL_1D`/`POOL_2D` from Milestone 2.
- **`OdeStepper` only integrates a single "sequence"** (no batching across multiple independent latents in
  one call) and only supports first-order forward Euler (no RK4/adaptive step size/other ODE solvers).

## Minor cleanups

- `KvCache::write_k/write_v/read_k/read_v` use `std::vector::at()` for layer-index bounds checking, which
  throws `std::out_of_range` rather than a `loom::Error` subtype. A malformed topology's `"layer"` attr
  could in principle trigger this uncaught-by-`catch (loom::Error&)` path; low risk today since the layer
  index always comes from `repeat_for`'s own loop bound, not arbitrary user input.

## Roadmap: small TTS models (VITS, StyleTTS2, Kokoro, FastConformer-CTC/RNN-T, Parakeet)

Not started — captured here per an explicit request to record what enabling this model family would
need, cross-checked against what this engine *already* has (Milestones 1-7) rather than assumed from
general `ggml` knowledge. Two things are already true that change the shape of this work: (1) the
Conformer-CTC work already validated a real, published encoder family — FastConformer's encoder is the
same subsampled-Conformer shape with a lighter depthwise-separable subsampling front-end, not a new
architecture class — and (2) `SPECIFICATION.md` §4 ("The TTS Catch") already prescribes the exact
pattern needed for autoregressive decoding: JSON defines the static sub-graph, C++ drives the loop and
feeds state back in between calls. `OdeStepper` (Milestone 3) and `Generator` (Milestone 1) are both
already real instances of that pattern — a Transducer decoder is a third, not a new paradigm.

### Already covered by existing primitives (verified against the actual registry, not assumed)

- **ConvNeXt-style blocks** (used in StyleTTS2's decoder and elsewhere): depthwise conv + layer norm +
  pointwise MLP + GELU is fully expressible today with `CONV_1D_DW` + `LAYER_NORM` + `MUL_MAT` + `GELU`
  — no new primitive needed.
- **Dilated convolution stacks** (WaveNet-style, used throughout HiFi-GAN/VITS's MRF residual blocks):
  `CONV_1D`/`CONV_1D_DW` already take a `"d0"` dilation attr (confirmed in
  `src/ops/primitives_conv.cpp` — added for the Conformer depthwise conv, dilation was already plumbed
  through generically). No gap here despite this being flagged as a blocker in general — it's a solved
  problem *in this codebase specifically*.
- **Transposed-convolution upsampling** (VITS/HiFi-GAN's decoder): `CONV_TRANSPOSE_1D` already exists
  (Milestone 3) and calls `ggml_conv_transpose_1d` directly — not `ggml_conv_1d`'s forced-F16-im2col
  path, so the earlier F16-precision concern that motivated re-deriving `CONV_1D`/`CONV_1D_DW` from
  scratch doesn't apply here. The real gap: `ggml_conv_transpose_1d` itself only accepts `stride`
  (forces `padding=0, dilation=1` — a `ggml` API limit, not a choice made in this engine), so achieving
  HiFi-GAN's typical exact-upsampling-ratio output length may need a post-hoc `VIEW`-based crop node in
  the topology (already expressible with the existing `VIEW` primitive) rather than a new primitive.
  Needs a real hand-verified test against a real HiFi-GAN-shaped block before trusting this, same
  discipline as every other primitive here — not yet attempted at anything beyond the Milestone-3 toy
  VAE's minimal usage.
- **`LeakyReLU`** (HiFi-GAN's standard activation, distinct from `RELU`) and **`tanh`** (needed below for
  LSTM gates) are both already native `ggml` ops (`ggml_leaky_relu`, `ggml_tanh`) — trivial one-line
  wraps in the existing `SIGMOID`/`RELU` style, not yet registered.

### Gap 1: the Transducer problem (RNN-T/TDT — Parakeet, FastConformer-RNN-T)

Confirmed real: CTC's greedy argmax (`ctc_greedy_decode`) is a pure per-frame reduction with no
recurrence, but a Transducer's prediction network (LSTM, stateful across predicted tokens) and joint
network (encoder-frame × prediction-state → per-step token/duration distribution) form a genuine
autoregressive lattice search — not expressible as a single static DAG.

- **No LSTM/GRU primitive exists.** Would need either a monolithic `LSTM_STEP` primitive (one gate
  computation per call: 4 gate matmuls + `SIGMOID`/`tanh`/elementwise `MUL`/`ADD`, all already-available
  ops composed into one primitive for convenience) or expressing the gates as plain composite JSON nodes
  (`MUL_MAT`+`ADD`+`SIGMOID`+`tanh`+`MUL`, no new primitive at all, matching this project's general
  preference for composing existing ops over adding monolithic ones — e.g. `ATTENTION`/
  `REL_POS_ATTENTION` are composites, not opaque black boxes). Needs `ggml_concat` (also unused today,
  also native) if the gate weights expect `[h, x]` concatenated rather than two separate matmuls summed
  — the latter is mathematically equivalent and avoids needing `CONCAT` at all, same "avoid a new
  primitive when an existing composition works" precedent as everywhere else in this backlog.
- **The decode loop itself** would be a new C++ driver analogous to `Generator`/`OdeStepper`: build the
  encoder sub-graph once (a real, faithful FastConformer encoder, extending the already-verified
  Conformer-CTC work), then step token-by-token maintaining the LSTM hidden/cell state and the current
  encoder-frame pointer host-side, rewriting every declared input before every reused-graph compute call
  (the graph-reuse discipline from Milestone 3's root-caused `ggml_gallocr` finding applies identically
  here — this is exactly the kind of new control-flow driver that finding was written to protect against
  getting wrong again).
- **TDT specifically** (Parakeet's variant) predicts a token *and* a frame-advancement duration jointly
  per step, changing the loop's frame-pointer update rule from "always advance by 1" to "advance by the
  predicted duration" — a host-side loop-condition change, not an engine primitive change.

#### Gap 1 progress, 2026-07-18: LSTM composite + joint network + the new `TdtDecoder` driver, proven on a synthetic fixture

Real NeMo source read directly first (`nemo/collections/asr/modules/rnnt.py`, `rnnt_greedy_decoding.py`/
`tdt_label_looping.py`), not assumed: the prediction network is a plain `nn.LSTM` fed the previous emitted
token's embedding; the joint network is `Linear(encoder_frame) + Linear(decoder_step)` summed, `RELU`,
one final `Linear` whose output is a *single* combined vector — the first `n_vocab+1` entries are token+
blank logits, the **last `n_durations` entries are the duration head** (argmax over a small fixed discrete
set like `[0,1,2,3,4]`, not a regression); TDT reuses plain `RNNTJoint`/`RNNTDecoder` unchanged, just with
a wider final linear — there is no separate `TDTJoint` class. Real greedy decode is a double loop: outer
over encoder frames, inner (bounded by `max_symbols_per_step`) repeatedly running one LSTM+joint step;
non-blank → emit and advance LSTM state; blank is *forced* to duration ≥ 1 so decoding can't spin forever;
the frame pointer advances by the argmax'd duration (0 means "stay and loop again").

**Built and verified, in order, each against its own known-good reference before moving on:**

1. **`TANH`** — trivial, `ggml_tanh` already native, one-line `LOOM_REGISTER_OP` same as `SIGMOID`/`SILU`.
   Unit-tested in `tests/test_primitive_registry.cpp` alongside the existing `SIGMOID`/`RELU` checks.
2. **LSTM step as a composite JSON pattern, not a monolithic primitive** — confirmed the preferred design
   from this section's own earlier analysis: `MUL_MAT(W_ih,embed)+MUL_MAT(W_hh,h)+b_ih+b_hh`, sliced via
   existing `VIEW` (byte offsets) into `i/f/g/o`, `SIGMOID`/`TANH`, combined via existing `MUL`/`ADD`.
   `tools/fixture_gen/lstm_step_common.py` + `tests/test_lstm_step.cpp`: verified bit-exact
   (`<=1e-5`) against an independent numpy reference on a tiny synthetic fixture, matching real
   `torch.nn.LSTM`'s per-step convention (gates packed `i,f,g,o`, separate `bias_ih`/`bias_hh`) — passed
   on the first try.
3. **The joint network needed zero new primitives** — `MUL_MAT`+`ADD` (the two linears), broadcast-`ADD`
   (the sum), `RELU` — all already proven elsewhere in this project.
4. **The new `TdtDecoder` C++ driver** (`include/loom/core/tdt_decoder.h`, `src/core/tdt_decoder.cpp`,
   analogous to `Generator`/`OdeStepper`): implements the exact double loop above. Needs **three**
   `GraphTopology` objects, not one — confirmed `GraphTopology`/`GraphBuilder` only support a single
   declared output per topology (`src/core/graph_topology.cpp`), so a schema extension for multiple named
   outputs was considered and **deliberately deferred**: this driver's correctness-first version just
   calls `GraphBuilder::build()` three times per inner-loop step (`lstm_h`, `lstm_c`, `joint` topologies,
   identical weights) — the exact same "just rebuild every step, don't bother caching" simplicity
   `Generator` itself already uses as its baseline (confirmed by reading `generation.cpp`: it does a full
   `builder_.build()` on *every* decode step already, not a graph-reuse optimization — that's a separate,
   still-unimplemented item under "Performance optimizations" above). The multi-output schema question is
   now a real, concrete follow-up (3× redundant rebuilds per step is a genuine inefficiency at real
   decode-loop scale, unlike a one-off unit test), not solved here.
5. **Verified on a small, hand-picked synthetic fixture** (`tools/fixture_gen/tdt_step_common.py`,
   `tests/test_tdt_decoder.cpp`) *before* any real checkpoint was involved, same "prove it small first"
   discipline as everything else in this project. The weights/encoder-output seed was searched (out of
   several hundred candidates) specifically so the decode naturally exercises a **blank-driven 2-frame
   skip** (`duration=2`) followed by a **non-blank emission also with `duration=2`** — genuine TDT
   dynamics, not just the trivial always-advance-by-1 case, and without ever relying on the driver's own
   defensive `max_symbols_per_step` safety net. **Result: bit-exact match, 5/5 checks passed.**

**One real bug found and fixed while building the reference itself**: the first version of the numpy
greedy-decode reference (and, before it was caught, the first version of `TdtDecoder::decode_greedy`)
hung indefinitely on this fixture's own randomly-generated weights — the inner loop can emit non-blank
tokens with `duration=0` indefinitely without ever advancing the frame pointer, and NeMo's own real
algorithm apparently relies on blank-forcing alone to guarantee termination (per this session's own
reading of `rnnt_greedy_decoding.py`), which doesn't cover this case. Fixed by adding an explicit
defensive fallback in both the numpy reference and the C++ driver: if `max_symbols_per_step` is exhausted
without the frame pointer ever advancing, force-advance by 1 frame anyway — not part of the textbook TDT
algorithm, but a real robustness requirement for production code (a pathological or undertrained model
should never be able to hang the decode loop).

Full `ctest` suite at that point: 47/47, zero regressions — proved the decode loop and its new
primitives/driver correct on synthetic data, not yet that they work on a real model. That gap is now
closed too, same day — see "Gap 1 CLOSED" below.

#### Gap 1 CLOSED, 2026-07-18: the real nvidia/parakeet-tdt-0.6b-v3 checkpoint runs end to end

Real checkpoint downloaded (`hf_hub_download`, `nvidia/parakeet-tdt-0.6b-v3`, ~2.34GB `.nemo` file) and
converted/verified for real, closing this gap completely — the first genuinely new model *family*
(Transducer/TDT, not another CTC or decoder-transformer) this engine runs.

**Two real, general bugs found and fixed before any of this could work, neither specific to this model:**

1. **`nemo_common.load_nemo()` assumed every `.nemo` file is gzipped** (`tarfile.open(path, "r:gz")`) — true
   for `stt_en_conformer_ctc_small.nemo`, but `parakeet-tdt-0.6b-v3.nemo` is a *plain, uncompressed* POSIX
   tar archive (confirmed via `file`/`tarfile.is_tarfile`) — NeMo's own save format apparently varies.
   Fixed by switching to `"r:*"` (auto-detect), re-verified the existing Conformer-CTC checkpoint still
   loads correctly afterward (shared code path).
2. **Extracting a real multi-GB `.nemo` archive into the system default temp dir (`/tmp`) hit a genuine
   "No space left on device"** — `/tmp` here is a small partition (28GB, 95% full, ~1.3GB free) entirely
   independent of `/home`'s large one, confirmed via `df -h`. Fixed by extracting next to the input file
   itself (`tempfile.TemporaryDirectory(dir=os.path.dirname(...))`) instead of trusting
   `tempfile.gettempdir()`'s default.

**A genuinely new engine primitive was needed, found by reading the real checkpoint's actual architecture
(not assumed from the earlier Conformer-CTC-small work):** the real FastConformer encoder's subsampling is
`dw_striding` (3 stages: one plain `Conv2d`, then two grouped/depthwise `Conv2d` + pointwise `Conv2d`
pairs) — genuinely different from Conformer-CTC-small's simpler 2-stage plain-`Conv2d` subsampling, and
loom had no depthwise-Conv2d primitive at all (`CONV_1D_DW` existed, `CONV_2D_DW` didn't). Added
**`CONV_2D_DW`** (`src/ops/primitives_conv.cpp`) — ggml already has a native `ggml_conv_2d_dw`, but (like
`ggml_conv_1d`/`ggml_conv_2d`, and unlike ggml's own `ggml_conv_2d` which already respects the kernel's own
dtype) it hardcodes `GGML_TYPE_F16` for its internal `im2col` regardless of the kernel's actual dtype
(confirmed by reading its source directly) — the same precision concern `CONV_1D`/`CONV_2D` were already
written to avoid. `CONV_2D_DW` is a faithful transcription of `ggml_conv_2d_dw`'s exact recipe with F32
substituted for that one hardcoded F16, not an independent re-derivation. Verified with a hand-computed
2-channel example (each channel convolved independently with its *own* kernel, checking both the math and
that channels don't cross-mix) in `tests/test_primitive_registry.cpp` before using it for anything real —
passed on the first try.

**Other real, checkpoint-specific facts confirmed directly against the real `model_config.yaml`/state dict
(not assumed from the config.json HF also publishes, nor from HF's own `modeling_parakeet.py` port, which
turned out to diverge from the real checkpoint on one point — see below):**
- `xscaling: false` — no `sqrt(d_model)` scaling anywhere (unlike Conformer-CTC-small).
- `use_bias: false` genuinely means **no bias anywhere in the encoder**: not just the self-attention/
  positional projections (expected, matches HF's own port), but **also the conv module's 3 convolutions**
  (`pointwise_conv1`/`depthwise_conv`/`pointwise_conv2`) — confirmed by checking the real state dict has no
  such bias tensors at all, contradicting HF `transformers`' `modeling_parakeet.py`, which hardcodes
  `bias=True` for those regardless of config. Trusted the raw checkpoint over the (possibly-diverged, or
  differently-configured) secondary library port.
- Real prediction network: `nn.LSTM(num_layers=2)` (not 1 — confirmed via `weight_ih_l0`/`weight_ih_l1`
  both present), `pred_hidden=640`. This required generalizing the just-built `TdtDecoder`/topology pattern
  from a single LSTM layer to a **stack of N layers** (layer 0 embeds the last token; layer i>0 takes
  layer i-1's `h_new` directly) — re-verified against the synthetic fixture at `N=2` (matching this real
  checkpoint's real depth, not simplified back to 1) before touching any real weights.
- Joint: `joint_hidden=640`, `num_classes=8192` (real tokens) + 1 (blank) + 5 (`durations=[0,1,2,3,4]`) =
  8198-wide final linear — exactly matching the design already proven on the synthetic fixture.
- Both `transformers`' local `ParakeetForTDT` (the class this checkpoint's own `config.json` names) and
  `nemo_toolkit`'s `ASRModel.from_pretrained()` path are unusable in this venv (a
  `huggingface_hub`/`transformers` version pin conflict breaks both imports) — confirmed before choosing to
  hand-roll `reference_forward_parakeet_tdt.py` directly from the raw state dict, not assumed necessary.

**One conversion-tooling bug, caught before it ever reached the C++ engine**: `TdtDecoder` shares *one*
`GgufModel` across every one of its internal `GraphBuilder`s (lstm-per-layer + joint) — the real conversion
script initially split decoder/joint weights across separate, non-overlapping small GGUFs (to avoid
duplicating them), which doesn't satisfy that assumption. Hit a real
`"unresolved input 'joint.enc.weight'"` `SchemaError` immediately, fixed by writing the *full* decoder+
joint weight set into every one of the small GGUFs, exactly matching the synthetic fixture's own
`write_one()` convention (a few tens of MB duplicated 5x, negligible next to the ~2.4GB encoder itself).

**Result**: `tests/test_e2e_parakeet_tdt.cpp` — the real FastConformer encoder's output is **bit-exact**
against the independent hand-rolled PyTorch reference (max abs diff `0.000000`, not just within tolerance)
on the very first full run after the bugs above were fixed, and the new `TdtDecoder`'s greedy decode over
that real encoder output produces the **exact same token sequence and frame indices** as the reference
(`[7618, 1815, 7883]` at frames `[3, 4, 10]`, on a random-noise test waveform — semantically meaningless
input, but structurally exactly what a real decode looks like). **11/11 checks passed.** Full `ctest`
suite: 48/48, zero regressions; the new test skips cleanly by default and passes for real with
`LOOM_PARAKEET_TDT_DIR` set.

**Explicitly out of scope, not attempted here**: detokenization to human-readable text. Verification above
is entirely at the token-id level, same as how the synthetic TDT fixture was checked — sufficient to prove
the decode-loop mechanics and the encoder are correct, but not sufficient for a human-readable transcript.
See the new roadmap item directly below for the real gap this leaves open and what it would take to close.

### DONE, 2026-07-18: SentencePiece **BPE** tokenizer support (`Vocab` gap surfaced by Parakeet-TDT)

Closed the same day it was raised — confirmed the "genuine hybrid of parts already built" hypothesis below
exactly, needing no new algorithm and no new GGUF schema. `Vocab` was extended in place (not a new class):
a `is_bpe_` flag set from `tokenizer.ggml.model` (`"t5"` → existing Viterbi path, `"llama"` → new
`encode_bpe()`), reusing `normalize()`/`normalize_prefix()`/the XCDA charsmap walk/`token_trie_`/`decode()`
completely unchanged. `encode_bpe()` is the standard "repeatedly merge the single highest-scoring adjacent
pair whose concatenation is a real vocab piece, leftmost wins ties" algorithm — a naive O(n²)-per-merge
scan (not SentencePiece's real priority-queue implementation) traded for simplicity, verified to produce
identical output. `tools/convert_nemo/tokenizer_common.py`'s `write_sentencepiece_vocab` now reads the real
`.model` protobuf's own `trainer_spec.model_type` (`UNIGRAM`/`BPE`) to pick `"t5"` vs `"llama"`
automatically instead of hardcoding `"t5"`, and `convert_parakeet_tdt.py` now writes the real tokenizer
into the encoder GGUF.

**Result: `tests/test_vocab_spm_bpe.cpp` — 17/17 checks pass**, exact id-sequence match (encode) and
exact-text match (decode round-trip) against the real `sentencepiece` Python library on the real
8192-piece Parakeet-TDT vocabulary, across plain text, an accented/non-ASCII character (genuinely
exercises the charsmap walk), and a multiple-consecutive-spaces case that only works because this
specific checkpoint's real `remove_extra_whitespaces` is `false` (unlike Conformer-CTC-small's `true`) —
`Vocab` already read this per-model from its own KV, so no bug there, just confirmation the existing code
was already correctly generic. All passed on the first real run, no debugging needed. Also updated
`tests/test_e2e_parakeet_tdt.cpp` to detokenize its real decoded token sequence end to end
(encoder → `TdtDecoder` → `Vocab::decode`) — produces `'Yeah.'`, a real, well-formed English fragment
(meaningless as a transcription, since the input is random test noise, not real speech, but structurally
exactly what a genuine transcription pipeline produces). Full `ctest` suite: 49/49, zero regressions.

**Original gap analysis, confirmed accurate in every particular once implemented:**

Confirmed by inspecting the real checkpoint's `tokenizer.model` directly (`pip install sentencepiece`'s own
real checkpoint's `tokenizer.model` directly (`pip install sentencepiece`'s own
`sentencepiece_model_pb2.ModelProto`, not assumed from the `config.json` field alone):
`m.trainer_spec.model_type == 2`, which the library's own `TrainerSpec.ModelType` enum
(`['UNIGRAM', 'BPE', 'WORD', 'CHAR']`) confirms is real SentencePiece **BPE** — genuinely different from
both tokenizer classes this engine already has: `loom::Vocab` implements only SentencePiece **UGM**
(unigram, GGUF `tokenizer.ggml.model == "t5"`, used by Conformer-CTC's Viterbi-scored segmentation) and
`loom::BpeVocab` implements GPT2-style **byte-level** BPE (GGUF `tokenizer.ggml.model == "gpt2"`, used by
Qwen3 — byte-to-`"Ġ"` remapping + a hand-written pretokenizer regex, no relation to SentencePiece's own
normalizer at all).

**The real gap is narrower than "write a third tokenizer from scratch" — it's a genuine hybrid of parts
already built**, confirmed by inspecting the real `ModelProto` directly:
- **The normalizer is identical to what `Vocab` (UGM) already implements**: `normalizer_spec.name ==
  "nmt_nfkc"`, a real, non-empty `precompiled_charsmap` (237561 bytes) present, `add_dummy_prefix == true`
  — the exact same XCDA-based charsmap walk + space-prefix convention `Vocab::normalize`/
  `normalize_prefix` already implement bit-for-bit, confirmed by inspecting the real bytes rather than
  assuming BPE-mode SentencePiece normalizes differently.
- **The encode algorithm is structurally the same *shape* `BpeVocab` already implements (greedy
  merge-by-rank), just rank-encoded differently.** Every piece's `.score` field is populated with
  monotonically-decreasing negative values by merge priority (confirmed real: the last 10 of 8192 pieces
  have scores `-7908.0` down to `-7917.0`, strictly decreasing) — i.e. merge rank is encoded as a score
  rather than `BpeVocab`'s explicit ordered `merges` array, but the actual algorithm ("repeatedly merge the
  highest-priority adjacent pair until none apply") is the same one already implemented, just keyed off
  `score` (higher/less-negative = higher priority) instead of an explicit rank list — reusing
  `BpeVocab::merge_rank_`'s whole approach, adapted to a different rank source, not a new algorithm.
- **The decode side is already exactly what `Vocab::decode` does** (join pieces, un-escape `▁` U+2581 back
  to a literal space) — no new logic needed there at all.
- **The GGUF schema tag is already a known, real llama.cpp convention, not something to invent**: the
  installed `gguf` Python package's own `SentencePieceVocab` class (`gguf/vocab.py`) — used by llama.cpp's
  own conversion tooling for exactly this case (the original LLaMA/Mistral tokenizers are *themselves* real
  SentencePiece BPE models, confirmed via this same class) — writes `tokenizer_model = "llama"` (not
  `"t5"` or `"gpt2"`) as the `tokenizer.ggml.model` KV value. `loom::Vocab::load` currently only recognizes
  `"t5"` and throws for anything else, so a `"llama"`-tagged GGUF would currently fail to load at all.

**What it actually took, confirming the plan above exactly**: extended `loom::Vocab` in place (the first
option considered, and it did fit better) — see the "DONE" summary at the top of this section for the real
result. No hand-traced synthetic fixture ended up being necessary before trusting it against the real
8192-piece vocabulary: `test_vocab_spm_bpe.cpp` compares directly against the real `sentencepiece` library
loaded against the same real `.model` file, the same "no meaningful toy substitute" reasoning
`test_vocab.cpp`'s own UGM test already used — correctness here is entirely about faithfully reproducing
one specific real model's behavior, which a synthetic vocab can't stand in for anyway.

### Gap 2: the vocoder blockers (VITS/StyleTTS2's HiFi-GAN decoder, Kokoro's ISTFTNet)

- **HiFi-GAN (VITS/StyleTTS2)**: per the "already covered" section above, this is graph-expressible with
  existing primitives (dilated `CONV_1D` MRF blocks, `CONV_TRANSPOSE_1D` upsampling, `LeakyReLU`) plus
  the two small additions noted (crop `VIEW`, `LeakyReLU`/`tanh` registration) — no fundamentally new
  capability needed, "just" a real conversion + reference-verification effort at real HiFi-GAN scale.
- **ISTFTNet (Kokoro)**: genuinely the hardest gap — `ggml` has no complex-number type or FFT. Two paths,
  in order of preference (verify the first before reaching for the second, same "prefer composing
  existing primitives" precedent as everywhere else in this file):
  1. **Invert the same STFT-via-convolution trick already verified in Milestone 5.** Forward STFT was
     expressed as cross-correlating framed audio against precomputed cos/sin DFT-basis kernels (verified
     bit-exact against real `torch.stft`, see `mel_common.py`/`BACKLOG.md`'s Milestone 5 section). The
     inverse DFT is likewise a fixed linear transform of each frame's spectral values (expressible as one
     `MUL_MAT` against a precomputed inverse-DFT basis matrix), and overlap-adding the resulting
     per-frame time-domain segments back into a single waveform is mathematically exactly what a
     transposed convolution computes (`CONV_TRANSPOSE_1D`, already implemented) — a windowed
     overlap-add is a transposed conv with the window folded into the kernel. **This is a promising,
     currently unverified hypothesis, not a confirmed plan** — it needs the same hand-computed
     small-example verification (and then a real-`numpy`-iSTFT cross-check) that every other primitive
     in this project got before being trusted, particularly around whether `CONV_TRANSPOSE_1D`'s exact
     accumulation semantics match true overlap-add without an off-by-one at frame boundaries. If it
     checks out, ISTFTNet needs zero new engine capability beyond what dynamic-length Conformer-CTC
     already has (`MUL_MAT` + `CONV_TRANSPOSE_1D` + the `floor()`-based dynamic-shape symbol-expression
     pattern from Milestone 7, since iSTFT's output length is itself a function of the frame count).
  2. **Fall back to a real FFT implementation** (a vendored lightweight C++ iFFT, or a `fftw3`
     dependency) only if (1) doesn't hold up under verification — this is explicitly the fallback, not
     the first move, to avoid taking on a new C/C++ dependency (this project currently has none beyond
     `ggml`/`nlohmann_json`, both fetched via `FetchContent`) before confirming it's actually necessary.
  Either way, the `ggml` graph's declared `"output"` stops at the magnitude/phase (or real/imag)
  spectrogram — the iSTFT reconstruction step runs host-side after `ggml_backend_graph_compute`, same
  "host logic, not a graph primitive" precedent as `ctc_greedy_decode`/`Vocab::decode`.

### Why Kokoro specifically is worth prioritizing if this roadmap is picked up

82M parameters (~350MB unquantized, meaningfully smaller quantized), fully feed-forward (no diffusion
steps, no autoregressive Transducer loop — sidesteps Gap 1 entirely), so it's reachable with *only* Gap
2 solved. If the iSTFT-via-existing-primitives hypothesis above holds, Kokoro would need no new ggml
primitives at all beyond the two trivial activation registrations — making it the cheapest, highest-payoff
next real-model target in this family, notably cheaper than VITS/StyleTTS2 (also need Gap 2, and are
larger/more architecturally involved). **Correction, 2026-07-18**: Parakeet does *not* need "both gaps" as
originally written here — it's ASR (audio-to-text), with no waveform synthesis stage at all, so Gap 2
(vocoder blockers) simply doesn't apply to it; Parakeet is gated on Gap 1 only (the Transducer/LSTM decode
loop).

**Candidate reference checkpoint**: <https://github.com/femelo/kokoro-deutsch> — a German fine-tune of
Kokoro-82M (PyTorch `.pth` checkpoints, trained via "a patched StyleTTS2 submodule", `misaki` phonemizer)
confirmed via its README, not assumed. Same base architecture/vocoder as upstream Kokoro-82M, so it's a
valid conversion target and a real checkpoint to verify the iSTFT hypothesis against — just note the
phonemizer (`misaki`, German-language-specific) is its own separate preprocessing stage upstream of the
graph, same "host-side, out of scope for the graph itself" boundary as mel-spectrogram extraction was
before Milestone 5 brought it in-graph (text→phoneme is a linguistic/lexicon problem, not a tensor op —
unlike mel extraction, unlikely to ever move into the `ggml` graph itself).

## Roadmap: Qwen3 family (Qwen3-0.6B, Qwen3-ASR-0.6B, Qwen3-TTS-0.6B)

Captured per an explicit request. **Qwen3-0.6B (base LLM) is now DONE — see "Resolved in Milestone 8"
above** — the other two remain not started. The user's own forks are the intended reference
implementations for the remaining ASR/TTS work — **not** the upstream repos they're forked from — since
they contain fixes/optimizations the user prefers:

- Qwen3-0.6B (base LLM): DONE (Milestone 8). No single reference repo was used — `llama.cpp` itself
  supports Qwen3 and was a legitimate reference for the plain architecture, but the real facts (attention
  shapes, QK-norm weight shapes, tokenizer regex/schema) were all confirmed directly against the real
  `Qwen/Qwen3-0.6B-Base` checkpoint and its `tokenizer.json`, not assumed from `llama.cpp`'s source.
- Qwen3-ASR-0.6B: <https://github.com/femelo/qwen3-asr.cpp> (the user's fork of `predict-woo/qwen3-asr.cpp`).
- Qwen3-TTS-0.6B (specifically "Qwen3-TTS-12Hz-0.6B-Base" per the fork's README): <https://github.com/femelo/qwen3-tts.cpp>
  (the user's fork of the same `predict-woo` lineage).

Per the user's framing, the strategic point of this item is broader than any single model: **because the
graph topology is data embedded in the GGUF rather than hardcoded C++, loom-engine could run models
`llama.cpp` doesn't officially support without needing a bespoke standalone engine per model** (the
situation `qwen3-asr.cpp`/`qwen3-tts.cpp` are themselves examples of — each is its own dedicated C++
program). The facts below were confirmed by fetching each fork's actual README (not assumed from general
knowledge of the Qwen3 family, since ASR/TTS variants are recent and their exact architectures aren't
something to guess at) — deeper source-level verification is still needed before conversion work starts,
same rigor as every other model in this backlog.

### Qwen3-0.6B (base LLM) — DONE, see "Resolved in Milestone 8" above

Was flagged here as likely the cheapest item in this entire backlog, since Milestone 1's toy LLM already
built the whole core stack it needed (RMSNorm, RoPE, KV-cached self-attention, SwiGLU FFN, autoregressive
greedy decoding). That held up — QK-norm, GQA, and tied embeddings all composed from existing primitives
with no engine changes, verified via new regression tests before touching the real checkpoint. The one
genuine gap (a real BPE tokenizer, since Qwen3 uses byte-level BPE, not SentencePiece unigram) is now
closed too, with full Unicode fidelity per explicit user decision. Full writeup, including what's
deliberately still narrower in scope (ASCII-only `\s`, Base not instruct/reasoning), is in "Resolved in
Milestone 8" above.

### Qwen3-ASR-0.6B

Per `qwen3-asr.cpp`'s README: an audio encoder feeds into the LLM via "audio-text embedding injection"
(module named `audio_injection.cpp/h` in that repo), then the LLM decodes **fully autoregressively** —
no CTC, no Transducer. That last point matters: this is architecturally simpler to integrate than
Parakeet's RNN-T (Gap 1 above) specifically *because* it reuses the existing `Generator`/`KvCache`
autoregressive pattern directly for the decode side — the genuinely new piece is the audio-encoder →
embedding-injection integration, not a new decode loop:

- **Audio encoder architecture is not yet confirmed** — the README doesn't state whether it's
  Whisper-style, Conformer-style, or something else specific to Qwen-Audio's lineage; needs real
  source-level (or weights-shape) inspection before assuming it maps onto the existing Conformer-CTC
  encoder work, same "confirm from the actual source, don't assume" discipline as every other model here.
- **Embedding injection is a new integration pattern**, distinct from both Gap 1 (Transducer) and the
  mel-frontend's "declared input" pattern: audio-encoder output embeddings need to be spliced into
  specific positions of the LLM's token-embedding sequence (replacing placeholder token embeddings)
  before the transformer stack runs — closer to a LLaVA-style vision-token-injection pattern than
  anything built so far. Likely needs a small amount of new host-side splicing logic (not a new ggml
  primitive — token embedding lookup + audio embedding are both just tensors, and overwriting a slice of
  one with the other is an existing `VIEW`/`ggml_set`-style operation), but the exact mechanism needs
  designing against the real architecture, not guessed at here.

### Qwen3-TTS-0.6B (12Hz-Base) — the most architecturally novel item in this whole roadmap

Per `qwen3-tts.cpp`'s README, this is a **neural-codec TTS**, not mel+vocoder like VITS/Kokoro: BPE text
tokenizer → ECAPA-TDNN speaker/x-vector encoder (from optional reference audio) → a **two-level**
autoregressive generator (a 28-layer "talker" producing codebook-0 per frame, plus a 5-layer "code
predictor" that then sequentially generates codebooks 1-15 *within* that same frame) → WavTokenizer
vocoder decoding the multi-codebook codes to a 24kHz waveform.

- **ECAPA-TDNN speaker encoder**: likely largely reachable already — TDNN layers are dilated `CONV_1D`
  (already supported), and x-vector extraction's statistics pooling (mean+std over time) is the same
  shape of reduction the mel-frontend's CMVN normalize already does (`SUM_ROWS`-based). SE-Res2Net's
  specific squeeze-excite gating would need confirming against the real architecture, not assumed solved
  by analogy alone.
- **Nested/hierarchical autoregressive generation is the real new gap here** — more involved than
  Parakeet's single RNN-T loop (Gap 1) or Qwen3-ASR's single LLM decode loop: an *outer* per-frame loop
  (the talker, itself a standard KV-cached autoregressive transformer — reuses the `Generator` pattern)
  with an *inner* per-codebook loop (the code predictor generating 15 more tokens *per outer step*,
  presumably its own smaller causal/KV-cached structure). This would need a new C++ driver nesting two
  reused-graph loops, doubling down on the graph-reuse discipline from Milestone 3's `ggml_gallocr`
  finding at two different loop granularities simultaneously — the most complex new control-flow driver
  proposed anywhere in this backlog, worth prototyping carefully rather than assuming it falls out of the
  existing `Generator`/`OdeStepper` patterns unchanged.
- **WavTokenizer's vocoder decoder is reportedly ConvNeXt-based** (consistent with WavTokenizer's
  published architecture, not yet confirmed against this specific fork's weights) — ties directly back
  to the "ConvNeXt-style blocks... fully expressible today" finding earlier in this file, another point
  in favor of the vocoder side of this model being cheaper than the nested-decode-loop side.

## Roadmap: investigate `executorch-ggml` for reusable ops-mapping concepts

Not started — captured per an explicit request to verify and possibly adopt reusable concepts from
<https://github.com/larryliu0820/executorch-ggml> (local clone: `/home/flavio/Dev/executorch-ggml`), an
ExecuTorch backend that lowers `torch.export`-produced ATen graphs to ggml compute graphs. Facts below
were confirmed by reading the actual repo (`README.md`, `PROGRESS.md`/`GRAPH_REBUILD.md`,
`schema/ggml_ir.fbs`, `docs/gguf-integration.md`, and representative files under
`python/executorch_ggml/ops/`, `runtime/ops/`), not assumed — it's a real, actively-benchmarked project
(claims beating `llama.cpp` itself on Qwen3-0.6B decode: 411 vs 377 tok/s on an A100, 331 vs 299 tok/s on
an Apple M4 Max, both Q8_0; and 122% of `llama.cpp`'s decode throughput on Qwen3.5-35B-A3B MoE).

**Where it's conceptually adjacent to loom-engine, not identical**: both projects interpret a
data-described graph at runtime instead of hardcoding architectures in C++, but the *source* of that graph
differs — `executorch-ggml` derives its IR automatically from `torch.export`'s ATen dialect (any model
that exports cleanly gets ggml coverage for free, gated by an op allow-list, with unsupported ops falling
back to ExecuTorch's own CPU executor), while loom-engine's JSON topology is hand-authored per architecture
by a conversion script. `executorch-ggml`'s IR is a FlatBuffer-serialized `OpCode` enum
(`schema/ggml_ir.fbs`, ~50 ops) embedded in a `.pte` file, playing a role similar to loom's JSON `"op"`
strings + `PrimitiveRegistry` — but the two extension models differ in a way worth noting as a point of
comparison rather than a gap: adding a new op to `executorch-ggml` needs edits in 5 places (FlatBuffer
enum, Python partitioner allow-list, Python ATen→IR mapping, C++ runtime builder call, regenerated
FlatBuffer headers, per its own README's "Extending to More Ops" section), vs. loom's single
self-registering `LOOM_REGISTER_OP(NAME, fn)` macro with no central switch statement — loom's approach is
simpler here specifically because it never needs to match an *externally-defined* op vocabulary (ATen's),
so this isn't something to change, just a confirmed point in favor of the existing design.

### The real prize: a generic `torch.export` → loom-topology converter, not just borrowed op-mapping ideas

Per explicit user direction: the actual value of `executorch-ggml` isn't its specific op-mapping choices
(above) — it's the *shape* of its conversion pipeline. Today, every model loom-engine has converted
(`convert_conformer_ctc.py`, `convert_qwen3.py`) is a **bespoke, hand-authored Python script per
architecture**: the topology JSON is written by hand, node by node, and every weight name is mapped by
hand. This doesn't scale to the rest of the roadmap in this file (VITS, StyleTTS2, Kokoro, Qwen3-ASR,
Qwen3-TTS, Parakeet, ...) — it's exactly the kind of duplicated bespoke-per-architecture work loom's whole
design is meant to avoid on the *runtime* side (a generic `GraphBuilder` + `PrimitiveRegistry` interpreting
data instead of hardcoded `llm_build_*`-style C++), but that avoidance currently stops at the conversion
boundary.

**The proposed shape, mirroring `executorch-ggml`'s pipeline but targeting loom's JSON topology instead of
its FlatBuffer IR**: `torch.export(model, args)` → a stable ATen FX graph → walk the graph nodes, mapping
each ATen op to one (or a short, fixed composition of a few) loom `PrimitiveRegistry` op(s), automatically
emitting both the topology JSON *and* a weight-name mapping (`state_dict` key → loom tensor name) — instead
of a human doing that translation by hand for every new model. Extending coverage to a new architecture
then means adding a handler for whichever *ATen ops* it uses that aren't mapped yet (write once, reused by
every future model that also uses that op), not writing an entirely new conversion script.

**The user's own caveat is the important part, and it's already anticipated elsewhere in this file**: not
every model exports as one clean functional graph. **VITS is the concrete example** — its stochastic
duration predictor involves sampling/control-flow, and its flow-based decoder and HiFi-GAN vocoder are
architecturally distinct stages, not one straight-through function `torch.export` can trace as a single
graph. This is *exactly* the same shape of problem already identified and scoped for two other models in
this backlog: **Gap 1** (the Transducer problem — RNN-T/TDT needs isolated Encoder/Prediction-Network/
Joint-Network sub-graphs with a custom C++ decode loop between them) and **Gap 2** (Qwen3-TTS's nested
two-level autoregressive generation — a talker sub-graph and a code-predictor sub-graph, driven by nested
C++ loops), both of which already establish the pattern `SPECIFICATION.md` §4 ("The TTS Catch") prescribes:
JSON describes each *static* sub-graph, C++ drives whatever loop/control-flow/sampling connects them. A
generic converter needs the *same* answer at the tooling level: call `torch.export()` **separately per
sub-module** (e.g. VITS's `TextEncoder`, `StochasticDurationPredictor`, `Flow`, `Generator`), producing
multiple topology JSON blobs in one GGUF, stitched together by a C++ driver — not a single monolithic
export attempt that breaks on the first piece of real control flow.

**Real tradeoffs to weigh before committing engineering time to this** (not yet prototyped):

- **The bespoke-per-model work doesn't disappear, it moves.** Any checkpoint that isn't already a clean,
  export-ready `torch.nn.Module` (most real checkpoints loaded from raw HF `state_dict`s, as both
  conversions so far have been) still needs *someone* to write a plain-PyTorch `nn.Module` reimplementing
  the architecture before it can be exported at all — arguably comparable effort to writing today's
  `reference_forward_*.py` scripts, just in `nn.Module` form instead of functional numpy. **The real payoff
  compounds over the number of models attempted**, not the first one: the op-mapping layer is write-once,
  reused-forever, unlike today's 100%-bespoke-every-time topology scripts.
- **Verification actually gets stronger, not weaker.** An auto-generated topology can be checked against
  the *original* PyTorch model's real forward pass directly (no possibility of a human mis-transcribing the
  op sequence by hand) — only the shared op-mapping layer itself needs to be correct, and it's verified once
  and reused, same "verify before trusting an existing mechanism in a new configuration" discipline as
  everything else in this backlog, just applied at the mapping-layer level instead of per-model.
- **This is a genuinely new, substantial subsystem** — an ATen-graph walker, an op-mapping table, and
  weight-name-mapping automation — not a quick add on top of existing conversion scripts.
- **Recommended first step if this gets picked up**: a minimal proof-of-concept against a model
  *already* hand-converted and verified — e.g. re-derive the Milestone-1 toy LLM's or Milestone 8's
  Qwen3-0.6B-Base's topology JSON via the generic converter and check it against the *same* existing
  reference/test harness those already have. That validates the whole approach cheaply, against a known-
  correct answer, before ever pointing it at a genuinely new, unconverted architecture like VITS.

#### POC done, 2026-07-17: the toy LLM, converter passes against the existing reference harness

Built the recommended POC above (`tools/convert_generic/`), targeting the Milestone-1 toy LLM (simplest
architecture available, already has an independent numpy reference). Result: **12/12 checks pass** in a new
`tests/test_e2e_toy_llm_generic.cpp`, comparing the auto-derived topology's generated tokens and per-step
logits against the exact same `reference_forward.py` fixture `test_e2e_toy_llm.cpp` already checks against
(both topologies share the same weights/seed, so this is a direct apples-to-apples check of the *converter*,
not a new numerical claim). Full `ctest` suite stays green (38/38, same skip count as before — this new test
follows the same `SKIP_RETURN_CODE 77` pattern as the real-model tests, since it needs a `torch` environment
to produce its GGUF; see `tests/test_e2e_toy_llm_generic.cpp`'s header comment for how to run it for real).

**What was built:**
- `tools/convert_generic/toy_llm_module.py` — a plain `torch.nn.Module` (`ToyLLM`) reproducing
  `reference_forward.py`'s forward pass op-for-op (verified eagerly against it first, `max abs diff ≈
  5.6e-9`, before ever exporting it), loading weights directly from `toy_llm_common.generate_weights()`.
- `tools/convert_generic/aten_to_loom.py` — the converter: walks `torch.export()`'s `ep.graph.nodes`,
  maps each node 1:1 via a small fixed table, resolves weight names via `graph_signature.inputs_to_parameters`
  plus one small qualname→GGUF-key rule, and emits the topology JSON.
- `tools/convert_generic/make_toy_llm_gguf_generic.py` — ties it together into a GGUF using the *same*
  `toy_llm_common` weights/hparams as the hand-written fixture.

**The concrete op-mapping table that came out of this** (`OP_MAP` in `aten_to_loom.py`):
`aten.embedding.default → GET_ROWS`, `aten.rms_norm.default → RMS_NORM` (asserts `weight is None` — this
POC always keeps the affine as a separate node), `aten.mul.Tensor → MUL`, `aten.add.Tensor → ADD`,
`aten.linear.default → MUL_MAT` (args reordered weight-first, per `ggml_mul_mat`'s convention),
`aten.view.default`/`aten.reshape.default → RESHAPE`, `aten.silu.default → SILU`, plus two **custom ops**
(`torch.library.custom_op`, so they survive export as single opaque nodes) for the two things with no ATen
equivalent: `loom::rope_neox → ROPE` and `loom::attention → ATTENTION`.

**Two real bugs found and fixed while getting this to actually run** (both concrete, worth remembering for
whoever picks this up next):
1. **`RESHAPE`'s `shape` attr needed reversing.** ATen's `view()`/`reshape()` args are plain numpy/PyTorch
   order (slowest-varying dim first); loom's `RESHAPE` feeds its `shape` attr straight into
   `ggml_reshape_*`, which is `ne`-order (fastest-varying first) — confirmed by reading
   `src/ops/primitives_basic.cpp:147-174` and cross-checking against `toy_llm_common.py`'s own
   hand-written topology, which already reverses this by construction. Missing this reversal didn't error
   at conversion time — it silently produced a topology that failed a `ggml_rope_ext` shape assertion
   (`a->ne[2] == b->ne[0]`) three build layers downstream, at `GraphBuilder::reserve()` time. **A real,
   generalizable lesson**: any op-mapping table translating between a numpy/PyTorch-convention IR (ATen,
   ONNX) and ggml's reversed-`ne` convention needs this reversal applied consistently at every
   shape-bearing attr, not just tensor axis order — easy to get right once, easy to silently get wrong
   per-op if each mapping is written independently.
2. **`ATTENTION`'s KV-cache side effects have no ATen equivalent, regardless of whether the source model
   uses `scaled_dot_product_attention`.** Confirmed empirically that `torch.export()`'s default output
   *does* keep `F.scaled_dot_product_attention` as a single `aten.scaled_dot_product_attention.default`
   node (not decomposed) — but even so, mapping it to `ATTENTION` would still need the converter to inject
   `kv_cache: true` / `layer: i`, information that exists nowhere in the ATen graph. This POC sidestepped
   the *separate* wrinkle of SDPA's `(batch, heads, seq, dim)` calling convention not matching loom's
   native head-minor layout (which would need its own transpose-wrapper unwrapping logic) by using a
   `loom::attention` custom op instead — a deliberate scoping choice, not a claim that SDPA-shaped graphs
   can't eventually be mapped directly.

**How the layer index gets recovered**: `node.meta["nn_module_stack"]` (confirmed to survive per-node even
on the fully-flattened Core ATen graph — see the empirical checks earlier in this section) is regex-matched
for `layers\.(\d+)` to fill in `ATTENTION`'s `layer` attr — this generalizes to any real model built from a
`nn.ModuleList` of per-layer submodules, not just this toy one.

**Still exactly as scoped, not attempted**: dynamic shapes (this POC's `n_tokens` substitution is a
literal-value special case, not real `torch.export.Dim` dynamic-shape export), op/subgraph fusion pattern
-matching, anything multi-graph (VITS). The weight-name mapping is still one small explicit per-model-family
rule (`_qualname_to_gguf_name`), not auto-derived — expected and already called out above, not a regression.

#### Round 2, same day: pointed the *same, unmodified* converter at real Qwen3-0.6B-Base — the op-mapping table held

Per the suggested next step above: reused `tools/convert_generic/aten_to_loom.py` completely unchanged
against `tools/convert_generic/qwen3_module.py`'s `Qwen3LLM` — a from-scratch `nn.Module` loading the real
checkpoint's BF16 weights directly (no `transformers`, same precedent as `convert_qwen3.py`), with genuine
GQA (16 query / 8 KV heads), per-head QK-norm, and tied embeddings. Verified eagerly against the real
reference first (`max abs diff ≈ 2.7e-5` against `expected_logits_step0.bin`, argmax match) before ever
exporting it — same discipline as the toy LLM.

**Result: `tests/test_e2e_qwen3_generic.cpp` passes 14/14 checks** against the exact same real-model
reference fixture `test_e2e_qwen3.cpp` already uses (same weights ⇒ byte-identical expected logits,
regardless of which conversion pipeline built the GGUF) — real 28-layer generation, correct GQA broadcast,
correct QK-norm, correct tied-embedding weight reuse, correct token sequence. Full `ctest` suite: 39/39,
same skip pattern.

**The actual signal this was testing: zero new op-mapping entries needed.** `torch.export()`'s distinct ATen
targets on the real Qwen3 graph are *exactly* the same 9 the toy LLM already had mapped —
`aten.embedding.default`, `aten.rms_norm.default`, `aten.mul.Tensor`, `aten.add.Tensor`,
`aten.linear.default`, `aten.view.default`, `aten.silu.default`, plus the two custom ops
(`loom.rope_neox.default`, `loom.attention.default`) — imported from `toy_llm_module.py`, not
redefined. Concretely, for free from the unmodified table:
- **QK-norm** decomposes to the exact same `rms_norm(weight=None)` + `mul` pair already used for
  `attn_norm`/`ffn_norm`, just applied to `q`/`k` instead of `cur` — the op-mapping table has no idea it's
  looking at "QK-norm" specifically, it's just RMS_NORM+MUL again.
- **GQA** needed no special-casing in the converter at all — `q`/`k`/`v` just arrive at `loom::attention`
  with different head counts (16 vs 8), and loom's real C++ `ATTENTION` primitive's existing
  `ggml_mul_mat` broadcast handles it, exactly as `tests/test_e2e_gqa.cpp` already proved for the hand
  -written topology. (The custom op's own *eager* reference body did need one fix completely unrelated to
  the converter — `F.scaled_dot_product_attention` needs `enable_gqa=True` when q/k head counts differ —
  but that's PyTorch-eager-execution plumbing, not a converter or op-mapping change.)
- **Tied embeddings** needed no special-casing either — `Qwen3LLM.token_embd` is one `nn.Parameter`
  referenced from two call sites (`F.embedding` and the final `F.linear`), so `torch.export` naturally
  emits one placeholder referenced by both consumer nodes, and the converter's existing
  `node_symbol`/`inputs_to_parameters` resolution just resolves both to the same `"token_embd.weight"`
  loom symbol without any tied-embedding-specific logic.
- **The `_qualname_to_gguf_name` weight-naming rule carried over unchanged too** — `Qwen3Layer`'s attribute
  names (`attn_q_norm`, `attn_k_norm`, etc.) were deliberately chosen to match the same
  `layers.N.xxx`/`.weight`-suffix convention the toy LLM used, so the one rule written for the toy model
  produced correct GGUF key names (`blk.0.attn_q_norm.weight`, ...) for the real one too.

This is real evidence for the backlog's central claim — the op-mapping layer, once written, is genuinely
write-once/reused, at least across two models sharing an attention-transformer shape. The next real test of
that claim is a model that *isn't* shaped like a decoder transformer at all (VITS, per the roadmap above) —
that's where new op-mapping entries and the actual multi-graph-linking question should first bite for real.

**Gating criterion, added 2026-07-18, before promoting this converter to the default path**: per explicit
user direction, the bespoke per-model scripts (`convert_qwen3.py`, `convert_nemo/...`) should *not* be
retired or de-emphasized yet, even though this result is a genuine, real win. The evidence so far only
covers two models sharing the exact same decoder-transformer shape — it doesn't yet say anything about
whether the op-mapping table (or the converter's design generally) holds up on something structurally
different. Two real costs also haven't gone away, and shouldn't be glossed over when comparing the two
approaches: **the per-model authoring effort hasn't been eliminated, it's moved** (from hand-writing
topology JSON to hand-writing an export-friendly `nn.Module` + registering `torch.library.custom_op`s for
anything with no ATen equivalent), and **the converter's own output is currently worse on one axis** — it
emits a fully flat, unrolled topology (704 nodes for Qwen3's 28 layers) with no `repeat_for` compaction,
where the hand-written scripts produce a compact, more readable one. Before treating the generic converter
as the default and retiring any bespoke script: (1) prove it on a model that *isn't* decoder-transformer
shaped — an encoder-only, non-autoregressive, conv-heavy architecture is the real test, not another LLM;
(2) either add `repeat_for`-compaction to the converter's own JSON emission, or explicitly accept the
flat-topology size/readability cost as a permanent tradeoff, not an oversight to fix later.

#### Gating criterion PASSED, 2026-07-18: generic converter proven against the real Conformer-CTC encoder

Point 1 of the gating criterion above is now satisfied: the *same* converter architecture (op-mapping
table + custom-op pattern) was pointed at the real `stt_en_conformer_ctc_small` checkpoint (Milestone 4/7)
— a genuinely non-decoder-transformer shape (encoder-only, non-autoregressive, subsampling `Conv2d`,
`LayerNorm` instead of `RMS_NORM`, depthwise `Conv1d`, `GLU`, relative-position self-attention with no
KV-cache). Scope deliberately narrower than the full model: the export-friendly `nn.Module`
(`tools/convert_generic/conformer_ctc_module.py`) takes `mel_input` as its declared input (skipping the
mel frontend, which exercises no new ops relative to the toy-LLM/Qwen3 POCs already done), loading the
same real checkpoint weights via the existing `tools/convert_nemo/nemo_common.load_nemo()`.

**Result: `tests/test_e2e_conformer_ctc_generic.cpp` passes 7/7 checks** against the real checkpoint,
reusing `test_e2e_conformer_ctc.cpp`'s own reference fixture (same weights/waveform seed ⇒ byte-identical
expected logits). Full `ctest` suite: 41/41, zero regressions.

**Unlike Round 2 (Qwen3), this genuinely DID need new op-mapping entries** — direct evidence for exactly
the question the gating criterion was designed to answer:
- `aten.layer_norm.default → LAYER_NORM` (+ separate `MUL`/`ADD` for the affine, same "weight=None asserted,
  affine kept external" pattern `RMS_NORM` already used).
- `aten.conv2d.default → CONV_2D`, `aten.conv1d.default → CONV_1D`/`CONV_1D_DW` (branches on the node's
  `groups` arg) — both needed real handling of ATen's variable-arity args (`stride`/`padding`/`dilation`/
  `groups` are silently omitted from `node.args` when they match the schema's own defaults — confirmed
  empirically via a standalone trace before trusting it, not assumed from the toy LLM's fixed-arity
  `aten.rms_norm.default` handling, which has a different default and never hits this).
- `aten.glu.default → GLU` (a real ATen op, convenient), `aten.relu.default → RELU`,
  `aten.squeeze.dim`/`aten.unsqueeze.default → RESHAPE` (no explicit target-shape arg like view/reshape —
  read the shape `torch.export` itself already computed via `node.meta["val"]`).
- **`aten.permute.default → PERMUTE`, the trickiest one**: `ggml_permute` and ATen's `permute` use
  genuinely different, inverse conventions — ggml's is a *destination*-encoding in reversed (ne-order) axis
  numbering (confirmed by reading `ggml_permute`'s C source directly), ATen's is a *source*-encoding in
  normal torch axis order. Derived the translation formula
  (`axes[k] = ndim-1 - dims.index(ndim-1-k)`, padded with identity beyond the input's real rank) and
  verified it numerically against several independent permutations (including non-involution 3- and
  4-cycles, both padded-from-3D and genuine full-4D cases) via a standalone numpy repro *before* trusting
  it in the converter — same rigor as the RESHAPE shape-order bug caught earlier in this file.
- A new `loom::rel_pos_attention` custom op (`tools/convert_generic/conformer_ops.py`) for
  `REL_POS_ATTENTION` — no ATen equivalent at all, same treatment `loom::attention`/`loom::rope_neox`
  already got.
- `_qualname_to_gguf_name` needed a small generalization: Conformer-CTC is the first model in this POC with
  biased `Linear`s, and the old rule (append `.weight` to anything not already ending in `.weight`) was
  silently mangling `*.bias` qualnames into `*.bias.weight`.

**Two real, general bugs found and fixed while building this — not model-specific workarounds**:
1. **A sentinel-collision bug in `_shape_attr`'s `n_tokens` substitution.** Converting a model with no real
   `n_tokens` dimension used `example_n_tokens=-1` as an "inert, never matches" sentinel — but `-1` is
   *also* PyTorch's own reshape/view "infer this dimension" sentinel value, so the substitution fired on
   every legitimate ATen `-1` entry, silently replacing it with the string `"n_tokens"` and defeating
   `RESHAPE`'s own `-1`-inference logic in `op_reshape`. Manifested as a 0-sized reshape target deep inside
   `ggml_reshape_2d`'s own assertion, nowhere near the actual bug — only found by adding a temporary debug
   trace to `GraphBuilder::build_node` (reverted after) and a Python shape-propagation simulator that
   cross-checked the topology JSON's own bookkeeping node-by-node. Fixed by excluding `-1` from the
   substitution unconditionally, regardless of what `example_n_tokens` is set to.
2. **`aten.reshape.default`/`aten.squeeze.dim`/`aten.unsqueeze.default` all need an unconditional `CONT`
   first, `aten.view.default` doesn't.** ATen's `.view()` is guaranteed to never copy (raises instead on
   non-contiguous input), matching `ggml_reshape_*`'s own hard contiguity requirement directly — but
   `.reshape()`/`.squeeze()`/`.unsqueeze()` are all copy-capable in ATen (they silently insert a copy for
   non-contiguous input rather than raising, with *no* separate `aten.contiguous.default` node appearing to
   signal it, unlike an explicit `.permute().contiguous()` call). Hit this for real twice — a
   `permute().reshape()` chain and a `permute().unsqueeze()` chain — both crashing inside
   `ggml_reshape_*`'s `GGML_ASSERT(ggml_is_contiguous(a))`. Fixed by always emitting an explicit `CONT`
   before `RESHAPE` for these three ATen targets specifically, matching the same unconditional-`CONT`-
   after-`PERMUTE` precedent the hand-written `convert_conformer_ctc.py` topology already uses throughout.

**Eager sanity check, run before ever exporting**: the new `ConformerCTC` nn.Module matched an independent
reference (`reference_forward_conformer.py`) to `max abs diff ≈ 2.2e-5` on the first try — real confirmation
that the subsampling/flatten/rel-pos-attention conventions (ground-truthed against NeMo's actual source,
`nemo/collections/asr/parts/submodules/subsampling.py`, not just comments-about-NeMo in the existing
hand-written converter) were derived correctly before any of the bugs above were even reachable.

**Point 2 of the gating criterion (`repeat_for` compaction) is still open** — this POC's output remains a
fully flat, unrolled topology (1137 nodes for a 16-layer encoder), same accepted tradeoff as Qwen3's Round 2.

#### Open discussion, to continue later: how to "link" multiple sub-graph exports together

Raised by the user, explicitly deferred for a later session — captured here as an open question with two
candidate directions, not a decision:

Once a model is split into N separate `torch.export()` calls (one per sub-module, per the VITS/Gap-1/Gap-2
pattern above), two things still need figuring out systematically: **(a) where to cut** — how to discover
the sub-module boundaries automatically rather than a human deciding them by hand every time — and **(b)
how to wire the pieces back together** — matching each sub-graph's output tensors to the next one's named
inputs, and identifying exactly where non-tensor control flow (sampling, iteration, discrete branching)
has to live in the C++ driver between them, analogous to how loom's own topology JSON already has a
`"inputs"`/`"output"` naming convention per graph.

**Confirmed 2026-07-17 by reading Netron's real source** (`lutzroeder/netron`, `source/pytorch.js` — fetched
directly, not assumed from memory, since the user asked specifically how Netron manages to visualize models
`torch.export` can't handle cleanly): Netron actually has **three tiers**, tried in order, and only the
first two produce a real op-level graph —

1. **`torch.export.ExportedProgram`** (when export succeeds): Netron walks `exported_program.graph.nodes`
   directly, resolving weights via `graph_signature.inputs_to_parameters`/`inputs_to_buffers` — the exact
   same FX/ATen graph `executorch-ggml` already targets. Nothing new relative to the generic-converter
   discussion above.
2. **TorchScript** (`torch.jit._script.RecursiveScriptModule`, from `.script()`/`.trace()` + `.save()`):
   confirmed as a genuinely different, more permissive IR, not just an older export path. TorchScript's
   graph has `prim::GetAttr` (submodule/attribute access) and `prim::CallMethod` (literally "call this
   method on this submodule") as first-class node kinds — meaning sub-module boundaries can exist as real
   graph nodes *without* being inlined, unlike ATen/FX which is always flat. Netron does apply an inlining
   pass (`torch._C._jit_pass_inline`) for some of its own format handlers, but that's a deliberate,
   optional pass over the IR, not an inherent limitation — the structure is there if you don't run it. This
   is the concrete mechanism behind the "TorchScript could do that" the user recalled, now verified rather
   than assumed — `prim::CallMethod` boundaries are exactly the kind of natural cut point the "where to
   cut" half of this problem needs, and TorchScript separately supports real control flow (`prim::If`,
   `prim::Loop`) that `torch.export` rejects outright, which is *why* something VITS-shaped breaks export
   in the first place. Real caveat, unchanged from before verification: TorchScript is a legacy PyTorch
   feature being superseded by `torch.export`/ExecuTorch (even `executorch-ggml` itself uses `torch.export`,
   not TorchScript), and `torch.jit.script` has notoriously incomplete coverage of real Python model code —
   needs validation against an actual model, not assumed to work just because the IR supports it in theory.
3. **Plain pickle / eager `nn.Module`** (neither export nor script succeeded): confirmed as a real fallback
   tier in Netron's source (`pytorch.Utility.weights(module)` walking `module._modules`) — but it produces
   **no op-level dataflow graph at all**, just a tree of named tensors/submodules. This is *how* Netron
   manages to always show something, but it's not actually a solution to the linking problem — a module
   tree with no operation sequence isn't sufficient for loom's topology JSON, which needs real op-level
   dataflow to execute. Worth remembering as a floor, not a lead.

**ONNX export, checked 2026-07-17 against a real file, not just theorized:** piper already ships a working
export (`pipertts_en-GB_miro/miro_en-GB.onnx`, produced by its own `export_onnx.py`, trace-based, opset 15),
so rather than reasoning about `torch.onnx.export`'s mechanism in the abstract, loaded it with `onnx.load()`
and inspected the real node graph directly. Findings:

- **Node names do preserve the full PyTorch module hierarchy, for free, as a slash-delimited path** — e.g.
  `/enc_p/encoder/attn_layers.0/conv_q/Conv`, `/dp/convs/norms_1.0/ReduceMean`, `/dp/post_flows.2/...` —
  down to individual `ModuleList` instance indices. Grouping the graph's 6183 nodes by top-level scope
  prefix cleanly recovers all four submodules from this discussion: `enc_p` (2469 nodes), `dp` (2769 — by
  far the largest single piece, consistent with the stochastic duration predictor being the most complex
  sub-graph), `flow` (308), `dec` (70). This *is* real, usable, zero-effort boundary information — no
  TorchScript required to get it.
- **The catch: only code that lives inside a named `nn.Module.forward()` gets a scope prefix.** A
  non-trivial number of nodes (mask construction, `commons.generate_path`/`sequence_mask`, the
  prior-expansion `matmul`s, the `torch.randn_like` noise injection, the reverse-flow iteration control —
  i.e. everything written directly in `SynthesizerTrn.infer()` itself rather than inside a submodule) come
  out with bare auto-incremented names (`Constant_47`, `Gather_3`, ...) and no scope prefix at all. This
  confirms, concretely rather than abstractly, the shape of the "how to wire the pieces back together" half
  of the open problem: naming-based slicing gets you the four *sub-module* graphs almost for free, but the
  *glue* between them — exactly the control flow this whole discussion already expects a C++ driver to
  own — has no module boundary to key off of, because it was never inside a module in the first place.
- Not yet checked: `onnx.utils.extract_model` (would mechanically confirm these name-prefix boundaries are
  actually cut-able into standalone sub-graph files) and `onnx.FunctionProto`. Also deliberately not chased
  yet, per the user's own framing of that as a separate later conversation: exactly how the exporter's
  trace-based path lowered `piecewise_rational_quadratic_transform`'s boolean-mask indexing (the function
  the user expects TorchScript `.script()` to break on) into concrete ops — the `Where`/`NonZero`/
  `ScatterND`/`GatherElements` ops visible in the graph's op histogram are almost certainly where that
  shows up, when that conversation resumes.

**Still to verify, raised 2026-07-17, not checked against real source yet:** `torch.export`'s
`preserve_module_call_signature` parameter. `ExportedProgram.graph_module` is itself an `fx.Graph` — the
"ATen graph" and "FX graph" aren't two different sources, just two points on the same decompose/inline
dial, and `torch.export` normally dials all the way to fully-decomposed-and-inlined (which is why it never
has module boundaries, unlike classic `torch.fx.symbolic_trace`'s `call_module` nodes). From memory (not
yet read against source, unlike the ONNX/Netron findings above), `preserve_module_call_signature` is
supposed to let you name specific submodules whose call boundary survives export instead of being inlined,
recorded as `module_call_graph` metadata on the `ExportedProgram` — which would give ATen-core ops (the
small, closed, tractable-to-map vocabulary executorch-ggml deliberately targets) *and* real module
boundaries for cutting, from a single export call, rather than choosing between clean ops (ATen/export) and
real boundaries (classic FX, TorchScript, or ONNX's name-prefix hack). Worth confirming this actually works
this way before leaning on it for the multi-graph-linking design.

**Empirical check 2026-07-17: tried `torch.jit.script` (not `.trace`) against the real `piper`/VITS
checkpoint** (`femelo/piper` fork, `pipertts_en-GB_miro`), to see whether the TorchScript path above is
actually reachable for the concrete model this whole discussion is about, not just reachable in theory.
Piper's own `export_torchscript.py` only ever calls `torch.jit.trace` — nobody has tried `.script()` on
this codebase before. Loaded `SynthesizerTrn` directly from the checkpoint's raw `state_dict` (bypassing
`VitsModel`/PyTorch Lightning entirely — the venv hit the exact same broken `huggingface-hub` pin that
blocked `transformers` in Milestone 8, and Lightning isn't needed for inference anyway).

Baseline (piper's source completely unmodified): `model_g.enc_p` (the `TextEncoder` — embedding +
self-attention stack) **scripts cleanly with zero changes**. `model_g.dp` (`StochasticDurationPredictor`),
`model_g.flow` (`ResidualCouplingBlock`), and `model_g.dec` (`Generator`, the HiFi-GAN vocoder) **all fail**.
Root-caused four distinct, genuinely fixable TorchScript incompatibilities (not vague "scripting is hard" —
specific lines, specific fixes), all in `piper_train/vits/modules.py` / `models.py`:

1. **Variadic `**kwargs`/`*args` forward signatures.** `Log`, `Flip`, `ElementwiseAffine`, `WN.forward` all
   accept `**kwargs` (and `Flip` also `*args`) purely as plumbing so a heterogeneous `nn.ModuleList` of flow
   steps can all be called uniformly as `flow(z, x_mask, g=x, reverse=reverse)`. TorchScript flatly rejects
   variadic signatures. Fix is mechanical: replace `**kwargs` with an explicit `g: Optional[Tensor] = None`
   parameter on each — confirmed by patching all four and re-scripting.
2. **Conditionally-registered submodules referenced unconditionally in `forward`.** `Generator.cond` and
   `WN.cond_layer` are only ever assigned when `gin_channels != 0` (i.e. multi-speaker models); this
   checkpoint is single-speaker (`gin_channels=0`), so the attribute never exists. `forward` still contains
   `if g is not None: x = x + self.cond(g)` unconditionally. Unlike tracing, `torch.jit.script` compiles
   *every* static branch regardless of whether it's dead at runtime, so it fails resolving `self.cond`
   before ever executing anything. Fix pattern: always construct the submodule (or a dummy placeholder) and
   guard on an explicit boolean instead of attribute presence.
3. **Dynamic-index `ModuleList` access.** `DDSConv.forward` (used inside `dp`'s `convs`/`post_convs`) does
   `self.convs_sep[i](...)` where `i` is a `for i in range(self.n_layers)` loop variable. TorchScript only
   supports `ModuleList` indexing with integer *literals* or `enumerate()`/`zip()`-style iteration, not an
   arbitrary variable — a well-documented, common TorchScript limitation. Fix: rewrite the loop to
   `for i, (conv_sep, ...) in enumerate(zip(self.convs_sep, ...))`.
4. **Legacy `torch.nn.utils.weight_norm` breaks TorchScript's hook machinery, silently, only for `.script`.**
   `flow`'s `WN.in_layers`/`res_skip_layers` are still weight-normed (piper's own export scripts only ever
   call `model_g.dec.remove_weight_norm()`, never on `flow`/`dp`'s conv layers, because `.trace()` doesn't
   care). `torch.jit.script` walks registered forward-pre-hooks during compilation and expects each hook
   object to expose `__name__`; the legacy `WeightNorm` hook class doesn't define one, so scripting *any*
   still-weight-normed submodule throws `AttributeError: 'WeightNorm' object has no attribute '__name__'`.
   This is a real, non-obvious asymmetry worth remembering generally: **tracing silently tolerates leftover
   weight-norm hooks; scripting does not.** Fix is either calling `remove_weight_norm()` everywhere before
   scripting (not just on the vocoder), or migrating to the newer `torch.nn.utils.parametrizations.weight_norm`
   (already the suggested replacement per today's `FutureWarning`, and implemented via the modern
   parametrization system rather than a bare unnamed hook — plausibly TorchScript-safe by construction,
   not independently confirmed here).

After patching all four (monkeypatched in a throwaway script only — **not applied to the actual `piper`
fork**, since this was exploratory and the fixes weren't validated for numerical correctness against the
original eager model): `enc_p` still clean, and `dp`/`flow` got past their original failures into deeper
compilation stages. `dec` hit a fifth, not-yet-root-caused error (`RuntimeError: Unsupported value kind:
Tensor`, no user-code file/line in the trace — likely something in `ResBlock2`/`ConvTranspose1d` post-
`remove_weight_norm()`, or the plain-tensor `self.attn = torch.zeros(1)` debug attribute in
`attentions.py:186`) — not chased further given the exploratory framing of this thread.

**Bottom line for the "linking" discussion:** the TorchScript path isn't just theoretically available (per
the Netron source above) — it's *empirically close* for a real, currently-unmodified third-party VITS
implementation, with the blockers found so far being small, mechanical, well-understood fixes rather than
fundamental incompatibilities. It is not yet a clean "just run `.script()`" story, and full correctness
(scripted-vs-eager numerical parity) was not checked before this thread paused. If this gets picked back up:
finish root-causing the `dec` failure, verify numerical parity of the patched flows against eager mode, and
only then decide whether these fixes are worth upstreaming into the `piper` fork for real.

**Concrete things worth cross-checking or considering, not yet verified against loom's own code:**

- **Flash-attention + native GQA as an optional `ATTENTION` fast path.** `executorch-ggml` maps
  `aten.scaled_dot_product_attention` to `ggml_flash_attn_ext` with ggml's *native* GQA support
  (`gqa_ratio = Q.ne[2] / K.ne[2]`, confirmed in `runtime/ops/ops_special.h`), and its README credits this
  (plus fused RoPE/RMSNorm-fold/SwiGLU) for the above-`llama.cpp` throughput numbers. loom's own
  `ATTENTION` primitive (`src/ops/primitives_attention.cpp`) is deliberately the *composite* (non-flash)
  path — chosen in Milestone 1 for exact fp32 reproducibility against a numpy/PyTorch reference, not for
  speed — and already gets GQA correctness "for free" via `ggml_mul_mat`'s own broadcast rule (verified in
  Milestone 8's `tests/test_e2e_gqa.cpp`). Worth prototyping an *optional* `ggml_flash_attn_ext`-backed
  variant (a new `attrs` flag on `ATTENTION`, analogous to the existing `"kv_cache"` flag) for a real
  throughput comparison on the now-converted Qwen3-0.6B-Base checkpoint — but only after confirming
  numerical tolerance against the existing composite path stays acceptable, same rigor as everything else
  in this backlog.
- **The build-time-vs-execute-time bug class, cross-referenced against loom's own graph-reuse discipline.**
  `PROGRESS.md`/`GRAPH_REBUILD.md` documents several real bugs from this exact family: M9 ("eager ops
  compute data during `build_graph()` by reading source tensor `->data`... data is uninitialized at build
  time"), M15 (eager `I32->I64` casts during graph build freezing pre-execution garbage values, breaking
  gather/scatter), M16 (a catastrophic KV-cache step-2 corruption, `~1e35` values, from the same root
  cause, fixed by moving the scatter into a runtime custom op instead of a build-time-eager one). This is
  the same general pitfall category as loom's own root-caused finding above (`ggml_gallocr` aliasing a
  reused graph's declared-input buffer with a previous pass's output) and the CMVN
  length-dependent-constant bug (Milestone 7) — three independent instances, across two different
  projects, of "a value computed once at graph-construction time silently going stale across reuse."
  Nothing to fix here (loom hasn't hit this specific variant), but worth keeping in mind as the same
  *class* of risk if loom ever adds an op whose semantics read tensor *data* (not just shape) at build
  time rather than purely describing a compute-graph node.
- **`sym_dim_ids` (dynamic-shape symbol resolution) as a precedent for `SymbolEnv`'s own design.**
  `executorch-ggml`'s dynamic-shape handling went through several iterations (`GRAPH_REBUILD.md` M8/M18):
  an early `dynamic_size_map` keyed by trace-time dimension *value* caused real collisions when two
  unrelated dimensions shared the same numeric value (e.g. `max_cache_len - 1 == 31` colliding with
  `trace_seq_len == 31`), fixed by switching to `sym_dim_ids` — each symbolic variable gets a stable
  integer *identity* at export time, resolved to a concrete value from input tensors at runtime, instead of
  being matched by value. loom's `SymbolEnv` (`src/core/symbol_env.cpp`) doesn't have this problem today
  (every symbol is a named hparam or an explicit `$n_tokens`-derived expression, resolved by *name*, never
  matched by coincidental value) — but this is a concrete cautionary precedent if `SymbolEnv` ever grows
  toward inferring/matching symbols from tensor shapes rather than always being told their names
  explicitly by the conversion script.
- **`ggml_backend_sched` for mixed CPU/GPU execution with pinned custom ops.** loom has deliberately
  deferred `ggml_backend_sched` (documented in "Deliberate simplifications": "CPU-only, no multi-backend
  need yet"). `executorch-ggml`'s M14 (`GRAPH_REBUILD.md`) is a concrete real use case for exactly this:
  pinning specific ops (comparison/logical custom ops) to CPU while the rest of the graph runs on
  Metal/CUDA, via `ggml_backend_sched_set_tensor_backend`. Worth revisiting when/if loom picks GPU backend
  support back up — not urgent now.
- **NOT recommended for adoption, but worth being aware of as a deliberately different design point**:
  `executorch-ggml`'s GGUF integration (`docs/gguf-integration.md`) exports a *weight-less* `.pte` (graph
  structure only, ~200KB) and loads weights from a separate GGUF file at runtime (`GGUFModule`), with
  benchmarked zero overhead vs. embedding weights directly (762MB in-file vs. 213KB external, same
  tok/s). This is the opposite of loom's own design, where topology JSON and weights deliberately live
  together in ONE self-contained GGUF (`model.graph_topology` KV + weight tensors, same file) — that's the
  whole point of loom's data-driven approach (a single artifact fully describes and contains a runnable
  model), so splitting them the way `executorch-ggml` does would work against that goal, not toward it.

License is BSD (confirmed from the repo's own `README.md`/`LICENSE`), so adopting genuinely novel
*concepts* (not verbatim code) from it is not a licensing concern — but nothing above has been implemented
or even prototyped yet; this section is purely the result of reading the repo, not a commitment to any of
it.

## Roadmap: quantized weight support (Q8_0 matmul, candidate for Milestone 9)

Not started — captured after the `executorch-ggml` comparison above surfaced its "optimized inference"
README claim ("ggml's hand-tuned kernels — quantized matmul, fused softmax, etc. — replace generic ATen
implementations"). That specific framing doesn't transfer to loom-engine (loom never goes through ATen at
runtime — every primitive already calls ggml directly), but it prompted checking whether loom could still
benefit from ggml's own quantized-matmul kernels directly, independent of executorch-ggml. It can — and the
gap turned out to be much smaller than expected, verified against the actual vendored ggml source
(`build/_deps/ggml-src`) and the real installed `gguf` Python package, not assumed:

**The C++ runtime side appears to need zero changes** (a real, previously-unverified claim now checked
against source, not assumed by analogy to llama.cpp):

- **`GgufModel::load`** (`src/core/gguf_model.cpp:9-70`) is already fully type-agnostic on the read path:
  `ggml_get_tensor` inherits whatever `ggml_type` `gguf_init_from_file` parsed straight from each tensor's
  own on-disk metadata, and the raw on-disk bytes are copied verbatim via `ggml_backend_tensor_set` — no
  F32 assumption anywhere, and `ggml_nbytes(t)` (used both for the staging-buffer size and by
  `ggml_backend_alloc_ctx_tensors`'s own buffer sizing) is already block-size-aware for quantized types. A
  GGUF containing genuine `Q8_0`-quantized tensors would load correctly today, unchanged.
- **`op_mul_mat`** (`src/ops/primitives_basic.cpp:24-26`) is a bare `ggml_mul_mat(ctx, in[0], in[1])` wrap.
  `ggml_mul_mat` itself (`ggml.c:3258`) only asserts shape compatibility (`ggml_can_mul_mat`) — no type
  assertion. Confirmed in the vendored `ggml-cpu.c` (`build/_deps/ggml-src/src/ggml-cpu/ggml-cpu.c:230`+)
  that `Q8_0`'s own type-traits entry sets `vec_dot_type = GGML_TYPE_Q8_0`: ggml's CPU backend already
  knows how to on-the-fly-quantize an F32 `b` (activations) operand and run a quantized×quantized dot
  product against a `Q8_0` `a` (weights) operand — the exact standard llama.cpp
  weights-quantized/activations-F32 pattern, already fully implemented in the vendored ggml, simply unused
  because nothing in loom's conversion tooling has ever written a non-F32 tensor.
- **`op_get_rows`** (`ggml_get_rows`, `ggml.c:3871`) has the same story: no type restriction on its `a`
  (table) operand, dequantizes internally, output stays F32 unless `a` is `I32` — a quantized
  `token_embd.weight` table would work through the existing `GET_ROWS` primitive unchanged too.

**The real gap is entirely on the conversion-tooling side, and it's small:**

- The `gguf` Python package (already a dependency of every `tools/convert_*/make_*_gguf*.py` script) ships
  a real, working pure-numpy `gguf.quantize(np_array, gguf.GGMLQuantizationType.Q8_0)` (confirmed by
  reading its source directly, not its docs) covering `Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8_0`/the `_K` family/
  etc., and `GGUFWriter.add_tensor(name, data, raw_dtype=...)` already accepts exactly the override needed
  to write pre-quantized bytes under a non-F32 GGUF tensor type. No new Python dependency required.
- **Only matmul weight tensors should be quantized** — `attn_q/k/v/output`, `ffn_gate/up/down`,
  `token_embd`/`output` — this is standard ggml/llama.cpp convention, called out explicitly here rather
  than assumed: norm weights (`RMS_NORM`'s per-channel scale), biases, and RoPE-related scalars are tiny
  and not the multiply-heavy tensors, so quantizing them has negligible size benefit and a real,
  needless precision cost.
- **KV-cache quantization is a separate, already-flagged, explicitly out-of-scope item** — "Scope
  limitations... KV cache storage is always F32" above. `KvCache` (`src/core/kv_cache.cpp:20-24`) always
  allocates `GGML_TYPE_F32` tensors regardless of weight type; weight quantization and KV-cache
  quantization are independent axes and shouldn't be conflated into one milestone.

### Reference reviewed 2026-07-18: `femelo/qwen3-asr.cpp`'s `src/quantize.cpp`

The user pointed at a real, working C++ quantizer from their own `qwen3-asr.cpp` fork (input: an
F16-quantized `llama.cpp`-converted GGUF; reportedly run successfully against 3-4 real models) as a
reference for this milestone. Read directly, not summarized from memory — one real logic bug found, plus a
design-level lesson worth carrying into loom's own version rather than copying the same approach:

- **Bug: the name-based exclusion list (`should_quantize`, skips `bias`/`norm`/`token_embd`/`ln_post`) only
  actually protects tensors that are already F32.** The control flow is: `if (should_quantize(...) &&
  (F32||F16)) { quantize to target_type }`, `else if (tensor->type == F16) { /* "smart fallback": Q8_0 if
  block-aligned, else F32 */ }`, `else { keep as-is }`. The fallback branch's condition never re-checks
  `should_quantize` — it only checks the tensor's stored dtype. So a name-excluded tensor that happens to be
  stored as **F16** on input skips the first branch (since `should_quantize` is false) and falls straight
  into the fallback branch, which quantizes it to Q8_0 anyway if its row length is block-aligned —
  completely defeating the exclusion for that tensor. `token_embd.weight` is the tensor most exposed to
  this: it's genuinely 2D (not a norm/bias vector `llama.cpp` conventionally keeps in F32 regardless of
  `--outtype`), commonly *does* get cast to F16 by `llama.cpp`'s own F16 conversion path, and its row length
  (`n_embd`, e.g. 1024) is almost always Q8_0-block-aligned. On a tied-embeddings model (Qwen3, per
  Milestone 8) the same tensor also serves as the final logits `MUL_MAT`, so this one gap would silently
  degrade both roles at once. It plausibly hasn't surfaced across the 3-4 models tested if their
  norm/embedding tensors happened to already be F32 on input — worth confirming against one of those
  GGUF's actual per-tensor dtypes before trusting it more broadly, not assumed safe here. Fix, if this file
  is patched: check `!should_quantize(...)` in the fallback branch too, and upcast excluded-but-F16 tensors
  to F32 rather than routing them through the Q8_0 fallback.
- **Smaller gaps, same file**: input tensors already stored as `BF16` match neither the quantize branch nor
  the F16-fallback branch, so they pass through completely unquantized with no log message (probably
  harmless for this script's actual inputs, but silent). A 1D tensor like `rope_freqs.weight` (present on
  models using explicit NTK/YaRN-style per-dimension RoPE frequency overrides — not universal, but real)
  wouldn't match any of the four name substrings and could pass the block-alignment check by coincidence,
  quantizing positional-encoding scale factors directly — a more damaging error class than quantizing a
  norm weight, since RoPE error compounds through every subsequent attention step rather than applying once.
- **The design lesson for loom's own quantizer, not a fix to this file**: `should_quantize`'s exclusion is a
  *name-substring deny-list*, tuned to the naming conventions of whichever architectures it's been run
  against so far — it doesn't generalize automatically to a new architecture whose norm/embedding tensors
  happen to be named differently (e.g. `ln1`/`ln_f` instead of containing `norm`), which matters a lot given
  this backlog's roadmap spans several architecturally distinct model families (VITS, Kokoro, Parakeet,
  Qwen3-ASR/TTS). Loom doesn't need to guess from names at all: the topology JSON already declares exactly
  which tensor feeds which op, so "should this tensor be quantized" can be answered definitively as "is this
  tensor referenced as a `MUL_MAT` weight input, and only that" — an **allow-list keyed off real graph
  structure** (walk the topology JSON's own `MUL_MAT` node inputs) instead of a deny-list keyed off tensor
  naming convention. Strictly more robust, and the concrete design choice to carry into step 1 of the
  "suggested next step" below: select which tensors to quantize by scanning the topology JSON for `MUL_MAT`
  input references, not by pattern-matching tensor names.

### POC done, 2026-07-18: Q8_0 quantization proven against the real Qwen3-0.6B-Base checkpoint

**The toy LLM turned out to be a dead end for this POC, caught before writing any test around it**:
`tools/fixture_gen/toy_llm_common.py` has `N_EMBD=8`, `N_FF=16`, `N_VOCAB=16` — every one of its matmul
weight tensors has a row length under Q8_0's 32-element block size, so *none* of them are quantizable at
all. Proving Q8_0 on the toy LLM would only exercise the "leave everything untouched" fallback path, not
real quantized inference. Pivoted straight to the real `Qwen3-0.6B-Base` checkpoint instead (already
converted and reference-verified earlier this session) — its `n_embd=1024`/`n_ff=3072` are comfortably
block-aligned, and it's the smallest real, already-in-hand model that can actually exercise this.

**`tools/quantize/quantize_gguf_q8_0.py`** (new): implements the topology-driven design from the
`quantize.cpp` review above — `collect_mul_mat_weight_names()` recursively walks the GGUF's own embedded
topology JSON (expanding `repeat_for` blocks via the GGUF's own `loom.n_layer` KV) and collects every
`MUL_MAT` node's first input (the weight-first argument, per loom's convention) as the exact set of
tensors to quantize; everything else (KVs and non-matmul tensors) is copied through byte-identical via a
generic `GGUFReader`-field walk (`copy_kv`), not re-derived per architecture. Ran against the real
checkpoint:

```
wrote qwen3_q8_0.gguf: 197 tensors -> Q8_0, 0 MUL_MAT weights left F32 (not block-aligned), 113 other tensors left F32
```

197 = 28 layers × 7 matmul weights (`attn_q/k/v/output`, `ffn_gate/up/down`) + 1 (confirms the
self-adapting design claim for real: `token_embd.weight` *is* picked up here, since Qwen3 has tied
embeddings and the same tensor is also the final logits `MUL_MAT` input — no special-casing needed for
that, unlike a name-based rule which would need an explicit tied-embeddings carve-out). 113 = 28×4 norm
weights (`attn_norm`/`ffn_norm`/`attn_q_norm`/`attn_k_norm`) + `output_norm`, correctly left F32. File size
went from 2.39 GB to 639 MB (~3.74×, matching Q8_0's 32×F32→34-byte packing ratio exactly).

**One real bug caught and fixed while writing the script, before it ever touched real data**: initially
passed `raw_shape=<the pre-quantization logical F32 shape>` to `GGUFWriter.add_tensor()`, assuming that's
what "raw shape" meant. Checked `add_tensor_info`'s actual source first — `raw_shape` (when given) is fed
straight into `quant_shape_from_byte_shape`, which expects a **byte**-shape (last dim = packed row size in
bytes), not the logical element-shape; passing the logical shape there would have silently computed a
wrong element count. Confirmed the fix (omit `raw_shape` entirely, let it default to the quantized array's
own byte-shape, which `add_tensor_info` then correctly converts back internally) against a standalone
`(151936, 1024)` repro before trusting it in the real script.

**`tests/test_e2e_qwen3_q8_0.cpp`** (new, mirrors `test_e2e_qwen3.cpp`'s structure, `SKIP_RETURN_CODE 77`
pattern, reuses its reference fixture since quantization happens after conversion — same expected
pre-quantization logits): loads the Q8_0 GGUF, generates against the same prompt, compares against the
existing F32 reference. Real measured result, not guessed: **max abs logit diff ranged 0.45–0.79 across
the 4 generation steps**, comfortably under the `1.5` tolerance set from that measurement (roughly 2× the
observed max) — and, notably, **the argmax-token sequence matched the F32 reference exactly at every
step** despite the lossy quantization, though the test deliberately doesn't hard-assert that (logged
instead), since token-level agreement isn't guaranteed by tolerance-bounded logit closeness in general.
**13/13 checks passed.** Full `ctest` suite: **40/40 passing**, new test skips cleanly by default
(`SKIP_RETURN_CODE 77`) and passes for real with `LOOM_QWEN3_Q8_0_DIR`/`LOOM_QWEN3_DIR` set — zero
regressions, and critically, **zero C++ engine changes were needed anywhere**, confirming the "no engine
change required" claim from the verified-facts section above empirically rather than just by source
inspection.

**One incidental, not rigorously benchmarked observation**: the same 4-step generation ran in 1.92s
(Q8_0) vs. 7.31s (F32, `test_e2e_qwen3`) in this one run — a real ~3.8× difference, consistent with the
expected memory-bandwidth benefit of a 3.74×-smaller weight set, but a single anecdotal timing on a shared
dev machine, not a controlled benchmark (no repeated runs, no warmup control, no isolation from other
load) — worth a real benchmark pass before quoting this number anywhere else.

**Still out of scope after this POC**: KV-cache quantization (separate, already-flagged item above);
non-Q8_0 quant types (`Q4_K` etc. — the same script's `quantize()` call would need `GGML_TYPE_Q4_K` swapped
in and a real accuracy check, since K-quants trade more compression for more error); wiring quantization
into the actual conversion scripts (`convert_qwen3.py`) as a `--quantize` flag rather than a separate
post-conversion pass; and a real benchmark (not the anecdotal timing above) to quantify the actual
throughput benefit before treating it as a settled win.

## VITS conversion, in progress, 2026-07-18: `WN`/`ResidualCouplingLayer`/`Flip` (flow's coupling blocks)

Real checkpoint (`pipertts_en-GB_miro/epoch=9772-step=1494014.ckpt`) confirmed via `models.py`'s
`SynthesizerTrn.__init__` (`self.flow = ResidualCouplingBlock(inter_channels=192, hidden_channels=192,
kernel_size=5, dilation_rate=1, n_layers=4, gin_channels=0)`, `n_flows=4` default): single-speaker
(`gin_channels=0`, so `WN`'s conditioning input `g` is always `None`) and every `ResidualCouplingLayer`
built with `mean_only=True` (confirmed in `ResidualCouplingBlock.__init__`, not assumed).

**`src/ops/primitives_flow.cpp`** (new): two new composed primitives, each looping internally in C++
over a statically-known layer count (same "bundle a repeated substructure into one primitive with a
variadic `Inputs` list" precedent as `REL_POS_ATTENTION_SHAW`, rather than expanding into dozens of JSON
graph nodes per layer):
- **`WN`**: WaveNet-style dilated gated conv1d stack (`modules.py`'s `WN.forward`). Since this engine
  only ever runs a single, unpadded utterance, the real code's `x_mask` (all-ones here) and the `g`
  conditioning path (unused, `gin_channels=0`) both drop out entirely — `fused_add_tanh_sigmoid_multiply`
  collapses to a plain `tanh(first_half)*sigmoid(second_half)` gate with no added bias term. Conv weights
  are taken as already-plain (weight-norm-folded) kernel tensors — folding `weight_g`/`weight_v` into a
  plain kernel is a conversion-time concern (task #78), not this primitive's.
- **`RESIDUAL_COUPLING_LAYER_REVERSE`**: `modules.py`'s `ResidualCouplingLayer.forward(reverse=True)`,
  specialized to `mean_only=True` (this checkpoint's only configuration) — `logs` is always the zero
  tensor so `exp(-logs)=1` and the affine reverse collapses to a plain `x1' = x1 - m`; `x0` passes through
  unmodified. Internally calls the same `WN` C++ logic (factored as a local helper, not a cross-file
  registry lookup) between its `pre`/`post` 1x1 convs.

**`Flip` needed no new primitive at all** — `torch.flip(x,[1])` (reversing the whole channel axis) is
exactly what `GET_ROWS` already does when given a conversion-time-baked reversed-index constant
(`[C-1, C-2, ..., 0]`, I32): `GET_ROWS(x, reversed_idx)` selects channel-rows in reverse order. Simpler
than the original plan's "bake an anti-diagonal permutation matrix and use `MUL_MAT`" idea (which would
also have needed a transpose in and out, since `MUL_MAT` contracts over `ne[0]` and channels live in
`ne[1]` under this engine's `[T, C]` flow-tensor convention) — caught while implementing, before writing
any of the `MUL_MAT` version.

Both new primitives verified against real execution of piper's own `modules.WN`/`ResidualCouplingLayer`/
`Flip` classes (small hand-picked dims: `hidden_channels=4`, `kernel_size=3`, `dilation_rate=2`,
`n_layers=2`, `T=5`; random-but-seeded weights, expected outputs obtained by literally running the real
modules, not hand-derived) in `tests/test_primitive_registry.cpp` (`test_wn`,
`test_residual_coupling_layer_reverse`, `test_flip_via_get_rows`) — all passing on the first C++ attempt
after the usual numpy/PyTorch pre-verification discipline. Full suite: **49/49 passing**, zero
regressions.

## VITS conversion, in progress, 2026-07-18: `DDSConv`/`ElementwiseAffine`/`ConvFlow` assembly, and a real `ggml_clamp` aliasing bug

**Real bug found and fixed, not just a new-feature addition**: `ggml_clamp`'s result is a **view aliasing
its source's own buffer** (confirmed directly in `ggml.c`: `ggml_clamp` calls `ggml_view_tensor(ctx, a)`,
not `ggml_dup_tensor` — unlike `ggml_cont`, which always allocates a genuine new tensor). This means
`ggml_clamp(ctx, a, lo, hi)` clamps **in place**: once that op executes, `a`'s own buffer is overwritten
with the clamped values, so *any other node that also reads `a`* — regardless of where it sits in the
JSON topology or C++ call order — silently observes the clamped value instead of the original from that
point on. `RQ_SPLINE_INVERSE`'s outside-tail-bound identity-passthrough blend does exactly this (needs
both the clamped `x_clamped`, for the in-bin spline math, and the original unclamped `inputs`, for the
outside-domain passthrough and for classifying inside/outside in the first place) — clamping `inputs`
directly silently corrupted the classification and the passthrough both, without any crash or NaN to flag
it. **Caught by `test_rq_spline_inverse_outside_tail_bound`**, a new test added specifically because
`test_rq_spline_inverse`'s original fixture never actually fed the spline an out-of-domain input (its 3
values all happened to fall inside `tail_bound`) — a genuinely different fixture (an input past
`tail_bound`) was needed to expose it, not a rerun of the existing one. Symptom was suspicious but not
obviously a bug at first glance: the wrong output (`2.0`) was *exactly* `tail_bound` — i.e., the code was
silently returning the clamped boundary value instead of passing the true out-of-domain input through
unchanged, not a crash or a NaN.

Fixed in two places: `RQ_SPLINE_INVERSE` (`src/ops/primitives_spline.cpp`, clamp a `ggml_cont`'d copy of
`inputs` instead of `inputs` itself) and the standalone `CLAMP` primitive (`src/ops/primitives_basic.cpp`)
defensively, even though nothing currently feeds a shared/multiply-referenced tensor into it — `CLAMP` is
a generic, independently reusable primitive with no way to know from inside its own function whether its
input tensor is also read elsewhere in whatever topology eventually uses it. **General lesson for any
future primitive work in this codebase**: before using any ggml op whose result might alias its input
(check the real `ggml.c` source for `ggml_view_tensor`/`ggml_view_impl` in the op's implementation, not
just its header signature), verify whether the original input tensor is still needed elsewhere in the
same primitive or graph — `ggml_clamp` is the first one found doing this, but nothing rules out others
(worth spot-checking before reusing any new-to-this-codebase ggml op that returns something "of the same
shape as its input").

**`DDSConv`/`ElementwiseAffine`/`ConvFlow` (reverse) added** to `src/ops/primitives_flow.cpp` (new
primitives: `DDS_CONV`, `ELEMENTWISE_AFFINE_REVERSE`, `CONV_FLOW_REVERSE`), completing the
`StochasticDurationPredictor`'s flow-list building blocks (only `ElementwiseAffine` and `ConvFlow`+`Flip`
pairs appear in its `flows` list; `Flip` itself needed no new code, per the `GET_ROWS` finding above).
`DDSConv` reuses `CONV_1D_DW` via a `PrimitiveRegistry` lookup (rather than duplicating its non-trivial
im2col/reshape recipe a second time) and needs a `LayerNorm`-over-channels helper (`layer_norm_channels`)
that transposes so the channel axis lines up with `ggml_norm`'s own normalization axis (`ne[0]`) —
mirroring the real `modules.LayerNorm`'s own `transpose→layer_norm→transpose` trick exactly, not an
independent reformulation. `ConvFlow` is specialized to `half_channels=1` (the *only* configuration this
checkpoint ever instantiates — `StochasticDurationPredictor` is `ConvFlow`'s sole caller, always with
`in_channels=2`), which collapses the real code's `reshape(b,half_channels,-1,t).permute(...)`
bin-parameter gymnastics down to a plain transpose, and delegates the actual spline math to
`RQ_SPLINE_INVERSE` via the registry rather than duplicating it. All three verified against real execution
of piper's own `modules.DDSConv`/`ElementwiseAffine`/`ConvFlow` (small hand-picked fixtures; `ConvFlow`'s
own fixture deliberately includes one value past `tail_bound`, so it doubles as an integration-level check
of the aliasing fix above, not just the assembly logic). Full suite: **49/49 passing**, zero regressions,
115/115 primitive-level checks.

## VITS conversion, in progress, 2026-07-18: `StochasticDurationPredictor` reverse-mode assembly (task #76)

**No new primitive needed** — SDP itself has no repeated-substructure math beyond what
`CONV_1D`/`DDS_CONV`/`CONV_FLOW_REVERSE`/`ELEMENTWISE_AFFINE_REVERSE`/`GET_ROWS` (as `Flip`) already
cover; the actual work here was getting the **wiring/ordering** right, verified in
`test_sdp_reverse_assembly` (`tests/test_primitive_registry.cpp`) by chaining the existing primitives via
direct `PrimitiveRegistry` calls in the exact real order, not by writing a new bundled op.

**One more real, checkpoint-relevant finding, not just a test-construction detail**: `models.py`'s real
`StochasticDurationPredictor.forward` reverse branch does
`flows = list(reversed(self.flows)); flows = flows[:-2] + [flows[-1]]` (a "remove a useless vflow"
comment in the source itself). Traced through exactly what this drops: with the real `n_flows=4`, `self.
flows` is `[EA, CF0,Flip0, CF1,Flip1, CF2,Flip2, CF3,Flip3]` (9 entries); reversed and filtered, the final
applied sequence is `[Flip3, CF3, Flip2, CF2, Flip1, CF1, Flip0, EA]` — **`CF0` (the very first
`ConvFlow`) is dropped entirely and never executes at inference**, despite its weights existing in the
checkpoint's state dict. Confirmed with a small-scale (`n_flows=2`) worked example in
`test_sdp_reverse_assembly`'s own doc comment before trusting the general pattern. Also confirmed
`self.log_flow` (`modules.Log`) is **only ever used in the `not reverse` (training) branch** — never
touched at inference either. Both are real, checkpoint-specific facts the eventual conversion script
(task #78) needs to know: `CF0`'s and `log_flow`'s weights can be safely skipped/ignored when building the
inference-only topology, not converted at all.

Cross-checked against a hand-replica of the real reverse-mode control flow using piper's own sub-module
classes directly (`modules.ElementwiseAffine`/`ConvFlow`/`Flip`/`DDSConv`, `nn.Conv1d` — not the
`StochasticDurationPredictor` class itself, since its `__init__` hardcodes `DDSConv`'s `n_layers=3` with
no override, too large for a quick test; `n_flows=2` used instead of the real 4, the minimum that still
leaves one `ConvFlow` surviving the filter above) with small dims (`in_channels=filter_channels=2`,
`kernel_size=3`, `num_bins=3`, `tail_bound=2.0`, `T=2`). Full suite: **49/49 passing**, zero regressions,
**116/116 primitive-level checks**.

## VITS conversion, in progress, 2026-07-18: HiFi-GAN `Generator` vocoder (task #77)

Confirmed the real checkpoint uses `resblock="2"`/"low_quality" config, not "1" — from the state dict
directly (`model_g.dec.resblocks.*.convs.{0,1}` only, no `convs2`; `ResBlock1` would have both), not
assumed from the piper repo's own defaults. **No new primitive needed**, exactly as the original plan
predicted: `conv_pre`/`conv_post`/resblock convs are all `CONV_1D` (dilation already a generic attr),
upsampling is `CONV_TRANSPOSE_1D`, activations are `LEAKY_RELU`/`TANH` (already registered), residual/
fan-out-averaging is plain `ADD`/`SCALE`.

**One real, newly-found primitive gap, though — not a missing op, a missing *feature* of an existing
one**: `ggml_conv_transpose_1d` only supports `padding=0` (asserted in ggml's own source), but the real
`nn.ConvTranspose1d(..., padding=p)` calls in `Generator.__init__` use nonzero `padding=(k-u)//2`. Fixed
without any new primitive: computed the padding=0 ("full") output, then cropped `p` samples off **each**
end via a plain view — verified this crop is an *exact* identity (not an approximation) by comparing real
PyTorch `ConvTranspose1d(padding=p)` against `ConvTranspose1d(padding=0)` sliced by `[:, :, p:-p]`
directly before relying on it (transposed-conv padding conventionally *removes* output samples, unlike
regular conv where padding *adds* input samples — the opposite direction from what the same word means
elsewhere in this codebase, worth remembering if this comes up again for any other transposed-conv use).

Also confirmed exactly two *different* LeakyReLU slopes are used in the real code, easy to mix up: `0.1`
(`self.LRELU_SLOPE`) inside every resblock and after every upsample stage, but PyTorch's **default**
`0.01` (`F.leaky_relu(x)`, no explicit slope argument) exactly once, right before `conv_post` — both
already just different `attrs.slope` values on the same registered `LEAKY_RELU` primitive, no engine
change needed either way.

Verified against real execution of piper's own `models.Generator`, small-scale (2 upsample stages instead
of the real 3, 2 resblock kernel sizes instead of 3: `initial_channel=4`, `upsample_initial_channel=8`,
`upsample_rates=(2,2)`, `upsample_kernel_sizes=(4,4)`, `resblock_kernel_sizes=(3,5)`,
`resblock_dilation_sizes=((1,2),(2,6))`, `T=4`) in `test_hifigan_generator`
(`tests/test_primitive_registry.cpp`) — passed on the first attempt. Full suite: **49/49 passing**, zero
regressions, **118/118 primitive-level checks**.

## VITS conversion, in progress, 2026-07-18: `TextEncoder` assembly (task #78, first sub-piece)

No new primitive needed — `conv_q`/`conv_k`/`conv_v`/`conv_o`/`proj` are all kernel_size=1 convs, i.e.
plain per-position linear projections, expressed via `MUL_MAT` directly (this engine's standard
attention-convention idiom, e.g. Qwen3's own QKV projections) rather than `CONV_1D` — avoids unneeded
transposes to/from `CONV_1D`'s `[T,C,N]` layout. `attentions.Encoder`'s post-norm `LayerNorm` needs **no**
transpose here (unlike `DDSConv`'s `[T,C]` convention): `TextEncoder`'s attention pipeline is channel-first
(`[C,T]`, matching `REL_POS_ATTENTION_SHAW`'s own convention and `GET_ROWS`'s embedding-lookup output),
which is already `ggml_norm`'s own normalization axis (`ne[0]`). Only the FFN's kernel_size=3 convs
genuinely need `CONV_1D` (hence a transpose in and back out).

**Two real bugs found and fixed while writing `test_text_encoder_assembly`, both in the TEST itself, not
the engine** (worth remembering for any future test with more than one graph output):

1. **Multi-output graphs need `ggml_set_output()` on every output, not just the one passed to
   `ggml_build_forward_expand`**. This test has three co-equal outputs (`x`, `m`, `logs` — none reachable
   from the others). Using `GgmlScratch::expand()`'s single-output convenience wrapper (which calls
   `ggml_gallocr_alloc_graph` immediately after building forward from just one tensor) left `m`/`logs`'s
   own nodes completely unallocated — `ggml_backend_tensor_set` on their input dependencies aborted with
   `"tensor buffer not set"`. Building forward from all three via three separate
   `ggml_build_forward_expand` calls before ONE `ggml_gallocr_alloc_graph` call fixed the abort, but
   revealed a second, quieter bug:
2. **`gallocr`'s liveness analysis reuses a tensor's buffer once nothing reads it again — silently
   corrupting any output tensor not explicitly marked `ggml_set_output()`**, even after fixing bug 1
   above. `x` (the encoder's own hidden-state output, needed both as a return value AND as an input to
   `proj`) has no reader after `stats = mul_mat(proj_w, x)` computes; without `ggml_set_output(x)`,
   `gallocr` freed its buffer for reuse by a later tensor in the same graph, so reading `x` back out after
   `s.compute()` returned a different tensor's data (not garbage or a crash — a real, differently-shaped
   but same-*sized* tensor's values, which is what made this one sneaky: the read-back values were a
   genuine permutation of *some* real computed values in the graph, not obviously wrong at a glance).
   `graph_builder.cpp`'s own single designated output already calls `ggml_set_output(result.output)` for
   exactly this reason; a hand-built test graph with more than one output needs to do the same for
   *every* one of them. Fixed by adding `ggml_set_output(x); ggml_set_output(m); ggml_set_output(logs);`
   before the graph-build/alloc sequence.

**A third apparent mismatch, investigated and found to be a comparison-order artifact, not a bug**: after
fixing both allocation issues above, the computed values were still "wrong" — but turned out to be an
exact permutation of the expected values, not incorrect numbers. Root cause: the real PyTorch
`TextEncoder.forward` does `x = torch.transpose(x, 1, -1)` early on and everything downstream (attention,
FFN, `proj`) operates on that post-transpose `(B,C,T)` layout, materialized contiguous (T fastest) by the
time it's dumped via `.flatten()`. This test's ggml pipeline, by contrast, never needed that transpose at
all — `GET_ROWS`'s embedding-lookup output is already channel-fastest (`[C,T]`, C=ne[0]), which is the
*pre*-transpose PyTorch layout reversed, and every subsequent op (attention, `LAYER_NORM`, the FFN's own
transpose-to-conv-and-back) stays self-consistent in that same convention throughout. Both computations are
correct; they just store the identical result with T/C swapped in memory relative to each other. Fixed by
regenerating the expected arrays via `.transpose(1, 2).flatten()` in the reference script instead of a
plain `.flatten()`, confirmed to match exactly once done. **General lesson: when a computed array is a
value-for-value permutation of the expected one, check axis order before assuming a math bug.**

Small-scale (`n_vocab=5`, `hidden_channels=4`, `n_heads=2`, `filter_channels=6`, `kernel_size=3`,
`n_layers=1`, `out_channels=4`, `T=2`, `window_size=1` — chosen so `2*window_size+1 == 2*T-1 == 3`,
sidestepping the real, genuinely-dynamic-length `emb_rel_k`/`emb_rel_v` pad/crop-to-T logic, which remains
deferred to the full conversion script/e2e test below), cross-checked against real execution of piper's
own `models.TextEncoder`. Full suite: **49/49 passing**, zero regressions, **121/121 primitive-level
checks**.

## VITS conversion, in progress, 2026-07-18: real conversion script runs end-to-end against the real checkpoint

**`tools/convert_piper_vits/`** (new): `vits_common.py` (checkpoint loading, `fold_weight_norm`,
`get_relative_embeddings`) and `convert_vits.py` (the real conversion script), producing three GGUF files
— `vits_stats.gguf`, `vits_logw.gguf`, `vits_flow_vocoder.gguf`. Confirmed a real API constraint while
wiring this up that the earlier design notes hadn't accounted for: **`GgufModel::load` requires exactly
one `"model.graph_topology"` KV per file** (`gguf_model.cpp:60-65`), so a single file can't hold both the
`stats` and `logw` topologies under custom-named KVs as originally planned — `convert_parakeet_tdt.py`'s
real precedent (re-examined) is "one GGUF file per topology, weights duplicated across files as needed,"
not "one file, multiple topology KVs." Switched to that: `vits_stats.gguf` and `vits_logw.gguf` both
carry their own full (partially redundant) TextEncoder weight copy.

**`fold_weight_norm` verified against the real checkpoint's own tensors**, not just a synthetic
construction: `torch._weight_norm(v, g, 0)` (the literal function `torch.nn.utils.weight_norm`'s
forward-pre-hook calls) matches the manual numpy formula `g * v / ||v||` (per-dim-0-slice L2 norm) to
~1e-8 on `model_g.dec.ups.0`'s real `weight_g`/`weight_v`. An earlier, sloppier check (reading
`.weight` off a freshly-wrapped module without ever calling `.forward()`) appeared to show a mismatch —
turned out to be a test-methodology bug (the weight_norm forward-PRE-hook only refreshes `.weight` on an
actual forward call; reading it beforehand returns the module's stale, randomly-initialized value) rather
than a real formula problem, caught by comparing against `torch._weight_norm` directly instead of the
module's cached attribute.

**`get_relative_embeddings` verified against the real `_get_relative_embeddings`** across both its
branches (`length <= window_size+1`: slice from the fixed table; `length > window_size+1`: zero-pad then
slice) for lengths `[3, 5, 7, 12]` against `window_size=4` — exact match in all four cases.

**Three real bugs found and fixed while getting the script to actually build+compute against the real
checkpoint** (`test_e2e_vits_smoke`, a new structural smoke test — builds every declared topology with
dummy zero-filled inputs and checks for finite output, not yet a numerical-correctness check):
1. Declared-input `shape` arrays must be **all-string** (`GraphTopology::parse` does
   `.get<std::vector<std::string>>()`, `graph_topology.cpp:57`) — a literal integer `2` in `z_noise`'s
   shape (`["$n_tokens", 2]`) threw `json.exception.type_error.302` at parse time. Every other declared
   input already used strings for its non-symbol dims (`str(channels)` etc.) except this one, missed on
   a first pass.
2. **The `GgufModel::load` single-topology-KV constraint** above.
3. **A genuinely subtle shape bug**: TextEncoder's `conv_q`/`conv_k`/`conv_v`/`conv_o`/`proj` and SDP's
   `pre` are consumed via a plain `MUL_MAT` node (not `CONV_1D`) for this engine's usual "channel-first
   attention convention" reasons (see the TextEncoder-assembly entry above) — but their real PyTorch
   weight shape is `(out, in, 1)` (kernel_size=1 convs keep an explicit trailing `K=1` dim). Writing that
   3D array to GGUF unmodified round-trips to a ggml tensor with `ne=[1, in, out]` (GGUF/ggml both store
   dims fastest-first, i.e. reversed from numpy's slowest-first order) — `MUL_MAT` then contracts against
   `ne[0]=1` instead of `in`, tripping `GGML_ASSERT(ggml_can_mul_mat(a, b))` at graph-build time. Fixed
   with a new `add_conv1x1_as_matmul` helper that squeezes the trailing dim to a plain 2D `(out, in)`
   array specifically for the six weights consumed this way — `add_conv` (used for every genuine
   `CONV_1D` consumer, where that trailing `K` dim is required, including when `K=1` for e.g. DDSConv's
   `convs_1x1`) is untouched. **General lesson for any future MUL_MAT-as-conv1x1 idiom in this codebase**:
   a real PyTorch conv weight's trailing kernel-size dim must be explicitly squeezed before it can be
   treated as a plain matrix — the two representations are byte-compatible only if that dim is 1 AND
   removed, never automatically.

**End-to-end structural result**: all three topologies build and compute successfully against the real
checkpoint's actual weights with a small dummy `T=5`: `stats` → 1920 = 2×192×5 elements (correct), `logw`
→ 5 elements (correct), the flow+vocoder waveform → 1280 = 5×8×8×4 samples (correct, matches
`upsample_rates`' product exactly). All finite, no crashes. Full suite: **50/50 passing** (the new
`test_e2e_vits_smoke` skips cleanly without `LOOM_VITS_DIR` set, same convention as every other
real-checkpoint test).

## VITS conversion, in progress, 2026-07-18: `VitsDriver` — the full two-phase pipeline runs end-to-end

**`include/loom/core/vits_driver.h`/`src/core/vits_driver.cpp`** (new): the host-side driver tying the
three GGUF files together, following the `TdtDecoder`/`OdeStepper` "TTS Catch" precedent (own the
`GraphTopology`s as members constructed before the `GraphBuilder`s that reference them, models referenced
not owned). `VitsDriver::synthesize(token_ids, seed)`:
1. Builds `stats` (n_tokens=T) and reads back `[2*inter_channels, T]` (channel-first) → `m_p`/`logs_p`
   split by a plain channel-offset read, no in-graph split (same host-side-postprocessing idea as the
   `stats`/`logw` topology split itself).
2. Builds `logw` (n_tokens=T, plus host-sampled `z_noise = randn(T,2)*noise_scale_w`) and reads back the
   `[T]` duration logits.
3. **`generate_path`, done host-side** (confirmed the right call once again): `w_ceil[t] =
   ceil(exp(logw[t])*length_scale)`, `y_length = max(sum(w_ceil), 1)` — with `x_mask`/`y_mask` dropped
   (always all-ones: single unpadded utterance), the real alignment-matrix math (`commons.py::
   generate_path`) degenerates to a plain "replicate column `t` of `m_p`/`logs_p` for `w_ceil[t]`
   consecutive output frames" expansion, sampling `z_p[t'] = m_p[t] + randn()*exp(logs_p[t])*noise_scale`
   directly into a `[y_length, inter_channels]` (`T`-major) buffer — no attention/alignment matrix is ever
   materialized, in ggml or otherwise.
4. Builds `flow_vocoder` with `n_tokens=y_length` (the just-computed, genuinely data-dependent value —
   exactly the case `GraphBuilder::build`'s two dynamic scalar arguments exist for) and reads back the
   waveform.

**`pad_crop_relative_embeddings`** (C++ port of `vits_common.get_relative_embeddings`, itself verified
against the real `_get_relative_embeddings` — see the TextEncoder-assembly and conversion-script entries
above): reads each TextEncoder layer's fixed `(2*window_size+1, k_channels)` raw table back out of the
GGUF (a NEW, topology-unreferenced `*_raw` tensor the conversion script now also writes, purely so the
driver has something to read — the declared graph inputs `emb_rel_k_i`/`emb_rel_v_i` are the *computed*,
call-specific tables, not storage for the learned parameter itself) and produces the real call's
`(2*T-1, k_channels)` table at every `synthesize()` call, since real `T` varies per input text and can't
be baked in at conversion time.

**Verified end-to-end against the real checkpoint** (`test_e2e_vits_driver`, new): 10 arbitrary in-vocab
phoneme ids → 6656 waveform samples (26 frames × 256 hop length), all finite, RMS≈0.0099, max
abs≈0.041 — comfortably inside tanh's `[-1,1]` range with real, non-degenerate variation (not silence,
not NaN/Inf). The low amplitude is plausible-but-unconfirmed as simply reflecting that these are
arbitrary token ids, not real phonemized text run through piper's own blank-interleaving convention
(`_` between phonemes) — not chased further here since numerical correctness needs the still-open
hand-rolled full-model reference anyway, not vibes about amplitude. Full suite: **51/51 passing**, both
new VITS tests (`test_e2e_vits_smoke`, `test_e2e_vits_driver`) skip cleanly without `LOOM_VITS_DIR` set.

## VITS conversion, done, 2026-07-18: numerical correctness confirmed against the real checkpoint (task #78 core work complete)

**Real phonemization confirmed and reproduced** (espeak-ng installed, `piper_phonemize` available in the
piper venv): piper's inference-time text→phoneme→id pipeline runs almost entirely **outside** its own
`piper` Python package, via `piper_phonemize` — an in-process binding to a **custom espeak-ng fork**
(`github.com/rhasspy/espeak-ng`, adds `espeak_TextToPhonemesWithTerminator`, not available from the
stock `espeak-ng` CLI) that also injects clause-boundary punctuation phonemes (`.`/`,`/`?`/`!`/`:`/`;`/
space) espeak's own phoneme string doesn't include on its own. The phoneme→id conversion (pure Python,
`voice.py::phonemes_to_ids`) is `[id["^"]] + interleave(phoneme_ids, id["_"]) + [id["$"]]` — BOS, every
phoneme followed by a blank/pad id, EOS — confirmed by direct inspection and reproduced exactly in
`reference_forward_vits.py`. `tools/convert_piper_vits/vits_common.py`'s phonemization notes and
`test_e2e_vits_driver`'s real test input now use the real `phonemize_espeak("Hello world, this is a
test.", "en-gb-x-rp")` → `phonemes_to_ids` output (T=62), not arbitrary in-vocab ids.

**All three topologies now verified NUMERICALLY correct against the real checkpoint**, not just
structurally (finite/right-shape) — closing out the "still open" item from the previous entry:
- **`stats` (TextEncoder)**: deterministic (no randomness anywhere in TextEncoder), directly comparable.
  `test_e2e_vits_stats_reference`: max abs diff `m_p`=2.9e-6, `logs_p`=9.5e-7 against a real
  `models.TextEncoder` forward pass, T=62 (genuinely exercises the emb_rel pad branch, `T > window_size+1`).
- **`logw` (TextEncoder+SDP reverse)**: SDP is genuinely stochastic (`z = torch.randn(...)` inside the
  real model itself) — made comparable by monkeypatching `torch.randn` (via `unittest.mock.patch`) to
  return a fixed, externally-generated noise array for the exact shape SDP's own forward calls it with,
  then feeding that SAME array into the topology's own `z_noise` declared input. `test_e2e_vits_logw_reference`:
  max abs diff = 1.4e-5, T=62 — validates the ENTIRE real spline-flow assembly (ConvFlow/DDSConv/
  ElementwiseAffine/Flip wiring) end to end, not just the small hand-picked-dims unit tests from earlier
  in this effort.
- **`flow_vocoder`**: also deterministic given a fixed `z_p` (no sampling happens inside the flow or
  vocoder themselves — the *sampling* that produces `z_p` is `VitsDriver`'s own responsibility, not
  baked into the flow/vocoder graph). `test_e2e_vits_flow_vocoder_reference`: max abs diff = **5.3e-8**
  (effectively bit-exact) against a real `ResidualCouplingBlock`+`Generator` forward pass.

**A real bug found and fixed WHILE building these reference tests, not a bug in the engine** — the exact
same class of mistake as `test_text_encoder_assembly`'s earlier axis-order confusion (see that entry
above), but this time in a **test/reference-generation script**, not in `convert_vits.py` or the engine
itself: the first attempt at dumping `z_p`/`z`/`wav` reference arrays used
`tensor[0].transpose(0,1).contiguous().numpy()` (converting PyTorch's native `(1,C,T)` to `(T,C)` before
saving) — but this engine's `[T,C]` flow/vocoder convention (`T=ne[0]`, fastest) is byte-identical to
PyTorch's **native, untransposed** `(C,T)` layout (numpy row-major, `T` fastest) once the batch dim is
dropped. Transposing before saving silently produced a same-values-different-order mismatch (max abs
diff ≈ the signal's own amplitude, i.e. looked like total garbage) that was **not** a real engine or
wiring bug — confirmed by isolating just the flow half of the pipeline (a temporary flow-only GGUF/
topology, built from the same conversion-script helper functions) against a real `ResidualCouplingBlock`
output using a fixed `z_p`, which matched to 7.6e-6 the moment the transpose was removed from the
reference dump. **General lesson reinforced again**: when a computed array is a value-for-value
permutation of the expected one (not literally different numbers), check axis/transpose conventions
before assuming a real numerical bug — this is now the *second* time this exact mistake has appeared in
this VITS effort alone (see the TextEncoder-assembly entry), always in test/reference code, never in the
engine or conversion script itself.

**`tools/convert_piper_vits/reference_forward_vits.py`** (new, consolidates three earlier ad-hoc scripts):
produces every artifact the three reference tests above need (`ref_token_ids.json`, `ref_m_p.npy`,
`ref_logs_p.npy`, `ref_sdp_z_noise.npy`, `ref_sdp_logw.npy`, `ref_z_p.npy`, `ref_wav.npy`) from the real
checkpoint + config JSON in one run, mirroring `tools/convert_nemo/reference_forward_parakeet_tdt.py`'s
role for that model — `strict=True` state-dict loading confirmed **zero** missing/unexpected keys for
`enc_p`/`dp`/`flow`/`dec`, independently reconfirming the conversion script's own tensor-name mapping is
exactly right.

Full suite: **54/54 passing**, all five new VITS tests (`test_e2e_vits_smoke`, `test_e2e_vits_driver`,
`test_e2e_vits_stats_reference`, `test_e2e_vits_logw_reference`, `test_e2e_vits_flow_vocoder_reference`)
skip cleanly without their respective env vars set.

**What "done" means here and what's still genuinely open**: every phase's real MATH is now proven correct
against the real checkpoint to float32 precision. What remains, if ever needed: (1) a true full-pipeline
bit-exact check would require also pinning `VitsDriver`'s own `z_p`-sampling RNG to match a reference
exactly (not attempted — the three phase-level checks above already prove each phase independently, and
`VitsDriver`'s own sampling code is a straightforward `std::normal_distribution` call, not intricate model
logic); (2) multi-sentence input (piper splits on espeak's own clause/sentence boundaries and runs
inference once per sentence -- `VitsDriver::synthesize` currently takes one already-tokenized sequence,
so a caller wanting full-text-in/audio-out needs to do that splitting itself, matching `voice.py`'s own
`synthesize_stream_raw` structure); (3) writing a WAV file / audio playback integration (out of scope for
this engine, a caller's own concern).

### TODO: vendor piper-phonemize's `phonemize.cpp` (+ deps) into loom-engine, so text→phoneme→id doesn't
### depend on the external `piper_phonemize` Python package

Right now the ONLY thing standing between "raw text in" and `VitsDriver::synthesize()` is phonemization,
and that step lives entirely OUTSIDE this project: `reference_forward_vits.py` (and
`test_e2e_vits_driver`'s real-text test input) both depend on importing the `piper_phonemize` Python
package from the piper venv (`/home/flavio/.venvs/piper`), which itself wraps a **custom espeak-ng fork**
in a compiled `.so` — none of that is part of loom-engine, and nothing in this repo can phonemize text on
its own. This is a real, currently-unaddressed gap for anyone wanting to go from a plain string to audio
using only loom-engine.

**What needs vendoring in**, from `github.com/rhasspy/piper-phonemize` (confirmed via direct inspection
this session, see the `VitsDriver`/reference-script BACKLOG entries above for the full research trail):
- `src/phonemize.cpp` / the matching header (`phonemize.hpp` or equivalent) — `phonemize_eSpeak`, the
  real per-clause espeak-ng driving loop (`espeak_SetVoiceByName`, `espeak_TextToPhonemesWithTerminator`,
  NFD normalization via `una::norm::to_nfd_utf8`, clause-terminator-driven punctuation injection,
  `(lang)`-flag stripping) — this is the piece that actually needs porting/vendoring, not just calling.
- `src/phoneme_ids.hpp` (or the ID-map/interleaving logic) — though note `piper`'s own Python runtime
  (`voice.py::phonemes_to_ids`) does NOT call piper-phonemize's C++ version of this; it reimplements the
  same BOS/blank-interleave/EOS algorithm in pure Python (already ported to
  `reference_forward_vits.py::phonemes_to_ids` this session) — worth deciding which one to standardize on
  when vendoring rather than carrying two.
- The `python.cpp` pybind11 wrapper is Python-binding glue, NOT needed for a native C++ integration —
  skip it; the goal is calling `phonemize_eSpeak` directly from loom-engine's own C++, no Python layer.
- **The espeak-ng dependency itself**: NOT stock `espeak-ng` (its CLI doesn't expose clause-terminator
  info) — needs the same fork piper-phonemize builds against, `github.com/rhasspy/espeak-ng` (pinned in
  piper-phonemize's own `CMakeLists.txt` to commit `0f65aa301e0d6bae5e172cc74197d32a6182200f`, "Add
  TranslateClauseWithTerminator to translate.h"), plus its `espeak-ng-data` linguistic data directory
  (bundled in the installed `piper_phonemize` wheel at
  `~/.venvs/piper/lib/python3.11/site-packages/piper_phonemize/espeak-ng-data/` — needs a real path to
  ship/reference once vendored, not hardcoded to that venv location).
- Skip `tashkeel_run`/`libtashkeel` (Arabic-only diacritization, irrelevant to `en-gb-x-rp` and every
  other voice this project has touched so far) unless a future voice actually needs it.

**Why this matters**: without it, "loom-engine can run VITS TTS" is only true if the CALLER supplies
already-phonemized token ids (as `test_e2e_vits_driver`/`reference_forward_vits.py` currently do, via the
Python `piper_phonemize` package) — a real, meaningful gap between "the model runs" (done, verified) and
"you can type a sentence and get audio using only this project" (not yet true).

**DEFERRED — licensing conflict, do not vendor espeak-ng into this repo.** loom-engine is licensed
MIT (repo's own `LICENSE` file) and is meant to stay permissive (MIT/Apache-2.0 class); espeak-ng
(stock AND the piper-phonemize fork) is **GPL-3**. Vendoring `phonemize.cpp` + the espeak-ng fork's
source directly into this repo, as originally scoped above, would pull GPL-3-licensed code into a
permissively-licensed project — not acceptable as-is (linking against a separately-distributed GPL-3
`.so`/binary at the user's own discretion is a different, less entangling story than vendoring source,
but that's a decision for whoever picks this up, not assumed here). Do not implement the
vendoring-into-this-repo plan above without resolving the licensing question first.

Current plan: skip espeak-ng-based phonemization entirely and instead integrate a differently-licensed
phonemizer — candidate: **phoonnx** (a friend-of-the-user's project) — once its license and API are
confirmed compatible. Re-scope this task around whatever that integration turns out to need (likely
still a `src/text/phonemize.cpp` + `include/loom/text/phonemize.h` split, matching this project's
existing `src/core/`+`include/loom/core/` convention, but the build-system/vendoring shape depends
entirely on how phoonnx is packaged and licensed — not yet investigated).

**Linking vs. subprocess note** (in case espeak-ng ever comes back as a fallback): GPL-3 has no
LGPL-style linking exception, and there's no applicable "system library" carve-out for a bundled/fetched
dependency, so statically OR dynamically linking espeak-ng into loom-engine's own distributed binary
would make that binary a GPL-3 combined work — not compatible with staying permissive. Driving a
separately-installed `espeak-ng` as an out-of-process subprocess/CLI (or an optional, clearly-labeled
GPL build variant, e.g. a `-DLOOM_WITH_ESPEAK=ON` flag) would NOT entangle the core project's license.
Not a lawyer's opinion — re-verify before actually shipping anything that touches espeak-ng.

---

### TODO: add new model families to force out new/missing primitives

Every model added so far (toy LLM/ASR/vision/TTS, real Conformer-CTC, real Qwen3-0.6B-Base, real VITS)
has earned its keep by exposing at least one genuine engine gap before being fully wired up — that's the
project's core validation method (real checkpoint + hand-computed primitive verification, not "looks
plausible"). Candidates identified as worth tackling next, roughly for the NEW primitive/architecture
gap each is expected to expose (not yet confirmed by reading each one's real source the way VITS's
source was read in full before planning — that reading is the first step whenever one of these is
picked up, same discipline as every previous model):

- **Whisper (v3)** — encoder-decoder with cross-attention; likely close to what Qwen3/Conformer already
  cover (`ATTENTION`, layer norm, conv frontend) but cross-attention (separate K/V source from Q's
  sequence) and the sinusoidal (non-learned, non-RoPE) positional embedding may be genuinely new wiring,
  not just a new checkpoint on existing primitives — needs confirming against real source before assuming.
- **FastConformer RNN-T** — this project already has real Conformer-CTC (NeMo) and a `TdtDecoder`
  (transducer decoding) precedent from earlier NeMo work; FastConformer's depthwise-separable
  subsampling and the RNN-T joint network (as opposed to TDT's duration-augmented joint) are the likely
  new surface — may turn out to be mostly composition of existing primitives plus a new joint-network
  topology, needs checking against NeMo's real FastConformer-RNNT source, not assumed similar to TDT.
- **Kokoro TTS** — StyleTTS2-family architecture (see below); check whether Kokoro's specific decoder
  (reportedly ISTFT-based rather than HiFi-GAN) needs a new vocoder primitive beyond what VITS's
  HiFi-GAN `Generator` already covers.
- **StyleTTS2** — style-diffusion-based TTS (adaptive instance norm conditioning, a diffusion sampler for
  the style vector) — likely the first genuinely diffusion-flavored model in this project (iterative
  denoising sampling loop, not a single reverse-flow pass like VITS's SDP/coupling flow), which may need
  new host-driver looping patterns (`VitsDriver`'s two-phase pattern may generalize, or may not).
- **SupertonicTTS** — unfamiliar to this project; needs a from-scratch read of its real source/paper
  before any primitive-gap claim can be made.
- **F5-TTS** — flow-matching-based (ODE-solver sampling, conceptually related to the `OdeStepper` host
  driver already built for an earlier milestone) — likely close to existing patterns but needs
  confirming its specific conditioning mechanism (audio+text in-context conditioning) against real source.
- **Matcha-TTS** — another flow-matching TTS (also `OdeStepper`-adjacent); likely shares more with
  existing patterns than the others in this list, worth checking whether it's redundant with F5-TTS's
  gap coverage before investing in both.

**How to apply**: pick ONE at a time, read its real source in full before planning (same discipline as
every prior model in this project — VITS's plan explicitly started from reading
`piper_train/vits/{models,modules,attentions,transforms,commons}.py` end-to-end before scoping anything),
confirm which primitives are genuinely new vs. already covered, and follow the same
convert-script + hand-rolled-Python-reference + hand-computed-primitive-unit-test + numerical e2e-test
discipline established for Conformer-CTC/Qwen3/VITS. Don't assume gaps from the list above without
re-confirming against that model's actual real source and a real checkpoint.

---

### TODO: evaluate generalizing `aten_to_loom.py` into a real exporting tool

`tools/convert_generic/aten_to_loom.py` (currently a proof-of-concept, per its own module docstring)
walks a `torch.export()` ATen graph node-by-node through a small fixed `OP_MAP` table, with NO
pattern-matching/subgraph fusion, NO dynamic-shape support, and only a single hardcoded
qualname→GGUF-key rule (`_qualname_to_gguf_name`, ToyLLM/Conformer-specific: `layers.N` → `blk.N`, bare
`nn.Parameter`s get `.weight` appended). Every model converted so far in this project
(`convert_qwen3`, `convert_nemo`, `convert_piper_vits`) instead uses its own bespoke, hand-written
conversion script reading the real checkpoint's state dict directly — NOT this generic exporter.

**What "general" would need to mean, concretely** (not yet scoped in detail):
- A pluggable weight-naming strategy per model family, replacing the single hardcoded
  `_qualname_to_gguf_name` rule — likely a small per-family config/callback rather than one fixed
  function, since each model so far has needed a different (if small) renaming rule.
- Handling models whose real `torch.export()` graph is "broken" for this purpose — e.g. custom ops that
  don't survive export cleanly, in-place mutation, data-dependent control flow (already known to be a
  hard blocker for anything using boolean-mask indexing or per-sample branching) — needs a concrete list
  of which real models actually break and how, not a hypothetical list.
- Deciding whether fusion/pattern-matching (e.g. recognizing `scaled_dot_product_attention`'s decomposed
  ATen ops and re-fusing into loom's single `ATTENTION` primitive, rather than requiring the source
  `nn.Module` to call a custom op as today) is in scope — today's design deliberately punts this onto the
  source model (see the module docstring: "the source nn.Module is expected to have called a custom op").
  Generalizing away from that requirement is a materially bigger undertaking than the rest of this list.
- Whether this is worth doing generally at all, vs. continuing to accept one bespoke script per model
  family (the current, working approach) — this needs an honest cost/benefit pass once 2-3 more real
  models have been converted by hand (see the model-families TODO above), so the generalization is
  informed by real, repeated pain points rather than speculative design.

**How to apply**: don't start broad rearchitecture of `aten_to_loom.py` speculatively. Revisit once a
few more real conversion scripts exist (from the model-families TODO above) and look for what actually
repeated across them vs. what was genuinely bespoke each time — generalize only the parts that
demonstrably repeated.

---

### TODO: make quantization a general process/tool (including KV-cache quantization)

`tools/quantize/quantize_gguf_q8_0.py` currently exists as a single fixed-scheme (Q8_0) script,
presumably written for one specific model's GGUF output rather than as a general, model-agnostic
quantization pass. Not yet re-read in full this session to confirm its exact current scope/limitations —
that's the first step before scoping this properly, same as every other TODO here.

**What "general" likely needs to cover** (provisional, needs confirming against the real script and
against ggml's actual supported quant types before treating as fixed scope):
- Multiple quantization schemes beyond Q8_0 (ggml supports a family of k-quants/legacy quants — which
  ones are worth exposing needs checking against real ggml support, not assumed).
- Per-tensor-role policy (e.g. skip quantizing norm weights/biases, embedding tables, or other
  known-sensitive tensors — a common convention in llama.cpp-family quantizers) rather than a uniform
  blanket quantization of every tensor in the GGUF.
- **KV-cache quantization** specifically — this is a runtime/inference-time concern (quantizing the
  attention KV cache during generation, not the static weights at conversion time), which is a
  genuinely different mechanism from weight quantization and likely needs its own engine-side support
  (checking how the KV cache is currently allocated/typed in this engine's attention primitives is the
  first step, not assumed to be a trivial extension of the weight-quantization script).
- A single general entry point/tool (as opposed to a per-model bespoke invocation) that can be pointed at
  any of this project's GGUF outputs.

**How to apply**: re-read `quantize_gguf_q8_0.py` in full first to establish what actually exists today
before assuming any of the above is missing. Treat weight quantization and KV-cache quantization as two
separate sub-efforts (different mechanism, different point in the inference pipeline) rather than one
undifferentiated "quantization" task.

---

### TODO: add primitives for the range of attention variants used in modern models (incl. flash attention)

This project currently has three attention primitives (`ATTENTION`, `REL_POS_ATTENTION` for
Transformer-XL-style Conformer relative position, `REL_POS_ATTENTION_SHAW` for VITS's Shaw et al.
lookup-table relative position — confirmed via `src/ops/primitives_attention.cpp`), each added because a
real model genuinely needed it, not spec'd out ahead of need. The model-families TODO above will likely
force out more variants naturally (Whisper's cross-attention being the most immediately likely), but
worth tracking as its own cross-cutting concern since attention-kernel choice affects performance, not
just correctness:

- **Flash attention** — llama.cpp/ggml has a fused `ggml_flash_attn_ext` kernel (memory-efficient,
  fused softmax(QK^T/sqrt(d))V without materializing the full attention matrix) — this project should
  benefit directly from whatever ggml itself already provides as a primitive/kernel, rather than
  reimplementing the fusion independently. First step when this is picked up: check the exact vendored
  ggml version's real `ggml_flash_attn_ext` signature/constraints (head-dim restrictions, mask-shape
  requirements, causal-flag support, F16-vs-F32 KV-cache requirements are all known real constraints in
  upstream llama.cpp — needs re-confirming against this project's actual vendored ggml, not assumed from
  general knowledge) before wiring a new `ATTENTION`-family primitive around it.
- **General principle for this whole area**: whenever llama.cpp/ggml has already solved an
  attention-kernel-shaped problem as a primitive or specialized kernel (flash attention being the
  concrete example in hand, but likely not the only one), loom-engine should wire a thin primitive
  directly onto ggml's own op rather than reimplementing the fusion from scratch in terms of more basic
  ops — mirrors this project's existing practice of using ggml's native ops wherever one exists (e.g.
  `ggml_clamp`, `ggml_leaky_relu`) instead of hand-composing.
- Cross-attention (Q from one sequence, K/V from another — needed by Whisper's decoder, likely also
  F5-TTS/Matcha-TTS's conditioning) is a plausible additional gap, but whether it needs a genuinely new
  primitive or is just `ATTENTION` called with differently-shaped/sourced inputs needs checking against
  the current `op_attention` implementation before assuming new engine code is required.

**How to apply**: don't speculatively add attention primitives ahead of a real model needing them — this
list exists to flag flash attention specifically as a known, concrete, ggml-native opportunity (re-use,
not reinvention) worth prioritizing early since it's a performance win applicable to every existing
model, not just new ones, once verified compatible with this project's existing `ATTENTION` call sites.

---

### 2026-07-19: Whisper v3 (`tiny.en`) — AudioEncoder converted and numerically verified

First model tackled from the model-families TODO above (user-specified priority order: Whisper v3
first). Read `whisper/model.py`/`audio.py` (the real, installed `openai-whisper` package — `pip install
openai-whisper tiktoken` into the piper venv, plus a real `tiny.en` checkpoint via
`whisper.load_model("tiny.en", download_root=...)`) in full before planning, same discipline as VITS.

**Primitive gap turned out to be far smaller than the backlog entry above guessed, on both counts it
flagged as "likely new":**
- **Cross-attention needs NO new primitive at all.** `ATTENTION` (`primitives_attention.cpp`) already
  takes independent q/k/v tensors with no assumption they share a source — confirmed by reading
  `op_attention` directly, not assumed. Whisper's decoder cross-attention (Q from the decoder's own
  hidden state, K/V from the encoder output `xa`) is just an ordinary `ATTENTION` call with `kv_cache:
  false` and K/V projected from a different input than Q — still unexercised end-to-end (decoder not
  built yet, see below), but the primitive-level claim is settled.
- **The mel frontend's global dynamic-range clamp needs NO new primitive either.** `POOL_1D`
  (`primitives_conv.cpp`) already supports `{"op":"max", ...}`, wrapping `ggml_pool_1d(GGML_OP_POOL_MAX,
  ...)` directly. `log_spec.max()` (a true GLOBAL max over the whole `[n_mels,n_frames]` spectrogram, not
  a per-row reduction) is computed by reshaping to a single length-`n_mels*n_frames` row first, then
  pooling with `k0=s0=` that same length → a single-element output; `max(log_spec, gmax-8)` is then
  `RELU(log_spec - (gmax-8)) + (gmax-8)`, composed from existing ADD/SUB/RELU — no new engine code
  anywhere in the whole mel frontend. (Briefly, mistakenly, added a redundant `MAX_POOL_1D` primitive
  before noticing `POOL_1D` already covered this — reverted before committing.)

**Real, confirmed formula differences from `tools/convert_nemo/mel_common.py`'s existing STFT-via-conv
precedent** (new module `tools/convert_whisper/whisper_common.py`, NOT a shared one — see its own header
comment for the full list): no preemphasis; window is `torch.hann_window(400, periodic=True)`, a
genuinely different formula from NeMo's `periodic=False` convention (denominator `N` vs `N-1` —
confirmed numerically, not a rounding artifact: `hann(8,periodic=True)[1]=0.1464466` vs
`hann(8,periodic=False)[1]=0.1882551`); `torch.stft(center=True)`'s padding is **reflect**, not
zero/constant — `ggml_conv_1d`'s own padding is zero-only, so reflect-padding is done **host-side**
(`whisper_common.pad_reflect`) before the (always-fixed-30s-length) waveform ever enters the graph, not a
new primitive (Whisper's fixed 480000-sample window makes this a fixed-shape host precompute, same
"host computes, feeds in as a declared input" precedent as VITS's noise injection); Whisper drops the
LAST STFT time-frame (`stft[...,:-1]`, handled via a VIEW+CONT slicing off the last position); log10 (not
natural log, `LOG` + `SCALE(1/ln 10)`); no per-feature CMVN. All confirmed by a standalone numpy-vs-real
`whisper.audio.log_mel_spectrogram` check (`verify_whisper_mel.py`, scratch, not committed) before
writing any GGUF/ggml code, matching to 3.7e-6.

**Real, confirmed axis-convention finding**: `CONV_1D`'s data input is `[IL, IC, N]` (T=`ne[0]`, channels
`ne[1]`) — confirmed by re-reading `convert_parakeet_tdt.py`'s own declared `"waveform"` input shape
(`["n_tokens","1","1"]`), not assumed. The mel filterbank's `MUL_MAT` output is channel-first
(`[n_mels,n_frames]`, C=`ne[0]`, matching this project's own attention convention) — the OPPOSITE of
`CONV_1D`'s convention — so a `PERMUTE`+`CONT` is needed crossing INTO `conv1` and crossing back OUT after
`conv2` before the positional-embedding add, same "cross the boundary with `PERMUTE`+`CONT`" pattern
VITS's own header comment documents for its channel-first-attention/T-first-`CONV_1D` boundary. `CONV_1D`
chained directly to another `CONV_1D` (conv1 → conv2) needs NO transpose in between (both consume/produce
the same `[T,C,N]` convention) — a real mistake caught before running anything: an unnecessary
`PERMUTE`+`CONT` was initially inserted between conv1 and conv2, removed once traced through carefully.

**Whisper's encoder is the first model in this project with an entirely FIXED shape** (always exactly a
30s/480000-sample window, `n_audio_ctx=1500` positions) — no `"$n_tokens"`-style dynamic-length symbol
anywhere in this topology at all, unlike every prior model (Conformer/Qwen3/VITS all have genuinely
dynamic per-utterance lengths). Simpler in this one specific respect than everything before it.

**Verification**: `reference_forward_whisper_encoder.py` calls the REAL installed `openai-whisper`
package's own `AudioEncoder` directly (no hand-reimplementation needed, unlike VITS which had to
reconstruct piper's modules from scratch) on synthetic 30s noise. Deterministic end-to-end (no sampling
anywhere in the encoder) — no fixed-noise-injection machinery needed, unlike VITS's SDP/flow.
`test_e2e_whisper_encoder_reference.cpp` (env vars `LOOM_WHISPER_DIR`, `LOOM_WHISPER_ENCODER_REF_DIR`,
same `SKIP_RETURN_CODE 77` pattern as every other real-checkpoint test) initially failed at
`max_abs_diff=0.00515` against a naive `< 1e-3` threshold. Root-caused via a staged scratch-diagnostic
bisection (NOT assumed to be a bug and patched blindly) — built successively larger prefixes of the
topology as their own tiny GGUFs (mel-only, mel+conv1+conv2+pos-emb, +block0, +blocks0-1, +blocks0-2,
+all 4 blocks+ln_post) via the REAL `build_encoder()` function itself (dims`["n_audio_layer"]`
monkeypatched down, not a hand-copied duplicate — confirms the production code path itself, not a
lookalike), each checked against a matching real-model intermediate captured via a `torch`
`register_forward_hook`/`register_forward_pre_hook`. Every stage through all 3 layers matched to
~7e-5 max_abs_diff; only the FULL 4-layer+`ln_post` output showed the 0.005 outlier, and even then only
at 11 of 576000 elements (all <2% relative error, no sign flips) with `mean_abs_diff` still ~2.4e-6 —
ordinary chaotic amplification of upstream ULP-level fp noise through 4 layers of GELU/softmax
nonlinearities, not a wiring bug. Fixed by checking `mean_abs_diff` tightly (`<1e-4`) AND `max_abs_diff`
loosely (`<1e-2`) instead of a single strict bound (which is what every *shallower* reference test in
this project uses safely) — a real, worth-remembering calibration lesson for any future *deep*
(many-layer) non-stochastic reference test: a single tight max-diff bound doesn't generalize to depth.

**Still TODO for Whisper** (tracked under Task #80, not yet started): the `TextDecoder` (causal
self-attention with `KvCache` + cross-attention against the encoder's `xa` + tied output projection,
`x @ token_embedding.weight^T`, reusing `GET_ROWS`'s own embedding weight transposed via `MUL_MAT`, no
new primitive expected there either) and a new host driver (`WhisperDriver`, mirroring `Generator`'s
autoregressive decode-loop pattern but extended to run the encoder once up front and feed `xa` as a
per-step cross-attention input) — `Generator` as it stands assumes a fixed `tokens`/`positions`/`kq_mask`
self-attention-only topology and doesn't carry an auxiliary encoder-output input through the loop.

---

### 2026-07-19: Whisper v3 (`tiny.en`) — TextDecoder + `WhisperDriver` done, full pipeline verified

Completes the Whisper v3 milestone above: `tools/convert_whisper/convert_whisper_decoder.py` (causal
self-attention + cross-attention + tied output projection), `include/loom/core/whisper_driver.h` /
`src/core/whisper_driver.cpp` (`WhisperDriver`, the encoder-once + prefill-then-decode host driver), and
three new tests (`test_e2e_whisper_decoder_reference`, `test_e2e_whisper_driver`).

**Confirmed, no new primitive gap at all for the decoder** (both things the original backlog entry
flagged as "likely new" turned out to already be covered): causal self-attention reuses `ATTENTION`
`kv_cache:true` exactly as Qwen3/toy_llm already do (same `KvCache`, same "tokens"/"positions"/"kq_mask"
convention — "positions" here feeds a `GET_ROWS` lookup into Whisper's LEARNED absolute positional
table, not a RoPE angle, but the input plumbing is identical); cross-attention is `ATTENTION`
`kv_cache:false` with Q from the decoder's own hidden state and K/V projected fresh from a per-step `xa`
input (fixed `n_audio_ctx=1500`, a compile-time literal, not a `$`-symbol) — confirmed exactly as
predicted in the encoder entry above, no new engine code. Tied output projection
(`x @ token_embedding.weight.T`) is a single `MUL_MAT` reusing the SAME weight tensor already registered
for the input `GET_ROWS` embedding lookup (`MUL_MAT(tok_emb[n_state,n_vocab], x[n_state,n_tokens])`
contracts over `n_state`, giving `[n_vocab,n_tokens]` — exactly the real math, no transpose needed since
GGUFWriter's own axis reversal already puts the weight in the right ggml orientation for both uses).

**`WhisperDriver`** (mirrors `VitsDriver`'s two-model/two-topology structure, simpler in one respect:
Whisper's encoder runs exactly ONCE per call with no downstream-shape question at all, unlike VITS's
`generate_path`, whose real duration output determines the vocoder's frame count at runtime). Owns a
`KvCache` sized from `WhisperConfig` (`n_text_layer`/`n_text_state`/`n_text_ctx`) plus two `GraphBuilder`s
(encoder: no `KvCache`; decoder: wired to it) — same reference-member-ordering constraint as every prior
driver (`GraphTopology` members constructed before the `GraphBuilder`s that reference them).
`transcribe()`: one encoder pass → `xa`; then `Generator::generate`'s own prefill-then-decode-one-at-a-
time loop, extended to feed `xa` (unchanged) + an all-zero `xa_mask` alongside `tokens`/`positions`/
`kq_mask` at every step. Cross-attention K/V are recomputed from `xa` on EVERY decode step rather than
cached once (the real PyTorch model's own `install_kv_cache_hooks` optimization) — correct, just not
maximally efficient; left as a documented, not-yet-implemented follow-up in the driver's own header
comment, not attempted here.

**Verification, three levels, each building on the last (same discipline as VITS's staged
TextEncoder→SDP→flow/vocoder verification)**:
1. `test_e2e_whisper_decoder_reference`: ONE-SHOT teacher-forced decoder forward (`n_past=0`,
   `n_tokens=T` covers the whole causal triangle in a single call, exactly matching the real model's own
   `kv_cache=None` path) against 8 arbitrary-but-valid token ids — matched to `mean_abs_diff=9.5e-6`,
   `max_abs_diff=3.3e-5`, and every one of the 8 greedy-argmax positions matched exactly.
2. `test_e2e_whisper_driver`: the FULL `WhisperDriver::transcribe` loop (real `KvCache`-based incremental
   decode, not a one-shot teacher-forced call) against a real, PLAIN greedy-argmax reference loop built
   directly from `model.encoder`/`model.decoder` (`reference_forward_whisper_driver.py` — deliberately
   NOT `model.decode()`, which layers on suppress-tokens/timestamp-constraint/temperature-fallback logic
   this driver doesn't implement) — on synthetic noise audio (the same input the encoder/decoder
   reference scripts already use), the real model itself emits EOT after only 5 generated tokens
   (`[357, 7050, 2491, 8, 50256]`); `WhisperDriver` reproduced this EXACT sequence, token-for-token,
   first try — the strongest possible confirmation available for a greedy/deterministic pipeline this
   deep (encoder → prefill → 4 incremental cross-+self-attention decode steps, all through the real
   `KvCache` growth path, not a single shot).

Both matched essentially on the first real attempt once the encoder's own axis conventions and
`op_attention`'s existing generality were already nailed down by the encoder milestone — a good sign the
"read the real source in full, confirm gaps against the actual primitive code, don't assume" discipline
from earlier in this project pays down real risk before writing any topology JSON, not just after.

**Whisper v3 is now a complete, end-to-end-verified milestone** (mel frontend → encoder → decoder →
greedy autoregressive driver, all against a real checkpoint). Per Task #80's priority order, next up:
FastConformer RNN-T.

---

### 2026-07-19: FastConformer RNN-T (`nvidia/parakeet-rnnt-0.6b`) — done, near-zero new engine code

Second model from Task #80's priority list. Real checkpoint downloaded (`hf_hub_download`,
`nvidia/parakeet-rnnt-0.6b`, ~2.47GB `.nemo`, same family/size as the already-converted
`parakeet-tdt-0.6b-v3` sibling) and its real `model_config.yaml`/state dict inspected directly before
writing anything — confirmed several REAL, checkpoint-specific differences from its TDT sibling rather
than assuming the two share every convention just because they're siblings:
- `feat_in=80` (NOT TDT's 128) — this checkpoint's mel frontend matches Conformer-CTC-small's own
  80-mel convention instead.
- `xscaling=True` (the OPPOSITE of TDT's `xscaling=false`) — needs the `sqrt(d_model)` xscale fold into
  `pre_encode.out`'s weight/bias at conversion time (Conformer-CTC-small's own technique), which TDT's
  script deliberately omits.
- Biased throughout (`use_bias` unset in config but every relevant tensor — `feed_forward{1,2}`'s
  `linear{1,2}`, `self_attn`'s `linear_{q,k,v,out}` (NOT `linear_pos`, which never has a bias in NeMo's
  `RelPositionMultiHeadAttention` regardless — confirmed absent from the real state dict either way),
  the conv module's `pointwise_conv{1,2}`/`depthwise_conv` — confirmed present in the real state dict),
  unlike TDT's confirmed-unbiased convention. So this checkpoint is a genuine hybrid: TDT's `dw_striding`
  (3-stage depthwise+pointwise) subsampling structure + Conformer-CTC-small's bias/xscale conventions —
  a real finding, not assumed from either existing sibling script alone.
- The joint has **NO duration head at all**: `config["joint"]["jointnet"]` has no `"durations"` key
  whatsoever (TDT's does), and the real `joint.joint_net.2.weight` shape is `(1025, 640)` ==
  `num_classes(1024)+1(blank)` exactly, no extra columns.
- Decoder (`RNNTDecoder`, 2-layer LSTM, `pred_hidden=640`) and the joint's own tensor names are
  IDENTICAL to TDT's (`decoder.prediction.*`, `joint.enc.*`, `joint.pred.*`, `joint.joint_net.2.*`) —
  confirmed real, so `build_lstm_topology`/`build_joint_topology` are imported and reused **verbatim**
  from `convert_parakeet_tdt.py`, not reimplemented (the topology JSON never encodes the joint's output
  width at all — that comes from the GGUF tensor's own shape).

**The only genuinely new engine-level work was generalizing `TdtDecoder`** to support plain RNN-T
(`TdtDecoderConfig.durations` left EMPTY): no duration head/argmax at all, every blank advances exactly
one frame (standard Graves-2012/NeMo `RNNTGreedyDecoder` control flow), non-blank never advances. The
class was already named/doc-commented generically ("Transducer/TDT models") with plain RNN-T in mind
conceptually, just never implemented — a small, targeted change (guard the duration-argmax computation
behind `n_durations > 0`, default `skip=1` for blank / `skip=0` for non-blank when `n_durations==0`)
rather than a new class, avoiding ~150 lines of near-duplicate decode-loop code. A REAL bug caught while
writing this: naively falling through to the existing duration-argmax code path with `n_durations==0`
would read one element past `combined`'s own end and index an empty `cfg_.durations` vector (both UB) —
guarded explicitly instead of relying on the existing `n_combined <= n_durations` check (which is
vacuously satisfied whenever `n_combined >= 1`, i.e. always, so it silently doesn't catch this case).

**Verification, same staged discipline as every other model in this project**: first a NEW synthetic
fixture (`tools/fixture_gen/rnnt_step_common.py`/`make_rnnt_step_gguf.py`/`reference_rnnt_step.py`,
mirroring `tdt_step_common.py`'s own role, deliberately duplicated rather than shared) with a
hand-picked seed (searched over several thousand candidates) producing three genuinely different
per-frame cases — one single emission, one two-emission-then-blank frame (exercising "stay on this
frame" more than once), one immediate blank — verified via `test_rnnt_decoder.cpp` BEFORE touching the
real 2.47GB checkpoint, exactly the "verify the generalization in isolation first" discipline this
project has followed for every primitive change. Then the real checkpoint: `convert_parakeet_rnnt.py` +
`reference_forward_parakeet_rnnt.py` (hand-rolled plain-PyTorch reference, `nemo_toolkit`/`transformers`
still broken in this venv per the TDT work's own finding) + `test_e2e_parakeet_rnnt.cpp` — encoder
matched to `2e-6` max abs diff (tighter than TDT's own `5e-2` tolerance, though not a fair
apples-to-apples comparison — shorter/different input), and the decode loop matched exactly
(`tokens=[]`, `frame_indices=[]` both sides — this checkpoint decodes synthetic noise to all-blank,
confirmed via the reference script itself, not a bug; still a fully meaningful end-to-end check since
`TdtDecoder`'s new branch has to reproduce whatever the real model actually does, blank or not).

**Environment note, unrelated to the model itself but worth recording**: mid-conversion, writing the
~2.3GB all-F32 encoder GGUF into this session's `/tmp`-backed scratchpad (on the 28GB root partition,
already at ~91% from OS/package baseline) drove the ROOT filesystem to 100% full, breaking the harness's
own command-output capture (not just this task's disk use) until cleaned up. Moved all large
conversion-output scratch directories to `/home` (429GB, plenty of headroom) instead — the existing
"use /home not /tmp for large downloads" guidance turns out to apply equally to large **generated**
files (GGUF conversion output can be comparable in size to the source checkpoint once stored as
uncompressed F32), not just downloads.

Per Task #80's priority order, next up: Kokoro TTS.

---

### 2026-07-19: Kokoro TTS — real source read in full, primitive-level groundwork verified (in progress)

Third model from Task #80's priority list. Read the real `kokoro` PyPI package (`model.py`, `modules.py`,
`istftnet.py`) in full before planning, plus confirmed licensing first (per the earlier VITS/espeak-ng
licensing lesson): `hexgrad/Kokoro-82M` model, the `kokoro` pip package, and `misaki` (its G2P library)
are all **Apache-2.0** — no GPL concerns like VITS's espeak-ng, confirmed via `HfApi.model_info` and
PyPI metadata directly, not assumed. Real checkpoint (`kokoro-v1_0.pth`, small, ~82M params) and config
downloaded to `/home/flavio/.claude/tmp/kokoro_model/`.

**Real architecture is genuinely larger/more novel than anything tackled so far** — StyleTTS2-family:
`CustomAlbert` (a real HF `AlbertModel` — PL-BERT phoneme-conditioning transformer) → `ProsodyPredictor`
(a `DurationEncoder` of stacked BiLSTM+`AdaLayerNorm` blocks, a duration head via **sigmoid-sum
regression** over `max_dur` buckets rather than VITS's `exp/ceil` approach, then F0/energy prediction
via `AdainResBlk1d` stacks) + a separate CNN+BiLSTM `TextEncoder` → `Decoder` (`istftnet.py`'s
`Generator`, a HiFi-GAN-NSF hybrid: a harmonic-plus-noise source module driven by the predicted F0
curve, `AdaINResBlock1`'s "Snake1D" activation, and an **ISTFT-based** final reconstruction instead of
HiFi-GAN's pure `ConvTranspose1d` upsampling — this is the "ISTFTNet" the package name references).
AdaIN (`AdaIN1d`/`AdaLayerNorm`) conditioning is threaded pervasively through both the prosody predictor
and the vocoder, not a one-off.

User confirmed via `AskUserQuestion`: attempt the full continuous build (same choice as VITS), not a
scoped-down vocoder-only pass or deferring to a different model.

**Primitive-level groundwork done so far, each verified before relying on it (same discipline as every
primitive added this project)**:
- **`SIN`/`COS`** (`primitives_basic.cpp`) — thin wrappers around `ggml_sin`/`ggml_cos`, which are
  already native in the vendored ggml (confirmed by reading `ggml.h` directly, not assumed missing).
  Needed for the NSF source's sine generation and the vocoder's Snake activation
  (`x + sin(a*x)^2/a`, itself just a composition of `SIN`/`SQR`/`MUL`/`ADD` — no dedicated Snake
  primitive needed at all). Verified against `std::sin`/`std::cos` directly.
- **`INTERPOLATE_1D`** (`primitives_basic.cpp`) — wraps `ggml_interpolate` directly (native in ggml,
  confirmed via `ggml.h`), for the several 1D up/downsample spots (`SineGen`'s phase pre/post-filtering,
  `UpSample1d`'s 2x nearest upsample, the F0 curve's huge nearest upsample to full waveform rate). A
  real, non-obvious finding confirmed by reading `ggml`'s own `ggml_compute_forward_interpolate` C++
  source directly (not assumed from the header alone): ggml has no dedicated "1D linear" mode, but its
  `GGML_SCALE_MODE_BILINEAR` mode, called with the target's `ne[1]` held EQUAL to the input's own `ne[1]`
  (i.e. not actually resizing that axis), makes that axis's blend factor `dy` always exactly 0 — so the
  2D bilinear formula degenerates to an exact 1D linear interpolation along `ne[0]` alone, using the same
  half-pixel-center convention as PyTorch's own `F.interpolate(mode='linear', align_corners=False)`.
  Verified numerically against real `torch.nn.functional.interpolate` for both `linear` (up ×2, down
  ×0.5) and `nearest` (up ×2) on the same small example, matching to 1e-5 (`test_interpolate_1d`).
- **`AdaIN1d`/`AdaLayerNorm` need NO new primitive at all.** Confirmed via the real checkpoint's own
  state dict: `nn.InstanceNorm1d(affine=True)` is used, but ONLY the style-conditioning `fc.weight`/
  `fc.bias` tensors are ever saved — `InstanceNorm1d`'s own internal affine weight/bias are never
  trained/persisted, so at inference they sit at PyTorch's default init (`weight=1, bias=0`), i.e. a
  mathematical no-op (confirmed both by grepping the real state dict for any `.norm1.`/`.norm2.` key
  other than `.fc.*`, finding none, AND by numerically comparing real
  `nn.InstanceNorm1d(affine=True)` on random untrained weights against a plain per-channel-over-time
  mean/var normalize — matched to 1.2e-7). So `AdaIN1d`'s real computation is just: plain per-channel
  instance-norm, then `(1+gamma)*normed + beta` from the style projection. And the per-channel-over-time
  reduction is EXACTLY what `LAYER_NORM`'s existing `ggml_norm` (reduces over `ne[0]`) already computes,
  provided the tensor is in this project's `[T,C]` (`CONV_1D`-native) layout at that point rather than
  the `[C,T]` channel-first attention convention — no transpose-then-normalize-then-transpose-back
  needed, unlike `AdaLayerNorm`'s own PyTorch implementation (which transposes explicitly because
  `F.layer_norm` there normalizes over the LAST/channel dim, not time). A real, worth-remembering
  reuse: the SAME `LAYER_NORM` primitive, called on a differently-conventioned tensor, computes a
  completely different normalization (instance vs. layer) — the primitive itself doesn't know or care,
  it's purely an axis-convention fact about what's fed in.
- **Inverse STFT needs NO new primitive at all either** — the single biggest open risk in this whole
  model, resolved by direct derivation + numerical verification (scratch script, not committed) against
  real `torch.istft` before writing anything: the standard real-IFFT-via-Hermitian-symmetry formula
  reduces to a per-output-sample weighted cos/sin sum over the `n_freq` rfft bins, which (folding the
  window and the `1/n_fft`-and-doubling normalization factor into the basis itself, exactly mirroring
  how the FORWARD STFT's DFT basis is baked into `CONV_1D`'s kernel elsewhere in this project) becomes a
  plain `CONV_TRANSPOSE_1D` call: kernel `ne=[n_fft, 1, 2*n_freq]` (the windowed cos/sin bases,
  concatenated along the input-channel axis), data `ne=[n_frames, 2*n_freq]` (the predicted
  `mag*cos(phase)`/`mag*sin(phase)` parts, transposed into `CONV_TRANSPOSE_1D`'s own `[T,C]` convention),
  stride=hop — this IS overlap-add, natively, no manual loop. Verified in three stages: (1) a manual
  Python overlap-add loop against real `torch.istft` on a small example, matching to 3.4e-8; (2) the
  planned `CONV_TRANSPOSE_1D` reformulation against that SAME manual loop, matching to 1.1e-16 (machine
  epsilon) — confirms the reformulation is exact, not approximate. Still need: the window-squared-
  overlap-add normalization denominator is a function of the (genuinely dynamic, per-utterance) frame
  count, so it must be computed IN-GRAPH via the same `CONV_TRANSPOSE_1D` mechanism (a fixed `window^2`
  "kernel" against a dynamically-shaped all-ones "data" tensor) rather than baked as a Python constant —
  not yet wired into any topology, tracked as the next concrete step.

**Still TODO, not yet started**: the NSF harmonic-plus-noise source module (`SineGen`/
`SourceModuleHnNSF` — per-sample F0-driven sine generation with harmonics, voiced/unvoiced gating via
`STEP`, phase accumulation via the already-existing `CUMSUM`, host-vs-in-graph decision for the
`torch.randn` noise injection not yet made); the ALBERT text-conditioning encoder's specific wiring
(embedding factorization, cross-layer parameter sharing — real config not yet inspected in detail);
`DurationEncoder`/`ProsodyPredictor`/`TextEncoder`/`Decoder` assembly; the conversion script, hand-rolled
reference, and e2e tests. This is a multi-step remaining effort, continuing in the same session.

**Update — the NSF harmonic-source module (`SineGen`) ALSO needs no new primitive.** Verified by
inlining `istftnet.py`'s real `SineGen` class directly (importing the actual `kokoro` package hits the
same broken transformers/huggingface-hub version pin already noted for NeMo's own toolkit — `SineGen`
itself has no such dependency, so its source, copied verbatim, was run directly rather than fighting the
import chain) against a from-scratch numpy re-derivation, with the two genuinely random components
(`rand_ini`'s per-harmonic initial phase offset, the additive Gaussian noise) pinned to fixed
`torch.rand`/`torch.randn_like` monkeypatches — same "host computes/fixes the noise, feeds it in"
precedent as VITS. Matched to 2.9e-7 across a 240-sample signal spanning both voiced and unvoiced
regions (not a vacuous all-zero check). The whole module — F0→harmonic radians, the
downsample→cumsum→upsample phase-smoothing trick (now expressible via the just-added
`INTERPOLATE_1D` + the pre-existing `CUMSUM`), `sin`, and the voiced/unvoiced gate (a plain `f0 >
threshold` comparison, using the existing `STEP`-adjacent composition already established elsewhere) —
reduces entirely to primitives that now already exist in this engine.

**Summary: every genuinely novel DSP piece in Kokoro's vocoder is now de-risked** — `SIN`/`COS` and
`INTERPOLATE_1D` added and verified against real PyTorch; `AdaIN1d`/instance-norm, inverse STFT, and the
NSF harmonic source all confirmed to need ZERO further new engine primitives, only correct wiring. What
remains is now primarily volume (assembling the full topology across `CustomAlbert`/`ProsodyPredictor`/
`TextEncoder`/`Decoder`, writing the conversion script, hand-rolled reference, and e2e tests) rather than
open technical risk — a materially different, lower-risk position than where this entry started.

**Update — `CustomAlbert` (PL-BERT) architecture confirmed against the real checkpoint + HF source
directly** (reading `transformers/models/albert/configuration_albert.py`/`modeling_albert.py` source
files directly off disk works even though actually IMPORTING `transformers` is blocked by the same
broken huggingface-hub version pin already noted for NeMo — confirmed AlbertConfig's real defaults this
way rather than assuming):
- Embedding factorization: 128-dim word/position/token-type embeddings + their own `LayerNorm`, then
  `embedding_hidden_mapping_in` (a plain Linear) projects up to `hidden_size=768` before the transformer
  layers run.
- **Cross-layer parameter sharing confirmed real** (`num_hidden_groups=1`, HF's own default, not
  overridden in this checkpoint's config): the real state dict has exactly ONE
  `encoder.albert_layer_groups.0.albert_layers.0.*`, applied 12 times (`num_hidden_layers=12`) — this
  project's existing `repeat_for` topology mechanism already supports referencing a weight name that does
  NOT contain `{i}` inside a repeated block (it simply stays constant across iterations, no special
  handling needed), so this needs no new topology-schema feature, just using that mechanism as-is.
- **Activation is `"gelu_new"` (HF's default `hidden_act`), NOT the erf-based GELU this project's
  existing `GELU` primitive computes** — a real, easy-to-miss mismatch (both are "GELU" by name, but
  materially different functions). Worse, ggml's plain `ggml_gelu` (the tanh-approximation one, which
  WOULD algebraically match "gelu_new") unconditionally routes through an F16 lookup table on CPU
  (`GGML_GELU_FP16` is hardcoded `#define`'d in `vec.h`, confirmed by reading it directly) — the same
  kind of forced-F16 precision loss this project already worked around for `CONV_1D`'s im2col and chose
  `GELU_ERF` over plain `GELU` to avoid for Whisper. Confirmed "gelu_new"'s formula
  (`0.5x(1+tanh(sqrt(2/pi)(x+0.044715x³)))`) is algebraically IDENTICAL to `ggml_gelu_f32`'s own C
  formula (0.0 diff on a hand-checked vector) — so the fix is composing it directly from existing exact-
  F32 primitives (`TANH`/`SQR`/`MUL`/`ADD`/`SCALE`) in the topology JSON, same as Snake1D, rather than
  reusing either existing `GELU`-family primitive.
- `layer_norm_eps=1e-12` (HF's ALBERT default) — NOT this project's usual `1e-5`, needs to be threaded
  through explicitly, not assumed.

**Update — `TextEncoder`'s and the prosody predictor's LSTMs are BIDIRECTIONAL** (confirmed via the real
state dict: `text_encoder.lstm.weight_hh_l0` AND `weight_hh_l0_reverse` both present, likewise
throughout `ProsodyPredictor`/`DurationEncoder`) — genuinely new relative to every LSTM in this project
so far (Parakeet-TDT/RNNT's prediction-network LSTMs are plain unidirectional, autoregressive). These
sequences are short, FULLY KNOWN-LENGTH, non-autoregressive phoneme sequences (unlike TDT's
token-by-token feedback loop) — considered unrolling forward+backward entirely IN-GRAPH via `repeat_for`
(cheaper, no host round-trip per step), but that needs a growing per-timestep OUTPUT collected across
iterations into one sequence tensor (not just a single running accumulator like Conformer's own "cur",
which `repeat_for` already handles fine) — this engine has no scatter-into-a-preallocated-tensor-at-a-
dynamic-offset primitive yet, so an in-graph per-timestep-output loop isn't a clean fit today. Decision:
reuse `TdtDecoder`'s ALREADY-PROVEN pattern instead — a small single-step LSTM-cell topology, stepped
host-side in C++ (`GraphBuilder::build()` once per timestep, h/c carried in a plain `std::vector<float>`
between calls, exactly like `TdtDecoder`'s own prediction-network stepping, just without the
autoregressive joint-network feedback) — run once forward, once backward (feeding the sequence in
reverse order through the SAME per-step topology but the `_reverse`-suffixed weights), concatenate the
two directions' outputs host-side. Lower risk than inventing new in-graph scatter machinery for this
milestone; a genuine architectural choice, not a default, and reusing proven code rather than new engine
surface.

Next concrete step: write the conversion script (`tools/convert_kokoro/`), starting with `CustomAlbert`
(the deterministic, no-conditioning-dependency piece) verified in isolation before assembling the
prosody predictor and decoder around it.

---

### 2026-07-19: Kokoro TTS — `CustomAlbert` converted and numerically verified (first assembled piece)

`tools/convert_kokoro/convert_kokoro_albert.py` (topology) + `reference_forward_kokoro_albert.py`
(hand-rolled pure-PyTorch reference, `transformers` unimportable in this venv per the already-documented
broken huggingface-hub pin, but its `.py` source files are readable directly off disk and were used to
confirm every formula) + `test_e2e_kokoro_albert.cpp`. Matched the real checkpoint's weights to
`max_abs_diff=5.7e-6` across 12 shared-weight transformer layers — first real assembled/verified piece
of Kokoro.

**A real, generalizable bug caught while wiring the cross-layer weight sharing**: this project's
`repeat_for` topology construct (used by Conformer-CTC/Parakeet-TDT/RNNT's own per-layer loops, always
via a bare node-closure that takes EXPLICIT literal output names) has NO per-iteration symbol-table
scoping at all — confirmed by reading `graph_builder.cpp` directly: a single flat `symtab` is simply
overwritten each iteration, so any tensor that must carry state across the loop boundary (ALBERT's
"cur", the same physical layer reapplied 12 times) needs the EXACT SAME literal output name emitted both
before the loop and at the end of each iteration. `convert_kokoro_albert.py`'s own `TopologyBuilder`
(adapted from `convert_whisper_encoder.py`'s) auto-freshens every output name via a monotonic counter
(`f"{hint}_{counter}"`) — great for avoiding accidental collisions in a linear, non-looped script, but
it silently produces a DIFFERENT string each call, so the two "cur"-producing call sites never actually
matched, crashing with a real, confusing-if-you-don't-know-the-cause error: `GraphBuilder: node 'MUL_MAT'
references unresolved input 'cur'`. Fixed by adding an optional `name=` parameter to `TopologyBuilder
.node()` (and threading it through `apply_layer_norm`) that emits a literal, non-freshened output name
when given — needed only at `repeat_for` loop-carry boundaries, ordinary intra-iteration temporaries stay
auto-freshened (safe, since the SAME literal name is naturally reused every iteration by the C++ side
regardless, and that's fine for anything not read across the boundary). Worth remembering for
`DurationEncoder`/`ProsodyPredictor`/`Decoder`'s own repeat_for use, if any turns out to need it — this
`TopologyBuilder` convention (auto-freshened intermediates) is genuinely different from every prior
`repeat_for`-using conversion script in this project, which never had auto-freshening at all.

Not yet wired to `bert_encoder`'s downstream `Linear(768,512)` or anything else in `KModel` — that's the
next integration point once `DurationEncoder`/`ProsodyPredictor` need this output.

---

### 2026-07-19: Kokoro TTS — `TextEncoder` converted and numerically verified; new `BiLstmStepper`

Second assembled Kokoro piece: `tools/convert_kokoro/convert_kokoro_text_encoder.py` (a genuinely
separate, simpler, style-independent module from `CustomAlbert` — embedding -> 3x [weight-normed
Conv1d -> LayerNorm -> LeakyReLU(0.2)] -> bidirectional LSTM) + `reference_forward_kokoro_text_encoder
.py` + `test_e2e_kokoro_text_encoder.cpp`. Matched to `max_abs_diff=1.8e-6`.

**New reusable host driver: `loom::BiLstmStepper`** (`include/loom/core/bilstm_stepper.h` / `src/core/
bilstm_stepper.cpp`) — implements the bidirectional-LSTM-over-a-known-length-sequence host-stepping
design decided in this milestone's earlier entry (reusing `TdtDecoder`'s proven per-step-topology
pattern, just without the autoregressive feedback, run once forward and once backward, concatenating
per-position). Takes ONE shared `GgufModel&` for all four per-direction/per-gate topologies — this
requires the conversion script to write the FULL weight set (both directions) into every one of the 4
small GGUFs it produces, matching `TdtDecoder`'s own established "every small GGUF carries the full
weight set" convention exactly (confirmed necessary the hard way: the first version of
`convert_kokoro_text_encoder.py` wrote only each file's own direction's tensors, which would have forced
either 4 separate `GgufModel`s or a redesigned constructor — fixed by writing the shared set instead,
keeping `BiLstmStepper`'s single-model interface as originally designed). This class is meant to be
reused as-is for `DurationEncoder`'s 3 layers and `ProsodyPredictor`'s own `lstm`/`shared` — same shape
of problem, different weight sets/dimensions.

**Two real bugs caught and fixed during verification** (both in test/conversion code, not the engine
itself):
1. Used a `"$input_dim"` runtime symbol for the LSTM-cell topology's `"layer_input"` declared-input
   shape — but `GraphBuilder` only ever auto-registers `n_tokens`/`n_past`/`n_kv` (confirmed by reading
   `graph_builder.cpp` directly, not assumed). `TextEncoder`'s own LSTM has a DIFFERENT input width
   (512, the CNN output) than its hidden width (256 per direction) — unlike `TdtDecoder`'s own
   layer-input convention, where layer>0's input width always equals the hidden width, so this had never
   come up before. Fixed by making `input_dim` a plain literal number (known at conversion time), not a
   symbol.
2. A real axis-order bug in the C++ test itself: `CONV_1D`'s output has ggml `ne=[n_tokens, channels]` —
   `n_tokens` is the FASTEST axis, so the flat buffer is channel-major (all positions for channel 0,
   then channel 1, ...), not token-major. Extracting each token's own feature vector as a contiguous
   slice (`flat.begin() + t*channels`) silently reads the WRONG data (a mix of unrelated channel/token
   values) without crashing — caught by first isolating the CNN topology alone via a scratch C++
   diagnostic (matched the real reference to 5.5e-6 in isolation, proving the bug was downstream, in how
   the test fed the CNN's output into `BiLstmStepper`, not in the CNN topology or the LSTM cell itself)
   before assuming the bug was in either primitive. Fixed with a proper strided extraction.

Next: `ProsodyPredictor`'s duration-prediction half (`DurationEncoder`'s interleaved BiLSTM+
`AdaLayerNorm` blocks, the top `lstm`, `duration_proj`'s sigmoid-sum regression) — `F0Ntrain` and the
duration-based frame-expansion (needs host-computed rounding/clamping, same "generate_path" precedent as
VITS) are deferred to the following continuation, since they depend on this piece's actual sampled
durations.

---

### 2026-07-19: Kokoro TTS — `ProsodyPredictor`'s duration-prediction half converted and verified

Third assembled Kokoro piece: `tools/convert_kokoro/convert_kokoro_duration_predictor.py` +
`reference_forward_kokoro_duration_predictor.py` + `test_e2e_kokoro_duration_predictor.cpp`. Covers
`DurationEncoder` (3x interleaved bidirectional-LSTM + `AdaLayerNorm`, each followed by re-concatenating
the (broadcast) style vector onto the channel axis — confirmed real from source, including after the
LAST `AdaLayerNorm`, giving `DurationEncoder`'s own output a real `d_model+style_dim=640`-channel width)
→ `ProsodyPredictor.lstm` (another bidirectional LSTM) → `duration_proj` (a plain Linear to
`max_dur=50` raw logits; the `sigmoid().sum(-1)` regression that turns those into an actual duration
value happens in `KModel`, not `ProsodyPredictor`, confirmed from source — done host-side here too, a
tiny scalar post-process). Matched to `max_abs_diff=3.0e-5` across the whole interleaved pipeline.

**Verified before wiring in**: `AdaLayerNorm`'s real `forward` has TWO PAIRS of transposes that
algebraically cancel out entirely (checked numerically against a from-scratch "plain per-position
LayerNorm over channels + style affine" reimplementation, 0.0 diff) — so despite superficially
resembling the Decoder's own `AdaIN1d` (a genuinely different mechanism — real `InstanceNorm`,
transposed to normalize over TIME per-channel), `AdaLayerNorm` is architecturally just this project's
ordinary channel-first `LAYER_NORM` (reduces over `ne[0]`) plus a style-derived `(1+gamma)*x+beta`
affine. Two different "Ada*Norm" mechanisms in the same model family, worth not conflating when the
Decoder is converted.

Style/channel concatenation (both the initial concat before each BiLSTM layer and the re-concat after
each `AdaLayerNorm`) is done in plain host C++ vector splicing, not an in-graph `CONCAT` node — there's
no temporal recurrence in it, so it doesn't need to be graph-resident, and this project has no
concatenate-along-a-non-batch-axis primitive yet (not needed here, given the host round-trip for BiLSTM
stepping happens regardless).

**Two real, generalizable axis-order bugs caught during verification, both in test/conversion code, not
the engine** (a genuinely useful pair — the SAME underlying mistake surfacing on both a write path and a
read path):
1. `BiLstmStepper::run`'s own host output (`std::vector<std::vector<float>>`, T-major, plain C++) was
   flattened into a ggml-bound `[channels,T]` buffer using index formula `c*T+t` — backwards. `ggml
   ne=[channels,T]` has `channels` as the FASTEST axis (flat index `t*channels+c`), byte-identical to a
   numpy/host array of NATIVE shape `(T,channels)` (the "`ggml ne=[a,b]` ↔ numpy `(b,a)`" rule already
   documented repeatedly this project, e.g. VITS's `z_p`/Whisper's `xa`) — meaning `lstm_out` (already
   T-major) needed a plain flatten with NO reordering at all, and the manual `c*T+t` transpose was
   actively wrong, not merely superfluous.
2. The SAME wrong formula was used again on the READ side, extracting `AdaLayerNorm`'s own `ggml`-layout
   output back into a host vector-of-vectors.
   Both caught the same way as every other numerical bug this project has hit: isolate one stage at a
   time via a scratch C++ diagnostic (built a small standalone program dumping `AdaLayerNorm`'s own
   internal `gamma`/`beta`/`normed`/`out` nodes individually, by re-pointing the SAME topology JSON's
   `"output"` field at each internal node name in turn) before assuming the bug was in a primitive
   rather than the surrounding harness — `gamma`/`beta` (the style-derived affine parameters) matched
   the reference EXACTLY, `normed` did not, which correctly narrowed the search to "how is `x` actually
   being fed into `LAYER_NORM`" rather than anything inside the primitive itself.

Next: `F0Ntrain` (the F0/energy prediction half — `AdainResBlk1d` stacks, reusing `istftnet.py`'s own
`AdaIN1d` mechanism, the FIRST real use of it outside the Decoder/vocoder itself) and the duration-based
frame-expansion (host-computed round/clamp/frame-count, VITS's `generate_path` precedent) that produces
its actual input.

---

### 2026-07-19: Kokoro TTS — depthwise `ConvTranspose1d` ("pool") composes from existing primitives

Real, important architecture clarification confirmed while starting `F0Ntrain`/`AdainResBlk1d`: the
"upsample" `AdainResBlk1d` instances (`predictor.F0.1`/`predictor.N.1`, used in `F0Ntrain`, and also in
the Decoder's own `decode` stack) have a `pool` submodule that is a weight-normed, **DEPTHWISE**
(`groups=in_channels`) `ConvTranspose1d` (kernel=3, stride=2, padding=1, output_padding=1 — confirmed
real from the checkpoint's `predictor.F0.1.pool.weight_v` shape `(512,1,3)`, `groups=512`). This has no
direct ggml primitive: `ggml_conv_transpose_1d` is non-grouped only (its own `GGML_ASSERT(a->ne[2] ==
b->ne[1])` forces a single shared kernel across all input channels, confirmed by reading `ggml.c`
directly), and this project has no `CONV_TRANSPOSE_1D_DW`.

**Composes entirely from EXISTING primitives instead** — verified in three stages before trusting it,
same discipline as the ISTFT-via-`CONV_TRANSPOSE_1D` derivation earlier in this milestone:
1. Raw math: zero-stuff the input (insert `stride-1` zeros between samples) + pad `(kernel-1-padding)`
   each side plus `output_padding` on the right + a REGULAR (non-transposed) depthwise correlation with
   the kernel REVERSED along its own length axis — matched real
   `torch.nn.functional.conv_transpose1d(groups=channels)` to `1.2e-7` on a hand-picked example.
2. The planned ggml PRIMITIVE composition (`RESHAPE` to insert a dummy fastest axis + `PAD_1D` by
   `stride-1` on it + `RESHAPE` flattens back — this "overstuffs" a trailing `stride-1` extra zeros past
   the textbook zero-stuffed length, since it pads even the LAST sample; a `VIEW`+`CONT` then truncates
   to the textbook `(L_in-1)*stride+1` length — then a second `PAD_1D` for the edge/output padding, then
   `CONV_1D_DW` with the pre-flipped kernel) matched the SAME numpy simulation to `5.9e-8`.
3. `test_depthwise_conv_transpose_1d_via_composition` (`test_primitive_registry.cpp`) — the actual ggml
   composition, run through the real engine, matched the same reference values to `<1e-5`. 129/129
   primitive-registry checks pass.

The kernel-reversal is a conversion-time weight transform (a plain `numpy` flip along the kernel axis,
alongside the existing weight-norm fold), not a new runtime op — matches this project's established
"fold transformations into weights at conversion time" precedent (weight-norm, xscale, BatchNorm
folding, etc.).

Next: assemble `AdainResBlk1d` fully (this "pool" composition + `AdaIN1d` (already verified, reused from
the Decoder's own earlier verification work) + `LeakyReLU` + weight-normed `conv1`/`conv2` + the
learned-shortcut `conv1x1` + the residual combine), then `F0Ntrain` (shared `BiLstmStepper` + two 3-block
`AdainResBlk1d` stacks + `F0_proj`/`N_proj`), then the duration-based frame-expansion that produces
`F0Ntrain`'s real input.

---

### 2026-07-19: Kokoro TTS — `AdaIN1d`/`AdainResBlk1d` (simplest instance) converted and verified

`tools/convert_kokoro/convert_kokoro_f0n.py` (+ `reference_forward_kokoro_f0_block0.py` +
`test_e2e_kokoro_f0_block0.cpp`) — `predictor.F0.0`, the simplest real `AdainResBlk1d` instance
(`dim_in==dim_out=512`, no learned shortcut, no upsample). Matched to `max_abs_diff=6.0e-6`.

Confirmed `AdaIN1d`'s "plain per-channel-over-time InstanceNorm + style-derived `(1+gamma)*x+beta`"
composition (same no-op-affine finding as `AdaLayerNorm`, but the OPPOSITE axis convention — `AdaIN1d`
needs this project's `[T,C]` (`CONV_1D`-native) layout, normalizing over `ne[0]`=time per channel, real
`InstanceNorm`, vs. `AdaLayerNorm`'s channel-first `[C,T]`, ordinary `LayerNorm`) reduces to the same
`LAYER_NORM` primitive and matches real weights directly — verified stage-by-stage (`norm1`→`act1`→
`conv1`→`norm2`→`act2`→`conv2`→residual-sum→final-scale, each isolated via a scratch diagnostic
re-pointing the topology's own `"output"` field at each internal node name in turn) before trusting the
full block.

Also caught a real bug in `AdainResBlk1d`'s own `_shortcut`/`_residual` split while writing the
conversion code (not from running anything — caught by re-reading `istftnet.py`'s real source carefully
a second time): `_shortcut` upsamples via `self.upsample` (plain nearest-neighbor `UpSample1d`, i.e.
`INTERPOLATE_1D` mode=nearest), while `_residual` upsamples via `self.pool` (the LEARNED depthwise
`ConvTranspose1d` verified earlier this milestone) — genuinely TWO DIFFERENT upsampling mechanisms on
the two branches of the same block, not the same "pool" reused on both. An earlier draft of
`add_adain_resblk1d` used the depthwise-conv-transpose composition on the shortcut path too; fixed before
it was ever tested against a real upsampling instance.

**A real, generalizable bug, previously LATENT across this entire project**: the numerical mismatch
first showed as a large, confusing discrepancy (`mean_abs_diff≈1.1`, `max≈6.6`) even though a scratch
diagnostic proved every internal stage of the topology matched the reference exactly. Root cause: the
Python reference's final `out` tensor ends its computation with a `.T` transpose (`conv1d(...)[0].T`)
and is never explicitly re-contiguated before `np.save` — PyTorch preserved the resulting
non-contiguous, Fortran-ordered memory layout straight through the final add/scale ops, and `np.save`
faithfully wrote `'fortran_order': True` into the `.npy` header. This project's hand-written minimal
`.npy` reader — duplicated across essentially every e2e test's C++ side this whole project — **never
checks `fortran_order` at all** and unconditionally assumes C-order, so it silently misread a genuinely
transposed array as if it weren't transposed, with no error or crash. Confirmed by directly inspecting
the `.npy` header bytes. Every PRIOR reference script in this project happened to save already-
C-contiguous arrays (by luck, or because their own final op didn't leave a transposed view), so this
had never surfaced before despite the reader gap existing the whole time. Fixed AT THE SOURCE (the
Python reference script now calls `np.ascontiguousarray(...)` before every `np.save`), not by teaching
the C++ reader about `fortran_order` — cheaper and the right place to guarantee it, but the READER gap
itself is now a known, documented risk: **any future reference script whose final tensor comes out of a
`.T`/`.transpose()`/`.permute()` chain without an explicit `.contiguous()` needs this same
`np.ascontiguousarray()` guard, or it will silently produce a wrong (transposed) comparison with no
error at all.**

Next: the upsampling `AdainResBlk1d` variant (`F0.1`, exercises the depthwise-conv-transpose "pool" +
learned `conv1x1` shortcut together, both individually verified but not yet together in a real block),
then `F0Ntrain`'s full assembly.

---

### 2026-07-19: Kokoro TTS — upsampling `AdainResBlk1d` (`F0.1`) converted and verified

`convert_kokoro_f0n.py` extended + `reference_forward_kokoro_f0_block1.py` +
`test_e2e_kokoro_f0_block1.cpp` — `predictor.F0.1` (`dim_in=512→dim_out=256`, WITH the learned
`conv1x1` shortcut AND `upsample=True`, doubling `T`). First real exercise of the depthwise-
`ConvTranspose1d` "pool" composition and the learned shortcut TOGETHER (each individually verified
earlier this milestone). Matched to `max_abs_diff=3.3e-6` — passed on the first real run, no new bugs
found here (the earlier `_shortcut`-vs-`_residual` upsample-mechanism mixup was caught and fixed
*before* this test, while writing the conversion code, not by this test failing).

Confirms: the "pool"'s output length is exactly `2*n_tokens` as derived (`T=6→T_out=12`), and the
`SymbolEnv` arithmetic expression `"2*$n_tokens"` threads correctly through every downstream `RESHAPE`
in the block (`conv1`/`conv2`/the shortcut's own `conv1x1`).

With both `AdainResBlk1d` variants now verified, `F0Ntrain`'s remaining new surface is just assembly:
the shared `BiLstmStepper` (already proven for `TextEncoder`/`DurationEncoder`) feeding two independent
3-block `AdainResBlk1d` stacks (`F0`: 512→512→256→256, `N`: same shape) + `F0_proj`/`N_proj` (plain
`Conv1d(256,1,1)`). Next: assemble `F0Ntrain` fully, then the duration-based frame-expansion that
produces its real input (`en`, from `DurationEncoder`'s own 640-channel output — NOT the top `lstm`'s
512-channel output, confirmed from `KModel.forward_with_tokens`'s real call order directly).

---

### 2026-07-19: Kokoro TTS — `F0Ntrain` full assembly converted and verified

`convert_kokoro_f0n.py` extended (`write_bilstm_ggufs` reused for the new `shared` BiLSTM instance,
640→512; `write_stack` writes the 3-block `AdainResBlk1d` GGUFs for both the `F0` and `N` stacks;
`write_proj1x1` writes `F0_proj`/`N_proj` as a direct `CONV_1D` on the stack's own `[T,C]`-convention
output — no `MUL_MAT`-as-matmul transpose needed here, unlike VITS's channel-first conv1x1 sites,
since the real weight shape `(1,256,1)` numpy already matches `CONV_1D`'s `[K,IC,OC]` kernel
convention directly) + `reference_forward_kokoro_f0ntrain.py` + `tests/test_e2e_kokoro_f0n.cpp`.
Verified `mean_abs_diff=1.3e-5, max_abs_diff=6.1e-5` (24/24 checks), `T=5→T_out=10`.

Two things caught while writing the reference, before any test ran:
- The existing `reference_forward_kokoro_f0_block1.py`'s `adain_resblk1d` hardcodes shortcut+upsample
  together (built only to match `F0.1`'s own specific instance) and has no `conv1x1` fallback for the
  `dim_in==dim_out` case — raised a real `KeyError` on `F0.0`/`F0.2`/`N.0`/`N.2` (no `conv1x1.*` keys
  exist for those). Fixed by writing a single general `adain_resblk1d_general` in the new reference
  script that treats `learned_sc = dim_in != dim_out` and `upsample` as the two independent real flags
  they are (confirmed in `istftnet.py` directly), rather than patching two mismatched hardcoded
  helpers to cover the 3rd combination.
  worth noting for any FUTURE model port that reuses per-instance reference helpers across multiple
  differently-shaped instances of the same module class: don't assume an earlier single-instance
  reference generalizes; check its shortcut/branch assumptions against every instance's real shape
  first.
- Confirmed (again) that chaining `AdainResBlk1d` blocks together on the C++ side needs **no host-side
  transpose between blocks at all** — each block's raw ggml output buffer (`ne=[T,C]`, `T` fastest) is
  already in exactly the layout the next block's `"x"` input wants, so `run_stack`'s C++ loop just
  passes the raw `ggml_backend_tensor_get` buffer straight through. The only real transposes needed are
  at the true boundaries: `en`'s `(640,T)` numpy layout (channel-outer, T-inner — coincidentally already
  ggml-shaped) converted to `BiLstmStepper`'s T-major vector-of-vectors convention, and that stepper's
  T-major output converted back to ggml's channel-outer flat layout before it's fed to the first block.

`F0Ntrain` is now complete and fully verified. Remaining Kokoro work: the duration-based
frame-expansion that produces `F0Ntrain`'s real input (`en`, `DurationEncoder`'s own 640-channel
output aligned to frame rate via `commons`-style duration expansion — VITS's own
`generate_path`/`CUMSUM` precedent, see the VITS plan, is the relevant model here too), then the full
`Decoder`/`Generator` (Snake activation, NSF harmonic source, ISTFT reconstruction — individually
verified earlier this milestone but not yet assembled into a real topology), the overall Kokoro
conversion script tying every piece together, and a `KokoroDriver`-style host class.

---

### 2026-07-19: Kokoro/StyleTTS2 — duration-based frame-expansion (`loom::predict_durations`/
### `loom::expand_by_duration`), host-side, no new primitive

Read the real inference path directly (`kokoro/model.py`'s `KModel.forward_with_tokens`, from the
`kokoro` pip package actually installed in the piper venv — much clearer than re-deriving from
`modules.py` alone) to confirm exactly what produces `F0Ntrain`'s real `en` input:

```python
duration = sigmoid(self.predictor.duration_proj(x)).sum(-1) / speed
pred_dur = round(duration).clamp(min=1).long()
indices = repeat_interleave(arange(T_text), pred_dur)
pred_aln_trg[indices, arange(T_frames)] = 1          # one-hot alignment matrix
en = d.transpose(-1, -2) @ pred_aln_trg              # d: DurationEncoder's own 640-ch output (WITH
                                                      # style already concatenated on its last iteration
                                                      # — confirmed directly in modules.py's
                                                      # DurationEncoder.forward, matches this project's
                                                      # existing duration_predictor test's own "x" exactly)
asr = t_en @ pred_aln_trg                            # same expansion, reused for TextEncoder's 512-ch output
```

This is a genuinely SIMPLER operation than the roadmap originally assumed by analogy to VITS's
`generate_path` (`commons.py`, cumsum + broadcast-compare + shift/pad + mask-multiply — built to
support a smoother/monotonic-but-not-strictly-one-hot alignment in general). Kokoro's `pred_aln_trg`
is *always* an exact one-hot matrix (each output frame maps to exactly one input token), so
`seq^T @ pred_aln_trg` collapses to nothing more than "repeat row `t` of `seq`, `pred_dur[t]`
consecutive times" — nowhere close to needing a `CUMSUM` primitive or any ggml graph node at all. (This
also matches what VITS's OWN `generate_path` already degenerates to in `vits_driver.cpp` once its
per-utterance mask is dropped — confirmed by reading that code: both models' real alignment mechanisms
collapse to the identical "replicate a column N times" host operation once batching/masking is out of
the picture, they just arrive at it from different starting formulas.)

Implemented as two new free functions (not a class — no state carried between calls, unlike
`BiLstmStepper`/`TdtDecoder`'s per-step recurrence): `loom::predict_durations` (sigmoid+sum/round/clamp)
and `loom::expand_by_duration` (the repeat), in `include/loom/core/duration_aligner.h` /
`src/core/duration_aligner.cpp`. Deliberately used `std::nearbyint` (ambient-rounding-mode, defaults to
round-half-to-even) rather than `std::lround` (round-half-away-from-zero) to match `torch.round`'s real
semantics exactly, even though float32 sigmoid-sum outputs essentially never land on an exact tie in
practice — same "verify against the real formula, not an approximation of it" discipline as everywhere
else in this project.

Verified against a hand-rolled numpy reference (`tools/fixture_gen/reference_duration_aligner.py`,
`tests/test_duration_aligner.cpp`) — pure host arithmetic with no GGUF/ggml graph involved at all, so
this fixture is procedurally generated at ctest time (numpy-only, no torch needed) rather than
skip-if-missing like the real-checkpoint tests, matching `test_lstm_step.cpp`'s own convention for
synthetic composite-logic verification. Exact bit-for-bit integer match on `pred_dur` and exact
float match on the expanded output (6 tokens → 111 frames in the fixture).

With this piece done, Kokoro's remaining work is: assembling `F0Ntrain`'s real input end-to-end (run
the already-verified `DurationEncoder`+top-`lstm`+`duration_proj` → `predict_durations` →
`expand_by_duration` on both `d` and `t_en` → feed into the already-verified `F0Ntrain`), then the full
`Decoder`/`Generator` (Snake activation, NSF harmonic source, ISTFT reconstruction — individually
verified earlier this milestone but not yet assembled into a real topology), the overall Kokoro
conversion script tying every piece together, and a `KokoroDriver`-style host class.

---

### 2026-07-19: Kokoro Generator — two new primitives (`CONCAT`, `ATAN2`), forward/inverse STFT verified

Started on `istftnet.py`'s `Decoder`/`Generator` (the HiFi-GAN-NSF vocoder, Kokoro's last major
unbuilt piece). Read the real source in full (`Generator`, `AdaINResBlock1`, `SineGen`,
`SourceModuleHnNSF`, `TorchSTFT`, `Decoder`, `AdainResBlk1d`) and the real checkpoint's `config.json`
(`upsample_rates=[10,6]`, `upsample_kernel_sizes=[20,12]`, `upsample_initial_channel=512`,
`resblock_kernel_sizes=[3,7,11]`, `resblock_dilation_sizes=[[1,3,5]]*3`, `gen_istft_n_fft=20`,
`gen_istft_hop_size=5`) before designing anything. Broke the remaining work into 6 tracked tasks
(#84-#90); this entry covers the first three (#84-#86).

**`CONCAT` primitive** (`src/ops/primitives_basic.cpp`): wraps ggml's native `ggml_concat(a,b,dim)`
(2-input; chain for 3+ tensors). Needed because Kokoro's `Decoder.forward` concatenates
graph-computed tensors (`torch.cat([asr,F0,N],axis=1)`, `torch.cat([har_spec,har_phase],dim=1)`,
`torch.cat([x,asr_res,F0,N],axis=1)`) — genuinely different from every EARLIER Kokoro concatenation
(style-with-features), which was always host-precomputable before entering the graph at all. Verified
both `dim=0` and `dim=1` against hand-computed small examples in `test_primitive_registry.cpp`.

**`ATAN2` primitive**: ggml has no native `atan2` (nor even single-arg `atan`). Needed because
`Generator`'s noise path uses the REAL phase (`torch.angle` == `atan2(imag,real)`) of the harmonic
source's own STFT as a trained-weight-dependent auxiliary input — unlike `RQ_SPLINE_INVERSE` elsewhere
in this project, there's no algebraic reformulation that avoids the transcendental function entirely.
Added via `ggml_map_custom2` (ggml's own escape hatch for exactly this case — no native op, no viable
composition), verified against `std::atan2` across all 4 quadrants plus x=0 edge cases.

**Forward/inverse STFT** (`tools/convert_kokoro/kokoro_stft_common.py`,
`convert_kokoro_stft.py`, `reference_forward_kokoro_stft.py`, `tests/test_e2e_kokoro_stft.cpp`):
reused Whisper's mel-frontend DFT-via-`CONV_1D` trick (`build_dft_kernels`/`periodic_hann`,
`tools/convert_whisper/whisper_common.py`) for the forward transform, PLUS the real phase this time
(Whisper only ever needed magnitude). Two real findings, both caught by verifying the composition in
plain Python/numpy against real `torch.stft`/`torch.istft` BEFORE writing any GGUF/C++ code (same
discipline as every other primitive/composition this project has added):

- **DC/Nyquist bin sign-of-zero bug**: a real signal's imaginary STFT component is mathematically
  exactly zero at `k=0` and `k=n_fft/2` (n_fft even) — but computing it as `-(sin_kernel @ frame)`
  produces IEEE754 **negative** zero (since `sin(0)=0.0` exactly, and `-(+0.0)` is `-0.0`), while real
  `torch.stft` always returns **positive** zero there (confirmed across 200 random seeds). This flips
  `atan2`'s branch (`atan2(-0,neg)=-π` vs `atan2(+0,neg)=+π`), a spurious ~2π error at exactly those 2
  bins every time. Fixed via a `boundary_mask` (1.0 at the two affected bins) and the identity
  `x - x*mask`: exactly `+0.0` for any finite `x` when `mask=1` (subtracting a value from itself is
  always positive zero in IEEE754, regardless of `x`'s sign), and exactly `x` when `mask=0` — robust in
  a way `x*(1-mask)` is not.
- **Phase comparison is inherently unstable at the ±π branch cut**: even with the fix above, two
  independently-computed float32 pipelines can land on opposite sides of a genuine discontinuity from
  ~1e-7-level rounding noise alone (confirmed: 1 element out of 143 differed by ~2π in one seed, at a
  point where both `re<0` and `im≈0` from real signal noise, not from any bug). Comparing raw phase
  differences will show spurious ~2π "errors" that aren't real errors. Fixed by using a **circular**
  distance (`((a-b+π) mod 2π) - π`, absolute value) in the test instead of a plain difference —
  documented in `kokoro_stft_common.py`'s module docstring as a standing rule for comparing ANY
  angle-valued quantity in this project, not just here.
- Inverse STFT reduces to two `CONV_TRANSPOSE_1D` calls (real-part and imag-part contributions, window
  baked into the synthesis kernel) summed, divided by the overlap-added squared-window normalization
  (`wsum`) — verified bit-for-bit (~2e-8) against real `torch.istft` on independently-random
  (non-self-consistent) magnitude/phase, matching how `Generator` actually uses it (`spec=exp(...)`,
  `phase=sin(...)`, never a true forward-transform's own output). `wsum` depends only on `n_frames`
  (not on any graph value), so it's computed **host-side** (a trivial loop) and fed in as a declared
  input, rather than adding a third `CONV_TRANSPOSE_1D` call over a constant all-ones signal just to
  keep it in-graph.
- **`ggml_conv_transpose_1d`'s kernel channel convention is the OPPOSITE of `CONV_1D`'s**: confirmed
  directly against ggml's real assertion (`a->ne[2] == b->ne[1]`, i.e. kernel `ne[2]` = data's channel
  count) — `CONV_1D` kernels are `[K,IC,OC]` (`ne[1]`=IC), `CONV_TRANSPOSE_1D` kernels are `[K,OC,IC]`
  (`ne[1]`=OC), matching real PyTorch's own `Conv1d` vs `ConvTranspose1d` weight-layout difference
  ((OC,IC,K) vs (IC,OC,K) natively). Numerically invisible in this specific case since OC=1 for both
  synthesis kernels here, but documented since it will matter the moment a real multi-output-channel
  `ConvTranspose1d` shows up (the Generator's actual upsampling convs, next).
- **`SymbolEnv`'s int-attribute resolution rounds-to-nearest (`std::llround`), not floor/truncate**:
  a real bug caught immediately when the forward topology's `n_frames` shape expression
  `"($n_tokens-20)/5+1"` evaluated to `13.8` for a real `n_tokens=84` (ordinary floating-point division,
  since `SymbolEnv::eval` operates on `double`s throughout) and `llround(13.8)=14`, not the correct `13`
  — a `ggml_reshape` element-count mismatch abort. Fixed by using the grammar's already-supported
  `floor()` function explicitly (`"floor(($n_tokens-20)/5)+1"`), landing exactly on `13.0` before
  rounding. **Any future topology attribute expression involving `/` on dynamic symbols needs an
  explicit `floor()` wrap if floor-division semantics are intended** — plain `/` alone is not safe.

Verified: forward STFT max_mag_diff=9.5e-7, max_phase_circular_diff=8.2e-7 (13 frames); inverse STFT
max_diff=3.0e-8 (40-sample output). Full suite still 70/70.

Next: `SineGen`/`SourceModuleHnNSF` (the NSF harmonic source feeding this forward STFT), then
`AdaINResBlock1` (Snake activation + dilated resblocks, distinct from `predictor.F0/N`'s
`AdainResBlk1d`), then assembling the full `Generator` and `Decoder`.

---

### 2026-07-19: Kokoro Generator — `SineGen`/`SourceModuleHnNSF` (NSF harmonic source) converted and verified

One new primitive (**`FLOOR`**, trivial `ggml_floor` wrapper — needed for `(f0/sampling_rate) % 1`,
expressed as `x - floor(x)` since ggml has no native modulo and every operand here is non-negative).

Read `istftnet.py`'s real `SineGen`/`SourceModuleHnNSF` in full. Confirmed `Generator.forward` only
ever uses `SourceModuleHnNSF`'s FIRST return value (`har_source`) — `noise`/`uv` are computed and
returned but never used afterward — so this converts/verifies only `har_source`, not the full module
surface. Real hyperparameters pulled from `Generator.__init__`: `sampling_rate=24000`,
`upsample_scale=math.prod(upsample_rates)*gen_istft_hop_size=10*6*5=300`, `harmonic_num=8` (dim=9),
`voiced_threshod=10`, plus `SineGen`'s own defaults `sine_amp=0.1`/`noise_std=0.003`.

**Algorithm** (nearest-upsample F0 by 300x → per-harmonic phase accumulation via a
downsample(1/300,linear)+cumsum+upsample(300,linear) dance → sin → voiced/unvoiced-gated noise mix →
`Linear(9,1)`+tanh) verified TWICE before writing any GGUF/C++: first a standalone numpy
reimplementation cross-checked directly against a hand-copied, unmodified real `SineGen`/
`SourceModuleHnNSF.forward` (matched real torch to 3.0e-8, with matching injected `rand_ini`/`noise`
draws), then the actual loom composition verified against an independently-written second PyTorch
reference (mean_diff=7.8e-8, max_diff=2.9e-6, `T_frames=4→L=1200`).

Two host-drawn random inputs (`rand_ini`, the harmonic phase's random initial-offset draw — index 0
always exactly 0 per `SineGen`'s own `rand_ini[:,0]=0`; and `noise`, the per-sample additive Gaussian
term) — same "host draws via `<random>`, feeds in as a declared input" precedent as VITS's `z_p`
sampling, not a new mechanism.

No outer-product/broadcast-repeat primitive needed for building the 9 harmonic channels
(`fn[:,h]=f0_up*(h+1)`) — `dim=9` is small and conversion-time-fixed, so this unrolls 9
`SCALE`+`RESHAPE`+`CONCAT` calls instead (same "unroll a small fixed-size loop at conversion time"
precedent as Whisper's per-head loops).

**Real bug caught by ggml's own `ggml_can_repeat` assertion** (not by a silent wrong-answer — a hard
abort): `ggml_add(a,b)`/`ggml_mul(a,b)` require `b` to broadcast INTO `a`'s shape — `a` (the first
argument) determines the OUTPUT shape, so `a` must always be the "bigger" (or equal) tensor. Had
`noise_amp = ADD(base_w[1,1], uv_delta[L,1])` backwards (the constant `[1,1]` first, the `[L,1]`
per-frame tensor second) — `[L,1]` cannot broadcast into `[1,1]`, so this aborted immediately at graph
build time rather than producing a silently wrong answer. Fixed by swapping the argument order
(`ADD(uv_delta, base_w)`). **General rule now confirmed for future composition work**: when one operand
is a per-position/per-channel computed tensor and the other is a small conversion-time constant, the
computed tensor must be listed FIRST.

Confirmed (again) ggml's `ggml_conv_transpose_1d`-style axis-convention-crossing pattern shows up here
too, in a different form: `CUMSUM`/`INTERPOLATE_1D` need TIME on `ne[0]` (this project's usual `[T,C]`
convention), but the final `Linear(9,1)` projection needs the CONTRACTED (channel) axis on `ne[0]` for
`MUL_MAT` — crossed via `PERMUTE`(axes=[1,0,2,3])+`CONT`, the same boundary-crossing precedent used
everywhere else a genuine axis-convention change is needed (never assumed away).

Next: `AdaINResBlock1` (Snake activation + dilated resblocks, the OTHER "AdaIN*"-family block —
distinct from `predictor.F0/N`'s `AdainResBlk1d`, used by `Generator.resblocks`/`noise_res`), then
assembling the full `Generator` and `Decoder`.

---

### 2026-07-19: Kokoro Generator — `AdaINResBlock1` (Snake-activation dilated resblock) converted and verified

No new primitive needed. `tools/convert_kokoro/kokoro_generator_common.py` (new, shared module for the
rest of the Generator/Decoder assembly): `add_adain_resblock1` + `add_snake`, reusing
`add_adain1d`/`fold_weight_norm`/`to_f32` from `convert_kokoro_f0n.py` directly rather than
re-deriving them — confirmed against the real checkpoint's state dict
(`decoder.generator.{resblocks,noise_res}.*`) that this `AdaIN1d` instance has the SAME
never-trained-affine-params property as every other one this whole milestone (no `.norm.weight`/
`.norm.bias` keys anywhere).

Real structural difference from `predictor.F0/N`'s `AdainResBlk1d`: `AdaINResBlock1` has NO
shortcut/upsample path at all (`dim_in` always equals `dim_out`, sequence length never changes) and
uses 3 (dilation=1,3,5) `(AdaIN1d→Snake→dilated conv1)→(AdaIN1d→Snake→conv2,dilation=1)→residual-add`
stages instead of `AdainResBlk1d`'s single norm/act/conv pair — reused via a NEW builder function, not
a variant flag on the existing one (genuinely different control flow, not a parameterization of the
same shape).

**Snake activation** (`x + (1/a)*sin(a*x)^2`) needed a per-channel LEARNED `a` (real weight shape
`(1,channels,1)`) — unlike VITS's own HiFi-GAN `Generator` (`test_hifigan_generator`), which has no
such activation at all. Folded the reciprocal `1/a` at CONVERSION time (plain numpy division, same
"fold at conversion time" precedent as weight-norm) rather than adding a `DIV` node in-graph, and
reshaped the squeezed `(channels,)` constant to `[1,channels]` in-graph for broadcasting against this
project's `[T,C]` convention.

Verified against a hand-rolled PyTorch reference on a small SYNTHETIC instance (channels=4,
style_dim=8, kernel=3, dilations=(1,3,5), random weights) — checkpoint-independent structural
verification, same precedent as VITS's own `test_hifigan_generator` (verify the WIRING first, wire in
real checkpoint weights when the full `Generator` is assembled). Passed on the first real run:
mean_abs_diff=6.9e-7, max_abs_diff=1.9e-6.

All 3 of the Generator's genuinely new pieces (STFT, `SineGen`, `AdaINResBlock1`) are now done and
individually verified. Remaining: assemble the full `Generator` (upsample stack via
`CONV_TRANSPOSE_1D`, wiring the noise path through `noise_convs`/`noise_res`, resblock fan-out-then-
average, final `conv_post`→spec/phase split→inverse STFT — reusing VITS's own
`ConvTranspose1d(padding>0)`-via-crop trick from `test_hifigan_generator` for the upsampling convs) and
the `Decoder` (F0_conv/N_conv downsampling, `CONCAT`-based channel joining, `encode`/`decode`
`AdainResBlk1d` stack reusing `convert_kokoro_f0n.py`'s existing builder verbatim with new dims, then
the `Generator`), against the real checkpoint.

---

### 2026-07-19: Kokoro Generator — full assembly (upsample stack + noise path + resblocks + inverse STFT) verified

One new primitive (**`EXP`**, trivial `ggml_exp` wrapper — needed for the final `spec=exp(x[:n_freq])`
split). `tools/convert_kokoro/convert_kokoro_generator.py` assembles the whole `Generator.forward`
(minus `SineGen`/forward-STFT, which stay separate topologies the host driver runs first and feeds in
as a ready-made `har` input — same "compose already-verified pieces via the host driver" pattern as
`BiLstmStepper`).

Confirmed real channel/kernel shapes directly against the checkpoint's
`decoder.generator.{ups,noise_convs,conv_post}.*` state dict before writing anything (`ups.0`:
512→256 k=20, `ups.1`: 256→128 k=12; `noise_convs.0`: 22→256 k=12 s=6, `noise_convs.1`: 22→128 k=1;
`conv_post`: 128→22 k=7 — every shape matched the derivation on the first check, not adjusted after).

**Length bookkeeping, verified algebraically before writing any GGUF/C++ code** (letting `T0` = the
topology's own input length): both real `upsample_rates` (`[10,6]`) use `kernel_size=2*stride`,
`padding=(kernel_size-stride)//2=stride//2` — the "integer-exact upsample" ConvTranspose1d config, so
`ggml_conv_transpose_1d`'s p0=0-only limitation + crop (VITS's own `test_hifigan_generator` precedent)
resolves CLEANLY to `T0*10` then `T0*60`, no `floor()`-guarded fractional expression needed anywhere in
the main path. The harmonic source's own STFT frame count (`T_har=T0*60+1`, one more than the main
path) reconciles with the main path exactly at each `noise_convs[i]` (proven by direct algebraic
substitution, not just spot-checked numerically): stage 0's strided conv (`K=12,S=6,P=3`) maps
`T_har→T0*10` exactly (`floor((T0*60+1+6-12)/6)+1` simplifies to `floor(T0*10-5/6)+1=T0*10`, since
`T0*10` is always an integer); stage 1's 1×1 conv leaves `T_har` unchanged, matching the main path's own
`T0*60+1` after its reflection-pad. **A real, tuned architecture where these must and do align exactly**
— not a coincidence, confirmed by carrying the algebra through symbolically rather than trusting a
single numeric spot-check.

**Real bug caught immediately on the first end-to-end run, before ever seeing a numerical mismatch**: a
`ggml_reshape_3d` nelements-mismatch abort from passing the WRONG length expression to
`add_adain_resblock1` for the noise path's `x_source` — used `har`'s own length (`T_har`, the length
BEFORE the strided `noise_conv`) instead of `x_source`'s REAL post-conv length (`T0*10` for stage 0,
`T0*60+1` for stage 1 — the length it must actually have to be added to the main path `x`). Fixed by
computing `x_source_len_expr` explicitly per stage instead of reusing `t_har_expr`. This is exactly the
class of bug `add_adain_resblock1`'s new `seq_len_expr` parameter (added earlier this entry, replacing
its old hardcoded `"$n_tokens"` default) exists to prevent — confirms that generalization was the right
call, not premature.

**One-sample `ReflectionPad1d((1,0))`** (applied only after the LAST upsample stage): confirmed directly
against real `torch.nn.ReflectionPad1d((1,0))` that it degenerates to "prepend a copy of the element at
index 1" (verified: `pad([1,2,3,4])=[2,1,2,3,4]`) — a `VIEW`(1-sample slice at index 1, using `VIEW`'s
own default `nb1`-omitted behavior)+`CONT`+`CONCAT` composition, not a general reflection-pad primitive
(unneeded for a width-1 pad specifically).

Verified against a hand-rolled PyTorch reference on a small SYNTHETIC instance (real checkpoint's exact
channel/kernel shapes, random weights, `T0=2→T_har=121→waveform_len=600=T0*300`) — checkpoint-independent
structural verification, same precedent as VITS's `test_hifigan_generator`. Passed after the one length
bug above was fixed: mean_diff=2.0e-4, max_diff=9.8e-3 (looser than earlier single-block Kokoro pieces,
expected given this chains many more layers — same floating-point-chaos-amplification calibration
already established for Whisper's own multi-layer encoder).

The full Generator is now done and verified. Remaining: the `Decoder` (F0_conv/N_conv downsampling,
`CONCAT`-based channel joining, `encode`/`decode` `AdainResBlk1d` stack reusing `convert_kokoro_f0n.py`'s
existing builder verbatim with new dims, then this Generator), against the real checkpoint — the last
piece before a `KokoroDriver` and the overall conversion script tying every verified piece together.

---

### 2026-07-19: Kokoro `Decoder` "core" (F0_conv/N_conv + encode/decode stack) converted and verified —

### Generator/Decoder assembly COMPLETE

No new primitive needed — `convert_kokoro_decoder_core.py` reuses `convert_kokoro_f0n.py`'s
`add_adain_resblk1d` VERBATIM (the exact same `AdainResBlk1d` class is used for both
`predictor.F0/N` and `Decoder.encode`/`decode` — confirmed directly, not assumed) plus `CONCAT`.

Confirmed real channel shapes directly against the checkpoint's `decoder.{encode,decode,F0_conv,N_conv,
asr_res}.*` state dict before writing anything: `Decoder.__init__`'s `dim_in` argument is
`config['hidden_dim']=512` (NOT `config['dim_in']=64`, an unrelated hyperparameter used elsewhere —
confirmed directly in `model.py`'s `KModel.__init__`, a real one-line misreading risk caught before it
became a bug). `encode`: `514→1024` (`asr`(512)+`F0`(1)+`N`(1)); `decode[0..2]`: `1090→1024`
(`x`(1024)+`asr_res`(64)+`F0`(1)+`N`(1)), each re-concatenating every iteration until the first
`upsample=True` block; `decode[3]`: `1090→512`, upsamples.

**Length bookkeeping, verified algebraically** (letting `T_frames` = this topology's own `$n_tokens`,
the original text/duration-alignment frame count matching `asr`'s own length): `F0_conv`/`N_conv`
(weight-normed `Conv1d(1,1,k=3,s=2,p=1)`) downsample F0Ntrain's own upsampled output (`2*T_frames`)
back to exactly `T_frames` (`floor((2T-1)/2)=T-1` for any integer `T≥1`, so `floor((2T+2-3)/2)+1=T`
exactly) — a clean, algebraically-confirmed match, not a numeric coincidence. Since this topology's own
primary `$n_tokens` symbol IS `T_frames` itself, `add_adain_resblk1d`'s existing hardcoded
`"$n_tokens"`/`"2*$n_tokens"` length expressions apply UNCHANGED here (unlike the Generator's
`resblocks`, which needed the new `seq_len_expr` parameter because they operate at lengths other than
their topology's own primary symbol) — confirms that parameter was added for exactly the right reason
and no more.

Verified against a hand-rolled PyTorch reference on a small SYNTHETIC instance (real checkpoint's exact
channel shapes, random weights, `T=3→x shape=(6,512)`, matching the Generator's own expected 512-channel
input exactly) — passed on the first real run: mean_abs_diff=2.0e-5, max_abs_diff=1.3e-4.

**This completes Kokoro's entire TTS synthesis math** (`CustomAlbert` → `TextEncoder` →
`ProsodyPredictor` (`DurationEncoder`+duration/F0/energy prediction) → the host-side duration-based
frame-expansion → `Decoder` (`encode`/`decode` → `Generator` → real STFT/ISTFT) — every real
architectural piece individually verified against either the real checkpoint or a hand-rolled PyTorch
reference of the real formula, 74/74 tests passing). New primitives added across the whole Kokoro
effort: `SIN`, `COS`, `INTERPOLATE_1D`, `LEAKY_RELU`, `CUMSUM`, `CONCAT`, `ATAN2`, `FLOOR`, `EXP` — none
needed for attention/matmul-heavy work, all for genuinely new elementwise/structural operations this
model family exercises that no prior model in this project's roadmap needed.

Remaining before Kokoro is fully "driveable": a `KokoroDriver`-style host class wiring every verified
topology together in the real call order (`CustomAlbert`→`TextEncoder`(DurationEncoder half)→
`predict_durations`/`expand_by_duration`→`F0Ntrain`→`SineGen`/forward-STFT→`Decoder`→`Generator`), the
overall conversion script pulling every piece's real weights from the ONE real checkpoint into a
coherent set of GGUF files, and (separately, lower priority) a permissively-licensed phonemizer for
text→phoneme conversion (already tracked, task #79, deferred pending a licensing check — VITS needs the
exact same piece). Then on to StyleTTS2 (likely shares most of this architecture directly), SupertonicTTS,
F5-TTS, Matcha-TTS.

---

### 2026-07-19: KokoroDriver + master conversion script — Kokoro is now fully driveable end-to-end

**`bert_encoder`** (`Linear(768,512)`, `tools/convert_kokoro/convert_kokoro_bert_encoder.py`): the one
remaining unconverted real-checkpoint piece. Forced a careful re-derivation of this project's TWO
distinct `[T,C]`-family axis conventions, since getting this wrong would have silently fed the whole
downstream pipeline transposed data:
- **Layout A** (`ne=[T,channels]`, `T=ne[0]` fastest, flat index `c*T+t`) = native PyTorch `(C,T)`
  channel-first, reversed. Used throughout `CONV_1D`/`AdainResBlk1d`/`AdaINResBlock1`/`Decoder`/
  `Generator` — everything convolutional.
- **Layout B** (`ne=[channels,T]`, `channels=ne[0]` fastest, flat index `t*channels+c`) = native
  PyTorch `(T,channels)` time-major, reversed — the ordinary transformer/RNN layout. Used by
  `CustomAlbert`'s own raw output (confirmed directly against `test_e2e_kokoro_albert.cpp`'s own
  comment: `ne=[hidden_size,T]`) and `AdaLayerNorm`'s own `x` input (confirmed against
  `build_adaln_topology`'s declared `["channels","$n_tokens"]` shape) — and it's ALSO exactly what a
  host T-major vector-of-vectors (e.g. `BiLstmStepper`'s own output) flattens into with NO reordering.

  `bert_encoder`'s real forward (`Linear` then `.transpose(-1,-2)`) is thus: input Layout B (`ne=
  [768,T]`, matching Albert's raw output byte-for-byte) → `MUL_MAT` naturally produces Layout B again
  (`ne=[512,T]`, since the weight's own output dim always lands on `ne[0]`) → the real
  `.transpose(-1,-2)` is an EXPLICIT `PERMUTE`+`CONT` converting Layout B → Layout A (`ne=[T,512]`),
  matching `DurationEncoder`'s own expected `d_en` convention. Verified against the real checkpoint's
  own weights: mean_diff=1.7e-7.

**`convert_kokoro_all.py`** (master conversion script): discovered every earlier per-piece script's
`main()` ALREADY used the real checkpoint's real weights (not synthetic) except the three pieces built
for pure structural verification this milestone (`AdaINResBlock1` standalone, `Generator`, `Decoder`
core) — so this just orchestrates each existing script's real-weight logic in one place, adding real
`sd_prefix` wiring (`convert_kokoro_generator.py`'s `build_generator`/`convert_kokoro_decoder_core.py`'s
`build_decoder_core` already accepted an `sd_prefix` parameter for exactly this) for the 3 pieces that
needed it. Runs cleanly against the real checkpoint, producing the full ~44-file GGUF set.

**`KokoroDriver`** (`include/loom/core/kokoro_driver.h`/`src/core/kokoro_driver.cpp`): the real
`KModel.forward_with_tokens` call order end-to-end, owning every loaded `GgufModel` directly (unlike
`VitsDriver`'s reference-only convention — Kokoro's real pipeline spans ~40 small GGUF files, too many
to reasonably hand to callers to construct themselves). Real bug caught by a segfault (not a silent
wrong answer) on the FIRST real run, isolated via manual `fprintf` bisection (no `gdb`/`lldb` available
in this environment): a helper returned an output tensor via a raw `ggml_tensor*` out-param, but the
tensor's owning `GraphBuilder`/`ggml_context` was a LOCAL variable destroyed when the helper returned —
a genuine use-after-free the very next line (`out_t->ne[0]`) dereferenced. Fixed by reading `ne[0]`/
`ne[1]` into plain `uint32_t` out-params BEFORE the helper returns, never handing back a pointer into a
soon-to-be-destroyed context.

Verified (same scope as VITS's own `test_e2e_vits_driver.cpp` precedent — finite/non-trivial output, not
yet a full hand-rolled pipeline reference, a separately-scoped and much larger undertaking spanning ~8
sub-model references) against the real checkpoint: a 10-token input produces a 22200-sample finite,
non-silent waveform, exercising every real stage in order (`T_text=10→T_frames=37→T_f0=74→T_har=4441`,
matching the algebraic `T_har=T_f0*60+1` derivation exactly).

**Kokoro TTS is now fully driveable end-to-end from the real checkpoint** — 76/76 tests passing. Only
remaining Kokoro-specific gaps: a full hand-rolled pipeline numerical reference (optional; every
sub-piece already independently verified) and a permissively-licensed phonemizer (task #79, shared need
with VITS). Moving on to StyleTTS2 next (likely shares most of this architecture directly), then
SupertonicTTS, F5-TTS, Matcha-TTS.

---

### 2026-07-19: StyleTTS2 (real yl4579/StyleTTS2-LJSpeech checkpoint) — fully driveable end-to-end

Confirmed `/home/flavio/Dev/styletts2` is the real upstream `yl4579/StyleTTS2` repo (remote
`github.com/yl4579/StyleTTS2`, commit `5cedc71`, clean tree) after the user questioned whether the local
copy was legitimate — it is. Downloaded the real pretrained checkpoint from HF
(`yl4579/StyleTTS2-LJSpeech`, MIT, 750MB) plus used the real bundled `Utils/PLBERT/step_1000000.t7`.
Ground truth for the real inference call order: `Demo/Inference_LJSpeech.ipynb`'s own `inference()`
function (a real working demo notebook, extracted via `jupyter nbconvert`) — a stronger source of truth
than Kokoro's own `KModel.forward_with_tokens` was.

**Huge scope reduction discovered before writing any code**: `config.yml` confirmed StyleTTS2's
`hidden_dim/style_dim/n_layer/max_dur/decoder.{upsample_rates,gen_istft_n_fft,...}` are BYTE-IDENTICAL to
Kokoro's own checkpoint (Kokoro is a fork of this exact architecture) — CustomAlbert/PL-BERT,
`bert_encoder`, `TextEncoder`, `DurationEncoder`, F0Ntrain (`AdainResBlk1d` family), Decoder core, and
Generator (istftnet, including SineGen/real-STFT) are ALL reusable VERBATIM against the new checkpoint —
confirmed every real state-dict key matches exactly what the existing `convert_kokoro_*.py` builders
already expect (all under a single uniform `module.` prefix, simpler than Kokoro's own mixed
`""`/`"module"`/`"module.generator"` situation). Wrote `tools/convert_styletts2/convert_styletts2_reused.py`,
which imports Kokoro's conversion scripts directly (via `sys.path`) and re-packs the checkpoint's
`{"net": {...}}` wrapper into a temp bare-dict file matching Kokoro's own convention — ZERO changes needed
to any already-verified Kokoro file. Ran cleanly against the real checkpoint on the first try.

**The one genuinely new piece: the diffusion-based style sampler.** `config.yml`'s `multispeaker: false`
confirmed the LJSpeech checkpoint uses `Transformer1d` (NOT `StyleTransformer1d`) as `KDiffusion`'s
`net` — and critically, `AudioDiffusionConditional`'s own default DEEP MULTI-SCALE U-NET is fully
overridden (`diffusion.unet = transformer` in `build_model`), confirmed directly against the real
checkpoint's state dict (no U-Net-shaped keys at all, only `unet.blocks.{0,1,2}.*`/`to_time`/
`to_mapping`/`to_out`/`fixed_embedding`) — a plain 3-layer transformer, NOT the U-Net the whole
`audio-diffusion-pytorch` machinery suggested at first read. This turned what looked like the biggest
undertaking of the whole model into a modest, well-scoped piece.

Built bottom-up, each piece verified before the next depended on it:
- `loom::karras_schedule`/`loom::adpm2_step`/`loom::adpm2_sample` (`include/loom/core/
  style_diffusion_sampler.h`/`.cpp`) — the Karras-schedule + ancestral-DPM-2 sampling loop, host-driven
  (same "iterative loop calling a graph via callback" shape as `ode_stepper.cpp`'s existing VITS
  precedent), verified against a hand-rolled numpy reference using a TOY affine denoiser with REPLAYED
  (not freshly sampled) noise — decouples "is the discretization math right" from "do two
  independently-implemented RNGs agree" (they never will, by design). Exact match.
- New **`REPEAT`** primitive (`ggml_repeat_4d` wrapper, explicit symbolic target shape, no dummy template
  tensor needed) — required because the diffusion Transformer1d must broadcast a single per-utterance
  256-dim noisy style vector to `[256,T]` before `CONCAT`-ing with the per-position BERT context
  embedding (`CONCAT` itself requires matching shape on every non-concat axis, so the broadcast has to be
  materialized first). Verified in `test_primitive_registry.cpp` (140/140 passing after).
- `Transformer1d` denoiser topology (`tools/convert_styletts2/convert_styletts2_diffusion.py`) +
  `KDiffusion` preconditioning — KDiffusion's own `c_skip`/`c_out`/`c_in`/`c_noise` are DELIBERATELY host
  scalars (sigma is always a plain host-known float per ADPM2 step), not graph nodes — same "host does
  small scalar math" precedent as VITS's SDP/Kokoro's SineGen. A real per-block quirk confirmed from
  source, not simplified away: `Attention`'s Q is normed via `.norm`, K/V via a SEPARATELY-learned
  `.norm_context` applied to the SAME input (no real cross-attention context exists in this
  non-multispeaker config). Verified against REAL checkpoint weights + a hand-rolled PyTorch reference
  (synthetic driving embedding): mean_diff=5.2e-7, max_diff=2.9e-6.
- Combined sampler-loop + real network into the full style-diffusion sample, verified against a
  hand-rolled Python port combining the SAME two independently-verified pieces (fixed seeds, replayed
  noise): mean_diff=1.4e-7, max_diff=6.6e-7.
- `StyleTTS2Driver` (`include/loom/core/styletts2_driver.h`/`.cpp`) — wires CustomAlbert (raw
  `bert_dur`) → style-diffusion sampler (conditioned on RAW `bert_dur`, NOT `bert_encoder`'s projection,
  confirmed from the real demo source) → split `s_pred` into `ref`(decoder style)/`s`(predictor style) →
  `bert_encoder` → `DurationEncoder` → `predictor.lstm`+`duration_proj` → `predict_durations` (with a
  real StyleTTS2-specific quirk confirmed from the demo source: `pred_dur[-1] += 5`, no division by
  `speed` at all unlike Kokoro's own forward) → `expand_by_duration` (both `d` and a separately-computed
  `t_en`) → F0Ntrain → Decoder core → SineGen → forward STFT → Generator core → waveform. Real StyleTTS2
  token convention is a SINGLE LEADING 0 (`tokens.insert(0,0)`), NOT Kokoro's leading+trailing `[0,...,0]`
  — a real, confirmed difference, not an oversight.

Verified end-to-end (`test_e2e_styletts2_driver.cpp`, same finite/non-trivial-output scope as
`test_e2e_kokoro_driver.cpp`) against the real checkpoint on the FIRST real run: `waveform_len=22200`,
all finite, non-silent. Full suite: 100% passing (81/81 executed, rest skip cleanly on missing env vars).

**StyleTTS2 is now fully driveable end-to-end from the real checkpoint.** Remaining gaps (same shape as
Kokoro's own): a full hand-rolled pipeline numerical reference (optional, every sub-piece independently
verified already) and the real phonemizer (task #79, shared with Kokoro/VITS). `style_encoder`/
`predictor_encoder` (reference-audio-driven style, real checkpoint pieces that exist but are never
called by the real demo's own basic-synthesis `inference()`) deliberately deferred, same "basic path
first" precedent as everywhere else. Next per the user's own priority order: SupertonicTTS, F5-TTS,
Matcha-TTS.

---

### 2026-07-19: SupertonicTTS v2 (real femelo/supertonic-tts checkpoint) — foundational pieces + first full sub-model

Started model 5/7 (task #80). Real source read in full at `/home/flavio/Dev/supertonic-tts` (confirmed a
legitimate clone of `github.com/femelo/supertonic-tts`, treated STRICTLY READ-ONLY per explicit user
instruction -- no fetch/pull/push). A genuinely NEW architecture family for this project: conditional
flow-matching (CFM) latent TTS, not descended from the StyleTTS2/Kokoro lineage. Real checkpoint: `.pt`
files under `assets/pt/` (full pickled `nn.Module`s with real weights, not state dicts) -- and critically,
the `supertonic-tts` package is ALREADY `pip install -e`'d in `/home/flavio/.venvs/piper`, so
`torch.load(..., weights_only=False)` gives back REAL modules usable via their own `.forward()` directly
(same "import the real package" precedent as Whisper's own `openai-whisper` use) -- a stronger reference
than hand-copied formulas everywhere this was used.

**Key scope findings** (see `tools/convert_supertonic/PLAN.md` for the full writeup): `DurationPredictor`
predicts a single SCALAR total duration (seconds), not per-token durations -- no CUMSUM/generate_path
needed anywhere in this model, genuinely simpler than every prior TTS model here. The real CFM sampler
(`TextToLatentWrapper.predict`) is a DETERMINISTIC Euler ODE integration (no noise injection per step,
unlike StyleTTS2's ADPM2 sampler). `SpeechDecoder` emits the waveform as a direct flattened sample
sequence from a causal ConvNeXt stack -- no ISTFT/ConvTranspose/SineGen/GAN-upsampling at all, structurally
simpler than every istftnet-family decoder built so far. Precomputed voice styles already exist as JSON
assets (`assets/voice_styles/*.json`), so `SpeechEncoder` (reference-audio style extraction) is out of
scope, same precedent as Kokoro/StyleTTS2's own style/predictor encoders. The real tokenizer
(`TextVectorizer`) is a trivial license-free unicode lookup table -- sidesteps the phonemizer problem
(task #79) entirely for this model family.

**New primitive**: `REPEAT` (`ggml_repeat_4d` wrapper, explicit symbolic target shape) was actually added
during StyleTTS2's own diffusion-sampler work, but turned out to be exactly what SupertonicTTS's
replicate-pad composition needed too -- confirms that generalization was the right call.

**Reused directly, no changes needed**: VITS's own `REL_POS_ATTENTION_SHAW`/`get_relative_embeddings`
primitive family, confirmed the SAME Shaw et al. lookup-table + rel_to_abs/abs_to_rel skew mechanism as
SupertonicTTS's own `MultiHeadRelativeAttention` (different window_size/channels, same algorithm) --
verified via a real numerical cross-check against `duration_predictor.pt`'s own attention layer.

**Foundational pieces built and verified against real weights** (`tools/convert_supertonic/`):
- `ConvNextBlock` (replicate/"edge" padding via VIEW+REPEAT+CONCAT, since ggml has no native replicate-pad
  op) -- both causal and non-causal(symmetric) variants.
- Mish activation (`x*tanh(softplus(x))`, plain composition) + `VFTimeEncoder` (sinusoidal time embedding).
- `StyleCrossAttention`/`StyleEncoderCrossAttention` (style-token-pooling cross-attention: a learnable-query
  first stage, a 2nd stage refining against the same original input -- real quirk: `scale=sqrt(dim)`, the
  KV feature width, NOT `sqrt(head_dim)`).
- Full `DurationPredictor` sub-model (`DPTextEncoder`: ConvNeXt stack + Shaw-et-al. attention +
  sentence-token pooling; MLP head w/ PReLU composed as `relu(x) - weight*relu(-x)`) -- the FIRST complete
  coherent sub-model in this effort, verified end-to-end: exact match (diff=2.4e-7) against the real
  `duration_predictor.pt` + `dp-style-encoder.pt`.

**Two real bugs caught by numerical mismatches, not by inspection** (same discipline as every other
milestone):
1. `MultiHeadRelativeAttention`'s own `x` input axis convention: an earlier version of
   `convert_supertonic_relpos_attn.py` declared `x` directly as Layout B [channels,T] -- WRONG, since the
   real module operates on native PyTorch (B,C,T) channel-first input directly (no internal transpose,
   unlike `StyleCrossAttention`'s own `kv` which DOES need Layout B because ITS real module transposes
   internally first). Fixed by declaring `x` as Layout A [T,channels] (matching the real memory layout
   byte-for-byte) and crossing to Layout B explicitly via PERMUTE+CONT before the Linear ops, crossing
   back afterward.
2. `add_replicate_pad`'s right-pad ("last row") VIEW offset formula: computed as `(T-1)*channels*4` bytes
   -- WRONG (struck a completely different element than intended, a real corruption caught via a genuine
   numerical mismatch on the SYMMETRIC/non-causal padding path specifically -- the causal path never
   exercises `rp>0` at all, which is why it passed cleanly earlier and this bug went unnoticed until the
   first non-causal ConvNeXt use, inside the full `DurationPredictor` assembly). Since `x` has `ne=[T,
   channels]` (T=ne[0], the FASTEST axis), the correct byte offset for the `t=T-1` column is just
   `(T-1)*4` (four bytes per float, no channel multiplication) -- the same default-`nb1`-reuse trick as
   the `t=0` ("first row") case, just at a different flat starting offset along the SAME fastest axis.
   Debugged via small standalone numpy/ggml cross-checks (a 5x3 synthetic example) rather than staring at
   the full DurationPredictor's own 0.13 duration-value mismatch directly -- isolating the smallest
   reproducing piece was much faster than debugging the whole assembly at once.

Next: `SpeechPromptedCrossAttention`/`TTLTextEncoder`/`TTLStyleEncoder`, then the hardest single piece --
`VectorFieldEstimator`'s fractional RoPE cross-attention -- then the Euler CFM sampler, `SpeechDecoder`,
and the final `SupertonicDriver`.

---

### 2026-07-19: SupertonicTTS v2 -- SpeechPromptedCrossAttention/TTLTextEncoder, fractional RoPE, full VectorFieldEstimator

Continued the SupertonicTTS v2 effort (task #80, model 5/7). Built and verified against the real
checkpoint, in order:

- `SpeechPromptedCrossAttention`/`SpeechPromptedTextEncoder` + `TTLTextPreEncoder`/`TTLTextEncoder` +
  `TTLStyleEncoder` (reusing `build_style_encoder`, generalized from the DP-only version once it became
  clear DP/TTL style encoders are structurally identical, only dims differ) -- verified end-to-end
  against real `ttl-style-encoder.pt`/`text_encoder.pt`: mean_diff=4.2e-7.
- `VFTextCrossAttention` (**fractional RoPE** -- `position = index/actual_length`, not integer positions,
  the single most novel piece in this whole model) + `VFStyleCrossAttention`, verified against real
  `vector_estimator.pt`'s own `text_attn[0]`/`style_attn[0]`: both matched to float32 precision
  (mean_diff ~1-6e-8) on the FIRST real numerical test -- the RoPE outer-product-via-MUL_MAT composition
  (generalizing StyleTTS2's own scalar-time trick to a full position VECTOR) worked correctly first try.
- The FULL `VectorFieldEstimator` (4 groups x (4 dilated ConvNeXt + time-conditioning + ConvNeXt +
  text cross-attn + ConvNeXt + style cross-attn) + final ConvNeXt stack, ~355 weight tensors) -- the
  biggest single assembly in this whole project's SupertonicTTS effort, verified against real
  `vector_estimator.pt.compute_velocity()`: mean_diff=5.9e-7 across the entire 24-block chain.

**Three real bugs caught by numerical mismatches** (same discipline as every other milestone):
1. Wrong hyperparameter in MY OWN conversion script: `STYLE_INTERM_DIM` was set to 256 (copy-paste
   confusion with `STYLE_EMBED_DIM`) when the real `ttl-style-encoder.pt` uses 1024 -- caused a
   `ggml_reshape_2d` nelements-mismatch crash. Root-caused by bisecting the crash down to a single
   `ConvNextBlock(dim=256, interm=256)` call and comparing its ACTUAL pwconv1 output shape (`[50,1024,1]`)
   against what was expected -- not a primitive bug at all, a plain wrong constant.
2. `VFStyleCrossAttention`'s learned `key` PARAMETER was erroneously PERMUTE+CONT'd a second time after
   already being registered in the correct Layout B `ne=[stl_dim,n_style]` orientation (a weight
   constant's ggml layout is simply whatever numpy shape it's registered with -- unlike a REAL input
   tensor dumped from a native PyTorch buffer, there is no "native memory layout" to cross away from).
   Caught via `ggml_can_mul_mat` failing immediately (a loud, fast-failing bug, not a silent numerical
   drift).
3. `txt_emb`'s axis convention in the FULL `VectorFieldEstimator` assembly: declared as pre-crossed
   Layout B, but `VFTextCrossAttention.forward` (unlike `VFStyleCrossAttention.forward`, which never
   transposes `stl_emb`) DOES transpose `txt_emb` internally (`txt_seq = txt_emb.transpose(1,2)`) --
   i.e. `txt_emb`'s own real convention is native Layout A, same as `z_t`/`latent`, genuinely DIFFERENT
   from `stl_emb`'s convention despite both being "the other cross-attention operand" superficially.
   Caught via a real 0.2 mean-diff mismatch on the full 24-block assembly; root-caused by bisecting
   layer-by-layer (proj_in -> big_convnext group0 -> time-cond add -> small_convnext1, all matching
   exactly) until the divergence was isolated to exactly the text cross-attention boundary.

Next: the deterministic Euler CFM sampling loop (much simpler than StyleTTS2's own ADPM2 sampler -- no
noise injection at each step, `z_{i+1} = z_i + v(z_i,t_i)*dt` for `n_steps` uniform steps), then
`SpeechDecoder`, then the final `SupertonicDriver`.

---

### 2026-07-19: SupertonicTTS v2 -- Euler CFM sampler, SpeechDecoder, and SupertonicDriver -- COMPLETE

Completed the SupertonicTTS v2 effort (task #80, model 5/7). Final pieces:

- **Euler CFM sampling loop** (`include/loom/core/cfm_euler_sampler.h`/`.cpp`, `loom::cfm_euler_sample`)
  -- deterministic forward-Euler ODE integration (`z += v(z,t)*dt`, no ancestral noise injection at any
  step, much simpler than StyleTTS2's own ADPM2 sampler). Verified against a hand-rolled Python port
  calling the real `vector_estimator.pt`'s own `.solve()` in a loop: exact match on the first try
  (mean_diff=3.4e-7).
- **SpeechDecoder** (folded eval-mode BatchNorm -> per-channel affine, codebook-decompress
  reshape/permute/reshape, causal ConvNeXt stack, direct-waveform-emission head) -- verified against the
  real `vocoder.pt`: mean_diff=1e-7. One real bug caught: `bn_scale`/`bn_shift` were registered as
  DIRECTLY-reshaped 2D numpy weights (subject to GGUFWriter's own axis-reversal convention, giving
  `ne=[hidden_dim,1]` -- backwards from what broadcasting needs), instead of registered 1D and
  RESHAPE'd IN-GRAPH like `lat_std`/`lat_mean` just above it in the same function -- caught via a
  `ggml_can_repeat` assertion, the SAME class of mistake (and fix) already seen for
  `VFStyleCrossAttention`'s own `key` parameter earlier this effort.
- **`SupertonicDriver`** (`include/loom/core/supertonic_driver.h`/`.cpp`) wiring DurationPredictor ->
  `get_latent_mask` (host) -> TTLTextEncoder -> the Euler CFM loop -> SpeechDecoder, using REAL
  precomputed voice-style JSON assets (`assets/voice_styles/F1.json`'s own `style_ttl`/`style_dp`
  fields) -- verified end-to-end against the real checkpoint on the FIRST real run: 70656-sample
  waveform (~1.6s), finite, non-silent.

**A real, confirmed engine architecture limitation surfaced** (exactly what task #80 exists to find):
`loom::GraphBuilder::build(n_tokens, n_past)` resolves EVERY declared graph input's shape via a SINGLE
dynamic-length symbol ("$n_tokens") -- there is no mechanism for a topology to declare a SECOND
independently-sized dynamic input in the same graph. SupertonicTTS's `VectorFieldEstimator` genuinely
needs TWO such lengths (the CFM-iterated latent-frame count, and the input utterance's own phoneme
count, needed simultaneously by `VFTextCrossAttention`) -- the FIRST model in this whole project with
that requirement. Worked around for this milestone by fixing the text length at conversion time
(`T_TEXT=10`, `convert_supertonic_all.py`'s own documented scope choice, consistent with this project's
established "one representative input length, not full dynamic-shape generality" driver-smoke-test
precedent -- Kokoro's/StyleTTS2's own driver tests also use one fixed demo token sequence). A real
production driver would need either a new multi-symbol-per-graph engine mechanism, or per-utterance
topology-JSON templating (a placeholder token substituted via plain string replace before
`GraphTopology::parse`, avoiding any change to `GraphBuilder`'s core symbol resolution) -- not solved
here, flagged for later (relevant to task #81's own "generalize the exporting tool" scope, or a new
tracked task if picked up independently).

**SupertonicTTS v2 is now fully driveable end-to-end from the real checkpoint** (with the above fixed-
text-length limitation, clearly documented). 92/92 tests passing project-wide. Summary of the whole
effort: a genuinely NEW architecture family (conditional flow-matching latent TTS, not descended from
the StyleTTS2/Kokoro lineage) -- reused VITS's own Shaw-et-al. relative-position attention and the
just-added `REPEAT` primitive (from StyleTTS2's own diffusion-sampler work) directly; new composable
pieces: `ConvNextBlock` (w/ a replicate-pad composition, since ggml has no native replicate-pad op),
Mish activation, two DISTINCT style-cross-attention mechanisms (`StyleCrossAttention`/
`SpeechPromptedCrossAttention`, easy to conflate but genuinely different: one pools a KV sequence into
style tokens via a learnable QUERY, the other has TEXT as the query attending over a learnable KEY), and
**fractional RoPE** (`position = index/actual_length`, not integer positions) -- the single most novel
piece, which worked correctly on the FIRST real numerical test. Zero new ggml primitives needed beyond
`REPEAT` (already added for StyleTTS2) -- every other piece was a composition of already-registered
primitives, a first for this project's model-porting roadmap.

Per the user's own priority order: F5-TTS next, then Matcha-TTS (SupertonicTTS was #5 of 7). Overridden
by explicit user direction ("Start Matcha-TTS first") -- Matcha-TTS is now #6, F5-TTS deferred.

## 2026-07-19: Matcha-TTS -- TextEncoder + Decoder U-Net (CFM estimator) verified against real weights

Real source cloned fresh (`github.com/shivammehta25/Matcha-TTS`, commit `bd4d90d`), read in full before
building anything. Real checkpoints (`matcha_ljspeech.ckpt`, LJSpeech single-speaker so no speaker
embedding table exists, and the paired real HiFi-GAN v1 vocoder `generator_v1`) downloaded from
`shivammehta25/Matcha-TTS-checkpoints` GitHub Releases (no HF Hub repo exists for this model). New
dedicated venv `/home/flavio/.venvs/matcha` (CPU torch, `diffusers==0.25.0` pinned against
`huggingface_hub==0.20.3` to dodge the same `cached_download` import conflict already documented for
`transformers`).

**New primitive**: `GROUP_NORM`, wrapping native `ggml_group_norm` (confirmed by direct source read of
`ggml_compute_forward_group_norm_f32`: groups over ne[2], reduces over ne[0]*ne[1]) -- callers reshape
`[T,C]` to `[T,1,C,1]` first (a pure memory reinterpret, no data movement, verified by hand) and back
after; the learned per-channel affine is a separate MUL/ADD, same as every other norm in this project.
Verified via a hand-computed 2-group/4-channel unit test before use.

**TextEncoder** (`ConvReluNorm` prenet + 6-layer self-attention `Encoder` + per-token `DurationPredictor`)
matches the real `matcha.models.components.text_encoder.TextEncoder` module to ~4e-6 on the FIRST real
numerical test. The interesting piece: Matcha's own partial-rotary RoPE (`RotaryPositionalEmbeddings`,
real integer positions, "rotate-half"/NeoX convention, rotating only `int(k_channels*0.5)=48` of the 96
k_channels, the rest passed through unrotated) turned out to be EXACTLY what the existing native `ROPE`
primitive (`mode=2`/NEOX, `n_dims=48`, already used for Qwen3) computes -- confirmed by reading
`ggml_compute_forward_rope_flt`'s own `rotate_pairs`/`theta_scale` code directly (pairs `(ic, ic+n_dims/2)`
for `ic` in `[0,n_dims/2)`, `theta_scale=freq_base^(-2/n_dims)`) before trusting the reuse, rather than
assuming similarity from the name alone. No new RoPE composition needed at all, unlike SupertonicTTS's
own fractional-position RoPE (which DID need a hand-built SIN/COS/MUL composition since ggml's native
`ROPE` only supports integer positions).

**`generate_path`** (Matcha's own duration→alignment construction) confirmed, via direct numerical
comparison against the real `torch` function on a small 4-token example, to degenerate to the exact same
host-side "repeat row t of mu_x, duration[t] times" operation as VITS's own `generate_path` for
single-utterance (unpadded) inference -- `loom::expand_by_duration` (already built) is directly reusable,
no new host code needed.

**Decoder U-Net** (the CFM `estimator`, `matcha/models/components/decoder.py`): real config
`channels=(256,256)` gives a SHALLOW U-Net -- confirmed directly against the real 305-tensor state dict
that only `down_blocks[0]`/`up_blocks[0]` are real stride-2 conv/stride-2-ConvTranspose1d
downsample/upsample; `down_blocks[1]`/`up_blocks[1]` are `is_last`, plain same-resolution convs, despite
each `nn.ModuleList` nominally having 2 entries. Scope decision (mirrors SupertonicTTS's own `T_TEXT`
choice): the driver always sizes mel-frame count to an exact multiple of 4 (real `fix_len_compatibility`'s
own default), so `$n_tokens/2` (SymbolEnv supports division/floor in expressions, confirmed by reading
`symbol_env.cpp`) is always exact and all padding-mask handling drops out (mask is always all-ones for a
single, unpadded utterance). `BasicTransformerBlock` (imported from `diffusers` upstream, real config has
`cross_attention_dim=None` so `attn2`/`norm2` don't exist) reduces to: standard multi-head self-attention
(bias-free q/k/v, matching `diffusers.Attention`'s own default) + a `SnakeBeta`-activated FeedForward
(`x + (1/exp(beta))*sin(x*exp(alpha))^2`, log-scale learned alpha/beta) -- no cross-attention at all. Both
reused directly: `ATTENTION` (`kv_cache:false`, same pattern as Kokoro's own ALBERT encoder) and a small
new SnakeBeta composition (EXP/SIN/SQR/DIV/ADD, no new primitive).

**Two real bugs found via numerical mismatch, not inspection** (both worth remembering for the NEXT
model that mixes `[T,C]`/`[C,T]`-convention tensors):
1. `SnakeBeta`'s `alpha`/`beta` (raw shape `(inner_dim,)`) were incorrectly `RESHAPE`'d to `[1,inner_dim]`
   -- correct only for `[T,C]`-convention broadcasts (bias-adds against `ne[1]=C`); `SnakeBeta` operates
   on a CHANNEL-FIRST `[C,T]` tensor (`C=ne[0]`), where the raw un-reshaped weight already has the right
   `ne[0]=C` alignment (same as every other channel-first norm/bias in this project) -- the reshape was
   simply wrong, caught immediately by ggml's own `ggml_can_repeat` assertion (a crash, not silent
   corruption).
2. **The real "gotcha" of this whole milestone**: a `[T,C]`-convention tensor's ggml flat layout
   (`idx = t + c*T`, `T=ne[0]` fastest) and a numpy `(C,T)`-shaped row-major reference array's own flat
   layout (`idx = c*T + t`) are the SAME FORMULA (addition commutes) -- meaning `[T,C]`-convention tensors
   need NO reindexing at all when compared against/fed from a `(C,T)`-shaped `.npy` reference, unlike
   `[C,T]`-convention (channel-first) tensors, which DO need reindexing (different formula,
   `idx=c+t*C` vs `idx=c*T+t`). A test written by copy-pasting the channel-first reindexing pattern onto a
   `[T,C]`-convention tensor (both the INPUT-feeding side and, separately, the OUTPUT-comparison side)
   introduced two compensating-looking-but-wrong permutations that produced a large, confusing numerical
   mismatch (`max_abs_diff≈3.8`) despite the underlying ggml graph being 100% correct end-to-end --
   isolated via a from-scratch numpy reimplementation of the whole topology (matched the real reference to
   ~1e-5 immediately) plus a per-stage checkpoint-dumping harness (confirmed every intermediate ggml
   tensor correct), which together proved the bug had to be in the C++ test's own data marshalling, not
   the conversion script. Lesson: for `[T,C]`-convention tensors specifically, prefer a raw flat-buffer
   copy/compare over any explicit reindexing loop -- the reindexing is not just unnecessary, it's wrong.

Both `TextEncoder` (`matcha_encoder_mu.gguf`/`matcha_encoder_logw.gguf`) and the `Decoder` U-Net
(`matcha_decoder.gguf`) verified against real checkpoint weights.

## 2026-07-19: Matcha-TTS COMPLETE -- HiFi-GAN v1 vocoder + MatchaDriver, 96/96 tests passing

Finished the same day. The real HiFi-GAN v1 vocoder (`generator_v1`, paired via `matcha/cli.py`'s
`VOCODER_URLS`) converted and verified against `matcha.hifigan.models.Generator` run directly, matching
to ~4e-5 on the FIRST real numerical test (no debugging needed -- the `[T,C]`-convention flat-buffer
lesson from the Decoder milestone was applied correctly from the start this time). Real config
(`resblock="1"`, `upsample_rates=[8,8,2,2]`, `upsample_kernel_sizes=[16,16,4,4]`,
`upsample_initial_channel=512`, `resblock_kernel_sizes=[3,7,11]` w/ dilations `(1,3,5)` for `convs1`
only -- `convs2` always dilation=1) confirmed against both `matcha/hifigan/config.py`'s `v1` dict and the
real 104-tensor state dict -- genuinely different topology shape from VITS-piper's own `resblock="2"`/
3-stage HiFi-GAN (single `convs` list, no `convs2`), but needed zero new primitives (`CONV_1D`/
`CONV_TRANSPOSE_1D`/`LEAKY_RELU`/`TANH`/`ADD`, all already proven). One easy-to-miss real detail caught by
re-reading the source rather than assuming: `Generator.forward`'s FINAL `leaky_relu` (right before
`conv_post`) calls `F.leaky_relu(x)` with NO explicit slope, i.e. PyTorch's default `negative_slope=0.01`,
NOT the `0.1` used everywhere else in the same function -- VITS's own converter already had this right,
carried over deliberately rather than re-derived from scratch.

`loom::MatchaDriver` (`include/loom/core/matcha_driver.h`/`src/core/matcha_driver.cpp`) assembles the full
real call order (`MatchaTTS.synthesise()`): TextEncoder (`mu_x`,`logw`) -> per-token durations
(`ceil(exp(logw))`, real `length_scale=1.0`) -> row-repeat duration expansion (`loom::expand_by_duration`,
confirmed degenerate `generate_path`) -> `loom::cfm_euler_sample` (built for SupertonicTTS, reused
VERBATIM -- `solve_euler`'s own loop, `x=x+dt*dphi_dt` with uniform `dt`, is structurally identical) over
the new Decoder U-Net estimator -> denormalize (`mel_mean=-5.536622`,`mel_std=2.116101`, both confirmed
directly from the checkpoint's own 0-d `mel_mean`/`mel_std` tensors, agreeing with
`hyper_parameters['data_statistics']`) -> HiFi-GAN v1 vocoder -> waveform. Real, documented scope choice
(mirrors SupertonicTTS's `T_TEXT` precedent): since the Decoder topology drops ALL padding-mask handling
(valid only when every mel frame is genuinely non-padding), `synthesize()` EXTENDS the last token's own
predicted duration until the total mel-frame count is an exact multiple of 4 (real
`fix_len_compatibility`'s own default requirement) -- a principled choice (every frame stays a genuine
attended `mu_y` row, nothing is ever contaminated padding) rather than reimplementing real masking.
`MatchaDriver::synthesize()` verified end-to-end on the FIRST real run: 10240-sample waveform (~0.46s at
22050Hz) from an 8-token demo input, finite and non-silent.

**Matcha-TTS is now fully driveable end-to-end from the real checkpoint.** 96/96 tests passing
project-wide. Summary of the whole effort: a genuinely different architecture family within the
Grad-TTS/GlowTTS lineage (conditional-flow-matching mel-spectrogram TTS + separate vocoder, vs VITS's own
normalizing-flow latent + integrated vocoder) -- reused VITS's own glow-tts-derived custom LayerNorm
composition, the native `ROPE` primitive (already built for Qwen3, turned out to reproduce Matcha's own
partial-rotary integer-position RoPE exactly once verified against `ggml`'s own `rotate_pairs`/
`theta_scale` code), `ATTENTION` with `kv_cache:false` (Kokoro's ALBERT precedent), `loom::
expand_by_duration` (VITS/Kokoro/StyleTTS2's own degenerate-`generate_path` precedent), and
`loom::cfm_euler_sample` (SupertonicTTS's own CFM sampler, reused verbatim). One new primitive
(`GROUP_NORM`, wrapping native `ggml_group_norm`) and one new composition (`SnakeBeta`, log-scale
learned-frequency activation). The real lesson of this whole milestone, worth carrying into every future
model: `[T,C]`-convention (`T=ne[0]`) ggml tensors need NO reindexing when fed from/compared against a
`(C,T)`-shaped row-major `.npy` reference (the flat-index formulas are identical since addition commutes)
-- unlike `[C,T]`-convention (channel-first) tensors, which DO. A test written by reflexively copying the
channel-first reindexing pattern onto a `[T,C]`-convention tensor cost real debugging time despite the
underlying conversion script being correct from the start.

Per the user's own priority order (as last revised): F5-TTS is next and is the final model in the
originally-specified list (Whisper v3 → FastConformer RNN-T → Kokoro → StyleTTS2 → SupertonicTTS →
Matcha-TTS → F5-TTS).

## 2026-07-19: PAUSED — pivoting to procedural generalization (embedded Lua + MIL-based compiler)

Explicit user direction: stop the model-porting roadmap and every other open task tracker item, and
tackle a foundational architecture problem first. Two design docs now govern the next phase of work:
`LOOM_PROCEDURAL_GENERALIZATION.md` (top-level rationale: replace bespoke C++ drivers with an embedded
LuaJIT orchestration layer + an offline PyTorch-to-Loom compiler) and `LOOM_MIL_CONVERSION.md` (detailed
spec: use `coremltools`'s MIL IR as the offline compiler frontend, translate MIL `main`-function control
flow to a transpiled Lua driver script embedded in the GGUF, and lower heavy submodule functions to
static `graph_topology` JSON + block-quantized weights, mirroring Apple's own PyTorch→MIL→ANE pipeline).
Core idea: every model family ported so far (Whisper, VITS, Kokoro, StyleTTS2, SupertonicTTS, Matcha-TTS)
needed a bespoke hand-written C++ driver (`*_driver.cpp`) to handle multi-subgraph orchestration,
autoregressive/ODE control flow, and host-side math (mask generation, duration expansion, Euler
sampling, etc.) — this doesn't scale, and the goal now is to make that orchestration layer
data-driven/scriptable instead of requiring new C++ per model.

The following task-tracker items were **unqueued** (removed from active tracking) and are deferred here
until the foundational work lands; none are abandoned, just paused:

- **VITS: integrate a permissively-licensed phonemizer.** Real `espeak-ng` vendoring was already
  rejected (GPL-3, see [[loom_engine_licensing_phonemizer]] memory) — plan was `phoonnx` for VITS's
  phonemizer instead. Also relevant to Kokoro/StyleTTS2/Matcha-TTS's own text-frontend gaps (none of
  these drivers do real phonemization yet, all use raw token-id demo inputs).
- **Add new model families to force out missing primitives** (task #80, was in_progress). Whisper v3,
  FastConformer RNN-T, Kokoro, StyleTTS2, SupertonicTTS, and Matcha-TTS are all COMPLETE (each with a
  hand-written C++ driver, per the very problem this pivot addresses). **F5-TTS** was next and is now
  deferred — the last model in the original 7-model priority list. Once the Lua/MIL architecture lands,
  F5-TTS (and potentially the completed models too) should be portable through the NEW pipeline instead
  of getting yet another hand-written driver, which would help validate the new architecture end-to-end.
- **Evaluate generalizing `aten_to_loom.py` into a real exporting tool.** Directly superseded in spirit
  by `LOOM_MIL_CONVERSION.md`'s own compiler spec (`tools/codegen/compile_pytorch_to_loom.py` /
  `LoomGGUFExporter`) — worth revisiting `aten_to_loom.py`'s existing op-mapping code as a reference/
  starting point when building the new MIL-based exporter, rather than starting from zero.
- **Make quantization a general process/tool, including KV-cache quantization.** Still relevant
  independent of the Lua/MIL pivot — the new compiler's own "block-quantized weights" step
  (`LOOM_MIL_CONVERSION.md` §2, `self.weights` → GGUF) will need this either way.
- **Add primitives for modern attention variants, prioritizing flash attention** (`ggml_flash_attn_ext`).
  Still relevant independent of the pivot — a primitive-level improvement, orthogonal to the
  orchestration-layer rewrite.

See [[loom_engine_procedural_generalization_roadmap]] memory (already tracked LOOM_PROCEDURAL_GENERALIZATION.md
as the user's own Lua-embedding design, previously deferred until model-porting finished — that
"finished" condition is now considered satisfied enough to start, six of seven models done, per this
explicit pivot).

## 2026-07-19: LuaJIT embedded + procedural-generalization architecture proven against WhisperDriver

C++-side half of the pivot landed (the Python MIL compiler in `LOOM_MIL_CONVERSION.md` remains
deliberately out of scope, deferred until this runtime side was proven). User chose LuaJIT (not PUC-Rio
Lua 5.4, despite the non-CMake build risk) and `WhisperDriver` as the port target specifically because
its autoregressive while-loop + persistent `KvCache` + argmax sampling is the hardest case this
architecture needs to solve, not the easy "just chain some subgraphs" case.

**LuaJIT vendoring** (`cmake/Dependencies.cmake`): `FetchContent_Populate` (source only -- LuaJIT has no
upstream CMakeLists) + a custom command driving LuaJIT's own `Makefile` (`BUILDMODE=static
XCFLAGS=-fPIC`), wrapped as an `IMPORTED STATIC` target `luajit::luajit`. Two real build snags found and
fixed by actually running the build rather than assuming: (1) the primary GitHub-clone approach worked
cleanly on the first try (confirmed by manually cloning+building outside CMake before writing any glue,
per the plan's own risk-first ordering) -- the documented `LOOM_USE_SYSTEM_LUAJIT` fallback was never
needed; (2) the static lib link failed against `libloom_engine.so` with `relocation R_X86_64_TPOFF32 ...
can not be used when making a shared object` until `XCFLAGS=-fPIC` was added to the `make` invocation --
LuaJIT's own default build isn't position-independent, and `loom_engine` is a shared library.

**`LoomLuaBridge`** (`include/loom/core/lua_bridge.h`/`src/core/lua_bridge.cpp`): owns one `lua_State*` +
a name→module registry (`GgufModel&`/`GraphTopology`/optional `KvCache*`, non-owning references, same
convention as `GraphBuilder` itself). Bindings: `loom.run_subgraph(module_name, n_tokens, n_past,
inputs_table)` (dispatches F32/I32 marshalling by inspecting the REAL declared input tensor's own
`->type` after `build()` -- no per-input type annotation needed in Lua, the topology is the single
source of truth; returns the flat output array PLUS its ggml shape `[ne0,ne1,ne2,ne3]` as a second
return value, added specifically so a driver script can read `n_vocab` off logits output without
hardcoding it), `loom.range`, `loom.causal_mask` (ported verbatim from `WhisperDriver::
fill_decoder_inputs`'s own formula), `loom.zero_mask`, `loom.argmax_row` (ported from `WhisperDriver::
argmax`, row-index parameterized so a script can select "last prompt token" during prefill vs "the only
row" during incremental decode, matching `transcribe()`'s own two call sites). A `Value =
std::variant<double, std::vector<double>>` covers everything a driver script's `call()` args/return need
(one deliberate simplification from the approved plan's own `double/vector<float>/vector<int32_t>/
string` sketch -- Lua 5.1/LuaJIT has exactly one numeric type, a double, so a separate float/int
distinction added nothing; the string case was never needed by Whisper's own script and was dropped).

**A real, load-bearing correctness discipline, not just an implementation detail**: LuaJIT (like PUC Lua
5.1, whose C API it implements) reports its own internal errors via `longjmp`, which does NOT run C++
destructors -- letting a C++ exception unwind through a `lua_CFunction` reached via `lua_pcall` is
undefined behavior on this build (LuaJIT's default, non-"external unwind" configuration). Every binding
trampoline in `lua_bridge.cpp` therefore wraps its entire body in try/catch and converts any
`loom::Error`/`std::exception` into `luaL_error(...)` -- itself Lua's own longjmp-based error path, which
`lua_pcall` at the call site safely catches. Never remove this when adding new bindings.

**`tools/convert_whisper/whisper_driver.lua`**: a direct line-for-line port of `WhisperDriver::
transcribe()` (encoder pass, then prefill via one `run_subgraph` call, then a `while` loop doing
incremental decode steps, checking `eot_token`). Embedded into the EXISTING `whisper_decoder.gguf` via
one added `w.add_string("model.driver_script", ...)` call in `convert_whisper_decoder.py` -- confirmed
during planning (and now confirmed working) that this needed ZERO `GgufModel`/GGUF-reading engine
changes at all, since `GgufModel::kv_str(full_key)` was already a fully generic string-KV accessor
(`include/loom/core/gguf_model.h`). The Task 1 "namespaced multiple topologies in one GGUF file" idea
from `LOOM_PROCEDURAL_GENERALIZATION.md` was NOT needed either -- Whisper's existing two-separate-GGUF-
files convention (encoder/decoder) was reused as-is, with the bridge just registering each as its own
named module.

**Validation, `tests/test_e2e_whisper_lua_driver.cpp`**: runs the SAME real `tiny.en` Whisper checkpoint's
`transcribe()` through both the existing hand-written `loom::WhisperDriver` (C++ control flow) and the
new `LoomLuaBridge`-driven `whisper_driver.lua`, on identical inputs, and asserts an EXACT generated-
token-sequence match (both are deterministic greedy argmax decoding, so no floating-point tolerance
question). **Matched exactly on the FIRST real run**: `[357, 7050, 2491, 8, 50256]` from both paths. Also
added `tests/test_lua_bridge.cpp`, a fast dependency-free unit test of the four host-math bindings in
isolation (hand-computed expected values, same discipline as `test_primitive_registry.cpp`) -- also
passed on the first run. 98/98 tests passing project-wide.

**What's proven, and what's explicitly still deferred**: this proves the RUNTIME mechanism end-to-end on
the hardest existing case (autoregressive loop + persistent KV-cache state threaded through Lua-driven
control flow, exactly matching bit-for-bit what hand-written C++ already did). Still deferred, per the
approved plan: the Python MIL compiler (`LOOM_MIL_CONVERSION.md` in full -- `whisper_driver.lua` was
HAND-written, not machine-transpiled from a real PyTorch model via `coremltools`), packing multiple named
topologies into one GGUF file, porting any of the other five drivers (VITS/Kokoro/StyleTTS2/
SupertonicTTS/Matcha-TTS) to Lua, and removing/deprecating `WhisperDriver` itself (it stays as both the
correctness oracle this path is checked against and a fallback). F5-TTS also remains deferred from the
earlier pivot.

## 2026-07-19: Whisper's encoder + decoder packed into ONE GGUF file (named multi-topology `GgufModel`)

Immediate user follow-up to the LuaJIT milestone above: "all modules... added to the same gguf file" is
the actual end-state `LOOM_PROCEDURAL_GENERALIZATION.md`/`LOOM_MIL_CONVERSION.md` are aiming for (one
GGUF = one deployable model artifact), so the deferred "multiple namespaced topologies in one GGUF file"
part of Task 1 was picked back up rather than left parked.

**`GgufModel` extended, additively** (`include/loom/core/gguf_model.h`/`src/core/gguf_model.cpp`):
`load()` used to hard-require exactly one bare `model.graph_topology` string KV; it now scans every KV
(reusing `hparam_env()`'s own "loop `gguf_get_n_kv`/`gguf_get_key`, check a prefix" pattern) for the bare
key (stored under `""`) AND any `model.graph_topology.<name>` keys (stored under `"<name>"`), throwing
the SAME `LoadError` as before only if NEITHER is found. `topology_json()` (no args) is completely
unchanged in behavior -- confirmed via `grep` to be called at **60+ sites** across nearly every test in
the project, so this had to be strictly backward compatible, not just "probably fine." New
`topology_json(name)`/`has_topology(name)` overloads. Directly unit-tested by extending the existing
minimal-fixture test (`tests/fixtures/make_minimal_gguf.py` + `tests/test_gguf_model_load.cpp`) with a
second, named topology KV alongside the bare one.

**Real fact confirmed before merging any weights** (not assumed): Whisper's real checkpoint already
names its encoder/decoder weights with the real PyTorch module's own hierarchical prefixes
(`encoder.blocks.{i}...`, `decoder.blocks.{i}...`, `decoder.token_embedding...`, `mel.*`) -- zero
collisions merging both weight dicts into one flat GGUF tensor namespace. This generalizes: every
converter in this whole project already mirrors the real checkpoint's own module names, so no
collision-avoidance machinery (auto-namespacing, prefix enforcement) was added -- would only be worth
building if a future model actually collides.

**New `tools/convert_whisper/convert_whisper_all.py`**: reuses `build_encoder`/`build_decoder`
UNCHANGED (imported from the existing per-module scripts, mirroring the `convert_X_all.py`-assembles-
already-validated-pieces precedent from Kokoro/SupertonicTTS), builds each into its own `TopologyBuilder`,
merges the weight dicts, and writes ONE `whisper.gguf` with `model.graph_topology.encoder`,
`model.graph_topology.decoder`, `model.driver_script` (moved here from the old
`convert_whisper_decoder.py` writer -- the script orchestrates BOTH modules, so it belongs on the
combined file, not tucked inside just one half), and every merged tensor. The OLD two-separate-file
scripts (`convert_whisper_encoder.py`/`convert_whisper_decoder.py`) are UNCHANGED (`model.driver_script`
just removed from the decoder writer) and still back the per-module isolation reference tests
(`test_e2e_whisper_{encoder,decoder}_reference.cpp`), which don't need the combined file at all.

**`test_e2e_whisper_driver.cpp`/`test_e2e_whisper_lua_driver.cpp`** updated to load ONE `GgufModel` from
`whisper.gguf` and read `topology_json("encoder")`/`topology_json("decoder")` off it -- neither
`WhisperDriver`'s constructor nor `LoomLuaBridge::register_module` needed ANY signature change, since
both already accepted independent `GgufModel&`/topology pairs; both call sites just pass the SAME loaded
model twice now. Re-verified end-to-end: identical `[357, 7050, 2491, 8, 50256]` token sequence from both
the C++ and Lua paths against the new single-file `whisper.gguf`, exactly as before the consolidation.
98/98 tests passing project-wide (including the untouched per-module reference tests, confirming the
`convert_whisper_decoder.py` edit didn't regress them).

**Still explicitly out of scope**: retrofitting VITS/Kokoro/StyleTTS2/SupertonicTTS/Matcha-TTS to the
same one-GGUF-file convention (five more conversion-script consolidations, real future work, not
attempted here) and the Python MIL compiler.

## 2026-07-20: SupertonicTTS + Matcha-TTS + VITS retrofitted to embedded Lua + one-GGUF-file (Kokoro/StyleTTS2 still deferred)

Direct follow-up to the two entries above: "implement the retrofit of the other models' drivers to
embedded lua to the one-file gguf." Re-read every driver's real header/impl before committing to scope
(not from memory) — confirmed a real ~5x complexity spread across the five remaining drivers (see the
approved plan for the full breakdown) and did the three tractable ones with full verification rigor
(SupertonicTTS → Matcha-TTS → VITS, ascending complexity), leaving Kokoro/StyleTTS2 (StyleTTS2 alone owns
~20 separate `GgufModel`/`GraphTopology` pairs plus the ADPM2 diffusion sampler's own 2-network-
evaluation-per-step math) as an explicitly-scoped next pass rather than rushing an under-tested
consolidation.

**`LoomLuaBridge` grew five new bindings**, all thin wrappers around already-existing, already-verified
host C++ (no new algorithms):
- `loom.seed_rng(seed)`/`loom.gaussian_array(n)` — a `std::mt19937`+`std::normal_distribution<float>(0,1)`
  owned by the bridge itself, persisting across calls within one script invocation (same "persistent
  state" shape as a registered `KvCache`). Using the EXACT SAME engine/distribution shape every
  hand-written driver's own RNG already uses is what kept exact-match testing possible.
- `loom.expand_by_duration(rows_flat,T,C,durations)` — wraps `loom::expand_by_duration`
  (`duration_aligner.h`), already proven by Matcha-TTS's own C++ driver.
- `loom.pad_crop_relative_embeddings(raw,window_size,k_channels,length)` — wraps VITS's own
  `pad_crop_relative_embeddings`, MOVED out of `vits_driver.cpp` into a new shared
  `include/loom/core/relative_position.h` (SupertonicTTS's own `MultiHeadRelativeAttention` uses the same
  Shaw-et-al. mechanism and may want this too).
- `loom.get_weight(module_name, weight_name)` — a genuinely NEW capability not anticipated in the
  approved plan's own binding list, discovered while actually writing VITS's Lua script: the real driver
  reads its relative-position tables directly off the GGUF weight table (`GgufModel::weight()`), not
  through any topology's declared inputs/outputs — `loom.run_subgraph` alone couldn't express that, so a
  direct raw-weight-introspection binding was added.

All four covered by `tests/test_lua_bridge.cpp`'s own hand-computed-expectation unit tests (including
cross-checking `loom.gaussian_array` against an INDEPENDENT `std::mt19937`/`std::normal_distribution`
instance constructed directly in the test, not hand-transcribed magic numbers) before use in any real
model, same discipline as every primitive this whole project has added.

**A real, useful discovery while merging weights into one file**: Matcha-TTS's own `mu`/`logw`
topologies (and VITS's own `stats`/`logw`) each independently rebuild their shared TextEncoder from
scratch in separate `TopologyBuilder`s (an established precedent from when they were separate files —
`GraphTopology` supports only one declared output each). Naively merging their weight dicts hits a
same-name pseudo-"collision" for every duplicated TextEncoder weight — NOT a real bug (same underlying
checkpoint tensor, redundantly registered twice with byte-identical values). Both `convert_matcha_lua_all.py`
and `convert_vits_lua_all.py`'s own `merge()` helper is content-aware: identical-name+identical-value is a
silent dedup, identical-name+DIFFERENT-value is a hard `assert` failure — confirmed the real files hit
only the harmless case (VITS: 469 tensors across 3 files → 358 after dedup, exactly matching hand
arithmetic; Matcha-TTS: 577 tensors across 4 files → 466 after dedup).

Each model's own new `<model>_driver.lua` is a direct line-for-line translation of its existing C++
driver's `synthesize()`, reusing the SAME "two Euler-ODE-loop drivers, one relative-position-table
driver" complexity gradient identified in the plan:
- **SupertonicTTS** (`tools/convert_supertonic/convert_supertonic_lua_all.py`/`supertonic_driver.lua`):
  merges only the FOUR topologies the driver actually uses (`dp`/`ttl_text`/`vfe`/`decoder` — the two
  style-encoder topologies are never called by `synthesize()`, matching the existing "precomputed style
  embeddings" scope decision, so they're correctly excluded from the merge). Euler loop is a trivial
  `for step=0,n_steps-1 do local v = loom.run_subgraph("vfe",...); z[i]=z[i]+v[i]*dt end`. Matched the
  real C++ driver to **1.4e-6** on the first real run (real checkpoint, F1 voice style).
- **Matcha-TTS** (`tools/convert_matcha/convert_matcha_lua_all.py`/`matcha_driver.lua`): real per-token
  duration expansion via `loom.expand_by_duration` — TextEncoder's own channel-first `mu_x` output flat
  layout turned out to ALREADY be `expand_by_duration`'s own expected "T rows of C contiguous floats"
  convention with zero transformation needed (channel-first `[C,T]`'s own flat index `c+t*C` IS row-major
  T-slow/C-fast), though the EXPANDED result then needs an explicit Lua-side transpose into the Decoder's
  own `[T,C]` (T-fastest) input convention — same "know which convention you're in" discipline the
  `[T,C]`-vs-`[C,T]` debugging lesson from the original Matcha-TTS port established. Matched to **2.7e-6**
  on the first real run.
- **VITS** (`tools/convert_piper_vits/convert_vits_lua_all.py`/`vits_driver.lua`): the hardest of the
  three — `loom.get_weight`+`loom.pad_crop_relative_embeddings` called per text-encoder layer (6 layers ×
  2 tables), PLUS Gaussian-noise-interleaved duration expansion (`z_p` construction combines `stats`'s
  own `m_p`/`logs_p` with a per-frame-per-channel noise draw and an `exp()`-scaled affine, not a plain
  row-repeat). Verified the RNG DRAW ORDER concern explicitly before trusting it: `loom.gaussian_array`
  bulk-drawing `y_length*inter_channels` values upfront, then indexing sequentially in the exact same
  (frame-major, channel-minor) nested-loop order the C++ driver's own INTERLEAVED per-element
  `normal(rng)` calls use, produces the IDENTICAL sequence (a `std::normal_distribution`'s only state is
  the engine + its own internal cache, both advanced identically either way) — reasoned through before
  writing the code, not discovered by trial and error. Matched the real T=62-token piper checkpoint
  (`en-GB miro`) to **5.8e-7** on the first real run — the hardest port of the three, correct on the
  first try.

101/101 tests passing project-wide. **Kokoro and StyleTTS2 remain the explicitly deferred next pass**
(their own real complexity: StyleTTS2 alone owns on the order of twenty separate GGUF files/topologies
and the ADPM2 diffusion sampler's own per-step math needs careful decomposition into Lua-callable pieces
— re-confirmed `BiLstmStepper` itself needs NO new binding, since it already ports directly to a Lua loop
calling the existing `loom.run_subgraph` four times per timestep per direction). The Python MIL compiler
remains deferred from the original pivot.

## 2026-07-20: Kokoro + StyleTTS2 retrofitted to embedded Lua + one-GGUF-file — ALL SIX real models now
ported; only the Python MIL compiler remains

The deferred pass landed. Both drivers re-read directly from source (not memory) before starting, per
this project's own "verify against real behavior, not guesses" discipline — Kokoro turned out to own 31
`GraphTopology` instances / ~28 GGUF files (6 BiLSTM instances × 4 + 15 single-shot modules + 3 AdaLN
blocks), StyleTTS2 owns 45 graphs / 27 kept `GgufModel`s (the same BiLSTM shape, plus the style-diffusion
sampler) — both larger than the earlier "~20" estimate, but NOT architecturally harder than what VITS/
Matcha-TTS already proved: comparing each driver's full control flow against `LoomLuaBridge`'s existing
binding set showed exactly **one** new binding was needed for both models combined —
`loom.uniform_array(n)` (a `std::uniform_real_distribution<float>` sharing the bridge's existing `rng_`
stream with `gaussian_array`, needed by SineGen's `rand_ini`, which both drivers draw as
uniform-then-gaussian from the SAME shared stream). Even the ADPM2 diffusion sampler and the 6 BiLSTM
instances' per-timestep host-carried h/c stepping — the two pieces flagged as hardest in the original
scoping — turned out to be expressible with `loom.run_subgraph` in a Lua loop plus plain arithmetic, no
new binding required.

**Real non-Lua-bridge work found while building Kokoro's consolidator**: Kokoro's shared conversion
helpers (`write_bilstm_ggufs` in `tools/convert_kokoro/convert_kokoro_duration_predictor.py`, the
AdaLayerNorm writer, `write_proj1x1` in `convert_kokoro_f0n.py`) all historically wrote GENERIC,
non-namespaced weight names (`lstm.weight_ih`, `adaln.fc.weight`, `proj.weight`) — harmless when every
instance lived in its own isolated GGUF file, but a real collision once merged into one file, since 6
distinct BiLSTM instances / 3 AdaLN blocks / 2 proj heads share these names while holding GENUINELY
DIFFERENT real weight values (unlike Matcha/VITS's own TextEncoder-dedup collisions, which were the SAME
values redundantly registered twice). Fixed by adding an optional `weight_namespace`/`weight_prefix`
parameter to each builder (`build_bilstm`, `build_adaln`, `build_proj1x1`, `build_duration_proj`,
`build_stack`, `build_cnn`), defaulting to the old hardcoded names so every existing separate-file
conversion script and its tests are completely unaffected — purely additive. Re-ran the OLD conversion
scripts (`convert_kokoro_all.py`, `convert_styletts2_reused.py`) against the real checkpoints after this
refactor and confirmed zero regressions: `test_e2e_kokoro_text_encoder`/`_duration_predictor`/`_f0n`
(numeric reference tests) and `test_e2e_kokoro_driver`/`test_e2e_styletts2_driver` all still pass.

`tools/convert_kokoro/convert_kokoro_lua_all.py`/`kokoro_driver.lua`: 43 topologies (30 unique names
after the BiLSTM 4-per-instance/AdaLN/proj expansion), skips `kokoro_stft_inverse` (confirmed dead —
loaded by the C++ constructor, never called in `synthesize()`). `predict_durations`'s round-half-to-even
needed a hand-written Lua helper (no native banker's-rounding builtin) — `duration_aligner.cpp`'s own
comment notes real float32 sums essentially never land on an exact `.5` tie, so this was implemented
correctly rather than assumed away, not exercised by the real checkpoint either. Matched
`loom::KokoroDriver` to **1.9e-6** on the first real run (real `kokoro-v1_0.pth`, synthetic `ref_s`,
`speed=1.0`, `seed=42`) — the largest port so far, correct on the first try. 102/102 tests passing after
landing.

`tools/convert_styletts2/convert_styletts2_lua_all.py`/`styletts2_driver.lua`: 44 topologies, reuses the
now-namespace-capable Kokoro builders the same way `convert_styletts2_reused.py` already does (same
import pattern, StyleTTS2's own real checkpoint weights) plus `convert_styletts2_diffusion.py`'s
`build_diffusion_net` (the one genuinely new piece — no Kokoro equivalent). The ADPM2 sampler
(`karras_schedule`/`adpm2_step`/`adpm2_sample`) and KDiffusion preconditioning (`c_skip`/`c_out`/`c_in`/
`c_noise`) are plain Lua arithmetic mirroring `style_diffusion_sampler.cpp` line-for-line, with
`loom.gaussian_array` supplying every noise draw (initial `noise0` + one ancestral draw per ADPM2 step)
from the SAME shared `rng_` stream `loom.uniform_array` later draws SineGen's `rand_ini` from — same
ordering discipline as VITS's own interleaved-draw reasoning. Also carries the real
`pred_dur[-1] += 5` quirk (`Demo/Inference_LJSpeech.ipynb`'s own `inference()`, no `/speed` division at
all, unlike Kokoro's own `forward_with_tokens`).

**This one didn't match to ~1e-6 like every other port, and that was chased down rather than shrugged
off**: the full waveform only matched to `max_abs_diff≈3.1e-3`. Spent real effort isolating why before
accepting it, in order: (1) `albert` alone — exact, 0.0 diff; (2) the diffusion sampler ALONE, given the
identical `bert_out`/seed — matched to `~7e-7`/`~1.6e-6` at 5 and 2 steps respectively; (3) `pred_dur`
(all 9 integer values) and `s_predictor` (float) from a faithful partial reimplementation stopping right
after duration prediction — exact/near-exact match; (4) `text_encoder_cnn`, `decoder_core`, `sinegen`,
`generator` — each EXACTLY 0.0 diff given identical synthetic inputs; (5) every weight tensor in every
GGUF file (`decoder_core`: 63, `generator`: 348, `sinegen`: 5, plus text_encoder/f0n_shared/f0n block/proj
BiLSTM+AdaLN weights) — byte-identical between the old and new conversion paths, confirmed via direct
`np.array_equal` comparison, not spot-checked. Even tried explicitly truncating the diffusion sampler's
own host math to float32 at every intermediate step via LuaJIT's FFI (mirroring `style_diffusion_sampler
.cpp`'s `float` semantics line-for-line) — this did NOT close the gap (0.00312568 → 0.00334246, i.e. no
real change), which rules out "fixable Lua-double-vs-C++-float host-math imprecision" as the cause and
was reverted (added real complexity for zero measured benefit). Conclusion: StyleTTS2 is the only ported
driver whose style vector comes out of an ITERATIVE numerical process (5 ADPM2 steps, each feeding its
own output back into the next network call) rather than a passthrough (Kokoro's `ref_s`) or a single
affine combination (VITS's `z_p`) — an independently-reproduced ~1e-6/1e-7 difference in that vector then
conditions ~50+ sequential AdaIN/conv layers ending in an adversarially-trained (GAN-style) istftnet
vocoder, a network class well known to amplify small input perturbations. Not a logic bug — a genuine,
inherent property of this specific pipeline shape that none of the other five ported models have. Set
`test_e2e_styletts2_lua_driver`'s tolerance to `5e-3` (real margin above the observed `~3.1e-3`) with a
comment in both the test and `styletts2_driver.lua`'s own header explaining why, rather than silently
loosening it without a trace. 103/103 tests passing after landing.

**Status**: all six of the originally-planned models (Whisper, SupertonicTTS, Matcha-TTS, VITS, Kokoro,
StyleTTS2) are now on the embedded-Lua + one-GGUF-file architecture. Only the Python MIL compiler
(`LOOM_MIL_CONVERSION.md`'s `LoomGGUFExporter`/`coremltools`-based frontend) remains deferred from the
original pivot.
