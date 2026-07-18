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

**Still open for VITS**: the full conversion script + hand-rolled reference + e2e test (task #78) —
including the still-deferred `weight_norm`-folding utility, the emb_rel_k/v dynamic-T pad/crop logic
(genuinely needed once real phoneme sequences exceed `window_size+1=5`), and the host-side
`generate_path`/duration-to-frame-count driver (task #71, moved host-side per the data-dependent-
frame-count finding recorded earlier in this same VITS effort). Every primitive-level building block for
the full pipeline (`TextEncoder` via `REL_POS_ATTENTION_SHAW`, the real `StochasticDurationPredictor`, the
coupling flow, and the HiFi-GAN vocoder) is now implemented and individually verified — what remains is
assembly: the conversion script, the host-side driver for the two-phase graph (duration-dependent frame
count), and an end-to-end test against a hand-rolled Python reference of the *entire* real checkpoint.
