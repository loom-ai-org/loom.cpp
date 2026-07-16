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
