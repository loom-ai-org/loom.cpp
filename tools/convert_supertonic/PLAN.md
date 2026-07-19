# SupertonicTTS v2 (real femelo/supertonic-tts checkpoint): conversion plan

## Context

Task #80, model 5/7 in the user's priority list. Real source read in full at
`/home/flavio/Dev/supertonic-tts` (an unofficial Apache-2.0 PyTorch port of Supertone's
"SupertonicTTS-2", confirmed a legitimate clone of `github.com/femelo/supertonic-tts`, treated
STRICTLY READ-ONLY per explicit user instruction -- no fetch/pull/push, `ls`/`grep`/`git log` only).
Real checkpoint: `.pt` files under `assets/pt/` (full pickled `nn.Module`s, `torch.save(self, path)` --
NOT state dicts), one per component. The `supertonic-tts` package is ALREADY `pip install -e`'d in
`/home/flavio/.venvs/piper` (onnx/librosa/lightning all present too) -- `torch.load(...,
weights_only=False)` on any `assets/pt/*.pt` file gives back a REAL, real-weighted `nn.Module` directly,
usable as ground truth via its own `.forward()` (same "import the real package as reference" precedent
as Whisper's own `openai-whisper`), not just hand-copied formulas.

Real inference entry point: `SpeechGenerator.predict()` (`src/supertonic_tts/models/speech_generator.py`).
Precomputed voice styles already exist as JSON assets (`assets/voice_styles/{F1..F5,M1..M5}.json`,
`style_ttl`: (1,50,256), `style_dp`: (1,8,16)) -- so `SpeechEncoder`/`SpeechPreprocessor` (extracting a
style from raw reference audio) are OUT OF SCOPE, same "basic synthesis from a precomputed style, not
from reference audio" precedent established for Kokoro's `style_encoder`/StyleTTS2's
`style_encoder`/`predictor_encoder`. The real tokenizer (`TextVectorizer`) is a trivial JSON unicode-index
lookup table (`assets/onnx/unicode_indexer.json`) -- NOT phoneme-based, no external phonemizer
dependency at all, unlike Kokoro/VITS's still-open task #79. This sidesteps that whole problem for this
model family; a REAL end-to-end text test (not just placeholder token ids) is achievable here.

## Real inference call order (`SpeechGenerator.predict`)

```
duration = dur_predictor.predict(txt_ids, stl_emb=dur_prd_stl_emb, txt_msk)   # scalar seconds, NOT per-token
duration /= speed
lat_msk = get_latent_mask(duration)          # host: samples -> latent-frame count -> boolean mask
lat_hat = lat_encoder.predict(txt_ids, stl_emb=ttl_enc_stl_emb, txt_msk, lat_msk, n_steps=10)
audio = speech_decoder.decode(lat_hat)       # (B, 144, T_lat) -> (B, T_lat*6*512) raw waveform samples
```

Key structural facts confirmed directly from source, not assumed:
- `DurationPredictor` predicts a SINGLE SCALAR total duration in seconds (`(B,)`, via
  `exp(MLP(cat(utt_emb, stl_emb)))`) -- NOT per-token durations. No CUMSUM/generate_path/frame-expansion
  needed anywhere in this model (genuinely simpler than every prior TTS model in this project). The
  per-token alignment is implicit, handled entirely by the VectorFieldEstimator's own cross-attention
  over text during CFM sampling -- there is no explicit duration-to-frame alignment step at all.
- `lat_encoder.predict` is a **deterministic Euler ODE integration** (conditional flow matching, CFM) --
  `z = randn(B,144,T)`; for 10 uniform steps `t=i/10`, `dt=1/10`: `v = vector_field.compute_velocity(z,
  txt_emb, stl_emb, lat_msk, txt_msk, t); z = (z + v*dt) * lat_msk`. NO noise injection at each step
  (unlike StyleTTS2's ADPM2 sampler) -- much simpler than that model's own sampling loop, closer to
  VITS's own `ode_stepper.cpp` shape (though that class's fixed 3-input-name convention doesn't fit this
  model's much richer conditioning, so a new SupertonicTTS-specific host loop is needed, not a reuse of
  `OdeStepper` itself).
- `SpeechDecoder.forward` takes the COMPRESSED 144-channel latent DIRECTLY (`TemporalLatentCompressor`'s
  own `decompress` interleaving is done INLINE inside the decoder itself via
  `reshape(B,24,6,T).permute(0,1,3,2).reshape(B,24,T*6)` -- confirmed directly, no separate compressor
  call needed at inference) and emits the waveform as a **direct flattened sample sequence** -- a
  fully-convolutional causal ConvNeXt stack ending in a `Conv1d(512,512,k=1)` head whose 512-channel
  output at each of `T*6` frame positions IS 512 raw consecutive audio samples, reshaped flat. No
  ISTFT/ConvTranspose1d/SineGen/GAN-upsampling-stack at all -- structurally simpler than every istftnet-
  family decoder this project has built (Kokoro/StyleTTS2's shared Decoder+Generator).

## Reusable components (already built/verified elsewhere this project)

- `DDS_CONV`, `WN` primitives (`components.py`'s own `DDSConv`/`WN` classes exist in the repo but are
  NOT used anywhere in the real inference path -- `predict()`'s call graph never touches them; only
  listed here for completeness, not needed).
- Relative-position Shaw-et-al. attention (`REL_POS_ATTENTION_SHAW`/`REL_TO_ABS_SHAW`/`ABS_TO_REL_SHAW`,
  built for VITS's `TextEncoder`) -- `components.py`'s `MultiHeadRelativeAttention` (used by
  `DPTextEncoder`/`TTLTextPreEncoder`) is the SAME Shaw et al. lookup-table + rel_shift mechanism,
  window_size=4 -- confirmed by direct comparison of `_get_relative_embeddings`/`_rel_to_abs`/
  `_abs_to_rel` against VITS's own already-verified implementation. Direct reuse expected, needs a
  small-example re-verification of the exact window/bucket math since window_size may differ (4 here vs
  VITS's own value) before trusting it, same discipline as every other primitive reuse in this project.
- `GELU`, `LAYER_NORM`, `CONV_1D`/`CONV_1D_DW`, `CONCAT`, `REPEAT` (just added for StyleTTS2), `SOFTMAX`,
  `SIN`/`COS`, `TANH`, `SIGMOID`, `RELU`, `SOFTPLUS` all already registered and directly applicable.

## Genuinely new pieces (primitives + compositions)

1. **Mish activation** (`x * tanh(softplus(x))`, used in `VFTimeEncoder`'s MLP) -- composable directly
   from existing `SOFTPLUS`+`TANH`+`MUL` primitives, no new primitive needed.
2. **Replicate ("edge") padding**, used throughout `ConvNextBlock` (both the symmetric non-causal case
   and the causal left-only case) -- ggml has no native replicate-pad op (only zero-pad via
   `ggml_pad_ext`, already wrapped as `PAD_1D`, plus `ggml_pad_reflect_1d`/`ggml_pad_ext_circular`, no
   "replicate" variant). Composable from existing primitives: extract the boundary row via `VIEW`, `REPEAT`
   it to the pad width, `CONCAT` onto the sequence -- exactly the same "materialize the broadcast, then
   CONCAT" pattern the `REPEAT` primitive was just built for (StyleTTS2's diffusion sampler). Pad widths
   here are always static (`(kernel_size-1)*dilation`, known at conversion time), so this is a pure
   composition, no new primitive.
3. **`ConvNextBlock`** (depthwise conv + LayerNorm + 2x pointwise conv + GELU + learned per-channel
   `gamma` scale + residual, optionally causal/dilated) -- a new reusable topology-builder function
   (mirrors this project's own `add_adain_resblk1d`-style per-model builder convention), composed
   entirely from primitives above (`CONV_1D_DW` for the depthwise conv, `CONV_1D` 1x1 for the pointwise
   convs, `LAYER_NORM`, `GELU`, replicate-pad composition above). Used by EVERY encoder/decoder in this
   model (`DPStyleEncoder`, `DPTextEncoder`, `TTLStyleEncoder`, `TTLTextPreEncoder`, `VectorFieldEstimator`,
   `SpeechDecoder`) -- the single highest-leverage piece to get right first.
4. **`StyleCrossAttention`/`StyleEncoderCrossAttention`** (style-token-pooling cross-attention: a
   LEARNABLE query parameter, tanh-gated 2-head split-concat mechanism, no softmax masking needed since
   it pools a FIXED-length latent segment) -- genuinely new composition, needed by `DPStyleEncoder`/
   `TTLStyleEncoder` (turning a variable-length compressed-latent crop into a fixed `n_style`-token style
   embedding).
5. **`SpeechPromptedCrossAttention`/`SpeechPromptedTextEncoder`** (text queries attend over style
   keys/values; keys are a LEARNABLE parameter, not derived from any input, gated via `tanh`) -- distinct
   from (4): here TEXT is the query side and STYLE is the value side (role-reversed vs (4), where style
   IS the query/pooling target). Needed by `TTLTextEncoder` to condition text embeddings on the TTL style.
6. **`VFTimeEncoder`** (sinusoidal `t*1000*freqs` embedding, concat sin/cos, 2-layer MLP w/ Mish) --
   trivial composition once Mish (1) exists.
7. **`VFTextCrossAttention`** (4-head cross-attention, latent queries / text keys+values, **fractional
   RoPE**: `position = (index / actual_length) * theta`, NOT the usual integer-position RoPE -- a real,
   confirmed-from-source quirk, not a simplification) -- this project's existing `ROPE` primitive almost
   certainly assumes INTEGER positions (built for Qwen3); needs checking whether it can be driven with a
   precomputed fractional-angle tensor directly (bypassing its own position→angle step) or needs a new
   primitive/variant. The single most novel piece in this whole model -- verify against a hand-computed
   small example (or the real PyTorch module's own forward, now that the real package is importable)
   before trusting it in the full `VectorFieldEstimator`.
8. **`VFStyleCrossAttention`** (2-head cross-attention, latent queries / TTL-style keys+values, keys
   derived via `tanh(W_key(fixed_constant_prototype))` -- structurally close to (4)/(5) but with its own
   distinct fixed-prototype-key mechanism) -- new composition, reusing the "tanh-gated key" idiom from
   (4)/(5) but on a per-model learned prototype rather than a shared class.
9. **`VectorFieldEstimator`** assembly: 4 groups × (4×dilated-ConvNext + time-conditioning-add + 1×ConvNext
   + `VFTextCrossAttention` + 1×ConvNext + `VFStyleCrossAttention`) + final 4×ConvNext stack -- the
   biggest single assembly in this model, built only after every sub-piece above is independently
   verified.
10. **Euler ODE host loop** (`z = randn(144,T)`; 10 uniform steps, `z += v*dt`, no noise injection) -- a
    new small host-side driver (own file, NOT a reuse of the existing VITS-specific `OdeStepper`, whose
    fixed 3-input-name convention doesn't fit this model's text+style+mask conditioning) -- otherwise the
    simplest sampling loop of any model family in this project so far.
11. **Frozen `BatchNorm1d` folding** (`SpeechDecoder.final_norm`, `SpeechEncoder.norm` -- unused since
    SpeechEncoder is out of scope) -- eval-mode BatchNorm reduces to a per-channel affine
    (`scale=weight/sqrt(running_var+eps)`, `shift=bias-running_mean*scale`), folded at CONVERSION time
    into a plain `MUL`+`ADD` -- same "fold at conversion time" precedent as weight-norm/Snake's
    reciprocal throughout this project. No new primitive.
12. **`DurationPredictor`** (scalar output: `exp(MLP(cat(utt_emb, stl_emb)))`, `PReLU` activation --
    PReLU is a single learned per-channel slope, composable from `RELU`+scaled-negative-part or checked
    against an existing primitive) -- the smallest complete sub-pipeline in this model, a natural first
    full-pipeline milestone once its own pieces ((3),(4)) are verified.
13. **`SpeechDecoder`**: causal `ConvNextBlock` stack + folded `BatchNorm` + causal head convs + `PReLU` +
    reshape-to-waveform -- verified against the real checkpoint once (3)/(11) are in place.

## Build order (bottom-up, each step verified before the next depends on it)

1. Replicate-pad composition + `ConvNextBlock` builder, verified against the REAL `nn.Module`'s own
   `.forward()` (torch.load(...).pt directly, real weights) on a small synthetic input -- both causal and
   non-causal/dilated variants.
2. Mish + `VFTimeEncoder`, verified against the real module.
3. `StyleCrossAttention`/`StyleEncoderCrossAttention`, verified against the real module (needed for BOTH
   `DPStyleEncoder` and `TTLStyleEncoder`, same class, different dims).
4. `MultiHeadRelativeAttention` reuse check (Shaw et al., window_size=4) -- re-verify against a small
   example / the real module before trusting the existing VITS-built primitive family unchanged.
5. `DPTextEncoder` (ConvNext stack + rel-pos attention + sentence-token pooling) + `DPStyleEncoder` +
   `DurationPredictor` -- FIRST full coherent sub-model, verified end-to-end against the real
   `duration_predictor.pt` + `dp-style-encoder.pt` + a real voice style JSON + real tokenized text.
6. `SpeechPromptedCrossAttention`/`SpeechPromptedTextEncoder`, `TTLTextPreEncoder`/`TTLTextEncoder`,
   `TTLStyleEncoder` -- verified against `text_encoder.pt`/`ttl-style-encoder.pt`.
7. `VFTextCrossAttention` (fractional RoPE -- the hardest single piece, verify the RoPE mechanism in
   isolation first), `VFStyleCrossAttention`, then the full `VectorFieldEstimator` assembly, verified
   against `vector_estimator.pt` (single `compute_velocity` call first, THEN the Euler loop).
8. Euler ODE host loop combining (6)+(7), verified against a hand-rolled/real-module Python port of
   `TextToLatentWrapper.predict`.
9. Folded-`BatchNorm` `SpeechDecoder`, verified against `vocoder.pt`.
10. `SupertonicDriver` host class wiring everything (`DurationPredictor` -> `get_latent_mask` (host) ->
    `TextToLatentWrapper` (Euler CFM) -> `SpeechDecoder`), using precomputed voice-style JSON assets and
    the real (license-free) `TextVectorizer` for genuine text input -- verified end-to-end against the
    real checkpoint, same finite/non-trivial-output scope as every prior driver test, ideally ALSO a real
    numerical match against `SpeechGenerator.predict()` itself since the real package is directly
    importable here (a stronger bar than Kokoro/StyleTTS2/VITS could reach).

## Verification plan

Same discipline as every other milestone: hand-computed or real-module-forward reference for every new
primitive/composition before trusting it in the full assembly; `tools/convert_supertonic/` mirrors
`tools/convert_kokoro/`'s layout; new `tests/test_e2e_supertonic_*.cpp` files, `SKIP_RETURN_CODE 77`
pattern; `BACKLOG.md` updated with real findings throughout. Since the REAL `supertonic_tts` package is
importable in `/home/flavio/.venvs/piper`, reference scripts can call real `nn.Module.forward()` directly
rather than hand-copying formulas where convenient -- still worth a hand-computed sanity check for the
genuinely novel pieces (fractional RoPE, the two style cross-attention variants) per this project's own
"verify before trusting" discipline, not a substitute for it.
