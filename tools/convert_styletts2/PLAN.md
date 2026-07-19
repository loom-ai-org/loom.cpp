# StyleTTS2 (real yl4579/StyleTTS2-LJSpeech checkpoint): conversion plan

## Context

Task #80, model 4/7 in the user's priority list. Real source read in full and confirmed authentic:
`/home/flavio/Dev/styletts2` is the actual upstream `yl4579/StyleTTS2` repo (remote `github.com/yl4579/
StyleTTS2`, commit `5cedc71`, clean tree) -- re-verified after a user prompt questioned whether the local
copy was legitimate; it is, no need to re-clone. Real pretrained checkpoint downloaded from HF
(`yl4579/StyleTTS2-LJSpeech`, MIT) to `/home/flavio/.claude/tmp/styletts2_model/ckpt/Models/LJSpeech/
epoch_2nd_00100.pth` (750MB) + `config.yml`. `Utils/PLBERT/step_1000000.t7` (a real pretrained PLBERT
checkpoint) is already bundled in the repo itself.

Ground truth for the real inference call order: `Demo/Inference_LJSpeech.ipynb`'s own `inference()`
function (extracted via `jupyter nbconvert --to script`) -- NOT re-derived from `models.py` alone, an
actual working demo notebook from the model authors. This is a stronger source of truth than Kokoro had
(Kokoro's own `KModel.forward_with_tokens` was the closest equivalent there).

## Key finding: StyleTTS2 and Kokoro are almost the same pipeline

`config.yml`'s `model_params` confirms byte-identical hyperparameters to what Kokoro's own checkpoint
already forced us to build: `hidden_dim=512`, `style_dim=128`, `n_layer=3`, `max_dur=50`,
`decoder.{upsample_rates=[10,6], upsample_kernel_sizes=[20,12], upsample_initial_channel=512,
resblock_kernel_sizes=[3,7,11], gen_istft_n_fft=20, gen_istft_hop_size=5, type=istftnet}`. Kokoro's own
model is a fork of this exact architecture. Confirmed directly against the real checkpoint's state dict
keys (all uniformly under a single `module.` prefix, simpler than Kokoro's mixed `""`/`"module"`/
`"module.generator"` situation):

- `bert.*` == Kokoro's `CustomAlbert` (PLBERT is literally `AlbertModel`, same `Utils/PLBERT/util.py`
  wrapper Kokoro's own bert descends from) -- reuse the existing CustomAlbert conversion/topology as-is,
  just pointed at this checkpoint's `bert` key.
- `bert_encoder.*` (`Linear(768,512)`) -- reuse `convert_kokoro_bert_encoder.py`'s `build_bert_encoder`
  verbatim (same Layout B->Layout A `PERMUTE`+`CONT` derivation already verified there).
- `text_encoder.*` (`embedding`+`cnn.{i}.{0,1}`+no LSTM needed at inference beyond what's already built)
  -- reuse Kokoro's TextEncoder topology builder verbatim.
- `predictor.text_encoder.lstms.*` (alternating BiLSTM / `AdaLayerNorm.fc`) == Kokoro's own
  `DurationEncoder` (`predictor.text_encoder` in both models) -- reuse verbatim.
- `predictor.{lstm,duration_proj,shared,F0,N,F0_proj,N_proj}.*` == Kokoro's F0Ntrain stack
  (`AdainResBlk1d`) -- reuse `convert_kokoro_f0n.py` verbatim.
- `decoder.{F0_conv,N_conv,asr_res,decode,encode,generator}.*` == Kokoro's Decoder+Generator, same
  channel/kernel shapes confirmed above -- reuse `convert_kokoro_decoder_core.py` +
  `convert_kokoro_generator.py` verbatim.
- Duration algorithm confirmed IDENTICAL in the real `inference()` source: `duration =
  torch.sigmoid(duration_proj(x)).sum(axis=-1)`, `pred_dur = torch.round(duration).clamp(min=1)` -- byte
  for byte what `loom::predict_durations` already implements (built for Kokoro). One extra quirk in this
  repo's own demo: `pred_dur[-1] += 5` (padding the last token's duration) -- real source, will replicate
  exactly, not an invented add-on.
- `pred_aln_trg` construction (`torch.zeros(...)` + a `for` loop writing 1s per duration span, then
  `d.transpose(-1,-2) @ pred_aln_trg`) is the EXACT SAME one-hot-matmul-as-row-repeat operation already
  proven to collapse to `loom::expand_by_duration` for Kokoro -- reuse directly, no new primitive.
- `style_encoder`/`predictor_encoder` (2D-conv mel-based reference-style extractor, spectral-norm
  `Conv2d`) exist in the checkpoint but are NEVER CALLED in the demo's own `inference()` function (only
  used by the separate, optional `compute_style()` helper for real-reference-audio timbre transfer) --
  deferred out of scope for now, same "basic synthesis path first" precedent as Kokoro deferring its own
  phonemizer; can be picked up later if reference-audio-driven style is wanted.
- `text_aligner` (ASRCNN) / `pitch_extractor` (JDCNet) are loaded by `build_model` but never invoked
  anywhere in `inference()` either (training-time-only components) -- not needed, not being ported.

## The one genuinely new piece: the diffusion-based style sampler

`config.yml`'s `multispeaker: false` confirms the LJSpeech checkpoint uses the plain `Transformer1d`
denoiser (not `StyleTransformer1d`) -- IMPORTANT SCOPE REDUCTION: `Model1d`/`AudioDiffusionConditional`'s
own "unet" default (`get_default_model_kwargs`, a deep multi-scale conv+attention U-Net) is fully
overridden in `build_model` (`diffusion.unet = transformer`) -- the actual real denoiser network run at
inference is just a 3-layer plain Transformer (`num_layers=3, num_heads=8, head_features=64,
multiplier=2`), NOT the U-Net. Confirmed directly against the real checkpoint's `diffusion.net.*` keys
(`blocks.{0,1,2}.{attention,feed_forward}.*`, `to_time`, `to_mapping`, `to_out`, `fixed_embedding` -- no
`use_context_features` branch present since `Transformer1d` wasn't given `context_features` in
`build_model`'s non-multispeaker branch, confirmed by the real key list containing no `to_features.*`).

Real call chain (`DiffusionSampler(model.diffusion.diffusion, sampler=ADPM2Sampler(),
sigma_schedule=KarrasSchedule(sigma_min=1e-4, sigma_max=3.0, rho=9.0), clamp=False)`, called with
`noise=randn(1,1,256)`, `embedding=bert_dur[0].unsqueeze(0)` (raw BERT hidden states, per-token, NOT
`bert_encoder`'s output), `num_steps=5..10`, `embedding_scale=1`):

1. `KarrasSchedule` -- pure host-side scalar math (`sigma_max**(1/rho) + i/(N-1)*(sigma_min**(1/rho) -
   sigma_max**(1/rho))) ** rho`, then a final 0 appended) -- no graph involvement at all.
2. `ADPM2Sampler.step` -- 2 calls to `fn` (the KDiffusion-preconditioned denoise) per outer step (one at
   `sigma_mid`, one at `sigma_down`... actually at `sigma` and `sigma_mid`, see real source), each
   requiring one host-side `torch.randn_like` draw for `sigma_up` noise injection -- same "host computes
   RNG, feeds in as declared f32 input" pattern already used for VITS's SDP/`z_p` sampling and Kokoro's
   SineGen. This is an iterative host-driven loop calling the SAME graph repeatedly with different scalar
   `sigma` inputs each time -- directly analogous to `ode_stepper.cpp`'s existing precedent (VITS's
   normalizing-flow ODE integration), a new small `style_diffusion_sampler.h/.cpp` will follow that same
   shape.
3. `KDiffusion.denoise_fn` preconditioning (`c_skip`,`c_out`,`c_in`,`c_noise` from `sigma`) -- pure
   elementwise graph math from a scalar `sigma` input, trivial with existing primitives (`SQRT`, `DIV`
   or reciprocal-via-`DIV`, `LOG`, `MUL`, `ADD` all already registered).
4. `Transformer1d` denoiser itself -- `LearnedPositionalEmbedding` (sinusoidal-ish: `freqs =
   time*weights*2pi`, concat `[time, sin(freqs), cos(freqs)]`) + 2 Linears w/ GELU (`to_time`) ->
   `to_mapping` (2x Linear+GELU, since `use_context_features=False` here, only time contributes) ->
   broadcast-add `mapping` into `x` before each block -> per-block: self-attention (Q from
   `LayerNorm(x)`, K/V from a SEPARATE `LayerNorm` applied to the SAME `x` -- two independently-learned
   affine params on identical input, a real quirk of `Attention`'s `context=default(context,x)` when no
   cross-attention context is given, not a bug to "simplify away") + residual, then FeedForward(GELU)
   + residual -> mean-pool over the token axis -> `to_out` (`Conv1d` 1x1, 1024->256). Standard
   multi-head softmax attention -- ATTENTION primitive (already used non-causal/no-cache in
   `convert_whisper_encoder.py`'s own encoder self-attention, exact same shape of usage: `kv_cache:
   false`, an all-zero `kq_mask` declared input) covers this directly, no new primitive needed.
   `x` itself is a single "pseudo-token" (`x.expand(-1, embedding.size(1), -1)` broadcasts the length-1
   noise vector across all T BERT-embedding positions before concatenation) -- so the actual attention
   sequence length is T (the phoneme count), same as everywhere else in this pipeline.

No new ggml primitive is expected to be strictly required (GELU, ATTENTION, LAYER_NORM, SIN/COS, basic
elementwise ops are all already registered) -- the novelty here is architectural composition (an
iterative sampling loop calling a graph N times with host-updated scalar/tensor inputs each call) and a
new host-side driver component (`style_diffusion_sampler.h/.cpp`), not new primitives. This is being
tracked anyway per task #80's own mandate to "force out missing primitives" -- if something IS missing
once building starts for real, it'll surface here same as every other model.

## Build order (bottom-up, each step verified before the next depends on it)

1. Confirm the reusable Kokoro builders (`convert_kokoro_bert_encoder.py`, TextEncoder builder,
   `convert_kokoro_f0n.py`'s `add_adain_resblk1d`/DurationEncoder pieces, `convert_kokoro_decoder_core.py`,
   `convert_kokoro_generator.py`) accept a single uniform `sd_prefix="module"` cleanly against this new
   checkpoint's key layout -- expect only plumbing, not new logic.
2. `KarrasSchedule` + `ADPM2Sampler` host-side loop, verified against a hand-rolled Python reference
   using a TRIVIAL toy denoiser (e.g. identity or a fixed linear map) first, isolating the discretization
   math from the real network -- same "verify the mechanism independent of real weights first" discipline
   as `test_hifigan_generator`/`test_e2e_kokoro_adainresblock1`.
3. `Transformer1d` denoiser topology + `KDiffusion` preconditioning, verified against the REAL checkpoint
   weights on a single `denoise_fn` call (no sampling loop yet) against a hand-rolled PyTorch reference.
4. Combine 2+3: full diffusion-based style sampling verified end-to-end against a hand-rolled Python
   port of `DiffusionSampler`+`ADPM2Sampler`+`KarrasSchedule`+the real `model.diffusion.diffusion.
   denoise_fn`, fixed seeds, for a handful of diffusion steps -- confirm close numerical match.
5. `StyleTTS2Driver` host class wiring everything (CustomAlbert -> bert_encoder(unused by diffusion
   path, only by predictor.text_encoder) -> diffusion sampler on raw BERT hidden states -> split
   `s_pred` into ref/s -> `predictor.text_encoder` (DurationEncoder) -> duration prediction/expansion ->
   F0Ntrain -> Decoder/Generator), verified end-to-end (finite/non-trivial output, same scope as
   `test_e2e_kokoro_driver.cpp`/`test_e2e_vits_driver.cpp`) against the real checkpoint, ideally also
   spot-checked against a hand-rolled full-pipeline Python port of the real `inference()` function since
   we have its literal source this time (stronger reference than Kokoro got).

## Verification plan

Same discipline as every other milestone this session: each new primitive/composition gets a
hand-computed or hand-rolled-PyTorch-reference unit test before being trusted in the full assembly;
`tools/convert_styletts2/` mirrors `tools/convert_kokoro/`'s layout (`*_common.py` helpers,
`reference_forward_*.py` hand-rolled ground truths, `convert_*.py` GGUF builders); new
`tests/test_e2e_styletts2_*.cpp` files, `SKIP_RETURN_CODE 77` pattern, gated on an env var pointing at
the converted GGUF directory; `BACKLOG.md` updated with real findings throughout.
