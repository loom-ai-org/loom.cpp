# Matcha-TTS: phased conversion plan

## Context

Task #80, done per explicit user direction ("Start Matcha-TTS first", overriding the previously-stated
F5-TTS-next order). Grad-TTS/GlowTTS-lineage conditional-flow-matching mel-spectrogram TTS + a separate
HiFi-GAN vocoder. Real source cloned fresh into `/home/flavio/Dev/Matcha-TTS`
(`github.com/shivammehta25/Matcha-TTS`, commit `bd4d90d`), read in full before planning anything:
`matcha/models/matcha_tts.py`, `matcha/models/components/{text_encoder,flow_matching,decoder,transformer}.py`,
`matcha/utils/model.py`, `matcha/hifigan/{models,config}.py`, real YAML configs.

Real checkpoints downloaded to `/home/flavio/.claude/tmp/matcha_model/ckpt/`:
`matcha_ljspeech.ckpt` (208MB, real LJSpeech Matcha-TTS weights) and `generator_v1` (53MB, paired real
HiFi-GAN v1 vocoder), from the URLs hardcoded in `matcha/cli.py`'s `MATCHA_URLS`/`VOCODER_URLS`
(`github.com/shivammehta25/Matcha-TTS-checkpoints` releases) since no HF Hub repo exists for this model.
Real state dict inspected directly (305 tensors) -- confirms every hyperparameter derived from the YAML
configs, and confirms the exact module shapes assumed below (see "Real checkpoint facts").

New dedicated venv `/home/flavio/.venvs/matcha` (CPU torch 2.13.0, `diffusers==0.25.0` pinned with
`huggingface_hub==0.20.3` to dodge the same `cached_download` import conflict already documented for
`transformers`, `conformer==0.3.2`, `matcha` itself installed editable `--no-deps`).

## Real inference call order (confirmed against source)

`MatchaTTS.synthesise()`:
1. `TextEncoder(x, x_lengths, spks=None)` -> `mu_x [B,80,T_x]`, `logw [B,1,T_x]`, `x_mask`.
2. `w_ceil = ceil(exp(logw)*x_mask) * length_scale` -- PER-TOKEN integer durations (unlike SupertonicTTS's
   single scalar total duration).
3. `y_lengths = sum(w_ceil)`, `y_max_length_ = fix_len_compatibility(y_max_length)` (round up to a multiple
   of `2^num_downsamplings_in_unet` -- default arg is 2, i.e. multiple of 4, REGARDLESS of the real U-Net
   only doing one actual downsample for this checkpoint's shallow `channels=[256,256]` config; harmless
   over-padding, must be replicated exactly for T_lat sizing).
4. `generate_path(w_ceil, attn_mask)` -> `attn` (alignment map) -> `mu_y = attn^T @ mu_x` (per-token means
   expanded to per-frame means, frame-length `y_max_length_`).
5. `CFM.forward(mu_y, y_mask, n_timesteps, temperature, spks)` == `solve_euler`: `z = randn*temperature`,
   `t_span = linspace(0,1,n_timesteps+1)`, loop `dphi_dt = estimator(x,mask,mu,t,spks,cond); x += dt*dphi_dt`
   -- STRUCTURALLY IDENTICAL to the already-built `loom::cfm_euler_sample` (SupertonicTTS), just needs a
   `VelocityFn` wrapping the new `Decoder` U-Net estimator instead of `VectorFieldEstimator`.
6. `denormalize(decoder_outputs, mel_mean=-5.536622, mel_std=2.116101)` -> real mel-spectrogram.
7. Separate HiFi-GAN v1 `Generator(mel)` -> waveform (NOT part of `MatchaTTS` itself; a second checkpoint).

## Real checkpoint facts (confirmed via direct state-dict inspection, 305 tensors)

- `n_vocab=178, n_spks=1` (LJSpeech, single-speaker -- so, like VITS's piper checkpoint, no speaker
  embedding table exists and `spks` stays `None` throughout), `spk_emb_dim=64` (unused, n_spks=1),
  `n_feats=80` (mel channels).
- Encoder: `n_channels=192, filter_channels=768, filter_channels_dp=256, n_heads=2, n_layers=6,
  kernel_size=3, prenet=True`. `encoder.prenet.*` (ConvReluNorm, 3 conv layers), `encoder.encoder.*`
  (6x `attn_layers`/`ffn_layers`/`norm_layers_1`/`norm_layers_2`), `encoder.proj_m` (mu, conv1x1),
  `encoder.proj_w.*` (DurationPredictor: 2 conv+norm layers + final proj to 1 channel).
- `attn_layers.N.conv_{q,k,v,o}`: all `192x192x1` (single-kernel "conv1x1" = a matmul with bias) --
  channel-preserving (no separate head-split projection width), 2 heads means 96 channels/head.
- Decoder (`decoder.estimator.*`): `down_blocks.0` (ResnetBlock1D 160->256 [160 = 2*80 mu/x concat, see
  below] + 1x BasicTransformerBlock + Downsample1D-as-strided-conv) -> `down_blocks.1` (ResnetBlock1D
  256->256 + 1x BasicTransformerBlock + a PLAIN conv, `is_last=True` so no real downsample) ->
  `mid_blocks.0`/`mid_blocks.1` (2x [ResnetBlock1D 256->256 + BasicTransformerBlock], SAME resolution as
  after the one real downsample) -> `up_blocks.0` (ResnetBlock1D 512->256 [512 = 256 decoder + 256 skip
  concat] + BasicTransformerBlock + real ConvTranspose1d upsample, kernel 4) -> `up_blocks.1` (ResnetBlock1D
  512->256 + BasicTransformerBlock + plain conv, `is_last=True`) -> `final_block` (Block1D 256->256) ->
  `final_proj` (256->80, conv1x1). `time_mlp`: `SinusoidalPosEmb(dim=160)` -> `Linear(160,1024)` -> SiLU ->
  `Linear(1024,1024)` (`time_embed_dim = channels[0]*4 = 1024`).  CONFIRMS the "shallow U-Net, exactly 1
  real down/upsample despite 2 tuple entries" hypothesis from source-reading -- `down_blocks.0.2` uses a
  strided conv (real downsample) while `down_blocks.1.2` is a plain same-resolution conv (`is_last`); the
  mirror holds for `up_blocks.0.2`/`up_blocks.1.2` (kernel 4 ConvTranspose1d vs plain conv).
- `attn1.to_{q,k,v}`: NO bias (confirmed: only `to_out.0.{weight,bias}` present) -- standard `diffusers`
  `Attention` default (`bias=False` on qkv projections). `attn_head_dim=64, num_heads=2` -> `to_q/k/v`
  project 256->128 (2*64), `to_out.0` projects 128->256 back.
- `ff.net.0` is `SnakeBeta` (`alpha`,`beta` both `(1024,)`, log-scale) + `proj` (256->1024); `ff.net.2` is
  the second linear (1024->256) -- confirms `FeedForward(dim=256, mult=4, activation_fn="snakebeta")`'s
  real shape, no GEGLU gating (single proj, not doubled, since SnakeBeta isn't gated like GEGLU).
- `norm1`/`norm3` only (no `norm2`) -- confirms `BasicTransformerBlock`'s real forward has NO
  cross-attention branch (`attn2=None`, `cross_attention_dim=None`), matching source reading.
- `block1.block.0`/`block2.block.0` = Conv1d (kernel 3); `block1.block.1`/`block2.block.1` shape `(C,)` =
  the GroupNorm learned affine (`nn.GroupNorm(groups=8, dim_out)`, confirmed from `decoder.py` source,
  default `eps=1e-5`) -- confirms `GROUP_NORM` (added this session, wraps native `ggml_group_norm`,
  verified via hand-computed unit test) plus separate MUL/ADD affine is exactly what's needed.
- `mel_mean=-5.536622, mel_std=2.116101` (scalars, stored directly in the checkpoint's `state_dict` as
  0-d tensors, not just `hyper_parameters['data_statistics']` -- both agree).

## Already fully covered by existing loom primitives / composables (verified, not assumed)

- HiFi-GAN vocoder: same `jik876/hifi-gan` `Generator` family already built for VITS's own decoder
  (`tools/convert_piper_vits/convert_vits.py`'s `build_flow_vocoder_topology`) -- CONV_1D, CONV_TRANSPOSE_1D,
  LEAKY_RELU, TANH, ADD all reused as-is. Real difference from VITS-piper: `resblock="1"` here (needs BOTH
  `convs1` AND `convs2` per ResBlock, VITS-piper's own checkpoint used `resblock="2"`, convs-only) and 4
  upsample stages (`[8,8,2,2]`) vs VITS-piper's 3 (`[8,8,4]`) -- a different topology-loop shape, no new
  primitive.
- CFM Euler loop: `loom::cfm_euler_sample` (SupertonicTTS) reused verbatim with a new `VelocityFn`.
- `attn_layers` (TextEncoder self-attention): partial-rotary REAL integer-position RoPE (rotates only
  `self.d = int(k_channels*0.5)` channels, "rotate-half" convention) -- GENUINELY DIFFERENT from both
  VITS's Shaw-et-al. relative-position lookup table AND SupertonicTTS's fractional-position RoPE. Needs a
  new RoPE variant composition (existing `add_rope`-style SIN/COS/MUL/etc. composables from SupertonicTTS
  should mostly carry over, but applied to only half the channel width with the other half passed through
  unrotated via VIEW+CONCAT) -- verify against a hand-computed example before trusting.
- `ConvReluNorm` prenet / `DurationPredictor`'s conv+LayerNorm+ReLU+dropout stack: plain CONV_1D + custom
  channel-axis LAYER_NORM (already used pervasively) + RELU, residual add on the prenet only.
- `generate_path`: same cumsum+sequence_mask+diff algorithmic family as VITS's own `generate_path`
  (already inlined in `vits_driver.cpp`), which was established to degenerate to a plain host-side
  "repeat row t of mu_x, duration[t] times" operation for single-utterance inference (no batching, mask
  always all-valid) -- STRONG working hypothesis given identical math, needs explicit re-verification
  against Matcha's own exact tensor shapes (task #110) before reuse, not just assumed.
- `BasicTransformerBlock`'s real-config-reduced forward: `norm1(x)` (plain LayerNorm, `nn.LayerNorm` not
  the custom channel-axis one) -> self-attention (standard scaled-dot-product softmax, no relative
  position bias at all here) + residual -> `norm3(x)` -> `FeedForward(SnakeBeta)` + residual.

## Real, confirmed engine additions needed

1. **`GROUP_NORM`** -- DONE this session (wraps native `ggml_group_norm`, verified via hand-computed
   2-group/4-channel unit test in `tests/test_primitive_registry.cpp`, 141/141 passing).
2. **`SnakeBeta` activation** -- `x + (1/exp(beta)) * sin(x*exp(alpha))^2`, log-scale alpha/beta (learned,
   per-channel, `(1024,)` in the FFN's hidden width) -- composable from existing EXP/SIN/SQR/MUL/ADD/DIV,
   no new primitive, but genuinely different from Kokoro's own plain (non-log-scale) Snake, which folded
   the reciprocal at conversion time -- here it must be computed at runtime since it's inside a learned
   log-scale parameter, not a static fold.
3. **Partial-rotary integer-position RoPE** -- needs verifying whether SupertonicTTS's own `add_rope`
   composable (built for FRACTIONAL positions) generalizes cleanly to real INTEGER positions restricted to
   half the channel width, or needs its own small variant.

## Build order (bottom-up, each step verified before the next depends on it)

1. ~~`GROUP_NORM` primitive~~ (done).
2. TextEncoder: `ConvReluNorm` prenet, partial-rotary RoPE + `MultiHeadAttention`, `FFN`, post-norm
   `Encoder` stack, `proj_m`/`proj_w` (`DurationPredictor`). Verify against real checkpoint with a
   hand-rolled Python reference (task #109).
3. Verify `generate_path`'s row-repeat reduction against a small hand-computed example with Matcha's own
   exact shapes (task #110).
4. Decoder U-Net: `SinusoidalPosEmb`+`TimestepEmbedding` time conditioning, `Block1D`/`ResnetBlock1D` (using
   `GROUP_NORM`), `BasicTransformerBlock` (SnakeBeta FFN, no cross-attn), the ONE real downsample/upsample
   + `is_last` plain-conv paths, skip-connection CONCATs (task #111).
5. Wire the CFM Euler loop against the new Decoder estimator (task #112).
6. HiFi-GAN v1 vocoder (`resblock="1"`, 4 upsample stages) (task #113).
7. `MatchaDriver`: full assembly + real e2e test, `SKIP_RETURN_CODE 77` pattern if checkpoint files are
   absent (task #114).

## Verification plan

Same discipline as every other model this session: each new primitive/composition gets a hand-computed
unit test before use; a `matcha_common.py` (mirroring `vits_common.py`/`supertonic_common.py`) holds shared
`TopologyBuilder` helpers; per-module `convert_matcha_*.py` + `reference_forward_matcha_*.py` pairs; a
final `convert_matcha_all.py` master script; `tests/test_e2e_matcha_*.cpp` per stage, compared against the
hand-rolled Python reference before trusting the full waveform output. `BACKLOG.md` updated with real
findings throughout.
