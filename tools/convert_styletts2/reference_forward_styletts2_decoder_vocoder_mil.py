#!/usr/bin/env python3
"""Numerical reference for the MIL-traced StyleTTS2 "decoder_vocoder" topology (export_styletts2_mil.py,
part of styletts2_mil.gguf). Reuses export_kokoro_mil.py's own DecoderVocoderWrapper (and its
trace-friendly AdainResBlk1d/SineGen/SourceModuleHnNSF/Generator/Decoder monkeypatches, applied at
`import export_kokoro_mil` time) in plain EAGER mode -- but against StyleTTS2's OWN `Decoder` instance
(Kokoro's real istftnet.Decoder class, StyleTTS2's own checkpoint weights loaded in).

UNLIKE reference_forward_kokoro_decoder_vocoder_mil.py's own arbitrary-synthetic-noise inputs (which work
fine for KOKORO's checkpoint), decoder_vocoder's inputs here are driven by a REAL forward pass through the
rest of the pipeline (CustomAlbert -> style-diffusion sampler -> bert_encoder -> DurationEncoder ->
duration prediction/frame-expansion -> TextEncoder -> F0Ntrain), using REAL StyleTTS2 weights throughout,
for a real (if arbitrary) phoneme sequence -- NOT the original untraced Decoder.forward, still the
DecoderVocoderWrapper (matching that script's own "declared-input reference" convention), but with
in-distribution rather than arbitrary inputs. This was a real, hard-won finding, not a style choice:
arbitrary-magnitude synthetic asr/F0_curve/N_curve/s (even shrunk toward near-zero) reliably drive this
SPECIFIC checkpoint's Decoder/Generator into its `torch.exp()`-based magnitude-reconstruction blow-up
regime (observed spec_logit reaching ~27, i.e. exp(27) ~ 5e11) for EVERY random seed tried -- Kokoro's own
checkpoint happens to stay bounded for that same out-of-distribution regime, but StyleTTS2's does not,
which is a real property of this trained checkpoint (confirmed via per-stage std/max instrumentation: the
decoder core itself stays bounded, std~1-4 through encode/decode/upsample stages; the explosion is
specifically in Generator.conv_post's raw output feeding `torch.exp`), not a loading or exporter bug.
Feeding genuinely in-distribution values (a real style vector's natural ~0.13-0.32 std, not an arbitrary
guess) keeps the whole pipeline in its trained operating regime, where the actual purpose of this test --
catching a real MIL-trace/ggml execution bug, not an artifact of driving the network somewhere it was
never trained to be stable -- is meaningful.

Usage:
  ~/.venvs/piper/bin/python3 tools/convert_styletts2/reference_forward_styletts2_decoder_vocoder_mil.py \\
      <epoch_2nd_00100.pth> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, for export_kokoro_mil
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import types  # noqa: E402
_stub = types.ModuleType("transformers.utils.versions")
_stub.require_version = lambda *a, **k: None
_stub.require_version_core = lambda *a, **k: None
sys.modules["transformers.utils.versions"] = _stub

from export_kokoro_mil import (  # noqa: E402
    DecoderVocoderWrapper, VerifiedSTFT, compute_wsum_np,
    _STFT_N_FFT, _STFT_HOP, _HARMONIC_NUM, _UPSAMPLE_SCALE,
)
from kokoro.istftnet import Decoder  # noqa: E402
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder  # noqa: E402
from transformers import AlbertConfig  # noqa: E402

from reference_forward_styletts2_diffusion import HP as DIFF_HP, transformer1d_forward  # noqa: E402

KOKORO_CONFIG_PATH = "/home/flavio/.claude/tmp/kokoro_model/config.json"


def load_submodule(module, state_dict):
    """Same fallback convention as kokoro.model.KModel.__init__ / export_styletts2_mil.py's own
    load_submodule."""
    try:
        module.load_state_dict(state_dict)
    except Exception:
        stripped = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
        module.load_state_dict(stripped, strict=False)


def karras_schedule(num_steps, sigma_min, sigma_max, rho):
    rho_inv = 1.0 / rho
    steps = np.arange(num_steps, dtype=np.float64)
    denom = max(num_steps - 1, 1)
    sigmas = (sigma_max ** rho_inv + (steps / denom) * (sigma_min ** rho_inv - sigma_max ** rho_inv)) ** rho
    return np.concatenate([sigmas, [0.0]])


def kdiffusion_denoise(x, sigma, embedding, sd, prefix, sigma_data):
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data * (sigma_data ** 2 + sigma ** 2) ** -0.5
    c_in = (sigma ** 2 + sigma_data ** 2) ** -0.5
    c_noise = np.log(sigma) * 0.25
    x_t = torch.from_numpy(x.astype(np.float32))
    x_pred = transformer1d_forward(x_t * float(c_in), torch.tensor(float(c_noise)), embedding, sd, prefix, DIFF_HP)
    return c_skip * x + c_out * x_pred.detach().numpy().astype(np.float64)


def adpm2_step(x, fn, sigma, sigma_next, noise):
    sigma_up = np.sqrt(sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2)
    sigma_down = np.sqrt(sigma_next ** 2 - sigma_up ** 2)
    sigma_mid = (sigma + sigma_down) / 2.0
    d = (x - fn(x, sigma)) / sigma
    x_mid = x + d * (sigma_mid - sigma)
    d_mid = (x_mid - fn(x_mid, sigma_mid)) / sigma_mid
    x_next = x + d_mid * (sigma_down - sigma)
    return x_next + noise * sigma_up


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <epoch_2nd_00100.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    kokoro_cfg = json.load(open(KOKORO_CONFIG_PATH))
    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net = sd_all["net"] if "net" in sd_all else sd_all

    torch.manual_seed(7)

    # --- real CustomAlbert -> raw bert_dur (real weights) ---
    bert = CustomAlbert(AlbertConfig(vocab_size=kokoro_cfg["n_token"], **kokoro_cfg["plbert"]))
    load_submodule(bert, net["bert"])
    bert.eval()

    phoneme_ids = [43, 62, 83, 61, 62, 47, 76, 46, 76, 56, 47, 12, 5, 90, 33, 61]
    token_ids = [0] + phoneme_ids  # real StyleTTS2's own single-leading-0 convention
    tokens = torch.tensor([token_ids], dtype=torch.long)
    T = tokens.shape[1]
    with torch.no_grad():
        bert_dur = bert(tokens, attention_mask=torch.ones_like(tokens))  # (1,T,768)

    # --- real style-diffusion sampler (ADPM2 + KDiffusion preconditioning over the REAL Transformer1d,
    #     already independently verified in reference_forward_styletts2_diffusion_sampler.py) driven by
    #     the REAL bert_dur (not a synthetic embedding this time) -- gives a genuinely in-distribution
    #     style vector, unlike an arbitrary guessed scale. ---
    diff_sd = net["diffusion"]
    diff_prefix = "module.unet"
    sigma_data = 0.45731624995853165
    num_steps = 5
    sigmas = karras_schedule(num_steps, 1e-4, 3.0, 9.0)
    rng_np = np.random.RandomState(7)
    noise0 = rng_np.normal(size=DIFF_HP["channels"])
    step_noises = rng_np.normal(size=(num_steps - 1, DIFF_HP["channels"]))
    embedding = bert_dur[0]  # (T,768) -- real bert_dur, real "embedding" convention
    fn = lambda x, sigma: kdiffusion_denoise(x, sigma, embedding, diff_sd, diff_prefix, sigma_data)
    x = sigmas[0] * noise0
    for i in range(num_steps - 1):
        x = adpm2_step(x, fn, sigmas[i], sigmas[i + 1], step_noises[i])
    style_dim = kokoro_cfg["style_dim"]
    s_decoder_np, s_predictor_np = x[:style_dim], x[style_dim:]
    print(f"real style vector: s_decoder std={s_decoder_np.std():.4f}, s_predictor std={s_predictor_np.std():.4f}")
    s_decoder = torch.from_numpy(s_decoder_np.astype(np.float32)).unsqueeze(0)
    s_predictor = torch.from_numpy(s_predictor_np.astype(np.float32)).unsqueeze(0)

    # --- bert_encoder -> d_en ---
    bert_encoder = torch.nn.Linear(bert.config.hidden_size, kokoro_cfg["hidden_dim"])
    load_submodule(bert_encoder, net["bert_encoder"])
    bert_encoder.eval()
    with torch.no_grad():
        d_en = bert_encoder(bert_dur).transpose(-1, -2)  # (1,512,T)

    # --- predictor (DurationEncoder + lstm + duration_proj + F0Ntrain), real weights ---
    predictor = ProsodyPredictor(style_dim=style_dim, d_hid=kokoro_cfg["hidden_dim"],
                                  nlayers=kokoro_cfg["n_layer"], max_dur=kokoro_cfg["max_dur"], dropout=0.0)
    load_submodule(predictor, net["predictor"])
    predictor.eval()

    text_lengths = torch.tensor([T])
    text_mask = torch.zeros(1, T, dtype=torch.bool)  # no padding, single utterance

    # ProsodyPredictor.forward needs `alignment` up front (real code computes duration BEFORE expansion,
    # then builds alignment from THAT duration) -- mirror Demo/Inference_LJSpeech.ipynb's own inference()
    # exactly: d = predictor.text_encoder(d_en, s, lengths, mask) -> lstm -> duration_proj -> pred_dur ->
    # alignment -> en = d.T @ alignment.
    with torch.no_grad():
        d = predictor.text_encoder(d_en, s_predictor, text_lengths, text_mask)  # (1,T,640)
        x_lstm, _ = predictor.lstm(d)
        duration = predictor.duration_proj(x_lstm)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze(0)).clamp(min=1).long()
        pred_dur[-1] += 5  # real quirk, see styletts2_driver.h

        T_frames = int(pred_dur.sum().item())
        pred_aln_trg = torch.zeros(T, T_frames)
        c_frame = 0
        for i in range(T):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i])] = 1
            c_frame += int(pred_dur[i])
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg  # (1,640,T_frames)
        F0_curve, N_curve = predictor.F0Ntrain(en, s_predictor)  # (1,T_frames_f0) each

        # --- real TextEncoder (separate plain conv+BiLSTM stack) -> t_en -> asr ---
        text_encoder = TextEncoder(channels=kokoro_cfg["hidden_dim"], kernel_size=kokoro_cfg["text_encoder_kernel_size"],
                                    depth=kokoro_cfg["n_layer"], n_symbols=kokoro_cfg["n_token"])
        load_submodule(text_encoder, net["text_encoder"])
        text_encoder.eval()
        t_en = text_encoder(tokens, text_lengths, text_mask)  # (1,512,T)
        asr = t_en @ pred_aln_trg  # (1,512,T_frames)

    print(f"T_frames={T_frames}, asr std={asr.std().item():.4f}, F0_curve std={F0_curve.std().item():.4f}, "
          f"N_curve std={N_curve.std().item():.4f}")

    # --- decoder_vocoder itself ---
    decoder = Decoder(dim_in=kokoro_cfg["hidden_dim"], style_dim=style_dim,
                       dim_out=kokoro_cfg["n_mels"], disable_complex=True, **kokoro_cfg["istftnet"])
    load_submodule(decoder, net["decoder"])
    decoder.eval()
    decoder.generator.verified_stft = VerifiedSTFT(_STFT_N_FFT, _STFT_HOP)
    wrapper = DecoderVocoderWrapper(decoder).eval()

    dim = _HARMONIC_NUM + 1
    t_f0 = F0_curve.shape[1]
    length = t_f0 * _UPSAMPLE_SCALE
    rng = torch.Generator().manual_seed(1234)
    rand_ini = torch.rand(1, dim, generator=rng)
    noise_in = torch.randn(1, length, dim, generator=rng)
    wsum = torch.from_numpy(compute_wsum_np(T_frames))

    with torch.no_grad():
        waveform = wrapper(asr, F0_curve, N_curve, s_decoder, rand_ini, noise_in, wsum)
    print(f"waveform std={waveform.std().item():.4f}, max_abs={waveform.abs().max().item():.4f}")

    def save(name, t):
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)))

    save("ref_styletts2_decoder_vocoder_asr", asr)
    save("ref_styletts2_decoder_vocoder_f0_curve", F0_curve)
    save("ref_styletts2_decoder_vocoder_n_curve", N_curve)
    save("ref_styletts2_decoder_vocoder_s", s_decoder)
    save("ref_styletts2_decoder_vocoder_rand_ini", rand_ini)
    save("ref_styletts2_decoder_vocoder_noise_in", noise_in)
    save("ref_styletts2_decoder_vocoder_wsum", wsum)
    save("ref_styletts2_decoder_vocoder_out", waveform)
    print(f"t_frames={T_frames}, waveform shape={tuple(waveform.shape)}, "
          f"mean={waveform.mean().item():.6f}, std={waveform.std().item():.6f}")


if __name__ == "__main__":
    main()
