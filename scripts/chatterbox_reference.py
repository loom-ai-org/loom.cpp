#!/usr/bin/env python3
"""Regenerate the Chatterbox oracle for family 9's fourth leaf.

Chatterbox is two models in sequence, and this runs both in the reference implementation with every
random draw written out:

* **T3**, a Llama-520M that turns text ids into S3 speech tokens under classifier-free guidance. The
  checkpoint SAMPLES (temperature 0.8, min_p 0.05), and two samplers running one algorithm from
  different RNG streams agree on nothing, so the exact oracle here is **guided greedy**: CFG and the
  repetition penalty applied exactly as `T3.inference` applies them, then an argmax where it draws.
  Both are deterministic and change the argmax, so the oracle still exercises both of them. Temperature
  and min_p cannot move an argmax, which is why they are absent here and graded by ear instead.
* **S3Gen**, which in-fills a mel after the built-in voice's prompt frames with a guided 10-step ODE,
  then vocodes it with HiFT. The flow's initial noise and the vocoder's NSF source draws (8 harmonic
  phases and a `[9, n_samples]` Gaussian) are recorded where the reference makes them and written
  out, for the reason `f5_tts_reference.py` gives: the same seed is not the same noise.

    ~/.venvs/piper/bin/python scripts/chatterbox_reference.py \
        --model ~/Dev/models/chatterbox --out $LOOM_FIXTURES/chatterbox_ref

Everything is float32 `.npy` (ids too, since `tests/support/npy_fixture.h` reads one dtype), plus
`meta.json`, and each draw is written in the layout the DRIVER takes it in: `noise` frame-major
`(n_frames, 80)`, `nsf_phase` `(9, 1)`, `nsf_noise` harmonic-major `(9, n_samples)`. The gate is
`tests/gate/test_e2e_chatterbox_lua_driver.cpp`.

**`--f64` adds `*_f64.npy`**: S3Gen re-run at float64 from the same tokens and the same draws. The vocoder's sine source is a cumulative sum over every output sample, so f32 accumulation
ORDER alone moves its phase, and the f64 arm is what says how far two correct f32 implementations
can be apart before a gate tolerance is chosen.

Requires the resemble-ai/chatterbox checkout (`--src`), which is not a dependency of this repo. Its
`perth` watermarker is stubbed out: loom ships Chatterbox WITHOUT the watermark (decided 2026-09-23;
the model card says why), so the oracle is the unwatermarked waveform.
"""
import argparse
import importlib.metadata as _metadata
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _stub_reference_imports() -> None:
    # `chatterbox/__init__.py` reads its own installed version, and it is a checkout, not installed.
    real_version = _metadata.version
    _metadata.version = lambda name: "0.0.0" if name == "chatterbox-tts" else real_version(name)
    perth = types.ModuleType("perth")

    class _NoWatermark:
        def apply_watermark(self, wav, sample_rate):
            return wav

    perth.PerthImplicitWatermarker = _NoWatermark
    sys.modules["perth"] = perth


def save(out: Path, name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    # C-order explicitly: `astype` keeps a transposed view's Fortran layout and `np.save` then records
    # `fortran_order: True`, which the engine's test reader refuses rather than reading transposed.
    np.save(out / f"{name}.npy", np.ascontiguousarray(array, dtype=np.float32))


def guided_greedy(t3, t3_cond, text_tokens, cfg_weight, repetition_penalty, max_new_tokens, out):
    """`T3.inference` with its multinomial draw replaced by an argmax, and nothing else changed."""
    from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

    hp = t3.hp
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    bos = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=bos, cfg_weight=cfg_weight)
    # prepare_input_embeds already appended BOS + speech_pos(0); T3.inference appends a SECOND
    # BOS + speech_pos(0) after it, so the prefill ends in two identical rows. Reproduced as written.
    bos_token = torch.tensor([[hp.start_speech_token]], dtype=torch.long)
    bos_embed = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)
    inputs_embeds = torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1)
    save(out, "t3_cond_emb", t3.prepare_conditioning(t3_cond)[0])
    save(out, "t3_prefill_cond", inputs_embeds[0])
    save(out, "t3_prefill_uncond", inputs_embeds[1])

    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    model = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb,
                                 speech_head=t3.speech_head)
    penalty = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
    output = model(inputs_embeds=inputs_embeds, past_key_values=None, use_cache=True,
                   output_hidden_states=True, return_dict=True)
    past = output.past_key_values
    generated = bos_token.clone()
    predicted = []
    for i in range(max_new_tokens):
        step = output.logits[:, -1, :]
        cond, uncond = step[0:1], step[1:2]
        logits = cond + cfg_weight * (cond - uncond)
        if i == 0:
            save(out, "t3_logits0_cond", cond[0])
            save(out, "t3_logits0_uncond", uncond[0])
        logits = penalty(generated[:1], logits)
        token = logits.argmax(dim=-1, keepdim=True)
        predicted.append(int(token))
        generated = torch.cat([generated, token], dim=1)
        if int(token) == hp.stop_speech_token:
            break
        embed = t3.speech_emb(token) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        output = model(inputs_embeds=torch.cat([embed, embed]), past_key_values=past,
                       output_hidden_states=True, return_dict=True)
        past = output.past_key_values
    return predicted, len_cond


class DrawRecorder:
    """Patches the three places S3Gen draws noise so each draw is recorded, or replayed at f64."""

    def __init__(self):
        self.draws = {}
        self.replay = None

    def install(self):
        from chatterbox.models.s3gen import flow_matching, hifigan
        recorder = self
        cfm = flow_matching.CausalConditionalCFM
        real_cfm_forward = cfm.forward

        def cfm_forward(module, mu, *args, **kwargs):
            real_randn_like = torch.randn_like

            def randn_like(x, *a, **k):
                if recorder.replay is not None:
                    return recorder.replay["flow_noise"].to(x.dtype)
                z = real_randn_like(x, *a, **k)
                recorder.draws["flow_noise"] = z.clone()
                return z
            torch.randn_like = randn_like
            try:
                return real_cfm_forward(module, mu, *args, **kwargs)
            finally:
                torch.randn_like = real_randn_like
        cfm.forward = torch.inference_mode()(cfm_forward)

        def sine_forward(module, f0):
            # SineGen.forward verbatim, with its two draws recorded or replayed.
            F_mat = torch.zeros((f0.size(0), module.harmonic_num + 1, f0.size(-1)),
                                dtype=f0.dtype, device=f0.device)
            for i in range(module.harmonic_num + 1):
                F_mat[:, i: i + 1, :] = f0 * (i + 1) / module.sampling_rate
            theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
            if recorder.replay is not None:
                phase_vec = recorder.replay["nsf_phase"].to(f0.dtype)
            else:
                u_dist = torch.distributions.uniform.Uniform(low=-np.pi, high=np.pi)
                phase_vec = u_dist.sample(sample_shape=(f0.size(0), module.harmonic_num + 1, 1))
                phase_vec[:, 0, :] = 0
                recorder.draws["nsf_phase"] = phase_vec.clone()
            sine_waves = module.sine_amp * torch.sin(theta_mat + phase_vec)
            uv = module._f02uv(f0).to(f0.dtype)
            noise_amp = uv * module.noise_std + (1 - uv) * module.sine_amp / 3
            if recorder.replay is not None:
                unit = recorder.replay["nsf_noise"].to(f0.dtype)
            else:
                unit = torch.randn_like(sine_waves)
                recorder.draws["nsf_noise"] = unit.clone()
            noise = noise_amp * unit
            sine_waves = sine_waves * uv + noise
            recorder.draws["sine_waves"] = sine_waves.clone()
            return sine_waves, uv, noise
        hifigan.SineGen.forward = torch.no_grad()(sine_forward)

        # HiFTGenerator.inference, split so that f0 and the source are captured on the way through.
        def hift_inference(module, speech_feat, cache_source=None):
            f0 = module.f0_predictor(speech_feat)
            s = module.f0_upsamp(f0[:, None]).transpose(1, 2)
            s, _, _ = module.m_source(s)
            s = s.transpose(1, 2)
            recorder.draws["f0"] = f0.clone()
            recorder.draws["source"] = s.clone()
            return module.decode(x=speech_feat, s=s), s
        hifigan.HiFTGenerator.inference = torch.inference_mode()(hift_inference)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the ResembleAI/chatterbox directory")
    ap.add_argument("--src", default="/home/flavio/Dev/chatterbox/src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="I don't really care what you call me.")
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--max-new-tokens", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f64", action="store_true", help="also re-run S3Gen at float64 from the same draws")
    args = ap.parse_args()

    _stub_reference_imports()
    sys.path.insert(0, args.src)
    from chatterbox.tts import ChatterboxTTS, punc_norm
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    recorder = DrawRecorder()
    recorder.install()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    tts = ChatterboxTTS.from_local(args.model, "cpu")
    conds = tts.conds
    if args.exaggeration != float(conds.t3.emotion_adv[0, 0, 0]):
        conds.t3 = T3Cond(speaker_emb=conds.t3.speaker_emb,
                          cond_prompt_speech_tokens=conds.t3.cond_prompt_speech_tokens,
                          emotion_adv=args.exaggeration * torch.ones(1, 1, 1))

    hp = tts.t3.hp
    text = punc_norm(args.text)
    text_tokens = tts.tokenizer.text_to_tokens(text)
    text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)
    save(out, "text_ids", text_tokens[0])

    with torch.inference_mode():
        predicted, len_cond = guided_greedy(tts.t3, conds.t3, text_tokens.long(), args.cfg_weight,
                                            args.repetition_penalty, args.max_new_tokens, out)
    save(out, "speech_tokens_raw", predicted)
    speech_tokens = torch.tensor([t for t in predicted if t < 6561], dtype=torch.long)
    save(out, "speech_tokens", speech_tokens)
    print(f"T3: {len(predicted)} ids, {len(speech_tokens)} speech tokens, "
          f"stopped={predicted[-1] == hp.stop_speech_token}")

    s3gen = tts.s3gen
    gen = conds.gen
    with torch.inference_mode():
        mel = s3gen.flow_inference(speech_tokens, ref_dict=dict(gen), finalize=True)
        wav, _ = s3gen.hift_inference(mel, None)
        wav[:, :len(s3gen.trim_fade)] *= s3gen.trim_fade
    save(out, "mel", mel[0])
    save(out, "wave", wav[0])
    for name in ("nsf_phase", "nsf_noise", "f0", "source"):
        save(out, name, recorder.draws[name][0])
    # FRAME-major, `(n_frames, 80)`: the layout the driver's `noise` input and the ODE state are in.
    # The reference draws it channel-major, `(1, 80, n_frames)`; transposing here keeps the gate from
    # having to know that.
    save(out, "noise", recorder.draws["flow_noise"][0].T)
    for name in ("prompt_token", "prompt_feat", "embedding"):
        save(out, f"voice_{name}", gen[name][0])
    save(out, "voice_speaker_emb", conds.t3.speaker_emb[0])
    save(out, "voice_cond_prompt_speech_tokens", conds.t3.cond_prompt_speech_tokens[0])
    print(f"S3Gen: mel {tuple(mel.shape)}, wave {tuple(wav.shape)} peak {float(wav.abs().max()):.4f}")

    if args.f64:
        recorder.replay = {k: v.double() for k, v in recorder.draws.items()}
        s3gen64 = s3gen.double()
        # A plain attribute rather than a buffer, so `.double()` leaves it at float32.
        s3gen64.mel2wav.stft_window = s3gen64.mel2wav.stft_window.double()
        # `mask_to_bias` asserts a 16/32-bit dtype; the arithmetic is the same at 64.
        from chatterbox.models.s3gen import decoder as s3gen_decoder
        s3gen_decoder.mask_to_bias = lambda mask, dtype: (1.0 - mask.to(dtype)) * -1.0e+10
        gen64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in gen.items()}
        with torch.inference_mode():
            mel64 = s3gen64.flow_inference(speech_tokens, ref_dict=gen64, finalize=True)
            wav64, _ = s3gen64.hift_inference(mel64, None)
            wav64[:, :len(s3gen64.trim_fade)] *= s3gen64.trim_fade
        save(out, "mel_f64", mel64[0])
        save(out, "wave_f64", wav64[0])
        save(out, "source_f64", recorder.draws["source"][0])
        d_mel = (mel64.float() - mel).abs().max().item()
        d_wav = (wav64.float() - wav).abs().max().item()
        print(f"f32 vs f64: mel max|d| {d_mel:.3e}, wave max|d| {d_wav:.3e}")

    (out / "meta.json").write_text(json.dumps({
        "text": args.text, "normalized_text": text, "seed": args.seed,
        "exaggeration": args.exaggeration, "cfg_weight": args.cfg_weight,
        "repetition_penalty": args.repetition_penalty, "len_cond": len_cond,
        "n_prompt_frames": int(gen["prompt_feat"].shape[1]),
        "n_speech_tokens": len(speech_tokens), "sample_rate": 24000,
        "flow_steps": 10, "flow_cfg_rate": 0.7,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
