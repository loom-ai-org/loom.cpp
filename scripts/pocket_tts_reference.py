#!/usr/bin/env python3
"""Regenerate the Pocket-TTS oracle for family 9's fifth leaf.

Pocket-TTS (kyutai-labs/pocket-tts) is a flow LM over CONTINUOUS latents: every autoregressive step
runs a 6-layer transformer over the previous 32-d latent, and a one-step flow head (Lagrangian Self
Distillation) turns the last hidden row plus a Gaussian draw into the next latent. There is no token
and no argmax anywhere in the loop, so the only thing that makes a run reproducible is its NOISE: one
`[32]` draw per step, scaled by `sqrt(temperature)`. This script pins the draws, runs the reference,
and writes out every step:

* `tokens`     the prepared text's SentencePiece ids (`prepare_text_prompt`, then the tokenizer)
* `noise`      `[n_steps, 32]` UNIT normals, one row per step taken; the reference draws `z * std`
* `hidden`     `[n_steps, 1024]`, the transformer's last row after `out_norm` (the flow head's `c`)
* `eos_logits` `[n_steps]`, `out_eos(hidden)` before the `> eos_threshold` comparison
* `latents`    `[n_frames, 32]`, the latents that reached Mimi (normalised space, before `emb_std`)
* `wave`       `[n_samples]`, what `TTSModel.generate_audio` returned

    ~/.venvs/piper/bin/python scripts/pocket_tts_reference.py \
        --model ~/Dev/models/pocket-tts/languages/english_2026-09 --out $LOOM_FIXTURES/pocket_tts_ref

Everything is float32 `.npy` (ids too, since `tests/support/npy_fixture.h` reads one dtype), plus
`meta.json`.

**The waveform is the reference's own streaming output**, and the script also decodes the same latents
in ONE Mimi call from a fresh state and checks the two agree. That is the claim the export rests on: the
streaming decoder is causal (left-padded convolutions, a transposed convolution whose tail is carried
to the next call, attention over a 250-frame window), so one call over every frame computes what the
chunked calls did.

**`--f64` adds `*_f64.npy`**: the whole loop re-run at float64 from the same tokens, the same voice and
the same draws. Every latent feeds the next step, so f32 rounding compounds along the sequence, and the
f64 arm is what says how far two correct f32 implementations can drift apart before a tolerance is
chosen (the lesson of `chatterbox_reference.py`'s NSF phase).

Requires the kyutai-labs/pocket-tts checkout (`--src`), which is not a dependency of this repo.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def save(out: Path, name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    # C-order explicitly, for `chatterbox_reference.py`'s reason: the engine's test reader refuses a
    # Fortran-ordered file rather than reading it transposed.
    np.save(out / f"{name}.npy", np.ascontiguousarray(array, dtype=np.float32))


def local_config(src: Path, model_dir: Path) -> Path:
    """The reference's own YAML for this checkpoint, with its `hf://` paths pointed at `model_dir`.

    The config is named after the checkpoint's directory (`languages/english_2026-09` <->
    `config/english_2026-09.yaml`), which is how the reference's own loader pairs them."""
    import yaml

    config_path = src / "pocket_tts" / "config" / f"{model_dir.name}.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["weights_path"] = str(model_dir / "model.safetensors")
    config.pop("weights_path_without_voice_cloning", None)
    config["flow_lm"]["lookup_table"]["tokenizer"] = "sentencepiece"
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(model_dir / "tokenizer.model")
    path = Path(tempfile.mkdtemp()) / f"{model_dir.name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


class Recorder:
    """Pins the flow head's draw and records every step's hidden row, EOS logit and latent.

    `FlowLMModel.forward` draws with `torch.nn.init.normal_(noise, mean=0, std=sqrt(temp))`; the patch
    fills that buffer with `z * std` from a pre-drawn table instead, which is the same function of the
    same unit draw."""

    def __init__(self, tts, unit_noise: torch.Tensor):
        self.tts = tts
        self.unit_noise = unit_noise
        self.step = 0
        self.hidden, self.eos_logits, self.latents, self.drawn = [], [], [], []

    def __enter__(self):
        flow_lm = self.tts.flow_lm
        self._normal = torch.nn.init.normal_
        self._forward = type(flow_lm).forward

        def normal_(tensor, mean=0.0, std=1.0, generator=None):
            assert tensor.shape[-1] == self.unit_noise.shape[-1] and mean == 0.0
            with torch.no_grad():
                tensor.copy_((self.unit_noise[self.step] * std).to(tensor.dtype).view_as(tensor))
            self.step += 1
            return tensor

        recorder = self

        def forward(module, sequence, text_embeddings, model_state, sampler_decode_steps, temp,
                    noise_clamp, eos_threshold):
            # Only the audio steps draw; the text prefill runs the same forward with an empty sequence,
            # and the reference discards what it returns.
            if sequence.shape[1] == 0:
                return recorder._forward(module, sequence, text_embeddings, model_state,
                                         sampler_decode_steps, temp, noise_clamp, eos_threshold)
            def record(_module, inputs, output):
                # Returns None: a forward hook that returns anything REPLACES the module's output.
                recorder.hidden.append(inputs[0][0].detach().clone())
                recorder.eos_logits.append(output[0, 0].detach().clone())

            hook = module.out_eos.register_forward_hook(record)
            # The row this step is about to draw. NOT the step's index: the text prefill runs the same
            # forward first, and it draws (and discards) a latent too.
            recorder.drawn.append(recorder.unit_noise[recorder.step].detach().clone())
            try:
                latent, is_eos = recorder._forward(module, sequence, text_embeddings, model_state,
                                                   sampler_decode_steps, temp, noise_clamp,
                                                   eos_threshold)
            finally:
                hook.remove()
            recorder.latents.append(latent[0].detach().clone())
            return latent, is_eos

        torch.nn.init.normal_ = normal_
        type(flow_lm).forward = forward
        return self

    def __exit__(self, *exc):
        torch.nn.init.normal_ = self._normal
        type(self.tts.flow_lm).forward = self._forward


class _TorchAtF64:
    """`torch`, except that `torch.float32` is float64. `FlowLMModel.forward` casts the transformer's
    output with `.to(torch.float32)`, which would put the whole flow head of an f64 arm back at f32."""

    float32 = torch.float64

    def __getattr__(self, name):
        return getattr(torch, name)


def _apply_rope_any_dtype(q, k, offset=0, max_period=10_000):
    """`pocket_tts.modules.rope.apply_rope` with its `.float()` casts and float32 aranges replaced by
    the operands' own dtype. The arithmetic is the reference's, line for line."""
    import math

    B, T, H, D = q.shape
    Hk = k.shape[2]
    dtype = q.dtype
    ds = torch.arange(D // 2, device=q.device, dtype=dtype)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = torch.arange(T, device=q.device, dtype=dtype)
    ts += offset
    ts = ts.view(-1, 1, 1)
    q = q.view(B, T, H, D // 2, 2)
    k = k.view(B, T, Hk, D // 2, 2)
    qr, qi = q[..., 0], q[..., 1]
    kr, ki = k[..., 0], k[..., 1]
    rotr, roti = torch.cos(freqs * ts), torch.sin(freqs * ts)
    qo = torch.stack([qr * rotr - qi * roti, qr * roti + qi * rotr], dim=-1)
    ko = torch.stack([kr * rotr - ki * roti, kr * roti + ki * rotr], dim=-1)
    return qo.view(B, T, H, D), ko.view(B, T, Hk, D)


def run(tts, voice: Path, text: str, unit_noise: torch.Tensor, dtype: torch.dtype):
    from pocket_tts.models import flow_lm as flow_lm_module
    from pocket_tts.models.model_state import _import_model_state
    from pocket_tts.modules import rope as rope_module

    if dtype == torch.float64:
        flow_lm_module.torch = _TorchAtF64()
        rope_module.apply_rope = _apply_rope_any_dtype
    tts = tts.to(dtype)
    tts.flow_lm.dtype = dtype
    state = _import_model_state(voice, tts.device)
    state = {m: {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in s.items()}
             for m, s in state.items()}
    with Recorder(tts, unit_noise.to(dtype)) as rec:
        wave = tts.generate_audio(state, text)
    return wave, rec


def one_shot_decode(tts, latents: torch.Tensor) -> torch.Tensor:
    """Every latent through Mimi in ONE call from a fresh state -- what the export's `mimi_decoder`
    computes -- against which the reference's streamed chunks are checked."""
    from pocket_tts.modules.stateful_module import init_states

    n = latents.shape[0]
    steps = int(tts.mimi.encoder_frame_rate / tts.mimi.frame_rate)
    state = init_states(tts.mimi, batch_size=1, sequence_length=n * steps)
    x = latents[None].to(tts.flow_lm.emb_std.dtype) * tts.flow_lm.emb_std + tts.flow_lm.emb_mean
    with torch.no_grad():
        return tts.mimi.decode_from_latent(x, state)[0, 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="a pocket-tts language directory (model.safetensors, tokenizer.model, embeddings/)")
    ap.add_argument("--src", default="/home/flavio/Dev/pocket-tts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--text", default=None, help="default: the reference's own default English text")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f64", action="store_true", help="also re-run the whole loop at float64")
    args = ap.parse_args()

    src, model_dir, out = Path(args.src), Path(args.model).expanduser(), Path(args.out)
    sys.path.insert(0, str(src))
    from pocket_tts.default_parameters import MAX_TOKEN_PER_CHUNK, get_default_text_for_language
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
    from pocket_tts.models.tts_model import TTSModel

    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    tts = TTSModel.load_model(config=str(local_config(src, model_dir)))
    tokenizer = tts.flow_lm.conditioner.tokenizer
    text = args.text or get_default_text_for_language("english")
    chunks = split_into_best_sentences(tokenizer, text, MAX_TOKEN_PER_CHUNK,
                                       tts.pad_with_spaces_for_short_inputs, tts.remove_semicolons,
                                       tts.append_terminal_punctuation, tts.capitalize_first_letter)
    if len(chunks) != 1:
        raise SystemExit(f"the oracle covers one chunk; {text!r} splits into {len(chunks)}")
    prepared, frames_after_eos_guess = prepare_text_prompt(
        chunks[0], tts.pad_with_spaces_for_short_inputs, tts.remove_semicolons,
        tts.append_terminal_punctuation, tts.capitalize_first_letter)
    tokens = tokenizer.encode(prepared)
    max_gen_len = tts._estimate_max_gen_len(len(tokens))

    generator = torch.Generator().manual_seed(args.seed)
    # One row per audio step plus the text prefill's, which the reference draws and discards.
    unit_noise = torch.randn(max_gen_len + 1, tts.flow_lm.ldim, generator=generator)
    voice = model_dir / "embeddings" / f"{args.voice}.safetensors"

    wave, rec = run(tts, voice, text, unit_noise, torch.float32)
    latents = torch.stack(rec.latents)
    n_steps = latents.shape[0]
    # The last step's latent is dropped when the loop breaks on EOS; every other one reached Mimi.
    n_frames = n_steps - (1 if n_steps < max_gen_len else 0)
    emitted = latents[:n_frames]
    single = one_shot_decode(tts, emitted)
    stream_gap = float((single - wave).abs().max())
    print(f"streamed vs one-shot Mimi: max|d| {stream_gap:.3e} over {wave.numel()} samples")
    if single.shape != wave.shape or stream_gap > 1e-4:
        raise SystemExit("the one-shot decode does not reproduce the streamed one; the export's "
                         "single mimi_decoder call would not either")

    eos = torch.stack(rec.eos_logits)
    over = (eos > tts.eos_threshold).nonzero().flatten().tolist()
    eos_step = next((s for s in over if s >= tts._MIN_FRAMES_BEFORE_EOS), None)
    save(out, "tokens", tokens)
    save(out, "noise", torch.stack(rec.drawn))
    save(out, "hidden", torch.stack(rec.hidden))
    save(out, "eos_logits", eos)
    save(out, "latents", emitted)
    save(out, "wave", wave)
    meta = {
        "text": text, "prepared_text": prepared, "voice": args.voice, "seed": args.seed,
        "temperature": tts.temp, "eos_threshold": tts.eos_threshold,
        "min_frames_before_eos": tts._MIN_FRAMES_BEFORE_EOS,
        "frames_after_eos": frames_after_eos_guess + 2, "max_gen_len": max_gen_len,
        "n_tokens": len(tokens), "n_steps": n_steps, "n_frames": n_frames, "eos_step": eos_step,
        "n_samples": int(wave.numel()), "peak": float(wave.abs().max()),
        "eos_margin": float((eos[eos_step] - tts.eos_threshold)) if eos_step is not None else None,
        "stream_vs_one_shot": stream_gap,
    }
    print(json.dumps(meta, indent=1))

    if args.f64:
        wave64, rec64 = run(tts, voice, text, unit_noise.double(), torch.float64)
        latents64 = torch.stack(rec64.latents)
        n = min(latents64.shape[0], latents.shape[0])
        save(out, "latents_f64", latents64[:n_frames])
        save(out, "wave_f64", wave64)
        meta["f64"] = {
            "n_steps": latents64.shape[0],
            "latents_max_abs_diff": float((latents64[:n].float() - latents[:n]).abs().max()),
            "wave_max_abs_diff": (float((wave64.float() - wave).abs().max())
                                  if wave64.shape == wave.shape else None),
        }
        print(json.dumps(meta["f64"], indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
