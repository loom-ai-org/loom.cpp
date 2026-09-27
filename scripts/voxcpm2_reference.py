#!/usr/bin/env python3
"""Regenerate the VoxCPM2 oracle for family 9's sixth leaf.

VoxCPM2 (OpenBMB/VoxCPM) is a diffusion-autoregressive TTS over CONTINUOUS latents: every step a local
DiT integrates one patch of 4 x 64-d AudioVAE latents from a Gaussian draw, guided (CFG-Zero*) by the
two LMs' last rows, and the patch is re-encoded into the next step's input. Nothing is sampled from a
distribution over ids, so the only thing that makes a run reproducible is its NOISE: one `[64, 4]`
draw per step. This script pins the draws, runs the reference's zero-shot path (`VoxCPM.generate`:
its text preparation, then `_generate_with_prompt_cache` without a prompt), and writes out every step:

* `tokens`       the text's ids as the reference's tokenizer wrapper returns them (no `<|audio_start|>`)
* `noise`        `[n_steps, 4, 64]` UNIT normals, PATCH-major (the reference draws `[1, 64, 4]`; this
                 is its transpose, the layout the engine's driver keeps)
* `mu`           `[n_steps, 2048]`, the DiT's conditioning (`lm_to_dit(lm) ++ res_to_dit(residual)`)
* `stop_logits`  `[n_steps, 2]`, the stop head on the row each patch was generated from
* `patches`      `[n_steps, 4, 64]`, every generated patch, patch-major
* `wave`         `[n_samples]` at 48 kHz, what the reference's non-streaming `generate` returned

    ~/.venvs/piper/bin/python scripts/voxcpm2_reference.py --model ~/Dev/models/voxcpm2 \\
        --out $LOOM_FIXTURES/voxcpm2_ref

Everything is float32 `.npy` (ids too, since `tests/support/npy_fixture.h` reads one dtype), plus
`meta.json`. `retry_badcase` is OFF: a run with pinned draws is a measurement, and the engine's driver
never retries one either.

**`--f64` adds `*_f64.npy`**: the whole loop re-run at float64 from the same tokens and the same
draws, in a fresh process after the f32 model is gone (2.3B parameters at 8 bytes is 18 GB). The
reference computes its RMS norm's variance, RoPE's rotation and RoPE's table in float32 whatever the
model's dtype, so the f64 arm patches those three to the input's dtype -- otherwise it would not be
an f64 arm. Every patch feeds the next step, so rounding compounds along the loop; the f64 arm is what
says how far two correct f32 implementations drift apart before a tolerance is chosen (Retro-055).

Requires the OpenBMB/VoxCPM checkout (`--src`), which is not a dependency of this repo.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_TEXT = ("VoxCPM2 brings multilingual support, creative voice design, and controllable voice "
                "cloning.")


def save(out: Path, name: str, value) -> None:
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    # C-order explicitly: the engine's test reader refuses a Fortran-ordered file rather than reading it
    # transposed (Retro-054).
    np.save(out / f"{name}.npy", np.ascontiguousarray(array, dtype=np.float32))


def patch_for_f64(voxcpm2_module) -> None:
    """The three places the reference computes in float32 whatever the model's dtype."""
    from voxcpm.modules.minicpm4 import model as minicpm

    def rms_layernorm(hidden, weight, eps):
        variance = hidden.pow(2).mean(dim=-1, keepdim=True)
        return hidden * torch.rsqrt(variance + eps) * weight

    def apply_rotary_pos_emb(q, k, cos, sin):
        return (q * cos) + (minicpm.rotate_half(q) * sin), (k * cos) + (minicpm.rotate_half(k) * sin)

    minicpm.rms_layernorm = rms_layernorm
    minicpm.apply_rotary_pos_emb = apply_rotary_pos_emb
    original = voxcpm2_module.get_dtype
    voxcpm2_module.get_dtype = lambda d: torch.float64 if d == "float64" else original(d)


def load(model_dir: str, dtype: torch.dtype):
    from voxcpm.model import voxcpm2 as voxcpm2_module

    if dtype == torch.float64:
        patch_for_f64(voxcpm2_module)
    model = voxcpm2_module.VoxCPM2Model.from_local(model_dir, optimize=False, device="cpu")
    model.config.dtype = "float64" if dtype == torch.float64 else "float32"
    model = model.to(dtype).eval()
    for lm in (model.base_lm, model.residual_lm):
        lm.setup_cache(1, model.config.max_length, "cpu", dtype)
        if dtype == torch.float64 and lm.rope_emb is not None:
            lm.rope_emb._set_cos_sin_cache(lm.rope_emb.max_position_embeddings, "cpu", torch.float64)
    for stack in (model.feat_encoder.encoder, model.feat_decoder.estimator.decoder):
        if dtype == torch.float64 and stack.rope_emb is not None:
            stack.rope_emb._set_cos_sin_cache(stack.rope_emb.max_position_embeddings, "cpu", torch.float64)
    # The AudioVAE decodes in float32 in the reference (`from_local` casts it back, and `_generate` casts
    # the latents to float32 on the way in); the f64 arm keeps the whole path at f64.
    model.audio_vae = model.audio_vae.to(dtype)
    if dtype == torch.float64:
        decode = model.audio_vae.decode
        model.audio_vae.decode = lambda z, sr_cond=None: decode(z.to(torch.float64), sr_cond)
    return model


def prepared_text(text: str) -> str:
    """`VoxCPM._generate`'s own preparation."""
    return re.sub(r"\s+", " ", text.replace("\n", " "))


def run(model, text: str, unit_noise: torch.Tensor):
    """The reference's zero-shot path with the draws pinned; returns every recorded step and the wave."""
    cfm = model.feat_decoder
    dtype = next(model.parameters()).dtype
    record = {"mu": [], "stop_logits": [], "patches": []}
    draws = iter(unit_noise)

    def forward(mu, n_timesteps, patch_size, cond, temperature=1.0, cfg_value=1.0, sway_sampling_coef=1.0,
                use_cfg_zero_star=True):
        # `UnifiedCFM.forward`, with `torch.randn` replaced by the next pinned draw (patch-major on disk,
        # channel-major here).
        z = next(draws).to(dtype).T.unsqueeze(0) * temperature
        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        record["mu"].append(mu[0].clone())
        out = cfm.solve_euler(x=z, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value,
                              use_cfg_zero_star=use_cfg_zero_star)
        record["patches"].append(out[0].T.clone())
        return out

    stop_head = model.stop_head
    original_stop = stop_head.forward

    def stop(x):
        logits = original_stop(x)
        record["stop_logits"].append(logits[0].clone())
        return logits

    stop_head.forward = stop
    cfm.forward = forward
    try:
        wav, _, _ = model.generate_with_prompt_cache(target_text=text, prompt_cache=None, min_len=2, max_len=4096,
                                                     inference_timesteps=10, cfg_value=2.0, retry_badcase=False)
    finally:
        del cfm.forward
        del stop_head.forward
    n = len(record["patches"])
    return {"mu": torch.stack(record["mu"]), "stop_logits": torch.stack(record["stop_logits"][:n]),
            "patches": torch.stack(record["patches"]), "wave": wav.reshape(-1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the openbmb/VoxCPM2 checkpoint directory")
    ap.add_argument("--src", default="/home/flavio/Dev/VoxCPM/src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=DEFAULT_TEXT, help="default: the model card's own example sentence")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f64", action="store_true", help="also re-run the whole loop at float64")
    ap.add_argument("--f64-only", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    sys.path.insert(0, args.src)
    torch.set_grad_enabled(False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = prepared_text(args.text)

    if args.f64_only:
        # The f64 arm, in its own process: the same draws, read back from the f32 run's files.
        model = load(args.model, torch.float64)
        unit_noise = torch.from_numpy(np.load(out / "noise.npy")).double()
        result = run(model, text, unit_noise)
        for key, value in result.items():
            save(out, f"{key}_f64", value)
        print(f"f64: {len(result['patches'])} patches")
        return

    model = load(args.model, torch.float32)
    tokens = model.text_tokenizer(text)
    budget = min(int(len(tokens) * 6.0 + 10), 4096)
    gen = torch.Generator().manual_seed(args.seed)
    unit_noise = torch.randn(budget, 4, 64, generator=gen)
    result = run(model, text, unit_noise)
    n = len(result["patches"])
    save(out, "tokens", tokens)
    save(out, "noise", unit_noise[:n])
    for key, value in result.items():
        save(out, key, value)
    meta = {"text": args.text, "prepared": text, "seed": args.seed, "n_tokens": len(tokens), "n_patches": n,
            "n_samples": int(result["wave"].numel()), "sample_rate": int(model.sample_rate), "cfg": 2.0,
            "timesteps": 10, "budget": budget}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    if args.f64:
        del model
        subprocess.run([sys.executable, __file__, "--model", args.model, "--src", args.src, "--out", args.out,
                        "--text", args.text, "--f64-only"], check=True)


if __name__ == "__main__":
    main()
