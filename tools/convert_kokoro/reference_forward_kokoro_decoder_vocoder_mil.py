#!/usr/bin/env python3
"""Numerical reference for the MIL-traced Kokoro "decoder_vocoder" topology (export_kokoro_mil.py's
DecoderVocoderWrapper). Reuses that SAME wrapper class (and its trace-friendly AdainResBlk1d/SineGen/
SourceModuleHnNSF/Generator/Decoder monkeypatches, already applied at `import export_kokoro_mil` time) in
plain EAGER mode against the real checkpoint -- the same "declared-input reference, not the original
un-traceable formula" convention export_vits_mil.py's own reference_forward_vits_widerange.py established
(rand_ini/noise_in/wsum are genuinely part of this topology's own declared input contract, not values a
reference script should independently resample). Every patch's own docstring in export_kokoro_mil.py
already documents why it's bit-level/mathematically equivalent to the original untraced code, so running
the WRAPPER eagerly (rather than the original unpatched Decoder.forward) is the correct ground truth for
this topology specifically, mirroring test_e2e_kokoro_mil_decoder_vocoder_smoke.cpp's own input shapes but
with real (not zero-filled) values and a fixed seed for reproducibility.

Usage:
  ~/.venvs/piper/bin/python3 tools/convert_kokoro/reference_forward_kokoro_decoder_vocoder_mil.py \\
      <kokoro-v1_0.pth> <config.json> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, for export_kokoro_mil
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import types  # noqa: E402
_stub = types.ModuleType("transformers.utils.versions")
_stub.require_version = lambda *a, **k: None
_stub.require_version_core = lambda *a, **k: None
sys.modules["transformers.utils.versions"] = _stub

from export_kokoro_mil import (  # noqa: E402
    DecoderVocoderWrapper, VerifiedSTFT, compute_wsum_np,
    _STFT_N_FFT, _STFT_HOP, _HARMONIC_NUM, _UPSAMPLE_SCALE,
)
from kokoro.model import KModel  # noqa: E402


def main():
    if len(sys.argv) < 4:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <config.json> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, config_path, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(config_path))
    model = KModel(repo_id="hexgrad/Kokoro-82M", config=cfg, model=ckpt_path, disable_complex=True)
    model.eval()
    model.decoder.generator.verified_stft = VerifiedSTFT(_STFT_N_FFT, _STFT_HOP)
    wrapper = DecoderVocoderWrapper(model.decoder).eval()

    rng = torch.Generator().manual_seed(1234)
    dim_in, t_frames = 512, 40  # deliberately different from build_decoder_vocoder_topology's own dummy_t_frames=40
    dim = _HARMONIC_NUM + 1
    t_f0 = 2 * t_frames
    length = t_f0 * _UPSAMPLE_SCALE

    asr = torch.randn(1, dim_in, t_frames, generator=rng) * 0.3
    f0_curve = torch.randn(1, t_f0, generator=rng) * 50 + 100
    n_curve = torch.rand(1, t_f0, generator=rng)
    s = torch.randn(1, 128, generator=rng) * 0.5
    rand_ini = torch.rand(1, dim, generator=rng)
    noise_in = torch.randn(1, length, dim, generator=rng)
    wsum = torch.from_numpy(compute_wsum_np(t_frames))

    with torch.no_grad():
        waveform = wrapper(asr, f0_curve, n_curve, s, rand_ini, noise_in, wsum)

    def save(name, t):
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32)))

    save("ref_decoder_vocoder_asr", asr)
    save("ref_decoder_vocoder_f0_curve", f0_curve)
    save("ref_decoder_vocoder_n_curve", n_curve)
    save("ref_decoder_vocoder_s", s)
    save("ref_decoder_vocoder_rand_ini", rand_ini)
    save("ref_decoder_vocoder_noise_in", noise_in)
    save("ref_decoder_vocoder_wsum", wsum)
    save("ref_decoder_vocoder_out", waveform)
    print(f"t_frames={t_frames}, waveform shape={tuple(waveform.shape)}, "
          f"mean={waveform.mean().item():.6f}, std={waveform.std().item():.6f}")


if __name__ == "__main__":
    main()
