"""Hand-rolled pure-PyTorch reference for Kokoro Generator's NSF harmonic source (istftnet.py's real
`SineGen`/`SourceModuleHnNSF.forward`, hand-copied verbatim -- `kokoro`/`transformers` can't be imported
in this venv, see BACKLOG.md), used as ground truth for tests/test_e2e_kokoro_sinegen.cpp. Matches
convert_kokoro_sinegen.py's own composition to max_diff=3.0e-8 in a standalone numpy cross-check done
BEFORE writing any GGUF/C++ code (see BACKLOG.md) -- this script re-derives the same real forward pass
independently (via real torch.nn.functional.interpolate/cumsum/sin, not the numpy reimplementation used
for that upfront cross-check) so the C++ test compares against an independent computation path.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_sinegen import HP


def real_sinegen_har_source(f0_curve, rand_ini, noise, l_linear_w, l_linear_b, hp):
    """f0_curve: (T_frames,). rand_ini: (dim,), index 0 must be 0. noise: (L,dim). l_linear_w: (1,dim).
    l_linear_b: (1,). Returns har_source (L,)."""
    scale = hp["upsample_scale"]
    dim = hp["harmonic_num"] + 1
    sampling_rate = hp["sampling_rate"]

    f0_up = F.interpolate(f0_curve[None, None, :], scale_factor=scale, mode="nearest")[0, 0]  # (L,)
    f0_b = f0_up[None, :, None]  # (1,L,1)
    fn = f0_b * torch.arange(1, dim + 1, dtype=torch.float32)[None, None, :]  # (1,L,dim)

    rad_values = (fn / sampling_rate) % 1
    rad_values = rad_values.clone()
    rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini[None, :]
    rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1 / scale, mode="linear").transpose(1, 2)
    phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
    phase = F.interpolate(phase.transpose(1, 2) * scale, scale_factor=scale, mode="linear").transpose(1, 2)
    sines = torch.sin(phase)

    sine_waves = sines * hp["sine_amp"]
    uv = (f0_b > hp["voiced_threshold"]).float()
    noise_amp = uv * hp["noise_std"] + (1 - uv) * hp["sine_amp"] / 3
    noise_term = noise_amp * noise[None]
    sine_waves_final = sine_waves * uv + noise_term

    har_source = torch.tanh(F.linear(sine_waves_final, l_linear_w, l_linear_b))  # (1,L,1)
    return har_source[0, :, 0]


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <gguf_dir> <ref_out_dir>", file=sys.stderr)
        sys.exit(1)
    gguf_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = HP
    dim = hp["harmonic_num"] + 1

    l_linear_w = torch.from_numpy(np.load(gguf_dir / "kokoro_sinegen_l_linear_w.npy"))
    l_linear_b = torch.from_numpy(np.load(gguf_dir / "kokoro_sinegen_l_linear_b.npy"))

    rng = np.random.RandomState(13)
    T_frames = 4
    L = T_frames * hp["upsample_scale"]
    f0_curve = torch.tensor([0.0, 180.0, 0.0, 260.0], dtype=torch.float32)  # mixes unvoiced/voiced frames
    rand_ini = rng.rand(dim).astype(np.float32)
    rand_ini[0] = 0.0
    noise = rng.normal(size=(L, dim)).astype(np.float32)

    with torch.no_grad():
        har_source = real_sinegen_har_source(f0_curve, torch.from_numpy(rand_ini), torch.from_numpy(noise),
                                              l_linear_w, l_linear_b, hp)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(arr))

    save("ref_sinegen_f0_curve", f0_curve.numpy())
    save("ref_sinegen_rand_ini", rand_ini)
    save("ref_sinegen_noise", noise)
    save("ref_sinegen_har_source", har_source.numpy())
    print(f"T_frames={T_frames}, L={L}, har_source[:5]={har_source.numpy()[:5]}")


if __name__ == "__main__":
    main()
