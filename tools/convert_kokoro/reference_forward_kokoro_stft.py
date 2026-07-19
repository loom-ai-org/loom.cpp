"""Real torch.stft/torch.istft ground truth for Kokoro's Generator STFT/ISTFT (istftnet.py's TorchSTFT),
used by tests/test_e2e_kokoro_stft.cpp. No `kokoro`/`transformers` import needed (both broken in this
venv, see BACKLOG.md) -- calls torch.stft/torch.istft directly, exactly mirroring TorchSTFT.transform/
inverse's own real calls.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from kokoro_stft_common import N_FFT, HOP_LENGTH, pad_reflect, compute_wsum


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <out_dir>", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    window = torch.hann_window(N_FFT, periodic=True, dtype=torch.float32)

    # --- forward: real waveform -> magnitude/phase (concatenated, matching istftnet.py's own cat) ---
    rng = np.random.RandomState(7)
    n_samples = 64
    waveform = rng.normal(scale=0.5, size=n_samples).astype(np.float32)
    waveform_t = torch.from_numpy(waveform).unsqueeze(0)
    Xc = torch.stft(waveform_t, N_FFT, HOP_LENGTH, N_FFT, window=window, center=True, return_complex=True)
    mag_fwd = Xc.abs()[0].numpy()      # (n_freq, n_frames)
    phase_fwd = Xc.angle()[0].numpy()  # (n_freq, n_frames)

    waveform_padded = pad_reflect(waveform, N_FFT // 2)

    # --- inverse: independent random (non-self-consistent) magnitude/phase -> waveform ---
    n_frames = 9
    n_freq = N_FFT // 2 + 1
    mag_inv = (rng.rand(n_freq, n_frames).astype(np.float32) * 0.5 + 0.1)
    phase_inv = ((rng.rand(n_freq, n_frames).astype(np.float32) * 2 - 1) * np.pi)
    spec = torch.from_numpy(mag_inv) * torch.exp(torch.from_numpy(phase_inv) * 1j)
    y = torch.istft(spec.unsqueeze(0), N_FFT, HOP_LENGTH, N_FFT, window=window, center=True)
    waveform_inv = y[0].numpy()
    wsum = compute_wsum(n_frames, N_FFT, HOP_LENGTH)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(arr))

    save("ref_stft_waveform_padded", waveform_padded)
    save("ref_stft_mag_fwd", mag_fwd)
    save("ref_stft_phase_fwd", phase_fwd)
    save("ref_stft_mag_inv", mag_inv)
    save("ref_stft_phase_inv", phase_inv)
    save("ref_stft_wsum", wsum)
    save("ref_stft_waveform_inv", waveform_inv)
    print(f"n_samples={n_samples}, n_frames_fwd={mag_fwd.shape[1]}, n_frames_inv={n_frames}, "
          f"waveform_inv_len={waveform_inv.shape[0]}")


if __name__ == "__main__":
    main()
