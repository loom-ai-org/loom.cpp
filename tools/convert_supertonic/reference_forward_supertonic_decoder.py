#!/usr/bin/env python3
"""Ground truth for SpeechDecoder (real source: speech_autoencoding/speech_autoencoder.py), the real
module directly (assets/pt/vocoder.pt) -- the FINAL stage of the whole SupertonicTTS pipeline, real
weights, synthetic compressed latent.

Usage: python3 reference_forward_supertonic_decoder.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    dec = torch.load(repo_root / "assets/pt/vocoder.pt", weights_only=False, map_location="cpu")
    dec.eval()

    torch.manual_seed(0)
    T = 4
    latent = torch.randn(1, 144, T)

    with torch.no_grad():
        wav = dec(latent)  # (1, T*6*512)

    latent.numpy().astype(np.float32).tofile(out_dir / "decoder_latent.bin")
    wav.numpy().astype(np.float32).tofile(out_dir / "decoder_expected_wav.bin")
    print(f"T={T}: wav shape {tuple(wav.shape)}, mean_abs={wav.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
