#!/usr/bin/env python3
"""Ground truth for the full DurationPredictor sub-model (DPTextEncoder + DPStyleEncoder +
MLP head w/ PReLU), the real modules directly (assets/pt/duration_predictor.pt,
assets/pt/dp-style-encoder.pt). Uses a synthetic compressed-latent crop (real shape (1,144,50), matching
`SpeechGenerator.crop_len=50`) and real tokenized text ids (via a small fixed sequence within the real
vocab_size=163, no real TextVectorizer dependency needed for this narrow sub-model check).

Usage: python3 reference_forward_supertonic_dp.py <supertonic-tts repo root> <out_dir>
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

    dp = torch.load(repo_root / "assets/pt/duration_predictor.pt", weights_only=False, map_location="cpu")
    dp_se = torch.load(repo_root / "assets/pt/dp-style-encoder.pt", weights_only=False, map_location="cpu")
    dp.eval()
    dp_se.eval()

    torch.manual_seed(0)
    T = 12
    txt_ids = torch.randint(1, 163, (1, T), dtype=torch.int64)
    txt_msk = torch.ones(1, 1, T)
    lat_crop = torch.randn(1, 144, 50)

    with torch.no_grad():
        stl_emb = dp_se(lat_crop)  # (1, 8, 16)
        duration = dp(txt_ids, stl_emb, txt_msk)  # (1,)

    txt_ids.numpy().astype(np.int32).tofile(out_dir / "dp_txt_ids.bin")
    lat_crop.numpy().astype(np.float32).tofile(out_dir / "dp_lat_crop.bin")
    stl_emb.numpy().astype(np.float32).tofile(out_dir / "dp_stl_emb.bin")
    duration.numpy().astype(np.float32).tofile(out_dir / "dp_expected_duration.bin")
    print(f"T={T}: duration={duration.item():.6f}")


if __name__ == "__main__":
    main()
