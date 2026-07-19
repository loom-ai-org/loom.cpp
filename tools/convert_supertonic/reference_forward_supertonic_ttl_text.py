#!/usr/bin/env python3
"""Ground truth for TTLStyleEncoder + TTLTextEncoder (real source: text_to_latent_encoding/encoders.py),
the real modules directly (assets/pt/ttl-style-encoder.pt, assets/pt/text_encoder.pt). Synthetic
compressed-latent crop (real shape (1,144,50)) + real tokenized text ids.

Usage: python3 reference_forward_supertonic_ttl_text.py <supertonic-tts repo root> <out_dir>
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

    ttl_se = torch.load(repo_root / "assets/pt/ttl-style-encoder.pt", weights_only=False, map_location="cpu")
    te = torch.load(repo_root / "assets/pt/text_encoder.pt", weights_only=False, map_location="cpu")
    ttl_se.eval()
    te.eval()

    torch.manual_seed(0)
    T = 10
    txt_ids = torch.randint(1, 163, (1, T), dtype=torch.int64)
    txt_msk = torch.ones(1, 1, T)
    lat_crop = torch.randn(1, 144, 50)

    with torch.no_grad():
        stl_emb = ttl_se(lat_crop)  # (1, 50, 256)
        txt_emb = te(txt_ids, stl_emb, txt_msk)  # (1, 256, T)

    txt_ids.numpy().astype(np.int32).tofile(out_dir / "ttl_txt_ids.bin")
    lat_crop.numpy().astype(np.float32).tofile(out_dir / "ttl_lat_crop.bin")
    stl_emb.numpy().astype(np.float32).tofile(out_dir / "ttl_expected_stl_emb.bin")
    txt_emb.numpy().astype(np.float32).tofile(out_dir / "ttl_expected_txt_emb.bin")
    print(f"T={T}: stl_emb {tuple(stl_emb.shape)}, txt_emb {tuple(txt_emb.shape)}, "
          f"mean_abs_txt_emb={txt_emb.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
