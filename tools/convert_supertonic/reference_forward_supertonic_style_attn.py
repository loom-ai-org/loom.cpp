#!/usr/bin/env python3
"""Ground truth for StyleEncoderCrossAttention (real source: components.py), the real module directly
(`dp-style-encoder.pt`'s own `style_token_layer`, dim=64/stl_dim=16/n_style=8) on a synthetic
ConvNeXt-stack-shaped input -- isolates the style-pooling cross-attention math from the ConvNeXt stack
(already independently verified) and proj_in.

Usage: python3 reference_forward_supertonic_style_attn.py <supertonic-tts repo root> <out_dir>
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

    enc = torch.load(repo_root / "assets/pt/dp-style-encoder.pt", weights_only=False, map_location="cpu")
    layer = enc.style_token_layer
    layer.eval()

    torch.manual_seed(0)
    T = 20
    x = torch.randn(1, 64, T)
    with torch.no_grad():
        out = layer(x)  # (1, 8, 16)

    x.numpy().astype(np.float32).tofile(out_dir / "style_attn_dp_x.bin")
    out.numpy().astype(np.float32).tofile(out_dir / "style_attn_dp_out.bin")
    print(f"T={T}: out shape {tuple(out.shape)}, out[0,0,:5]={out[0, 0, :5].tolist()}")


if __name__ == "__main__":
    main()
