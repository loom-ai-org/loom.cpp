#!/usr/bin/env python3
"""Ground truth for reusing VITS's own REL_POS_ATTENTION_SHAW primitive family against SupertonicTTS's
`MultiHeadRelativeAttention` (real source: components.py -- confirmed byte-for-byte the same Shaw et al.
lookup-table + rel_to_abs/abs_to_rel skew mechanism as VITS's own `attentions.Encoder`, just a different
window_size/channel count). Runs the REAL module directly
(`duration_predictor.pt`'s own `sentence_encoder.attn_layers[0]`, channels=64/n_heads=2/window_size=4).

Usage: python3 reference_forward_supertonic_relpos_attn.py <supertonic-tts repo root> <out_dir>
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
    attn = dp.sentence_encoder.attn_layers[0]
    attn.eval()

    torch.manual_seed(0)
    T = 15  # > window_size+1=5, exercises the zero-pad branch of _get_relative_embeddings
    x = torch.randn(1, 64, T)
    attn_mask = torch.ones(1, 1, T, T)
    with torch.no_grad():
        out = attn(x, attn_mask)  # (1, 64, T)

    x.numpy().astype(np.float32).tofile(out_dir / "relpos_attn_x.bin")
    out.numpy().astype(np.float32).tofile(out_dir / "relpos_attn_out.bin")
    print(f"T={T}: out shape {tuple(out.shape)}, mean_abs={out.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
