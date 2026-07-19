#!/usr/bin/env python3
"""Ground truth for VFTextCrossAttention (fractional RoPE -- the hardest single piece in this whole
SupertonicTTS effort) and VFStyleCrossAttention, the real modules directly
(assets/pt/vector_estimator.pt's own `text_attn[0]`/`style_attn[0]`). Synthetic latent/text/style inputs.

Usage: python3 reference_forward_supertonic_vf_attn.py <supertonic-tts repo root> <out_dir>
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

    ve = torch.load(repo_root / "assets/pt/vector_estimator.pt", weights_only=False, map_location="cpu")
    text_attn = ve.text_attn[0]
    style_attn = ve.style_attn[0]
    text_attn.eval()
    style_attn.eval()

    torch.manual_seed(0)
    L, T = 7, 11
    latent = torch.randn(1, 512, L)
    txt_emb = torch.randn(1, 256, T)
    stl_emb = torch.randn(1, 50, 256)
    lat_msk = torch.ones(1, 1, L)
    txt_msk = torch.ones(1, 1, T)

    with torch.no_grad():
        text_out = text_attn(latent, txt_emb, lat_msk, txt_msk)  # (1,512,L)
        style_out = style_attn(latent, stl_emb, lat_msk)  # (1,512,L)

    latent.numpy().astype(np.float32).tofile(out_dir / "vf_attn_latent.bin")
    txt_emb.numpy().astype(np.float32).tofile(out_dir / "vf_attn_txt_emb.bin")
    stl_emb.numpy().astype(np.float32).tofile(out_dir / "vf_attn_stl_emb.bin")
    text_out.numpy().astype(np.float32).tofile(out_dir / "vf_attn_expected_text_out.bin")
    style_out.numpy().astype(np.float32).tofile(out_dir / "vf_attn_expected_style_out.bin")

    # Host-computed fractional positions, real lat_len=L, txt_len=T (single unpadded utterance).
    lat_frac = (np.arange(L, dtype=np.float32) / L)
    txt_frac = (np.arange(T, dtype=np.float32) / T)
    lat_frac.tofile(out_dir / "vf_attn_lat_frac.bin")
    txt_frac.tofile(out_dir / "vf_attn_txt_frac.bin")

    print(f"L={L}, T={T}: text_out mean_abs={text_out.abs().mean().item():.4f}, "
          f"style_out mean_abs={style_out.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
