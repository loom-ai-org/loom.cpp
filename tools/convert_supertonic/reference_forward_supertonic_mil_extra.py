#!/usr/bin/env python3
"""Two extra reference fixtures for the MIL-traced SupertonicTTS export (export_supertonic_mil.py),
needed because that export's `dp`/`vfe` topologies fix `T_TEXT` at exactly `T_TEXT_FIXED = 10` (see that
script's own module docstring) -- the EXISTING `reference_forward_supertonic_dp.py` (T=12) and
`reference_forward_supertonic_vfe.py` (T=6) fixtures don't match that shape, so this dumps the same real
modules' own `.forward()`/`.compute_velocity()` again at T=10 instead. `ttl_text`/`decoder` need no new
fixture: `reference_forward_supertonic_ttl_text.py` already uses T=10 unmodified, and `decoder` never
touches the text axis at all.

Usage: python3 reference_forward_supertonic_mil_extra.py <supertonic-tts repo root> <out_dir>
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

    # --- dp, T=10 (same recipe as reference_forward_supertonic_dp.py, T=12 there) ---
    dp = torch.load(repo_root / "assets/pt/duration_predictor.pt", weights_only=False, map_location="cpu")
    dp_se = torch.load(repo_root / "assets/pt/dp-style-encoder.pt", weights_only=False, map_location="cpu")
    dp.eval()
    dp_se.eval()

    torch.manual_seed(0)
    T = 10
    txt_ids = torch.randint(1, 163, (1, T), dtype=torch.int64)
    txt_msk = torch.ones(1, 1, T)
    lat_crop = torch.randn(1, 144, 50)
    with torch.no_grad():
        stl_emb = dp_se(lat_crop)
        duration = dp(txt_ids, stl_emb, txt_msk)

    txt_ids.numpy().astype(np.int32).tofile(out_dir / "dp_mil_txt_ids.bin")
    stl_emb.numpy().astype(np.float32).tofile(out_dir / "dp_mil_stl_emb.bin")
    duration.numpy().astype(np.float32).tofile(out_dir / "dp_mil_expected_duration.bin")
    print(f"dp T={T}: duration={duration.item():.6f}")

    # --- vfe, T=10 (same recipe as reference_forward_supertonic_vfe.py, T=6 there), L=9 unchanged ---
    ve = torch.load(repo_root / "assets/pt/vector_estimator.pt", weights_only=False, map_location="cpu")
    ve.eval()

    torch.manual_seed(0)
    L, T2 = 9, 10
    z_t = torch.randn(1, 144, L)
    txt_emb = torch.randn(1, 256, T2)
    stl_emb2 = torch.randn(1, 50, 256)
    lat_msk = torch.ones(1, 1, L)
    txt_msk2 = torch.ones(1, 1, T2)
    t = torch.tensor([0.3])
    with torch.no_grad():
        v = ve.compute_velocity(z_t, txt_emb, stl_emb2, lat_msk, txt_msk2, t)

    z_t.numpy().astype(np.float32).tofile(out_dir / "vfe_mil_z_t.bin")
    txt_emb.numpy().astype(np.float32).tofile(out_dir / "vfe_mil_txt_emb.bin")
    stl_emb2.numpy().astype(np.float32).tofile(out_dir / "vfe_mil_stl_emb.bin")
    v.numpy().astype(np.float32).tofile(out_dir / "vfe_mil_expected_v.bin")
    print(f"vfe L={L}, T={T2}: v mean_abs={v.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
