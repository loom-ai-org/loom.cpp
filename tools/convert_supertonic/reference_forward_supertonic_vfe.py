#!/usr/bin/env python3
"""Ground truth for the FULL VectorFieldEstimator.compute_velocity (real source:
vector_field_estimator.py), the real module directly (assets/pt/vector_estimator.pt). ONE velocity call
(no ODE loop yet -- that's the next milestone). Synthetic z_t/txt_emb/stl_emb, real weights throughout.

Usage: python3 reference_forward_supertonic_vfe.py <supertonic-tts repo root> <out_dir>
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
    ve.eval()

    torch.manual_seed(0)
    L, T = 9, 6
    z_t = torch.randn(1, 144, L)
    txt_emb = torch.randn(1, 256, T)
    stl_emb = torch.randn(1, 50, 256)
    lat_msk = torch.ones(1, 1, L)
    txt_msk = torch.ones(1, 1, T)
    t = torch.tensor([0.3])

    with torch.no_grad():
        v = ve.compute_velocity(z_t, txt_emb, stl_emb, lat_msk, txt_msk, t)  # (1,144,L)

    z_t.numpy().astype(np.float32).tofile(out_dir / "vfe_z_t.bin")
    txt_emb.numpy().astype(np.float32).tofile(out_dir / "vfe_txt_emb.bin")
    stl_emb.numpy().astype(np.float32).tofile(out_dir / "vfe_stl_emb.bin")
    v.numpy().astype(np.float32).tofile(out_dir / "vfe_expected_v.bin")
    np.array([0.3], dtype=np.float32).tofile(out_dir / "vfe_t.bin")
    (np.arange(L, dtype=np.float32) / L).tofile(out_dir / "vfe_lat_frac.bin")
    (np.arange(T, dtype=np.float32) / T).tofile(out_dir / "vfe_txt_frac.bin")

    print(f"L={L}, T={T}: v shape {tuple(v.shape)}, mean_abs={v.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
