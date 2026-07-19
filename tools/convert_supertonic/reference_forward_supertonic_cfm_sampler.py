#!/usr/bin/env python3
"""Ground truth for the FULL CFM Euler sampling loop (real source:
text_to_latent_encoding/latent_encoder.py's `TextToLatentWrapper.predict`, default solver="euler"),
combining the real `vector_estimator.pt` module's own `.solve()` (already independently verified as
`compute_velocity`) with the deterministic Euler loop -- the first point in this project's SupertonicTTS
effort where the sampler loop and the real network run together. Fixed `z0` (dumped, not regenerated) so
the C++ side can replay the EXACT SAME initial condition.

Usage: python3 reference_forward_supertonic_cfm_sampler.py <supertonic-tts repo root> <out_dir>
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
    n_steps = 5
    z0 = torch.randn(1, 144, L)
    txt_emb = torch.randn(1, 256, T)
    stl_emb = torch.randn(1, 50, 256)
    lat_msk = torch.ones(1, 1, L)
    txt_msk = torch.ones(1, 1, T)

    z = z0.clone()
    dt = torch.tensor([1.0 / n_steps])
    with torch.no_grad():
        for i in range(n_steps):
            t = torch.tensor([i / n_steps])
            z = ve.solve(z, txt_emb, stl_emb, lat_msk, txt_msk, t, dt)

    z0.numpy().astype(np.float32).tofile(out_dir / "cfm_z0.bin")
    txt_emb.numpy().astype(np.float32).tofile(out_dir / "cfm_txt_emb.bin")
    stl_emb.numpy().astype(np.float32).tofile(out_dir / "cfm_stl_emb.bin")
    z.numpy().astype(np.float32).tofile(out_dir / "cfm_expected_z_final.bin")
    (np.arange(L, dtype=np.float32) / L).tofile(out_dir / "cfm_lat_frac.bin")
    (np.arange(T, dtype=np.float32) / T).tofile(out_dir / "cfm_txt_frac.bin")

    print(f"L={L}, T={T}, n_steps={n_steps}: z_final mean_abs={z.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
