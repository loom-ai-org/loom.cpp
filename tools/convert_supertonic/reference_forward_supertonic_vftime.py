#!/usr/bin/env python3
"""Ground truth for VFTimeEncoder (real source: vector_field_estimator.py), the real module directly
(assets/pt/vector_estimator.pt) -- sinusoidal t*1000*freqs embedding + 2-layer MLP w/ Mish.

Usage: python3 reference_forward_supertonic_vftime.py <supertonic-tts repo root> <out_dir>
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
    te = ve.time_encoder
    te.eval()

    for t_val, name in [(0.0, "t0"), (0.37, "t037"), (0.9, "t09")]:
        t = torch.tensor([t_val])
        with torch.no_grad():
            out = te(t)  # (1, 64, 1)
        np.array([t_val], dtype=np.float32).tofile(out_dir / f"vftime_{name}_t.bin")
        out.squeeze().detach().numpy().astype(np.float32).tofile(out_dir / f"vftime_{name}_out.bin")
        print(f"t={t_val}: out[:5]={out.squeeze()[:5].tolist()}")

    # Dump the real freqs buffer + scale for the C++ conversion script (a fixed constant, no separate
    # conversion script identity needed -- reuse this dump directly).
    te.freqs.detach().numpy().astype(np.float32).tofile(out_dir / "vftime_freqs.bin")


if __name__ == "__main__":
    main()
