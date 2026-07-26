"""Produces a SECOND flow_vocoder reference fixture at a more REALISTIC z_p scale/length than
reference_forward_vits.py's own Tp=8/std=0.5 one -- found necessary after discovering that the bespoke
ggml flow_vocoder topology (convert_vits.py's build_flow_vocoder_topology, backing
vits_flow_vocoder.gguf) diverges from the real PyTorch ResidualCouplingBlock+Generator by ~0.22 (absolute,
against a ~0.01-0.02 rms signal) for a REAL end-to-end z_p (T=194, values up to +-10), while matching to
~1e-6 for the small-scale Tp=8/std=0.5 z_p the existing reference_forward_vits.py/
test_e2e_vits_flow_vocoder_reference.cpp pair has ever exercised. The MIL-traced flow_vocoder topology
(export_vits_mil.py) matches the SAME real z_p to ~1.2e-6 -- see BACKLOG.md's VITS entry.

This script deliberately does NOT touch reference_forward_vits.py's own existing Tp=8 fixture (that
fixture, and the bespoke test comparing against it, both stay green -- this is an ADDITIONAL, wider-range
check, not a fix to the narrower one). z_p here mimics real generate_path output's own statistics
(observed empirically: roughly std~0.7, with occasional |z_p|>5 outliers from large `exp(logs_p)*
noise_scale` terms) via a large-std Gaussian mixture rather than reproducing generate_path itself
(reproducing the engine's own mt19937 stream in Python isn't needed -- this is a FIXED, externally-
supplied z_p test, same "isolate the wiring from RNG" precedent as reference_forward_vits.py's own
Tp=8 case).

Usage: python reference_forward_vits_widerange.py <checkpoint.ckpt> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/flavio/Dev/piper/src/python")
from piper_train.vits.models import ResidualCouplingBlock, Generator

from vits_common import load_piper_checkpoint


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.ckpt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    full_sd = load_piper_checkpoint(ckpt_path)
    sd = {k[len("model_g."):]: v for k, v in full_sd.items() if k.startswith("model_g.")}

    flow = ResidualCouplingBlock(192, 192, 5, 1, 4, n_flows=4, gin_channels=0)
    missing, unexpected = flow.load_state_dict({k[len("flow."):]: v for k, v in sd.items() if k.startswith("flow.")}, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    flow.eval()
    dec = Generator(192, "2", (3, 5, 7), ((1, 2), (2, 6), (3, 12)), (8, 8, 4), 256, (16, 16, 8), gin_channels=0)
    missing, unexpected = dec.load_state_dict({k[len("dec."):]: v for k, v in sd.items() if k.startswith("dec.")}, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    dec.eval()

    torch.manual_seed(11)
    Tp = 194
    z_p = torch.randn(1, 192, Tp) * 0.7
    outlier_mask = torch.rand(1, 192, Tp) < 0.01
    z_p = torch.where(outlier_mask, torch.randn(1, 192, Tp) * 8, z_p)
    with torch.no_grad():
        y_mask = torch.ones(1, 1, Tp)
        z = flow(z_p, y_mask, g=None, reverse=True)
        wav = dec(z, g=None)
    np.save(out_dir / "ref_z_p_wide.npy", z_p[0].numpy())  # (192, Tp)
    np.save(out_dir / "ref_wav_wide.npy", wav[0, 0].numpy())  # (Tp*256,)
    print(f"z_p range: [{z_p.min().item():.3f}, {z_p.max().item():.3f}], std={z_p.std().item():.3f}")
    print("wav rms:", wav.pow(2).mean().sqrt().item(), "max abs:", wav.abs().max().item())


if __name__ == "__main__":
    main()
