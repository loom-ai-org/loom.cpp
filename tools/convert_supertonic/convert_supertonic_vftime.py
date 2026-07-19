"""Standalone verification of VFTimeEncoder (sinusoidal time embedding + Mish MLP), against the real
`vector_estimator.pt`'s own `time_encoder`. Mish (`x*tanh(softplus(x))`) is a plain composition from
existing SOFTPLUS/TANH/MUL primitives, no new primitive needed.

Usage: python3 convert_supertonic_vftime.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

from supertonic_common import TopologyBuilder, to_f32, write_gguf

N_FREQS = 32
MLP_HIDDEN = 256
MLP_OUT = 64
SCALE = 1000.0


def mish(tb, x, out_hint):
    sp = tb.node("SOFTPLUS", [x], None, f"{out_hint}_softplus")
    t = tb.node("TANH", [sp], None, f"{out_hint}_tanh")
    return tb.node("MUL", [x, t], None, out_hint)


def build_vftime(tb, sd, freqs):
    tb.weight("vftime.freqs", freqs.reshape(N_FREQS))
    angles = tb.node("MUL", ["vftime.freqs", "t"], None, "angles_raw")  # freqs[32] * t[1] (broadcast)
    angles = tb.node("SCALE", [angles], {"s": SCALE}, "angles")
    sin_a = tb.node("SIN", [angles], None, "sin_a")
    cos_a = tb.node("COS", [angles], None, "cos_a")
    embed = tb.node("CONCAT", [sin_a, cos_a], {"dim": 0}, "embed")  # [64]

    w1 = tb.weight("vftime.linear1.weight", to_f32(sd["linear1.weight"]))
    b1 = tb.weight("vftime.linear1.bias", to_f32(sd["linear1.bias"]))
    h = tb.node("ADD", [tb.node("MUL_MAT", [w1, embed], None, "h1_mm"), b1], None, "h1")
    h = mish(tb, h, "h1_mish")

    w2 = tb.weight("vftime.linear2.weight", to_f32(sd["linear2.weight"]))
    b2 = tb.weight("vftime.linear2.bias", to_f32(sd["linear2.bias"]))
    out = tb.node("ADD", [tb.node("MUL_MAT", [w2, h], None, "h2_mm"), b2], None, "out")  # [64]
    return out


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ve = torch.load(repo_root / "assets/pt/vector_estimator.pt", weights_only=False, map_location="cpu")
    te_sd = ve.time_encoder.state_dict()
    freqs = to_f32(ve.time_encoder.freqs)

    tb = TopologyBuilder()
    out = build_vftime(tb, te_sd, freqs)
    inputs = [{"name": "t", "dtype": "f32", "shape": ["1"]}]
    write_gguf(out_dir / "supertonic_vftime.gguf", tb.topology(inputs, out), tb.weights,
               "loom-supertonic-vftime")


if __name__ == "__main__":
    main()
