"""Standalone verification of StyleEncoderCrossAttention (add_style_encoder_cross_attention,
supertonic_common.py) against the real `dp-style-encoder.pt`'s own `style_token_layer` (dim=64,
stl_dim=16, n_style=8). Input `x` is declared Layout A [T,64] (matching every other ConvNeXt-stack
output convention in this project) -- the real module's own forward does `x_0 = x.transpose(1,2)`
BEFORE its Linear ops, i.e. genuinely needs Layout B [64,T], so this script crosses that boundary with
an explicit PERMUTE+CONT, same pattern used everywhere else in this project.

Usage: python3 convert_supertonic_style_attn.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import torch

from supertonic_common import TopologyBuilder, add_style_encoder_cross_attention, write_gguf

DIM = 64
STL_DIM = 16
N_STYLE = 8


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = torch.load(repo_root / "assets/pt/dp-style-encoder.pt", weights_only=False, map_location="cpu")
    sd = enc.style_token_layer.state_dict()

    tb = TopologyBuilder()
    x_ct_p = tb.node("PERMUTE", ["x"], {"axes": [1, 0, 2, 3]}, "x_ct_p")
    x_ct = tb.node("CONT", [x_ct_p], None, "x_ct")  # [64, T] Layout B
    out = add_style_encoder_cross_attention(tb, x_ct, "sca", sd, "", DIM, STL_DIM, N_STYLE, "$n_tokens", "out")

    inputs = [{"name": "x", "dtype": "f32", "shape": ["$n_tokens", str(DIM)]}]
    write_gguf(out_dir / "supertonic_style_attn_dp.gguf", tb.topology(inputs, out), tb.weights,
               "loom-supertonic-style-attn")


if __name__ == "__main__":
    main()
