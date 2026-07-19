"""Standalone verification of VFTextCrossAttention (fractional RoPE) + VFStyleCrossAttention against
the real `vector_estimator.pt`'s own `text_attn[0]`/`style_attn[0]` modules. `lat_frac_pos`/`txt_frac_pos`
are declared graph inputs, host-computed for this specific (L,T) -- see
`add_vf_text_cross_attention`'s own docstring for why (L,T only known at real call time).

Usage: python3 convert_supertonic_vf_attn.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import torch

from supertonic_common import (TopologyBuilder, write_gguf, add_vf_text_cross_attention,
                                add_vf_style_cross_attention)

LAT_DIM = 512
TXT_DIM = 256
STL_DIM = 256
N_STYLE = 50
TEXT_N_HEADS = 4
TEXT_HEAD_DIM = 64
STYLE_N_HEADS = 2
STYLE_HEAD_DIM = 128


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ve = torch.load(repo_root / "assets/pt/vector_estimator.pt", weights_only=False, map_location="cpu")
    text_sd = ve.text_attn[0].state_dict()
    style_sd = ve.style_attn[0].state_dict()

    L, T = 7, 11

    def cross_to_cb(tb, x_ta, out_hint):
        """`latent`/`txt_emb` are native (B,C,T) tensors that `VFTextCrossAttention.forward` itself
        transposes internally (`x_seq = latent.transpose(1,2)`) -- i.e. their OWN memory layout is
        Layout A [T,C] (T fastest, byte-identical to the real tensor), genuinely DIFFERENT from
        `stl_emb` (already native Layout B, never transposed in the real forward). Cross explicitly,
        same pattern as every other Layout-A-native real input in this project."""
        p = tb.node("PERMUTE", [x_ta], {"axes": [1, 0, 2, 3]}, f"{out_hint}_p")
        return tb.node("CONT", [p], None, out_hint)

    tb = TopologyBuilder()
    latent_cb = cross_to_cb(tb, "latent", "latent_cb")
    txt_emb_cb = cross_to_cb(tb, "txt_emb", "txt_emb_cb")
    out = add_vf_text_cross_attention(tb, latent_cb, txt_emb_cb, "vf_text_attn", text_sd, "", LAT_DIM,
                                       TXT_DIM, TEXT_N_HEADS, TEXT_HEAD_DIM, "lat_frac", "txt_frac",
                                       str(L), str(T), "text_out_cb")
    out_p = tb.node("PERMUTE", [out], {"axes": [1, 0, 2, 3]}, "text_out_p")
    out_ta = tb.node("CONT", [out_p], None, "text_out")  # back to Layout A, matches the real (1,512,L) dump
    inputs = [
        {"name": "latent", "dtype": "f32", "shape": [str(L), str(LAT_DIM)]},
        {"name": "txt_emb", "dtype": "f32", "shape": [str(T), str(TXT_DIM)]},
        {"name": "lat_frac", "dtype": "f32", "shape": [str(L)]},
        {"name": "txt_frac", "dtype": "f32", "shape": [str(T)]},
    ]
    write_gguf(out_dir / "supertonic_vf_text_attn.gguf", tb.topology(inputs, out_ta), tb.weights,
               "loom-supertonic-vf-text-attn")

    tb2 = TopologyBuilder()
    latent_cb2 = cross_to_cb(tb2, "latent", "latent_cb")
    out2 = add_vf_style_cross_attention(tb2, latent_cb2, "stl_emb", "vf_style_attn", style_sd, "", LAT_DIM,
                                         STL_DIM, N_STYLE, STYLE_N_HEADS, STYLE_HEAD_DIM, str(L),
                                         "style_out_cb")
    out2_p = tb2.node("PERMUTE", [out2], {"axes": [1, 0, 2, 3]}, "style_out_p")
    out2_ta = tb2.node("CONT", [out2_p], None, "style_out")
    inputs2 = [
        {"name": "latent", "dtype": "f32", "shape": [str(L), str(LAT_DIM)]},
        {"name": "stl_emb", "dtype": "f32", "shape": [str(STL_DIM), str(N_STYLE)]},
    ]
    write_gguf(out_dir / "supertonic_vf_style_attn.gguf", tb2.topology(inputs2, out2_ta), tb2.weights,
               "loom-supertonic-vf-style-attn")


if __name__ == "__main__":
    main()
