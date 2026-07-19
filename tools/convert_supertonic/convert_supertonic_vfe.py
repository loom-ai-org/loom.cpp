"""Converts the FULL VectorFieldEstimator (real source: vector_field_estimator.py), the biggest single
assembly in this project's SupertonicTTS effort -- verified only after every sub-piece (ConvNeXt,
VFTimeEncoder, VFTextCrossAttention w/ fractional RoPE, VFStyleCrossAttention) was independently
confirmed against the real checkpoint. `stl_emb` is declared Layout B directly (matching
`TTLStyleEncoder`'s own real output convention -- `VFStyleCrossAttention.forward` never transposes it).
`txt_emb`, like `z_t`, is native Layout A (`VFTextCrossAttention.forward` DOES transpose it internally
via `txt_seq = txt_emb.transpose(1,2)` -- real quirk confirmed from source, NOT foldable away: a real bug
caught here via a large numerical mismatch when an earlier version of this script wrongly declared
`txt_emb` as pre-crossed Layout B) -- so this script crosses it explicitly, same as `z_t`->`x`.
`lat_frac_pos`/`txt_frac_pos` are declared graph inputs, host-computed for this specific (L,T) (see
`add_vf_text_cross_attention`'s own docstring).

Usage: python3 convert_supertonic_vfe.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import torch

from supertonic_common import TopologyBuilder, write_gguf, build_vector_field_estimator, cross_ta_to_cb

HP = {
    "latent_dim": 144,
    "hidden_dim": 512,
    "interm_dim": 1024,
    "txt_dim": 256,
    "n_groups": 4,
    "n_cn_layers": 4,
    "time_emb_dim": 64,
    "n_text_heads": 4,
    "text_head_dim": 64,
    "n_style_heads": 2,
    "style_head_dim": 128,
    "n_style": 50,
    "stl_dim": 256,
}


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ve = torch.load(repo_root / "assets/pt/vector_estimator.pt", weights_only=False, map_location="cpu")
    sd = ve.state_dict()

    L, T = 9, 6
    tb = TopologyBuilder()
    txt_emb_cb = cross_ta_to_cb(tb, "txt_emb", "txt_emb_cb")
    v = build_vector_field_estimator(tb, sd, "z_t", txt_emb_cb, "stl_emb", "t", "lat_frac", "txt_frac", HP,
                                      str(L), str(T), "v")
    inputs = [
        {"name": "z_t", "dtype": "f32", "shape": [str(L), str(HP["latent_dim"])]},
        {"name": "txt_emb", "dtype": "f32", "shape": [str(T), str(HP["txt_dim"])]},
        {"name": "stl_emb", "dtype": "f32", "shape": [str(HP["stl_dim"]), str(HP["n_style"])]},
        {"name": "t", "dtype": "f32", "shape": ["1"]},
        {"name": "lat_frac", "dtype": "f32", "shape": [str(L)]},
        {"name": "txt_frac", "dtype": "f32", "shape": [str(T)]},
    ]
    write_gguf(out_dir / "supertonic_vfe.gguf", tb.topology(inputs, v), tb.weights, "loom-supertonic-vfe")


if __name__ == "__main__":
    main()
