"""Converts SupertonicTTS v2's full DurationPredictor sub-model (DPTextEncoder + DPStyleEncoder's
`style_token_layer` -- style embedding fed in as a declared graph input, matching how the real
`SpeechGenerator.predict()` calls `dur_predictor.predict(txt_ids, stl_emb=..., txt_msk)` with a
PRECOMPUTED style embedding -- + MLP head w/ PReLU). This is the FIRST full coherent sub-model in this
project's SupertonicTTS effort -- every piece (ConvNeXt, relative-position attention reuse, style
cross-attention) already independently verified in isolation.

`emb_rel_k`/`emb_rel_v` tables are windowed for the SPECIFIC T+1 this standalone test targets (see
`add_multihead_relative_attention`'s own docstring for why a real per-call driver would instead declare
these as graph inputs, computed host-side per utterance -- deferred to the eventual SupertonicDriver).

Usage: python3 convert_supertonic_dp.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

from supertonic_common import TopologyBuilder, to_f32, write_gguf, build_dp_text_encoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_piper_vits"))
from vits_common import get_relative_embeddings  # noqa: E402

DIM = 64
INTERM_DIM = 256
N_CN_LAYERS = 6
N_ATTN_LAYERS = 2
N_HEADS = 2
WINDOW_SIZE = 4
STL_EMBED_DIM = 128  # 8*16, flattened DP style embedding


def build_duration_predictor(tb, sd, T):
    t_plus_1 = T + 1
    tables = []
    for i in range(N_ATTN_LAYERS):
        p = f"attn_layers.{i}"
        emb_rel_k_raw = to_f32(sd[f"{p}.emb_rel_k"]).squeeze(0)  # (9, 32)
        emb_rel_v_raw = to_f32(sd[f"{p}.emb_rel_v"]).squeeze(0)
        tables.append({
            "k": get_relative_embeddings(emb_rel_k_raw, WINDOW_SIZE, t_plus_1),
            "v": get_relative_embeddings(emb_rel_v_raw, WINDOW_SIZE, t_plus_1),
        })

    utt_emb = build_dp_text_encoder(tb, sd, DIM, INTERM_DIM, N_CN_LAYERS, N_ATTN_LAYERS, N_HEADS,
                                     WINDOW_SIZE, tables, "$n_tokens", T, "utt_emb")  # [64]

    # stl_emb: declared input, real (1,8,16) -> flattened [128] (row-major: n_style-major, stl_dim-minor,
    # matching `stl_emb.reshape(B,-1)` in the real DurationPredictor.forward exactly).
    x = tb.node("CONCAT", [utt_emb, "stl_emb"], {"dim": 0}, "mlp_in")  # [64+128=192]

    w0 = tb.weight("dp.layers0.weight", to_f32(sd["layers.0.weight"]))
    b0 = tb.weight("dp.layers0.bias", to_f32(sd["layers.0.bias"]))
    h = tb.node("ADD", [tb.node("MUL_MAT", [w0, x], None, "l0_mm"), b0], None, "l0")  # [128]

    # PReLU(x) = relu(x) - weight*relu(-x) (single learned scalar `weight`, num_parameters=1).
    prelu_w = tb.weight("dp.prelu_weight", to_f32(sd["activation.weight"]))
    relu_pos = tb.node("RELU", [h], None, "prelu_pos")
    neg_h = tb.node("SCALE", [h], {"s": -1.0}, "prelu_negh")
    relu_neg = tb.node("RELU", [neg_h], None, "prelu_relu_neg")
    relu_neg_scaled = tb.node("MUL", [relu_neg, prelu_w], None, "prelu_relu_neg_scaled")
    h = tb.node("SUB", [relu_pos, relu_neg_scaled], None, "prelu_out")

    w1 = tb.weight("dp.layers1.weight", to_f32(sd["layers.1.weight"]))
    b1 = tb.weight("dp.layers1.bias", to_f32(sd["layers.1.bias"]))
    h = tb.node("ADD", [tb.node("MUL_MAT", [w1, h], None, "l1_mm"), b1], None, "l1")  # [1]

    return tb.node("EXP", [h], None, "duration")


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    dp = torch.load(repo_root / "assets/pt/duration_predictor.pt", weights_only=False, map_location="cpu")
    # `build_dp_text_encoder` reads DPTextEncoder's own keys (unprefixed); the top-level DurationPredictor
    # also carries `layers.{0,1}.*`/`activation.weight` (the MLP head) under those SAME unprefixed names
    # already (real `DurationPredictor.state_dict()`'s own top-level keys) -- merge both into one dict.
    sd = dict(dp.sentence_encoder.state_dict())
    for k, v in dp.state_dict().items():
        if k.startswith("layers.") or k.startswith("activation."):
            sd[k] = v

    T = 12
    tb = TopologyBuilder()
    out = build_duration_predictor(tb, sd, T)
    inputs = [
        {"name": "txt_ids", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "stl_emb", "dtype": "f32", "shape": [str(STL_EMBED_DIM)]},
    ]
    write_gguf(out_dir / "supertonic_dp.gguf", tb.topology(inputs, out), tb.weights, "loom-supertonic-dp")


if __name__ == "__main__":
    main()
