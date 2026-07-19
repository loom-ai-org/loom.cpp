"""Standalone verification that VITS's own REL_POS_ATTENTION_SHAW primitive family (built for
`attentions.Encoder`) is directly reusable for SupertonicTTS's `MultiHeadRelativeAttention` (real
source: components.py) -- confirmed the SAME Shaw et al. lookup-table + rel_to_abs/abs_to_rel skew
mechanism, just channels=64/n_heads=2/window_size=4 here vs whatever VITS's own checkpoint used. Reuses
`tools/convert_piper_vits/vits_common.py`'s own `get_relative_embeddings` (host-side dynamic-T table
construction) directly via sys.path, same "import the other tool's already-verified helper" precedent as
convert_styletts2_reused.py's reuse of Kokoro's own conversion scripts.

Usage: python3 convert_supertonic_relpos_attn.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

from supertonic_common import TopologyBuilder, to_f32, write_gguf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_piper_vits"))
from vits_common import get_relative_embeddings  # noqa: E402

CHANNELS = 64
N_HEADS = 2
WINDOW_SIZE = 4
HEAD_DIM = CHANNELS // N_HEADS


def add_conv1x1_as_matmul(tb, prefix, sd, name):
    w = to_f32(sd[f"{name}.weight"]).squeeze(-1)  # (out,in,1) -> (out,in)
    b = to_f32(sd[f"{name}.bias"])
    return tb.weight(f"{prefix}.weight", w), tb.weight(f"{prefix}.bias", b)


def build_relpos_attn(tb, sd, T):
    """`x` is declared Layout A [T,64] (T=ne[0] fastest -- matches the real module's own native PyTorch
    (B,64,T) channel-first input BYTE-FOR-BYTE, since T is the fastest-varying/last torch axis there).
    This is a DIFFERENT convention from StyleCrossAttention's own `kv` (which needed Layout B because the
    real module transposes internally BEFORE its Linear ops) -- `MultiHeadRelativeAttention.forward`
    does NOT transpose x at all, so crossing to Layout B [64,T] (channels=ne[0], what conv1x1-as-matmul
    needs) is this script's own job, same "cross the boundary with PERMUTE+CONT" pattern used everywhere
    in this project. (A real bug caught here: an earlier version of this script declared `x` directly as
    Layout B without this crossing, silently reinterpreting T-fastest data as if channels were fastest --
    caught via a real numerical mismatch against the reference, not by inspection.)
    emb_rel_k/v tables are HOST-COMPUTED for this SPECIFIC T (via get_relative_embeddings) and baked as
    fixed constants here -- this standalone verification targets one fixed T, unlike the eventual driver
    which recomputes them per call (see VITS's own convert_vits.py precedent: declared inputs there,
    since T varies per real utterance)."""
    x_cb_p = tb.node("PERMUTE", ["x"], {"axes": [1, 0, 2, 3]}, "x_cb_p")
    x_cb = tb.node("CONT", [x_cb_p], None, "x_cb")  # [64, T] Layout B

    qw, qb = add_conv1x1_as_matmul(tb, "attn.q", sd, "conv_q")
    kw, kb = add_conv1x1_as_matmul(tb, "attn.k", sd, "conv_k")
    vw, vb = add_conv1x1_as_matmul(tb, "attn.v", sd, "conv_v")
    ow, ob = add_conv1x1_as_matmul(tb, "attn.o", sd, "conv_o")

    q = tb.node("ADD", [tb.node("MUL_MAT", [qw, x_cb], None, "q_mm"), qb], None, "q_b")
    k = tb.node("ADD", [tb.node("MUL_MAT", [kw, x_cb], None, "k_mm"), kb], None, "k_b")
    v = tb.node("ADD", [tb.node("MUL_MAT", [vw, x_cb], None, "v_mm"), vb], None, "v_b")
    q = tb.node("RESHAPE", [q], {"shape": [HEAD_DIM, N_HEADS, T]}, "q_r")
    k = tb.node("RESHAPE", [k], {"shape": [HEAD_DIM, N_HEADS, T]}, "k_r")
    v = tb.node("RESHAPE", [v], {"shape": [HEAD_DIM, N_HEADS, T]}, "v_r")

    emb_rel_k_raw = to_f32(sd["emb_rel_k"]).squeeze(0)  # (9,32)
    emb_rel_v_raw = to_f32(sd["emb_rel_v"]).squeeze(0)
    emb_rel_k_t = get_relative_embeddings(emb_rel_k_raw, WINDOW_SIZE, T)  # (2T-1, 32)
    emb_rel_v_t = get_relative_embeddings(emb_rel_v_raw, WINDOW_SIZE, T)
    ek = tb.weight("attn.emb_rel_k", emb_rel_k_t)
    ev = tb.weight("attn.emb_rel_v", emb_rel_v_t)

    mask = tb.weight("attn.mask_zero", np.zeros((T, T), dtype=np.float32))
    attn = tb.node("REL_POS_ATTENTION_SHAW", [q, k, v, ek, ev, mask],
                   {"scale": 1.0 / float(np.sqrt(HEAD_DIM))}, "attn_out")
    out_cb = tb.node("ADD", [tb.node("MUL_MAT", [ow, attn], None, "o_mm"), ob], None, "out_cb")  # [C,T] Layout B

    out_p = tb.node("PERMUTE", [out_cb], {"axes": [1, 0, 2, 3]}, "out_p")
    return tb.node("CONT", [out_p], None, "out")  # [T,C] Layout A -- matches the real (B,C,T) output byte-for-byte


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    dp = torch.load(repo_root / "assets/pt/duration_predictor.pt", weights_only=False, map_location="cpu")
    sd = dp.sentence_encoder.attn_layers[0].state_dict()

    T = 15
    tb = TopologyBuilder()
    out = build_relpos_attn(tb, sd, T)
    # "x" is Layout A [T,C] (T=ne[0]) -- matches the real (B,64,T) input's own memory layout byte-for-byte.
    inputs = [{"name": "x", "dtype": "f32", "shape": [str(T), str(CHANNELS)]}]
    write_gguf(out_dir / "supertonic_relpos_attn.gguf", tb.topology(inputs, out), tb.weights,
               "loom-supertonic-relpos-attn")


if __name__ == "__main__":
    main()
