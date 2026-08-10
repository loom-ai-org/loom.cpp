"""Shared hyperparameters, weights, synthetic image, and JSON graph topology for the Milestone-2 toy
vision encoder fixture (CONV_2D patch embed -> non-causal self-attention encoder blocks -> GELU MLP).
Generic scope: validates the CONV_2D/GELU/no-cache-ATTENTION primitives via a ViT-*shaped* encoder, not a
faithful ViT reproduction (no class token, no learned position embeddings -- see BACKLOG.md).

Imported by both make_toy_vision_gguf.py (writes the .gguf) and reference_forward_vision.py (computes the
same forward pass in pure numpy), so the two are guaranteed to agree on weights/image (same seeds, same
RNG call order) without the reference ever parsing the GGUF binary format back out.
"""
import json

import numpy as np

N_EMBD = 8
N_LAYER = 2
N_HEAD = 2
N_FF = 16
HEAD_DIM = N_EMBD // N_HEAD
RMS_NORM_EPS = 1e-5

IMG_C, IMG_H, IMG_W = 3, 8, 8
PATCH = 4
N_TOKENS = (IMG_H // PATCH) * (IMG_W // PATCH)  # 4 patches

SEED = 2024


def hparams() -> dict:
    return {
        "n_embd": N_EMBD, "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd_head": HEAD_DIM,
        "n_ff": N_FF, "rms_norm_eps": RMS_NORM_EPS,
    }


def generate_image() -> np.ndarray:
    """(N, IC, IH, IW) = (1, 3, 8, 8) -- gguf-py reverses this into ggml ne=[IW,IH,IC,N], the layout
    CONV_2D's `data` operand expects."""
    rng = np.random.default_rng(SEED + 1)
    return rng.normal(scale=1.0, size=(1, IMG_C, IMG_H, IMG_W)).astype(np.float32)


def generate_weights() -> dict:
    rng = np.random.default_rng(SEED)

    def rnd(*shape):
        return rng.normal(scale=0.1, size=shape).astype(np.float32)

    w = {"patch_embed.weight": rnd(N_EMBD, IMG_C, PATCH, PATCH)}  # (OC,IC,KH,KW) -> ne=[KW,KH,IC,OC]
    for i in range(N_LAYER):
        w[f"blk.{i}.attn_norm.weight"] = rnd(N_EMBD)
        w[f"blk.{i}.attn_q.weight"] = rnd(N_EMBD, N_EMBD)
        w[f"blk.{i}.attn_k.weight"] = rnd(N_EMBD, N_EMBD)
        w[f"blk.{i}.attn_v.weight"] = rnd(N_EMBD, N_EMBD)
        w[f"blk.{i}.attn_output.weight"] = rnd(N_EMBD, N_EMBD)
        w[f"blk.{i}.ffn_norm.weight"] = rnd(N_EMBD)
        w[f"blk.{i}.mlp_fc1.weight"] = rnd(N_FF, N_EMBD)
        w[f"blk.{i}.mlp_fc2.weight"] = rnd(N_EMBD, N_FF)
    w["output_norm.weight"] = rnd(N_EMBD)
    return w


def build_topology() -> dict:
    return {
        "version": 1,
        "inputs": [
            {"name": "kq_mask", "dtype": "f32", "shape": ["n_tokens", "n_tokens"]},
        ],
        "output": "cur",
        "nodes": [
            # Patch embed: CONV_2D -> [OW,OH,OC,N] -> permute/cont/reshape -> [n_embd, n_tokens].
            {"op": "CONV_2D", "inputs": ["patch_embed.weight", "image.data"], "outputs": ["patches"],
             "attrs": {"s0": PATCH, "s1": PATCH, "p0": 0, "p1": 0, "d0": 1, "d1": 1}},
            {"op": "PERMUTE", "inputs": ["patches"], "outputs": ["patches"], "attrs": {"axes": [1, 2, 0, 3]}},
            {"op": "CONT", "inputs": ["patches"], "outputs": ["patches"]},
            {"op": "RESHAPE", "inputs": ["patches"], "outputs": ["cur"], "attrs": {"shape": ["n_embd", "n_tokens"]}},

            {"repeat_for": "$n_layer", "index_var": "i", "nodes": [
                {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["attn_normed"], "attrs": {"eps": "$rms_norm_eps"}},
                {"op": "MUL", "inputs": ["attn_normed", "blk.{i}.attn_norm.weight"], "outputs": ["attn_normed"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_q.weight", "attn_normed"], "outputs": ["q"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_k.weight", "attn_normed"], "outputs": ["k"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_v.weight", "attn_normed"], "outputs": ["v"]},
                {"op": "RESHAPE", "inputs": ["q"], "outputs": ["q"], "attrs": {"shape": ["n_embd_head", "n_head", "n_tokens"]}},
                {"op": "RESHAPE", "inputs": ["k"], "outputs": ["k"], "attrs": {"shape": ["n_embd_head", "n_head", "n_tokens"]}},
                {"op": "RESHAPE", "inputs": ["v"], "outputs": ["v"], "attrs": {"shape": ["n_embd_head", "n_head", "n_tokens"]}},
                {"op": "ATTENTION", "inputs": ["q", "k", "v", "kq_mask"], "outputs": ["attn_out"],
                 "attrs": {"kv_cache": False, "scale": "1/sqrt($n_embd_head)"}},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_output.weight", "attn_out"], "outputs": ["attn_proj"]},
                {"op": "ADD", "inputs": ["cur", "attn_proj"], "outputs": ["cur"]},
                {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["ffn_normed"], "attrs": {"eps": "$rms_norm_eps"}},
                {"op": "MUL", "inputs": ["ffn_normed", "blk.{i}.ffn_norm.weight"], "outputs": ["ffn_normed"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.mlp_fc1.weight", "ffn_normed"], "outputs": ["ffn_hidden"]},
                {"op": "GELU", "inputs": ["ffn_hidden"], "outputs": ["ffn_hidden"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.mlp_fc2.weight", "ffn_hidden"], "outputs": ["ffn_out"]},
                {"op": "ADD", "inputs": ["cur", "ffn_out"], "outputs": ["cur"]},
            ]},

            {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["cur"], "attrs": {"eps": "$rms_norm_eps"}},
            {"op": "MUL", "inputs": ["cur", "output_norm.weight"], "outputs": ["cur"]},
        ],
    }


def topology_json() -> str:
    return json.dumps(build_topology())
