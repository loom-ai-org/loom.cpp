"""Shared hyperparameters, weights, synthetic input features, and JSON graph topology for the
Milestone-2 toy ASR encoder fixture (CONV_1D subsampling -> non-causal self-attention encoder blocks ->
GELU MLP). Generic scope: validates the CONV_1D/GELU/no-cache-ATTENTION primitives via a conv-subsampled
transformer encoder pattern -- the same general shape Zipformer/Conformer-style ASR encoders use, not a
faithful Zipformer reproduction (no multi-branch downsampling, no bypass modules -- see BACKLOG.md).

Imported by both make_toy_asr_gguf.py (writes the .gguf) and reference_forward_asr.py (computes the same
forward pass in pure numpy), so the two are guaranteed to agree on weights/features (same seeds, same RNG
call order) without the reference ever parsing the GGUF binary format back out.
"""
import json

import numpy as np

N_EMBD = 8
N_LAYER = 2
N_HEAD = 2
N_FF = 16
HEAD_DIM = N_EMBD // N_HEAD
RMS_NORM_EPS = 1e-5

IN_CHANNELS = 4
IN_LENGTH = 16
KERNEL = 3
STRIDE = 2
PADDING = 1
N_TOKENS = (IN_LENGTH + 2 * PADDING - KERNEL) // STRIDE + 1  # 8

SEED = 3030


def hparams() -> dict:
    return {
        "n_embd": N_EMBD, "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd_head": HEAD_DIM,
        "n_ff": N_FF, "rms_norm_eps": RMS_NORM_EPS,
    }


def generate_features() -> np.ndarray:
    """(N, IC, IL) = (1, 4, 16) -- gguf-py reverses this into ggml ne=[IL,IC,N], the layout CONV_1D's
    `data` operand expects."""
    rng = np.random.default_rng(SEED + 1)
    return rng.normal(scale=1.0, size=(1, IN_CHANNELS, IN_LENGTH)).astype(np.float32)


def generate_weights() -> dict:
    rng = np.random.default_rng(SEED)

    def rnd(*shape):
        return rng.normal(scale=0.1, size=shape).astype(np.float32)

    w = {"conv_subsample.weight": rnd(N_EMBD, IN_CHANNELS, KERNEL)}  # (OC,IC,K) -> ne=[K,IC,OC]
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
            # Conv1d subsample: CONV_1D -> [OL,OC,N] -> permute/cont/reshape -> [n_embd, n_tokens].
            {"op": "CONV_1D", "inputs": ["conv_subsample.weight", "features.data"], "outputs": ["frames"],
             "attrs": {"s0": STRIDE, "p0": PADDING, "d0": 1}},
            {"op": "PERMUTE", "inputs": ["frames"], "outputs": ["frames"], "attrs": {"axes": [1, 0, 2, 3]}},
            {"op": "CONT", "inputs": ["frames"], "outputs": ["frames"]},
            {"op": "RESHAPE", "inputs": ["frames"], "outputs": ["cur"], "attrs": {"shape": ["n_embd", "n_tokens"]}},

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
