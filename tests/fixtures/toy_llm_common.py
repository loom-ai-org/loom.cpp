"""Shared hyperparameters, weight generation, and JSON graph topology for the Milestone-1 toy LLM
fixture, used by both make_toy_llm_gguf.py (writes the .gguf) and reference_forward.py (computes the
same forward pass in pure numpy). Both import this module rather than one re-deriving the other's
output, so the two are guaranteed to agree on weights (same seed, same RNG call order) without
reference_forward.py ever having to parse the GGUF binary format back out.
"""
import json

import numpy as np

N_VOCAB = 16
N_EMBD = 8
N_LAYER = 2
N_HEAD = 2
N_HEAD_KV = 2
N_FF = 16
N_CTX_TRAIN = 32
HEAD_DIM = N_EMBD // N_HEAD

ROPE_FREQ_BASE = 10000.0
ROPE_FREQ_SCALE = 1.0
RMS_NORM_EPS = 1e-5

SEED = 1234


def hparams() -> dict:
    return {
        "n_vocab": N_VOCAB, "n_embd": N_EMBD, "n_layer": N_LAYER, "n_head": N_HEAD,
        "n_head_kv": N_HEAD_KV, "n_embd_head_k": HEAD_DIM, "n_embd_head_v": HEAD_DIM,
        "n_ff": N_FF, "n_ctx_train": N_CTX_TRAIN, "rope_dims": HEAD_DIM,
        "rope_freq_base": ROPE_FREQ_BASE, "rope_freq_scale": ROPE_FREQ_SCALE,
        "rms_norm_eps": RMS_NORM_EPS,
    }


def generate_weights() -> dict:
    """Returns every weight as a numpy array in its natural (non-ggml-reversed) shape, e.g. a Linear
    layer's weight is (n_out, n_in) as usual -- MUL_MAT(W, x) computes x @ W.T (see reference_forward.py)."""
    rng = np.random.default_rng(SEED)

    def rnd(*shape):
        return rng.normal(scale=0.1, size=shape).astype(np.float32)

    w = {"token_embd.weight": rnd(N_VOCAB, N_EMBD)}
    for i in range(N_LAYER):
        w[f"blk.{i}.attn_norm.weight"] = rnd(N_EMBD)
        w[f"blk.{i}.attn_q.weight"] = rnd(N_HEAD * HEAD_DIM, N_EMBD)
        w[f"blk.{i}.attn_k.weight"] = rnd(N_HEAD_KV * HEAD_DIM, N_EMBD)
        w[f"blk.{i}.attn_v.weight"] = rnd(N_HEAD_KV * HEAD_DIM, N_EMBD)
        w[f"blk.{i}.attn_output.weight"] = rnd(N_EMBD, N_HEAD * HEAD_DIM)
        w[f"blk.{i}.ffn_norm.weight"] = rnd(N_EMBD)
        w[f"blk.{i}.ffn_gate.weight"] = rnd(N_FF, N_EMBD)
        w[f"blk.{i}.ffn_up.weight"] = rnd(N_FF, N_EMBD)
        w[f"blk.{i}.ffn_down.weight"] = rnd(N_EMBD, N_FF)
    w["output_norm.weight"] = rnd(N_EMBD)
    w["output.weight"] = rnd(N_VOCAB, N_EMBD)
    return w


def build_topology() -> dict:
    return {
        "version": 1,
        "inputs": [
            {"name": "tokens", "dtype": "i32", "shape": ["n_tokens"]},
            {"name": "positions", "dtype": "i32", "shape": ["n_tokens"]},
            {"name": "kq_mask", "dtype": "f32", "shape": ["n_kv", "n_tokens"]},
        ],
        "output": "logits",
        "nodes": [
            {"op": "GET_ROWS", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"]},
            {"repeat_for": "$n_layer", "index_var": "i", "nodes": [
                {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["attn_normed"], "attrs": {"eps": "$rms_norm_eps"}},
                {"op": "MUL", "inputs": ["attn_normed", "blk.{i}.attn_norm.weight"], "outputs": ["attn_normed"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_q.weight", "attn_normed"], "outputs": ["q"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_k.weight", "attn_normed"], "outputs": ["k"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_v.weight", "attn_normed"], "outputs": ["v"]},
                {"op": "RESHAPE", "inputs": ["q"], "outputs": ["q"], "attrs": {"shape": ["n_embd_head_k", "n_head", "n_tokens"]}},
                {"op": "RESHAPE", "inputs": ["k"], "outputs": ["k"], "attrs": {"shape": ["n_embd_head_k", "n_head_kv", "n_tokens"]}},
                {"op": "RESHAPE", "inputs": ["v"], "outputs": ["v"], "attrs": {"shape": ["n_embd_head_v", "n_head_kv", "n_tokens"]}},
                {"op": "ROPE", "inputs": ["q", "positions"], "outputs": ["q"], "attrs": {
                    "n_dims": "$rope_dims", "mode": 2, "n_ctx_orig": "$n_ctx_train",
                    "freq_base": "$rope_freq_base", "freq_scale": "$rope_freq_scale",
                    "ext_factor": 0.0, "attn_factor": 1.0, "beta_fast": 32.0, "beta_slow": 1.0,
                }},
                {"op": "ROPE", "inputs": ["k", "positions"], "outputs": ["k"], "attrs": {
                    "n_dims": "$rope_dims", "mode": 2, "n_ctx_orig": "$n_ctx_train",
                    "freq_base": "$rope_freq_base", "freq_scale": "$rope_freq_scale",
                    "ext_factor": 0.0, "attn_factor": 1.0, "beta_fast": 32.0, "beta_slow": 1.0,
                }},
                {"op": "ATTENTION", "inputs": ["q", "k", "v", "kq_mask"], "outputs": ["attn_out"],
                 "attrs": {"layer": "{i}", "scale": "1/sqrt($n_embd_head_k)"}},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_output.weight", "attn_out"], "outputs": ["attn_proj"]},
                {"op": "ADD", "inputs": ["cur", "attn_proj"], "outputs": ["cur"]},
                {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["ffn_normed"], "attrs": {"eps": "$rms_norm_eps"}},
                {"op": "MUL", "inputs": ["ffn_normed", "blk.{i}.ffn_norm.weight"], "outputs": ["ffn_normed"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.ffn_gate.weight", "ffn_normed"], "outputs": ["ffn_gate"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.ffn_up.weight", "ffn_normed"], "outputs": ["ffn_up"]},
                {"op": "SWIGLU", "inputs": ["ffn_gate", "ffn_up"], "outputs": ["ffn_act"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.ffn_down.weight", "ffn_act"], "outputs": ["ffn_out"]},
                {"op": "ADD", "inputs": ["cur", "ffn_out"], "outputs": ["cur"]},
            ]},
            {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["cur"], "attrs": {"eps": "$rms_norm_eps"}},
            {"op": "MUL", "inputs": ["cur", "output_norm.weight"], "outputs": ["cur"]},
            {"op": "MUL_MAT", "inputs": ["output.weight", "cur"], "outputs": ["logits"]},
        ],
    }


def topology_json() -> str:
    return json.dumps(build_topology())
