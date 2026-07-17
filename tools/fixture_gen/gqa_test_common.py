"""Shared hyperparameters, weight generation, and JSON graph topology for the GQA+QK-norm regression
fixture, used by both make_gqa_test_gguf.py and reference_forward_gqa.py.

A standalone sibling of toy_llm_common.py (deliberately not sharing code with it, same convention as
tests/fixtures/make_attention_test_gguf.py's own self-contained smaller fixture) -- its two real
differences are (1) N_HEAD_KV < N_HEAD (4 query heads sharing 2 KV heads, 2:1 grouping), which the toy LLM
fixture never exercised (it uses N_HEAD == N_HEAD_KV == 2), and (2) per-head QK-norm (RMSNorm on q/k
before RoPE, a Qwen3-specific addition the toy LLM fixture doesn't have at all). This is BACKLOG.md's
"verify before trusting an existing mechanism in a new configuration" regression test for both: ATTENTION's
GQA broadcast (src/ops/primitives_attention.cpp's `ggml_mul_mat(kp, qp)`, relying on ggml's own
n_head_kv -> n_head broadcast rule) and RMS_NORM/MUL's broadcast over a [head_dim, n_head, n_tokens]
tensor with a [head_dim]-shaped weight (two extra broadcast dims, vs. the one dim every other existing
norm-weight usage in this codebase exercises) -- both run BEFORE the real Qwen3-0.6B-Base checkpoint (16
query / 8 KV heads, real QK-norm weights) depends on either.
"""
import json

import numpy as np

N_VOCAB = 16
N_EMBD = 8
N_LAYER = 2
N_HEAD = 4
N_HEAD_KV = 2  # 2:1 grouping -- the one thing this fixture exists to exercise
N_FF = 16
N_CTX_TRAIN = 32
HEAD_DIM = N_EMBD // N_HEAD

ROPE_FREQ_BASE = 10000.0
ROPE_FREQ_SCALE = 1.0
RMS_NORM_EPS = 1e-5

SEED = 4242


def hparams() -> dict:
    return {
        "n_vocab": N_VOCAB, "n_embd": N_EMBD, "n_layer": N_LAYER, "n_head": N_HEAD,
        "n_head_kv": N_HEAD_KV, "n_embd_head_k": HEAD_DIM, "n_embd_head_v": HEAD_DIM,
        "n_ff": N_FF, "n_ctx_train": N_CTX_TRAIN, "rope_dims": HEAD_DIM,
        "rope_freq_base": ROPE_FREQ_BASE, "rope_freq_scale": ROPE_FREQ_SCALE,
        "rms_norm_eps": RMS_NORM_EPS,
    }


def generate_weights() -> dict:
    rng = np.random.default_rng(SEED)

    def rnd(*shape):
        return rng.normal(scale=0.1, size=shape).astype(np.float32)

    w = {"token_embd.weight": rnd(N_VOCAB, N_EMBD)}
    for i in range(N_LAYER):
        w[f"blk.{i}.attn_norm.weight"] = rnd(N_EMBD)
        w[f"blk.{i}.attn_q.weight"] = rnd(N_HEAD * HEAD_DIM, N_EMBD)
        w[f"blk.{i}.attn_k.weight"] = rnd(N_HEAD_KV * HEAD_DIM, N_EMBD)
        w[f"blk.{i}.attn_v.weight"] = rnd(N_HEAD_KV * HEAD_DIM, N_EMBD)
        # QK-norm weights: RMSNorm applied per-head (over HEAD_DIM only), so shape is [HEAD_DIM] -- not
        # [N_HEAD*HEAD_DIM] -- broadcast across every head and token by the topology's MUL node below.
        w[f"blk.{i}.attn_q_norm.weight"] = rnd(HEAD_DIM)
        w[f"blk.{i}.attn_k_norm.weight"] = rnd(HEAD_DIM)
        w[f"blk.{i}.attn_output.weight"] = rnd(N_EMBD, N_HEAD * HEAD_DIM)
        w[f"blk.{i}.ffn_norm.weight"] = rnd(N_EMBD)
        w[f"blk.{i}.ffn_gate.weight"] = rnd(N_FF, N_EMBD)
        w[f"blk.{i}.ffn_up.weight"] = rnd(N_FF, N_EMBD)
        w[f"blk.{i}.ffn_down.weight"] = rnd(N_EMBD, N_FF)
    w["output_norm.weight"] = rnd(N_EMBD)
    w["output.weight"] = rnd(N_VOCAB, N_EMBD)
    return w


def build_topology() -> dict:
    # Identical structure to toy_llm_common.py's build_topology() -- every head-count reference here is a
    # "$"-prefixed symbol name (n_head, n_head_kv), never a literal, so this JSON is already fully generic
    # over the query/KV head ratio; only the hparam VALUES fed in via GGUF KVs make this fixture GQA.
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
                # QK-norm (Qwen3-specific): per-head RMSNorm on q/k, before RoPE. RMS_NORM normalizes
                # along ne[0] (HEAD_DIM) independently per (head, token) slice; the weight is [HEAD_DIM],
                # broadcast by MUL across the n_head and n_tokens dims -- this fixture's reason to exist.
                {"op": "RMS_NORM", "inputs": ["q"], "outputs": ["q"], "attrs": {"eps": "$rms_norm_eps"}},
                {"op": "MUL", "inputs": ["q", "blk.{i}.attn_q_norm.weight"], "outputs": ["q"]},
                {"op": "RMS_NORM", "inputs": ["k"], "outputs": ["k"], "attrs": {"eps": "$rms_norm_eps"}},
                {"op": "MUL", "inputs": ["k", "blk.{i}.attn_k_norm.weight"], "outputs": ["k"]},
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
