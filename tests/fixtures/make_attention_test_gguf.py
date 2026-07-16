#!/usr/bin/env python3
"""Generates a tiny single-layer attention-based toy LLM GGUF for test_generation_smoke.cpp.

Exercises the full milestone-1 op set (GET_ROWS, RMS_NORM, MUL, MUL_MAT, RESHAPE, ROPE, ATTENTION,
SWIGLU, ADD) through GraphBuilder + KvCache + Generator, as an integration smoke test ahead of Phase 4's
larger, numerically-verified-against-numpy toy-LLM fixture.

Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

N_VOCAB, N_EMBD, N_HEAD, N_HEAD_KV, N_LAYER, N_FF, N_CTX_TRAIN = 8, 4, 2, 2, 1, 8, 16
HEAD_DIM = N_EMBD // N_HEAD


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("attention_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    def rnd(*shape):
        return rng.normal(scale=0.1, size=shape).astype(np.float32)

    w = GGUFWriter(str(out_path), "loom-attention-test-fixture")
    w.add_string("loom.architecture", "attention_test")
    w.add_uint32("loom.n_vocab", N_VOCAB)
    w.add_uint32("loom.n_embd", N_EMBD)
    w.add_uint32("loom.n_layer", N_LAYER)
    w.add_uint32("loom.n_head", N_HEAD)
    w.add_uint32("loom.n_head_kv", N_HEAD_KV)
    w.add_uint32("loom.n_embd_head_k", HEAD_DIM)
    w.add_uint32("loom.n_embd_head_v", HEAD_DIM)
    w.add_uint32("loom.n_ff", N_FF)
    w.add_uint32("loom.n_ctx_train", N_CTX_TRAIN)
    w.add_uint32("loom.rope_dims", HEAD_DIM)
    w.add_float32("loom.rope_freq_base", 10000.0)
    w.add_float32("loom.rope_freq_scale", 1.0)
    w.add_float32("loom.rms_norm_eps", 1e-5)

    topology = {
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
    w.add_string("model.graph_topology", json.dumps(topology))

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))
    for i in range(N_LAYER):
        w.add_tensor(f"blk.{i}.attn_norm.weight", rnd(N_EMBD))
        w.add_tensor(f"blk.{i}.attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(f"blk.{i}.attn_k.weight", rnd(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(f"blk.{i}.attn_v.weight", rnd(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(f"blk.{i}.attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(f"blk.{i}.ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(f"blk.{i}.ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(f"blk.{i}.ffn_down.weight", rnd(N_EMBD, N_FF))
    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
