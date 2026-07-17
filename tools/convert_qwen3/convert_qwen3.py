#!/usr/bin/env python3
"""Converts a real Qwen3-0.6B-Base checkpoint (HuggingFace format: config.json + model.safetensors +
tokenizer.json, e.g. as downloaded from https://huggingface.co/Qwen/Qwen3-0.6B-Base) into a loom-engine
GGUF: hparam KVs, the embedded JSON graph topology, the byte-level-BPE vocab (see qwen3_tokenizer.py),
and every weight tensor.

Deliberately avoids the `transformers` library entirely (same precedent as tools/convert_nemo/, which
hand-parses a .nemo archive instead of depending on the NeMo toolkit) -- config.json/tokenizer.json are
plain JSON, and weights load via the `safetensors` package's own torch loader (needed only to convert
BF16 -> F32; numpy has no native bfloat16 support).

Tensor name mapping and shapes were confirmed directly against the real checkpoint's safetensors header
(not assumed): `model.layers.{i}.self_attn.q_proj.weight` is [2048, 1024] (16 heads * 128 head_dim,
up-projected from hidden_size=1024 -- head_dim is an INDEPENDENT hparam here, not n_embd/n_head), k/v_proj
are [1024, 1024] (8 KV heads * 128), q_norm/k_norm are [128] (per-head QK-norm, confirming the same
[head_dim]-shaped-weight design already proven in tests/test_e2e_gqa.cpp's synthetic GQA+QK-norm
fixture), and `tie_word_embeddings=true` means there is no separate lm_head weight in the checkpoint at
all -- the topology's final logits MUL_MAT reuses "token_embd.weight" by name directly (GraphBuilder's
symbol table already resolves a name to the same tensor wherever referenced, so this needs no engine
change and writes no duplicate tensor data).

Usage: python3 convert_qwen3.py <hf_checkpoint_dir> <out.gguf>
Requires: pip install gguf numpy torch safetensors
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen3_tokenizer


def hparams(config: dict) -> dict:
    return {
        "n_vocab": config["vocab_size"],
        "n_embd": config["hidden_size"],
        "n_layer": config["num_hidden_layers"],
        "n_head": config["num_attention_heads"],
        "n_head_kv": config["num_key_value_heads"],
        "n_embd_head_k": config["head_dim"],
        "n_embd_head_v": config["head_dim"],
        "n_ff": config["intermediate_size"],
        "n_ctx_train": config["max_position_embeddings"],
        "rope_dims": config["head_dim"],
        "rope_freq_base": float(config["rope_theta"]),
        "rope_freq_scale": 1.0,
        "rms_norm_eps": float(config["rms_norm_eps"]),
    }


def build_topology(hp: dict) -> dict:
    # Same structure as tools/fixture_gen/gqa_test_common.py's proven QK-norm+GQA topology (verified
    # against numpy in tests/test_e2e_gqa.cpp), plus tied embeddings: the final MUL_MAT reads
    # "token_embd.weight" directly instead of a separate "output.weight" (see module docstring).
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
            {"op": "MUL_MAT", "inputs": ["token_embd.weight", "cur"], "outputs": ["logits"]}, # tied embeddings
        ],
    }


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <hf_checkpoint_dir> <out.gguf>", file=sys.stderr)
        sys.exit(1)
    hf_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads((hf_dir / "config.json").read_text())
    tokenizer_json = json.loads((hf_dir / "tokenizer.json").read_text())
    hp = hparams(config)

    state = load_file(str(hf_dir / "model.safetensors"))

    def put(writer: GGUFWriter, name: str, hf_key: str) -> None:
        t = state[hf_key].to(torch.float32).numpy()
        writer.add_tensor(name, np.ascontiguousarray(t))

    w = GGUFWriter(str(out_path), "loom-qwen3")
    w.add_string("loom.architecture", "qwen3")
    for key in ("n_vocab", "n_embd", "n_layer", "n_head", "n_head_kv", "n_embd_head_k",
                "n_embd_head_v", "n_ff", "n_ctx_train", "rope_dims"):
        w.add_uint32(f"loom.{key}", hp[key])
    for key in ("rope_freq_base", "rope_freq_scale", "rms_norm_eps"):
        w.add_float32(f"loom.{key}", hp[key])
    w.add_string("model.graph_topology", json.dumps(build_topology(hp)))

    qwen3_tokenizer.write_bpe_vocab(
        w, tokenizer_json,
        vocab_size=hp["n_vocab"],
        bos_token_id=config["bos_token_id"],
        eos_token_id=config["eos_token_id"],
    )

    put(w, "token_embd.weight", "model.embed_tokens.weight")
    for i in range(hp["n_layer"]):
        prefix = f"model.layers.{i}"
        put(w, f"blk.{i}.attn_norm.weight", f"{prefix}.input_layernorm.weight")
        put(w, f"blk.{i}.attn_q.weight", f"{prefix}.self_attn.q_proj.weight")
        put(w, f"blk.{i}.attn_k.weight", f"{prefix}.self_attn.k_proj.weight")
        put(w, f"blk.{i}.attn_v.weight", f"{prefix}.self_attn.v_proj.weight")
        put(w, f"blk.{i}.attn_q_norm.weight", f"{prefix}.self_attn.q_norm.weight")
        put(w, f"blk.{i}.attn_k_norm.weight", f"{prefix}.self_attn.k_norm.weight")
        put(w, f"blk.{i}.attn_output.weight", f"{prefix}.self_attn.o_proj.weight")
        put(w, f"blk.{i}.ffn_norm.weight", f"{prefix}.post_attention_layernorm.weight")
        put(w, f"blk.{i}.ffn_gate.weight", f"{prefix}.mlp.gate_proj.weight")
        put(w, f"blk.{i}.ffn_up.weight", f"{prefix}.mlp.up_proj.weight")
        put(w, f"blk.{i}.ffn_down.weight", f"{prefix}.mlp.down_proj.weight")
    put(w, "output_norm.weight", "model.norm.weight")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {out_path} ({hp['n_layer']} layers, n_embd={hp['n_embd']}, "
          f"n_head={hp['n_head']}/{hp['n_head_kv']} (Q/KV), n_vocab={hp['n_vocab']})")


if __name__ == "__main__":
    main()
