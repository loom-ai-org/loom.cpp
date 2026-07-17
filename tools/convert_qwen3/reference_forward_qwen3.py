#!/usr/bin/env python3
"""Independent numpy ground truth for a converted Qwen3-0.6B-Base GGUF, used by test_e2e_qwen3.cpp to
verify loom-engine's real-model logits within the usual 1e-3 tolerance.

Reuses tools/fixture_gen/reference_forward.py's rope_neox/silu helpers (already validated against ggml's
own GGML_ROPE_TYPE_NEOX behavior) and the same QK-norm-augmented, GQA-generic attention math already
proven in tools/fixture_gen/reference_forward_gqa.py -- this is that exact same math, just fed the real
checkpoint's weights/hparams/tied-embedding convention instead of a synthetic fixture's.

Avoids `transformers` (see convert_qwen3.py's docstring for why) -- loads weights via `safetensors`
directly and casts BF16 -> F32 via torch (numpy has no native bfloat16), then does the actual forward
pass in plain numpy so this script's own correctness doesn't depend on torch's operator semantics
matching ggml's (only the initial dtype cast does).

Usage: python3 reference_forward_qwen3.py <hf_checkpoint_dir> <out_dir> --prompt <id> [<id> ...] --n-new N
Requires: pip install numpy torch safetensors
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixture_gen"))
from reference_forward import rope_neox, silu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_qwen3 import hparams


def rms_norm(x: np.ndarray, eps: float) -> np.ndarray:
    mean_sq = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(mean_sq + eps)).astype(np.float32)


def forward(tokens: list, weights: dict, hp: dict) -> np.ndarray:
    n_tokens = len(tokens)
    n_head, n_head_kv, head_dim = hp["n_head"], hp["n_head_kv"], hp["n_embd_head_k"]
    eps = hp["rms_norm_eps"]
    positions = np.arange(n_tokens, dtype=np.int32)

    cur = weights["token_embd.weight"][np.array(tokens, dtype=np.int64)]  # (T, n_embd)

    for i in range(hp["n_layer"]):
        attn_normed = rms_norm(cur, eps) * weights[f"blk.{i}.attn_norm.weight"]

        q = attn_normed @ weights[f"blk.{i}.attn_q.weight"].T
        k = attn_normed @ weights[f"blk.{i}.attn_k.weight"].T
        v = attn_normed @ weights[f"blk.{i}.attn_v.weight"].T
        q = q.reshape(n_tokens, n_head, head_dim)
        k = k.reshape(n_tokens, n_head_kv, head_dim)
        v = v.reshape(n_tokens, n_head_kv, head_dim)

        q = rms_norm(q, eps) * weights[f"blk.{i}.attn_q_norm.weight"]
        k = rms_norm(k, eps) * weights[f"blk.{i}.attn_k_norm.weight"]

        q = rope_neox(q, positions, hp["rope_dims"], hp["rope_freq_base"], hp["rope_freq_scale"])
        k = rope_neox(k, positions, hp["rope_dims"], hp["rope_freq_base"], hp["rope_freq_scale"])

        scale = 1.0 / np.sqrt(head_dim)
        causal = np.triu(np.full((n_tokens, n_tokens), -np.inf, dtype=np.float32), k=1)
        group = n_head // n_head_kv
        attn_out = np.zeros((n_tokens, n_head, head_dim), dtype=np.float32)
        for h in range(n_head):
            kv_h = h // group
            scores = (q[:, h, :] @ k[:, kv_h, :].T) * scale + causal
            scores = scores - scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            attn_out[:, h, :] = probs @ v[:, kv_h, :]
        attn_out = attn_out.reshape(n_tokens, n_head * head_dim)

        attn_proj = attn_out @ weights[f"blk.{i}.attn_output.weight"].T
        cur = cur + attn_proj

        ffn_normed = rms_norm(cur, eps) * weights[f"blk.{i}.ffn_norm.weight"]
        gate = ffn_normed @ weights[f"blk.{i}.ffn_gate.weight"].T
        up = ffn_normed @ weights[f"blk.{i}.ffn_up.weight"].T
        act = silu(gate) * up
        ffn_out = act @ weights[f"blk.{i}.ffn_down.weight"].T
        cur = cur + ffn_out

    cur = rms_norm(cur, eps) * weights["output_norm.weight"]
    logits = cur @ weights["token_embd.weight"].T  # tied embeddings
    return logits.astype(np.float32)


def load_weights_numpy(hf_dir: Path, hp: dict) -> dict:
    state = load_file(str(hf_dir / "model.safetensors"))

    def get(hf_key: str) -> np.ndarray:
        return state[hf_key].to(torch.float32).numpy()

    weights = {"token_embd.weight": get("model.embed_tokens.weight")}
    for i in range(hp["n_layer"]):
        prefix = f"model.layers.{i}"
        weights[f"blk.{i}.attn_norm.weight"] = get(f"{prefix}.input_layernorm.weight")
        weights[f"blk.{i}.attn_q.weight"] = get(f"{prefix}.self_attn.q_proj.weight")
        weights[f"blk.{i}.attn_k.weight"] = get(f"{prefix}.self_attn.k_proj.weight")
        weights[f"blk.{i}.attn_v.weight"] = get(f"{prefix}.self_attn.v_proj.weight")
        weights[f"blk.{i}.attn_q_norm.weight"] = get(f"{prefix}.self_attn.q_norm.weight")
        weights[f"blk.{i}.attn_k_norm.weight"] = get(f"{prefix}.self_attn.k_norm.weight")
        weights[f"blk.{i}.attn_output.weight"] = get(f"{prefix}.self_attn.o_proj.weight")
        weights[f"blk.{i}.ffn_norm.weight"] = get(f"{prefix}.post_attention_layernorm.weight")
        weights[f"blk.{i}.ffn_gate.weight"] = get(f"{prefix}.mlp.gate_proj.weight")
        weights[f"blk.{i}.ffn_up.weight"] = get(f"{prefix}.mlp.up_proj.weight")
        weights[f"blk.{i}.ffn_down.weight"] = get(f"{prefix}.mlp.down_proj.weight")
    weights["output_norm.weight"] = get("model.norm.weight")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--prompt", type=int, nargs="+", required=True)
    parser.add_argument("--n-new", type=int, required=True)
    args = parser.parse_args()

    hf_dir = Path(args.hf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((hf_dir / "config.json").read_text())
    hp = hparams(config)
    weights = load_weights_numpy(hf_dir, hp)

    seq = list(args.prompt)
    generated = []
    for step in range(args.n_new):
        logits = forward(seq, weights, hp)
        last_logits = logits[-1]
        last_logits.tofile(out_dir / f"expected_logits_step{step}.bin")
        next_tok = int(np.argmax(last_logits))
        generated.append(next_tok)
        seq.append(next_tok)
        print(f"step {step}: sampled token {next_tok}", file=sys.stderr)

    (out_dir / "expected_generated_tokens.json").write_text(json.dumps(generated))


if __name__ == "__main__":
    main()
