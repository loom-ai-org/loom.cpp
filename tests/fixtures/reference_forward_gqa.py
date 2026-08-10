#!/usr/bin/env python3
"""Independent numpy ground truth for the GQA+QK-norm regression fixture (see gqa_test_common.py).

Adapted from reference_forward.py's forward() (same RoPE/GQA-attention/SwiGLU math, already generic over
any n_head/n_head_kv ratio via `group = n_head // n_head_kv; kv_h = h // group`) with one addition: a
per-head RMSNorm on q/k immediately after reshape, before RoPE -- matching this fixture's topology
(gqa_test_common.py's build_topology(), which inserts RMS_NORM+MUL nodes in that same spot).

Requires: pip install numpy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gqa_test_common as common
from reference_forward import rope_neox, silu


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

        # QK-norm: per-head RMSNorm (last axis == head_dim), weight shape (head_dim,) broadcasts over
        # both the head and token axes -- exactly what gqa_test_common.py's RMS_NORM+MUL nodes compute.
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
    logits = cur @ weights["output.weight"].T
    return logits.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--prompt", type=int, nargs="+", required=True)
    parser.add_argument("--n-new", type=int, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = common.generate_weights()
    hp = common.hparams()

    seq = list(args.prompt)
    generated = []
    for step in range(args.n_new):
        logits = forward(seq, weights, hp)
        last_logits = logits[-1]
        last_logits.tofile(out_dir / f"expected_logits_step{step}.bin")
        next_tok = int(np.argmax(last_logits))
        generated.append(next_tok)
        seq.append(next_tok)

    (out_dir / "expected_generated_tokens.json").write_text(json.dumps(generated))


if __name__ == "__main__":
    main()
