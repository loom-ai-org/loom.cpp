#!/usr/bin/env python3
"""Independent numpy re-implementation of the toy LLM's forward pass (see toy_llm_common.py), used as
the ground truth test_e2e_toy_llm.cpp compares loom-engine's C++ output against.

Recomputes the FULL non-cached causal forward pass over the whole token sequence at every generation
step, rather than maintaining its own KV cache -- for a causal decoder this is mathematically identical
to incremental cached decoding (attention only ever looks at the causal past), so it exercises the same
math loom-engine's KV-cache path implements without needing a second cache implementation here.

For each generated step, dumps that step's last-position logits as a raw float32 binary blob (no .npy
parsing needed on the C++ side) plus the full greedy-argmax token sequence as a small JSON file.

Requires: pip install numpy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_llm_common as common


def rms_norm(x: np.ndarray, eps: float) -> np.ndarray:
    mean_sq = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(mean_sq + eps)).astype(np.float32)


def rope_neox(x: np.ndarray, positions: np.ndarray, n_dims: int, freq_base: float, freq_scale: float) -> np.ndarray:
    """x: (n_tokens, n_head, head_dim). Rotates the first n_dims dims of each head in NEOX-paired style
    (pairs (i, i + n_dims/2)), matching ggml_rope_ext's GGML_ROPE_TYPE_NEOX branch bit-for-bit given
    ext_factor=0 (no YaRN) and attn_factor=1 (see ggml-cpu/ops.cpp: rope_yarn + rotate_pairs)."""
    half = n_dims // 2
    out = x.copy()
    pos = positions.astype(np.float64)
    for i in range(half):
        theta = pos * freq_scale * (freq_base ** (-2.0 * i / n_dims))
        cos_t = np.cos(theta).astype(np.float32)
        sin_t = np.sin(theta).astype(np.float32)
        x0 = x[:, :, i].copy()
        x1 = x[:, :, i + half].copy()
        out[:, :, i] = x0 * cos_t[:, None] - x1 * sin_t[:, None]
        out[:, :, i + half] = x0 * sin_t[:, None] + x1 * cos_t[:, None]
    return out


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


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
