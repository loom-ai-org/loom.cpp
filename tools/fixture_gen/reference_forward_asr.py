#!/usr/bin/env python3
"""Independent numpy re-implementation of the toy ASR encoder's forward pass (see toy_asr_common.py),
used as the ground truth test_e2e_toy_asr.cpp compares loom-engine's C++ output against.

Requires: pip install numpy
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_common as enc
import toy_asr_common as common


def forward() -> np.ndarray:
    hp = common.hparams()
    weights = common.generate_weights()
    features = common.generate_features()

    conv_out = enc.conv1d(features, weights["conv_subsample.weight"], stride=common.STRIDE, padding=common.PADDING)
    n, oc, ol = conv_out.shape
    # Matches the engine's CONV_1D -> PERMUTE([1,0,2,3]) -> CONT -> RESHAPE sequence: ggml's conv output
    # ne=[OL,OC,N] gets flattened to [n_embd, n_tokens] with tokens ordered OL-fastest, then N --
    # transpose(0,2,1) + reshape reproduces that exact token ordering in numpy.
    cur = conv_out.transpose(0, 2, 1).reshape(n * ol, oc)

    eps = hp["rms_norm_eps"]
    scale = 1.0 / np.sqrt(hp["n_embd_head"])
    for i in range(hp["n_layer"]):
        attn_normed = enc.rms_norm(cur, eps) * weights[f"blk.{i}.attn_norm.weight"]
        q = attn_normed @ weights[f"blk.{i}.attn_q.weight"].T
        k = attn_normed @ weights[f"blk.{i}.attn_k.weight"].T
        v = attn_normed @ weights[f"blk.{i}.attn_v.weight"].T
        attn_out = enc.multi_head_self_attention(q, k, v, hp["n_head"], scale)
        attn_proj = attn_out @ weights[f"blk.{i}.attn_output.weight"].T
        cur = cur + attn_proj

        ffn_normed = enc.rms_norm(cur, eps) * weights[f"blk.{i}.ffn_norm.weight"]
        hidden = enc.gelu_erf(ffn_normed @ weights[f"blk.{i}.mlp_fc1.weight"].T)
        ffn_out = hidden @ weights[f"blk.{i}.mlp_fc2.weight"].T
        cur = cur + ffn_out

    cur = enc.rms_norm(cur, eps) * weights["output_norm.weight"]
    return cur.astype(np.float32)  # (n_tokens, n_embd)


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    forward().tofile(out_dir / "expected_output.bin")


if __name__ == "__main__":
    main()
