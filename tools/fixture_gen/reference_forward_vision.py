#!/usr/bin/env python3
"""Independent numpy re-implementation of the toy vision encoder's forward pass (see
toy_vision_common.py), used as the ground truth test_e2e_toy_vision.cpp compares loom-engine's C++ output
against.

Requires: pip install numpy
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_common as enc
import toy_vision_common as common


def conv2d(data: np.ndarray, kernel: np.ndarray, stride, padding) -> np.ndarray:
    """data: (N,IC,IH,IW), kernel: (OC,IC,KH,KW) -> (N,OC,OH,OW). Small/naive on purpose -- these toy
    fixtures are a handful of pixels/patches, clarity matters far more than speed here."""
    n, ic, ih, iw = data.shape
    oc, ic2, kh, kw = kernel.shape
    assert ic == ic2
    s_h, s_w = stride
    p_h, p_w = padding
    if p_h or p_w:
        data = np.pad(data, ((0, 0), (0, 0), (p_h, p_h), (p_w, p_w)))
    oh = (ih + 2 * p_h - kh) // s_h + 1
    ow = (iw + 2 * p_w - kw) // s_w + 1

    out = np.zeros((n, oc, oh, ow), dtype=np.float32)
    for ni in range(n):
        for o in range(oc):
            for y in range(oh):
                for x in range(ow):
                    patch = data[ni, :, y * s_h:y * s_h + kh, x * s_w:x * s_w + kw]
                    out[ni, o, y, x] = np.sum(patch * kernel[o])
    return out


def forward() -> np.ndarray:
    hp = common.hparams()
    weights = common.generate_weights()
    image = common.generate_image()

    conv_out = conv2d(image, weights["patch_embed.weight"], stride=(common.PATCH, common.PATCH), padding=(0, 0))
    n, oc, oh, ow = conv_out.shape
    # Matches the engine's CONV_2D -> PERMUTE([1,2,0,3]) -> CONT -> RESHAPE sequence: ggml's conv output
    # ne=[OW,OH,OC,N] gets flattened to [n_embd, n_tokens] with tokens ordered OW-fastest, then OH, then
    # N -- transpose(0,2,3,1) + reshape reproduces that exact token ordering in numpy.
    cur = conv_out.transpose(0, 2, 3, 1).reshape(n * oh * ow, oc)

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
