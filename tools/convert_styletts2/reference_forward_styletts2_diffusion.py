"""Hand-rolled PyTorch ground truth for StyleTTS2's `Transformer1d` diffusion denoiser (real source:
Modules/diffusion/modules.py's `Transformer1d`/`TransformerBlock`/`Attention`/`AttentionBase`/
`FeedForward`/`TimePositionalEmbedding`/`LearnedPositionalEmbedding`, re-typed here directly rather than
importing the real package -- same "no framework dependency, hand-copy the real math" precedent as every
other reference script in this project). Loads REAL checkpoint weights (like
convert_styletts2_diffusion.py itself), but drives with a SYNTHETIC `embedding` (no real BERT forward
needed to check the denoiser network's OWN math in isolation) -- same "real weights + synthetic driving
input" scope as this project's individual-piece verification precedent (a notch better than Kokoro's own
Generator/Decoder-core tests, which used synthetic weights too).

`embedding_scale=1.0` is assumed throughout (the real demo's own basic-synthesis value) -- the
classifier-free-guidance branch (`embedding_scale != 1.0`) is out of scope, matching PLAN.md's own
basic-synthesis-first scope decision.

Usage: python3 reference_forward_styletts2_diffusion.py <epoch_2nd_00100.pth> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HP = {
    "channels": 256,
    "context_embedding_features": 768,
    "num_layers": 3,
    "num_heads": 8,
    "head_features": 64,
    "mid_features": 512,
    "multiplier": 2,
}


def learned_positional_embedding(time, weights):
    # time: scalar tensor (shape ()); weights: (half_dim,)
    x = time.view(1)                       # [1]
    freqs = x * weights.unsqueeze(0) * 2 * np.pi   # [1, half_dim]
    fouriered = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [1, dim]
    fouriered = torch.cat([x.unsqueeze(0), fouriered], dim=-1)  # [1, dim+1]
    return fouriered.squeeze(0)  # [dim+1]


def linear(x, w, b=None):
    out = x @ w.t()
    if b is not None:
        out = out + b
    return out


def attention_base(q, k, v, num_heads, head_dim, to_out_w, to_out_b):
    T = q.shape[0]
    q = q.view(T, num_heads, head_dim).transpose(0, 1)  # [h,T,d]
    k = k.view(T, num_heads, head_dim).transpose(0, 1)
    v = v.view(T, num_heads, head_dim).transpose(0, 1)
    scale = head_dim ** -0.5
    sim = torch.einsum("h n d, h m d -> h n m", q, k) * scale
    attn = sim.softmax(dim=-1)
    out = torch.einsum("h n m, h m d -> h n d", attn, v)  # [h,T,d]
    out = out.transpose(0, 1).reshape(T, num_heads * head_dim)  # [T, mid]
    return linear(out, to_out_w, to_out_b)


def attention(x, sd, prefix, hp):
    features = x.shape[-1]
    norm_w, norm_b = sd[f"{prefix}.norm.weight"], sd[f"{prefix}.norm.bias"]
    normctx_w, normctx_b = sd[f"{prefix}.norm_context.weight"], sd[f"{prefix}.norm_context.bias"]
    xn = F.layer_norm(x, (features,), norm_w, norm_b, eps=1e-5)
    ctxn = F.layer_norm(x, (features,), normctx_w, normctx_b, eps=1e-5)

    q = linear(xn, sd[f"{prefix}.to_q.weight"])
    kv = linear(ctxn, sd[f"{prefix}.to_kv.weight"])
    mid = hp["mid_features"]
    k, v = kv[:, :mid], kv[:, mid:]
    return attention_base(q, k, v, hp["num_heads"], hp["head_features"],
                           sd[f"{prefix}.attention.to_out.weight"], sd[f"{prefix}.attention.to_out.bias"])


def feed_forward(x, sd, prefix):
    h = linear(x, sd[f"{prefix}.0.weight"], sd[f"{prefix}.0.bias"])
    h = F.gelu(h)
    return linear(h, sd[f"{prefix}.2.weight"], sd[f"{prefix}.2.bias"])


def transformer_block(x, sd, prefix, hp):
    x = attention(x, sd, f"{prefix}.attention", hp) + x
    x = feed_forward(x, sd, f"{prefix}.feed_forward") + x
    return x


def transformer1d_forward(x_in, time_scalar, embedding, sd, prefix, hp):
    """x_in: [channels]; time_scalar: scalar tensor; embedding: [T, context_embedding_features].
    Returns [channels] (the raw model_out, KDiffusion preconditioning NOT applied here)."""
    T = embedding.shape[0]
    channels = hp["channels"]

    lpe_weights = sd[f"{prefix}.to_time.0.0.weights"]
    fouriered_full = learned_positional_embedding(time_scalar, lpe_weights)  # [channels+1]
    time_emb = linear(fouriered_full, sd[f"{prefix}.to_time.0.1.weight"], sd[f"{prefix}.to_time.0.1.bias"])
    time_emb = F.gelu(time_emb)

    mapping = linear(time_emb, sd[f"{prefix}.to_mapping.0.weight"], sd[f"{prefix}.to_mapping.0.bias"])
    mapping = F.gelu(mapping)
    mapping = linear(mapping, sd[f"{prefix}.to_mapping.2.weight"], sd[f"{prefix}.to_mapping.2.bias"])
    mapping = F.gelu(mapping)  # [features]

    x_rep = x_in.unsqueeze(0).expand(T, -1)          # [T, channels]
    x = torch.cat([x_rep, embedding], dim=-1)         # [T, features]

    for i in range(hp["num_layers"]):
        x = x + mapping.unsqueeze(0)
        x = transformer_block(x, sd, f"{prefix}.blocks.{i}", hp)

    x_mean = x.mean(dim=0)  # [features]
    to_out_w = sd[f"{prefix}.to_out.1.weight"].squeeze(-1)  # (channels, features, 1) -> (channels, features)
    to_out_b = sd[f"{prefix}.to_out.1.bias"]
    model_out = linear(x_mean, to_out_w, to_out_b)  # [channels]
    return model_out


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <epoch_2nd_00100.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net = sd_all["net"] if "net" in sd_all else sd_all
    sd = net["diffusion"]
    prefix = "module.unet"

    torch.manual_seed(0)
    T = 6
    x_in = torch.randn(HP["channels"])
    time_scalar = torch.tensor(0.7)
    embedding = torch.randn(T, HP["context_embedding_features"])

    model_out = transformer1d_forward(x_in, time_scalar, embedding, sd, prefix, HP)

    x_in.numpy().astype(np.float32).tofile(out_dir / "diff_x_in.bin")
    np.array([time_scalar.item()], dtype=np.float32).tofile(out_dir / "diff_time.bin")
    embedding.numpy().astype(np.float32).tofile(out_dir / "diff_embedding.bin")
    model_out.detach().numpy().astype(np.float32).tofile(out_dir / "diff_expected_model_out.bin")
    print(f"T={T}, model_out[:5]={model_out[:5].tolist()}")


if __name__ == "__main__":
    main()
