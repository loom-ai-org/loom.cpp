"""Hand-rolled pure-PyTorch reimplementation of Kokoro's `CustomAlbert` (a real HF `AlbertModel`
returning just `last_hidden_state`), used as the ground truth test_e2e_kokoro_albert.cpp compares
loom-engine's C++ output against.

Deliberately hand-rolled rather than importing `transformers.AlbertModel` directly -- the same broken
huggingface-hub version pin already hit for NeMo's own toolkit blocks importing `transformers` in this
venv (confirmed, not assumed: `ImportError: huggingface-hub>=0.34.0,<1.0 is required ... but found
huggingface-hub==1.23.0`). Every formula/shape/eps here was confirmed by reading
`transformers/models/albert/{configuration,modeling}_albert.py`'s real source directly off disk (that
works fine -- only actually importing the package triggers the version check) -- see
convert_kokoro_albert.py's own module docstring for the full trail of confirmed real details (embedding
factorization, cross-layer weight sharing, "gelu_new", post-LN ordering, eps=1e-12, default
token_type_ids=0, absolute position embeddings).

Deterministic end to end -- no sampling anywhere in this piece, so this is a plain exact-match check.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_albert import HP


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * x * (1.0 + 0.044715 * x * x)))


def albert_forward(token_ids, sd, hp):
    """token_ids: 1D python list of phoneme ids (already including Kokoro's own [0, *ids, 0]
    bos/eos-sentinel wrapping -- that's the caller's responsibility, matching KModel.forward's own
    `input_ids = [0, *input_ids, 0]` line). Returns (T, hidden_size) numpy array."""
    T = len(token_ids)
    eps = hp["ln_eps"]
    tokens = torch.tensor(token_ids, dtype=torch.long)
    positions = torch.arange(T, dtype=torch.long)

    word_emb = F.embedding(tokens, sd["module.embeddings.word_embeddings.weight"])
    pos_emb = F.embedding(positions, sd["module.embeddings.position_embeddings.weight"])
    type_row0 = sd["module.embeddings.token_type_embeddings.weight"][0]
    x = word_emb + pos_emb + type_row0
    x = F.layer_norm(x, (hp["embedding_size"],), sd["module.embeddings.LayerNorm.weight"],
                      sd["module.embeddings.LayerNorm.bias"], eps)

    x = F.linear(x, sd["module.encoder.embedding_hidden_mapping_in.weight"],
                 sd["module.encoder.embedding_hidden_mapping_in.bias"])

    p = "module.encoder.albert_layer_groups.0.albert_layers.0"
    n_head, head_dim = hp["n_head"], hp["head_dim"]
    for _ in range(hp["n_layer"]):
        q = F.linear(x, sd[f"{p}.attention.query.weight"], sd[f"{p}.attention.query.bias"])
        k = F.linear(x, sd[f"{p}.attention.key.weight"], sd[f"{p}.attention.key.bias"])
        v = F.linear(x, sd[f"{p}.attention.value.weight"], sd[f"{p}.attention.value.bias"])
        q = q.view(T, n_head, head_dim).transpose(0, 1)  # (n_head, T, head_dim)
        k = k.view(T, n_head, head_dim).transpose(0, 1)
        v = v.view(T, n_head, head_dim).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).transpose(0, 1).reshape(T, n_head * head_dim)
        attn_out = F.linear(ctx, sd[f"{p}.attention.dense.weight"], sd[f"{p}.attention.dense.bias"])
        x = F.layer_norm(x + attn_out, (hp["hidden_size"],), sd[f"{p}.attention.LayerNorm.weight"],
                          sd[f"{p}.attention.LayerNorm.bias"], eps)

        ffn_h = F.linear(x, sd[f"{p}.ffn.weight"], sd[f"{p}.ffn.bias"])
        ffn_h = gelu_new(ffn_h)
        ffn_out = F.linear(ffn_h, sd[f"{p}.ffn_output.weight"], sd[f"{p}.ffn_output.bias"])
        x = F.layer_norm(x + ffn_out, (hp["hidden_size"],), sd[f"{p}.full_layer_layer_norm.weight"],
                          sd[f"{p}.full_layer_layer_norm.bias"], eps)

    return x.detach().numpy()


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["bert"]

    # Arbitrary but valid (< vocab_size=178) phoneme ids, already wrapped with Kokoro's own [0, ..., 0]
    # bos/eos sentinel convention (KModel.forward: `input_ids = [0, *input_ids, 0]`) -- semantic content
    # doesn't matter for this deterministic architecture-correctness check.
    phoneme_ids = [43, 62, 83, 61, 62, 47, 76, 46, 76, 56, 47]
    token_ids = [0] + phoneme_ids + [0]

    with torch.no_grad():
        out = albert_forward(token_ids, sd, HP)

    np.save(out_dir / "ref_albert_tokens.npy", np.array(token_ids, dtype=np.int32))
    np.save(out_dir / "ref_albert_out.npy", out)
    print(f"tokens={token_ids}, out shape={out.shape}, mean={out.mean():.6f}, std={out.std():.6f}")


if __name__ == "__main__":
    main()
