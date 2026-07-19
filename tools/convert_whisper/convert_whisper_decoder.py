"""Converts OpenAI Whisper's TextDecoder into a loom-engine GGUF file: token+positional embedding,
n_layer ResidualAttentionBlocks (causal self-attention w/ persistent KvCache + cross-attention against
the encoder's `xa` + MLP), final LayerNorm, tied output projection (`x @ token_embedding.weight.T`).
Companion to convert_whisper_encoder.py -- separate GGUF/topology, same multi-file precedent as VITS
(GraphTopology supports exactly one declared output per topology; the encoder's `xa` output becomes a
per-step INPUT here, produced once by `loom::WhisperDriver` and fed unchanged to every decode step).

Real checkpoint layout (same `tiny.en` torch.load dict as the encoder): decoder.token_embedding.weight
((n_vocab,n_state), tied with the output projection -- no separate output layer exists at all),
decoder.positional_embedding (a LEARNED nn.Parameter, shape (n_text_ctx,n_state) -- genuinely different
from the encoder's fixed sinusoidal buffer, looked up by absolute position via GET_ROWS, same mechanism
as token embedding, NOT RoPE), decoder.blocks.{i}.attn.* (causal self-attention, same Linear shapes as
the encoder's attn), decoder.blocks.{i}.cross_attn.* (same shapes, K/V projected from `xa` instead of the
decoder's own hidden state), decoder.blocks.{i}.{attn_ln,cross_attn_ln,mlp_ln}, decoder.blocks.{i}.mlp.
{0,2}, decoder.ln (final).

Self-attention uses ATTENTION with kv_cache=true (persistent per-layer KvCache, causal mask, exactly
Generator's own convention: "tokens"/"positions"/"kq_mask" inputs, growing n_past/n_kv) -- `positions`
here means something different than Qwen3's (a GET_ROWS index into the LEARNED positional_embedding
table, not a RoPE angle), but the input NAME and host-side construction (0..n_past+n_tokens-1) are
identical. Cross-attention uses ATTENTION with kv_cache=false: K/V are projected fresh from the `xa`
input on EVERY decode step (not cached across steps the way the real PyTorch model optimizes via
`install_kv_cache_hooks`'s "calculate keys/values once" trick) -- simpler and still correct, just
redundant compute; left as a documented future optimization for `loom::WhisperDriver`, not attempted
here. `xa`'s own length (n_audio_ctx) is a FIXED literal (1500 for every checkpoint size, confirmed in
whisper's own ModelDimensions), unlike the decoder's own `$n_tokens`, which grows exactly like every
other autoregressive decoder in this project.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

from convert_whisper_encoder import TopologyBuilder, add_linear, add_layer_norm, apply_layer_norm, to_f32


def build_decoder(tb, sd, dims):
    n_state = dims["n_text_state"]
    n_head = dims["n_text_head"]
    n_layer = dims["n_text_layer"]
    head_dim = n_state // n_head
    n_audio_ctx = dims["n_audio_ctx"]
    eps = 1e-5

    tok_emb = tb.weight("decoder.token_embedding.weight", to_f32(sd["decoder.token_embedding.weight"]))
    x = tb.node("GET_ROWS", [tok_emb, "tokens"], None, "tok_emb")  # [n_state, n_tokens]

    pos_emb_table = tb.weight("decoder.positional_embedding", to_f32(sd["decoder.positional_embedding"]))
    pos = tb.node("GET_ROWS", [pos_emb_table, "positions"], None, "pos_emb")  # [n_state, n_tokens]
    x = tb.node("ADD", [x, pos], None, "x_emb")

    for i in range(n_layer):
        p = f"decoder.blocks.{i}"

        # --- causal self-attention (persistent KvCache, layer=i) ---
        resid = x
        xn = apply_layer_norm(tb, x, f"{p}.attn_ln", sd, f"{p}.attn_ln", eps, f"attn_ln_out_{i}")
        qw, qb = add_linear(tb, f"{p}.attn.query", sd, f"{p}.attn.query")
        kw, _ = add_linear(tb, f"{p}.attn.key", sd, f"{p}.attn.key", has_bias=False)
        vw, vb = add_linear(tb, f"{p}.attn.value", sd, f"{p}.attn.value")
        ow, ob = add_linear(tb, f"{p}.attn.out", sd, f"{p}.attn.out")
        q = tb.node("ADD", [tb.node("MUL_MAT", [qw, xn], None, "q"), qb], None, "q_b")
        k = tb.node("MUL_MAT", [kw, xn], None, "k_b")
        v = tb.node("ADD", [tb.node("MUL_MAT", [vw, xn], None, "v"), vb], None, "v_b")
        q = tb.node("RESHAPE", [q], {"shape": [head_dim, n_head, "$n_tokens"]}, "q_r")
        k = tb.node("RESHAPE", [k], {"shape": [head_dim, n_head, "$n_tokens"]}, "k_r")
        v = tb.node("RESHAPE", [v], {"shape": [head_dim, n_head, "$n_tokens"]}, "v_r")
        attn = tb.node("ATTENTION", [q, k, v, "kq_mask"],
                       {"kv_cache": True, "layer": i, "scale": 1.0 / float(np.sqrt(head_dim))}, "self_attn")
        o = tb.node("ADD", [tb.node("MUL_MAT", [ow, attn], None, "o"), ob], None, "o_b")
        x = tb.node("ADD", [resid, o], None, "res_self")

        # --- cross-attention against the encoder's `xa` (kv_cache=false, K/V recomputed every step) ---
        resid = x
        xn = apply_layer_norm(tb, x, f"{p}.cross_attn_ln", sd, f"{p}.cross_attn_ln", eps, f"xattn_ln_out_{i}")
        cqw, cqb = add_linear(tb, f"{p}.cross_attn.query", sd, f"{p}.cross_attn.query")
        ckw, _ = add_linear(tb, f"{p}.cross_attn.key", sd, f"{p}.cross_attn.key", has_bias=False)
        cvw, cvb = add_linear(tb, f"{p}.cross_attn.value", sd, f"{p}.cross_attn.value")
        cow, cob = add_linear(tb, f"{p}.cross_attn.out", sd, f"{p}.cross_attn.out")
        cq = tb.node("ADD", [tb.node("MUL_MAT", [cqw, xn], None, "cq"), cqb], None, "cq_b")
        ck = tb.node("MUL_MAT", [ckw, "xa"], None, "ck_b")
        cv = tb.node("ADD", [tb.node("MUL_MAT", [cvw, "xa"], None, "cv"), cvb], None, "cv_b")
        cq = tb.node("RESHAPE", [cq], {"shape": [head_dim, n_head, "$n_tokens"]}, "cq_r")
        ck = tb.node("RESHAPE", [ck], {"shape": [head_dim, n_head, str(n_audio_ctx)]}, "ck_r")
        cv = tb.node("RESHAPE", [cv], {"shape": [head_dim, n_head, str(n_audio_ctx)]}, "cv_r")
        xattn = tb.node("ATTENTION", [cq, ck, cv, "xa_mask"],
                        {"kv_cache": False, "scale": 1.0 / float(np.sqrt(head_dim))}, "cross_attn")
        co = tb.node("ADD", [tb.node("MUL_MAT", [cow, xattn], None, "co"), cob], None, "co_b")
        x = tb.node("ADD", [resid, co], None, "res_cross")

        # --- MLP ---
        resid = x
        xn = apply_layer_norm(tb, x, f"{p}.mlp_ln", sd, f"{p}.mlp_ln", eps, f"mlp_ln_out_{i}")
        w1, b1 = add_linear(tb, f"{p}.mlp.0", sd, f"{p}.mlp.0")
        w2, b2 = add_linear(tb, f"{p}.mlp.2", sd, f"{p}.mlp.2")
        hmlp = tb.node("ADD", [tb.node("MUL_MAT", [w1, xn], None, "mlp1"), b1], None, "mlp1_b")
        hmlp = tb.node("GELU", [hmlp], None, "mlp_gelu")
        hmlp = tb.node("ADD", [tb.node("MUL_MAT", [w2, hmlp], None, "mlp2"), b2], None, "mlp2_b")
        x = tb.node("ADD", [resid, hmlp], None, "res_mlp")

    x = apply_layer_norm(tb, x, "decoder.ln", sd, "decoder.ln", eps, "ln_final_out")
    # Tied output projection: MUL_MAT(tok_emb[n_state,n_vocab], x[n_state,n_tokens]) contracts over
    # n_state -> [n_vocab, n_tokens], exactly `x @ token_embedding.weight.T`'s math -- the SAME weight
    # tensor already registered above for the input embedding lookup, no separate weight/transpose.
    logits = tb.node("MUL_MAT", [tok_emb, x], None, "logits")
    return logits


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.pt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    dims = checkpoint["dims"]
    sd = checkpoint["model_state_dict"]

    tb = TopologyBuilder()
    logits = build_decoder(tb, sd, dims)

    n_state = dims["n_text_state"]
    n_audio_ctx = dims["n_audio_ctx"]
    inputs = [
        {"name": "tokens", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "positions", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "kq_mask", "dtype": "f32", "shape": ["$n_kv", "$n_tokens"]},
        {"name": "xa", "dtype": "f32", "shape": [str(n_state), str(n_audio_ctx)]},
        {"name": "xa_mask", "dtype": "f32", "shape": [str(n_audio_ctx), "$n_tokens"]},
    ]
    topo = tb.topology(inputs, logits)

    writer = GGUFWriter(str(out_dir / "whisper_decoder.gguf"), "loom-whisper-decoder")
    writer.add_string("model.graph_topology", json.dumps(topo))
    # hparams needed by loom::WhisperDriver to size its KvCache (same convention Generator/GgufModel::
    # hparam_u32 already use for Qwen3/toy_llm's own decode loop).
    writer.add_uint32("loom.n_layer", dims["n_text_layer"])
    writer.add_uint32("loom.n_head_kv", dims["n_text_head"])
    writer.add_uint32("loom.n_embd_head_k", n_state // dims["n_text_head"])
    writer.add_uint32("loom.n_embd_head_v", n_state // dims["n_text_head"])
    for name, arr in tb.weights.items():
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {out_dir / 'whisper_decoder.gguf'}, {len(tb.weights)} weights")


if __name__ == "__main__":
    main()
