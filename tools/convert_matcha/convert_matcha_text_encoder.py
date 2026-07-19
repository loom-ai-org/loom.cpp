"""Converts Matcha-TTS's real TextEncoder (`matcha/models/components/text_encoder.py`) into two
loom-engine GGUF topologies -- `matcha_encoder_mu.gguf` (ends at `proj_m`) and
`matcha_encoder_logw.gguf` (ends at `proj_w`, the per-token DurationPredictor) -- mirroring VITS's own
established "GraphTopology supports exactly one declared output; two files sharing one duplicated
TextEncoder body" precedent (see convert_piper_vits/convert_vits.py's `build_text_sdp_topologies`).

Real hyperparameters confirmed directly against the real checkpoint's `hyper_parameters` dict AND its
305-tensor state dict (n_vocab=178, n_spks=1 single-speaker so no speaker embedding table exists,
n_channels=192, filter_channels=768, n_heads=2, n_layers=6, kernel_size=3, prenet=True,
filter_channels_dp=256).

Real RoPE: `MultiHeadAttention.query_rotary_pe`/`key_rotary_pe` are `RotaryPositionalEmbeddings(d=
k_channels*0.5=48)` -- REAL integer positions, "rotate-half" (NeoX) convention, rotating only the
FIRST 48 of the 96 k_channels (the other 48 passed through unrotated). Confirmed directly against
ggml's own `ggml_compute_forward_rope_flt`/`rotate_pairs` (GGML_ROPE_TYPE_NEOX branch: pairs
`(ic, ic+n_dims/2)` for `ic` in `[0, n_dims/2)`, `theta_scale = freq_base^(-2/n_dims)`) that the
EXISTING native `ROPE` primitive (`mode=2`, `n_dims=48`, `freq_base=10000`, `ext_factor=0` to disable
YaRN) reproduces this exactly -- no new primitive/composition needed, same primitive already used for
Qwen3.

Tensor-layout convention: TextEncoder's whole pipeline (emb, prenet, encoder, proj_m/proj_w) is
channel-first `[C, T]` (C = ne[0], matching GET_ROWS's own embedding-lookup convention) -- SAME
convention established for VITS's own TextEncoder. CONV_1D-consuming pieces (prenet's kernel=5 convs,
FFN's kernel=3 convs, DurationPredictor's kernel=3 convs) transpose to `[T, C]` and back at each
boundary via `tb.transpose_2d`, same as VITS.
"""
import sys
from pathlib import Path

import numpy as np

from matcha_common import (
    TopologyBuilder, add_conv, add_conv1x1_as_matmul, add_glowtts_layer_norm,
    apply_glowtts_layer_norm, load_matcha_checkpoint, to_f32, write_gguf,
)

HP = {
    "n_vocab": 178,
    "n_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "filter_channels_dp": 256,
    "dp_kernel_size": 3,
    "prenet_kernel_size": 5,
    "prenet_n_layers": 3,
    "ln_eps": 1e-4,  # matcha's own text_encoder.LayerNorm default (NOT VITS's 1e-5)
    "rope_dims": 48,  # k_channels(96) * 0.5
    "rope_freq_base": 10000.0,
}


def build_conv_relu_norm(tb, x_ct, prefix, sd, name, n_layers, kernel_size, channels, eps, out_hint="crn"):
    """`ConvReluNorm`: n_layers of (Conv1d(kernel_size) -> channel LayerNorm -> ReLU), then a
    zero-init-but-now-trained kernel_size=1 `proj` conv, added residually to the ORIGINAL input
    (`x_org`), matching `ConvReluNorm.forward` exactly (dropout skipped at inference).
    """
    x_org = x_ct
    x = x_ct
    for i in range(n_layers):
        w, b = add_conv(tb, f"{prefix}.conv_layers.{i}", sd, f"{name}.conv_layers.{i}")
        xt = tb.transpose_2d(x, f"{out_hint}{i}_xt")
        xt3 = tb.node("RESHAPE", [xt], {"shape": [-1, channels, 1]}, f"{out_hint}{i}_xt3")
        h = tb.node("CONV_1D", [w, xt3], {"s0": 1, "p0": kernel_size // 2, "d0": 1}, f"{out_hint}{i}_h")
        h = tb.node("ADD", [h, tb.node("RESHAPE", [b], {"shape": [1, channels, 1]}, f"{out_hint}{i}_b_r")],
                     None, f"{out_hint}{i}_hb")
        h2d = tb.node("RESHAPE", [h], {"shape": [-1, channels]}, f"{out_hint}{i}_h2d")
        x_ct2 = tb.transpose_2d(h2d, f"{out_hint}{i}_ct")
        gamma, beta = add_glowtts_layer_norm(tb, f"{prefix}.norm_layers.{i}", sd, f"{name}.norm_layers.{i}")
        x_ct2 = apply_glowtts_layer_norm(tb, x_ct2, gamma, beta, channels, eps, f"{out_hint}{i}_ln")
        x = tb.node("RELU", [x_ct2], None, f"{out_hint}{i}_relu")
    proj_w, proj_b = add_conv1x1_as_matmul(tb, f"{prefix}.proj", sd, f"{name}.proj")
    proj_out = tb.node("MUL_MAT", [proj_w, x], None, f"{out_hint}_proj")
    proj_out = tb.node("ADD", [proj_out, proj_b], None, f"{out_hint}_proj_b")
    return tb.node("ADD", [x_org, proj_out], None, out_hint)


def build_ffn(tb, x_ct, prefix, sd, name, channels, filter_channels, kernel_size, out_hint="ffn"):
    """`FFN`: conv_1 (channels->filter_channels) -> ReLU -> conv_2 (filter_channels->channels).
    No LayerNorm inside FFN itself (that's applied by the caller, `Encoder`, afterward).
    """
    w1, b1 = add_conv(tb, f"{prefix}.conv_1", sd, f"{name}.conv_1")
    w2, b2 = add_conv(tb, f"{prefix}.conv_2", sd, f"{name}.conv_2")
    xt = tb.transpose_2d(x_ct, out_hint + "_xt")
    xt3 = tb.node("RESHAPE", [xt], {"shape": [-1, channels, 1]}, out_hint + "_xt3")
    h = tb.node("CONV_1D", [w1, xt3], {"s0": 1, "p0": kernel_size // 2, "d0": 1}, out_hint + "_h")
    h = tb.node("ADD", [h, tb.node("RESHAPE", [b1], {"shape": [1, filter_channels, 1]}, out_hint + "_b1_r")],
                 None, out_hint + "_hb")
    h = tb.node("RELU", [h], None, out_hint + "_relu")
    h2 = tb.node("CONV_1D", [w2, h], {"s0": 1, "p0": kernel_size // 2, "d0": 1}, out_hint + "_h2")
    h2 = tb.node("ADD", [h2, tb.node("RESHAPE", [b2], {"shape": [1, channels, 1]}, out_hint + "_b2_r")],
                 None, out_hint + "_h2b")
    h2d = tb.node("RESHAPE", [h2], {"shape": [-1, channels]}, out_hint + "_h2d")
    return tb.transpose_2d(h2d, out_hint + "_ct")


def build_encoder(tb, x_ct, sd, prefix, name, hp, out_hint="enc"):
    """`Encoder`: n_layers of (self-attention w/ partial-rotary RoPE -> residual -> LayerNorm ->
    FFN -> residual -> LayerNorm), post-norm throughout. Attention uses `kv_cache: False` (plain,
    non-causal, single-utterance self-attention -- same established pattern as Kokoro's ALBERT
    encoder), mask is a declared `attn_mask` input (all-zero additive bias, no padding).
    """
    channels = hp["n_channels"]
    n_heads = hp["n_heads"]
    head_dim = channels // n_heads
    n_layers = hp["n_layers"]
    eps = hp["ln_eps"]

    x = x_ct
    for i in range(n_layers):
        p = f"{prefix}.attn_layers.{i}"
        qw, qb = add_conv1x1_as_matmul(tb, f"{p}.conv_q", sd, f"{name}.attn_layers.{i}.conv_q")
        kw, kb = add_conv1x1_as_matmul(tb, f"{p}.conv_k", sd, f"{name}.attn_layers.{i}.conv_k")
        vw, vb = add_conv1x1_as_matmul(tb, f"{p}.conv_v", sd, f"{name}.attn_layers.{i}.conv_v")
        ow, ob = add_conv1x1_as_matmul(tb, f"{p}.conv_o", sd, f"{name}.attn_layers.{i}.conv_o")

        q = tb.node("ADD", [tb.node("MUL_MAT", [qw, x], None, f"q{i}"), qb], None, f"q{i}_b")
        k = tb.node("ADD", [tb.node("MUL_MAT", [kw, x], None, f"k{i}"), kb], None, f"k{i}_b")
        v = tb.node("ADD", [tb.node("MUL_MAT", [vw, x], None, f"v{i}"), vb], None, f"v{i}_b")
        q = tb.node("RESHAPE", [q], {"shape": [head_dim, n_heads, "$n_tokens"]}, f"q{i}_r")
        k = tb.node("RESHAPE", [k], {"shape": [head_dim, n_heads, "$n_tokens"]}, f"k{i}_r")
        v = tb.node("RESHAPE", [v], {"shape": [head_dim, n_heads, "$n_tokens"]}, f"v{i}_r")

        rope_attrs = {"n_dims": hp["rope_dims"], "mode": 2, "n_ctx_orig": 0,
                      "freq_base": hp["rope_freq_base"], "freq_scale": 1.0,
                      "ext_factor": 0.0, "attn_factor": 1.0, "beta_fast": 32.0, "beta_slow": 1.0}
        q = tb.node("ROPE", [q, "positions"], rope_attrs, f"q{i}_rope")
        k = tb.node("ROPE", [k, "positions"], rope_attrs, f"k{i}_rope")

        attn = tb.node("ATTENTION", [q, k, v, "attn_mask"],
                        {"kv_cache": False, "scale": 1.0 / float(np.sqrt(head_dim))}, f"attn{i}")
        o = tb.node("ADD", [tb.node("MUL_MAT", [ow, attn], None, f"o{i}"), ob], None, f"o{i}_b")

        x = tb.node("ADD", [x, o], None, f"res1_{i}")
        g1, b1 = add_glowtts_layer_norm(tb, f"{prefix}.norm_layers_1.{i}", sd, f"{name}.norm_layers_1.{i}")
        x = apply_glowtts_layer_norm(tb, x, g1, b1, channels, eps, f"ln1_{i}")

        ffn_out = build_ffn(tb, x, f"{prefix}.ffn_layers.{i}", sd, f"{name}.ffn_layers.{i}",
                             channels, hp["filter_channels"], hp["kernel_size"], f"ffn{i}")
        x = tb.node("ADD", [x, ffn_out], None, f"res2_{i}")
        g2, b2 = add_glowtts_layer_norm(tb, f"{prefix}.norm_layers_2.{i}", sd, f"{name}.norm_layers_2.{i}")
        x = apply_glowtts_layer_norm(tb, x, g2, b2, channels, eps, f"ln2_{i}")

    return x


def build_duration_predictor(tb, x_ct, sd, hp, out_hint="dp"):
    """`DurationPredictor`: conv_1(kernel3) -> ReLU -> LayerNorm -> conv_2(kernel3) -> ReLU ->
    LayerNorm -> proj(kernel1, ->1 channel). Mask-multiplies before every conv are skipped (single,
    unpadded utterance -- mask is always 1).
    """
    channels = hp["n_channels"]
    fc = hp["filter_channels_dp"]
    k = hp["dp_kernel_size"]
    w1, b1 = add_conv(tb, "encoder.proj_w.conv_1", sd, "encoder.proj_w.conv_1")
    xt = tb.transpose_2d(x_ct, "dp_xt")
    xt3 = tb.node("RESHAPE", [xt], {"shape": [-1, channels, 1]}, "dp_xt3")
    h = tb.node("CONV_1D", [w1, xt3], {"s0": 1, "p0": k // 2, "d0": 1}, "dp_h1")
    h = tb.node("ADD", [h, tb.node("RESHAPE", [b1], {"shape": [1, fc, 1]}, "dp_b1_r")], None, "dp_h1b")
    h2d = tb.node("RESHAPE", [h], {"shape": [-1, fc]}, "dp_h1_2d")
    h_ct = tb.transpose_2d(h2d, "dp_h1_ct")
    h_ct = tb.node("RELU", [h_ct], None, "dp_relu1")
    g1, b1n = add_glowtts_layer_norm(tb, "encoder.proj_w.norm_1", sd, "encoder.proj_w.norm_1")
    h_ct = apply_glowtts_layer_norm(tb, h_ct, g1, b1n, fc, hp["ln_eps"], "dp_ln1")

    w2, b2 = add_conv(tb, "encoder.proj_w.conv_2", sd, "encoder.proj_w.conv_2")
    xt2 = tb.transpose_2d(h_ct, "dp_xt2")
    xt2_3 = tb.node("RESHAPE", [xt2], {"shape": [-1, fc, 1]}, "dp_xt2_3")
    h2 = tb.node("CONV_1D", [w2, xt2_3], {"s0": 1, "p0": k // 2, "d0": 1}, "dp_h2")
    h2 = tb.node("ADD", [h2, tb.node("RESHAPE", [b2], {"shape": [1, fc, 1]}, "dp_b2_r")], None, "dp_h2b")
    h2_2d = tb.node("RESHAPE", [h2], {"shape": [-1, fc]}, "dp_h2_2d")
    h2_ct = tb.transpose_2d(h2_2d, "dp_h2_ct")
    h2_ct = tb.node("RELU", [h2_ct], None, "dp_relu2")
    g2, b2n = add_glowtts_layer_norm(tb, "encoder.proj_w.norm_2", sd, "encoder.proj_w.norm_2")
    h2_ct = apply_glowtts_layer_norm(tb, h2_ct, g2, b2n, fc, hp["ln_eps"], "dp_ln2")

    proj_w, proj_b = add_conv1x1_as_matmul(tb, "encoder.proj_w.proj", sd, "encoder.proj_w.proj")
    logw = tb.node("MUL_MAT", [proj_w, h2_ct], None, "logw_mm")
    logw = tb.node("ADD", [logw, proj_b], None, out_hint)
    return logw


def build_text_encoder_body(tb, sd, token_ids_name):
    """emb -> prenet -> Encoder. Shared by both the `mu` and `logw` topologies (each rebuilds this
    from scratch into its own TopologyBuilder, mirroring VITS's own established duplication
    precedent since GraphTopology supports exactly one declared output per file).
    """
    channels = HP["n_channels"]
    emb_w = tb.weight("encoder.emb.weight", to_f32(sd["encoder.emb.weight"]))
    x = tb.node("GET_ROWS", [emb_w, token_ids_name], None, "emb")
    x = tb.node("SCALE", [x], {"s": float(np.sqrt(channels))}, "emb_scaled")

    x = build_conv_relu_norm(tb, x, "encoder.prenet", sd, "encoder.prenet",
                              HP["prenet_n_layers"], HP["prenet_kernel_size"], channels, HP["ln_eps"], "prenet")
    x = build_encoder(tb, x, sd, "encoder.encoder", "encoder.encoder", HP, "enc")
    return x


def build_mu_topology(sd):
    tb = TopologyBuilder()
    x = build_text_encoder_body(tb, sd, "tokens")
    proj_m_w, proj_m_b = add_conv1x1_as_matmul(tb, "encoder.proj_m", sd, "encoder.proj_m")
    mu = tb.node("MUL_MAT", [proj_m_w, x], None, "mu_mm")
    mu = tb.node("ADD", [mu, proj_m_b], None, "mu")

    inputs = [
        {"name": "tokens", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "positions", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "attn_mask", "dtype": "f32", "shape": ["$n_tokens", "$n_tokens"]},
    ]
    return tb.topology(inputs, mu), tb.weights, tb.int32_weights


def build_logw_topology(sd):
    tb = TopologyBuilder()
    x = build_text_encoder_body(tb, sd, "tokens")
    logw = build_duration_predictor(tb, x, sd, HP, "logw")

    inputs = [
        {"name": "tokens", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "positions", "dtype": "i32", "shape": ["$n_tokens"]},
        {"name": "attn_mask", "dtype": "f32", "shape": ["$n_tokens", "$n_tokens"]},
    ]
    return tb.topology(inputs, logw), tb.weights, tb.int32_weights


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <matcha_ljspeech.ckpt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = load_matcha_checkpoint(ckpt_path)

    mu_topo, mu_weights, mu_int32 = build_mu_topology(sd)
    write_gguf(out_dir / "matcha_encoder_mu.gguf", "matcha_encoder_mu", HP, mu_topo, mu_weights, mu_int32)

    logw_topo, logw_weights, logw_int32 = build_logw_topology(sd)
    write_gguf(out_dir / "matcha_encoder_logw.gguf", "matcha_encoder_logw", HP, logw_topo, logw_weights, logw_int32)


if __name__ == "__main__":
    main()
