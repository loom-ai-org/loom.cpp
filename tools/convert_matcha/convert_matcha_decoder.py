"""Converts Matcha-TTS's real `Decoder` U-Net (the CFM `estimator`, `matcha/models/components/
decoder.py`) into a single loom-engine GGUF topology (`matcha_decoder.gguf`) computing one Euler
velocity evaluation `dphi_dt = estimator(x, mask, mu, t, spks=None, cond=None)`.

Real config (`channels=(256,256)`) gives a SHALLOW U-Net: only `down_blocks[0]`'s downsample is a
real stride-2 conv (`down_blocks[1]` is `is_last`, a plain same-resolution conv); the mirror holds for
`up_blocks[0]` (real stride-2 ConvTranspose1d) vs `up_blocks[1]` (`is_last`, plain conv) -- confirmed
directly against the real 305-tensor state dict (see PLAN.md's "Real checkpoint facts"). So there is
exactly ONE real downsample and ONE real upsample despite each `nn.ModuleList` having 2 entries.

Scope decision (mirrors SupertonicTTS's own T_TEXT fixed-length choice): this topology assumes the
caller (MatchaDriver) always sizes its mel-frame count `$n_tokens` to a MULTIPLE OF 4 (real
`fix_len_compatibility`'s own default `num_downsamplings_in_unet=2` requirement) -- `$n_tokens/2` is
then always an exact integer, so no fractional-length edge cases need handling in this topology.
Padding-mask handling is dropped entirely (every real `x*mask`/`h*mask` multiply is a no-op for a
single, unpadded, exactly-multiple-of-4-length utterance).

Tensor-layout convention: exactly mirrors the real code's own `rearrange` calls -- `ResnetBlock1D`/
`Block1D`/`Downsample1D`/`Upsample1D`/the time-embedding MLP injection all operate in `[T,C]` (conv,
T=ne[0]) convention; `BasicTransformerBlock` (LayerNorm/Attention/SnakeBeta-FeedForward) operates in
`[C,T]` (channel-first, C=ne[0]) convention, crossed via `tb.transpose_2d` at each `rearrange`
boundary, same established pattern as VITS's/Matcha's own TextEncoder FFN boundary.
"""
import sys
from pathlib import Path

import numpy as np

from matcha_common import (
    TopologyBuilder, add_conv, add_conv1x1_as_matmul, add_linear, add_linear_no_bias,
    apply_std_layer_norm, build_group_norm, load_matcha_checkpoint, mish, to_f32, write_gguf,
)

HP = {
    "n_feats": 80,
    "channels": 256,
    "time_embed_dim": 1024,  # channels[0] * 4
    "n_groups": 8,
    "gn_eps": 1e-5,
    "ln_eps": 1e-5,
    "num_heads": 2,
    "attention_head_dim": 64,
    "ff_mult": 4,
}


def sinusoidal_time_embedding(tb, dim, out_hint="tpe"):
    """`SinusoidalPosEmb(dim=in_channels=160)`: a PURE function of `dim` (no learned params) -- the
    frequency table is computed here (numpy, conversion-time) and baked as a constant weight, same
    "bake a fixed constant" precedent as the mel frontend's DFT kernels / RQ_SPLINE's boundary
    constants. `scale=1000` (real code's own default, never overridden by `Decoder.forward`).
    """
    half_dim = dim // 2
    emb_const = np.log(10000.0) / (half_dim - 1)
    freqs = np.exp(np.arange(half_dim, dtype=np.float64) * -emb_const).astype(np.float32)
    freqs_w = tb.weight(f"{out_hint}.freqs", freqs)
    angles = tb.node("SCALE", [tb.node("MUL", [freqs_w, "t"], None, f"{out_hint}_raw")], {"s": 1000.0},
                      f"{out_hint}_angles")
    sin_a = tb.node("SIN", [angles], None, f"{out_hint}_sin")
    cos_a = tb.node("COS", [angles], None, f"{out_hint}_cos")
    return tb.node("CONCAT", [sin_a, cos_a], {"dim": 0}, out_hint)


def build_time_mlp(tb, sd, in_channels, time_embed_dim, out_hint="time_mlp"):
    """`TimestepEmbedding(in_channels, time_embed_dim, act_fn="silu")`: Linear -> SiLU -> Linear."""
    raw = sinusoidal_time_embedding(tb, in_channels, out_hint + "_pe")
    w1, b1 = add_linear(tb, f"{out_hint}.linear_1", sd, "decoder.estimator.time_mlp.linear_1")
    h = tb.node("ADD", [tb.node("MUL_MAT", [w1, raw], None, out_hint + "_h1_mm"), b1], None, out_hint + "_h1")
    h = tb.node("SILU", [h], None, out_hint + "_h1_silu")
    w2, b2 = add_linear(tb, f"{out_hint}.linear_2", sd, "decoder.estimator.time_mlp.linear_2")
    return tb.node("ADD", [tb.node("MUL_MAT", [w2, h], None, out_hint + "_h2_mm"), b2], None, out_hint)


def build_block1d(tb, x_tc, prefix, sd, name, dim_in, dim_out, hp, out_hint="blk"):
    """`Block1D`: Conv1d(kernel3,pad1) -> GroupNorm(8) -> Mish. Mask multiplies dropped (see module
    docstring's scope decision).
    """
    w, b = add_conv(tb, f"{prefix}.block.0", sd, f"{name}.block.0")
    x3 = tb.node("RESHAPE", [x_tc], {"shape": [-1, dim_in, 1]}, out_hint + "_x3")
    h = tb.node("CONV_1D", [w, x3], {"s0": 1, "p0": 1, "d0": 1}, out_hint + "_conv")
    h = tb.node("ADD", [h, tb.node("RESHAPE", [b], {"shape": [1, dim_out, 1]}, out_hint + "_b_r")],
                None, out_hint + "_convb")
    h2d = tb.node("RESHAPE", [h], {"shape": [-1, dim_out]}, out_hint + "_h2d")
    h_gn = build_group_norm(tb, h2d, f"{prefix}.block.1", sd, f"{name}.block.1", dim_out, hp["n_groups"],
                             hp["gn_eps"], out_hint + "_gn")
    return mish(tb, h_gn, out_hint + "_mish")


def build_resnet_block1d(tb, x_tc, time_emb, prefix, sd, name, dim_in, dim_out, hp, out_hint="res"):
    """`ResnetBlock1D`: block1 -> (+time MLP, broadcast over T) -> block2 -> + res_conv(x)."""
    h = build_block1d(tb, x_tc, f"{prefix}.block1", sd, f"{name}.block1", dim_in, dim_out, hp, out_hint + "_b1")

    te_mish = mish(tb, time_emb, out_hint + "_te_mish")
    w_mlp, b_mlp = add_linear(tb, f"{prefix}.mlp.1", sd, f"{name}.mlp.1")
    t_cond = tb.node("ADD", [tb.node("MUL_MAT", [w_mlp, te_mish], None, out_hint + "_tcond_mm"), b_mlp],
                      None, out_hint + "_tcond")
    t_cond_r = tb.node("RESHAPE", [t_cond], {"shape": [1, dim_out]}, out_hint + "_tcond_r")
    h = tb.node("ADD", [h, t_cond_r], None, out_hint + "_plus_t")  # broadcast over T (ne[0])

    h = build_block1d(tb, h, f"{prefix}.block2", sd, f"{name}.block2", dim_out, dim_out, hp, out_hint + "_b2")

    res_w, res_b = add_conv(tb, f"{prefix}.res_conv", sd, f"{name}.res_conv")
    x3 = tb.node("RESHAPE", [x_tc], {"shape": [-1, dim_in, 1]}, out_hint + "_resx3")
    res = tb.node("CONV_1D", [res_w, x3], {"s0": 1, "p0": 0, "d0": 1}, out_hint + "_resconv")
    res = tb.node("ADD", [res, tb.node("RESHAPE", [res_b], {"shape": [1, dim_out, 1]}, out_hint + "_resb_r")],
                  None, out_hint + "_resconvb")
    res2d = tb.node("RESHAPE", [res], {"shape": [-1, dim_out]}, out_hint + "_res2d")
    return tb.node("ADD", [h, res2d], None, out_hint)


def build_snakebeta_ffn(tb, x_ct, prefix, sd, name, dim, inner_dim, out_hint="ff"):
    """`FeedForward(activation_fn="snakebeta")`: SnakeBeta(Linear(dim,inner_dim)) -> Linear(inner_dim,dim).
    `SnakeBeta(x) = y + (1/(exp(beta)+eps)) * sin(y*exp(alpha))^2`, `y = proj(x)`, alpha/beta log-scale.
    """
    proj_w, proj_b = add_linear(tb, f"{prefix}.net.0.proj", sd, f"{name}.net.0.proj")
    y = tb.node("ADD", [tb.node("MUL_MAT", [proj_w, x_ct], None, out_hint + "_proj_mm"), proj_b], None, out_hint + "_y")

    # alpha/beta raw shape (inner_dim,) -> ne[0]=inner_dim, already aligned with `y`'s own channel-first
    # [inner_dim,T] convention (C=ne[0]) -- no reshape needed (unlike the [T,C]-convention bias-adds
    # elsewhere in this file, which DO need a [1,C]/[1,C,1] reshape to broadcast over ne[0]=T).
    alpha = tb.weight(f"{prefix}.net.0.alpha", to_f32(sd[f"{name}.net.0.alpha"]))
    beta = tb.weight(f"{prefix}.net.0.beta", to_f32(sd[f"{name}.net.0.beta"]))
    alpha_exp = tb.node("EXP", [alpha], None, out_hint + "_alpha_exp")
    beta_exp = tb.node("EXP", [beta], None, out_hint + "_beta_exp")

    eps_bump = tb.weight(f"{out_hint}.eps_bump", np.full(inner_dim, 1e-9, dtype=np.float32))
    denom = tb.node("ADD", [beta_exp, eps_bump], None, out_hint + "_denom")

    s = tb.node("SIN", [tb.node("MUL", [y, alpha_exp], None, out_hint + "_y_alpha")], None, out_hint + "_sin")
    s2 = tb.node("SQR", [s], None, out_hint + "_sin2")
    term = tb.node("DIV", [s2, denom], None, out_hint + "_term")
    snake_out = tb.node("ADD", [y, term], None, out_hint + "_snake")

    w2, b2 = add_linear(tb, f"{prefix}.net.2", sd, f"{name}.net.2")
    return tb.node("ADD", [tb.node("MUL_MAT", [w2, snake_out], None, out_hint + "_out_mm"), b2], None, out_hint)


def build_basic_transformer_block(tb, x_ct, prefix, sd, name, dim, hp, mask_name, out_hint="btb"):
    """`BasicTransformerBlock` real-config-reduced forward: norm1 -> self-attn (no cross-attn) ->
    residual -> norm3 -> FeedForward(SnakeBeta) -> residual. `timestep` kwarg is a no-op here (only
    used by AdaLayerNorm variants, not `norm_type="layer_norm"`).
    """
    n1_w = tb.weight(f"{prefix}.norm1.weight", to_f32(sd[f"{name}.norm1.weight"]))
    n1_b = tb.weight(f"{prefix}.norm1.bias", to_f32(sd[f"{name}.norm1.bias"]))
    normed1 = apply_std_layer_norm(tb, x_ct, n1_w, n1_b, hp["ln_eps"], out_hint + "_ln1")

    n_heads = hp["num_heads"]
    head_dim = hp["attention_head_dim"]
    qw = add_linear_no_bias(tb, f"{prefix}.attn1.to_q", sd, f"{name}.attn1.to_q")
    kw = add_linear_no_bias(tb, f"{prefix}.attn1.to_k", sd, f"{name}.attn1.to_k")
    vw = add_linear_no_bias(tb, f"{prefix}.attn1.to_v", sd, f"{name}.attn1.to_v")
    ow, ob = add_linear(tb, f"{prefix}.attn1.to_out.0", sd, f"{name}.attn1.to_out.0")

    q = tb.node("MUL_MAT", [qw, normed1], None, out_hint + "_q")
    k = tb.node("MUL_MAT", [kw, normed1], None, out_hint + "_k")
    v = tb.node("MUL_MAT", [vw, normed1], None, out_hint + "_v")
    q = tb.node("RESHAPE", [q], {"shape": [head_dim, n_heads, -1]}, out_hint + "_q_r")
    k = tb.node("RESHAPE", [k], {"shape": [head_dim, n_heads, -1]}, out_hint + "_k_r")
    v = tb.node("RESHAPE", [v], {"shape": [head_dim, n_heads, -1]}, out_hint + "_v_r")

    attn = tb.node("ATTENTION", [q, k, v, mask_name], {"kv_cache": False, "scale": 1.0 / float(np.sqrt(head_dim))},
                    out_hint + "_attn")
    o = tb.node("ADD", [tb.node("MUL_MAT", [ow, attn], None, out_hint + "_o_mm"), ob], None, out_hint + "_o")
    x_ct = tb.node("ADD", [x_ct, o], None, out_hint + "_res1")

    n3_w = tb.weight(f"{prefix}.norm3.weight", to_f32(sd[f"{name}.norm3.weight"]))
    n3_b = tb.weight(f"{prefix}.norm3.bias", to_f32(sd[f"{name}.norm3.bias"]))
    normed3 = apply_std_layer_norm(tb, x_ct, n3_w, n3_b, hp["ln_eps"], out_hint + "_ln3")

    inner_dim = dim * hp["ff_mult"]
    ff_out = build_snakebeta_ffn(tb, normed3, f"{prefix}.ff", sd, f"{name}.ff", dim, inner_dim, out_hint + "_ff")
    return tb.node("ADD", [x_ct, ff_out], None, out_hint)


def crop_conv_transpose_output(tb, x_full_tc, channels, pad, out_len_expr, out_hint):
    """Crops `pad` elements from EACH side of a raw (unpadded) CONV_TRANSPOSE_1D output along ne[0]
    (T) -- same VIEW-based crop as VITS's own HiFi-GAN upsample stages (`ggml_conv_transpose_1d` has
    no padding parameter, always emitting the "full" `(T-1)*s0+K` length).
    """
    return tb.node("VIEW", [x_full_tc], {"shape": [out_len_expr, channels], "offset": pad * 4}, out_hint)


def build_up_or_down_transformer_stack(tb, x_tc, mask_name, prefix, sd, name, channels, hp, out_hint):
    """`x -> rearrange(c,t) -> [BasicTransformerBlock] * n_blocks (n_blocks=1 in the real config) ->
    rearrange back`.
    """
    x_ct = tb.transpose_2d(x_tc, out_hint + "_ct")
    x_ct = build_basic_transformer_block(tb, x_ct, f"{prefix}.0", sd, f"{name}.0", channels, hp, mask_name, out_hint + "_btb0")
    return tb.transpose_2d(x_ct, out_hint + "_tc")


def build_decoder(sd, hp):
    tb = TopologyBuilder()
    n_feats = hp["n_feats"]
    ch = hp["channels"]
    time_embed_dim = hp["time_embed_dim"]
    in_channels = 2 * n_feats  # x (CFM latent) concat mu (conditioning)

    x_in = tb.node("CONCAT", ["z", "mu"], {"dim": 1}, "x_in")  # [T, 160]
    time_emb = build_time_mlp(tb, sd, in_channels, time_embed_dim, "decoder.estimator.time_mlp")

    # --- down_blocks[0]: real downsample (stride-2 conv) ---
    d0 = build_resnet_block1d(tb, x_in, time_emb, "decoder.estimator.down_blocks.0.0", sd,
                               "decoder.estimator.down_blocks.0.0", in_channels, ch, hp, "d0_res")
    d0 = build_up_or_down_transformer_stack(tb, d0, "attn_mask_full", "decoder.estimator.down_blocks.0.1", sd,
                                             "decoder.estimator.down_blocks.0.1", ch, hp, "d0_tf")
    hidden0 = d0  # skip connection, FULL resolution
    dsw, dsb = add_conv(tb, "decoder.estimator.down_blocks.0.2.conv", sd, "decoder.estimator.down_blocks.0.2.conv")
    d0_3 = tb.node("RESHAPE", [d0], {"shape": [-1, ch, 1]}, "d0_down_x3")
    d0_down = tb.node("CONV_1D", [dsw, d0_3], {"s0": 2, "p0": 1, "d0": 1}, "d0_down_conv")
    d0_down = tb.node("ADD", [d0_down, tb.node("RESHAPE", [dsb], {"shape": [1, ch, 1]}, "d0_down_b_r")],
                       None, "d0_down_convb")
    d0_down = tb.node("RESHAPE", [d0_down], {"shape": [-1, ch]}, "d0_down_2d")  # [T/2, 256]

    # --- down_blocks[1]: is_last (plain conv, no real downsample) ---
    d1 = build_resnet_block1d(tb, d0_down, time_emb, "decoder.estimator.down_blocks.1.0", sd,
                               "decoder.estimator.down_blocks.1.0", ch, ch, hp, "d1_res")
    d1 = build_up_or_down_transformer_stack(tb, d1, "attn_mask_half", "decoder.estimator.down_blocks.1.1", sd,
                                             "decoder.estimator.down_blocks.1.1", ch, hp, "d1_tf")
    hidden1 = d1  # skip connection, HALF resolution
    d1w, d1b = add_conv(tb, "decoder.estimator.down_blocks.1.2", sd, "decoder.estimator.down_blocks.1.2")
    d1_3 = tb.node("RESHAPE", [d1], {"shape": [-1, ch, 1]}, "d1_last_x3")
    x = tb.node("CONV_1D", [d1w, d1_3], {"s0": 1, "p0": 1, "d0": 1}, "d1_last_conv")
    x = tb.node("ADD", [x, tb.node("RESHAPE", [d1b], {"shape": [1, ch, 1]}, "d1_last_b_r")], None, "d1_last_convb")
    x = tb.node("RESHAPE", [x], {"shape": [-1, ch]}, "d1_last_2d")  # [T/2, 256]

    # --- mid_blocks: 2x [resnet + transformer], all at HALF resolution ---
    for i in range(2):
        x = build_resnet_block1d(tb, x, time_emb, f"decoder.estimator.mid_blocks.{i}.0", sd,
                                  f"decoder.estimator.mid_blocks.{i}.0", ch, ch, hp, f"mid{i}_res")
        x = build_up_or_down_transformer_stack(tb, x, "attn_mask_half", f"decoder.estimator.mid_blocks.{i}.1", sd,
                                                f"decoder.estimator.mid_blocks.{i}.1", ch, hp, f"mid{i}_tf")

    # --- up_blocks[0]: concat hidden1 (half-res skip), real upsample (ConvTranspose1d, stride2) ---
    x = tb.node("CONCAT", [x, hidden1], {"dim": 1}, "u0_concat")  # [T/2, 512]
    x = build_resnet_block1d(tb, x, time_emb, "decoder.estimator.up_blocks.0.0", sd,
                              "decoder.estimator.up_blocks.0.0", 2 * ch, ch, hp, "u0_res")
    x = build_up_or_down_transformer_stack(tb, x, "attn_mask_half", "decoder.estimator.up_blocks.0.1", sd,
                                            "decoder.estimator.up_blocks.0.1", ch, hp, "u0_tf")
    usw, usb = add_conv(tb, "decoder.estimator.up_blocks.0.2.conv", sd, "decoder.estimator.up_blocks.0.2.conv")
    x_full_raw = tb.node("CONV_TRANSPOSE_1D", [usw, x], {"s0": 2}, "u0_up_full")
    x_full_raw = tb.node("ADD", [x_full_raw, tb.node("RESHAPE", [usb], {"shape": [1, ch]}, "u0_up_b_r")],
                          None, "u0_up_biased")
    x = crop_conv_transpose_output(tb, x_full_raw, ch, 1, "$n_tokens", "u0_up_cropped")  # back to FULL res

    # --- up_blocks[1]: concat hidden0 (full-res skip), is_last (plain conv, no real upsample) ---
    x = tb.node("CONCAT", [x, hidden0], {"dim": 1}, "u1_concat")  # [T, 512]
    x = build_resnet_block1d(tb, x, time_emb, "decoder.estimator.up_blocks.1.0", sd,
                              "decoder.estimator.up_blocks.1.0", 2 * ch, ch, hp, "u1_res")
    x = build_up_or_down_transformer_stack(tb, x, "attn_mask_full", "decoder.estimator.up_blocks.1.1", sd,
                                            "decoder.estimator.up_blocks.1.1", ch, hp, "u1_tf")
    u1w, u1b = add_conv(tb, "decoder.estimator.up_blocks.1.2", sd, "decoder.estimator.up_blocks.1.2")
    x_3 = tb.node("RESHAPE", [x], {"shape": [-1, ch, 1]}, "u1_last_x3")
    x = tb.node("CONV_1D", [u1w, x_3], {"s0": 1, "p0": 1, "d0": 1}, "u1_last_conv")
    x = tb.node("ADD", [x, tb.node("RESHAPE", [u1b], {"shape": [1, ch, 1]}, "u1_last_b_r")], None, "u1_last_convb")
    x = tb.node("RESHAPE", [x], {"shape": [-1, ch]}, "u1_last_2d")  # [T, 256]

    # --- final_block + final_proj ---
    x = build_block1d(tb, x, "decoder.estimator.final_block", sd, "decoder.estimator.final_block", ch, ch, hp,
                       "final_block")
    fp_w, fp_b = add_conv1x1_as_matmul(tb, "decoder.estimator.final_proj", sd, "decoder.estimator.final_proj")
    x_ct_final = tb.transpose_2d(x, "final_ct")
    out = tb.node("ADD", [tb.node("MUL_MAT", [fp_w, x_ct_final], None, "final_mm"), fp_b], None, "final_b")
    out = tb.transpose_2d(out, "dphi_dt")  # back to [T, n_feats]

    inputs = [
        {"name": "z", "dtype": "f32", "shape": ["$n_tokens", str(n_feats)]},
        {"name": "mu", "dtype": "f32", "shape": ["$n_tokens", str(n_feats)]},
        {"name": "t", "dtype": "f32", "shape": ["1"]},
        {"name": "attn_mask_full", "dtype": "f32", "shape": ["$n_tokens", "$n_tokens"]},
        {"name": "attn_mask_half", "dtype": "f32", "shape": ["$n_tokens/2", "$n_tokens/2"]},
    ]
    return tb.topology(inputs, out), tb.weights, tb.int32_weights


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <matcha_ljspeech.ckpt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = load_matcha_checkpoint(ckpt_path)
    topo, weights, int32_names = build_decoder(sd, HP)
    write_gguf(out_dir / "matcha_decoder.gguf", "matcha_decoder", HP, topo, weights, int32_names)


if __name__ == "__main__":
    main()
