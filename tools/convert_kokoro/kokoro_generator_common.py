"""Reusable topology builders for Kokoro Generator's `AdaINResBlock1` (istftnet.py) -- the OTHER
"AdaIN*"-family resblock, distinct from `predictor.F0/N`'s `AdainResBlk1d` (convert_kokoro_f0n.py):
`AdaINResBlock1` has NO shortcut/upsample path at all (always `dim_in==dim_out`, always same-length),
uses 3 (dilation=1,3,5) `(AdaIN1d+Snake+conv1(dilated))` -> `(AdaIN1d+Snake+conv2(dilation=1))` ->
residual-add stages instead of `AdainResBlk1d`'s single norm1/act/conv1/norm2/act/conv2 pair. Used by
`Generator.resblocks` (kernel_size from `resblock_kernel_sizes=[3,7,11]`) and `Generator.noise_res`
(kernel_size=7 for all but the last upsample stage, 11 for the last).

Confirmed against the real checkpoint's state dict (`decoder.generator.{resblocks,noise_res}.*`):
  - `AdaIN1d`'s `InstanceNorm1d(affine=True)` has NO `.norm.weight`/`.norm.bias` keys anywhere here
    either -- the SAME never-trained-affine-params finding as every other `AdaIN1d`/`AdaLayerNorm`
    instance this whole milestone, so `add_adain1d` (originally written for F0Ntrain, same [T,C]
    convention) is reused here VERBATIM, not re-derived.
  - `alpha1`/`alpha2`: real shape `(1,channels,1)` (`nn.Parameter(torch.ones(1,channels,1))`) -- squeezed
    to a flat `(channels,)` array at conversion time and reshaped to `[1,channels]` in-graph for
    broadcasting against this project's `[T,C]` convention (channels=ne[1]).
  - `convs1`/`convs2`: ordinary weight-normed `Conv1d(channels,channels,kernel_size,dilation=d,
    padding=get_padding(kernel_size,d)=(kernel_size*d-d)//2)` -- "same"-length padding, confirmed by
    `get_padding`'s real formula in istftnet.py.

Snake activation (`xt + (1/a)*sin(a*xt)^2`) with a per-channel LEARNED `a` (not a scalar, unlike VITS's
Generator which has no such activation at all) needs the reciprocal `inv_alpha=1/alpha` folded at
CONVERSION time (plain numpy division), not a DIV node in-graph -- same "fold at conversion time"
precedent as weight-norm.
"""
import numpy as np

from convert_kokoro_f0n import add_adain1d, fold_weight_norm, to_f32


def get_padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2


def sk(sd_prefix, name):
    """Joins an sd_prefix and a state-dict key name, tolerating an empty prefix (standalone/synthetic
    fixtures with no real checkpoint prefix at all) without producing a stray leading '.'."""
    return f"{sd_prefix}.{name}" if sd_prefix else name


def add_snake(tb, x, prefix, sd, sd_prefix, channels, out_hint):
    """x: [T,channels]. `alpha` real shape (1,channels,1) -> squeezed to (channels,) here."""
    alpha = sd[f"{sd_prefix}"].detach().cpu().numpy().reshape(-1).astype(np.float32)
    inv_alpha = (1.0 / alpha).astype(np.float32)
    alpha_w = tb.weight(f"{prefix}.alpha", alpha)
    inv_alpha_w = tb.weight(f"{prefix}.inv_alpha", inv_alpha)
    alpha_r = tb.node("RESHAPE", [alpha_w], {"shape": [1, channels]}, f"{out_hint}_alpha_r")
    inv_alpha_r = tb.node("RESHAPE", [inv_alpha_w], {"shape": [1, channels]}, f"{out_hint}_inv_alpha_r")
    ax = tb.node("MUL", [x, alpha_r], None, f"{out_hint}_ax")
    sin_sq = tb.node("SQR", [tb.node("SIN", [ax], None, f"{out_hint}_sin")], None, f"{out_hint}_sinsq")
    term = tb.node("MUL", [sin_sq, inv_alpha_r], None, f"{out_hint}_term")
    return tb.node("ADD", [x, term], None, out_hint)


def add_adain_resblock1(tb, x, style_name, prefix, sd, sd_prefix, channels, style_dim, eps, kernel_size,
                         dilations, out_hint, seq_len_expr="$n_tokens"):
    """x: [T,channels]. Returns [T,channels] (same length/channels always -- no shortcut, no upsample).
    `seq_len_expr` defaults to the topology's own main "$n_tokens" symbol (F0Ntrain's own usage, where
    every block operates at that same length) but MUST be passed explicitly whenever this is reused at a
    length other than the topology's primary one (e.g. Generator.resblocks, which run at T0*10/T0*60 --
    real lengths derived from upsampling T0, not T0 itself) -- otherwise the internal RESHAPE calls would
    target the wrong size and abort on a nelements mismatch at graph-build time."""
    for i, d in enumerate(dilations):
        r = add_adain1d(tb, x, style_name, f"{prefix}.adain1.{i}", sd, sk(sd_prefix, f"adain1.{i}"),
                         channels, style_dim, eps, f"{out_hint}_n1_{i}")
        r = add_snake(tb, r, f"{prefix}.snake1.{i}", sd, sk(sd_prefix, f"alpha1.{i}"), channels, f"{out_hint}_s1_{i}")
        pad1 = get_padding(kernel_size, d)
        w1 = tb.weight(f"{prefix}.convs1.{i}.weight",
                       fold_weight_norm(sd[sk(sd_prefix, f"convs1.{i}.weight_g")], sd[sk(sd_prefix, f"convs1.{i}.weight_v")]))
        b1 = tb.weight(f"{prefix}.convs1.{i}.bias", to_f32(sd[sk(sd_prefix, f"convs1.{i}.bias")]))
        r3 = tb.node("RESHAPE", [r], {"shape": [seq_len_expr, channels, 1]}, f"{out_hint}_r3_{i}")
        r = tb.node("CONV_1D", [w1, r3], {"s0": 1, "p0": pad1, "d0": d}, f"{out_hint}_c1_raw_{i}")
        r = tb.node("ADD", [r, tb.node("RESHAPE", [b1], {"shape": [1, channels, 1]}, f"{out_hint}_c1_b_{i}")],
                    None, f"{out_hint}_c1_biased_{i}")
        r = tb.node("RESHAPE", [r], {"shape": [seq_len_expr, channels]}, f"{out_hint}_c1_2d_{i}")

        r = add_adain1d(tb, r, style_name, f"{prefix}.adain2.{i}", sd, sk(sd_prefix, f"adain2.{i}"),
                         channels, style_dim, eps, f"{out_hint}_n2_{i}")
        r = add_snake(tb, r, f"{prefix}.snake2.{i}", sd, sk(sd_prefix, f"alpha2.{i}"), channels, f"{out_hint}_s2_{i}")
        pad2 = get_padding(kernel_size, 1)
        w2 = tb.weight(f"{prefix}.convs2.{i}.weight",
                       fold_weight_norm(sd[sk(sd_prefix, f"convs2.{i}.weight_g")], sd[sk(sd_prefix, f"convs2.{i}.weight_v")]))
        b2 = tb.weight(f"{prefix}.convs2.{i}.bias", to_f32(sd[sk(sd_prefix, f"convs2.{i}.bias")]))
        r3b = tb.node("RESHAPE", [r], {"shape": [seq_len_expr, channels, 1]}, f"{out_hint}_r3b_{i}")
        r = tb.node("CONV_1D", [w2, r3b], {"s0": 1, "p0": pad2, "d0": 1}, f"{out_hint}_c2_raw_{i}")
        r = tb.node("ADD", [r, tb.node("RESHAPE", [b2], {"shape": [1, channels, 1]}, f"{out_hint}_c2_b_{i}")],
                    None, f"{out_hint}_c2_biased_{i}")
        r = tb.node("RESHAPE", [r], {"shape": [seq_len_expr, channels]}, f"{out_hint}_c2_2d_{i}")

        x = tb.node("ADD", [r, x], None, f"{out_hint}_sum_{i}" if i < len(dilations) - 1 else out_hint)
    return x
