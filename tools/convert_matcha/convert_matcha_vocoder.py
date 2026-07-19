"""Converts the real HiFi-GAN v1 vocoder checkpoint (`generator_v1`, paired with Matcha-TTS's own
LJSpeech checkpoint via `matcha/cli.py`'s `VOCODER_URLS`) into a loom-engine GGUF topology
(`matcha_vocoder.gguf`): `mel [T,80] -> waveform [T*256]`.

Real config confirmed both from `matcha/hifigan/config.py`'s `v1` dict AND directly against the real
104-tensor state dict: `resblock="1"` (needs BOTH `convs1` AND `convs2` per resblock -- genuinely
DIFFERENT structure from VITS-piper's own HiFi-GAN checkpoint, which used `resblock="2"`, a single
`convs` list), `upsample_rates=[8,8,2,2]` (4 stages, hop_size=8*8*2*2=256, vs VITS-piper's 3 stages),
`upsample_kernel_sizes=[16,16,4,4]`, `upsample_initial_channel=512`,
`resblock_kernel_sizes=[3,7,11]`, each with dilations `(1,3,5)` for `convs1` (`convs2` always
dilation=1, confirmed from `matcha/hifigan/models.py::ResBlock1`). Real `Generator.forward`'s FINAL
`leaky_relu` (after the upsample/resblock loop, before `conv_post`) uses PyTorch's DEFAULT
`negative_slope=0.01` (no explicit slope arg in `F.leaky_relu(x)`), NOT the `0.1` used everywhere
else -- confirmed by re-reading `models.py` directly, same detail VITS's own converter already got
right (`convert_piper_vits/convert_vits.py`'s own `final_lrelu` node).

Tensor-layout convention: `[T,C]` (T=ne[0], CONV_1D's own native convention) throughout, matching
VITS's own HiFi-GAN vocoder conversion exactly.
"""
import sys
from pathlib import Path

from matcha_common import (
    TopologyBuilder, add_wn_conv, load_hifigan_checkpoint, write_gguf,
)

HP = {
    "n_feats": 80,
    "resblock_kernel_sizes": (3, 7, 11),
    "resblock_dilations": (1, 3, 5),  # convs1's per-layer dilation; convs2 always dilation=1
    "upsample_rates": (8, 8, 2, 2),
    "upsample_kernel_sizes": (16, 16, 4, 4),
    "upsample_initial_channel": 512,
}


def build_resblock1(tb, x_tc, ch, prefix, sd, name, kernel_size, dilations, out_hint):
    """`ResBlock1`: for each of 3 (convs1[k],convs2[k]) pairs -- lrelu -> convs1[k](dilation=d) ->
    lrelu -> convs2[k](dilation=1) -> residual add.
    """
    x = x_tc
    for k, d in enumerate(dilations):
        c1w, c1b = add_wn_conv(tb, f"{prefix}.convs1.{k}", sd, f"{name}.convs1.{k}")
        c2w, c2b = add_wn_conv(tb, f"{prefix}.convs2.{k}", sd, f"{name}.convs2.{k}")

        pad1 = (kernel_size * d - d) // 2
        xt = tb.node("LEAKY_RELU", [x], {"slope": 0.1}, f"{out_hint}{k}_lr1")
        xt3 = tb.node("RESHAPE", [xt], {"shape": [-1, ch, 1]}, f"{out_hint}{k}_xt3")
        xt = tb.node("CONV_1D", [c1w, xt3], {"s0": 1, "p0": pad1, "d0": d}, f"{out_hint}{k}_c1")
        xt = tb.node("ADD", [xt, tb.node("RESHAPE", [c1b], {"shape": [1, ch, 1]}, f"{out_hint}{k}_c1b_r")],
                      None, f"{out_hint}{k}_c1_b")
        xt = tb.node("RESHAPE", [xt], {"shape": [-1, ch]}, f"{out_hint}{k}_c1_2d")

        pad2 = (kernel_size * 1 - 1) // 2
        xt = tb.node("LEAKY_RELU", [xt], {"slope": 0.1}, f"{out_hint}{k}_lr2")
        xt3b = tb.node("RESHAPE", [xt], {"shape": [-1, ch, 1]}, f"{out_hint}{k}_xt3b")
        xt = tb.node("CONV_1D", [c2w, xt3b], {"s0": 1, "p0": pad2, "d0": 1}, f"{out_hint}{k}_c2")
        xt = tb.node("ADD", [xt, tb.node("RESHAPE", [c2b], {"shape": [1, ch, 1]}, f"{out_hint}{k}_c2b_r")],
                      None, f"{out_hint}{k}_c2_b")
        xt = tb.node("RESHAPE", [xt], {"shape": [-1, ch]}, f"{out_hint}{k}_c2_2d")

        x = tb.node("ADD", [xt, x], None, f"{out_hint}{k}_res")
    return x


def build_vocoder(sd, hp):
    tb = TopologyBuilder()
    n_feats = hp["n_feats"]
    upsample_rates = hp["upsample_rates"]
    upsample_kernel_sizes = hp["upsample_kernel_sizes"]
    resblock_kernel_sizes = hp["resblock_kernel_sizes"]
    dilations = hp["resblock_dilations"]
    num_kernels = len(resblock_kernel_sizes)
    upsample_initial_channel = hp["upsample_initial_channel"]

    conv_pre_w, conv_pre_b = add_wn_conv(tb, "conv_pre", sd, "conv_pre")
    mel3 = tb.node("RESHAPE", ["mel"], {"shape": [-1, n_feats, 1]}, "gen_in")
    x = tb.node("CONV_1D", [conv_pre_w, mel3], {"s0": 1, "p0": 3, "d0": 1}, "conv_pre_out")
    x = tb.node("ADD", [x, tb.node("RESHAPE", [conv_pre_b], {"shape": [1, upsample_initial_channel, 1]}, "cpb_r")],
                 None, "conv_pre_b")
    x = tb.node("RESHAPE", [x], {"shape": [-1, upsample_initial_channel]}, "conv_pre_2d")

    running_product = 1
    ch_out = upsample_initial_channel
    for stage in range(len(upsample_rates)):
        u = upsample_rates[stage]
        kk = upsample_kernel_sizes[stage]
        pad = (kk - u) // 2
        ch_out = upsample_initial_channel // (2 ** (stage + 1))
        up_w, up_b = add_wn_conv(tb, f"ups.{stage}", sd, f"ups.{stage}")

        x = tb.node("LEAKY_RELU", [x], {"slope": 0.1}, f"up{stage}_lrelu")
        x_full = tb.node("CONV_TRANSPOSE_1D", [up_w, x], {"s0": u}, f"up{stage}_full")
        x_full = tb.node("ADD", [x_full, tb.node("RESHAPE", [up_b], {"shape": [1, ch_out]}, f"up{stage}_b_r")],
                          None, f"up{stage}_biased")
        running_product *= u
        crop_shape_expr = f"$n_tokens*{running_product}"
        x = tb.node("VIEW", [x_full], {"shape": [crop_shape_expr, ch_out], "offset": pad * 4}, f"up{stage}_cropped")

        summed = None
        for j in range(num_kernels):
            resblock_idx = stage * num_kernels + j
            rk = resblock_kernel_sizes[j]
            prefix = f"resblocks.{resblock_idx}"
            rb_out = build_resblock1(tb, x, ch_out, prefix, sd, prefix, rk, dilations, f"rb{resblock_idx}_")
            summed = rb_out if summed is None else tb.node("ADD", [summed, rb_out], None, f"rb{resblock_idx}_sum")
        x = tb.node("SCALE", [summed], {"s": 1.0 / num_kernels}, f"stage{stage}_avg")

    conv_post_w, conv_post_b = add_wn_conv(tb, "conv_post", sd, "conv_post")
    x = tb.node("LEAKY_RELU", [x], {"slope": 0.01}, "final_lrelu")  # real code's DEFAULT slope, not 0.1
    x3 = tb.node("RESHAPE", [x], {"shape": [-1, ch_out, 1]}, "final_3d")
    x = tb.node("CONV_1D", [conv_post_w, x3], {"s0": 1, "p0": 3, "d0": 1}, "wav_pre_bias")
    x = tb.node("ADD", [x, tb.node("RESHAPE", [conv_post_b], {"shape": [1, 1, 1]}, "conv_post_b_r")], None, "wav_pre_tanh")
    wav = tb.node("TANH", [x], None, "wav")
    wav = tb.node("RESHAPE", [wav], {"shape": [-1]}, "wav_1d")

    inputs = [{"name": "mel", "dtype": "f32", "shape": ["$n_tokens", str(n_feats)]}]
    return tb.topology(inputs, wav), tb.weights, tb.int32_weights


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <generator_v1> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = load_hifigan_checkpoint(ckpt_path)
    topo, weights, int32_names = build_vocoder(sd, HP)
    write_gguf(out_dir / "matcha_vocoder.gguf", "matcha_vocoder", HP, topo, weights, int32_names)


if __name__ == "__main__":
    main()
