"""Assembles Kokoro's Generator "core" (istftnet.py's `Generator.forward`, the upsample stack + noise
path + resblocks + conv_post + inverse STFT -- NOT the `SineGen`/forward-STFT piece that produces `har`,
which stays a SEPARATE topology (convert_kokoro_sinegen.py/convert_kokoro_stft.py, already verified) run
first by the host driver and fed in here as a ready-made "har" input, same "compose already-verified
pieces via the host driver" pattern as `BiLstmStepper` feeding per-block `GraphBuilder::build()` calls
elsewhere in this project).

Real config (`config.json`'s `istftnet`): `upsample_rates=[10,6]`, `upsample_kernel_sizes=[20,12]`,
`upsample_initial_channel=512`, `resblock_kernel_sizes=[3,7,11]`, `resblock_dilation_sizes=[[1,3,5]]*3`,
`gen_istft_n_fft=20`, `gen_istft_hop_size=5`. Real channel/kernel shapes confirmed directly against the
checkpoint's `decoder.generator.{ups,noise_convs,conv_post}.*` state-dict entries before writing any of
this (see BACKLOG.md).

Length bookkeeping (verified algebraically, not assumed -- see BACKLOG.md for the derivation): letting
`T0` = this topology's own "$n_tokens" (the Decoder's own decode-stack output length, i.e. `x`'s length),
`upsample_rates=[10,6]` both use `kernel_size=2*stride` with `padding=(kernel_size-stride)//2=stride//2`,
which is EXACTLY the "integer-exact upsample" ConvTranspose1d config -- `ups[0]` maps `T0 -> T0*10`,
`ups[1]` maps `T0*10 -> T0*60`, both via `ggml_conv_transpose_1d`'s p0=0-only limitation + a crop (VITS's
own `test_hifigan_generator` precedent), never a "floor()"-guarded fractional formula (both resolve to a
clean exact multiplication). The one-sample `ReflectionPad1d((1,0))` (applied only after the LAST
upsample stage) degenerates to "prepend a copy of the element at index 1" (confirmed directly against
real `torch.nn.ReflectionPad1d((1,0))`) -- a `VIEW`(1-sample slice at index 1)+`CONT`+`CONCAT` composition,
not a new primitive. `har`'s own frame count (`T0*60+1`, ONE more than the main path's `T0*60`, from the
forward-STFT's own `n_frames` formula) is reconciled by `noise_convs[i]`'s own stride/padding formula --
proven algebraically to land on exactly `T0*10` (stage 0, strided conv downsamples `har`) and exactly
`T0*60+1` (stage 1, a 1x1 conv that doesn't change length, matching the reflection-padded main path).

No new primitive needed beyond this milestone's `EXP` (trivial `ggml_exp` wrapper, needed for
`spec=exp(x[:n_freq])`) -- everything else composes from primitives already verified earlier this
milestone (`CONV_TRANSPOSE_1D`+crop, `AdaINResBlock1`, `CONCAT`, the inverse-STFT node sequence).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

from convert_kokoro_f0n import fold_weight_norm, to_f32
from convert_kokoro_stft import build_inverse_synth_kernels
from kokoro_generator_common import add_adain_resblock1, sk

HP = {
    "upsample_rates": [10, 6],
    "upsample_kernel_sizes": [20, 12],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilations": (1, 3, 5),
    "gen_istft_n_fft": 20,
    "gen_istft_hop_size": 5,
    "style_dim": 128,
    "ada_ln_eps": 1e-5,
}


class TopologyBuilder:
    def __init__(self):
        self.nodes = []
        self.weights = {}
        self._counter = 0

    def _fresh(self, hint):
        self._counter += 1
        return f"{hint}_{self._counter}"

    def node(self, op, inputs, attrs=None, out_hint="t", name=None):
        out = name if name is not None else self._fresh(out_hint)
        entry = {"op": op, "inputs": list(inputs), "outputs": [out]}
        if attrs:
            entry["attrs"] = attrs
        self.nodes.append(entry)
        return out

    def weight(self, name, array):
        self.weights[name] = np.asarray(array)
        return name

    def topology(self, inputs, output):
        return {"version": 1, "inputs": inputs, "output": output, "nodes": self.nodes}


def add_conv_transpose1d_crop(tb, x, prefix, sd, sd_prefix, kernel_size, stride, cropped_len_expr, out_hint):
    """x: [T_in,in_ch]. Real weight-normed ConvTranspose1d(in_ch,out_ch,kernel_size,stride,
    padding=(kernel_size-stride)//2) via ggml's p0=0-only CONV_TRANSPOSE_1D + a crop (VITS's own
    test_hifigan_generator precedent) -- `cropped_len_expr` is the caller-supplied POST-crop length (this
    project's real upsample configs always resolve this to an exact clean multiplication, verified
    algebraically per-call site, not derived generically here)."""
    w = tb.weight(f"{prefix}.weight", fold_weight_norm(sd[sk(sd_prefix, "weight_g")], sd[sk(sd_prefix, "weight_v")]))
    b = sd[sk(sd_prefix, "bias")]
    out_ch = b.shape[0]
    b_w = tb.weight(f"{prefix}.bias", to_f32(b))
    full = tb.node("CONV_TRANSPOSE_1D", [w, x], {"s0": stride}, f"{out_hint}_full")
    full_biased = tb.node("ADD", [full, tb.node("RESHAPE", [b_w], {"shape": [1, out_ch]}, f"{out_hint}_b_r")],
                          None, f"{out_hint}_full_biased")
    padding = (kernel_size - stride) // 2
    cropped_view = tb.node("VIEW", [full_biased], {"shape": [cropped_len_expr, out_ch], "offset": padding * 4},
                            f"{out_hint}_view")
    return tb.node("CONT", [cropped_view], None, out_hint)


def add_reflect_pad1_left(tb, x, channels, out_hint):
    """x: [T,channels]. Real `nn.ReflectionPad1d((1,0))` degenerates (for a width-1 left pad) to
    "prepend a copy of the element at index 1" -- confirmed directly against real
    torch.nn.ReflectionPad1d((1,0)) (see BACKLOG.md), not assumed from the general reflection formula."""
    slice_view = tb.node("VIEW", [x], {"shape": [1, channels], "offset": 4}, f"{out_hint}_slice")
    slice_cont = tb.node("CONT", [slice_view], None, f"{out_hint}_slice_cont")
    return tb.node("CONCAT", [slice_cont, x], {"dim": 0}, out_hint)


def add_conv1d_same(tb, x, prefix, sd, sd_prefix, channels_in, channels_out, kernel_size, seq_len_expr, out_hint):
    """x: [T,channels_in]. Ordinary weight-normed "same"-padding Conv1d(stride=1,dilation=1)."""
    w = tb.weight(f"{prefix}.weight", fold_weight_norm(sd[sk(sd_prefix, "weight_g")], sd[sk(sd_prefix, "weight_v")]))
    b = tb.weight(f"{prefix}.bias", to_f32(sd[sk(sd_prefix, "bias")]))
    pad = (kernel_size - 1) // 2
    x3 = tb.node("RESHAPE", [x], {"shape": [seq_len_expr, channels_in, 1]}, f"{out_hint}_x3")
    conv = tb.node("CONV_1D", [w, x3], {"s0": 1, "p0": pad, "d0": 1}, f"{out_hint}_raw")
    biased = tb.node("ADD", [conv, tb.node("RESHAPE", [b], {"shape": [1, channels_out, 1]}, f"{out_hint}_b_r")],
                      None, f"{out_hint}_biased")
    return tb.node("RESHAPE", [biased], {"shape": [seq_len_expr, channels_out]}, out_hint)


def add_strided_conv1d(tb, x, prefix, sd, sd_prefix, channels_in, channels_out, kernel_size, stride, padding,
                        seq_len_expr, out_hint):
    """x: [T,channels_in]. Ordinary (non-weight-normed -- noise_convs are plain nn.Conv1d) strided Conv1d.
    Output length is whatever ggml's own CONV_1D formula computes (not pre-declared via an expression --
    only the INPUT reshape needs `seq_len_expr`); verified algebraically to land on the exact real
    architecture's own intended length at each real call site (see module docstring)."""
    w = tb.weight(f"{prefix}.weight", to_f32(sd[sk(sd_prefix, "weight")]))
    b = tb.weight(f"{prefix}.bias", to_f32(sd[sk(sd_prefix, "bias")]))
    x3 = tb.node("RESHAPE", [x], {"shape": [seq_len_expr, channels_in, 1]}, f"{out_hint}_x3")
    conv = tb.node("CONV_1D", [w, x3], {"s0": stride, "p0": padding, "d0": 1}, f"{out_hint}_raw")
    biased = tb.node("ADD", [conv, tb.node("RESHAPE", [b], {"shape": [1, channels_out, 1]}, f"{out_hint}_b_r")],
                      None, f"{out_hint}_biased")
    # -1 must be a JSON integer literal (not the string "-1") to trigger RESHAPE's "infer this dimension"
    # handling (op_reshape checks `v.is_number_integer()`, so a string would be evaluated as an ordinary
    # -1 dimension instead of triggering inference).
    return tb.node("RESHAPE", [biased], {"shape": [-1, channels_out]}, out_hint)


def build_generator(hp, sd, sd_prefix):
    """Inputs: "x" [$n_tokens,512], "style" [style_dim], "har" [$n_tokens*60+1, 2*n_freq], "wsum"
    [(T_har-1)*hop+n_fft]. Output: waveform [$n_tokens*300]."""
    tb = TopologyBuilder()
    n_fft = hp["gen_istft_n_fft"]
    hop = hp["gen_istft_hop_size"]
    n_freq = n_fft // 2 + 1
    style_dim = hp["style_dim"]
    eps = hp["ada_ln_eps"]
    dilations = hp["resblock_dilations"]
    kernel_sizes = hp["resblock_kernel_sizes"]
    num_kernels = len(kernel_sizes)
    upsample_rates = hp["upsample_rates"]
    upsample_kernel_sizes = hp["upsample_kernel_sizes"]
    num_upsamples = len(upsample_rates)
    uic = hp["upsample_initial_channel"]

    t_har_expr = "$n_tokens*60+1"

    x = "x"
    t_expr = "$n_tokens"
    for i in range(num_upsamples):
        ch_in = uic // (2 ** i)
        ch_out = uic // (2 ** (i + 1))
        stride = upsample_rates[i]
        k = upsample_kernel_sizes[i]

        x = tb.node("LEAKY_RELU", [x], {"slope": 0.1}, f"stage{i}_lrelu")

        stride_f0 = 1
        for r in upsample_rates[i + 1:]:
            stride_f0 *= r
        if i + 1 < num_upsamples:
            noise_k = stride_f0 * 2
            noise_pad = (stride_f0 + 1) // 2
            noise_res_k = 7
        else:
            noise_k = 1
            noise_pad = 0
            noise_res_k = 11
        # x_source's REAL length after the strided conv is whatever matches x's own length post-ups[i]
        # (proven algebraically equal, see module docstring: T0*10 for stage 0, T0*60+1 for the reflect-
        # padded last stage) -- NOT t_har_expr (har's own, always-121-style length), a real bug caught
        # here before ever running anything (add_adain_resblock1's RESHAPE calls need x_source's actual
        # post-conv length, not the pre-conv "har" length it was strided-conv'd FROM).
        t_next_expr = f"({t_expr})*{stride}"
        x_source_len_expr = t_next_expr if i < num_upsamples - 1 else f"({t_next_expr})+1"

        x_source = add_strided_conv1d(tb, "har", f"stage{i}.noise_conv", sd, sk(sd_prefix, f"noise_convs.{i}"),
                                       2 * n_freq, ch_out, noise_k, stride_f0, noise_pad, t_har_expr,
                                       f"stage{i}_xsource_raw")
        x_source = add_adain_resblock1(tb, x_source, "style", f"stage{i}.noise_res", sd,
                                        sk(sd_prefix, f"noise_res.{i}"), ch_out, style_dim, eps, noise_res_k,
                                        dilations, f"stage{i}_xsource", seq_len_expr=x_source_len_expr)

        x = add_conv_transpose1d_crop(tb, x, f"stage{i}.ups", sd, sk(sd_prefix, f"ups.{i}"), k, stride,
                                       t_next_expr, f"stage{i}_ups")
        t_expr = t_next_expr
        if i == num_upsamples - 1:
            x = add_reflect_pad1_left(tb, x, ch_out, f"stage{i}_reflect")
            t_expr = f"({t_expr})+1"

        x = tb.node("ADD", [x, x_source], None, f"stage{i}_x_plus_source")

        xs = None
        for j in range(num_kernels):
            rb = add_adain_resblock1(tb, x, "style", f"stage{i}.resblock{j}", sd,
                                      sk(sd_prefix, f"resblocks.{i * num_kernels + j}"), ch_out, style_dim, eps,
                                      kernel_sizes[j], dilations, f"stage{i}_rb{j}", seq_len_expr=t_expr)
            xs = rb if xs is None else tb.node("ADD", [xs, rb], None, f"stage{i}_rbsum{j}")
        x = tb.node("SCALE", [xs], {"s": 1.0 / num_kernels}, f"stage{i}_x_avg")

    ch_final = uic // (2 ** num_upsamples)
    x = tb.node("LEAKY_RELU", [x], {"slope": 0.01}, "final_lrelu")  # PyTorch's *default* slope, not 0.1
    conv_out = add_conv1d_same(tb, x, "conv_post", sd, sk(sd_prefix, "conv_post"), ch_final, 2 * n_freq, 7,
                                t_expr, "conv_post_out")

    spec = tb.node("EXP", [tb.node("VIEW", [conv_out], {"shape": [t_expr, n_freq], "offset": 0}, "spec_view")],
                    None, "spec")
    phase_offset = f"({t_expr})*{n_freq}*4"
    phase = tb.node("SIN", [tb.node("VIEW", [conv_out], {"shape": [t_expr, n_freq], "offset": phase_offset},
                                     "phase_view")], None, "phase")

    cos_synth, neg_sin_synth = build_inverse_synth_kernels(n_fft)
    cos_w = tb.weight("stft.cos_synth", cos_synth)
    neg_sin_w = tb.weight("stft.neg_sin_synth", neg_sin_synth)
    re = tb.node("MUL", [spec, tb.node("COS", [phase], None, "cosph")], None, "re")
    im = tb.node("MUL", [spec, tb.node("SIN", [phase], None, "sinph")], None, "im")
    out_len_full_expr = f"(({t_expr})-1)*{hop}+{n_fft}"
    re_contrib = tb.node("CONV_TRANSPOSE_1D", [cos_w, re], {"s0": hop}, "re_contrib")
    im_contrib = tb.node("CONV_TRANSPOSE_1D", [neg_sin_w, im], {"s0": hop}, "im_contrib")
    numerator = tb.node("ADD", [re_contrib, im_contrib], None, "numerator")
    numerator_1d = tb.node("RESHAPE", [numerator], {"shape": [out_len_full_expr]}, "numerator_1d")
    normalized = tb.node("DIV", [numerator_1d, "wsum"], None, "normalized")
    pad = n_fft // 2
    cropped_len_expr = f"{out_len_full_expr}-{2 * pad}"
    out = tb.node("VIEW", [normalized], {"shape": [cropped_len_expr], "offset": pad * 4}, "waveform")

    inputs = [
        {"name": "x", "dtype": "f32", "shape": ["$n_tokens", str(uic)]},
        {"name": "style", "dtype": "f32", "shape": [str(style_dim)]},
        {"name": "har", "dtype": "f32", "shape": [t_har_expr, str(2 * n_freq)]},
        {"name": "wsum", "dtype": "f32", "shape": [out_len_full_expr]},
    ]
    return tb.topology(inputs, out), tb.weights


def make_synthetic_state_dict(rng, hp):
    """Real checkpoint weights get wired in when this is tied into the full Decoder (task #90) -- this
    standalone converter verifies the WIRING with synthetic weights of the REAL shapes (confirmed against
    the checkpoint directly, see module docstring), same precedent as VITS's test_hifigan_generator and
    this milestone's own test_e2e_kokoro_adainresblock1.cpp."""
    sd = {}
    uic = hp["upsample_initial_channel"]
    n_fft = hp["gen_istft_n_fft"]
    n_freq = n_fft // 2 + 1
    style_dim = hp["style_dim"]
    dilations = hp["resblock_dilations"]
    kernel_sizes = hp["resblock_kernel_sizes"]
    upsample_rates = hp["upsample_rates"]
    upsample_kernel_sizes = hp["upsample_kernel_sizes"]
    num_upsamples = len(upsample_rates)

    def randn(shape, scale=0.2):
        return torch.from_numpy(rng.normal(scale=scale, size=shape).astype(np.float32))

    def add_resblock1_weights(prefix, channels, kernel_size):
        for i in range(len(dilations)):
            for norm in ("adain1", "adain2"):
                sd[f"{prefix}.{norm}.{i}.fc.weight"] = randn((2 * channels, style_dim))
                sd[f"{prefix}.{norm}.{i}.fc.bias"] = randn((2 * channels,), 0.1)
            for a in ("alpha1", "alpha2"):
                sd[f"{prefix}.{a}.{i}"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(1, channels, 1)).astype(np.float32))
            for c in ("convs1", "convs2"):
                sd[f"{prefix}.{c}.{i}.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(channels, 1, 1)).astype(np.float32))
                sd[f"{prefix}.{c}.{i}.weight_v"] = randn((channels, channels, kernel_size))
                sd[f"{prefix}.{c}.{i}.bias"] = randn((channels,), 0.1)

    for i in range(num_upsamples):
        ch_in = uic // (2 ** i)
        ch_out = uic // (2 ** (i + 1))
        k = upsample_kernel_sizes[i]
        sd[f"ups.{i}.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(ch_in, 1, 1)).astype(np.float32))
        sd[f"ups.{i}.weight_v"] = randn((ch_in, ch_out, k))
        sd[f"ups.{i}.bias"] = randn((ch_out,), 0.1)

        stride_f0 = 1
        for r in upsample_rates[i + 1:]:
            stride_f0 *= r
        noise_k = stride_f0 * 2 if i + 1 < num_upsamples else 1
        noise_res_k = 7 if i + 1 < num_upsamples else 11
        sd[f"noise_convs.{i}.weight"] = randn((ch_out, 2 * n_freq, noise_k))
        sd[f"noise_convs.{i}.bias"] = randn((ch_out,), 0.1)
        add_resblock1_weights(f"noise_res.{i}", ch_out, noise_res_k)

        for j, ks in enumerate(kernel_sizes):
            add_resblock1_weights(f"resblocks.{i * len(kernel_sizes) + j}", ch_out, ks)

    ch_final = uic // (2 ** num_upsamples)
    sd["conv_post.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(2 * n_freq, 1, 1)).astype(np.float32))
    sd["conv_post.weight_v"] = randn((2 * n_freq, ch_final, 7))
    sd["conv_post.bias"] = randn((2 * n_freq,), 0.1)
    return sd


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <out_dir>", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = HP

    rng = np.random.RandomState(31)
    sd = make_synthetic_state_dict(rng, hp)
    np.savez(out_dir / "kokoro_generator_sd.npz", **{k: v.numpy() for k, v in sd.items()})

    topo, weights = build_generator(hp, sd, "")
    w = GGUFWriter(str(out_dir / "kokoro_generator.gguf"), "loom-kokoro-generator")
    w.add_string("model.graph_topology", json.dumps(topo))
    for name, arr in weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'kokoro_generator.gguf'}, {len(weights)} weights")


if __name__ == "__main__":
    main()
