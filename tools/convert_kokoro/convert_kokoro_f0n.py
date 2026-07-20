"""Converts Kokoro's `F0Ntrain` (modules.py's `ProsodyPredictor.F0Ntrain` -- the F0/energy prediction
half of `ProsodyPredictor`, NOT yet the frame-expansion that produces its real input, see BACKLOG.md).
Builds `AdaIN1d` and `AdainResBlk1d` as reusable topology builders first (verified against a single real,
simplest block instance -- `predictor.F0.0`, `dim_in=dim_out=512`, no learned shortcut, no upsample --
before tackling the upsampling variant, matching this project's usual bottom-up discipline), then
`F0Ntrain`'s full shared-BiLSTM + two 3-block stacks + `F0_proj`/`N_proj` assembly.

Real architecture confirmed against `istftnet.py`'s `AdaIN1d`/`AdainResBlk1d` classes + the real
checkpoint's state dict:
  - `AdaIN1d`'s `InstanceNorm1d(affine=True)` has NEVER-TRAINED/-SAVED affine params (confirmed no
    `.norm.weight`/`.norm.bias` tensors exist in the state dict for ANY `AdaIN1d` instance, matching the
    already-verified "no-op affine" finding from earlier in this milestone) -- so this is exactly the
    same "plain InstanceNorm + style-derived `(1+gamma)*x+beta`" composition already verified for
    `AdaLayerNorm`, just with the OPPOSITE tensor-axis convention: `AdaLayerNorm` operates on this
    project's channel-first `[C,T]` (normalizing over `ne[0]`=channels, ordinary LayerNorm), while
    `AdaIN1d` needs `[T,C]` (`CONV_1D`'s own convention, normalizing over `ne[0]`=TIME per channel, real
    InstanceNorm) -- the SAME `LAYER_NORM` primitive, a genuinely different axis convention is what makes
    it compute a different normalization; not a coincidence, not a shortcut, a real distinct fact about
    what's fed in each time.
  - `AdainResBlk1d`'s `conv1`/`conv2` are ORDINARY (non-depthwise) weight-normed `Conv1d(dim,dim,
    kernel=3,padding=1)` -- this project's plain `CONV_1D` primitive, weight-norm folded at conversion
    time (same `fold_weight_norm` helper as `TextEncoder`'s own convs).
  - `predictor.F0.0`/`predictor.N.0` (`dim_in=dim_out=512`) have NO `conv1x1` (no learned shortcut, since
    `dim_in==dim_out`) and NO `pool` (no upsample) -- confirmed via the real state dict (no
    `F0.0.conv1x1.*`/`F0.0.pool.*` keys at all) -- the simplest possible instance, converted/verified
    FIRST before the upsampling variant (`F0.1`, which needs the depthwise-ConvTranspose1d composition
    verified earlier in this milestone, plus the learned `conv1x1` shortcut).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

from convert_kokoro_duration_predictor import write_bilstm_ggufs

HP = {
    "style_dim": 128,
    "ln_eps": 1e-5,  # nn.InstanceNorm1d's own default eps
    "leaky_slope": 0.2,
}


def to_f32(t):
    return t.detach().cpu().numpy().astype(np.float32)


def fold_weight_norm(weight_g, weight_v):
    g = weight_g.detach().numpy() if torch.is_tensor(weight_g) else np.asarray(weight_g)
    v = weight_v.detach().numpy() if torch.is_tensor(weight_v) else np.asarray(weight_v)
    norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape([-1] + [1] * (v.ndim - 1))
    return (g * v / norm).astype(np.float32)


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
        arr = np.asarray(array)
        if name in self.weights and self.weights[name].shape != arr.shape:
            raise ValueError(f"weight {name!r} already registered with a different shape")
        self.weights[name] = arr
        return name

    def topology(self, inputs, output):
        return {"version": 1, "inputs": inputs, "output": output, "nodes": self.nodes}


def add_adain1d(tb, x, style_name, prefix, sd, sd_prefix, channels, style_dim, eps, out_hint):
    """x: [T,channels] (CONV_1D convention). Returns [T,channels]. See module docstring for why plain
    LAYER_NORM on this axis convention is exactly InstanceNorm1d."""
    normed = tb.node("LAYER_NORM", [x], {"eps": eps}, f"{out_hint}_normed")
    fc_w = tb.weight(f"{prefix}.fc.weight", to_f32(sd[f"{sd_prefix}.fc.weight"]))
    fc_b = tb.weight(f"{prefix}.fc.bias", to_f32(sd[f"{sd_prefix}.fc.bias"]))
    h = tb.node("ADD", [tb.node("MUL_MAT", [fc_w, style_name], None, f"{out_hint}_h_mm"), fc_b], None, f"{out_hint}_h")
    gamma = tb.node("VIEW", [h], {"shape": [channels], "offset": 0}, f"{out_hint}_gamma")
    beta = tb.node("VIEW", [h], {"shape": [channels], "offset": channels * 4}, f"{out_hint}_beta")
    gamma_r = tb.node("RESHAPE", [gamma], {"shape": [1, channels]}, f"{out_hint}_gamma_r")
    beta_r = tb.node("RESHAPE", [beta], {"shape": [1, channels]}, f"{out_hint}_beta_r")
    gamma_p1 = tb.node("ADD", [gamma_r, tb.weight(f"{prefix}.one", np.array([1.0], dtype=np.float32))],
                       None, f"{out_hint}_gamma_p1")
    scaled = tb.node("MUL", [normed, gamma_p1], None, f"{out_hint}_scaled")
    return tb.node("ADD", [scaled, beta_r], None, out_hint)


def add_depthwise_conv_transpose_upsample(tb, x, prefix, sd, sd_prefix, channels, out_hint):
    """The "pool" ConvTranspose1d(kernel=3,stride=2,padding=1,output_padding=1,groups=channels) --
    composed entirely from existing primitives, verified in test_primitive_registry.cpp's
    test_depthwise_conv_transpose_1d_via_composition BEFORE being used here. x: [T,channels]."""
    w_raw = fold_weight_norm(sd[f"{sd_prefix}.weight_g"], sd[f"{sd_prefix}.weight_v"])  # (channels,1,3)
    w_flipped = w_raw[:, :, ::-1].copy()
    # GGUFWriter reverses numpy axis order -> ggml ne=[3,1,channels], matching CONV_1D_DW's own kernel
    # convention (K,1,channels).
    kernel = tb.weight(f"{prefix}.pool.weight", w_flipped)
    bias = tb.weight(f"{prefix}.pool.bias", to_f32(sd[f"{sd_prefix}.bias"]))

    stride, kernel_size, padding, output_padding = 2, 3, 1, 1
    d3 = tb.node("RESHAPE", [x], {"shape": [1, "$n_tokens", channels]}, f"{out_hint}_d3")
    stuffed3 = tb.node("PAD_1D", [d3], {"lp0": 0, "rp0": stride - 1}, f"{out_hint}_stuffed3")
    overstuffed = tb.node("RESHAPE", [stuffed3], {"shape": [f"$n_tokens*{stride}", channels]}, f"{out_hint}_overstuffed")
    std_len = f"($n_tokens-1)*{stride}+1"
    truncated_view = tb.node("VIEW", [overstuffed], {"shape": [std_len, channels]}, f"{out_hint}_trunc_v")
    truncated = tb.node("CONT", [truncated_view], None, f"{out_hint}_trunc")
    pad_each = kernel_size - 1 - padding
    padded = tb.node("PAD_1D", [truncated], {"lp0": pad_each, "rp0": pad_each + output_padding}, f"{out_hint}_padded")
    conv = tb.node("CONV_1D_DW", [kernel, padded], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_conv")
    bias_r = tb.node("RESHAPE", [bias], {"shape": [1, channels]}, f"{out_hint}_bias_r")
    return tb.node("ADD", [conv, bias_r], None, out_hint)


def build_proj1x1(sd, sd_name, prefix="proj"):
    """F0_proj/N_proj: plain Conv1d(256,1,kernel_size=1) -- applied directly via CONV_1D on the
    AdainResBlk1d stack's own [T,C] output convention (T=ne[0]), NOT a MUL_MAT-as-matmul trick (which
    would need a [C,T] channel-first transpose first, unlike VITS's own conv1x1-as-matmul sites, which
    were channel-first ALREADY for attention reasons) -- the real weight shape (1,256,1) numpy -> ggml
    ne=[1,256,1] = [K,IC,OC] already matches CONV_1D's own kernel convention directly, no squeeze needed
    at all. `prefix` namespaces the weight names (default "proj", matching this module's historical
    hardcoded name) -- F0_proj and N_proj are two distinct instances with genuinely different weight
    values, a real collision once consolidated into one combined GGUF file."""
    tb = TopologyBuilder()
    w = tb.weight(f"{prefix}.weight", to_f32(sd[f"{sd_name}.weight"]))
    b = tb.weight(f"{prefix}.bias", to_f32(sd[f"{sd_name}.bias"]))
    x3 = tb.node("RESHAPE", ["x"], {"shape": ["$n_tokens", 256, 1]}, "x3")
    conv = tb.node("CONV_1D", [w, x3], {"s0": 1, "p0": 0, "d0": 1}, "conv")
    conv = tb.node("ADD", [conv, tb.node("RESHAPE", [b], {"shape": [1, 1, 1]}, "bias_r")], None, "conv_biased")
    out = tb.node("RESHAPE", [conv], {"shape": ["$n_tokens"]}, "out")
    inputs = [{"name": "x", "dtype": "f32", "shape": ["$n_tokens", "256"]}]
    return tb.topology(inputs, out), tb.weights


def build_stack(sd, name_prefix, sd_names, dims, hp):
    """Returns {i: (topo, weights)} for i=0,1,2 -- same per-block prefixing (`{name_prefix}_{i}`, already
    collision-free since add_adain_resblk1d takes an explicit `prefix` per block) as write_stack's own
    file-per-block convention, just returned in memory instead of written to disk."""
    result = {}
    for i, (sd_name, (dim_in, dim_out, upsample)) in enumerate(zip(sd_names, dims)):
        tb = TopologyBuilder()
        out = add_adain_resblk1d(tb, "x", "style", f"{name_prefix}_{i}", sd, sd_name, dim_in, dim_out,
                                  hp["style_dim"], hp["ln_eps"], hp["leaky_slope"], upsample=upsample,
                                  out_hint="out")
        inputs = [
            {"name": "x", "dtype": "f32", "shape": ["$n_tokens", str(dim_in)]},
            {"name": "style", "dtype": "f32", "shape": [str(hp["style_dim"])]},
        ]
        result[i] = (tb.topology(inputs, out), tb.weights)
    return result


def add_adain_resblk1d(tb, x, style_name, prefix, sd, sd_prefix, dim_in, dim_out, style_dim, eps,
                        leaky_slope, upsample, out_hint):
    """x: [T,dim_in]. Returns [T,dim_out]. `upsample`: bool. `dim_in != dim_out` implies a learned 1x1
    shortcut conv (confirmed real: AdainResBlk1d.learned_sc = dim_in != dim_out)."""
    # --- shortcut ---
    seq_len_expr = "2*$n_tokens" if upsample else "$n_tokens"
    sc = x
    if upsample:
        # NOTE: AdainResBlk1d._shortcut's own `self.upsample` is the SEPARATE, plain nearest-neighbor
        # UpSample1d (INTERPOLATE_1D, mode=nearest) -- NOT the same learned depthwise-ConvTranspose1d
        # "pool" the residual path uses. Confirmed from istftnet.py's real _shortcut/_residual split
        # directly: `_shortcut` calls `self.upsample(x)` (UpSample1d), `_residual` calls `self.pool(x)`
        # (the learned ConvTranspose1d) -- genuinely different upsampling mechanisms on the two branches,
        # not the same "pool" reused. This was WRONG in an earlier draft of this function (reused the
        # depthwise-conv-transpose composition on the shortcut path too) -- fixed to use plain
        # INTERPOLATE_1D (nearest, x2) here instead, matching the real module structure.
        sc = tb.node("INTERPOLATE_1D", [sc], {"ne0": "2*$n_tokens", "mode": "nearest"}, f"{out_hint}_sc_upsampled")
    if dim_in != dim_out:
        w1x1 = tb.weight(f"{prefix}.conv1x1.weight",
                          fold_weight_norm(sd[f"{sd_prefix}.conv1x1.weight_g"], sd[f"{sd_prefix}.conv1x1.weight_v"]))
        sc3 = tb.node("RESHAPE", [sc], {"shape": [seq_len_expr, dim_in, 1]}, f"{out_hint}_sc3")
        sc = tb.node("CONV_1D", [w1x1, sc3], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_sc_conv")
        sc = tb.node("RESHAPE", [sc], {"shape": [seq_len_expr, dim_out]}, f"{out_hint}_sc_2d")

    # --- residual ---
    # The depthwise-ConvTranspose1d composition's real output length is exactly 2*n_tokens (matches
    # PyTorch's own ConvTranspose1d(kernel=3,stride=2,padding=1,output_padding=1)'s L_out=2*L_in formula,
    # confirmed by direct derivation earlier this milestone) -- "2*$n_tokens" is an ordinary SymbolEnv
    # arithmetic expression, same convention as e.g. n_pos_expr's "2*(...)-1" elsewhere in this project.
    seq_len_expr = "2*$n_tokens" if upsample else "$n_tokens"

    r = add_adain1d(tb, x, style_name, f"{prefix}.norm1", sd, f"{sd_prefix}.norm1", dim_in, style_dim, eps,
                     f"{out_hint}_norm1")
    r = tb.node("LEAKY_RELU", [r], {"slope": leaky_slope}, f"{out_hint}_act1")
    if upsample:
        r = add_depthwise_conv_transpose_upsample(tb, r, f"{prefix}.pool", sd, f"{sd_prefix}.pool", dim_in,
                                                   f"{out_hint}_pool")
    conv1_w = tb.weight(f"{prefix}.conv1.weight", fold_weight_norm(sd[f"{sd_prefix}.conv1.weight_g"], sd[f"{sd_prefix}.conv1.weight_v"]))
    conv1_b = tb.weight(f"{prefix}.conv1.bias", to_f32(sd[f"{sd_prefix}.conv1.bias"]))
    r3 = tb.node("RESHAPE", [r], {"shape": [seq_len_expr, dim_in, 1]}, f"{out_hint}_r3")
    r = tb.node("CONV_1D", [conv1_w, r3], {"s0": 1, "p0": 1, "d0": 1}, f"{out_hint}_conv1_raw")
    r = tb.node("ADD", [r, tb.node("RESHAPE", [conv1_b], {"shape": [1, dim_out, 1]}, f"{out_hint}_conv1_bias_r")],
                None, f"{out_hint}_conv1_biased")
    r = tb.node("RESHAPE", [r], {"shape": [seq_len_expr, dim_out]}, f"{out_hint}_conv1_2d")

    r = add_adain1d(tb, r, style_name, f"{prefix}.norm2", sd, f"{sd_prefix}.norm2", dim_out, style_dim, eps,
                     f"{out_hint}_norm2")
    r = tb.node("LEAKY_RELU", [r], {"slope": leaky_slope}, f"{out_hint}_act2")
    conv2_w = tb.weight(f"{prefix}.conv2.weight", fold_weight_norm(sd[f"{sd_prefix}.conv2.weight_g"], sd[f"{sd_prefix}.conv2.weight_v"]))
    conv2_b = tb.weight(f"{prefix}.conv2.bias", to_f32(sd[f"{sd_prefix}.conv2.bias"]))
    r3b = tb.node("RESHAPE", [r], {"shape": [seq_len_expr, dim_out, 1]}, f"{out_hint}_r3b")
    r = tb.node("CONV_1D", [conv2_w, r3b], {"s0": 1, "p0": 1, "d0": 1}, f"{out_hint}_conv2_raw")
    r = tb.node("ADD", [r, tb.node("RESHAPE", [conv2_b], {"shape": [1, dim_out, 1]}, f"{out_hint}_conv2_bias_r")],
                None, f"{out_hint}_conv2_biased")
    r = tb.node("RESHAPE", [r], {"shape": [seq_len_expr, dim_out]}, f"{out_hint}_conv2_2d")

    summed = tb.node("ADD", [r, sc], None, f"{out_hint}_sum")
    return tb.node("SCALE", [summed], {"s": float(1.0 / np.sqrt(2.0))}, out_hint)


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["predictor"]
    hp = HP

    def write_gguf(path, topology, weights):
        w = GGUFWriter(str(path), "loom-kokoro-f0n")
        w.add_string("model.graph_topology", json.dumps(topology))
        for name, arr in weights.items():
            w.add_tensor(name, arr.astype(np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print(f"wrote {path}, {len(weights)} weights")

    # --- F0.0 alone: the simplest AdainResBlk1d instance (no shortcut conv, no upsample) ---
    tb = TopologyBuilder()
    out = add_adain_resblk1d(tb, "x", "style", "f0_0", sd, "module.F0.0", 512, 512, hp["style_dim"],
                              hp["ln_eps"], hp["leaky_slope"], upsample=False, out_hint="out")
    inputs = [
        {"name": "x", "dtype": "f32", "shape": ["$n_tokens", "512"]},
        {"name": "style", "dtype": "f32", "shape": [str(hp["style_dim"])]},
    ]
    write_gguf(out_dir / "kokoro_f0_block0.gguf", tb.topology(inputs, out), tb.weights)

    # --- F0.1: dim_in=512 -> dim_out=256, WITH learned conv1x1 shortcut AND upsample (the "pool"
    #     depthwise-ConvTranspose1d in the residual path + plain nearest INTERPOLATE_1D in the
    #     shortcut path -- both verified individually, first combined together here). ---
    tb = TopologyBuilder()
    out = add_adain_resblk1d(tb, "x", "style", "f0_1", sd, "module.F0.1", 512, 256, hp["style_dim"],
                              hp["ln_eps"], hp["leaky_slope"], upsample=True, out_hint="out")
    inputs = [
        {"name": "x", "dtype": "f32", "shape": ["$n_tokens", "512"]},
        {"name": "style", "dtype": "f32", "shape": [str(hp["style_dim"])]},
    ]
    write_gguf(out_dir / "kokoro_f0_block1.gguf", tb.topology(inputs, out), tb.weights)

    # --- Full F0Ntrain assembly: shared BiLSTM (640->512) -> two independent 3-block AdainResBlk1d
    #     stacks (F0: 512->512->256->256, N: same shape) -> F0_proj/N_proj (plain Conv1d(256,1,
    #     kernel=1), consumed as a 1x1-conv-as-matmul, same "squeeze the trailing K=1 dim" precedent as
    #     VITS's own conv1x1-as-matmul weight sites). AdainResBlk1d itself has NO recurrence at all
    #     (pure per-position conv/norm/activation), so each block is ONE ordinary graph call -- only the
    #     `shared` BiLSTM needs `loom::BiLstmStepper`'s host-stepping. ---
    write_bilstm_ggufs(out_dir, "kokoro_f0n_shared_lstm", "module.shared", sd, 256, 512 + hp["style_dim"])

    block_dims = [(512, 512, False), (512, 256, True), (256, 256, False)]
    for i, (topo, weights) in build_stack(sd, "kokoro_f0n_f0", ["module.F0.0", "module.F0.1", "module.F0.2"],
                                           block_dims, hp).items():
        write_gguf(out_dir / f"kokoro_f0n_f0_block{i}.gguf", topo, weights)
    for i, (topo, weights) in build_stack(sd, "kokoro_f0n_n", ["module.N.0", "module.N.1", "module.N.2"],
                                           block_dims, hp).items():
        write_gguf(out_dir / f"kokoro_f0n_n_block{i}.gguf", topo, weights)

    topo, weights = build_proj1x1(sd, "module.F0_proj")
    write_gguf(out_dir / "kokoro_f0n_f0_proj.gguf", topo, weights)
    topo, weights = build_proj1x1(sd, "module.N_proj")
    write_gguf(out_dir / "kokoro_f0n_n_proj.gguf", topo, weights)


if __name__ == "__main__":
    main()
