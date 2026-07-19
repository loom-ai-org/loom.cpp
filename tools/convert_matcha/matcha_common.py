"""Shared helpers for converting the real Matcha-TTS checkpoint into loom-engine GGUF files.

Mirrors `convert_piper_vits/vits_common.py`'s `TopologyBuilder` idiom exactly. Real checkpoint:
`/home/flavio/.claude/tmp/matcha_model/ckpt/matcha_ljspeech.ckpt` (LJSpeech, single-speaker,
n_spks=1 -- no speaker embedding table exists, `spks` stays unused throughout, same simplification
as VITS's own piper checkpoint).
"""
import json

import numpy as np
import torch
from gguf import GGUFWriter


def load_matcha_checkpoint(ckpt_path):
    """Loads the real Lightning `.ckpt`. Tensor names are NOT prefixed (unlike piper's `model_g.`) --
    `state_dict` keys are already `encoder.*`/`decoder.*` directly.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def to_f32(tensor):
    arr = tensor.detach().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    return arr.astype(np.float32)


def load_hifigan_checkpoint(ckpt_path):
    """The real HiFi-GAN vocoder checkpoint (`generator_v1`) stores the generator's own state dict
    under a top-level `"generator"` key (confirmed directly: `torch.load(...).keys() ==
    ['generator']`), unlike the Matcha checkpoint's own `state_dict`/`hyper_parameters` layout.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["generator"]


def fold_weight_norm(weight_g, weight_v):
    """Folds a PyTorch `weight_norm`-reparametrized (`weight_g`,`weight_v`) pair into a plain weight
    tensor: `w = g*v/||v||`, norm over every dim except dim 0 -- same formula/precedent already
    verified for VITS's own HiFi-GAN vocoder (`convert_piper_vits/vits_common.py`).
    """
    g = weight_g.detach().numpy() if torch.is_tensor(weight_g) else np.asarray(weight_g)
    v = weight_v.detach().numpy() if torch.is_tensor(weight_v) else np.asarray(weight_v)
    norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape([-1] + [1] * (v.ndim - 1))
    return (g * v / norm).astype(np.float32)


class TopologyBuilder:
    def __init__(self):
        self.nodes = []
        self.weights = {}
        self.int32_weights = set()
        self._counter = 0

    def _fresh(self, hint):
        self._counter += 1
        return f"{hint}_{self._counter}"

    def node(self, op, inputs, attrs=None, out_hint="t"):
        out = self._fresh(out_hint)
        entry = {"op": op, "inputs": list(inputs), "outputs": [out]}
        if attrs:
            entry["attrs"] = attrs
        self.nodes.append(entry)
        return out

    def weight(self, name, array, is_int32=False):
        if name in self.weights:
            existing = self.weights[name]
            if existing.shape != np.asarray(array).shape:
                raise ValueError(f"weight {name!r} already registered with a different shape")
        else:
            self.weights[name] = np.asarray(array)
            if is_int32:
                self.int32_weights.add(name)
        return name

    def transpose_2d(self, x, out_hint="t2d"):
        p = self.node("PERMUTE", [x], {"axes": [1, 0, 2, 3]}, out_hint + "_p")
        return self.node("CONT", [p], None, out_hint)

    def topology(self, inputs, output):
        return {"version": 1, "inputs": inputs, "output": output, "nodes": self.nodes}


def add_conv(tb, prefix, sd, name):
    """Plain (no weight_norm) Conv1d: weight (out,in,k) + bias (out,), used verbatim via CONV_1D."""
    tb.weight(f"{prefix}.weight", to_f32(sd[f"{name}.weight"]))
    tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"]))
    return f"{prefix}.weight", f"{prefix}.bias"


def add_wn_conv(tb, prefix, sd, name):
    """Registers a weight_norm'd conv's FOLDED weight (+ plain bias) -- same as `vits_common.py`'s
    own helper, used by the real HiFi-GAN vocoder's `conv_pre`/`conv_post`/`ups`/`resblocks` convs.
    """
    folded = fold_weight_norm(sd[f"{name}.weight_g"], sd[f"{name}.weight_v"])
    tb.weight(f"{prefix}.weight", folded)
    tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"]))
    return f"{prefix}.weight", f"{prefix}.bias"


def add_conv1x1_as_matmul(tb, prefix, sd, name):
    """A kernel_size=1 Conv1d used via MUL_MAT instead of CONV_1D -- squeeze the trailing K=1 dim
    (same convention/reasoning as vits_common.py's own helper of the same name: GGUF's axis-reversal
    means an unsqueezed (out,in,1) array loads back as ne=[1,in,out], contracting MUL_MAT against the
    wrong axis).
    """
    w = to_f32(sd[f"{name}.weight"])
    if w.ndim == 3:
        assert w.shape[-1] == 1, f"{name}.weight: expected a squeezable kernel_size=1 conv, got {w.shape}"
        w = w.reshape(w.shape[0], w.shape[1])
    tb.weight(f"{prefix}.weight", w)
    tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"]))
    return f"{prefix}.weight", f"{prefix}.bias"


def add_linear_no_bias(tb, prefix, sd, name):
    """A real bias-free `nn.Linear` (`bias=False`) -- `diffusers`' own `Attention` class default for
    its `to_q`/`to_k`/`to_v` projections (confirmed: only `to_out.0.{weight,bias}` has a bias in the
    real checkpoint).
    """
    w = tb.weight(f"{prefix}.weight", to_f32(sd[f"{name}.weight"]))
    return w


def build_group_norm(tb, x_tc, prefix, sd, name, channels, n_groups, eps, out_hint="gn"):
    """`nn.GroupNorm(n_groups, channels)` applied to `x` in [T,C] (CONV_1D/conv) convention --
    reshape to [T,1,C,1] (moving C to ne[2], matching GROUP_NORM's own native-`ggml_group_norm`
    convention), normalize, reshape back, then the learned per-channel affine (separate MUL/ADD, same
    pattern as every other norm in this project since GROUP_NORM itself never applies one).
    """
    gamma = tb.weight(f"{prefix}.weight", to_f32(sd[f"{name}.weight"]))
    beta = tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"]))
    x4 = tb.node("RESHAPE", [x_tc], {"shape": [-1, 1, channels, 1]}, out_hint + "_4d")
    normed4 = tb.node("GROUP_NORM", [x4], {"n_groups": n_groups, "eps": eps}, out_hint + "_normed4d")
    normed = tb.node("RESHAPE", [normed4], {"shape": [-1, channels]}, out_hint + "_normed")
    scaled = tb.node("MUL", [normed, tb.node("RESHAPE", [gamma], {"shape": [1, channels]}, out_hint + "_g_r")],
                      None, out_hint + "_scaled")
    return tb.node("ADD", [scaled, tb.node("RESHAPE", [beta], {"shape": [1, channels]}, out_hint + "_b_r")],
                   None, out_hint)


def add_glowtts_layer_norm(tb, prefix, sd, name):
    """Matcha's own `text_encoder.py::LayerNorm` (channel-axis mean/var, eps=1e-4, learned gamma/beta)
    -- same custom-channel-axis LAYER_NORM composition (LAYER_NORM + MUL gamma + ADD beta) already used
    for VITS's own glow-tts-derived LayerNorm, just a different default eps (VITS: 1e-5, Matcha: 1e-4).
    """
    tb.weight(f"{prefix}.gamma", to_f32(sd[f"{name}.gamma"]))
    tb.weight(f"{prefix}.beta", to_f32(sd[f"{name}.beta"]))
    return f"{prefix}.gamma", f"{prefix}.beta"


def apply_glowtts_layer_norm(tb, x, gamma, beta, channels, eps, out_hint="ln"):
    """Applies channel-axis LayerNorm to `x` in [C,T] convention (C=ne[0]) -- LAYER_NORM normalizes
    over ne[0] directly (ggml_norm's own convention), matching torch.mean(x,1,keepdim=True) over the
    channel axis when x is (B,C,T) -- so no reshape/permute needed here, unlike CONV_1D-convention
    ([T,C]) callers which would need one.
    """
    normed = tb.node("LAYER_NORM", [x], {"eps": eps}, out_hint + "_n")
    scaled = tb.node("MUL", [normed, gamma], None, out_hint + "_g")
    return tb.node("ADD", [scaled, beta], None, out_hint)


def add_linear(tb, prefix, sd, name):
    """A real `nn.Linear`: weight (out,in) + bias (out,), used via MUL_MAT directly (no squeeze
    needed, unlike `add_conv1x1_as_matmul` -- Linear weights have no trailing kernel dim).
    """
    w = tb.weight(f"{prefix}.weight", to_f32(sd[f"{name}.weight"]))
    b = tb.weight(f"{prefix}.bias", to_f32(sd[f"{name}.bias"]))
    return w, b


def apply_std_layer_norm(tb, x_ct, gamma, beta, eps, out_hint="ln"):
    """Standard `nn.LayerNorm` (mean/var over the whole channel vector, matching torch's own
    default when applied to a (...,C) tensor) applied to `x` in [C,T] (C=ne[0]) convention -- same
    LAYER_NORM(ggml_norm)+MUL+ADD composition as `apply_glowtts_layer_norm`, just a different
    (standard torch) eps default (1e-5, not Matcha's own custom glow-tts LayerNorm's 1e-4) and no
    custom mean/var formula difference (`ggml_norm` already computes plain full mean+variance over
    ne[0], identical to `torch.nn.LayerNorm`'s own reduction over its last/normalized axis once the
    tensor is in this [C,T] convention).
    """
    normed = tb.node("LAYER_NORM", [x_ct], {"eps": eps}, out_hint + "_n")
    scaled = tb.node("MUL", [normed, gamma], None, out_hint + "_g")
    return tb.node("ADD", [scaled, beta], None, out_hint)


def mish(tb, x, out_hint):
    """Mish(x) = x * tanh(softplus(x)) -- same composition already verified for SupertonicTTS's own
    Mish usage (tools/convert_supertonic/supertonic_common.py).
    """
    sp = tb.node("SOFTPLUS", [x], None, f"{out_hint}_softplus")
    t = tb.node("TANH", [sp], None, f"{out_hint}_tanh")
    return tb.node("MUL", [x, t], None, out_hint)


def write_gguf(path, architecture, hparams, topology, weights, int32_names=()):
    w = GGUFWriter(str(path), architecture)
    w.add_string("loom.architecture", architecture)
    for key, value in hparams.items():
        if isinstance(value, float):
            w.add_float32(f"loom.{key}", value)
        elif isinstance(value, int):
            w.add_uint32(f"loom.{key}", value)
        else:
            w.add_string(f"loom.{key}", str(value))
    w.add_string("model.graph_topology", json.dumps(topology))
    for name, arr in weights.items():
        if name in int32_names:
            w.add_tensor(name, arr.astype(np.int32))
        else:
            w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path} ({len(weights)} tensors)")
