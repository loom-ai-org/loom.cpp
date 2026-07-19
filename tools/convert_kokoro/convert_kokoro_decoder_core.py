"""Assembles Kokoro's `Decoder` "core" (istftnet.py's `Decoder.forward`, everything EXCEPT the final
`self.generator(...)` call -- the Generator is already its own separately-verified topology,
convert_kokoro_generator.py, run next by the host driver, same "compose already-verified pieces via the
host driver" pattern as `BiLstmStepper`/F0Ntrain's own block-chaining).

Real channel/shape bookkeeping (confirmed directly against the checkpoint's `decoder.{encode,decode,
F0_conv,N_conv,asr_res}.*` state dict before writing anything -- see BACKLOG.md): `Decoder.__init__` is
called with `dim_in=config['hidden_dim']=512` (NOT `config['dim_in']=64`, a different, unrelated
hyperparameter -- confirmed directly in `model.py`'s `KModel.__init__`). Letting `T_frames` = this
topology's own "$n_tokens" (the original text/duration-alignment frame count, matching `asr`'s own
length):
  - `F0_conv`/`N_conv`: weight-normed `Conv1d(1,1,kernel=3,stride=2,padding=1)`, applied to `F0_curve`/`N`
    at F0Ntrain's own OUTPUT length (`2*T_frames`, F0Ntrain upsamples once) -> downsamples back to
    `T_frames` (`floor((2*T_frames+2-3)/2)+1 = T_frames` exactly, verified algebraically: `floor((2T-1)/2)
    = T-1` for any integer `T>=1`).
  - `encode = AdainResBlk1d(512+2=514, 1024, style_dim)` on `cat([asr,F0,N])` -- reuses
    `convert_kokoro_f0n.py`'s existing `add_adain_resblk1d` VERBATIM (same class used for
    `predictor.F0/N`), no upsample, no new dims logic needed.
  - `decode[0..2] = AdainResBlk1d(1024+2+64=1090, 1024, style_dim)`, each re-concatenating
    `[x,asr_res,F0,N]` (`asr_res`: a separate `Conv1d(512,64,kernel=1)` downsample of `asr`, computed
    once, threaded into every decode block).
  - `decode[3] = AdainResBlk1d(1090, 512, style_dim, upsample=True)` -- upsamples to `2*T_frames`,
    matching the Generator's own expected input length exactly.
  `add_adain_resblk1d`'s existing hardcoded `"$n_tokens"`/`"2*$n_tokens"` length expressions (from
  F0Ntrain) are directly correct here UNCHANGED (no `seq_len_expr` generalization needed, unlike the
  Generator's resblocks) because this topology's own primary `$n_tokens` symbol IS `T_frames` -- every
  non-upsampling block really does operate at exactly `$n_tokens`.

No new primitive needed (reuses `CONCAT`, `AdainResBlk1d`'s composition, ordinary `CONV_1D`).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

from convert_kokoro_f0n import add_adain_resblk1d, fold_weight_norm, to_f32

HP = {
    "style_dim": 128,
    "ln_eps": 1e-5,
    "leaky_slope": 0.2,
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


def add_strided_conv1x1ch(tb, x, prefix, sd, sd_prefix, kernel_size, stride, padding, seq_len_expr, out_hint):
    """x: [T_in,1] (a single-channel curve, e.g. F0_curve/N unsqueezed). Real weight-normed
    Conv1d(1,1,kernel_size,stride,padding). Output length left to ggml's own CONV_1D formula (not
    pre-declared -- verified algebraically to be exactly seq_len_expr's "$n_tokens" at the real call
    site, see module docstring)."""
    w = tb.weight(f"{prefix}.weight", fold_weight_norm(sd[f"{sd_prefix}.weight_g"], sd[f"{sd_prefix}.weight_v"]))
    b = tb.weight(f"{prefix}.bias", to_f32(sd[f"{sd_prefix}.bias"]))
    x3 = tb.node("RESHAPE", [x], {"shape": [seq_len_expr, 1, 1]}, f"{out_hint}_x3")
    conv = tb.node("CONV_1D", [w, x3], {"s0": stride, "p0": padding, "d0": 1}, f"{out_hint}_raw")
    biased = tb.node("ADD", [conv, tb.node("RESHAPE", [b], {"shape": [1, 1, 1]}, f"{out_hint}_b_r")],
                      None, f"{out_hint}_biased")
    return tb.node("RESHAPE", [biased], {"shape": [-1, 1]}, out_hint)


def add_conv1x1(tb, x, prefix, sd, sd_prefix, ch_in, ch_out, seq_len_expr, out_hint):
    """x: [T,ch_in]. Real weight-normed Conv1d(ch_in,ch_out,kernel=1) (asr_res)."""
    w = tb.weight(f"{prefix}.weight", fold_weight_norm(sd[f"{sd_prefix}.weight_g"], sd[f"{sd_prefix}.weight_v"]))
    b = tb.weight(f"{prefix}.bias", to_f32(sd[f"{sd_prefix}.bias"]))
    x3 = tb.node("RESHAPE", [x], {"shape": [seq_len_expr, ch_in, 1]}, f"{out_hint}_x3")
    conv = tb.node("CONV_1D", [w, x3], {"s0": 1, "p0": 0, "d0": 1}, f"{out_hint}_raw")
    biased = tb.node("ADD", [conv, tb.node("RESHAPE", [b], {"shape": [1, ch_out, 1]}, f"{out_hint}_b_r")],
                      None, f"{out_hint}_biased")
    return tb.node("RESHAPE", [biased], {"shape": [seq_len_expr, ch_out]}, out_hint)


def build_decoder_core(hp, sd, sd_prefix=""):
    """Inputs: "asr" [$n_tokens,512], "f0_curve" [2*$n_tokens,1], "n_curve" [2*$n_tokens,1], "style"
    [style_dim]. Output: "x" [2*$n_tokens,512] (fed to the Generator topology next). `sd_prefix` lets
    this be pointed at the real checkpoint's own "module." nesting (sd_all["decoder"]'s keys are all
    "module.encode.*"/"module.decode.*"/etc.) without changing any internal key strings."""
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    tb = TopologyBuilder()
    style_dim = hp["style_dim"]
    eps = hp["ln_eps"]
    leaky = hp["leaky_slope"]

    f0_2t_expr = "2*$n_tokens"
    F0 = add_strided_conv1x1ch(tb, "f0_curve", "F0_conv", sd, p("F0_conv"), 3, 2, 1, f0_2t_expr, "F0")
    N = add_strided_conv1x1ch(tb, "n_curve", "N_conv", sd, p("N_conv"), 3, 2, 1, f0_2t_expr, "N")

    x = tb.node("CONCAT", ["asr", F0], {"dim": 1}, "asr_f0")
    x = tb.node("CONCAT", [x, N], {"dim": 1}, "asr_f0_n")
    x = add_adain_resblk1d(tb, x, "style", "encode", sd, p("encode"), 514, 1024, style_dim, eps, leaky,
                            upsample=False, out_hint="encoded")

    asr_res = add_conv1x1(tb, "asr", "asr_res", sd, p("asr_res.0"), 512, 64, "$n_tokens", "asr_res")

    decode_dims = [(1090, 1024, False), (1090, 1024, False), (1090, 1024, False), (1090, 512, True)]
    res = True
    for i, (dim_in, dim_out, upsample) in enumerate(decode_dims):
        if res:
            x = tb.node("CONCAT", [x, asr_res], {"dim": 1}, f"decode{i}_cat1")
            x = tb.node("CONCAT", [x, F0], {"dim": 1}, f"decode{i}_cat2")
            x = tb.node("CONCAT", [x, N], {"dim": 1}, f"decode{i}_cat3")
        x = add_adain_resblk1d(tb, x, "style", f"decode.{i}", sd, p(f"decode.{i}"), dim_in, dim_out, style_dim,
                                eps, leaky, upsample=upsample, out_hint=f"decode{i}_out")
        if upsample:
            res = False

    inputs = [
        {"name": "asr", "dtype": "f32", "shape": ["$n_tokens", "512"]},
        {"name": "f0_curve", "dtype": "f32", "shape": [f0_2t_expr, "1"]},
        {"name": "n_curve", "dtype": "f32", "shape": [f0_2t_expr, "1"]},
        {"name": "style", "dtype": "f32", "shape": [str(style_dim)]},
    ]
    return tb.topology(inputs, x), tb.weights


def make_synthetic_state_dict(rng, hp):
    """Real checkpoint weights get wired in when the overall Kokoro conversion script is assembled --
    this standalone converter verifies the WIRING with synthetic weights of the REAL shapes (confirmed
    against the checkpoint directly, see module docstring)."""
    sd = {}
    style_dim = hp["style_dim"]

    def randn(shape, scale=0.2):
        return torch.from_numpy(rng.normal(scale=scale, size=shape).astype(np.float32))

    def add_resblk_weights(prefix, dim_in, dim_out, learned_sc, upsample):
        sd[f"{prefix}.norm1.fc.weight"] = randn((2 * dim_in, style_dim))
        sd[f"{prefix}.norm1.fc.bias"] = randn((2 * dim_in,), 0.1)
        sd[f"{prefix}.norm2.fc.weight"] = randn((2 * dim_out, style_dim))
        sd[f"{prefix}.norm2.fc.bias"] = randn((2 * dim_out,), 0.1)
        sd[f"{prefix}.conv1.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(dim_out, 1, 1)).astype(np.float32))
        sd[f"{prefix}.conv1.weight_v"] = randn((dim_out, dim_in, 3))
        sd[f"{prefix}.conv1.bias"] = randn((dim_out,), 0.1)
        sd[f"{prefix}.conv2.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(dim_out, 1, 1)).astype(np.float32))
        sd[f"{prefix}.conv2.weight_v"] = randn((dim_out, dim_out, 3))
        sd[f"{prefix}.conv2.bias"] = randn((dim_out,), 0.1)
        if learned_sc:
            sd[f"{prefix}.conv1x1.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(dim_out, 1, 1)).astype(np.float32))
            sd[f"{prefix}.conv1x1.weight_v"] = randn((dim_out, dim_in, 1))
        if upsample:
            sd[f"{prefix}.pool.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(dim_in, 1, 1)).astype(np.float32))
            sd[f"{prefix}.pool.weight_v"] = randn((dim_in, 1, 3))
            sd[f"{prefix}.pool.bias"] = randn((dim_in,), 0.1)

    for name, k in (("F0_conv", 3), ("N_conv", 3)):
        sd[f"{name}.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(1, 1, 1)).astype(np.float32))
        sd[f"{name}.weight_v"] = randn((1, 1, k))
        sd[f"{name}.bias"] = randn((1,), 0.1)

    sd["asr_res.0.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(64, 1, 1)).astype(np.float32))
    sd["asr_res.0.weight_v"] = randn((64, 512, 1))
    sd["asr_res.0.bias"] = randn((64,), 0.1)

    add_resblk_weights("encode", 514, 1024, True, False)
    decode_dims = [(1090, 1024, False), (1090, 1024, False), (1090, 1024, False), (1090, 512, True)]
    for i, (dim_in, dim_out, upsample) in enumerate(decode_dims):
        add_resblk_weights(f"decode.{i}", dim_in, dim_out, dim_in != dim_out, upsample)
    return sd


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <out_dir>", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = HP

    rng = np.random.RandomState(51)
    sd = make_synthetic_state_dict(rng, hp)
    np.savez(out_dir / "kokoro_decoder_core_sd.npz", **{k: v.numpy() for k, v in sd.items()})

    topo, weights = build_decoder_core(hp, sd)
    w = GGUFWriter(str(out_dir / "kokoro_decoder_core.gguf"), "loom-kokoro-decoder-core")
    w.add_string("model.graph_topology", json.dumps(topo))
    for name, arr in weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'kokoro_decoder_core.gguf'}, {len(weights)} weights")


if __name__ == "__main__":
    main()
