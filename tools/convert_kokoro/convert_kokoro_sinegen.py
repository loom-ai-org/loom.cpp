"""Converts Kokoro Generator's NSF harmonic source (istftnet.py's `SineGen`+`SourceModuleHnNSF`,
real hyperparameters hardcoded in `Generator.__init__`: `sampling_rate=24000`,
`upsample_scale=math.prod(upsample_rates)*gen_istft_hop_size=10*6*5=300`, `harmonic_num=8`,
`voiced_threshod=10`; `SineGen`'s own defaults `sine_amp=0.1`, `noise_std=0.003`) into a single loom
topology producing `har_source` (real `Generator.forward` never uses `SourceModuleHnNSF`'s other two
return values, `noise`/`uv`, after this point -- confirmed directly in the real source, so this doesn't
compute them at all).

Algorithm (nearest-upsample F0 by `upsample_scale` -> per-harmonic phase accumulation via a
downsample(1/scale,linear)+cumsum+upsample(scale,linear) dance -> sin -> voiced/unvoiced-gated noise mix
-> Linear(dim,1)+tanh) verified in plain Python/numpy against the REAL hand-copied `SineGen`/
`SourceModuleHnNSF.forward` (with matched injected randomness) to max_diff=3.0e-8 BEFORE writing this --
see BACKLOG.md. Two host-drawn random inputs match VITS's own noise-injection precedent (host `<random>`,
fed in as declared inputs): `rand_ini` (the harmonic phase's random initial-offset draw, `dim` floats,
index 0 always exactly 0 per `SineGen._f02sine`'s own `rand_ini[:,0]=0`) and `noise` (`SourceModuleHnNSF`'s
per-sample additive Gaussian noise, `[L,dim]` floats, `L=T_frames*upsample_scale`).

No new primitive needed beyond this milestone's `FLOOR` (added for the `x % 1 = x - floor(x)` reduction,
valid since every operand here is non-negative) -- `harmonic_num+1=9` per-harmonic channels are built by
unrolling 9 `SCALE`+`RESHAPE`+`CONCAT` calls (a fixed, conversion-time-known count) rather than adding a
new outer-product/broadcast-repeat primitive just for this.
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

HP = {
    "sampling_rate": 24000.0,
    "upsample_scale": 300,
    "harmonic_num": 8,
    "voiced_threshold": 10.0,
    "sine_amp": 0.1,
    "noise_std": 0.003,
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


def build_sinegen(hp, l_linear_w, l_linear_b):
    """l_linear_w: (1, dim) numpy (real nn.Linear(dim,1) native weight shape). l_linear_b: (1,) numpy."""
    dim = hp["harmonic_num"] + 1
    scale = hp["upsample_scale"]
    tb = TopologyBuilder()

    l_expr = f"$n_tokens*{scale}"
    f0_up = tb.node("INTERPOLATE_1D", ["f0_curve"], {"ne0": l_expr, "mode": "nearest"}, "f0_up")

    # fn[:, h] = f0_up * (h+1), h=0..dim-1 -- unrolled (dim is small & conversion-time-fixed), concatenated
    # along the channel axis (dim=1) rather than adding a broadcast-repeat primitive just for this.
    fn = None
    for h in range(1, dim + 1):
        scaled_h = tb.node("SCALE", [f0_up], {"s": float(h)}, f"fn_h{h}")
        col_h = tb.node("RESHAPE", [scaled_h], {"shape": [l_expr, 1]}, f"fn_col{h}")
        fn = col_h if fn is None else tb.node("CONCAT", [fn, col_h], {"dim": 1}, f"fn_cat{h}")

    rad_scaled = tb.node("SCALE", [fn], {"s": 1.0 / hp["sampling_rate"]}, "rad_scaled")
    rad_full = tb.node("SUB", [rad_scaled, tb.node("FLOOR", [rad_scaled], None, "rad_floor")], None, "rad_full")

    rand_ini_row = tb.node("RESHAPE", ["rand_ini"], {"shape": [1, dim]}, "rand_ini_row")
    rad_offset = tb.node("PAD_1D", [rand_ini_row], {"lp0": 0, "rp0": f"{l_expr}-1"}, "rad_offset")
    rad_with_ini = tb.node("ADD", [rad_full, rad_offset], None, "rad_with_ini")

    rad_down = tb.node("INTERPOLATE_1D", [rad_with_ini], {"ne0": "$n_tokens", "mode": "linear"}, "rad_down")
    phase_low = tb.node("SCALE", [tb.node("CUMSUM", [rad_down], None, "cumsum_low")],
                         {"s": 2.0 * np.pi}, "phase_low")
    phase_pre = tb.node("SCALE", [phase_low], {"s": float(scale)}, "phase_pre")
    phase_full = tb.node("INTERPOLATE_1D", [phase_pre], {"ne0": l_expr, "mode": "linear"}, "phase_full")
    sine_waves = tb.node("SCALE", [tb.node("SIN", [phase_full], None, "sines")], {"s": hp["sine_amp"]}, "sine_waves")

    f0_up_col = tb.node("RESHAPE", [f0_up], {"shape": [l_expr, 1]}, "f0_up_col")
    threshold_w = tb.weight("sinegen.threshold", np.array([[hp["voiced_threshold"]]], dtype=np.float32))
    uv = tb.node("STEP", [tb.node("SUB", [f0_up_col, threshold_w], None, "f0_minus_thresh")], None, "uv")

    base_w = tb.weight("sinegen.noise_amp_base", np.array([[hp["sine_amp"] / 3.0]], dtype=np.float32))
    delta_w = tb.weight("sinegen.noise_amp_delta",
                         np.array([[hp["noise_std"] - hp["sine_amp"] / 3.0]], dtype=np.float32))
    # ggml_add(a,b) requires b to broadcast INTO a's shape (a determines the output shape) -- uv_delta
    # ([L,1]) must be the FIRST arg here, base_w ([1,1]) the second, not the other way around (a real bug
    # caught by ggml's own ggml_can_repeat assertion firing when this was reversed).
    noise_amp = tb.node("ADD", [tb.node("MUL", [uv, delta_w], None, "uv_delta"), base_w], None, "noise_amp")
    noise_scaled = tb.node("MUL", ["noise", noise_amp], None, "noise_scaled")

    sine_waves_final = tb.node("ADD", [tb.node("MUL", [sine_waves, uv], None, "sine_gated"), noise_scaled],
                                None, "sine_waves_final")

    # Cross the [T,C]->[C,T] convention boundary (CUMSUM/INTERPOLATE_1D need time on ne[0]; MUL_MAT needs
    # the contracted (channel) axis on ne[0]) via PERMUTE+CONT, same precedent as every other genuine
    # axis-convention crossing in this project.
    sine_t = tb.node("CONT", [tb.node("PERMUTE", [sine_waves_final], {"axes": [1, 0, 2, 3]}, "sine_perm")],
                      None, "sine_t")
    w = tb.weight("sinegen.l_linear.weight", l_linear_w)
    b = tb.weight("sinegen.l_linear.bias", l_linear_b)
    har_pre = tb.node("ADD", [tb.node("MUL_MAT", [w, sine_t], None, "har_mm"),
                              tb.node("RESHAPE", [b], {"shape": [1, 1]}, "har_bias_r")], None, "har_pre")
    har_source = tb.node("TANH", [har_pre], None, "har_pre_tanh")
    out = tb.node("RESHAPE", [har_source], {"shape": [l_expr]}, "har_source")

    inputs = [
        {"name": "f0_curve", "dtype": "f32", "shape": ["$n_tokens"]},
        {"name": "rand_ini", "dtype": "f32", "shape": [str(dim)]},
        {"name": "noise", "dtype": "f32", "shape": [l_expr, str(dim)]},
    ]
    return tb.topology(inputs, out), tb.weights


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <out_dir>", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    # l_linear is part of the real checkpoint's `decoder.generator.m_source.l_linear.*` -- pulled from a
    # real checkpoint by the (still-to-come) overall Kokoro conversion script; this standalone converter
    # takes them as an argument-free placeholder call sized from HP for now (this piece is pure math, no
    # checkpoint dependency at all until wired into the full Decoder assembly), matching the STFT
    # converter's own "no real weights involved" structure -- swap in real weights when Decoder assembly
    # (task #90) wires this into the real checkpoint.
    rng = np.random.RandomState(11)
    dim = HP["harmonic_num"] + 1
    l_linear_w = (rng.normal(scale=0.3, size=(1, dim))).astype(np.float32)
    l_linear_b = (rng.normal(scale=0.1, size=(1,))).astype(np.float32)

    topo, weights = build_sinegen(HP, l_linear_w, l_linear_b)
    w = GGUFWriter(str(out_dir / "kokoro_sinegen.gguf"), "loom-kokoro-sinegen")
    w.add_string("model.graph_topology", json.dumps(topo))
    for name, arr in weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'kokoro_sinegen.gguf'}, {len(weights)} weights")

    np.save(out_dir / "kokoro_sinegen_l_linear_w.npy", l_linear_w)
    np.save(out_dir / "kokoro_sinegen_l_linear_b.npy", l_linear_b)


if __name__ == "__main__":
    main()
