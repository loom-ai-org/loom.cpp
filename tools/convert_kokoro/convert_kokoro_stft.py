"""Builds Kokoro Generator's forward/inverse STFT as loom topologies (istftnet.py's TorchSTFT, real
config gen_istft_n_fft=20, gen_istft_hop_size=5) -- see kokoro_stft_common.py's module docstring for the
full derivation/verification notes (ATAN2 boundary-bin fix, CONV_TRANSPOSE_1D-based ISTFT, host-computed
wsum). No real checkpoint weights involved at all (every tensor here is a conversion-time-baked constant
DFT/window kernel), so this can be run standalone.

Usage: python3 convert_kokoro_stft.py <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

from kokoro_stft_common import N_FFT, HOP_LENGTH, build_forward_dft_kernels, build_inverse_synth_kernels


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


def write_gguf(path, topology, weights):
    w = GGUFWriter(str(path), "loom-kokoro-stft")
    w.add_string("model.graph_topology", json.dumps(topology))
    for name, arr in weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {path}, {len(weights)} weights")


def build_forward(n_fft, hop):
    """Input: "waveform_padded" [n_samples_padded] (host reflect-padded by n_fft//2 each side already).
    Output: "har" [n_frames, 2*n_freq] -- magnitude then phase concatenated along channels (dim=1 in
    PyTorch's (B,C,T) convention), matching istftnet.py's own
    `har = torch.cat([har_spec, har_phase], dim=1)` directly."""
    n_freq = n_fft // 2 + 1
    cos_kernel, neg_sin_kernel, boundary_mask = build_forward_dft_kernels(n_fft)

    tb = TopologyBuilder()
    cos_w = tb.weight("stft.cos_kernel", cos_kernel)
    neg_sin_w = tb.weight("stft.neg_sin_kernel", neg_sin_kernel)
    mask_w = tb.weight("stft.boundary_mask", boundary_mask)

    # SymbolEnv's int-attr resolution rounds-to-nearest (std::llround), NOT floor/truncate -- plain "/"
    # would give e.g. (84-20)/5+1=13.8, which llround's to 14 (WRONG, real conv output has 13 frames).
    # floor() is explicitly supported in the grammar for exactly this reason.
    n_frames_expr = f"floor(($n_tokens-{n_fft})/{hop})+1"
    x3 = tb.node("RESHAPE", ["waveform_padded"], {"shape": ["$n_tokens", 1, 1]}, "x3")
    re = tb.node("CONV_1D", [cos_w, x3], {"s0": hop, "p0": 0, "d0": 1}, "re_raw")
    re = tb.node("RESHAPE", [re], {"shape": [n_frames_expr, n_freq]}, "re")
    im_raw = tb.node("CONV_1D", [neg_sin_w, x3], {"s0": hop, "p0": 0, "d0": 1}, "im_raw3")
    im_raw = tb.node("RESHAPE", [im_raw], {"shape": [n_frames_expr, n_freq]}, "im_raw")
    im_bad = tb.node("MUL", [im_raw, mask_w], None, "im_bad")
    im = tb.node("SUB", [im_raw, im_bad], None, "im")

    mag = tb.node("SQRT", [tb.node("ADD", [tb.node("SQR", [re], None, "re2"),
                                            tb.node("SQR", [im], None, "im2")], None, "sumsq")], None, "mag")
    phase = tb.node("ATAN2", [im, re], None, "phase")
    out = tb.node("CONCAT", [mag, phase], {"dim": 1}, "har")

    inputs = [{"name": "waveform_padded", "dtype": "f32", "shape": ["$n_tokens"]}]
    return tb.topology(inputs, out), tb.weights


def build_inverse(n_fft, hop):
    """Inputs: "magnitude"/"phase" [n_frames, n_freq], "wsum" [(n_frames-1)*hop+n_fft] (host-precomputed,
    see kokoro_stft_common.compute_wsum). Output: "waveform" [(n_frames-1)*hop+n_fft-2*(n_fft//2)]
    (center-cropped, matching torch.istft(center=True)'s own default)."""
    n_freq = n_fft // 2 + 1
    pad = n_fft // 2
    cos_synth, neg_sin_synth = build_inverse_synth_kernels(n_fft)

    tb = TopologyBuilder()
    cos_w = tb.weight("stft.cos_synth", cos_synth)
    neg_sin_w = tb.weight("stft.neg_sin_synth", neg_sin_synth)

    re = tb.node("MUL", ["magnitude", tb.node("COS", ["phase"], None, "cosph")], None, "re")
    im = tb.node("MUL", ["magnitude", tb.node("SIN", ["phase"], None, "sinph")], None, "im")

    out_len_full_expr = f"($n_tokens-1)*{hop}+{n_fft}"
    re_contrib = tb.node("CONV_TRANSPOSE_1D", [cos_w, re], {"s0": hop}, "re_contrib")
    im_contrib = tb.node("CONV_TRANSPOSE_1D", [neg_sin_w, im], {"s0": hop}, "im_contrib")
    numerator = tb.node("ADD", [re_contrib, im_contrib], None, "numerator")
    numerator_1d = tb.node("RESHAPE", [numerator], {"shape": [out_len_full_expr]}, "numerator_1d")
    normalized = tb.node("DIV", [numerator_1d, "wsum"], None, "normalized")
    cropped_len_expr = f"{out_len_full_expr}-{2 * pad}"
    out = tb.node("VIEW", [normalized], {"shape": [cropped_len_expr], "offset": pad * 4}, "waveform")

    inputs = [
        {"name": "magnitude", "dtype": "f32", "shape": ["$n_tokens", str(n_freq)]},
        {"name": "phase", "dtype": "f32", "shape": ["$n_tokens", str(n_freq)]},
        {"name": "wsum", "dtype": "f32", "shape": [out_len_full_expr]},
    ]
    return tb.topology(inputs, out), tb.weights


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <out_dir>", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    fwd_topo, fwd_weights = build_forward(N_FFT, HOP_LENGTH)
    write_gguf(out_dir / "kokoro_stft_forward.gguf", fwd_topo, fwd_weights)

    inv_topo, inv_weights = build_inverse(N_FFT, HOP_LENGTH)
    write_gguf(out_dir / "kokoro_stft_inverse.gguf", inv_topo, inv_weights)


if __name__ == "__main__":
    main()
