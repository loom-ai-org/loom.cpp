"""Builds a single standalone `AdaINResBlock1` GGUF topology (istftnet.py's Generator resblock, distinct
from `predictor.F0/N`'s `AdainResBlk1d`) from the synthetic state dict
`reference_forward_kokoro_adainresblock1.py` produces -- verifies the WIRING against a hand-rolled
PyTorch reference (tests/test_e2e_kokoro_adainresblock1.cpp), same "checkpoint-independent structural
verification first" precedent as VITS's own test_hifigan_generator. Real checkpoint weights get wired in
when the full Generator is assembled (task #89).

Usage: python3 convert_kokoro_adainresblock1.py <ref_dir> <out_dir>
(ref_dir must already contain adainresblock1_sd.npz, produced by the reference script above.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

from kokoro_generator_common import add_adain_resblock1

CHANNELS = 4
STYLE_DIM = 8
KERNEL_SIZE = 3
DILATIONS = (1, 3, 5)
EPS = 1e-5


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


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <ref_dir> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ref_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(ref_dir / "adainresblock1_sd.npz")
    sd = {k: torch.from_numpy(npz[k]) for k in npz.files}

    tb = TopologyBuilder()
    out = add_adain_resblock1(tb, "x", "style", "resblock", sd, "", CHANNELS, STYLE_DIM, EPS,
                               KERNEL_SIZE, DILATIONS, out_hint="out")
    inputs = [
        {"name": "x", "dtype": "f32", "shape": ["$n_tokens", str(CHANNELS)]},
        {"name": "style", "dtype": "f32", "shape": [str(STYLE_DIM)]},
    ]
    topo = tb.topology(inputs, out)

    w = GGUFWriter(str(out_dir / "kokoro_adainresblock1.gguf"), "loom-kokoro-adainresblock1")
    w.add_string("model.graph_topology", json.dumps(topo))
    for name, arr in tb.weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'kokoro_adainresblock1.gguf'}, {len(tb.weights)} weights")


if __name__ == "__main__":
    main()
