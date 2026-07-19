"""Converts Kokoro's `bert_encoder` (`model.py`'s `KModel.__init__`: `self.bert_encoder =
torch.nn.Linear(self.bert.config.hidden_size, config['hidden_dim'])`, i.e. plain `Linear(768,512)`)
into its own tiny topology. Real call site: `d_en = self.bert_encoder(bert_dur).transpose(-1,-2)`.

Axis convention (confirmed directly against `test_e2e_kokoro_albert.cpp`'s own comment before assuming
anything): `CustomAlbert`'s real output is TIME-MAJOR/channel-LAST (`ggml ne=[768,T]`, 768=ne[0] --
"native PyTorch, byte-identical to ggml ne=[hidden_size,T]", the NATURAL transformer `(B,T,H)` layout,
reversed by ggml -- a genuinely DIFFERENT convention from `CONV_1D`-family components' own `[T,C]`
convention (`ne=[T,channels]`, channel-first `(B,C,T)` reversed)). So: `x` input here is `ne=[768,T]`
directly (no transpose needed on the way in, matching Albert's own raw output byte-for-byte) ->
`MUL_MAT` naturally produces `ne=[512,T]` (still time-major, matching real `bert_encoder(bert_dur)`
BEFORE its own `.transpose(-1,-2)`) -> the real `.transpose(-1,-2)` is then an EXPLICIT
`PERMUTE`+`CONT` crossing into `DurationEncoder`'s own channel-first convention (`ne=[T,512]`, matching
`test_e2e_kokoro_duration_predictor.cpp`'s own `d_en` convention directly) -- the standard "explicit
transpose at a real axis-convention crossing" precedent used throughout this project, not assumed away.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter


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


def build_bert_encoder(tb, sd, sd_prefix=""):
    """x: [768,$n_tokens] (ne[0]=768, CustomAlbert's own real time-major output convention -- NOT
    [T,C]). Returns [$n_tokens,512] (T=ne[0], channel-first -- DurationEncoder's own convention,
    matching the real `.transpose(-1,-2)`)."""
    def p(name):
        return f"{sd_prefix}.{name}" if sd_prefix else name

    w = tb.weight("bert_encoder.weight", sd[p("weight")].detach().cpu().numpy().astype(np.float32))
    b = tb.weight("bert_encoder.bias", sd[p("bias")].detach().cpu().numpy().astype(np.float32))

    mm = tb.node("MUL_MAT", [w, "x"], None, "mm")  # ne=[512,$n_tokens] (still time-major)
    mm_biased = tb.node("ADD", [mm, tb.node("RESHAPE", [b], {"shape": [512, 1]}, "b_r")], None, "mm_biased")
    # The real `.transpose(-1,-2)`: time-major [512,T] -> channel-first [T,512].
    out = tb.node("CONT", [tb.node("PERMUTE", [mm_biased], {"axes": [1, 0, 2, 3]}, "out_perm")], None, "out")
    return out


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["bert_encoder"]

    tb = TopologyBuilder()
    out = build_bert_encoder(tb, sd, "module")
    inputs = [{"name": "x", "dtype": "f32", "shape": ["768", "$n_tokens"]}]
    topo = tb.topology(inputs, out)

    w = GGUFWriter(str(out_dir / "kokoro_bert_encoder.gguf"), "loom-kokoro-bert-encoder")
    w.add_string("model.graph_topology", json.dumps(topo))
    for name, arr in tb.weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'kokoro_bert_encoder.gguf'}, {len(tb.weights)} weights")


if __name__ == "__main__":
    main()
