"""Converts Kokoro's `TextEncoder` (modules.py -- NOT `CustomAlbert`/PL-BERT, a separate, simpler,
style-independent module: embedding -> 3x [weight-normed Conv1d -> LayerNorm -> LeakyReLU(0.2)] ->
bidirectional LSTM) into loom-engine GGUFs, verified in isolation.

Real hparams confirmed against config.json + the real checkpoint's state dict: channels=hidden_dim=512,
kernel_size=text_encoder_kernel_size=5 (padding=(5-1)//2=2), depth=n_layer=3, n_symbols=n_token=178 (a
SEPARATE embedding table from CustomAlbert's own word_embeddings -- confirmed different tensor,
`text_encoder.embedding.weight` (178,512), not tied to `bert.embeddings.word_embeddings.weight`
(178,128)). The real forward's `masked_fill_`/padding-mask logic is a no-op for every real call this
project makes (always a single, unpadded utterance -- same "no real masking needed" precedent as VITS/
Whisper/CustomAlbert), so none of it is wired here at all.

The LSTM is BIDIRECTIONAL (`nn.LSTM(channels, channels//2, 1, bidirectional=True)`) -- driven by the new
`loom::BiLstmStepper` host driver (see BACKLOG.md's Kokoro research notes for why this is host-stepped
rather than unrolled in-graph). This script writes TWO small per-direction LSTM-cell topologies (h_new/
c_new each, matching TdtDecoder's own per-layer split) referencing the checkpoint's `lstm.weight_*_l0`
(forward) and `lstm.weight_*_l0_reverse` (backward) tensors -- structurally IDENTICAL graphs, genuinely
different weight tensors, same reasoning as every other per-direction/per-layer topology split in this
project (GGUF weight references are static strings baked in at conversion time).

Produces `kokoro_text_encoder_cnn.gguf` (embedding + conv stack, output [T,C]) and
`kokoro_text_encoder_lstm_{h,c}_{fwd,bwd}.gguf` (4 small LSTM-cell topologies, all referencing the SAME
underlying `lstm.*` tensors so a single shared GgufModel works for all four, matching TdtDecoder's own
"every small GGUF carries the full weight set" convention).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

from convert_kokoro_duration_predictor import build_bilstm, write_bilstm_ggufs

HP = {
    "channels": 512,
    "kernel_size": 5,
    "depth": 3,
    "n_symbols": 178,
    "hidden_per_dir": 256,  # channels // 2
    "ln_eps": 1e-5,  # modules.py's own LayerNorm class default (NOT ALBERT's 1e-12)
    "leaky_slope": 0.2,
}


def fold_weight_norm(weight_g, weight_v):
    g = weight_g.detach().numpy() if torch.is_tensor(weight_g) else np.asarray(weight_g)
    v = weight_v.detach().numpy() if torch.is_tensor(weight_v) else np.asarray(weight_v)
    norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).reshape([-1] + [1] * (v.ndim - 1))
    return (g * v / norm).astype(np.float32)


def to_f32(t):
    return t.detach().cpu().numpy().astype(np.float32)


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


def build_cnn_topology(tb, sd, hp):
    c = hp["channels"]
    k = hp["kernel_size"]
    pad = (k - 1) // 2
    eps = hp["ln_eps"]

    emb_w = tb.weight("te.embedding.weight", to_f32(sd["module.embedding.weight"]))
    x = tb.node("GET_ROWS", [emb_w, "tokens"], None, "emb")  # [C, T] channel-first (GET_ROWS convention)
    xp = tb.node("PERMUTE", [x], {"axes": [1, 0, 2, 3]}, "x_tc_p")
    x = tb.node("CONT", [xp], None, "x_tc")  # [T, C] -- CONV_1D's own convention

    for i in range(hp["depth"]):
        p = f"module.cnn.{i}.0"
        conv_w = tb.weight(f"te.cnn.{i}.weight", fold_weight_norm(sd[f"{p}.weight_g"], sd[f"{p}.weight_v"]))
        conv_b = tb.weight(f"te.cnn.{i}.bias", to_f32(sd[f"{p}.bias"]))
        x3 = tb.node("RESHAPE", [x], {"shape": ["$n_tokens", c, 1]}, f"cnn{i}_in3")
        h = tb.node("CONV_1D", [conv_w, x3], {"s0": 1, "p0": pad, "d0": 1}, f"cnn{i}_raw")
        h = tb.node("ADD", [h, tb.node("RESHAPE", [conv_b], {"shape": [1, c, 1]}, f"cnn{i}_bias_r")],
                    None, f"cnn{i}_biased")
        h2 = tb.node("RESHAPE", [h], {"shape": ["$n_tokens", c]}, f"cnn{i}_2d")
        # LayerNorm(channels) here normalizes over the CHANNEL axis (modules.py's own LayerNorm class:
        # transpose to put channels last, F.layer_norm, transpose back) -- our [T,C] tensor already has
        # C at ne[1], not ne[0], so this needs the channel-first convention: transpose to [C,T] first.
        hp_ = tb.node("PERMUTE", [h2], {"axes": [1, 0, 2, 3]}, f"cnn{i}_ct_p")
        hc = tb.node("CONT", [hp_], None, f"cnn{i}_ct")  # [C, T]
        normed = tb.node("LAYER_NORM", [hc], {"eps": eps}, f"cnn{i}_ln_normed")
        g = tb.weight(f"te.cnn.{i}.ln_gamma", to_f32(sd[f"module.cnn.{i}.1.gamma"]))
        b = tb.weight(f"te.cnn.{i}.ln_beta", to_f32(sd[f"module.cnn.{i}.1.beta"]))
        normed = tb.node("MUL", [normed, g], None, f"cnn{i}_ln_mul")
        normed = tb.node("ADD", [normed, b], None, f"cnn{i}_ln_out")
        act = tb.node("LEAKY_RELU", [normed], {"slope": hp["leaky_slope"]}, f"cnn{i}_act")
        act_p = tb.node("PERMUTE", [act], {"axes": [1, 0, 2, 3]}, f"cnn{i}_out_tc_p")
        x = tb.node("CONT", [act_p], None, f"cnn{i}_out")  # back to [T, C] for the next conv/output

    return x


def build_cnn(sd, hp):
    """Topology + weights for the CNN front-end alone (embedding -> 3x [Conv1d -> LayerNorm ->
    LeakyReLU]), returned rather than written to disk -- lets a one-GGUF-file consolidator harvest this
    piece directly instead of round-tripping through a standalone file."""
    tb = TopologyBuilder()
    out = build_cnn_topology(tb, sd, hp)
    inputs = [{"name": "tokens", "dtype": "i32", "shape": ["$n_tokens"]}]
    return tb.topology(inputs, out), tb.weights


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["text_encoder"]

    topo, weights = build_cnn(sd, HP)
    w = GGUFWriter(str(out_dir / "kokoro_text_encoder_cnn.gguf"), "loom-kokoro-text-encoder-cnn")
    w.add_string("model.graph_topology", json.dumps(topo))
    for name, arr in weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'kokoro_text_encoder_cnn.gguf'}, {len(weights)} weights")

    # Every small GGUF carries the FULL weight set (both directions), matching TdtDecoder's own
    # "every small GGUF carries the full weight set" convention -- required so a single shared
    # GgufModel (loaded from ANY one of these 4 files) can resolve every one of the 4 topologies'
    # weight references, which is exactly what loom::BiLstmStepper's single-model constructor assumes.
    write_bilstm_ggufs(out_dir, "kokoro_text_encoder_lstm", "module.lstm", sd, HP["hidden_per_dir"], HP["channels"])
    print(f"wrote 4 LSTM-cell GGUFs to {out_dir}")


if __name__ == "__main__":
    main()
