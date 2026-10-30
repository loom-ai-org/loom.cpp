#!/usr/bin/env python3
"""A bidirectional GRU as two per-timestep cell topologies, for tests/ci/test_gru_sweep.cpp.

The engine's recurrent sweep (`cell_sweep`, behind `loom.run_recurrent` and
`loom.run_bi_recurrent_and_retain`) steps two cell shapes, told apart by what the topology declares: an
LSTM's (`c_prev` in, `h_new` and `c_new` out) and a GRU's (no `c_prev`, `h_new` out). This is the GRU's,
with the node list loom-exporter's `recurrent._gru_cell_topology` emits: torch.nn.GRU's gate blocks
`[r, z, n]`, the two biases apart because `b_hn` sits inside `r * (...)`, and
`h' = n + z * (h - n)`. The reference is an independent float64 numpy GRU, carried in the file as JSON.

Also one cell the sweep must REFUSE: two outputs and no `c_prev`, which would drop a cell state.

Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

INPUT, HIDDEN, SEQ = 3, 4, 6

DRIVER = f"""
function forward(inputs)
    local out = loom.run_recurrent('gru_fwd', inputs.sequence, {SEQ}, {INPUT}, {HIDDEN}, false)
    return out
end

function backward(inputs)
    local out = loom.run_recurrent('gru_bwd', inputs.sequence, {SEQ}, {INPUT}, {HIDDEN}, true)
    return out
end

function bidirectional(inputs)
    loom.run_bi_recurrent_and_retain('gru_fwd', 'gru_bwd', inputs.sequence, {SEQ}, {INPUT}, {HIDDEN})
    return loom.get_output('gru_fwd', 1)
end

function two_outputs_no_c(inputs)
    loom.run_recurrent('bad', inputs.sequence, {SEQ}, {INPUT}, {HIDDEN}, false)
    return {{0}}
end
"""


def cell(prefix: str) -> dict:
    h, f32 = HIDDEN, 4

    def views(src, names):
        return [{"op": "VIEW", "inputs": [src], "outputs": [n], "attrs": {"shape": [h], "offset": k * h * f32}}
                for k, n in enumerate(names)]

    return {
        "version": 1,
        "inputs": [{"name": "layer_input", "dtype": "f32", "shape": [str(INPUT)]},
                   {"name": "h_prev", "dtype": "f32", "shape": [str(h)]}],
        "outputs": ["h_new"],
        "nodes": [
            {"op": "MUL_MAT", "inputs": [f"{prefix}weight_ih", "layer_input"], "outputs": ["gx_w"]},
            {"op": "ADD", "inputs": ["gx_w", f"{prefix}bias_ih"], "outputs": ["gx"]},
            {"op": "MUL_MAT", "inputs": [f"{prefix}weight_hh", "h_prev"], "outputs": ["gh_w"]},
            {"op": "ADD", "inputs": ["gh_w", f"{prefix}bias_hh"], "outputs": ["gh"]},
            *views("gx", ("x_r", "x_z", "x_n")),
            *views("gh", ("h_r", "h_z", "h_n")),
            {"op": "ADD", "inputs": ["x_r", "h_r"], "outputs": ["r_pre"]},
            {"op": "SIGMOID", "inputs": ["r_pre"], "outputs": ["r"]},
            {"op": "ADD", "inputs": ["x_z", "h_z"], "outputs": ["z_pre"]},
            {"op": "SIGMOID", "inputs": ["z_pre"], "outputs": ["z"]},
            {"op": "MUL", "inputs": ["r", "h_n"], "outputs": ["r_hn"]},
            {"op": "ADD", "inputs": ["x_n", "r_hn"], "outputs": ["n_pre"]},
            {"op": "TANH", "inputs": ["n_pre"], "outputs": ["n"]},
            {"op": "SUB", "inputs": ["h_prev", "n"], "outputs": ["h_minus_n"]},
            {"op": "MUL", "inputs": ["z", "h_minus_n"], "outputs": ["z_h_minus_n"]},
            {"op": "ADD", "inputs": ["n", "z_h_minus_n"], "outputs": ["h_new"]},
        ],
    }


def reference(seq: np.ndarray, w: dict, reverse: bool) -> np.ndarray:
    """torch.nn.GRU's documented step, in float64."""
    def sig(v):
        return 1.0 / (1.0 + np.exp(-v))
    h = np.zeros(HIDDEN)
    out = np.zeros((SEQ, HIDDEN))
    for t in (reversed(range(SEQ)) if reverse else range(SEQ)):
        gx = w["weight_ih"] @ seq[t] + w["bias_ih"]
        gh = w["weight_hh"] @ h + w["bias_hh"]
        r = sig(gx[:HIDDEN] + gh[:HIDDEN])
        z = sig(gx[HIDDEN:2 * HIDDEN] + gh[HIDDEN:2 * HIDDEN])
        n = np.tanh(gx[2 * HIDDEN:] + r * gh[2 * HIDDEN:])
        h = (1 - z) * n + z * h
        out[t] = h
    return out


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("gru_sweep.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(29)
    weights = {}
    per_dir = {}
    for d in ("fwd", "bwd"):
        w = {"weight_ih": rng.normal(scale=0.6, size=(3 * HIDDEN, INPUT)),
             "weight_hh": rng.normal(scale=0.6, size=(3 * HIDDEN, HIDDEN)),
             "bias_ih": rng.normal(scale=0.3, size=3 * HIDDEN),
             "bias_hh": rng.normal(scale=0.3, size=3 * HIDDEN)}
        w = {k: v.astype(np.float32) for k, v in w.items()}
        per_dir[d] = {k: v.astype(np.float64) for k, v in w.items()}
        weights.update({f"{d}.{k}": v for k, v in w.items()})
    seq = rng.normal(size=(SEQ, INPUT)).astype(np.float32).astype(np.float64)
    fwd, bwd = reference(seq, per_dir["fwd"], False), reference(seq, per_dir["bwd"], True)

    bad = cell("fwd.")
    bad["outputs"] = ["h_new", "n"]

    w = GGUFWriter(str(out_path), "loom-gru-sweep-fixture")
    w.add_string("loom.architecture", "gru_sweep_test")
    w.add_string("model.graph_topology.gru_fwd", json.dumps(cell("fwd.")))
    w.add_string("model.graph_topology.gru_bwd", json.dumps(cell("bwd.")))
    w.add_string("model.graph_topology.bad", json.dumps(bad))
    w.add_string("model.driver_script", DRIVER)
    w.add_string("test.sequence", json.dumps(seq.reshape(-1).tolist()))
    w.add_string("test.forward", json.dumps(fwd.reshape(-1).tolist()))
    w.add_string("test.backward", json.dumps(bwd.reshape(-1).tolist()))
    w.add_string("test.bidirectional", json.dumps(np.concatenate([fwd, bwd], 1).reshape(-1).tolist()))
    for name, array in weights.items():
        w.add_tensor(name, array)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
