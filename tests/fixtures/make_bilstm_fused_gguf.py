#!/usr/bin/env python3
"""A two-layer BiLSTM for tests/ci/test_bilstm_fused.cpp.

`loom.run_bi_lstm_and_retain` runs a BiLSTM as one graph and reads the weights and the gate order from
the per-timestep cell topology. Layer 0 uses loom-exporter's gate order `[i, f, o, z]`. Layer 1 uses
torch.nn.LSTM's `[i, f, g, o]` and reads layer 0's retained output by reference. The reference is an
independent float64 numpy BiLSTM, carried in the file as JSON.

Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

INPUT, HIDDEN, SEQ = 3, 4, 7

DRIVER = f"""
local function two_layers(fn, seq, layout)
    fn('l0_fwd', 'l0_bwd', seq, {SEQ}, {INPUT}, {HIDDEN})
    fn('l1_fwd', 'l1_bwd', {{from = 'l0_fwd'}}, {SEQ}, {2 * HIDDEN}, {HIDDEN}, layout)
    return loom.get_output('l1_fwd', 1)
end

function fused(inputs) return two_layers(loom.run_bi_lstm_and_retain, inputs.sequence) end
function stepped(inputs) return two_layers(loom.run_bi_recurrent_and_retain, inputs.sequence) end
function fused_layout_a(inputs) return two_layers(loom.run_bi_lstm_and_retain, inputs.sequence, 'layout_a') end
function stepped_layout_a(inputs) return two_layers(loom.run_bi_recurrent_and_retain, inputs.sequence, 'layout_a') end

function refuses_gru(inputs)
    loom.run_bi_lstm_and_retain('gru', 'gru', inputs.sequence, {SEQ}, {INPUT}, {HIDDEN})
end
"""

# Gate block order -> the name each block's VIEW takes.
EXPORTER_ORDER = ("i", "f", "o", "z")
TORCH_ORDER = ("i", "f", "z", "o")


def lstm_cell(prefix: str, n_in: int, order: tuple) -> dict:
    h = HIDDEN
    views = [{"op": "VIEW", "inputs": ["gates"], "outputs": [f"{n}_pre"],
              "attrs": {"shape": [h], "offset": k * h * 4}} for k, n in enumerate(order)]
    return {
        "version": 1,
        "inputs": [{"name": "layer_input", "dtype": "f32", "shape": [str(n_in)]},
                   {"name": "h_prev", "dtype": "f32", "shape": [str(h)]},
                   {"name": "c_prev", "dtype": "f32", "shape": [str(h)]}],
        "outputs": ["h_new", "c_new"],
        "nodes": [
            {"op": "MUL_MAT", "inputs": [f"{prefix}weight_ih", "layer_input"], "outputs": ["gates_x"]},
            {"op": "MUL_MAT", "inputs": [f"{prefix}weight_hh", "h_prev"], "outputs": ["gates_h"]},
            {"op": "ADD", "inputs": ["gates_x", "gates_h"], "outputs": ["gates_sum"]},
            {"op": "ADD", "inputs": ["gates_sum", f"{prefix}bias"], "outputs": ["gates"]},
            *views,
            {"op": "SIGMOID", "inputs": ["i_pre"], "outputs": ["i"]},
            {"op": "SIGMOID", "inputs": ["f_pre"], "outputs": ["f"]},
            {"op": "SIGMOID", "inputs": ["o_pre"], "outputs": ["o"]},
            {"op": "TANH", "inputs": ["z_pre"], "outputs": ["z"]},
            {"op": "MUL", "inputs": ["f", "c_prev"], "outputs": ["fc"]},
            {"op": "MUL", "inputs": ["i", "z"], "outputs": ["iz"]},
            {"op": "ADD", "inputs": ["fc", "iz"], "outputs": ["c_new"]},
            {"op": "TANH", "inputs": ["c_new"], "outputs": ["tanh_c"]},
            {"op": "MUL", "inputs": ["o", "tanh_c"], "outputs": ["h_new"]},
        ],
    }


def gru_cell() -> dict:
    return {
        "version": 1,
        "inputs": [{"name": "layer_input", "dtype": "f32", "shape": [str(INPUT)]},
                   {"name": "h_prev", "dtype": "f32", "shape": [str(HIDDEN)]}],
        "outputs": ["h_new"],
        "nodes": [{"op": "MUL_MAT", "inputs": ["gru.weight", "layer_input"], "outputs": ["h_new"]}],
    }


def direction(seq: np.ndarray, w: dict, order: tuple, reverse: bool) -> np.ndarray:
    def sig(v):
        return 1.0 / (1.0 + np.exp(-v))
    h, c = np.zeros(HIDDEN), np.zeros(HIDDEN)
    out = np.zeros((len(seq), HIDDEN))
    for t in (reversed(range(len(seq))) if reverse else range(len(seq))):
        g = w["weight_ih"] @ seq[t] + w["weight_hh"] @ h + w["bias"]
        gate = {n: g[k * HIDDEN:(k + 1) * HIDDEN] for k, n in enumerate(order)}
        c = sig(gate["f"]) * c + sig(gate["i"]) * np.tanh(gate["z"])
        h = sig(gate["o"]) * np.tanh(c)
        out[t] = h
    return out


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("bilstm_fused.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    seq = rng.normal(size=(SEQ, INPUT)).astype(np.float32).astype(np.float64)

    w = GGUFWriter(str(out_path), "loom-bilstm-fused-fixture")
    w.add_string("loom.architecture", "bilstm_fused_test")
    x = seq
    for layer, (n_in, order) in enumerate(((INPUT, EXPORTER_ORDER), (2 * HIDDEN, TORCH_ORDER))):
        halves = []
        for d in ("fwd", "bwd"):
            prefix = f"l{layer}.{d}."
            weights = {"weight_ih": rng.normal(scale=0.6, size=(4 * HIDDEN, n_in)),
                       "weight_hh": rng.normal(scale=0.6, size=(4 * HIDDEN, HIDDEN)),
                       "bias": rng.normal(scale=0.3, size=4 * HIDDEN)}
            weights = {k: v.astype(np.float32) for k, v in weights.items()}
            for k, v in weights.items():
                w.add_tensor(prefix + k, v)
            w.add_string(f"model.graph_topology.l{layer}_{d}", json.dumps(lstm_cell(prefix, n_in, order)))
            halves.append(direction(x, {k: v.astype(np.float64) for k, v in weights.items()}, order, d == "bwd"))
        x = np.concatenate(halves, 1)
    w.add_tensor("gru.weight", rng.normal(size=(HIDDEN, INPUT)).astype(np.float32))
    w.add_string("model.graph_topology.gru", json.dumps(gru_cell()))
    w.add_string("model.driver_script", DRIVER)
    w.add_string("test.sequence", json.dumps(seq.reshape(-1).tolist()))
    w.add_string("test.rows", json.dumps(x.reshape(-1).tolist()))
    w.add_string("test.layout_a", json.dumps(x.T.reshape(-1).tolist()))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
