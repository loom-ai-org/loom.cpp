"""Tiny synthetic fixture for the Transducer/TDT prediction network's LSTM step, expressed as a composite
loom topology (MUL_MAT/ADD/VIEW/SIGMOID/TANH/MUL -- no monolithic LSTM primitive, matching this project's
general preference for composing existing ops, per BACKLOG.md's "Gap 1" design decision). Verifies the
composite's math bit-exact against an independent numpy reference before this pattern is trusted inside
the real TdtDecoder driver.

Matches real torch.nn.LSTM's per-step convention exactly: weight_ih/weight_hh are [4*H, in_features]/
[4*H, H] with gates packed in (i, f, g, o) order (input, forget, cell/candidate, output), separate
bias_ih/bias_hh (both added, not pre-summed -- matches nn.LSTM's own parameterization), and
c' = f*c + i*g, h' = o*tanh(c').
"""
import numpy as np

INPUT_SIZE = 3
HIDDEN = 4


def hparams() -> dict:
    return {"input_size": INPUT_SIZE, "hidden": HIDDEN}


def generate_weights() -> dict:
    rng = np.random.default_rng(11)

    def rnd(*shape):
        return rng.normal(scale=0.3, size=shape).astype(np.float32)

    return {
        "weight_ih": rnd(4 * HIDDEN, INPUT_SIZE),
        "weight_hh": rnd(4 * HIDDEN, HIDDEN),
        "bias_ih": rnd(4 * HIDDEN),
        "bias_hh": rnd(4 * HIDDEN),
    }


def generate_inputs() -> dict:
    rng = np.random.default_rng(13)
    return {
        "x": rng.normal(scale=0.5, size=INPUT_SIZE).astype(np.float32),
        "h_prev": rng.normal(scale=0.5, size=HIDDEN).astype(np.float32),
        "c_prev": rng.normal(scale=0.5, size=HIDDEN).astype(np.float32),
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def reference_step(x, h_prev, c_prev, w):
    gates = w["weight_ih"] @ x + w["weight_hh"] @ h_prev + w["bias_ih"] + w["bias_hh"]  # (4H,)
    i, f, g, o = np.split(gates, 4)
    i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_new = f * c_prev + i * g
    h_new = o * np.tanh(c_new)
    return h_new.astype(np.float32), c_new.astype(np.float32)


def build_topology(output_name: str) -> dict:
    """output_name must be "h_new" or "c_new" -- GraphTopology only supports one declared output per
    topology (confirmed in src/core/graph_topology.cpp), so the two are tested via two separate builds
    sharing the same weights/inputs rather than one topology with two outputs."""
    assert output_name in ("h_new", "c_new")
    h = HIDDEN
    f32 = 4  # bytes per f32 element, for VIEW's byte-offset "offset" attr
    return {
        "version": 1,
        "inputs": [
            {"name": "x", "dtype": "f32", "shape": [str(INPUT_SIZE)]},
            {"name": "h_prev", "dtype": "f32", "shape": [str(h)]},
            {"name": "c_prev", "dtype": "f32", "shape": [str(h)]},
        ],
        "output": output_name,
        "nodes": [
            {"op": "MUL_MAT", "inputs": ["weight_ih", "x"], "outputs": ["gates_x"]},
            {"op": "MUL_MAT", "inputs": ["weight_hh", "h_prev"], "outputs": ["gates_h"]},
            {"op": "ADD", "inputs": ["gates_x", "gates_h"], "outputs": ["gates_sum"]},
            {"op": "ADD", "inputs": ["gates_sum", "bias_ih"], "outputs": ["gates_b1"]},
            {"op": "ADD", "inputs": ["gates_b1", "bias_hh"], "outputs": ["gates"]},
            {"op": "VIEW", "inputs": ["gates"], "outputs": ["i_pre"], "attrs": {"shape": [h], "offset": 0 * h * f32}},
            {"op": "VIEW", "inputs": ["gates"], "outputs": ["f_pre"], "attrs": {"shape": [h], "offset": 1 * h * f32}},
            {"op": "VIEW", "inputs": ["gates"], "outputs": ["g_pre"], "attrs": {"shape": [h], "offset": 2 * h * f32}},
            {"op": "VIEW", "inputs": ["gates"], "outputs": ["o_pre"], "attrs": {"shape": [h], "offset": 3 * h * f32}},
            {"op": "SIGMOID", "inputs": ["i_pre"], "outputs": ["i"]},
            {"op": "SIGMOID", "inputs": ["f_pre"], "outputs": ["f"]},
            {"op": "TANH", "inputs": ["g_pre"], "outputs": ["g"]},
            {"op": "SIGMOID", "inputs": ["o_pre"], "outputs": ["o"]},
            {"op": "MUL", "inputs": ["f", "c_prev"], "outputs": ["fc"]},
            {"op": "MUL", "inputs": ["i", "g"], "outputs": ["ig"]},
            {"op": "ADD", "inputs": ["fc", "ig"], "outputs": ["c_new"]},
            {"op": "TANH", "inputs": ["c_new"], "outputs": ["tanh_c"]},
            {"op": "MUL", "inputs": ["o", "tanh_c"], "outputs": ["h_new"]},
        ],
    }
