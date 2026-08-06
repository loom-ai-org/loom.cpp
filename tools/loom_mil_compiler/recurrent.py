"""
MIL `lstm`/`gru` op -> per-timestep cell topology + weights (EXPORT-IMPROVEMENT-BACKLOG.md item 4).

ggml has no native LSTM/GRU op, but this engine already has a full, hand-written LSTM implementation:
`src/core/bilstm_stepper.cpp`'s `BiLstmStepper` (used today by `styletts2_driver.cpp`/`kokoro_driver.cpp`,
models still on the old hand-written-driver architecture) plus a hand-authored Python topology builder,
`tools/convert_kokoro/convert_kokoro_duration_predictor.py`'s `build_lstm_cell_topology()`, which composes
the LSTM cell entirely from already-registered primitives (`MUL_MAT`/`ADD`/`SIGMOID`/`TANH`/`MUL`/`VIEW`
-- no new ggml kernel needed). This module generalizes that exact node sequence to be driven by a REAL
traced MIL `lstm`/`gru` op's own weight/bias tensor values instead of a hand-picked state-dict slice, so
`exporter.py` can auto-detect `torch.nn.LSTM`/`torch.nn.GRU` wherever coremltools' torch frontend produces
one (confirmed: it maps them directly to opaque `lstm`/`gru` ops, with NO lowering pass -- unlike
`complex_stft`, no MIL pass ever decomposes these into primitive ops, so detecting the op type itself is
exact, not a pattern-match).

Two things confirmed by directly tracing real `torch.nn.LSTM`/`torch.nn.GRU` modules (not assumed from the
docstrings alone -- MIL's `lstm` docstring's own prose is a little ambiguous about bias packing, so the
gate order and pre-summed-bias behavior below were verified against real traced op values AND a numpy
reproduction of a real `nn.LSTM` forward pass before being trusted):

1. **Gate order is `[i, f, o, z]`** (input, forget, output, cell-candidate) in `weight_ih`/`weight_hh` --
   NOT `torch.nn.LSTM`'s own native `[i, f, g, o]` state-dict packing. coremltools' torch frontend already
   permutes gate blocks when it builds the `lstm` op, so reading `op.inputs["weight_ih"].val` gives data
   already in MIL's own `ifoz` order -- this module's own `VIEW` offsets must match that order, not the
   reference `convert_kokoro_duration_predictor.py` script's offsets, which slice a raw PyTorch state dict
   still in `ifgo` order.
2. **`bias` is a single pre-summed `4*H` tensor** (`bias_ih + bias_hh`, already combined by the torch
   frontend), not two separate `4*H` halves the way `nn.LSTM`'s own `bias_ih_l0`/`bias_hh_l0` state-dict
   entries are -- so the cell topology only needs ONE bias-add, not two.

**Bidirectional handling**: a `bidirectional=True` `nn.LSTM` traces as a SINGLE `lstm` op with
`direction="bidirectional"`, packing BOTH directions' weights (`weight_ih`/`weight_hh`/`bias` for forward,
`weight_ih_back`/`weight_hh_back`/`bias_back` for backward) into that one op -- confirmed directly; it is
*not* two separate `lstm` ops the way a naive reading of "HF's own nn.LSTM(bidirectional=True) decomposes
that way" might suggest. `build_lstm_cell_topologies` returns up to two directions' worth of `{h, c}`
topology pairs accordingly, matching `BiLstmStepper`'s own 4-topology (h_fwd/c_fwd/h_bwd/c_bwd) shape.

**GRU is NOT handled here.** MIL does have an opaque `gru` op schema (confirmed via
`coremltools.converters.mil.mil.ops.defs.iOS15.recurrent.gru`), and the original plan for this module
assumed `torch.nn.GRU` would trace to it exactly like `torch.nn.LSTM` traces to `lstm`. That assumption
was wrong, found only by actually tracing a real `nn.GRU` (with and without an explicit initial hidden
state, uni-directional) through this pipeline's real `torch.jit.trace` + `ct.convert(...)` path: it
decomposes into a `while_loop` + `slice_by_index` + consts instead -- coremltools' torch frontend's `gru`
handler is simply never reached this way in this environment/torch version. That makes GRU a fundamentally
different problem than LSTM here: there is no opaque op to detect and swap for a stepper call, only a
raw, real recurrent Python loop already unrolled into MIL control-flow ops -- closer to the "bespoke
workflow's `transpile_operation` already has real `while_loop` handling" case than to "detect one opaque
op." Revisiting GRU support needs a concrete target model to trace and inspect first, not more
speculation from the schema alone.
"""
import numpy as np


def _val(op, key, default=None):
    v = op.inputs.get(key)
    if v is None or not hasattr(v, "val") or v.val is None:
        return default
    return v.val


def _require_activation(op, key, expected: str, op_label: str) -> None:
    got = _val(op, key, expected)
    if got != expected:
        raise NotImplementedError(
            f"{op_label} op '{op.name}' has {key}='{got}', which this exporter doesn't support "
            f"(only '{expected}' is)."
        )


def _lstm_cell_topology(hidden_dim: int, input_dim: int, weight_prefix: str) -> dict:
    """One LSTM cell step, gate order [i, f, o, z] (MIL's own `lstm` op convention -- see module
    docstring), single pre-summed `weight_prefix + 'bias'`.

    **Declares BOTH outputs, which halves the work of every LSTM in this project.** A cell step computes
    `h_new` and `c_new` from one gate stack, but `GraphTopology` allowed only one declared output when
    this was written, so the established precedent (convert_kokoro_duration_predictor.py's own
    `build_lstm_cell_topology`) was to emit the identical node list twice and vary only the declared
    output -- and every caller then ran both, computing the gates, the four VIEWs and the six
    elementwise ops a second time to read the other half of the same result. P2 added multi-output
    topologies; this is that precedent retired. Kokoro and StyleTTS2 each drive six BiLSTMs over a whole
    sequence, so it is not a marginal saving.

    Output ORDER is the contract: `["h_new", "c_new"]`, which is what `run_bi_lstm`'s two capture
    variables and `l_run_recurrent`'s two reads both assume, and what `{from = ..., index = 2}` means
    for a caller threading the cell state onward."""
    h = hidden_dim
    f32 = 4
    return {
        "version": 1,
        "inputs": [
            {"name": "layer_input", "dtype": "f32", "shape": [str(input_dim)]},
            {"name": "h_prev", "dtype": "f32", "shape": [str(h)]},
            {"name": "c_prev", "dtype": "f32", "shape": [str(h)]},
        ],
        "outputs": ["h_new", "c_new"],
        "nodes": [
            {"op": "MUL_MAT", "inputs": [f"{weight_prefix}weight_ih", "layer_input"], "outputs": ["gates_x"]},
            {"op": "MUL_MAT", "inputs": [f"{weight_prefix}weight_hh", "h_prev"], "outputs": ["gates_h"]},
            {"op": "ADD", "inputs": ["gates_x", "gates_h"], "outputs": ["gates_sum"]},
            {"op": "ADD", "inputs": ["gates_sum", f"{weight_prefix}bias"], "outputs": ["gates"]},
            {"op": "VIEW", "inputs": ["gates"], "outputs": ["i_pre"], "attrs": {"shape": [h], "offset": 0 * h * f32}},
            {"op": "VIEW", "inputs": ["gates"], "outputs": ["f_pre"], "attrs": {"shape": [h], "offset": 1 * h * f32}},
            {"op": "VIEW", "inputs": ["gates"], "outputs": ["o_pre"], "attrs": {"shape": [h], "offset": 2 * h * f32}},
            {"op": "VIEW", "inputs": ["gates"], "outputs": ["z_pre"], "attrs": {"shape": [h], "offset": 3 * h * f32}},
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


def build_lstm_cell_topologies(op, weight_namespace: str) -> dict:
    """`op` is a MIL `lstm` Operation. `weight_namespace` prefixes every weight tensor name this produces
    (avoids collisions across multiple LSTM instances sharing one GGUF, the same convention
    `write_bilstm_ggufs` uses per-instance today). Returns:
        {
            "hidden_dim": H, "input_dim": I, "bidirectional": bool,
            "forward": topo_json,          # declares BOTH outputs, in the order ["h_new", "c_new"]
            "backward": topo_json | None,   # same, for the reverse direction of a bidirectional LSTM
            "weights": {namespaced_tensor_name: np.ndarray},
        }
    """
    _require_activation(op, "recurrent_activation", "sigmoid", "lstm")
    _require_activation(op, "cell_activation", "tanh", "lstm")
    _require_activation(op, "activation", "tanh", "lstm")

    direction = _val(op, "direction", "forward")
    if direction not in ("forward", "reverse", "bidirectional"):
        raise NotImplementedError(f"lstm op '{op.name}' has direction='{direction}', which this exporter doesn't support.")
    bidirectional = direction == "bidirectional"

    weight_ih = op.inputs["weight_ih"].val
    hidden_dim = weight_ih.shape[0] // 4
    input_dim = weight_ih.shape[1]

    weights = {}

    def _register_direction(suffix: str, ih, hh, bias):
        prefix = f"{weight_namespace}{suffix}"
        weights[f"{prefix}weight_ih"] = np.asarray(ih, dtype=np.float32)
        weights[f"{prefix}weight_hh"] = np.asarray(hh, dtype=np.float32)
        weights[f"{prefix}bias"] = np.asarray(bias, dtype=np.float32)
        return _lstm_cell_topology(hidden_dim, input_dim, prefix)

    forward_topos = _register_direction(
        "fwd.", op.inputs["weight_ih"].val, op.inputs["weight_hh"].val,
        _val(op, "bias", np.zeros(4 * hidden_dim, dtype=np.float32)),
    )
    # A standalone (non-bidirectional) reverse LSTM still only carries ONE weight set, under the same
    # "forward" input keys MIL uses regardless -- it's the caller (the exporter, via the returned
    # `direction`) that decides which time-direction to walk the sequence in, not which dict key the
    # weights happen to live under.
    backward_topos = None
    if bidirectional:
        backward_topos = _register_direction(
            "bwd.", op.inputs["weight_ih_back"].val, op.inputs["weight_hh_back"].val,
            _val(op, "bias_back", np.zeros(4 * hidden_dim, dtype=np.float32)),
        )

    return {
        "hidden_dim": hidden_dim,
        "input_dim": input_dim,
        "direction": direction,
        "bidirectional": bidirectional,
        "forward": forward_topos,
        "backward": backward_topos,
        "weights": weights,
    }
