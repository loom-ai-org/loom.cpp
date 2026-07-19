"""Tiny synthetic fixture for a full greedy PLAIN RNN-T (not TDT) decode step -- same token embedding ->
LSTM stack -> joint network machinery as tdt_step_common.py, but the joint has NO duration head at all
(output width is exactly n_vocab+1, not n_vocab+1+n_durations), and the reference greedy decode uses
standard RNN-T control flow: a blank ALWAYS advances exactly one frame; a non-blank NEVER advances the
frame (stays put, emitting further symbols on the same frame up to max_symbols_per_step) -- there is no
predicted "duration" concept at all, unlike TDT.

Exercises `loom::TdtDecoder`'s new plain-RNN-T mode (`TdtDecoderConfig.durations` left empty) against an
independent numpy reference of the real NeMo RNN-T greedy-decoding algorithm, BEFORE the real
nvidia/parakeet-rnnt-0.6b checkpoint is involved -- same "synthetic fixture first" discipline as
tdt_step_common.py's own role for TDT.

N_LSTM_LAYERS=2 for the same reason as tdt_step_common.py: matches the real checkpoint's actual
prediction-network depth (confirmed against nvidia/parakeet-rnnt-0.6b's own state dict, see BACKLOG.md),
not simplified to 1 -- this exercises the driver's inter-layer chaining for real.
"""
import numpy as np

N_VOCAB = 3       # real tokens 0,1,2
BLANK_ID = N_VOCAB  # 3 -- NeMo convention: blank is the last "extra" vocab entry
PRED_HIDDEN = 4
N_LSTM_LAYERS = 2
N_EMBD = 3        # encoder frame width
JOINT_HIDDEN = 4
N_FRAMES = 3
MAX_SYMBOLS_PER_STEP = 3


def hparams() -> dict:
    return {
        "n_vocab": N_VOCAB, "blank_id": BLANK_ID, "pred_hidden": PRED_HIDDEN, "n_lstm_layers": N_LSTM_LAYERS,
        "n_embd": N_EMBD, "joint_hidden": JOINT_HIDDEN, "n_frames": N_FRAMES,
        "max_symbols_per_step": MAX_SYMBOLS_PER_STEP,
    }


def generate_weights() -> dict:
    # Seed hand-picked (searched over several thousand candidates, same discipline as
    # tdt_step_common.py's own seed search) so the decode below naturally exercises three genuinely
    # different per-frame cases without ever relying on the driver's own MAX_SYMBOLS_PER_STEP safety
    # net: frame 0 emits exactly one non-blank symbol then blanks, frame 1 emits TWO non-blank symbols
    # (exercising "stay on this frame" more than once) before blanking, and frame 2 blanks immediately
    # with no emission at all -- token/frame sequence [1,1,1] at frames [0,1,1].
    rng = np.random.default_rng(8585)

    def rnd(*shape):
        return rng.normal(scale=0.3, size=shape).astype(np.float32)

    h = PRED_HIDDEN
    w = {
        "embed.weight": rnd(N_VOCAB + 1, h),
        "joint.enc.weight": rnd(JOINT_HIDDEN, N_EMBD),
        "joint.enc.bias": rnd(JOINT_HIDDEN),
        "joint.pred.weight": rnd(JOINT_HIDDEN, h),
        "joint.pred.bias": rnd(JOINT_HIDDEN),
        "joint.out.weight": rnd(N_VOCAB + 1, JOINT_HIDDEN),  # NO duration columns -- the whole point
        "joint.out.bias": rnd(N_VOCAB + 1),
    }
    for layer in range(N_LSTM_LAYERS):
        w[f"lstm.{layer}.weight_ih"] = rnd(4 * h, h)
        w[f"lstm.{layer}.weight_hh"] = rnd(4 * h, h)
        w[f"lstm.{layer}.bias_ih"] = rnd(4 * h)
        w[f"lstm.{layer}.bias_hh"] = rnd(4 * h)
    return w


def generate_encoder_output() -> np.ndarray:
    rng = np.random.default_rng(1122)
    return rng.normal(scale=0.4, size=(N_FRAMES, N_EMBD)).astype(np.float32)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_step(layer_input, h_prev, c_prev, w, layer: int):
    gates = (w[f"lstm.{layer}.weight_ih"] @ layer_input + w[f"lstm.{layer}.weight_hh"] @ h_prev
             + w[f"lstm.{layer}.bias_ih"] + w[f"lstm.{layer}.bias_hh"])
    i, f, g, o = np.split(gates, 4)
    i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_new = f * c_prev + i * g
    h_new = o * np.tanh(c_new)
    return h_new.astype(np.float32), c_new.astype(np.float32)


def lstm_stack_step(last_label, h_prev_layers, c_prev_layers, w):
    layer_input = w["embed.weight"][last_label]
    h_new_layers, c_new_layers = [], []
    for layer in range(N_LSTM_LAYERS):
        h_new, c_new = lstm_step(layer_input, h_prev_layers[layer], c_prev_layers[layer], w, layer)
        h_new_layers.append(h_new)
        c_new_layers.append(c_new)
        layer_input = h_new
    return h_new_layers, c_new_layers, h_new_layers[-1]


def joint(encoder_frame, decoder_out, w):
    f_proj = w["joint.enc.weight"] @ encoder_frame + w["joint.enc.bias"]
    g_proj = w["joint.pred.weight"] @ decoder_out + w["joint.pred.bias"]
    activated = np.maximum(f_proj + g_proj, 0.0)  # RELU
    return (w["joint.out.weight"] @ activated + w["joint.out.bias"]).astype(np.float32)


def reference_greedy_decode(encoder_output: np.ndarray, w: dict):
    """Standard RNN-T greedy control flow (Graves 2012 / NeMo's own RNNTGreedyDecoder): blank ALWAYS
    advances exactly one frame; non-blank NEVER advances (stays on the same frame, up to
    MAX_SYMBOLS_PER_STEP total emissions per frame). No duration concept at all, unlike TDT. Returns
    (tokens, frame_indices)."""
    h_layers = [np.zeros(PRED_HIDDEN, dtype=np.float32) for _ in range(N_LSTM_LAYERS)]
    c_layers = [np.zeros(PRED_HIDDEN, dtype=np.float32) for _ in range(N_LSTM_LAYERS)]
    last_label = BLANK_ID  # NeMo's SOS sentinel for the very first step

    tokens, frame_indices = [], []
    time_idx = 0
    while time_idx < N_FRAMES:
        f = encoder_output[time_idx]
        symbols_added = 0
        while symbols_added < MAX_SYMBOLS_PER_STEP:
            h_new_layers, c_new_layers, top_h = lstm_stack_step(last_label, h_layers, c_layers, w)
            combined = joint(f, top_h, w)
            k = int(np.argmax(combined))
            symbols_added += 1
            if k != BLANK_ID:
                tokens.append(k)
                frame_indices.append(time_idx)
                h_layers, c_layers, last_label = h_new_layers, c_new_layers, k
                continue  # stay on this frame
            break  # blank: advance to the next frame
        time_idx += 1
    return tokens, frame_indices


def build_lstm_topology(layer: int, output_name: str) -> dict:
    """Identical structure to tdt_step_common.py's own build_lstm_topology (duplicated per this
    project's usual per-fixture convention, not shared -- the LSTM-cell composite has nothing TDT- or
    RNN-T-specific about it at all)."""
    assert output_name in ("h_new", "c_new")
    h = PRED_HIDDEN
    f32 = 4
    if layer == 0:
        inputs = [
            {"name": "last_label", "dtype": "i32", "shape": ["1"]},
            {"name": "h_prev", "dtype": "f32", "shape": [str(h)]},
            {"name": "c_prev", "dtype": "f32", "shape": [str(h)]},
        ]
        embed_nodes = [
            {"op": "GET_ROWS", "inputs": ["embed.weight", "last_label"], "outputs": ["embed_row"]},
            {"op": "RESHAPE", "inputs": ["embed_row"], "outputs": ["layer_input_resolved"], "attrs": {"shape": [h]}},
        ]
    else:
        inputs = [
            {"name": "layer_input", "dtype": "f32", "shape": [str(h)]},
            {"name": "h_prev", "dtype": "f32", "shape": [str(h)]},
            {"name": "c_prev", "dtype": "f32", "shape": [str(h)]},
        ]
        embed_nodes = [
            {"op": "RESHAPE", "inputs": ["layer_input"], "outputs": ["layer_input_resolved"], "attrs": {"shape": [h]}},
        ]

    p = f"lstm.{layer}."
    return {
        "version": 1,
        "inputs": inputs,
        "output": output_name,
        "nodes": embed_nodes + [
            {"op": "MUL_MAT", "inputs": [p + "weight_ih", "layer_input_resolved"], "outputs": ["gates_x"]},
            {"op": "MUL_MAT", "inputs": [p + "weight_hh", "h_prev"], "outputs": ["gates_h"]},
            {"op": "ADD", "inputs": ["gates_x", "gates_h"], "outputs": ["gates_sum"]},
            {"op": "ADD", "inputs": ["gates_sum", p + "bias_ih"], "outputs": ["gates_b1"]},
            {"op": "ADD", "inputs": ["gates_b1", p + "bias_hh"], "outputs": ["gates"]},
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


def build_joint_topology() -> dict:
    return {
        "version": 1,
        "inputs": [
            {"name": "encoder_frame", "dtype": "f32", "shape": [str(N_EMBD)]},
            {"name": "decoder_out", "dtype": "f32", "shape": [str(PRED_HIDDEN)]},
        ],
        "output": "combined",
        "nodes": [
            {"op": "MUL_MAT", "inputs": ["joint.enc.weight", "encoder_frame"], "outputs": ["f_proj_mm"]},
            {"op": "ADD", "inputs": ["f_proj_mm", "joint.enc.bias"], "outputs": ["f_proj"]},
            {"op": "MUL_MAT", "inputs": ["joint.pred.weight", "decoder_out"], "outputs": ["g_proj_mm"]},
            {"op": "ADD", "inputs": ["g_proj_mm", "joint.pred.bias"], "outputs": ["g_proj"]},
            {"op": "ADD", "inputs": ["f_proj", "g_proj"], "outputs": ["summed"]},
            {"op": "RELU", "inputs": ["summed"], "outputs": ["activated"]},
            {"op": "MUL_MAT", "inputs": ["joint.out.weight", "activated"], "outputs": ["combined_mm"]},
            {"op": "ADD", "inputs": ["combined_mm", "joint.out.bias"], "outputs": ["combined"]},
        ],
    }
