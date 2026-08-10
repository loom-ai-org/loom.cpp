"""Tiny synthetic fixture for a full greedy TDT (Token-and-Duration Transducer) decode step: token
embedding lookup -> a STACK of LSTM-step composites (see lstm_step_common.py for the single-layer version
this generalizes) -> joint network (encoder-frame projection + decoder-step projection, summed, RELU,
final linear -> combined token+duration logits). Verifies loom's new TdtDecoder C++ driver's control flow
(frame-pointer advance via the predicted duration, blank forcing duration>=1, LSTM state carried only on
non-blank, multi-layer LSTM chaining) against an independent numpy reference of NeMo's real greedy-TDT
algorithm (see BACKLOG.md's research notes) -- BEFORE the real Parakeet-TDT-0.6B-v3 checkpoint is involved.

N_LSTM_LAYERS=2 deliberately matches the real checkpoint's own `decoder.prediction.dec_rnn.lstm` (a real
torch.nn.LSTM(num_layers=2), confirmed against the real state dict -- weight_ih_l0/weight_ih_l1 both
present) rather than testing only the single-layer case, which wouldn't exercise inter-layer chaining at
all (layer i>0's input is layer i-1's h_new, not a fresh embedding lookup).

Per-layer topologies (GraphTopology only supports one declared output each):
  - "lstm_h"/"lstm_c" for layer 0: embeds last_label (GET_ROWS), then the composite gates.
  - "lstm_h"/"lstm_c" for layer i>0: takes "layer_input" (f32, the previous layer's h_new) directly,
    no embedding lookup -- same composite gates otherwise.
  - "joint": encoder_frame + decoder_out (the TOP layer's h_new) -> combined
    [n_vocab+1+n_durations] logits (first n_vocab+1 are token+blank, last n_durations are the duration
    head, matching real TDT's single-combined-linear-output convention).
"""
import numpy as np

N_VOCAB = 3       # real tokens 0,1,2
BLANK_ID = N_VOCAB  # 3 -- NeMo convention: blank is the last "extra" vocab entry
PRED_HIDDEN = 4
N_LSTM_LAYERS = 2   # matches the real checkpoint's real depth, not simplified to 1
N_EMBD = 3        # encoder frame width
JOINT_HIDDEN = 4
DURATIONS = [0, 1, 2]
N_DURATIONS = len(DURATIONS)
N_FRAMES = 3
MAX_SYMBOLS_PER_STEP = 3


def hparams() -> dict:
    return {
        "n_vocab": N_VOCAB, "blank_id": BLANK_ID, "pred_hidden": PRED_HIDDEN, "n_lstm_layers": N_LSTM_LAYERS,
        "n_embd": N_EMBD, "joint_hidden": JOINT_HIDDEN, "durations": DURATIONS, "n_frames": N_FRAMES,
        "max_symbols_per_step": MAX_SYMBOLS_PER_STEP,
    }


def generate_weights() -> dict:
    # Seed hand-picked (searched over several hundred candidates) so the decode below naturally exercises
    # both a blank-driven multi-frame skip (duration=2) and a non-blank emission with duration=2, without
    # relying on the MAX_SYMBOLS_PER_STEP safety-net fallback -- not an arbitrary choice.
    rng = np.random.default_rng(122)

    def rnd(*shape):
        return rng.normal(scale=0.3, size=shape).astype(np.float32)

    h = PRED_HIDDEN
    w = {
        "embed.weight": rnd(N_VOCAB + 1, h),  # + blank row
        "joint.enc.weight": rnd(JOINT_HIDDEN, N_EMBD),
        "joint.enc.bias": rnd(JOINT_HIDDEN),
        "joint.pred.weight": rnd(JOINT_HIDDEN, h),
        "joint.pred.bias": rnd(JOINT_HIDDEN),
        "joint.out.weight": rnd(N_VOCAB + 1 + N_DURATIONS, JOINT_HIDDEN),
        "joint.out.bias": rnd(N_VOCAB + 1 + N_DURATIONS),
    }
    for layer in range(N_LSTM_LAYERS):
        w[f"lstm.{layer}.weight_ih"] = rnd(4 * h, h)  # input_size == hidden_size == h for every layer,
        w[f"lstm.{layer}.weight_hh"] = rnd(4 * h, h)  # matching the real checkpoint (layer i>0's input is
        w[f"lstm.{layer}.bias_ih"] = rnd(4 * h)       # layer i-1's h-sized output, not a separate embed dim)
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
    """Runs the full N_LSTM_LAYERS stack for one step. h_prev_layers/c_prev_layers: lists, one per layer.
    Returns (h_new_layers, c_new_layers, top_h -- the final layer's h_new, fed to the joint)."""
    layer_input = w["embed.weight"][last_label]
    h_new_layers, c_new_layers = [], []
    for layer in range(N_LSTM_LAYERS):
        h_new, c_new = lstm_step(layer_input, h_prev_layers[layer], c_prev_layers[layer], w, layer)
        h_new_layers.append(h_new)
        c_new_layers.append(c_new)
        layer_input = h_new  # next layer's input is this layer's output, not a fresh embedding
    return h_new_layers, c_new_layers, h_new_layers[-1]


def joint(encoder_frame, decoder_out, w):
    f_proj = w["joint.enc.weight"] @ encoder_frame + w["joint.enc.bias"]
    g_proj = w["joint.pred.weight"] @ decoder_out + w["joint.pred.bias"]
    activated = np.maximum(f_proj + g_proj, 0.0)  # RELU
    return (w["joint.out.weight"] @ activated + w["joint.out.bias"]).astype(np.float32)


def reference_greedy_decode(encoder_output: np.ndarray, w: dict):
    """Real NeMo greedy-TDT control flow (see module docstring for the source), in plain numpy. Returns
    (tokens, frame_indices) -- frame_indices[i] is the encoder frame token i was emitted at."""
    h_layers = [np.zeros(PRED_HIDDEN, dtype=np.float32) for _ in range(N_LSTM_LAYERS)]
    c_layers = [np.zeros(PRED_HIDDEN, dtype=np.float32) for _ in range(N_LSTM_LAYERS)]
    last_label = BLANK_ID  # NeMo's SOS sentinel for the very first step

    tokens, frame_indices = [], []
    time_idx = 0
    while time_idx < N_FRAMES:
        f = encoder_output[time_idx]
        symbols_added = 0
        advanced = False
        while symbols_added < MAX_SYMBOLS_PER_STEP:
            h_new_layers, c_new_layers, top_h = lstm_stack_step(last_label, h_layers, c_layers, w)
            combined = joint(f, top_h, w)
            token_logits = combined[: N_VOCAB + 1]
            duration_logits = combined[N_VOCAB + 1 :]
            k = int(np.argmax(token_logits))
            d_idx = int(np.argmax(duration_logits))
            skip = DURATIONS[d_idx]
            if k != BLANK_ID:
                tokens.append(k)
                frame_indices.append(time_idx)
                h_layers, c_layers, last_label = h_new_layers, c_new_layers, k
            elif skip == 0:
                skip = 1  # blank is forced to advance at least one frame
            symbols_added += 1
            time_idx += skip
            if skip > 0:
                advanced = True
                break
        if not advanced:
            # Defensive termination bound, not part of the real TDT algorithm itself (which relies on
            # blank-forcing alone): guards against a pathological model that keeps emitting non-blank
            # tokens with duration 0 forever, which would otherwise spin on the same frame indefinitely.
            # Hit this for real against this fixture's own random weights before adding it.
            time_idx += 1
    return tokens, frame_indices


def build_lstm_topology(layer: int, output_name: str) -> dict:
    """layer 0 embeds last_label (GET_ROWS); layer>0 takes "layer_input" (f32, the previous layer's h_new)
    directly -- same composite gates (MUL_MAT/ADD/VIEW/SIGMOID/TANH/MUL) either way."""
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
