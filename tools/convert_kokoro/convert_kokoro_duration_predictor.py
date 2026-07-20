"""Converts Kokoro's `ProsodyPredictor`'s DURATION-PREDICTION half (modules.py: `DurationEncoder` ->
`ProsodyPredictor.lstm` -> `duration_proj`) into loom-engine GGUFs, verified in isolation. Deliberately
excludes `F0Ntrain` (the F0/energy prediction half) and the duration-based frame-expansion that produces
its own input -- both depend on this piece's actual SAMPLED durations (host-computed
round/clamp/frame-count, same "generate_path" precedent as VITS), so they're a separate, later
continuation (see BACKLOG.md).

Real architecture confirmed against modules.py's `DurationEncoder`/`ProsodyPredictor` source + the real
checkpoint's state dict:
  - `DurationEncoder.lstms` interleaves 3x BIDIRECTIONAL LSTM layers (`lstms.{0,2,4}`, each
    `nn.LSTM(d_model+style_dim=640, d_model//2=256, bidirectional=True)` -> concatenated 512-wide
    output) with 3x `AdaLayerNorm` (`lstms.{1,3,5}`, `channels=d_model=512`) -- confirmed real via the
    state dict's own `predictor.text_encoder.lstms.{0,2,4}.weight_ih_l0` (BiLSTM) vs
    `lstms.{1,3,5}.fc.weight` (AdaLayerNorm) tensor names.
  - `AdaLayerNorm`'s own `forward` has TWO PAIRS of transposes that algebraically CANCEL OUT entirely
    (verified numerically against a from-scratch "plain per-position LayerNorm over channels + style
    affine" reimplementation, 0.0 diff, before trusting this) -- so despite superficially resembling
    `AdaIN1d` (used in the Decoder/`istftnet.py`, a genuinely DIFFERENT mechanism, transposed to
    normalize over TIME per-channel, i.e. real InstanceNorm), `AdaLayerNorm` is architecturally just this
    project's ordinary channel-first `LAYER_NORM` (reduces over `ne[0]`, exactly like `CustomAlbert`'s
    own LayerNorm usage) plus a style-derived `(1+gamma)*x+beta` affine, `eps=1e-5` (modules.py's own
    `AdaLayerNorm.__init__` default). Two DIFFERENT "Ada*Norm" mechanisms in the same model family --
    worth remembering not to conflate them when the Decoder is converted later.
  - After EVERY `AdaLayerNorm` (including the LAST one), `DurationEncoder.forward` re-concatenates the
    (broadcast, per-position-identical) style vector back onto the channel axis before the next BiLSTM
    layer -- confirmed from the source directly, not assumed -- so `DurationEncoder`'s own final output
    is `d_model+style_dim=640` channels wide, matching `ProsodyPredictor.lstm`'s real input width
    (`predictor.lstm.weight_ih_l0` shape `(1024,640)`, confirmed).
  - `duration_proj` (a `LinearNorm`-wrapped `nn.Linear(512,max_dur=50)`) outputs raw per-bucket logits;
    the `sigmoid().sum(-1)` duration regression happens in `KModel.forward_with_tokens`, NOT inside
    `ProsodyPredictor` itself -- this script's own topology stops at the raw `[T,50]` logits (the
    sigmoid-sum + round/clamp is a tiny host-side step, done in the reference/test directly, matching
    this project's "host does small scalar post-processing" precedent).

Style/channel-concatenation itself (NOT the BiLSTM recurrence, NOT AdaLayerNorm's actual normalize+
affine math) is done in PLAIN HOST C++ (just vector splicing) rather than an in-graph CONCAT node --
there's no temporal recurrence in it at all, so it doesn't need to be graph-resident for correctness,
and this project has no CONCAT-along-a-non-batch-axis primitive yet (not needed here, given the host
round-trip already happens for the BiLSTM stepping regardless).

Produces: `kokoro_duration_adaln_{0,1,2}.gguf` (3 standalone AdaLayerNorm topologies, one per instance,
each `{"x": [C,T], "style": [style_dim]} -> [C,T]`), `kokoro_duration_lstm_{0,1,2}_{h,c}_{fwd,bwd}.gguf`
(3 BiLSTM instances x 4 = 12 small LSTM-cell topologies, all structurally identical to
convert_kokoro_text_encoder.py's own, just different weights/dims), `kokoro_duration_top_lstm_*.gguf`
(the same, for `ProsodyPredictor.lstm`), and `kokoro_duration_proj.gguf` (a plain Linear topology).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

HP = {
    "d_model": 512,
    "style_dim": 128,
    "hidden_per_dir": 256,  # d_model // 2
    "max_dur": 50,
    "ada_ln_eps": 1e-5,
}


def to_f32(t):
    return t.detach().cpu().numpy().astype(np.float32)


def write_gguf(path, topology, weights, architecture="loom-kokoro-duration-predictor"):
    w = GGUFWriter(str(path), architecture)
    w.add_string("model.graph_topology", json.dumps(topology))
    for name, arr in weights.items():
        w.add_tensor(name, arr.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def build_adaln_topology(channels, style_dim, eps, weight_prefix="adaln"):
    """`{"x": [channels, T], "style": [style_dim]} -> [channels, T]` -- plain channel-first LAYER_NORM
    (reduces over ne[0]) + a style-derived `(1+gamma)*x+beta` affine (see module docstring for why this
    is algebraically what AdaLayerNorm's real, transpose-heavy forward reduces to). `style_dim` is a
    plain literal (128, known at conversion time) -- GraphBuilder only ever auto-registers
    n_tokens/n_past/n_kv (see convert_kokoro_text_encoder.py's own note on this same gotcha).
    `weight_prefix` namespaces the weight tensor names (default "adaln", matching this module's
    historical hardcoded names) -- needed because DurationEncoder has THREE AdaLayerNorm instances with
    genuinely different weight values that would otherwise collide under one combined GGUF file (see
    build_adaln's own docstring)."""
    return {
        "version": 1,
        "inputs": [
            {"name": "x", "dtype": "f32", "shape": [str(channels), "$n_tokens"]},
            {"name": "style", "dtype": "f32", "shape": [str(style_dim)]},
        ],
        "output": "out",
        "nodes": [
            {"op": "LAYER_NORM", "inputs": ["x"], "outputs": ["normed"], "attrs": {"eps": eps}},
            {"op": "MUL_MAT", "inputs": [f"{weight_prefix}.fc.weight", "style"], "outputs": ["h_mm"]},
            {"op": "ADD", "inputs": ["h_mm", f"{weight_prefix}.fc.bias"], "outputs": ["h"]},
            {"op": "VIEW", "inputs": ["h"], "outputs": ["gamma"], "attrs": {"shape": [channels], "offset": 0}},
            {"op": "VIEW", "inputs": ["h"], "outputs": ["beta"], "attrs": {"shape": [channels], "offset": channels * 4}},
            {"op": "RESHAPE", "inputs": ["gamma"], "outputs": ["gamma_r"], "attrs": {"shape": [channels, 1]}},
            {"op": "RESHAPE", "inputs": ["beta"], "outputs": ["beta_r"], "attrs": {"shape": [channels, 1]}},
            {"op": "ADD", "inputs": ["gamma_r", f"{weight_prefix}.one"], "outputs": ["gamma_p1"]},
            {"op": "MUL", "inputs": ["normed", "gamma_p1"], "outputs": ["scaled"]},
            {"op": "ADD", "inputs": ["scaled", "beta_r"], "outputs": ["out"]},
        ],
    }


def build_adaln(sd, lstm_idx, hp, weight_prefix="adaln"):
    """Topology + weights for one AdaLayerNorm instance (`text_encoder.lstms.{lstm_idx}`), namespaced
    under `weight_prefix` so multiple instances can coexist in one combined GGUF without a weight-name
    collision (each instance's real fc.weight/fc.bias values genuinely differ)."""
    topo = build_adaln_topology(hp["d_model"], hp["style_dim"], hp["ada_ln_eps"], weight_prefix)
    weights = {
        f"{weight_prefix}.fc.weight": to_f32(sd[f"module.text_encoder.lstms.{lstm_idx}.fc.weight"]),
        f"{weight_prefix}.fc.bias": to_f32(sd[f"module.text_encoder.lstms.{lstm_idx}.fc.bias"]),
        f"{weight_prefix}.one": np.array([1.0], dtype=np.float32),
    }
    return topo, weights


def build_lstm_cell_topology(output_name, hidden_dim, input_dim, weight_prefix):
    """Same structure as convert_kokoro_text_encoder.py's own build_lstm_cell_topology (duplicated per
    this project's usual per-tool convention -- the LSTM-cell composite has nothing model-specific
    about it at all)."""
    assert output_name in ("h_new", "c_new")
    h = hidden_dim
    f32 = 4
    return {
        "version": 1,
        "inputs": [
            {"name": "layer_input", "dtype": "f32", "shape": [str(input_dim)]},
            {"name": "h_prev", "dtype": "f32", "shape": [str(h)]},
            {"name": "c_prev", "dtype": "f32", "shape": [str(h)]},
        ],
        "output": output_name,
        "nodes": [
            {"op": "MUL_MAT", "inputs": [f"{weight_prefix}weight_ih", "layer_input"], "outputs": ["gates_x"]},
            {"op": "MUL_MAT", "inputs": [f"{weight_prefix}weight_hh", "h_prev"], "outputs": ["gates_h"]},
            {"op": "ADD", "inputs": ["gates_x", "gates_h"], "outputs": ["gates_sum"]},
            {"op": "ADD", "inputs": ["gates_sum", f"{weight_prefix}bias_ih"], "outputs": ["gates_b1"]},
            {"op": "ADD", "inputs": ["gates_b1", f"{weight_prefix}bias_hh"], "outputs": ["gates"]},
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


def build_bilstm(sd_prefix, sd, hidden_dim, input_dim, weight_namespace="lstm"):
    """Returns {"h_fwd": (topo, weights), "c_fwd": ..., "h_bwd": ..., "c_bwd": ...} for one BiLSTM
    instance. `weight_namespace` (default "lstm", matching this module's historical hardcoded prefix)
    namespaces the weight tensor names -- Kokoro has SIX distinct BiLSTM instances (this module's own 3
    DurationEncoder layers + top_lstm, plus TextEncoder's and F0Ntrain's own, each in their own script)
    that would otherwise all write identically-named-but-different-valued `lstm.weight_ih` etc. tensors,
    a real collision once consolidated into one combined GGUF file."""
    result = {}
    for output_name, direction, fname_suffix in (
        ("h_new", "", "h_fwd"), ("c_new", "", "c_fwd"),
        ("h_new", "reverse.", "h_bwd"), ("c_new", "reverse.", "c_bwd"),
    ):
        weight_prefix = f"{weight_namespace}.{direction}" if direction else f"{weight_namespace}."
        topo = build_lstm_cell_topology(output_name, hidden_dim, input_dim, weight_prefix)
        weights = {}
        for dp, suffix in ((f"{weight_namespace}.", ""), (f"{weight_namespace}.reverse.", "_reverse")):
            for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                weights[f"{dp}{kind}"] = to_f32(sd[f"{sd_prefix}.{kind}_l0{suffix}"])
        result[fname_suffix] = (topo, weights)
    return result


def write_bilstm_ggufs(out_dir, name_prefix, sd_prefix, sd, hidden_dim, input_dim, weight_namespace="lstm"):
    """Writes the 4 small (h/c x fwd/bwd) GGUFs for one BiLSTM instance, each carrying the FULL weight
    set (both directions) -- matching TdtDecoder's/convert_kokoro_text_encoder.py's own established
    convention, required for loom::BiLstmStepper's single-shared-model constructor."""
    for fname_suffix, (topo, weights) in build_bilstm(sd_prefix, sd, hidden_dim, input_dim, weight_namespace).items():
        write_gguf(out_dir / f"{name_prefix}_{fname_suffix}.gguf", topo, weights)


def build_duration_proj(sd, hp, weight_prefix="duration_proj"):
    """Plain Linear(512, max_dur=50), namespaced under `weight_prefix` (default matches the historical
    hardcoded name -- only one instance of this exists, so no collision risk today, but kept consistent
    with the other build_* helpers here)."""
    topo = {
        "version": 1,
        "inputs": [{"name": "x", "dtype": "f32", "shape": [str(hp["d_model"])]}],
        "output": "out",
        "nodes": [
            {"op": "MUL_MAT", "inputs": [f"{weight_prefix}.weight", "x"], "outputs": ["mm"]},
            {"op": "ADD", "inputs": ["mm", f"{weight_prefix}.bias"], "outputs": ["out"]},
        ],
    }
    weights = {
        f"{weight_prefix}.weight": to_f32(sd["module.duration_proj.linear_layer.weight"]),
        f"{weight_prefix}.bias": to_f32(sd["module.duration_proj.linear_layer.bias"]),
    }
    return topo, weights


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["predictor"]
    hp = HP

    # --- 3x AdaLayerNorm (lstms.1, lstms.3, lstms.5) ---
    for i, lstm_idx in enumerate((1, 3, 5)):
        topo, weights = build_adaln(sd, lstm_idx, hp)
        write_gguf(out_dir / f"kokoro_duration_adaln_{i}.gguf", topo, weights)
    print(f"wrote 3 AdaLayerNorm GGUFs to {out_dir}")

    # --- 3x BiLSTM (lstms.0, lstms.2, lstms.4), all input_dim=d_model+style_dim=640 ---
    input_dim = hp["d_model"] + hp["style_dim"]
    for i, lstm_idx in enumerate((0, 2, 4)):
        write_bilstm_ggufs(out_dir, f"kokoro_duration_lstm_{i}", f"module.text_encoder.lstms.{lstm_idx}",
                            sd, hp["hidden_per_dir"], input_dim)
    print(f"wrote 3x4=12 DurationEncoder BiLSTM GGUFs to {out_dir}")

    # --- ProsodyPredictor's own top `lstm` (same shape as the DurationEncoder's own BiLSTM layers) ---
    write_bilstm_ggufs(out_dir, "kokoro_duration_top_lstm", "module.lstm", sd, hp["hidden_per_dir"], input_dim)
    print(f"wrote 4 top-lstm GGUFs to {out_dir}")

    # --- duration_proj: plain Linear(512, max_dur=50) ---
    proj_topo, proj_weights = build_duration_proj(sd, hp)
    write_gguf(out_dir / "kokoro_duration_proj.gguf", proj_topo, proj_weights)
    print(f"wrote kokoro_duration_proj.gguf")


if __name__ == "__main__":
    main()
