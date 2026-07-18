#!/usr/bin/env python3
"""Converts the real NVIDIA nvidia/parakeet-tdt-0.6b-v3 checkpoint (a .nemo file) into loom-engine GGUFs:
one for the FastConformer encoder (mel frontend + encoder, no CTC/decoder head -- this is TDT, not CTC),
and one pair (h/c) per LSTM layer plus one for the joint network, for the new TdtDecoder C++ driver
(tools/fixture_gen/tdt_step_common.py's synthetic-fixture pattern, extended to real weights/hparams).

Real hparams/weight-schema confirmed directly against this checkpoint's own model_config.yaml and
model_weights.ckpt state dict (not assumed from general NeMo/HF-transformers knowledge -- see BACKLOG.md):
  - Encoder: 24 layers, d_model=1024, n_heads=8, ff_expansion=4 (ff_hidden=4096), conv_kernel_size=9,
    feat_in=128 (n_mels), subsampling="dw_striding" (3 stages: one plain Conv2d, then two
    depthwise+pointwise Conv2d pairs -- needs the new CONV_2D_DW primitive, absent for Conformer-CTC-
    small's simpler 2-stage plain-Conv2d subsampling), xscaling=false (NO sqrt(d_model) scaling, unlike
    Conformer-CTC-small), use_bias=false -- confirmed from the real state dict, not just the config flag,
    that this means NO bias anywhere in the encoder: not on self-attn/pos projections (expected), and
    -- a real, checkpoint-specific finding -- ALSO not on the conv module's 3 convs (pointwise_conv1/
    depthwise_conv/pointwise_conv2), unlike HF transformers' own `modeling_parakeet.py` port, which
    hardcodes bias=True for those regardless of config (verified this checkpoint's real state dict has no
    such bias tensors before trusting either secondary source).
  - Decoder: RNNTDecoder, a real 2-layer stacked nn.LSTM (pred_hidden=640, confirmed via
    decoder.prediction.dec_rnn.lstm.weight_ih_l0 AND _l1 both present), embedding table
    decoder.prediction.embed.weight [8193, 640] (8192 real tokens + 1 blank row).
  - Joint: RNNTJoint, joint_hidden=640, activation=relu, num_classes=8192, num_extra_outputs=5
    (durations=[0,1,2,3,4]) -> final linear width 8192+1(blank)+5 = 8198.

Requires: pip install torch gguf numpy pyyaml librosa
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mel_common
import nemo_common as common
import tokenizer_common


def hparams(config: dict) -> dict:
    enc = config["encoder"]
    dec = config["decoder"]["prednet"]
    joint = config["joint"]["jointnet"]
    n_embd = enc["d_model"]
    n_head = enc["n_heads"]
    return {
        "n_layers": enc["n_layers"], "n_embd": n_embd, "n_head": n_head, "head_dim": n_embd // n_head,
        "ff_hidden": n_embd * enc["ff_expansion_factor"], "conv_kernel_size": enc["conv_kernel_size"],
        "conv_padding": (enc["conv_kernel_size"] - 1) // 2, "feat_in": enc["feat_in"],
        "subsampling_conv_channels": enc["subsampling_conv_channels"],
        "subsampling_num_layers": 3,  # log2(subsampling_factor=8) -- confirmed real (dw_striding), not 2
        "ln_eps": 1e-5, "bn_eps": 1e-5,
        "pred_hidden": dec["pred_hidden"], "pred_rnn_layers": dec["pred_rnn_layers"],
        "joint_hidden": joint["joint_hidden"],
        "num_classes": config["joint"]["num_classes"],  # real tokens, blank NOT included yet
        "blank_id": config["joint"]["num_classes"],      # blank is the next id after all real tokens
        "durations": config["joint"]["jointnet"].get("durations") or [0, 1, 2, 3, 4],
    }


def build_encoder_topology(hp: dict) -> dict:
    nodes = []

    def node(op, inputs, outputs, attrs=None):
        n = {"op": op, "inputs": inputs, "outputs": outputs}
        if attrs is not None:
            n["attrs"] = attrs
        nodes.append(n)
        return outputs[0]

    def linear_nobias(x, w, out):
        return node("MUL_MAT", [w, x], [out])

    def layer_norm(x, w, b, out):
        ln = node("LAYER_NORM", [x], [out + "_ln"], {"eps": "$ln_eps"})
        scaled = node("MUL", [ln, w], [out + "_lnw"])
        return node("ADD", [scaled, b], [out])

    def half_step_ff(x, prefix, out):
        normed = layer_norm(x, f"blk.{{i}}.{prefix}.norm.weight", f"blk.{{i}}.{prefix}.norm.bias", f"{out}_normed")
        hidden_raw = linear_nobias(normed, f"blk.{{i}}.{prefix}.linear1.weight", f"{out}_hidden_raw")
        hidden = node("SILU", [hidden_raw], [f"{out}_hidden"])
        ff_out = linear_nobias(hidden, f"blk.{{i}}.{prefix}.linear2.weight", f"{out}_ff")
        return node("ADD", ["cur", ff_out], ["cur"])

    # ---- Mel-spectrogram front-end -- identical to convert_conformer_ctc.py's, just parametrized by
    # this checkpoint's own n_mels/hop_length (128/160 vs Conformer-CTC-small's 80/160). ----
    node("PAD_1D", ["waveform"], ["wav_padded"], {"lp0": 1, "rp0": 0})
    node("VIEW", ["wav_padded"], ["wav_prev"], {"shape": ["n_tokens", 1, 1]})
    node("VIEW", ["wav_padded"], ["wav_curr"], {"shape": ["n_tokens", 1, 1], "offset": 4})
    prev_scaled = node("SCALE", ["wav_prev"], ["wav_prev_scaled"], {"s": hp["preemph"]})
    node("SUB", ["wav_curr", prev_scaled], ["preemph_x"])

    stft_attrs = {"s0": hp["hop_length"], "p0": hp["stft_pad"], "d0": 1}
    node("CONV_1D", ["mel.cos_kernel", "preemph_x"], ["stft_cos"], stft_attrs)
    node("CONV_1D", ["mel.sin_kernel", "preemph_x"], ["stft_sin"], stft_attrs)
    node("SQR", ["stft_cos"], ["cos_sq"])
    node("SQR", ["stft_sin"], ["sin_sq"])
    node("ADD", ["cos_sq", "sin_sq"], ["power"])

    node("PERMUTE", ["power"], ["power_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["power_p"], ["power_t"])
    node("MUL_MAT", ["mel.filterbank", "power_t"], ["mel_raw"])
    guarded = node("ADD", ["mel_raw", "mel.log_guard"], ["mel_guarded"])
    log_mel = node("LOG", [guarded], ["log_mel"])

    node("PERMUTE", [log_mel], ["logmel_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["logmel_p"], ["logmel_t"])
    node("SUM_ROWS", ["logmel_t"], ["sum_t"])
    node("SCALE", ["sum_t"], ["mean_t"], {"s": f"1/({mel_common_t_mel_expr(hp)})"})
    node("RESHAPE", ["mean_t"], ["mean"], {"shape": [hp["n_mels"], 1, 1]})
    node("SUB", [log_mel, "mean"], ["centered"])
    node("PERMUTE", ["centered"], ["centered_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["centered_p"], ["centered_t"])
    node("SQR", ["centered_t"], ["centered_sq_t"])
    node("SUM_ROWS", ["centered_sq_t"], ["sumsq_t"])
    node("SCALE", ["sumsq_t"], ["var_t"], {"s": f"1/(({mel_common_t_mel_expr(hp)}) - 1)"})
    node("SQRT", ["var_t"], ["std_t"])
    node("ADD", ["std_t", "mel.norm_eps"], ["std_t_guarded"])
    node("RESHAPE", ["std_t_guarded"], ["std"], {"shape": [hp["n_mels"], 1, 1]})
    node("DIV", ["centered", "std"], ["normalized"])
    node("RESHAPE", ["normalized"], ["mel_input"], {"shape": [hp["n_mels"], -1, 1, 1]})

    # ---- Real dw_striding subsampling front-end (3 stages: plain Conv2d, then 2x [depthwise Conv2d +
    #      pointwise Conv2d], each followed by ReLU) -- genuinely different from Conformer-CTC-small's
    #      simpler 2-stage plain-Conv2d subsampling, confirmed against this checkpoint's real state dict
    #      (encoder.pre_encode.conv.{0,2,3,5,6}) and NeMo's own subsampling.py source. ----
    c = hp["subsampling_conv_channels"]
    node("CONV_2D", ["pre_encode.conv0.weight", "mel_input"], ["sub0_raw"],
         {"s0": 2, "s1": 2, "p0": 1, "p1": 1, "d0": 1, "d1": 1})
    bias0 = node("RESHAPE", ["pre_encode.conv0.bias"], ["conv0_bias_r"], {"shape": [1, 1, c, 1]})
    node("ADD", ["sub0_raw", bias0], ["sub0_biased"])
    node("RELU", ["sub0_biased"], ["sub0"])

    prev = "sub0"
    for stage in (1, 2):
        dw_idx = 2 if stage == 1 else 5
        pw_idx = 3 if stage == 1 else 6
        dw_raw = node("CONV_2D_DW", [f"pre_encode.conv{dw_idx}.weight", prev], [f"sub{stage}_dw_raw"],
                      {"s0": 2, "s1": 2, "p0": 1, "p1": 1, "d0": 1, "d1": 1})
        dw_bias = node("RESHAPE", [f"pre_encode.conv{dw_idx}.bias"], [f"sub{stage}_dw_bias_r"], {"shape": [1, 1, c, 1]})
        dw_biased = node("ADD", [dw_raw, dw_bias], [f"sub{stage}_dw_biased"])
        pw_raw = node("CONV_2D", [f"pre_encode.conv{pw_idx}.weight", dw_biased], [f"sub{stage}_pw_raw"],
                      {"s0": 1, "s1": 1, "p0": 0, "p1": 0, "d0": 1, "d1": 1})
        pw_bias = node("RESHAPE", [f"pre_encode.conv{pw_idx}.bias"], [f"sub{stage}_pw_bias_r"], {"shape": [1, 1, c, 1]})
        pw_biased = node("ADD", [pw_raw, pw_bias], [f"sub{stage}_pw_biased"])
        prev = node("RELU", [pw_biased], [f"sub{stage}"])

    node("PERMUTE", [prev], ["sub_perm"], {"axes": [0, 2, 1, 3]})
    node("CONT", ["sub_perm"], ["sub_cont"])
    node("RESHAPE", ["sub_cont"], ["sub_flat"], {"shape": [hp["flattened_subsample_dim"], -1]})
    # No xscale here (xscaling=false, confirmed real) -- unlike Conformer-CTC-small's pre_encode.out.
    linear_nobias("sub_flat", "pre_encode.out.weight", "sub_out_mm")
    node("ADD", ["sub_out_mm", "pre_encode.out.bias"], ["cur"])

    # ---- 24x Conformer layers (same structure as Conformer-CTC-small; NO bias anywhere here per this
    #      checkpoint's real use_bias=false) ----
    layer_nodes = []
    saved_nodes, nodes = nodes, layer_nodes
    half_step_ff("cur", "feed_forward1", "ff1")

    sa_normed = layer_norm("cur", "blk.{i}.norm_self_att.weight", "blk.{i}.norm_self_att.bias", "sa_normed")
    q_flat = linear_nobias(sa_normed, "blk.{i}.self_attn.linear_q.weight", "q_flat")
    k_flat = linear_nobias(sa_normed, "blk.{i}.self_attn.linear_k.weight", "k_flat")
    v_flat = linear_nobias(sa_normed, "blk.{i}.self_attn.linear_v.weight", "v_flat")
    node("RESHAPE", [q_flat], ["q"], {"shape": ["$head_dim", "$n_head", -1]})
    node("RESHAPE", [k_flat], ["k"], {"shape": ["$head_dim", "$n_head", -1]})
    node("RESHAPE", [v_flat], ["v"], {"shape": ["$head_dim", "$n_head", -1]})
    p_flat = node("MUL_MAT", ["blk.{i}.self_attn.linear_pos.weight", "pos_emb_raw"], ["p_flat"])
    node("RESHAPE", [p_flat], ["p"], {"shape": ["$head_dim", "$n_head", -1]})
    node("REL_POS_ATTENTION", ["q", "k", "v", "p", "blk.{i}.self_attn.pos_bias_u", "blk.{i}.self_attn.pos_bias_v", "kq_mask"],
         ["attn_ctx"], {"scale": "1/sqrt($head_dim)"})
    attn_proj = linear_nobias("attn_ctx", "blk.{i}.self_attn.linear_out.weight", "attn_proj")
    node("ADD", ["cur", attn_proj], ["cur"])

    conv_normed = layer_norm("cur", "blk.{i}.norm_conv.weight", "blk.{i}.norm_conv.bias", "conv_normed")
    node("PERMUTE", [conv_normed], ["conv_in_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["conv_in_p"], ["conv_in"])
    node("CONV_1D", ["blk.{i}.conv.pointwise_conv1.weight", "conv_in"], ["pw1_raw"], {"s0": 1, "p0": 0, "d0": 1})
    node("GLU", ["pw1_raw"], ["glu_out"])
    node("CONV_1D_DW", ["blk.{i}.conv.depthwise_conv.weight", "glu_out"], ["dw_raw"],
         {"s0": 1, "p0": "$conv_padding", "d0": 1})
    bn_scale = node("RESHAPE", ["blk.{i}.conv.batch_norm.scale"], ["bn_scale_r"], {"shape": [1, "$n_embd", 1]})
    bn_shift = node("RESHAPE", ["blk.{i}.conv.batch_norm.shift"], ["bn_shift_r"], {"shape": [1, "$n_embd", 1]})
    node("MUL", ["dw_raw", bn_scale], ["bn_scaled"])
    node("ADD", ["bn_scaled", bn_shift], ["bn_out"])
    node("SILU", ["bn_out"], ["swish_out"])
    node("CONV_1D", ["blk.{i}.conv.pointwise_conv2.weight", "swish_out"], ["pw2_raw"], {"s0": 1, "p0": 0, "d0": 1})
    node("PERMUTE", ["pw2_raw"], ["conv_result_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["conv_result_p"], ["conv_result"])
    node("ADD", ["cur", "conv_result"], ["cur"])

    half_step_ff("cur", "feed_forward2", "ff2")
    layer_norm("cur", "blk.{i}.norm_out.weight", "blk.{i}.norm_out.bias", "cur")

    nodes = saved_nodes
    nodes.append({"repeat_for": "$n_layers", "index_var": "i", "nodes": layer_nodes})

    return {
        "version": 1,
        "inputs": [
            {"name": "waveform", "dtype": "f32", "shape": ["n_tokens", "1", "1"]},
            {"name": "pos_emb_raw", "dtype": "f32", "shape": [str(hp["n_embd"]), n_pos_expr(hp)]},
            {"name": "kq_mask", "dtype": "f32", "shape": [n_subsampled_expr(hp), n_subsampled_expr(hp)]},
        ],
        "output": "cur",
        "nodes": nodes,
    }


def conv_stride_out_expr(in_expr: str, pad: int, kernel: int, stride: int) -> str:
    return f"floor((({in_expr}) + {2 * pad} - {kernel})/{stride}) + 1"


def mel_common_t_mel_expr(hp: dict) -> str:
    return f"floor($n_tokens/{hp['hop_length']}) + 1"


def n_subsampled_expr(hp: dict) -> str:
    t = mel_common_t_mel_expr(hp)
    for _ in range(hp["subsampling_num_layers"]):
        t = conv_stride_out_expr(t, 1, 3, 2)
    return t


def n_pos_expr(hp: dict) -> str:
    return f"2*({n_subsampled_expr(hp)}) - 1"


def build_lstm_topology(layer: int, output_name: str, pred_hidden: int) -> dict:
    """Same composite pattern as tools/fixture_gen/tdt_step_common.py's build_lstm_topology, real weight
    names (decoder.prediction.dec_rnn.lstm.{weight,bias}_{ih,hh}_l{layer})."""
    assert output_name in ("h_new", "c_new")
    h = pred_hidden
    f32 = 4
    p = f"decoder.prediction.dec_rnn.lstm."
    if layer == 0:
        inputs = [
            {"name": "last_label", "dtype": "i32", "shape": ["1"]},
            {"name": "h_prev", "dtype": "f32", "shape": [str(h)]},
            {"name": "c_prev", "dtype": "f32", "shape": [str(h)]},
        ]
        embed_nodes = [
            {"op": "GET_ROWS", "inputs": ["decoder.prediction.embed.weight", "last_label"], "outputs": ["embed_row"]},
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
    return {
        "version": 1,
        "inputs": inputs,
        "output": output_name,
        "nodes": embed_nodes + [
            {"op": "MUL_MAT", "inputs": [f"{p}weight_ih_l{layer}", "layer_input_resolved"], "outputs": ["gates_x"]},
            {"op": "MUL_MAT", "inputs": [f"{p}weight_hh_l{layer}", "h_prev"], "outputs": ["gates_h"]},
            {"op": "ADD", "inputs": ["gates_x", "gates_h"], "outputs": ["gates_sum"]},
            {"op": "ADD", "inputs": ["gates_sum", f"{p}bias_ih_l{layer}"], "outputs": ["gates_b1"]},
            {"op": "ADD", "inputs": ["gates_b1", f"{p}bias_hh_l{layer}"], "outputs": ["gates"]},
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


def build_joint_topology(n_embd: int, pred_hidden: int) -> dict:
    return {
        "version": 1,
        "inputs": [
            {"name": "encoder_frame", "dtype": "f32", "shape": [str(n_embd)]},
            {"name": "decoder_out", "dtype": "f32", "shape": [str(pred_hidden)]},
        ],
        "output": "combined",
        "nodes": [
            {"op": "MUL_MAT", "inputs": ["joint.enc.weight", "encoder_frame"], "outputs": ["f_proj_mm"]},
            {"op": "ADD", "inputs": ["f_proj_mm", "joint.enc.bias"], "outputs": ["f_proj"]},
            {"op": "MUL_MAT", "inputs": ["joint.pred.weight", "decoder_out"], "outputs": ["g_proj_mm"]},
            {"op": "ADD", "inputs": ["g_proj_mm", "joint.pred.bias"], "outputs": ["g_proj"]},
            {"op": "ADD", "inputs": ["f_proj", "g_proj"], "outputs": ["summed"]},
            {"op": "RELU", "inputs": ["summed"], "outputs": ["activated"]},
            {"op": "MUL_MAT", "inputs": ["joint.joint_net.2.weight", "activated"], "outputs": ["combined_mm"]},
            {"op": "ADD", "inputs": ["combined_mm", "joint.joint_net.2.bias"], "outputs": ["combined"]},
        ],
    }


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.nemo> <out_dir> [--n-samples N]", file=sys.stderr)
        sys.exit(1)
    nemo_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_samples = 16000  # 1.0s @ 16kHz, arbitrary default -- see note in convert_conformer_ctc.py
    if "--n-samples" in sys.argv:
        n_samples = int(sys.argv[sys.argv.index("--n-samples") + 1])

    config, state, tokenizer_model_bytes = common.load_nemo(nemo_path)
    hp = hparams(config)
    hp.update(mel_common.mel_hparams(hp["feat_in"]))
    hp["n_mels"] = hp["feat_in"]
    hp["n_tokens"] = n_samples

    t_mel = n_samples // hp["hop_length"] + 1
    f_dims = [hp["feat_in"]]
    for _ in range(hp["subsampling_num_layers"]):
        f_dims.append((f_dims[-1] + 2 * 1 - 3) // 2 + 1)
    hp["flattened_subsample_dim"] = hp["subsampling_conv_channels"] * f_dims[-1]
    t_dims = [t_mel]
    for _ in range(hp["subsampling_num_layers"]):
        t_dims.append((t_dims[-1] + 2 * 1 - 3) // 2 + 1)
    n_subsampled = t_dims[-1]
    hp["n_subsampled"] = n_subsampled
    hp["n_pos"] = 2 * n_subsampled - 1

    # --- Encoder GGUF ---
    w = GGUFWriter(str(out_dir / "parakeet_encoder.gguf"), "loom-parakeet-tdt-encoder")
    w.add_string("loom.architecture", "parakeet_tdt_encoder")
    for key in ("n_layers", "n_embd", "n_head", "head_dim", "ff_hidden", "conv_kernel_size", "conv_padding"):
        w.add_uint32(f"loom.{key}", hp[key])
    w.add_float32("loom.ln_eps", hp["ln_eps"])
    w.add_uint32("loom.n_samples", hp["n_tokens"])
    w.add_uint32("loom.n_subsampled", hp["n_subsampled"])
    w.add_uint32("loom.n_pos", hp["n_pos"])
    w.add_string("model.graph_topology", json.dumps(build_encoder_topology(hp)))
    if tokenizer_model_bytes is not None:
        tokenizer_common.write_sentencepiece_vocab(w, tokenizer_model_bytes)

    def put(name, tensor):
        arr = tensor.detach().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
        w.add_tensor(name, arr.astype(np.float32))

    cos_kernel, sin_kernel = mel_common.build_dft_kernels(hp["n_fft"], hp["win_length"])
    mel_fb = mel_common.build_mel_filterbank(hp["sample_rate"], hp["n_fft"], hp["n_mels"])
    put("mel.cos_kernel", cos_kernel)
    put("mel.sin_kernel", sin_kernel)
    put("mel.filterbank", mel_fb)
    put("mel.log_guard", np.array([hp["log_guard"]]))
    put("mel.norm_eps", np.array([hp["norm_eps"]]))

    put("pre_encode.conv0.weight", state["encoder.pre_encode.conv.0.weight"])
    put("pre_encode.conv0.bias", state["encoder.pre_encode.conv.0.bias"])
    put("pre_encode.conv2.weight", state["encoder.pre_encode.conv.2.weight"])
    put("pre_encode.conv2.bias", state["encoder.pre_encode.conv.2.bias"])
    put("pre_encode.conv3.weight", state["encoder.pre_encode.conv.3.weight"])
    put("pre_encode.conv3.bias", state["encoder.pre_encode.conv.3.bias"])
    put("pre_encode.conv5.weight", state["encoder.pre_encode.conv.5.weight"])
    put("pre_encode.conv5.bias", state["encoder.pre_encode.conv.5.bias"])
    put("pre_encode.conv6.weight", state["encoder.pre_encode.conv.6.weight"])
    put("pre_encode.conv6.bias", state["encoder.pre_encode.conv.6.bias"])
    put("pre_encode.out.weight", state["encoder.pre_encode.out.weight"])
    put("pre_encode.out.bias", state["encoder.pre_encode.out.bias"])

    for i in range(hp["n_layers"]):
        p = f"encoder.layers.{i}"
        put(f"blk.{i}.feed_forward1.norm.weight", state[f"{p}.norm_feed_forward1.weight"])
        put(f"blk.{i}.feed_forward1.norm.bias", state[f"{p}.norm_feed_forward1.bias"])
        put(f"blk.{i}.feed_forward1.linear1.weight", state[f"{p}.feed_forward1.linear1.weight"])
        put(f"blk.{i}.feed_forward1.linear2.weight", state[f"{p}.feed_forward1.linear2.weight"] * 0.5)

        put(f"blk.{i}.norm_self_att.weight", state[f"{p}.norm_self_att.weight"])
        put(f"blk.{i}.norm_self_att.bias", state[f"{p}.norm_self_att.bias"])
        put(f"blk.{i}.self_attn.pos_bias_u", state[f"{p}.self_attn.pos_bias_u"])
        put(f"blk.{i}.self_attn.pos_bias_v", state[f"{p}.self_attn.pos_bias_v"])
        for proj in ("q", "k", "v", "out", "pos"):
            put(f"blk.{i}.self_attn.linear_{proj}.weight", state[f"{p}.self_attn.linear_{proj}.weight"])

        put(f"blk.{i}.norm_conv.weight", state[f"{p}.norm_conv.weight"])
        put(f"blk.{i}.norm_conv.bias", state[f"{p}.norm_conv.bias"])
        put(f"blk.{i}.conv.pointwise_conv1.weight", state[f"{p}.conv.pointwise_conv1.weight"])
        put(f"blk.{i}.conv.depthwise_conv.weight", state[f"{p}.conv.depthwise_conv.weight"])
        bn_scale, bn_shift = common.fold_batchnorm(state, f"{p}.conv.batch_norm", hp["bn_eps"])
        put(f"blk.{i}.conv.batch_norm.scale", bn_scale)
        put(f"blk.{i}.conv.batch_norm.shift", bn_shift)
        put(f"blk.{i}.conv.pointwise_conv2.weight", state[f"{p}.conv.pointwise_conv2.weight"])

        put(f"blk.{i}.feed_forward2.norm.weight", state[f"{p}.norm_feed_forward2.weight"])
        put(f"blk.{i}.feed_forward2.norm.bias", state[f"{p}.norm_feed_forward2.bias"])
        put(f"blk.{i}.feed_forward2.linear1.weight", state[f"{p}.feed_forward2.linear1.weight"])
        put(f"blk.{i}.feed_forward2.linear2.weight", state[f"{p}.feed_forward2.linear2.weight"] * 0.5)

        put(f"blk.{i}.norm_out.weight", state[f"{p}.norm_out.weight"])
        put(f"blk.{i}.norm_out.bias", state[f"{p}.norm_out.bias"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_dir / 'parakeet_encoder.gguf'}")

    # --- LSTM (per-layer h/c) + joint GGUFs ---
    def write_small(path, topology, tensor_names):
        ww = GGUFWriter(str(path), "loom-parakeet-tdt-decoder")
        ww.add_string("loom.architecture", "parakeet_tdt_decoder")
        ww.add_string("model.graph_topology", json.dumps(topology))
        for name in tensor_names:
            arr = state[name].detach().numpy() if hasattr(state[name], "detach") else np.asarray(state[name])
            ww.add_tensor(name, arr.astype(np.float32))
        ww.write_header_to_file()
        ww.write_kv_data_to_file()
        ww.write_tensors_to_file()
        ww.close()

    # TdtDecoder uses ONE shared GgufModel for every one of its internal GraphBuilders (lstm_h/lstm_c per
    # layer + joint) -- confirmed the hard way (a real "unresolved input 'joint.enc.weight'" crash) that
    # this means every one of these small GGUFs needs ALL decoder+joint tensors, not just the ones its own
    # topology references, exactly mirroring tools/fixture_gen/tdt_step_common.py's synthetic-fixture
    # convention (write_one() there also writes the full weight set into every file). A few tens of MB
    # duplicated 5x is negligible next to the ~2.4GB encoder checkpoint.
    all_decoder_joint_tensor_names = ["decoder.prediction.embed.weight"]
    for layer in range(hp["pred_rnn_layers"]):
        for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            all_decoder_joint_tensor_names.append(f"decoder.prediction.dec_rnn.lstm.{kind}_l{layer}")
    all_decoder_joint_tensor_names += ["joint.enc.weight", "joint.enc.bias", "joint.pred.weight", "joint.pred.bias",
                                       "joint.joint_net.2.weight", "joint.joint_net.2.bias"]

    for layer in range(hp["pred_rnn_layers"]):
        write_small(out_dir / f"parakeet_lstm_h_{layer}.gguf",
                    build_lstm_topology(layer, "h_new", hp["pred_hidden"]), all_decoder_joint_tensor_names)
        write_small(out_dir / f"parakeet_lstm_c_{layer}.gguf",
                    build_lstm_topology(layer, "c_new", hp["pred_hidden"]), all_decoder_joint_tensor_names)

    write_small(out_dir / "parakeet_joint.gguf", build_joint_topology(hp["n_embd"], hp["pred_hidden"]),
                all_decoder_joint_tensor_names)

    print(f"wrote {hp['pred_rnn_layers']} LSTM layer(s) + joint GGUFs to {out_dir}")
    print(f"hparams: n_layers={hp['n_layers']} n_embd={hp['n_embd']} pred_hidden={hp['pred_hidden']} "
          f"pred_rnn_layers={hp['pred_rnn_layers']} blank_id={hp['blank_id']} durations={hp['durations']}")


if __name__ == "__main__":
    main()
