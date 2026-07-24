#!/usr/bin/env python3
"""Converts an NVIDIA NeMo Conformer-CTC checkpoint (a .nemo file, e.g.
nvidia/stt_en_conformer_ctc_small) into a loom-engine GGUF file: weights + hyperparameters under the
"loom." KV namespace + the JSON graph topology under "model.graph_topology".

Scope (see BACKLOG.md): the Conformer encoder + CTC decoder, AND mel-spectrogram extraction (STFT +
power + mel filterbank + log + per-feature CMVN normalize, matching NeMo's AudioToMelSpectrogramPreprocessor
exactly). The declared runtime input is now "waveform" (raw 16kHz PCM samples), not a precomputed mel
tensor -- mel_common.py builds the STFT's DFT basis kernels and the mel filterbank as constant tensors
baked into the GGUF, so the whole preprocessing pipeline runs inside the ggml graph. STFT-via-convolution
(cross-correlating framed audio against precomputed cos/sin DFT-basis kernels) is the same trick used by
ONNX-exportable audio frontends since torch.stft itself isn't graph-friendly; ggml's CONV_1D's own
zero-padding matches NeMo's stft(center=True, pad_mode="constant") exactly, so no reflect-pad primitive
is needed. The checkpoint's SentencePiece unigram vocab is also written into the GGUF, using llama.cpp's
own "tokenizer.ggml.*" KV schema (see tokenizer_common.py) -- CTC greedy decode + detokenization happen
host-side (loom::ctc_greedy_decode + loom::Vocab), not as graph nodes, same "host logic, not a graph
primitive" precedent as everything else non-tensor-shaped in this engine.

Sequence length is genuinely dynamic, per SPECIFICATION.md §4 ("rebuilding the compute graph from
scratch for every forward pass... injecting the exact dimensions"): "waveform"'s shape was always
`["n_tokens", "1", "1"]` (n_tokens = raw sample count, GraphBuilder's one true runtime symbol), and
"pos_emb_raw"/"kq_mask" (whose sizes depend on n_subsampled, a non-trivial derived function of
n_tokens via the mel-frontend's STFT-conv stride then the Conformer's own two subsampling convs) are
now ALSO full SymbolEnv expressions in terms of "$n_tokens" -- see n_subsampled_expr()/n_pos_expr()
below -- evaluated fresh on every GraphBuilder::build() call, not hardcoded literals computed once
here. hp["n_subsampled"]/hp["n_pos"]/hp["n_tokens"] (the Python numbers) are kept only for the
loom.n_subsampled/loom.n_pos/loom.n_samples hparam KVs, which now describe just the *default* length
used to regenerate the bundled test fixture, not a hard constraint on real usage.

Three kinds of constants get folded/baked directly at conversion time, matching this project's
established "fold constants once, avoid needing a new primitive" precedent (see BACKLOG.md's
BatchNorm-folding note from Milestone 3):
  - BatchNorm (eval mode has no batch-statistics dependency) folds to a per-channel scale+shift.
  - The Conformer's half-step feed-forward residual (0.5x) and xscale (sqrt(d_model), applied once right
    after subsampling) both fold into the weight/bias of the linear layer that immediately precedes
    them, since scaling commutes through a Linear layer's output exactly.
  - The mel frontend's DFT basis (cos/sin kernels) and mel filterbank matrix are pure functions of fixed
    hyperparameters (sample rate, n_fft, window, n_mels) -- computed once in Python (mel_common.py) and
    baked in as ordinary constant GGUF weight tensors, never touched by the checkpoint's own state dict.

Requires: pip install torch pyyaml gguf numpy librosa sentencepiece
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mel_common
import nemo_common as common
import tokenizer_common


def conv_stride_out_expr(in_expr: str, pad: int, kernel: int, stride: int) -> str:
    """A SymbolEnv expression string for the standard conv output-length formula
    floor((in + 2*pad - kernel)/stride) + 1, applied to another expression string (not a number) so
    these compose into a single nested formula. Needs SymbolEnv's floor() (added alongside this
    change): the halfway values this formula routinely produces for even-length inputs would otherwise
    get rounded the WRONG way by GraphBuilder's outer std::llround (rounds away from zero, not down)."""
    return f"floor((({in_expr}) + {2 * pad} - {kernel})/{stride}) + 1"


def t_mel_expr(hp: dict) -> str:
    """Mel-frontend STFT frame count as a function of $n_tokens (raw waveform samples). Reduces to
    floor($n_tokens/hop_length) + 1 because 2*stft_pad == n_fft exactly by construction (stft_pad =
    n_fft//2) -- same simplification already used when this was computed as a plain Python number."""
    return f"floor($n_tokens/{hp['hop_length']}) + 1"


def valid_frames_expr(hp: dict) -> str:
    """Real NeMo's own CMVN-valid frame count (see mel_common.py's "Real-NeMo gotcha" docstring note):
    NeMo's `get_seq_len` computes floor($n_tokens/hop_length), exactly t_mel_expr(hp) - 1 -- ALWAYS one
    less than the real STFT frame count, even for a full-length utterance with no true padding. CMVN
    mean/variance must reduce over only this many leading frames, and the final normalized frame at index
    t_mel-1 must be zeroed, or the output silently diverges from real NeMo (confirmed against the actual
    checkpoint's nemo_asr preprocessor output -- this is a distinct bug from the encoder's own
    calc_length/all_paddings masking, which genuinely is a no-op here)."""
    return f"floor($n_tokens/{hp['hop_length']})"


def n_subsampled_expr(hp: dict) -> str:
    """Post-subsampling encoder frame count as a function of $n_tokens: the mel-frame count above, run
    through the Conformer's own two stride-2/pad-1/kernel-3 subsampling convs (time axis)."""
    t1 = conv_stride_out_expr(t_mel_expr(hp), 1, 3, 2)
    return conv_stride_out_expr(t1, 1, 3, 2)


def n_pos_expr(hp: dict) -> str:
    return f"2*({n_subsampled_expr(hp)}) - 1"


def build_topology(hp: dict) -> dict:
    nodes = []

    def node(op, inputs, outputs, attrs=None):
        n = {"op": op, "inputs": inputs, "outputs": outputs}
        if attrs is not None:
            n["attrs"] = attrs
        nodes.append(n)
        return outputs[0]

    def linear(x, w, b, out):
        mm = node("MUL_MAT", [w, x], [out + "_mm"])
        return node("ADD", [mm, b], [out])

    def layer_norm(x, w, b, out):
        ln = node("LAYER_NORM", [x], [out + "_ln"], {"eps": "$ln_eps"})
        scaled = node("MUL", [ln, w], [out + "_lnw"])
        return node("ADD", [scaled, b], [out])

    def broadcast_bias_reshape(bias_name, channels, out):
        # Reshapes a plain 1D [channels] bias to [1, channels, 1] so it broadcasts over CONV_1D's
        # ne=[T, channels, N] output on the channel axis (ne[1]), not the length axis (ne[0]).
        return node("RESHAPE", [bias_name], [out], {"shape": [1, channels, 1]})

    def broadcast_bias_reshape_2d(bias_name, channels, out):
        # CONV_2D's output is ne=[OW, OH, OC, N] (channels at ne[2], not ne[1] like CONV_1D) -- see
        # op_conv_2d's final PERMUTE to [OW,OH,OC,N]. Needed only for the two subsampling conv2d biases.
        return node("RESHAPE", [bias_name], [out], {"shape": [1, 1, channels, 1]})

    def half_step_ff(x, prefix, out):
        # linear2's weight/bias are pre-scaled by 0.5 at conversion time (see module docstring), so this
        # is just LN -> Linear -> Swish -> Linear -> residual-add, no separate scale node needed.
        normed = layer_norm(x, f"blk.{{i}}.{prefix}.norm.weight", f"blk.{{i}}.{prefix}.norm.bias", f"{out}_normed")
        hidden_raw = linear(normed, f"blk.{{i}}.{prefix}.linear1.weight", f"blk.{{i}}.{prefix}.linear1.bias", f"{out}_hidden_raw")
        hidden = node("SILU", [hidden_raw], [f"{out}_hidden"])
        ff_out = linear(hidden, f"blk.{{i}}.{prefix}.linear2.weight", f"blk.{{i}}.{prefix}.linear2.bias", f"{out}_ff")
        return node("ADD", ["cur", ff_out], ["cur"])

    # ---- Mel-spectrogram front-end (preemphasis -> STFT-via-CONV_1D -> power -> mel filterbank -> log
    #      -> per-feature CMVN normalize), matching NeMo's FilterbankFeatures exactly. Produces "mel_input"
    #      with the same [n_mels, T_mel, 1, 1] shape/name the rest of the topology already expects. ----
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
    node("ADD", ["cos_sq", "sin_sq"], ["power"])  # [T_mel, n_freq, 1]

    node("PERMUTE", ["power"], ["power_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["power_p"], ["power_t"])  # [n_freq, T_mel, 1]
    node("MUL_MAT", ["mel.filterbank", "power_t"], ["mel_raw"])  # [n_mels, T_mel, 1]
    guarded = node("ADD", ["mel_raw", "mel.log_guard"], ["mel_guarded"])
    log_mel = node("LOG", [guarded], ["log_mel"])

    # per-feature (per-mel-bin) CMVN over the time axis: unbiased variance, epsilon-guarded std.
    # Real NeMo treats the LAST STFT frame as invalid -- excluded from mean/var, zeroed in the final
    # output (see valid_frames_expr()'s own docstring). "valid" below means "the first T_mel-1 frames".
    node("PERMUTE", [log_mel], ["logmel_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["logmel_p"], ["logmel_t"])  # [T_mel, n_mels, 1]
    node("VIEW", ["logmel_t"], ["logmel_valid_t"], {"shape": [valid_frames_expr(hp), hp["n_mels"], 1]})
    node("SUM_ROWS", ["logmel_valid_t"], ["sum_t"])
    # T_mel/valid-frame-count are functions of the runtime $n_tokens (see t_mel_expr/valid_frames_expr),
    # NOT the conversion-time default -- these scale factors must be symbol expressions, not fixed Python
    # numbers baked in, or CMVN normalize silently divides by the wrong count for every length but the
    # default one.
    node("SCALE", ["sum_t"], ["mean_t"], {"s": f"1/({valid_frames_expr(hp)})"})
    node("RESHAPE", ["mean_t"], ["mean"], {"shape": [hp["n_mels"], 1, 1]})
    node("SUB", [log_mel, "mean"], ["centered"])  # [n_mels, T_mel, 1]
    node("PERMUTE", ["centered"], ["centered_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["centered_p"], ["centered_t"])  # [T_mel, n_mels, 1]
    node("VIEW", ["centered_t"], ["centered_valid_t"], {"shape": [valid_frames_expr(hp), hp["n_mels"], 1]})
    node("SQR", ["centered_valid_t"], ["centered_sq_t"])
    node("SUM_ROWS", ["centered_sq_t"], ["sumsq_t"])
    node("SCALE", ["sumsq_t"], ["var_t"], {"s": f"1/(({valid_frames_expr(hp)}) - 1)"})
    node("SQRT", ["var_t"], ["std_t"])
    node("ADD", ["std_t", "mel.norm_eps"], ["std_t_guarded"])
    node("RESHAPE", ["std_t_guarded"], ["std"], {"shape": [hp["n_mels"], 1, 1]})
    node("DIV", ["centered", "std"], ["normalized"])  # [n_mels, T_mel, 1]
    # Zero the last (structurally-always-invalid, see above) frame: permute T_mel back to ne[0], drop it,
    # zero-pad it back with PAD_1D, permute back.
    node("PERMUTE", ["normalized"], ["normalized_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["normalized_p"], ["normalized_t"])  # [T_mel, n_mels, 1]
    node("VIEW", ["normalized_t"], ["normalized_valid_t"], {"shape": [valid_frames_expr(hp), hp["n_mels"], 1]})
    node("PAD_1D", ["normalized_valid_t"], ["normalized_padded_t"], {"lp0": 0, "rp0": 1})  # [T_mel, n_mels, 1]
    node("PERMUTE", ["normalized_padded_t"], ["normalized_final_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["normalized_final_p"], ["normalized_final"])  # [n_mels, T_mel, 1]
    node("RESHAPE", ["normalized_final"], ["mel_input"], {"shape": [hp["n_mels"], -1, 1, 1]})

    # ---- Subsampling front-end (CONV_2D x2 + ReLU, "same"-ish striding subsampling) ----
    node("CONV_2D", ["pre_encode.conv0.weight", "mel_input"], ["sub0_raw"],
         {"s0": 2, "s1": 2, "p0": 1, "p1": 1, "d0": 1, "d1": 1})
    bias0 = broadcast_bias_reshape_2d("pre_encode.conv0.bias", hp["subsampling_conv_channels"], "conv0_bias_r")
    node("ADD", ["sub0_raw", bias0], ["sub0_biased"])
    node("RELU", ["sub0_biased"], ["sub0"])

    node("CONV_2D", ["pre_encode.conv1.weight", "sub0"], ["sub1_raw"],
         {"s0": 2, "s1": 2, "p0": 1, "p1": 1, "d0": 1, "d1": 1})
    bias1 = broadcast_bias_reshape_2d("pre_encode.conv1.bias", hp["subsampling_conv_channels"], "conv1_bias_r")
    node("ADD", ["sub1_raw", bias1], ["sub1_biased"])
    node("RELU", ["sub1_biased"], ["sub1"])

    # ne=[F,T',C,1] -> permute to [F,C,T',1] (channel-slower/freq-faster flatten, matching NeMo's own
    # transpose(1,2).reshape(b,t,-1)) -> flatten (F,C) -> [F*C, T'] -> Linear -> [n_embd, T'].
    node("PERMUTE", ["sub1"], ["sub1_perm"], {"axes": [0, 2, 1, 3]})
    node("CONT", ["sub1_perm"], ["sub1_cont"])
    node("RESHAPE", ["sub1_cont"], ["sub1_flat"], {"shape": [hp["flattened_subsample_dim"], -1]})
    # pre_encode.out's weight/bias are pre-scaled by xscale=sqrt(n_embd) at conversion time (module
    # docstring) -- this IS the encoder's running "cur", already xscaled.
    linear("sub1_flat", "pre_encode.out.weight", "pre_encode.out.bias", "cur")

    # ---- 16x Conformer layers ----
    layer_nodes = []
    saved_nodes, nodes = nodes, layer_nodes  # temporarily redirect `node()` into a per-layer list
    half_step_ff("cur", "feed_forward1", "ff1")

    # Self-attention (full residual).
    sa_normed = layer_norm("cur", "blk.{i}.norm_self_att.weight", "blk.{i}.norm_self_att.bias", "sa_normed")
    q_flat = linear(sa_normed, "blk.{i}.self_attn.linear_q.weight", "blk.{i}.self_attn.linear_q.bias", "q_flat")
    k_flat = linear(sa_normed, "blk.{i}.self_attn.linear_k.weight", "blk.{i}.self_attn.linear_k.bias", "k_flat")
    v_flat = linear(sa_normed, "blk.{i}.self_attn.linear_v.weight", "blk.{i}.self_attn.linear_v.bias", "v_flat")
    node("RESHAPE", [q_flat], ["q"], {"shape": ["$head_dim", "$n_head", -1]})
    node("RESHAPE", [k_flat], ["k"], {"shape": ["$head_dim", "$n_head", -1]})
    node("RESHAPE", [v_flat], ["v"], {"shape": ["$head_dim", "$n_head", -1]})
    p_flat = node("MUL_MAT", ["blk.{i}.self_attn.linear_pos.weight", "pos_emb_raw"], ["p_flat"])
    node("RESHAPE", [p_flat], ["p"], {"shape": ["$head_dim", "$n_head", -1]})
    node("REL_POS_ATTENTION", ["q", "k", "v", "p", "blk.{i}.self_attn.pos_bias_u", "blk.{i}.self_attn.pos_bias_v", "kq_mask"],
         ["attn_ctx"], {"scale": "1/sqrt($head_dim)"})
    attn_proj = linear("attn_ctx", "blk.{i}.self_attn.linear_out.weight", "blk.{i}.self_attn.linear_out.bias", "attn_proj")
    node("ADD", ["cur", attn_proj], ["cur"])

    # Conv module (full residual): LN -> transpose to [T,C] -> pointwise/GLU/depthwise/BN/Swish/pointwise -> transpose back.
    conv_normed = layer_norm("cur", "blk.{i}.norm_conv.weight", "blk.{i}.norm_conv.bias", "conv_normed")
    node("PERMUTE", [conv_normed], ["conv_in_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["conv_in_p"], ["conv_in"])
    node("CONV_1D", ["blk.{i}.conv.pointwise_conv1.weight", "conv_in"], ["pw1_raw"], {"s0": 1, "p0": 0, "d0": 1})
    pw1_bias = broadcast_bias_reshape("blk.{i}.conv.pointwise_conv1.bias", 2 * hp["n_embd"], "pw1_bias_r")
    node("ADD", ["pw1_raw", pw1_bias], ["pw1_biased"])
    node("GLU", ["pw1_biased"], ["glu_out"])
    node("CONV_1D_DW", ["blk.{i}.conv.depthwise_conv.weight", "glu_out"], ["dw_raw"],
         {"s0": 1, "p0": "$conv_padding", "d0": 1})
    dw_bias = broadcast_bias_reshape("blk.{i}.conv.depthwise_conv.bias", hp["n_embd"], "dw_bias_r")
    node("ADD", ["dw_raw", dw_bias], ["dw_out"])
    bn_scale = broadcast_bias_reshape("blk.{i}.conv.batch_norm.scale", hp["n_embd"], "bn_scale_r")
    bn_shift = broadcast_bias_reshape("blk.{i}.conv.batch_norm.shift", hp["n_embd"], "bn_shift_r")
    node("MUL", ["dw_out", bn_scale], ["bn_scaled"])
    node("ADD", ["bn_scaled", bn_shift], ["bn_out"])
    node("SILU", ["bn_out"], ["swish_out"])
    node("CONV_1D", ["blk.{i}.conv.pointwise_conv2.weight", "swish_out"], ["pw2_raw"], {"s0": 1, "p0": 0, "d0": 1})
    pw2_bias = broadcast_bias_reshape("blk.{i}.conv.pointwise_conv2.bias", hp["n_embd"], "pw2_bias_r")
    node("ADD", ["pw2_raw", pw2_bias], ["pw2_biased"])
    node("PERMUTE", ["pw2_biased"], ["conv_result_p"], {"axes": [1, 0, 2, 3]})
    node("CONT", ["conv_result_p"], ["conv_result"])
    node("ADD", ["cur", "conv_result"], ["cur"])

    half_step_ff("cur", "feed_forward2", "ff2")
    layer_norm("cur", "blk.{i}.norm_out.weight", "blk.{i}.norm_out.bias", "cur")

    nodes = saved_nodes  # restore
    nodes.append({"repeat_for": "$n_layers", "index_var": "i", "nodes": layer_nodes})

    # ---- CTC decoder ----
    linear("cur", "decoder.weight", "decoder.bias", "logits")

    return {
        "version": 1,
        "inputs": [
            {"name": "waveform", "dtype": "f32", "shape": ["n_tokens", "1", "1"]},
            {"name": "pos_emb_raw", "dtype": "f32", "shape": [str(hp["n_embd"]), n_pos_expr(hp)]},
            {"name": "kq_mask", "dtype": "f32", "shape": [n_subsampled_expr(hp), n_subsampled_expr(hp)]},
        ],
        "output": "logits",
        "nodes": nodes,
    }


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.nemo> <out.gguf> [--n-samples N]", file=sys.stderr)
        sys.exit(1)
    nemo_path, out_path = sys.argv[1], sys.argv[2]
    # n_samples is NOT a hard constraint on the converted GGUF anymore -- pos_emb_raw/kq_mask's shapes
    # are now $n_tokens expressions (see n_subsampled_expr()/n_pos_expr()), evaluated fresh for whatever
    # length GraphBuilder::build() is actually called with. This value only sizes: (1) the
    # loom.n_samples/n_subsampled/n_pos hparam KVs (informative defaults, e.g. for a caller that wants a
    # sensible starting point), and (2) the reference fixture reference_forward_conformer.py generates
    # for testing. loom_cli --wav uses the real audio's own sample count directly.
    n_samples = 10240  # 0.64s @ 16kHz; gives t_mel=65, n_subsampled=17 (unchanged from the prior milestone).
    if "--n-samples" in sys.argv:
        n_samples = int(sys.argv[sys.argv.index("--n-samples") + 1])

    config, state, tokenizer_model_bytes = common.load_nemo(nemo_path)
    hp = common.hparams(config)
    hp.update(mel_common.mel_hparams(hp["feat_in"]))
    hp["n_mels"] = hp["feat_in"]
    hp["n_tokens"] = n_samples
    # floor((n_samples + 2*stft_pad - n_fft)/hop_length) + 1; 2*stft_pad == n_fft exactly (stft_pad =
    # n_fft//2), so this reduces to floor(n_samples/hop_length)+1 -- the same im2col output-length formula
    # CONV_1D itself uses internally.
    t_mel = n_samples // hp["hop_length"] + 1
    hp["t_mel"] = t_mel
    # Fixed by feat_in=80 and the two stride-2/pad-1/kernel-3 conv2d layers, independent of n_tokens.
    f1 = (hp["feat_in"] + 2 * 1 - 3) // 2 + 1
    f2 = (f1 + 2 * 1 - 3) // 2 + 1
    hp["flattened_subsample_dim"] = hp["subsampling_conv_channels"] * f2
    # Same stride-2/pad-1/kernel-3 formula, applied to the time axis instead of frequency.
    t1 = (t_mel + 2 * 1 - 3) // 2 + 1
    n_subsampled = (t1 + 2 * 1 - 3) // 2 + 1
    hp["n_subsampled"] = n_subsampled
    hp["n_pos"] = 2 * n_subsampled - 1

    xscale = float(np.sqrt(hp["n_embd"]))

    w = GGUFWriter(out_path, "loom-conformer-ctc")
    w.add_string("loom.architecture", "conformer_ctc")
    for key in ("n_layers", "n_embd", "n_head", "head_dim", "ff_hidden", "conv_kernel_size", "conv_padding"):
        w.add_uint32(f"loom.{key}", hp[key])
    w.add_float32("loom.ln_eps", hp["ln_eps"])
    # Fixed conversion-time shapes, stored as real KVs (not just baked into the topology JSON) so callers
    # like loom_cli can read them back instead of hardcoding them -- mirrors llama.cpp's practice of
    # storing hparams as first-class KVs.
    w.add_uint32("loom.n_samples", hp["n_tokens"])
    w.add_uint32("loom.n_subsampled", hp["n_subsampled"])
    w.add_uint32("loom.n_pos", hp["n_pos"])
    w.add_uint32("loom.num_classes", hp["num_classes"])
    w.add_string("model.graph_topology", __import__("json").dumps(build_topology(hp)))

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
    put("pre_encode.conv1.weight", state["encoder.pre_encode.conv.2.weight"])
    put("pre_encode.conv1.bias", state["encoder.pre_encode.conv.2.bias"])
    put("pre_encode.out.weight", state["encoder.pre_encode.out.weight"] * xscale)
    put("pre_encode.out.bias", state["encoder.pre_encode.out.bias"] * xscale)
    put("decoder.weight", state["decoder.decoder_layers.0.weight"].squeeze(-1))
    put("decoder.bias", state["decoder.decoder_layers.0.bias"])

    for i in range(hp["n_layers"]):
        p = f"encoder.layers.{i}"
        put(f"blk.{i}.feed_forward1.norm.weight", state[f"{p}.norm_feed_forward1.weight"])
        put(f"blk.{i}.feed_forward1.norm.bias", state[f"{p}.norm_feed_forward1.bias"])
        put(f"blk.{i}.feed_forward1.linear1.weight", state[f"{p}.feed_forward1.linear1.weight"])
        put(f"blk.{i}.feed_forward1.linear1.bias", state[f"{p}.feed_forward1.linear1.bias"])
        put(f"blk.{i}.feed_forward1.linear2.weight", state[f"{p}.feed_forward1.linear2.weight"] * 0.5)
        put(f"blk.{i}.feed_forward1.linear2.bias", state[f"{p}.feed_forward1.linear2.bias"] * 0.5)

        put(f"blk.{i}.norm_self_att.weight", state[f"{p}.norm_self_att.weight"])
        put(f"blk.{i}.norm_self_att.bias", state[f"{p}.norm_self_att.bias"])
        put(f"blk.{i}.self_attn.pos_bias_u", state[f"{p}.self_attn.pos_bias_u"])
        put(f"blk.{i}.self_attn.pos_bias_v", state[f"{p}.self_attn.pos_bias_v"])
        for proj in ("q", "k", "v", "out"):
            put(f"blk.{i}.self_attn.linear_{proj}.weight", state[f"{p}.self_attn.linear_{proj}.weight"])
            put(f"blk.{i}.self_attn.linear_{proj}.bias", state[f"{p}.self_attn.linear_{proj}.bias"])
        put(f"blk.{i}.self_attn.linear_pos.weight", state[f"{p}.self_attn.linear_pos.weight"])

        put(f"blk.{i}.norm_conv.weight", state[f"{p}.norm_conv.weight"])
        put(f"blk.{i}.norm_conv.bias", state[f"{p}.norm_conv.bias"])
        put(f"blk.{i}.conv.pointwise_conv1.weight", state[f"{p}.conv.pointwise_conv1.weight"])
        put(f"blk.{i}.conv.pointwise_conv1.bias", state[f"{p}.conv.pointwise_conv1.bias"])
        put(f"blk.{i}.conv.depthwise_conv.weight", state[f"{p}.conv.depthwise_conv.weight"])
        put(f"blk.{i}.conv.depthwise_conv.bias", state[f"{p}.conv.depthwise_conv.bias"])
        bn_scale, bn_shift = common.fold_batchnorm(state, f"{p}.conv.batch_norm", hp["bn_eps"])
        put(f"blk.{i}.conv.batch_norm.scale", bn_scale)
        put(f"blk.{i}.conv.batch_norm.shift", bn_shift)
        put(f"blk.{i}.conv.pointwise_conv2.weight", state[f"{p}.conv.pointwise_conv2.weight"])
        put(f"blk.{i}.conv.pointwise_conv2.bias", state[f"{p}.conv.pointwise_conv2.bias"])

        put(f"blk.{i}.feed_forward2.norm.weight", state[f"{p}.norm_feed_forward2.weight"])
        put(f"blk.{i}.feed_forward2.norm.bias", state[f"{p}.norm_feed_forward2.bias"])
        put(f"blk.{i}.feed_forward2.linear1.weight", state[f"{p}.feed_forward2.linear1.weight"])
        put(f"blk.{i}.feed_forward2.linear1.bias", state[f"{p}.feed_forward2.linear1.bias"])
        put(f"blk.{i}.feed_forward2.linear2.weight", state[f"{p}.feed_forward2.linear2.weight"] * 0.5)
        put(f"blk.{i}.feed_forward2.linear2.bias", state[f"{p}.feed_forward2.linear2.bias"] * 0.5)

        put(f"blk.{i}.norm_out.weight", state[f"{p}.norm_out.weight"])
        put(f"blk.{i}.norm_out.bias", state[f"{p}.norm_out.bias"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
