#!/usr/bin/env python3
"""Independent plain-PyTorch reimplementation of the real Parakeet-TDT-0.6B-v3 forward pass -- mel
frontend, FastConformer encoder (dw_striding subsampling, no bias, no xscale), LSTM prediction network
(2 stacked layers), joint network, and greedy TDT decode -- used as the ground truth
test_e2e_parakeet_tdt.cpp compares loom-engine's C++ output (encoder + the new TdtDecoder driver) against.

Deliberately hand-rolled rather than using nemo_toolkit or transformers (both are broken in this venv --
a huggingface_hub/transformers version conflict -- confirmed before choosing this path, not assumed) --
same "avoid the heavier dependency chain, read weights directly" precedent as every other reference script
in this project.

Requires: pip install torch pyyaml numpy librosa
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mel_common
import nemo_common as common
from convert_parakeet_tdt import hparams


def compute_mel_features(waveform: np.ndarray, hp: dict) -> np.ndarray:
    """Same algorithm as reference_forward_conformer.py's compute_mel_features, parametrized by this
    checkpoint's own n_mels=128 (mel_common.py's functions are already n_mels-generic)."""
    x = torch.from_numpy(waveform).unsqueeze(0).double()
    x = torch.cat((x[:, :1], x[:, 1:] - hp["preemph"] * x[:, :-1]), dim=1)

    window = torch.hann_window(hp["win_length"], periodic=False, dtype=torch.float64)
    stft = torch.stft(x, n_fft=hp["n_fft"], hop_length=hp["hop_length"], win_length=hp["win_length"],
                       window=window, center=True, pad_mode="constant", return_complex=True)
    power = stft.real ** 2 + stft.imag ** 2

    fb = torch.from_numpy(mel_common.build_mel_filterbank(hp["sample_rate"], hp["n_fft"], hp["n_mels"]).astype(np.float64))
    mel = torch.matmul(fb.unsqueeze(0), power)
    log_mel = torch.log(mel + hp["log_guard"])

    mean = log_mel.mean(dim=2, keepdim=True)
    var = ((log_mel - mean) ** 2).sum(dim=2, keepdim=True) / (log_mel.shape[2] - 1)
    std = torch.sqrt(var) + hp["norm_eps"]
    normalized = (log_mel - mean) / std

    return normalized.squeeze(0).transpose(0, 1).to(torch.float32).numpy()  # (t_mel, n_mels)


def sinusoidal_pos_emb(n_subsampled: int, n_embd: int) -> np.ndarray:
    length = n_subsampled
    positions = np.arange(length - 1, -length, -1, dtype=np.float64)
    div_term = np.exp(np.arange(0, n_embd, 2, dtype=np.float64) * -(math.log(10000.0) / n_embd))
    pe = np.zeros((2 * length - 1, n_embd), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions[:, None] * div_term[None, :])
    pe[:, 1::2] = np.cos(positions[:, None] * div_term[None, :])
    return pe


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    b, h, qlen, pos_len = x.size()
    x = F.pad(x, pad=(1, 0))
    x = x.view(b, h, -1, qlen)
    x = x[:, :, 1:].view(b, h, qlen, pos_len)
    return x


def rel_pos_mhsa(x: torch.Tensor, pos_emb: torch.Tensor, state: dict, prefix: str, hp: dict) -> torch.Tensor:
    n_head, head_dim = hp["n_head"], hp["head_dim"]
    b, t, _ = x.shape
    q = F.linear(x, state[f"{prefix}.linear_q.weight"]).view(b, t, n_head, head_dim)
    k = F.linear(x, state[f"{prefix}.linear_k.weight"]).view(b, t, n_head, head_dim)
    v = F.linear(x, state[f"{prefix}.linear_v.weight"]).view(b, t, n_head, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    n_batch_pos = pos_emb.size(0)
    p = F.linear(pos_emb, state[f"{prefix}.linear_pos.weight"]).view(n_batch_pos, -1, n_head, head_dim)
    p = p.transpose(1, 2)

    q_with_bias_u = (q + state[f"{prefix}.pos_bias_u"]).transpose(1, 2)
    q_with_bias_v = (q + state[f"{prefix}.pos_bias_v"]).transpose(1, 2)

    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
    matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
    matrix_bd = rel_shift(matrix_bd)
    matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

    scores = (matrix_ac + matrix_bd) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, n_head * head_dim)
    return F.linear(out, state[f"{prefix}.linear_out.weight"])


def conformer_layer(x: torch.Tensor, pos_emb: torch.Tensor, state: dict, i: int, hp: dict) -> torch.Tensor:
    p = f"encoder.layers.{i}"
    eps = hp["ln_eps"]
    n_embd = hp["n_embd"]

    residual = x
    y = F.layer_norm(x, (n_embd,), state[f"{p}.norm_feed_forward1.weight"], state[f"{p}.norm_feed_forward1.bias"], eps)
    y = F.linear(y, state[f"{p}.feed_forward1.linear1.weight"])
    y = F.silu(y)
    y = F.linear(y, state[f"{p}.feed_forward1.linear2.weight"])
    x = residual + 0.5 * y

    residual = x
    y = F.layer_norm(x, (n_embd,), state[f"{p}.norm_self_att.weight"], state[f"{p}.norm_self_att.bias"], eps)
    y = rel_pos_mhsa(y, pos_emb, state, f"{p}.self_attn", hp)
    x = residual + y

    residual = x
    y = F.layer_norm(x, (n_embd,), state[f"{p}.norm_conv.weight"], state[f"{p}.norm_conv.bias"], eps)
    y = y.transpose(1, 2)
    y = F.conv1d(y, state[f"{p}.conv.pointwise_conv1.weight"])
    y = F.glu(y, dim=1)
    y = F.conv1d(y, state[f"{p}.conv.depthwise_conv.weight"], padding=hp["conv_padding"], groups=n_embd)
    y = F.batch_norm(y, state[f"{p}.conv.batch_norm.running_mean"], state[f"{p}.conv.batch_norm.running_var"],
                      state[f"{p}.conv.batch_norm.weight"], state[f"{p}.conv.batch_norm.bias"],
                      training=False, eps=hp["bn_eps"])
    y = F.silu(y)
    y = F.conv1d(y, state[f"{p}.conv.pointwise_conv2.weight"])
    y = y.transpose(1, 2)
    x = residual + y

    residual = x
    y = F.layer_norm(x, (n_embd,), state[f"{p}.norm_feed_forward2.weight"], state[f"{p}.norm_feed_forward2.bias"], eps)
    y = F.linear(y, state[f"{p}.feed_forward2.linear1.weight"])
    y = F.silu(y)
    y = F.linear(y, state[f"{p}.feed_forward2.linear2.weight"])
    x = residual + 0.5 * y

    return F.layer_norm(x, (n_embd,), state[f"{p}.norm_out.weight"], state[f"{p}.norm_out.bias"], eps)


def encoder_forward(mel: np.ndarray, state: dict, hp: dict):
    """mel: (T_in, feat_in) numpy. Returns encoder_out (n_subsampled, n_embd) numpy."""
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # (1,1,T_in,feat_in)

    x = F.conv2d(x, state["encoder.pre_encode.conv.0.weight"], state["encoder.pre_encode.conv.0.bias"], stride=2, padding=1)
    x = F.relu(x)
    for dw_idx, pw_idx in ((2, 3), (5, 6)):
        x = F.conv2d(x, state[f"encoder.pre_encode.conv.{dw_idx}.weight"], state[f"encoder.pre_encode.conv.{dw_idx}.bias"],
                     stride=2, padding=1, groups=hp["subsampling_conv_channels"])
        x = F.conv2d(x, state[f"encoder.pre_encode.conv.{pw_idx}.weight"], state[f"encoder.pre_encode.conv.{pw_idx}.bias"])
        x = F.relu(x)

    b, c, t, f = x.size()
    x = x.transpose(1, 2).reshape(b, t, c * f)
    x = F.linear(x, state["encoder.pre_encode.out.weight"], state["encoder.pre_encode.out.bias"])
    # NO xscale (xscaling=false, confirmed real for this checkpoint).

    n_subsampled = x.size(1)
    pos_emb = torch.from_numpy(sinusoidal_pos_emb(n_subsampled, hp["n_embd"])).unsqueeze(0)

    for i in range(hp["n_layers"]):
        x = conformer_layer(x, pos_emb, state, i, hp)

    return x.squeeze(0).detach().numpy()


def lstm_stack_step(last_label: int, h_layers, c_layers, state: dict, hp: dict):
    embed = state["decoder.prediction.embed.weight"][last_label].numpy()
    layer_input = embed
    h_new_layers, c_new_layers = [], []
    for layer in range(hp["pred_rnn_layers"]):
        p = f"decoder.prediction.dec_rnn.lstm."
        w_ih = state[f"{p}weight_ih_l{layer}"].numpy()
        w_hh = state[f"{p}weight_hh_l{layer}"].numpy()
        b_ih = state[f"{p}bias_ih_l{layer}"].numpy()
        b_hh = state[f"{p}bias_hh_l{layer}"].numpy()
        gates = w_ih @ layer_input + w_hh @ h_layers[layer] + b_ih + b_hh
        i, f_, g, o = np.split(gates, 4)
        i, f_, g, o = 1 / (1 + np.exp(-i)), 1 / (1 + np.exp(-f_)), np.tanh(g), 1 / (1 + np.exp(-o))
        c_new = f_ * c_layers[layer] + i * g
        h_new = o * np.tanh(c_new)
        h_new_layers.append(h_new.astype(np.float32))
        c_new_layers.append(c_new.astype(np.float32))
        layer_input = h_new
    return h_new_layers, c_new_layers, h_new_layers[-1]


def joint(encoder_frame: np.ndarray, decoder_out: np.ndarray, state: dict) -> np.ndarray:
    f_proj = state["joint.enc.weight"].numpy() @ encoder_frame + state["joint.enc.bias"].numpy()
    g_proj = state["joint.pred.weight"].numpy() @ decoder_out + state["joint.pred.bias"].numpy()
    activated = np.maximum(f_proj + g_proj, 0.0)
    return (state["joint.joint_net.2.weight"].numpy() @ activated + state["joint.joint_net.2.bias"].numpy()).astype(np.float32)


def greedy_decode_tdt(encoder_output: np.ndarray, state: dict, hp: dict):
    """Real NeMo greedy-TDT control flow. Returns (tokens, frame_indices)."""
    n_frames = encoder_output.shape[0]
    blank_id = hp["blank_id"]
    durations = hp["durations"]
    max_symbols = 10

    h_layers = [np.zeros(hp["pred_hidden"], dtype=np.float32) for _ in range(hp["pred_rnn_layers"])]
    c_layers = [np.zeros(hp["pred_hidden"], dtype=np.float32) for _ in range(hp["pred_rnn_layers"])]
    last_label = blank_id

    tokens, frame_indices = [], []
    time_idx = 0
    while time_idx < n_frames:
        f = encoder_output[time_idx]
        symbols_added = 0
        advanced = False
        while symbols_added < max_symbols:
            h_new_layers, c_new_layers, top_h = lstm_stack_step(last_label, h_layers, c_layers, state, hp)
            combined = joint(f, top_h, state)
            token_logits = combined[: hp["num_classes"] + 1]
            duration_logits = combined[hp["num_classes"] + 1 :]
            k = int(np.argmax(token_logits))
            d_idx = int(np.argmax(duration_logits))
            skip = durations[d_idx]
            if k != blank_id:
                tokens.append(k)
                frame_indices.append(time_idx)
                h_layers, c_layers, last_label = h_new_layers, c_new_layers, k
            elif skip == 0:
                skip = 1
            symbols_added += 1
            time_idx += skip
            if skip > 0:
                advanced = True
                break
        if not advanced:
            time_idx += 1
    return tokens, frame_indices


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.nemo> <out_dir> [--n-samples N]", file=sys.stderr)
        sys.exit(1)
    nemo_path, out_dir = sys.argv[1], Path(sys.argv[2])
    n_samples = 16000
    if "--n-samples" in sys.argv:
        n_samples = int(sys.argv[sys.argv.index("--n-samples") + 1])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config, state, _ = common.load_nemo(nemo_path)
    hp = hparams(config)
    hp.update(mel_common.mel_hparams(hp["feat_in"]))
    hp["n_mels"] = hp["feat_in"]

    rng = np.random.default_rng(2024)
    waveform = rng.normal(scale=0.1, size=n_samples).astype(np.float32)
    mel = compute_mel_features(waveform, hp)

    encoder_out = encoder_forward(mel, state, hp)
    tokens, frame_indices = greedy_decode_tdt(encoder_out, state, hp)

    waveform.tofile(out_dir / "waveform.bin")
    pos_emb = sinusoidal_pos_emb(encoder_out.shape[0], hp["n_embd"])
    pos_emb.astype(np.float32).tofile(out_dir / "pos_emb_raw.bin")
    encoder_out.astype(np.float32).tofile(out_dir / "expected_encoder_output.bin")
    import json
    (out_dir / "expected_decode.json").write_text(json.dumps({"tokens": tokens, "frame_indices": frame_indices}))

    print(f"n_samples={n_samples} t_mel={mel.shape[0]} n_subsampled={encoder_out.shape[0]}")
    print(f"encoder_out shape={encoder_out.shape}, mean={encoder_out.mean():.6f}, std={encoder_out.std():.6f}")
    print(f"decoded tokens={tokens}")
    print(f"frame_indices={frame_indices}")


if __name__ == "__main__":
    main()
