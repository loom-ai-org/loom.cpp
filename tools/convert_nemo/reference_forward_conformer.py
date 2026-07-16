#!/usr/bin/env python3
"""Independent plain-PyTorch reimplementation of NVIDIA NeMo's Conformer-CTC forward pass -- mel-spectrogram
frontend, Conformer encoder, and CTC decoder -- used as the ground truth test_e2e_conformer_ctc.cpp
compares loom-engine's C++ output against.

Deliberately does NOT reuse convert_conformer_ctc.py's folding tricks (the 0.5 half-step scale and the
xscale factor are applied directly here, not pre-folded into weights) -- this is meant as a genuinely
independent check of the SAME published algorithm, not a mirror of the converter's own code path. Uses
torch's own F.conv2d/F.conv1d(groups=...)/F.layer_norm/F.glu/F.batch_norm as trusted building blocks
(not nemo_toolkit), per the project's decision to avoid that heavier dependency chain.

The mel frontend here also deliberately does NOT reuse convert_conformer_ctc.py's conv-based DFT trick
(cross-correlating against precomputed cos/sin kernels) -- it calls torch.stft directly, so this is a
genuine independent check that the conv trick actually reproduces a real DFT, not just a restatement of
the same formula. It DOES reuse mel_common.py's mel filterbank construction (same librosa call both
sides need), since re-deriving librosa's mel-filter formula independently would risk a subtle mismatch
unrelated to anything this test is actually meant to catch.

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


def compute_mel_features(waveform: np.ndarray, hp: dict) -> np.ndarray:
    """waveform: (n_samples,) float32. Returns (t_mel, n_mels) log-mel, per-feature CMVN-normalized --
    verbatim NeMo FilterbankFeatures algorithm (see mel_common.py's docstring for the exact source-quoted
    steps), computed independently via real torch.stft + librosa's mel filterbank."""
    x = torch.from_numpy(waveform).unsqueeze(0).double()  # (1, n_samples)
    x = torch.cat((x[:, :1], x[:, 1:] - hp["preemph"] * x[:, :-1]), dim=1)  # preemphasis

    window = torch.hann_window(hp["win_length"], periodic=False, dtype=torch.float64)
    stft = torch.stft(x, n_fft=hp["n_fft"], hop_length=hp["hop_length"], win_length=hp["win_length"],
                       window=window, center=True, pad_mode="constant", return_complex=True)  # (1,n_freq,t_mel)
    power = stft.real ** 2 + stft.imag ** 2  # mag_power=2.0, guard=0 at inference (use_grads=False)

    fb = torch.from_numpy(mel_common.build_mel_filterbank(hp["sample_rate"], hp["n_fft"], hp["n_mels"]).astype(np.float64))
    mel = torch.matmul(fb.unsqueeze(0), power)  # (1, n_mels, t_mel)
    log_mel = torch.log(mel + hp["log_guard"])

    # per_feature normalize: single full utterance, no padding, so valid_mask is all-True.
    mean = log_mel.mean(dim=2, keepdim=True)
    var = ((log_mel - mean) ** 2).sum(dim=2, keepdim=True) / (log_mel.shape[2] - 1)  # unbiased (N-1)
    std = torch.sqrt(var) + hp["norm_eps"]
    normalized = (log_mel - mean) / std

    return normalized.squeeze(0).transpose(0, 1).to(torch.float32).numpy()  # (t_mel, n_mels)


def sinusoidal_pos_emb(n_subsampled: int, n_embd: int) -> np.ndarray:
    """(n_pos, n_embd), n_pos = 2*n_subsampled - 1. Verbatim algorithm from NeMo's
    RelPositionalEncoding.extend_pe/create_pe: positions run from +(length-1) down to -(length-1)."""
    length = n_subsampled
    positions = np.arange(length - 1, -length, -1, dtype=np.float64)
    div_term = np.exp(np.arange(0, n_embd, 2, dtype=np.float64) * -(math.log(10000.0) / n_embd))
    pe = np.zeros((2 * length - 1, n_embd), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions[:, None] * div_term[None, :])
    pe[:, 1::2] = np.cos(positions[:, None] * div_term[None, :])
    return pe


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """Verbatim from NeMo's RelPositionMultiHeadAttention.rel_shift."""
    b, h, qlen, pos_len = x.size()
    x = F.pad(x, pad=(1, 0))
    x = x.view(b, h, -1, qlen)
    x = x[:, :, 1:].view(b, h, qlen, pos_len)
    return x


def rel_pos_mhsa(x: torch.Tensor, pos_emb: torch.Tensor, state: dict, prefix: str, hp: dict) -> torch.Tensor:
    n_head, head_dim = hp["n_head"], hp["head_dim"]
    b, t, _ = x.shape
    q = F.linear(x, state[f"{prefix}.linear_q.weight"], state[f"{prefix}.linear_q.bias"]).view(b, t, n_head, head_dim)
    k = F.linear(x, state[f"{prefix}.linear_k.weight"], state[f"{prefix}.linear_k.bias"]).view(b, t, n_head, head_dim)
    v = F.linear(x, state[f"{prefix}.linear_v.weight"], state[f"{prefix}.linear_v.bias"]).view(b, t, n_head, head_dim)
    k = k.transpose(1, 2)  # (b,h,t,d)
    v = v.transpose(1, 2)

    n_batch_pos = pos_emb.size(0)
    p = F.linear(pos_emb, state[f"{prefix}.linear_pos.weight"]).view(n_batch_pos, -1, n_head, head_dim)
    p = p.transpose(1, 2)  # (batch_pos,h,pos_len,d)

    q_with_bias_u = (q + state[f"{prefix}.pos_bias_u"]).transpose(1, 2)  # (b,h,t,d)
    q_with_bias_v = (q + state[f"{prefix}.pos_bias_v"]).transpose(1, 2)

    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))         # (b,h,t,t)
    matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))         # (b,h,t,pos_len)
    matrix_bd = rel_shift(matrix_bd)
    matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]

    scores = (matrix_ac + matrix_bd) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, n_head * head_dim)
    return F.linear(out, state[f"{prefix}.linear_out.weight"], state[f"{prefix}.linear_out.bias"])


def conformer_layer(x: torch.Tensor, pos_emb: torch.Tensor, state: dict, i: int, hp: dict) -> torch.Tensor:
    p = f"encoder.layers.{i}"
    eps = hp["ln_eps"]
    n_embd = hp["n_embd"]

    # FF1, half-step residual.
    residual = x
    y = F.layer_norm(x, (n_embd,), state[f"{p}.norm_feed_forward1.weight"], state[f"{p}.norm_feed_forward1.bias"], eps)
    y = F.linear(y, state[f"{p}.feed_forward1.linear1.weight"], state[f"{p}.feed_forward1.linear1.bias"])
    y = F.silu(y)
    y = F.linear(y, state[f"{p}.feed_forward1.linear2.weight"], state[f"{p}.feed_forward1.linear2.bias"])
    x = residual + 0.5 * y

    # Self-attention, full residual.
    residual = x
    y = F.layer_norm(x, (n_embd,), state[f"{p}.norm_self_att.weight"], state[f"{p}.norm_self_att.bias"], eps)
    y = rel_pos_mhsa(y, pos_emb, state, f"{p}.self_attn", hp)
    x = residual + y

    # Conv module, full residual.
    residual = x
    y = F.layer_norm(x, (n_embd,), state[f"{p}.norm_conv.weight"], state[f"{p}.norm_conv.bias"], eps)
    y = y.transpose(1, 2)  # (b,c,t)
    y = F.conv1d(y, state[f"{p}.conv.pointwise_conv1.weight"], state[f"{p}.conv.pointwise_conv1.bias"])
    y = F.glu(y, dim=1)
    y = F.conv1d(y, state[f"{p}.conv.depthwise_conv.weight"], state[f"{p}.conv.depthwise_conv.bias"],
                 padding=hp["conv_padding"], groups=n_embd)
    y = F.batch_norm(y, state[f"{p}.conv.batch_norm.running_mean"], state[f"{p}.conv.batch_norm.running_var"],
                      state[f"{p}.conv.batch_norm.weight"], state[f"{p}.conv.batch_norm.bias"],
                      training=False, eps=hp["bn_eps"])
    y = F.silu(y)
    y = F.conv1d(y, state[f"{p}.conv.pointwise_conv2.weight"], state[f"{p}.conv.pointwise_conv2.bias"])
    y = y.transpose(1, 2)
    x = residual + y

    # FF2, half-step residual.
    residual = x
    y = F.layer_norm(x, (n_embd,), state[f"{p}.norm_feed_forward2.weight"], state[f"{p}.norm_feed_forward2.bias"], eps)
    y = F.linear(y, state[f"{p}.feed_forward2.linear1.weight"], state[f"{p}.feed_forward2.linear1.bias"])
    y = F.silu(y)
    y = F.linear(y, state[f"{p}.feed_forward2.linear2.weight"], state[f"{p}.feed_forward2.linear2.bias"])
    x = residual + 0.5 * y

    return F.layer_norm(x, (n_embd,), state[f"{p}.norm_out.weight"], state[f"{p}.norm_out.bias"], eps)


def forward(mel: np.ndarray, state: dict, hp: dict):
    """mel: (T_in, feat_in) numpy. Returns (encoder_out, logits) as numpy arrays, encoder_out shape
    (n_subsampled, n_embd), logits shape (n_subsampled, num_classes)."""
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # (1,1,T_in,feat_in) == NeMo's (b,c,t,f)

    x = F.conv2d(x, state["encoder.pre_encode.conv.0.weight"], state["encoder.pre_encode.conv.0.bias"], stride=2, padding=1)
    x = F.relu(x)
    x = F.conv2d(x, state["encoder.pre_encode.conv.2.weight"], state["encoder.pre_encode.conv.2.bias"], stride=2, padding=1)
    x = F.relu(x)

    b, c, t, f = x.size()
    x = x.transpose(1, 2).reshape(b, t, c * f)  # channel-slower, freq-faster flatten
    x = F.linear(x, state["encoder.pre_encode.out.weight"], state["encoder.pre_encode.out.bias"])
    x = x * math.sqrt(hp["n_embd"])  # xscale

    n_subsampled = x.size(1)
    pos_emb = torch.from_numpy(sinusoidal_pos_emb(n_subsampled, hp["n_embd"])).unsqueeze(0)

    for i in range(hp["n_layers"]):
        x = conformer_layer(x, pos_emb, state, i, hp)

    logits = F.linear(x, state["decoder.decoder_layers.0.weight"].squeeze(-1), state["decoder.decoder_layers.0.bias"])

    return x.squeeze(0).detach().numpy(), logits.squeeze(0).detach().numpy(), pos_emb.squeeze(0).detach().numpy()


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <model.nemo> <out_dir> [--n-samples N]", file=sys.stderr)
        sys.exit(1)
    nemo_path, out_dir = sys.argv[1], Path(sys.argv[2])
    n_samples = 10240  # must match convert_conformer_ctc.py's default exactly (see its main() comment).
    if "--n-samples" in sys.argv:
        n_samples = int(sys.argv[sys.argv.index("--n-samples") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    config, state, _tokenizer_model_bytes = common.load_nemo(nemo_path)
    hp = common.hparams(config)
    hp.update(mel_common.mel_hparams(hp["feat_in"]))
    hp["n_mels"] = hp["feat_in"]

    rng = np.random.default_rng(2024)
    waveform = rng.normal(scale=0.1, size=n_samples).astype(np.float32)
    mel = compute_mel_features(waveform, hp)

    encoder_out, logits, pos_emb = forward(mel, state, hp)

    waveform.tofile(out_dir / "waveform.bin")
    pos_emb.astype(np.float32).tofile(out_dir / "pos_emb_raw.bin")
    encoder_out.astype(np.float32).tofile(out_dir / "expected_encoder_output.bin")
    logits.astype(np.float32).tofile(out_dir / "expected_logits.bin")

    print(f"n_samples={n_samples} t_mel={mel.shape[0]} n_subsampled={encoder_out.shape[0]} n_pos={pos_emb.shape[0]}")
    print(f"encoder_out shape={encoder_out.shape} logits shape={logits.shape}")
    print("nan in logits?", bool(np.isnan(logits).any()))


if __name__ == "__main__":
    main()
