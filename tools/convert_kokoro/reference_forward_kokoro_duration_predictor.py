"""Hand-rolled pure-PyTorch reference for Kokoro's `DurationEncoder` -> `ProsodyPredictor.lstm` ->
`duration_proj` (the duration-prediction half of `ProsodyPredictor`, see convert_kokoro_duration_predictor
.py's own module docstring for the full architecture confirmation trail), used as the ground truth
test_e2e_kokoro_duration_predictor.cpp compares loom-engine's C++ output against.

Takes `d_en` (a (d_model=512, T) sequence -- what `KModel.forward_with_tokens` calls
`self.bert_encoder(bert_dur).transpose(-1,-2)`, i.e. CustomAlbert's own output run through
`bert_encoder`'s Linear(768,512)) and a `style` vector (128-dim) as direct inputs, rather than
re-deriving them from `CustomAlbert`'s own already-separately-verified forward pass -- keeps this piece's
verification independent of upstream pieces (same "each stage checked against its own direct input"
discipline as every other model in this project).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_duration_predictor import HP


def ada_layer_norm(x_ct, style, fc_weight, fc_bias, channels, eps):
    """x_ct: (channels, T). Verified algebraically equivalent to the real, transpose-heavy
    AdaLayerNorm.forward (0.0 diff on a hand-checked example) -- see module docstring."""
    x_tc = x_ct.T  # (T, channels)
    normed = F.layer_norm(x_tc, (channels,), eps=eps)
    h = F.linear(style, fc_weight, fc_bias)  # (2*channels,)
    gamma, beta = h[:channels], h[channels:]
    out = (1 + gamma) * normed + beta
    return out.T  # back to (channels, T)


def bilstm_forward(x_tc, w_ih_f, w_hh_f, b_ih_f, b_hh_f, w_ih_b, w_hh_b, b_ih_b, b_hh_b, hidden_per_dir):
    """x_tc: (T, input_dim). Returns (T, 2*hidden_per_dir)."""
    T = x_tc.shape[0]
    h = hidden_per_dir

    def step(x_t, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh):
        gates = w_ih @ x_t + w_hh @ h_prev + b_ih + b_hh
        i, f, g, o = gates[:h], gates[h:2*h], gates[2*h:3*h], gates[3*h:]
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_new = f * c_prev + i * g
        return o * torch.tanh(c_new), c_new

    out = torch.zeros(T, 2 * h)
    h_f = torch.zeros(h)
    c_f = torch.zeros(h)
    for t in range(T):
        h_f, c_f = step(x_tc[t], h_f, c_f, w_ih_f, w_hh_f, b_ih_f, b_hh_f)
        out[t, :h] = h_f
    h_b = torch.zeros(h)
    c_b = torch.zeros(h)
    for t in reversed(range(T)):
        h_b, c_b = step(x_tc[t], h_b, c_b, w_ih_b, w_hh_b, b_ih_b, b_hh_b)
        out[t, h:] = h_b
    return out


def duration_encoder_forward(d_en, style, sd, hp):
    """d_en: (d_model, T). style: (style_dim,). Returns (d_model+style_dim, T) -- DurationEncoder's own
    final output, INCLUDING the style re-concatenation after the last AdaLayerNorm (confirmed real, see
    module docstring)."""
    T = d_en.shape[1]
    style_bcast = style.unsqueeze(1).expand(-1, T)  # (style_dim, T)
    x = torch.cat([d_en, style_bcast], dim=0)  # (d_model+style_dim, T)

    for i, lstm_idx in enumerate((0, 2, 4)):
        p = f"module.text_encoder.lstms.{lstm_idx}"
        out = bilstm_forward(
            x.T, sd[f"{p}.weight_ih_l0"], sd[f"{p}.weight_hh_l0"], sd[f"{p}.bias_ih_l0"], sd[f"{p}.bias_hh_l0"],
            sd[f"{p}.weight_ih_l0_reverse"], sd[f"{p}.weight_hh_l0_reverse"], sd[f"{p}.bias_ih_l0_reverse"],
            sd[f"{p}.bias_hh_l0_reverse"], hp["hidden_per_dir"])
        x = out.T  # (d_model, T)

        ada_idx = lstm_idx + 1
        ap = f"module.text_encoder.lstms.{ada_idx}"
        x = ada_layer_norm(x, style, sd[f"{ap}.fc.weight"], sd[f"{ap}.fc.bias"], hp["d_model"], hp["ada_ln_eps"])
        x = torch.cat([x, style_bcast], dim=0)  # re-concat, every time including after the last AdaLN

    return x


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["predictor"]
    hp = HP

    # Arbitrary but fixed (T, d_model) / (style_dim,) inputs -- semantic content doesn't matter for
    # this deterministic architecture-correctness check, only real weights and real shapes do.
    rng = np.random.RandomState(0)
    T = 9
    d_en = torch.from_numpy(rng.normal(scale=0.3, size=(hp["d_model"], T)).astype(np.float32))
    style = torch.from_numpy(rng.normal(scale=0.3, size=(hp["style_dim"],)).astype(np.float32))

    with torch.no_grad():
        d = duration_encoder_forward(d_en, style, sd, hp)  # (640, T)

        top = bilstm_forward(
            d.T, sd["module.lstm.weight_ih_l0"], sd["module.lstm.weight_hh_l0"], sd["module.lstm.bias_ih_l0"],
            sd["module.lstm.bias_hh_l0"], sd["module.lstm.weight_ih_l0_reverse"], sd["module.lstm.weight_hh_l0_reverse"],
            sd["module.lstm.bias_ih_l0_reverse"], sd["module.lstm.bias_hh_l0_reverse"], hp["hidden_per_dir"])  # (T, 512)

        duration_logits = F.linear(top, sd["module.duration_proj.linear_layer.weight"],
                                    sd["module.duration_proj.linear_layer.bias"])  # (T, 50)
        duration = torch.sigmoid(duration_logits).sum(dim=-1)  # (T,) -- KModel's own sigmoid-sum, speed=1

    np.save(out_dir / "ref_duration_d_en.npy", d_en.numpy())
    np.save(out_dir / "ref_duration_style.npy", style.numpy())
    np.save(out_dir / "ref_duration_logits.npy", duration_logits.numpy())
    np.save(out_dir / "ref_duration_values.npy", duration.numpy())
    print(f"T={T}, duration_logits shape={duration_logits.shape}, duration={duration.numpy()}")


if __name__ == "__main__":
    main()
