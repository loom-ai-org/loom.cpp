"""Hand-rolled pure-PyTorch reference for Kokoro's `TextEncoder` (modules.py), used as the ground truth
test_e2e_kokoro_text_encoder.cpp compares loom-engine's C++ output (CNN topology + the new
`loom::BiLstmStepper` host driver) against. Deterministic end to end -- no sampling anywhere in this
piece (unlike VITS/Kokoro's own stochastic pieces), so this is a plain exact-match check. The real
module's `masked_fill_`/padding-mask logic is skipped entirely (a no-op for a real single-utterance,
unpadded call, same precedent as every other model in this project).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_text_encoder import HP, fold_weight_norm


def text_encoder_forward(token_ids, sd, hp):
    """token_ids: 1D python list of phoneme ids. Returns (T, channels) numpy array -- NOTE: this is the
    TRANSPOSE of the real module's own (B,channels,T) return convention (channel-first) -- kept as
    (T,channels) here purely because that's the natural shape of loom::BiLstmStepper's own host-side
    output (a plain per-timestep vector list), not because the real model's output layout is different;
    the eventual full-assembly step will transpose back to channel-first before feeding this into
    anything else, same as every other T-first/C-first boundary crossing in this project."""
    T = len(token_ids)
    c = hp["channels"]
    tokens = torch.tensor(token_ids, dtype=torch.long)

    x = F.embedding(tokens, sd["module.embedding.weight"])  # (T, c)
    x = x.T.unsqueeze(0)  # (1, c, T) -- matches the real module's own conv1d-ready layout

    for i in range(hp["depth"]):
        p = f"module.cnn.{i}.0"
        w = torch.from_numpy(fold_weight_norm(sd[f"{p}.weight_g"], sd[f"{p}.weight_v"]))
        b = sd[f"{p}.bias"]
        x = F.conv1d(x, w, b, padding=(hp["kernel_size"] - 1) // 2)
        x = F.layer_norm(x.transpose(1, -1), (c,), sd[f"module.cnn.{i}.1.gamma"],
                          sd[f"module.cnn.{i}.1.beta"], hp["ln_eps"]).transpose(1, -1)
        x = F.leaky_relu(x, hp["leaky_slope"])

    x = x.transpose(1, 2)[0]  # (T, c) -- ready for the bidirectional LSTM

    w_ih_f = sd["module.lstm.weight_ih_l0"]
    w_hh_f = sd["module.lstm.weight_hh_l0"]
    b_ih_f = sd["module.lstm.bias_ih_l0"]
    b_hh_f = sd["module.lstm.bias_hh_l0"]
    w_ih_b = sd["module.lstm.weight_ih_l0_reverse"]
    w_hh_b = sd["module.lstm.weight_hh_l0_reverse"]
    b_ih_b = sd["module.lstm.bias_ih_l0_reverse"]
    b_hh_b = sd["module.lstm.bias_hh_l0_reverse"]
    h = hp["hidden_per_dir"]

    def lstm_step(x_t, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh):
        gates = w_ih @ x_t + w_hh @ h_prev + b_ih + b_hh
        i, f, g, o = gates[:h], gates[h:2*h], gates[2*h:3*h], gates[3*h:]
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    out = torch.zeros(T, 2 * h)
    h_f = torch.zeros(h)
    c_f = torch.zeros(h)
    for t in range(T):
        h_f, c_f = lstm_step(x[t], h_f, c_f, w_ih_f, w_hh_f, b_ih_f, b_hh_f)
        out[t, :h] = h_f
    h_b = torch.zeros(h)
    c_b = torch.zeros(h)
    for t in reversed(range(T)):
        h_b, c_b = lstm_step(x[t], h_b, c_b, w_ih_b, w_hh_b, b_ih_b, b_hh_b)
        out[t, h:] = h_b

    return out.detach().numpy()


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["text_encoder"]

    phoneme_ids = [43, 62, 83, 61, 62, 47, 76, 46, 76, 56, 47]

    with torch.no_grad():
        out = text_encoder_forward(phoneme_ids, sd, HP)

    np.save(out_dir / "ref_text_encoder_tokens.npy", np.array(phoneme_ids, dtype=np.int32))
    np.save(out_dir / "ref_text_encoder_out.npy", out)
    print(f"tokens={phoneme_ids}, out shape={out.shape}, mean={out.mean():.6f}, std={out.std():.6f}")


if __name__ == "__main__":
    main()
