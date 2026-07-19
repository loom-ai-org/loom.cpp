"""Hand-rolled pure-PyTorch reference for Kokoro's full `ProsodyPredictor.F0Ntrain` (modules.py):
shared BiLSTM (640->512) -> two independent 3-block `AdainResBlk1d` stacks (F0: 512->512->256->256, N:
same shape) -> `F0_proj`/`N_proj` (plain `Conv1d(256,1,kernel=1)`). Used as the ground truth
test_e2e_kokoro_f0ntrain.cpp compares loom-engine's C++ output against.

Takes `en` (a (640,T) sequence -- what `KModel.forward_with_tokens` calls
`d.transpose(-1,-2) @ pred_aln_trg`, i.e. `DurationEncoder`'s own 640-channel output ALIGNMENT-EXPANDED
to frame rate) and `style` as direct inputs, rather than re-deriving them from the (separately verified)
duration-prediction/frame-expansion pieces -- keeps this piece's verification independent (same
discipline as every other stage this whole project).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_f0n import HP, fold_weight_norm
from reference_forward_kokoro_f0_block0 import adain1d


def bilstm_forward(x_tc, w_ih_f, w_hh_f, b_ih_f, b_hh_f, w_ih_b, w_hh_b, b_ih_b, b_hh_b, hidden_per_dir):
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


def adain_resblk1d_general(x_tc, style, sd, sd_prefix, dim_in, dim_out, eps, leaky_slope, upsample):
    """Generalizes reference_forward_kokoro_f0_block0.adain_resblk1d_simple (no shortcut/no upsample) and
    reference_forward_kokoro_f0_block1.adain_resblk1d (shortcut+upsample, hardcoded together) into the one
    real AdainResBlk1d structure needed for all 3 stack positions: block0/block2 (dim_in==dim_out, no
    shortcut conv, no upsample) and block1 (dim_in!=dim_out, WITH shortcut conv, upsample) -- real
    `learned_sc = dim_in != dim_out` and `upsample` are independent flags (confirmed in istftnet.py)."""
    learned_sc = dim_in != dim_out
    x_ct = x_tc.T.unsqueeze(0)  # (1, dim_in, T)

    sc = x_ct
    if upsample:
        sc = F.interpolate(sc, scale_factor=2, mode="nearest")
    if learned_sc:
        w1x1 = torch.from_numpy(fold_weight_norm(sd[f"{sd_prefix}.conv1x1.weight_g"], sd[f"{sd_prefix}.conv1x1.weight_v"]))
        sc = F.conv1d(sc, w1x1)  # no bias
    sc_tc = sc[0].T  # (T_out, dim_out)

    r = adain1d(x_tc, style, sd[f"{sd_prefix}.norm1.fc.weight"], sd[f"{sd_prefix}.norm1.fc.bias"], dim_in, eps)
    r = F.leaky_relu(r, leaky_slope)
    r_ct = r.T.unsqueeze(0)
    if upsample:
        w_pool = torch.from_numpy(fold_weight_norm(sd[f"{sd_prefix}.pool.weight_g"], sd[f"{sd_prefix}.pool.weight_v"]))
        b_pool = sd[f"{sd_prefix}.pool.bias"]
        r_ct = F.conv_transpose1d(r_ct, w_pool, b_pool, stride=2, padding=1, output_padding=1, groups=dim_in)
    w1 = torch.from_numpy(fold_weight_norm(sd[f"{sd_prefix}.conv1.weight_g"], sd[f"{sd_prefix}.conv1.weight_v"]))
    b1 = sd[f"{sd_prefix}.conv1.bias"]
    r_ct = F.conv1d(r_ct, w1, b1, padding=1)
    r = r_ct[0].T  # (T_out, dim_out)

    r = adain1d(r, style, sd[f"{sd_prefix}.norm2.fc.weight"], sd[f"{sd_prefix}.norm2.fc.bias"], dim_out, eps)
    r = F.leaky_relu(r, leaky_slope)
    w2 = torch.from_numpy(fold_weight_norm(sd[f"{sd_prefix}.conv2.weight_g"], sd[f"{sd_prefix}.conv2.weight_v"]))
    b2 = sd[f"{sd_prefix}.conv2.bias"]
    r = F.conv1d(r.T.unsqueeze(0), w2, b2, padding=1)[0].T

    return (r + sc_tc) / np.sqrt(2.0)


def f0ntrain(en_ct, style, sd, hp):
    """en_ct: (640, T). Returns (F0, N), each (T*2,) -- F0_proj/N_proj squeeze the channel dim to 1,
    and the middle AdainResBlk1d in each stack doubles T."""
    x_tc = en_ct.T  # (T, 640) -- BiLSTM's own batch_first input convention
    shared_out = bilstm_forward(
        x_tc, sd["module.shared.weight_ih_l0"], sd["module.shared.weight_hh_l0"], sd["module.shared.bias_ih_l0"],
        sd["module.shared.bias_hh_l0"], sd["module.shared.weight_ih_l0_reverse"], sd["module.shared.weight_hh_l0_reverse"],
        sd["module.shared.bias_ih_l0_reverse"], sd["module.shared.bias_hh_l0_reverse"], 256)  # (T, 512)

    def run_stack(prefix):
        x = shared_out  # (T, 512)
        dims = [(512, 512, False), (512, 256, True), (256, 256, False)]
        for i, (dim_in, dim_out, upsample) in enumerate(dims):
            x = adain_resblk1d_general(x, style, sd, f"{prefix}.{i}", dim_in, dim_out, hp["ln_eps"], hp["leaky_slope"], upsample)
        return x  # (T*2, 256)

    f0_feat = run_stack("module.F0")
    n_feat = run_stack("module.N")

    f0_proj_w = sd["module.F0_proj.weight"]  # (1,256,1)
    f0_proj_b = sd["module.F0_proj.bias"]
    n_proj_w = sd["module.N_proj.weight"]
    n_proj_b = sd["module.N_proj.bias"]
    F0 = F.conv1d(f0_feat.T.unsqueeze(0), f0_proj_w, f0_proj_b)[0, 0]  # (T*2,)
    N = F.conv1d(n_feat.T.unsqueeze(0), n_proj_w, n_proj_b)[0, 0]
    return F0, N


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["predictor"]
    hp = HP

    rng = np.random.RandomState(3)
    T = 5
    en = torch.from_numpy(rng.normal(scale=0.3, size=(640, T)).astype(np.float32))
    style = torch.from_numpy(rng.normal(scale=0.3, size=(hp["style_dim"],)).astype(np.float32))

    with torch.no_grad():
        F0, N = f0ntrain(en, style, sd, hp)

    np.save(out_dir / "ref_f0ntrain_en.npy", np.ascontiguousarray(en.numpy()))
    np.save(out_dir / "ref_f0ntrain_style.npy", np.ascontiguousarray(style.numpy()))
    np.save(out_dir / "ref_f0ntrain_F0.npy", np.ascontiguousarray(F0.numpy()))
    np.save(out_dir / "ref_f0ntrain_N.npy", np.ascontiguousarray(N.numpy()))
    print(f"T={T}, F0 shape={F0.shape}, N shape={N.shape}")
    print("F0", F0.numpy())
    print("N", N.numpy())


if __name__ == "__main__":
    main()
