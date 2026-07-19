"""Hand-rolled pure-PyTorch reference for predictor.F0.1 -- the UPSAMPLING AdainResBlk1d instance
(dim_in=512, dim_out=256, WITH a learned conv1x1 shortcut AND upsample=True), used as the ground truth
test_e2e_kokoro_f0_block1.cpp compares loom-engine's C++ output against. Real ops (F.interpolate,
F.conv_transpose1d) are used directly here since this is the ground-truth reference, not the engine
composition (which instead composes the depthwise ConvTranspose1d from RESHAPE/PAD_1D/CONV_1D_DW,
verified separately in test_primitive_registry.cpp).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_f0n import HP, fold_weight_norm
from reference_forward_kokoro_f0_block0 import adain1d


def adain_resblk1d(x_tc, style, sd, sd_prefix, dim_in, dim_out, eps, leaky_slope, upsample):
    """x_tc: (T, dim_in). Returns (T_out, dim_out) -- T_out = 2*T if upsample else T."""
    x_ct = x_tc.T.unsqueeze(0)  # (1, dim_in, T) -- native conv/interpolate layout

    # --- shortcut: plain nearest UpSample1d, then the learned conv1x1 (dim_in!=dim_out here) ---
    sc = x_ct
    if upsample:
        sc = F.interpolate(sc, scale_factor=2, mode="nearest")
    w1x1 = torch.from_numpy(fold_weight_norm(sd[f"{sd_prefix}.conv1x1.weight_g"], sd[f"{sd_prefix}.conv1x1.weight_v"]))
    sc = F.conv1d(sc, w1x1)  # no bias (real AdainResBlk1d's conv1x1 has bias=False)

    # --- residual: norm1 -> act -> pool (learned depthwise ConvTranspose1d) -> conv1 -> norm2 -> act -> conv2 ---
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

    sc_tc = sc[0].T  # (T_out, dim_out)
    out = (r + sc_tc) / np.sqrt(2.0)
    return out


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["predictor"]
    hp = HP

    rng = np.random.RandomState(2)
    T = 6
    x = torch.from_numpy(rng.normal(scale=0.4, size=(T, 512)).astype(np.float32))
    style = torch.from_numpy(rng.normal(scale=0.3, size=(hp["style_dim"],)).astype(np.float32))

    with torch.no_grad():
        out = adain_resblk1d(x, style, sd, "module.F0.1", 512, 256, hp["ln_eps"], hp["leaky_slope"], upsample=True)

    np.save(out_dir / "ref_f0block1_x.npy", np.ascontiguousarray(x.numpy()))
    np.save(out_dir / "ref_f0block1_style.npy", np.ascontiguousarray(style.numpy()))
    np.save(out_dir / "ref_f0block1_out.npy", np.ascontiguousarray(out.numpy()))
    print(f"T={T}, out shape={out.shape}, mean={out.mean().item():.6f}, std={out.std().item():.6f}")


if __name__ == "__main__":
    main()
