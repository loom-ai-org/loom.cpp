"""Hand-rolled pure-PyTorch reference for a single AdainResBlk1d instance (predictor.F0.0 -- the
simplest case, dim_in=dim_out=512, no learned shortcut, no upsample), used as the ground truth
test_e2e_kokoro_f0_block0.cpp compares loom-engine's C++ output against.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_f0n import HP, fold_weight_norm


def adain1d(x_tc, style, fc_weight, fc_bias, channels, eps):
    """x_tc: (T, channels) -- matches the engine's own [T,channels] convention directly (no transpose
    needed here, unlike AdaLayerNorm's channel-first convention): InstanceNorm1d normalizes each channel
    over TIME, i.e. over dim 0 of a (T,channels) tensor -- exactly what plain per-COLUMN layer_norm
    would give if we transpose to put channels last... concretely: normalize over T for each channel."""
    mean = x_tc.mean(dim=0, keepdim=True)
    var = x_tc.var(dim=0, keepdim=True, unbiased=False)
    normed = (x_tc - mean) / torch.sqrt(var + eps)
    h = F.linear(style, fc_weight, fc_bias)  # (2*channels,)
    gamma, beta = h[:channels], h[channels:]
    return (1 + gamma) * normed + beta


def adain_resblk1d_simple(x_tc, style, sd, sd_prefix, channels, eps, leaky_slope):
    """No shortcut conv, no upsample -- predictor.F0.0/N.0's own case."""
    r = adain1d(x_tc, style, sd[f"{sd_prefix}.norm1.fc.weight"], sd[f"{sd_prefix}.norm1.fc.bias"], channels, eps)
    r = F.leaky_relu(r, leaky_slope)
    w1 = torch.from_numpy(fold_weight_norm(sd[f"{sd_prefix}.conv1.weight_g"], sd[f"{sd_prefix}.conv1.weight_v"]))
    b1 = sd[f"{sd_prefix}.conv1.bias"]
    r = F.conv1d(r.T.unsqueeze(0), w1, b1, padding=1)[0].T

    r = adain1d(r, style, sd[f"{sd_prefix}.norm2.fc.weight"], sd[f"{sd_prefix}.norm2.fc.bias"], channels, eps)
    r = F.leaky_relu(r, leaky_slope)
    w2 = torch.from_numpy(fold_weight_norm(sd[f"{sd_prefix}.conv2.weight_g"], sd[f"{sd_prefix}.conv2.weight_v"]))
    b2 = sd[f"{sd_prefix}.conv2.bias"]
    r = F.conv1d(r.T.unsqueeze(0), w2, b2, padding=1)[0].T

    out = (r + x_tc) / np.sqrt(2.0)
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

    rng = np.random.RandomState(1)
    T = 7
    x = torch.from_numpy(rng.normal(scale=0.4, size=(T, 512)).astype(np.float32))
    style = torch.from_numpy(rng.normal(scale=0.3, size=(hp["style_dim"],)).astype(np.float32))

    with torch.no_grad():
        out = adain_resblk1d_simple(x, style, sd, "module.F0.0", 512, hp["ln_eps"], hp["leaky_slope"])

    # `out`'s own internal `.T` chain (conv1d(...)[0].T inside adain_resblk1d_simple) leaves it
    # non-contiguous (Fortran-ordered) -- np.save happily writes that layout verbatim
    # (`fortran_order: True` in the .npy header), but this project's hand-written minimal .npy reader
    # (used by every e2e test's C++ side) never checks that flag and always assumes C-order, silently
    # misreading a transposed array as if it weren't transposed at all. A real bug caught via a scratch
    # diagnostic that isolated the actual topology output (matched perfectly) from the test's own
    # reference-loading code (didn't) -- fixed at the source here rather than teaching every test's
    # reader about fortran_order: always save a C-contiguous array.
    np.save(out_dir / "ref_f0block0_x.npy", np.ascontiguousarray(x.numpy()))
    np.save(out_dir / "ref_f0block0_style.npy", np.ascontiguousarray(style.numpy()))
    np.save(out_dir / "ref_f0block0_out.npy", np.ascontiguousarray(out.numpy()))
    print(f"T={T}, out shape={out.shape}, mean={out.mean().item():.6f}, std={out.std().item():.6f}")


if __name__ == "__main__":
    main()
