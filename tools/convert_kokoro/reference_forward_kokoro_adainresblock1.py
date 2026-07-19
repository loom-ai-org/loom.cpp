"""Hand-rolled pure-PyTorch reference for Kokoro Generator's `AdaINResBlock1` (istftnet.py, real class
verbatim), used as ground truth for tests/test_e2e_kokoro_adainresblock1.cpp. Uses a small SYNTHETIC
instance (channels=4, style_dim=8, kernel_size=3, dilations=(1,3,5)) with random weights -- this piece is
checkpoint-independent (verifies the WIRING, same as VITS's own test_hifigan_generator precedent), the
real checkpoint's actual weights get wired in later when the full Generator is assembled (task #89).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_f0n import fold_weight_norm


def get_padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2


def adain1d(x_tc, style, fc_weight, fc_bias, channels, eps):
    mean = x_tc.mean(dim=0, keepdim=True)
    var = x_tc.var(dim=0, keepdim=True, unbiased=False)
    normed = (x_tc - mean) / torch.sqrt(var + eps)
    h = F.linear(style, fc_weight, fc_bias)
    gamma, beta = h[:channels], h[channels:]
    return (1 + gamma) * normed + beta


def adain_resblock1(x_tc, style, sd, channels, style_dim, eps, kernel_size, dilations):
    """x_tc: (T,channels). Returns (T,channels)."""
    x = x_tc
    for i, d in enumerate(dilations):
        r = adain1d(x, style, sd[f"adain1.{i}.fc.weight"], sd[f"adain1.{i}.fc.bias"], channels, eps)
        alpha1 = sd[f"alpha1.{i}"].reshape(-1)
        r = r + (1.0 / alpha1) * torch.sin(alpha1 * r) ** 2
        w1 = torch.from_numpy(fold_weight_norm(sd[f"convs1.{i}.weight_g"], sd[f"convs1.{i}.weight_v"]))
        b1 = sd[f"convs1.{i}.bias"]
        r = F.conv1d(r.T.unsqueeze(0), w1, b1, dilation=d, padding=get_padding(kernel_size, d))[0].T

        r = adain1d(r, style, sd[f"adain2.{i}.fc.weight"], sd[f"adain2.{i}.fc.bias"], channels, eps)
        alpha2 = sd[f"alpha2.{i}"].reshape(-1)
        r = r + (1.0 / alpha2) * torch.sin(alpha2 * r) ** 2
        w2 = torch.from_numpy(fold_weight_norm(sd[f"convs2.{i}.weight_g"], sd[f"convs2.{i}.weight_v"]))
        b2 = sd[f"convs2.{i}.bias"]
        r = F.conv1d(r.T.unsqueeze(0), w2, b2, dilation=1, padding=get_padding(kernel_size, 1))[0].T

        x = r + x
    return x


def make_synthetic_state_dict(rng, channels, style_dim, kernel_size, dilations):
    sd = {}
    for i in range(len(dilations)):
        for norm in ("adain1", "adain2"):
            sd[f"{norm}.{i}.fc.weight"] = torch.from_numpy(rng.normal(scale=0.2, size=(2 * channels, style_dim)).astype(np.float32))
            sd[f"{norm}.{i}.fc.bias"] = torch.from_numpy(rng.normal(scale=0.1, size=(2 * channels,)).astype(np.float32))
        for a in ("alpha1", "alpha2"):
            sd[f"{a}.{i}"] = torch.from_numpy((rng.uniform(0.5, 1.5, size=(1, channels, 1))).astype(np.float32))
        for c in ("convs1", "convs2"):
            sd[f"{c}.{i}.weight_g"] = torch.from_numpy(rng.uniform(0.5, 1.5, size=(channels, 1, 1)).astype(np.float32))
            sd[f"{c}.{i}.weight_v"] = torch.from_numpy(rng.normal(scale=0.2, size=(channels, channels, kernel_size)).astype(np.float32))
            sd[f"{c}.{i}.bias"] = torch.from_numpy(rng.normal(scale=0.1, size=(channels,)).astype(np.float32))
    return sd


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <out_dir>", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    channels, style_dim, kernel_size, dilations, eps = 4, 8, 3, (1, 3, 5), 1e-5
    rng = np.random.RandomState(21)
    sd = make_synthetic_state_dict(rng, channels, style_dim, kernel_size, dilations)

    T = 9
    x = torch.from_numpy(rng.normal(scale=0.4, size=(T, channels)).astype(np.float32))
    style = torch.from_numpy(rng.normal(scale=0.3, size=(style_dim,)).astype(np.float32))

    with torch.no_grad():
        out = adain_resblock1(x, style, sd, channels, style_dim, eps, kernel_size, dilations)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(arr))

    save("ref_adainresblock1_x", x.numpy())
    save("ref_adainresblock1_style", style.numpy())
    save("ref_adainresblock1_out", out.numpy())

    # Persist the synthetic state dict as a single .npz (keys preserve real dotted names directly, npz
    # supports arbitrary string keys) so convert_kokoro_adainresblock1.py can build the exact same GGUF
    # weights from it -- same "share the fixture" pattern as kokoro_sinegen's l_linear_{w,b}.npy.
    np.savez(out_dir / "adainresblock1_sd.npz", **{k: v.numpy() for k, v in sd.items()})

    print(f"T={T}, channels={channels}, out mean={out.mean().item():.6f}, std={out.std().item():.6f}")


if __name__ == "__main__":
    main()
