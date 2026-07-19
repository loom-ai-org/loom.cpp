"""Hand-rolled pure-PyTorch reference for Kokoro's `Decoder` "core" (istftnet.py's real
`Decoder.forward`, everything except the final `self.generator(...)` call), used as ground truth for
tests/test_e2e_kokoro_decoder_core.cpp. Uses the SAME synthetic state dict
convert_kokoro_decoder_core.py's own `make_synthetic_state_dict` produces -- checkpoint-independent
structural verification, same precedent as every other Generator/Decoder piece this milestone.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_decoder_core import HP
from convert_kokoro_f0n import fold_weight_norm


def adain1d(x_tc, style, fc_weight, fc_bias, channels, eps):
    mean = x_tc.mean(dim=0, keepdim=True)
    var = x_tc.var(dim=0, keepdim=True, unbiased=False)
    normed = (x_tc - mean) / torch.sqrt(var + eps)
    h = F.linear(style, fc_weight, fc_bias)
    gamma, beta = h[:channels], h[channels:]
    return (1 + gamma) * normed + beta


def adain_resblk1d(x_tc, style, sd, prefix, dim_in, dim_out, eps, leaky_slope, upsample):
    learned_sc = dim_in != dim_out
    x_ct = x_tc.T.unsqueeze(0)

    sc = x_ct
    if upsample:
        sc = F.interpolate(sc, scale_factor=2, mode="nearest")
    if learned_sc:
        w1x1 = torch.from_numpy(fold_weight_norm(sd[f"{prefix}.conv1x1.weight_g"], sd[f"{prefix}.conv1x1.weight_v"]))
        sc = F.conv1d(sc, w1x1)
    sc_tc = sc[0].T

    r = adain1d(x_tc, style, sd[f"{prefix}.norm1.fc.weight"], sd[f"{prefix}.norm1.fc.bias"], dim_in, eps)
    r = F.leaky_relu(r, leaky_slope)
    r_ct = r.T.unsqueeze(0)
    if upsample:
        w_pool = torch.from_numpy(fold_weight_norm(sd[f"{prefix}.pool.weight_g"], sd[f"{prefix}.pool.weight_v"]))
        b_pool = sd[f"{prefix}.pool.bias"]
        r_ct = F.conv_transpose1d(r_ct, w_pool, b_pool, stride=2, padding=1, output_padding=1, groups=dim_in)
    w1 = torch.from_numpy(fold_weight_norm(sd[f"{prefix}.conv1.weight_g"], sd[f"{prefix}.conv1.weight_v"]))
    b1 = sd[f"{prefix}.conv1.bias"]
    r_ct = F.conv1d(r_ct, w1, b1, padding=1)
    r = r_ct[0].T

    r = adain1d(r, style, sd[f"{prefix}.norm2.fc.weight"], sd[f"{prefix}.norm2.fc.bias"], dim_out, eps)
    r = F.leaky_relu(r, leaky_slope)
    w2 = torch.from_numpy(fold_weight_norm(sd[f"{prefix}.conv2.weight_g"], sd[f"{prefix}.conv2.weight_v"]))
    b2 = sd[f"{prefix}.conv2.bias"]
    r = F.conv1d(r.T.unsqueeze(0), w2, b2, padding=1)[0].T

    return (r + sc_tc) / np.sqrt(2.0)


def decoder_core_forward(asr_tc, f0_curve, n_curve, style, sd):
    """asr_tc: (T,512). f0_curve/n_curve: (2T,). Returns x: (2T,512)."""
    w_f0 = torch.from_numpy(fold_weight_norm(sd["F0_conv.weight_g"], sd["F0_conv.weight_v"]))
    b_f0 = sd["F0_conv.bias"]
    F0 = F.conv1d(f0_curve[None, None, :], w_f0, b_f0, stride=2, padding=1)[0].T  # (T,1)
    w_n = torch.from_numpy(fold_weight_norm(sd["N_conv.weight_g"], sd["N_conv.weight_v"]))
    b_n = sd["N_conv.bias"]
    N = F.conv1d(n_curve[None, None, :], w_n, b_n, stride=2, padding=1)[0].T  # (T,1)

    x = torch.cat([asr_tc, F0, N], dim=1)  # (T,514)
    x = adain_resblk1d(x, style, sd, "encode", 514, 1024, HP["ln_eps"], HP["leaky_slope"], upsample=False)

    w_ar = torch.from_numpy(fold_weight_norm(sd["asr_res.0.weight_g"], sd["asr_res.0.weight_v"]))
    b_ar = sd["asr_res.0.bias"]
    asr_res = F.conv1d(asr_tc.T.unsqueeze(0), w_ar, b_ar)[0].T  # (T,64)

    decode_dims = [(1090, 1024, False), (1090, 1024, False), (1090, 1024, False), (1090, 512, True)]
    res = True
    for i, (dim_in, dim_out, upsample) in enumerate(decode_dims):
        if res:
            x = torch.cat([x, asr_res, F0, N], dim=1)
        x = adain_resblk1d(x, style, sd, f"decode.{i}", dim_in, dim_out, HP["ln_eps"], HP["leaky_slope"], upsample)
        if upsample:
            res = False
    return x


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <gguf_dir> <ref_out_dir>", file=sys.stderr)
        sys.exit(1)
    gguf_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(gguf_dir / "kokoro_decoder_core_sd.npz")
    sd = {k: torch.from_numpy(npz[k]) for k in npz.files}

    rng = np.random.RandomState(61)
    T = 3
    style_dim = HP["style_dim"]
    asr = torch.from_numpy(rng.normal(scale=0.3, size=(T, 512)).astype(np.float32))
    f0_curve = torch.from_numpy(rng.normal(scale=0.3, size=(2 * T,)).astype(np.float32))
    n_curve = torch.from_numpy(rng.normal(scale=0.3, size=(2 * T,)).astype(np.float32))
    style = torch.from_numpy(rng.normal(scale=0.3, size=(style_dim,)).astype(np.float32))

    with torch.no_grad():
        x = decoder_core_forward(asr, f0_curve, n_curve, style, sd)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(arr))

    save("ref_decoder_core_asr", asr.numpy())
    save("ref_decoder_core_f0_curve", f0_curve.numpy())
    save("ref_decoder_core_n_curve", n_curve.numpy())
    save("ref_decoder_core_style", style.numpy())
    save("ref_decoder_core_x", x.numpy())
    print(f"T={T}, x shape={x.shape} (expect ({2*T},512))")


if __name__ == "__main__":
    main()
