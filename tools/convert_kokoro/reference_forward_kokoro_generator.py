"""Hand-rolled pure-PyTorch reference for Kokoro's Generator "core" (istftnet.py's real
`Generator.forward`, hand-copied verbatim except that `har`/`f0`-derived pieces are taken as direct
inputs -- SineGen/forward-STFT are separately verified topologies, see BACKLOG.md), used as ground truth
for tests/test_e2e_kokoro_generator.cpp. Uses the SAME synthetic state dict
convert_kokoro_generator.py's own `make_synthetic_state_dict` produces (loaded from
kokoro_generator_sd.npz) -- checkpoint-independent structural verification, same precedent as VITS's own
test_hifigan_generator and this milestone's test_e2e_kokoro_adainresblock1.cpp.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from convert_kokoro_f0n import fold_weight_norm
from convert_kokoro_generator import HP


def adain1d(x_tc, style, fc_weight, fc_bias, channels, eps):
    mean = x_tc.mean(dim=0, keepdim=True)
    var = x_tc.var(dim=0, keepdim=True, unbiased=False)
    normed = (x_tc - mean) / torch.sqrt(var + eps)
    h = F.linear(style, fc_weight, fc_bias)
    gamma, beta = h[:channels], h[channels:]
    return (1 + gamma) * normed + beta


def get_padding(kernel_size, dilation):
    return (kernel_size * dilation - dilation) // 2


def adain_resblock1(x_tc, style, sd, prefix, channels, style_dim, eps, kernel_size, dilations):
    x = x_tc
    for i, d in enumerate(dilations):
        r = adain1d(x, style, sd[f"{prefix}.adain1.{i}.fc.weight"], sd[f"{prefix}.adain1.{i}.fc.bias"], channels, eps)
        alpha1 = sd[f"{prefix}.alpha1.{i}"].reshape(-1)
        r = r + (1.0 / alpha1) * torch.sin(alpha1 * r) ** 2
        w1 = torch.from_numpy(fold_weight_norm(sd[f"{prefix}.convs1.{i}.weight_g"], sd[f"{prefix}.convs1.{i}.weight_v"]))
        b1 = sd[f"{prefix}.convs1.{i}.bias"]
        r = F.conv1d(r.T.unsqueeze(0), w1, b1, dilation=d, padding=get_padding(kernel_size, d))[0].T

        r = adain1d(r, style, sd[f"{prefix}.adain2.{i}.fc.weight"], sd[f"{prefix}.adain2.{i}.fc.bias"], channels, eps)
        alpha2 = sd[f"{prefix}.alpha2.{i}"].reshape(-1)
        r = r + (1.0 / alpha2) * torch.sin(alpha2 * r) ** 2
        w2 = torch.from_numpy(fold_weight_norm(sd[f"{prefix}.convs2.{i}.weight_g"], sd[f"{prefix}.convs2.{i}.weight_v"]))
        b2 = sd[f"{prefix}.convs2.{i}.bias"]
        r = F.conv1d(r.T.unsqueeze(0), w2, b2, dilation=1, padding=get_padding(kernel_size, 1))[0].T

        x = r + x
    return x


def generator_forward(x_tc, style, har_tc, sd, hp):
    """x_tc: (T0,512). har_tc: (T_har,2*n_freq). Returns waveform (T0*300,)."""
    upsample_rates = hp["upsample_rates"]
    upsample_kernel_sizes = hp["upsample_kernel_sizes"]
    num_upsamples = len(upsample_rates)
    kernel_sizes = hp["resblock_kernel_sizes"]
    num_kernels = len(kernel_sizes)
    dilations = hp["resblock_dilations"]
    style_dim = hp["style_dim"]
    eps = hp["ada_ln_eps"]
    uic = hp["upsample_initial_channel"]
    n_fft = hp["gen_istft_n_fft"]
    hop = hp["gen_istft_hop_size"]
    n_freq = n_fft // 2 + 1

    x_ct = x_tc.T.unsqueeze(0)  # (1,512,T0)
    har_ct = har_tc.T.unsqueeze(0)  # (1,2*n_freq,T_har)

    for i in range(num_upsamples):
        ch_out = uic // (2 ** (i + 1))
        stride = upsample_rates[i]
        k = upsample_kernel_sizes[i]
        padding = (k - stride) // 2

        x_ct = F.leaky_relu(x_ct, 0.1)

        stride_f0 = 1
        for r in upsample_rates[i + 1:]:
            stride_f0 *= r
        if i + 1 < num_upsamples:
            noise_k, noise_pad, noise_res_k = stride_f0 * 2, (stride_f0 + 1) // 2, 7
        else:
            noise_k, noise_pad, noise_res_k = 1, 0, 11
        x_source_ct = F.conv1d(har_ct, sd[f"noise_convs.{i}.weight"], sd[f"noise_convs.{i}.bias"],
                                stride=stride_f0, padding=noise_pad)
        x_source_tc = adain_resblock1(x_source_ct[0].T, style, sd, f"noise_res.{i}", ch_out, style_dim, eps,
                                       noise_res_k, dilations)

        w_up = torch.from_numpy(fold_weight_norm(sd[f"ups.{i}.weight_g"], sd[f"ups.{i}.weight_v"]))
        b_up = sd[f"ups.{i}.bias"]
        x_ct = F.conv_transpose1d(x_ct, w_up, b_up, stride=stride, padding=padding)
        if i == num_upsamples - 1:
            x_ct = F.pad(x_ct, (1, 0), mode="reflect")
        x_tc2 = x_ct[0].T + x_source_tc

        xs = None
        for j in range(num_kernels):
            rb = adain_resblock1(x_tc2, style, sd, f"resblocks.{i * num_kernels + j}", ch_out, style_dim, eps,
                                  kernel_sizes[j], dilations)
            xs = rb if xs is None else xs + rb
        x_tc2 = xs / num_kernels
        x_ct = x_tc2.T.unsqueeze(0)

    x_ct = F.leaky_relu(x_ct)  # default slope 0.01
    w_post = torch.from_numpy(fold_weight_norm(sd["conv_post.weight_g"], sd["conv_post.weight_v"]))
    b_post = sd["conv_post.bias"]
    x_ct = F.conv1d(x_ct, w_post, b_post, padding=3)

    spec = torch.exp(x_ct[:, :n_freq, :])
    phase = torch.sin(x_ct[:, n_freq:, :])
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
    waveform = torch.istft(spec * torch.exp(phase * 1j), n_fft, hop, n_fft, window=window, center=True)
    return waveform[0]


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <gguf_dir> <ref_out_dir>", file=sys.stderr)
        sys.exit(1)
    gguf_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = HP

    npz = np.load(gguf_dir / "kokoro_generator_sd.npz")
    sd = {k: torch.from_numpy(npz[k]) for k in npz.files}

    rng = np.random.RandomState(41)
    T0 = 2
    uic = hp["upsample_initial_channel"]
    style_dim = hp["style_dim"]
    n_fft = hp["gen_istft_n_fft"]
    n_freq = n_fft // 2 + 1
    T_har = T0 * 60 + 1

    x = torch.from_numpy(rng.normal(scale=0.3, size=(T0, uic)).astype(np.float32))
    style = torch.from_numpy(rng.normal(scale=0.3, size=(style_dim,)).astype(np.float32))
    har = torch.from_numpy(rng.normal(scale=0.3, size=(T_har, 2 * n_freq)).astype(np.float32))

    with torch.no_grad():
        waveform = generator_forward(x, style, har, sd, hp)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(arr))

    save("ref_generator_x", x.numpy())
    save("ref_generator_style", style.numpy())
    save("ref_generator_har", har.numpy())
    save("ref_generator_waveform", waveform.numpy())
    print(f"T0={T0}, T_har={T_har}, waveform_len={waveform.shape[0]} (expect {T0*300})")


if __name__ == "__main__":
    main()
