#!/usr/bin/env python3
"""Independent numpy re-implementation of the toy VAE decoder's forward pass (see toy_vae_common.py),
used as the ground truth test_e2e_toy_vae.cpp compares loom-engine's C++ output against.

Requires: pip install numpy
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_common as enc
import toy_vae_common as common


def conv_transpose1d(data: np.ndarray, kernel: np.ndarray, stride: int) -> np.ndarray:
    """data: (IC,IL), kernel: (IC,OC,K) -- PyTorch's ConvTranspose1d weight convention -- -> (OC,OL),
    OL=(IL-1)*stride+K. Small/naive on purpose, same rationale as the other reference conv helpers."""
    ic, il = data.shape
    ic2, oc, k = kernel.shape
    assert ic == ic2
    ol = (il - 1) * stride + k

    out = np.zeros((oc, ol), dtype=np.float32)
    for o in range(oc):
        for frame in range(il):
            for kk in range(k):
                out[o, frame * stride + kk] += np.sum(data[:, frame] * kernel[:, o, kk])
    return out


def forward() -> np.ndarray:
    weights = common.generate_weights()
    latent = common.generate_latent()  # (n_embd, t_latent), gguf-ready (C,T) layout

    upsampled = conv_transpose1d(latent, weights["upsample.weight"], stride=common.STRIDE_UP)  # (OC,OL)
    upsampled = enc.gelu_erf(upsampled)

    data = upsampled[np.newaxis, :, :]  # (OC,OL) -> (1,IC,IL) for conv1d
    decoded = enc.conv1d(data, weights["refine.weight"], stride=1, padding=common.PADDING_REFINE)  # (1,OC,OL)
    return decoded[0].astype(np.float32)  # (n_embd, t_out) -- already gguf-ready (C,T), no transpose needed


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    forward().tofile(out_dir / "expected_output.bin")


if __name__ == "__main__":
    main()
