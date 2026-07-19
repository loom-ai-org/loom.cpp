"""Ground truth for the Decoder U-Net conversion: loads the REAL
`matcha.models.components.decoder.Decoder` directly with real checkpoint weights, and runs its real
forward pass (`dphi_dt = estimator(x, mask, mu, t, spks=None, cond=None)`) on a small, hand-crafted
input -- isolates the U-Net from the rest of the pipeline (TextEncoder, duration expansion), same
"test each piece against a small hand-crafted input" precedent as every other component test this
session. `T=8` (a multiple of 4, matching the conversion's own scope decision that mel-frame counts are
always exactly-multiple-of-4 and mask is always all-ones/unpadded).

Usage: python3 reference_forward_matcha_decoder.py <matcha_ljspeech.ckpt> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

from matcha.models.components.decoder import Decoder


def main():
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    sd = ckpt["state_dict"]

    n_feats = hp["encoder"]["encoder_params"]["n_feats"]
    decoder = Decoder(in_channels=2 * n_feats, out_channels=n_feats, **hp["decoder"])
    dec_sd = {k[len("decoder.estimator."):]: v for k, v in sd.items() if k.startswith("decoder.estimator.")}
    missing, unexpected = decoder.load_state_dict(dec_sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    decoder.eval()

    torch.manual_seed(0)
    T = 8
    x = torch.randn(1, n_feats, T)
    mu = torch.randn(1, n_feats, T)
    mask = torch.ones(1, 1, T)
    t = torch.tensor([0.37])

    with torch.no_grad():
        dphi_dt = decoder(x, mask, mu, t, spks=None, cond=None)

    np.save(out_dir / "ref_decoder_x.npy", x.squeeze(0).numpy().astype(np.float32))       # (80, T)
    np.save(out_dir / "ref_decoder_mu.npy", mu.squeeze(0).numpy().astype(np.float32))     # (80, T)
    np.save(out_dir / "ref_decoder_t.npy", t.numpy().astype(np.float32))                  # (1,)
    np.save(out_dir / "ref_decoder_dphi_dt.npy", dphi_dt.squeeze(0).numpy().astype(np.float32))  # (80, T)
    print("dphi_dt shape", dphi_dt.shape)
    print("dphi_dt[0,:5]", dphi_dt[0, 0, :5])


if __name__ == "__main__":
    main()
