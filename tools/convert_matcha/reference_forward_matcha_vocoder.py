"""Ground truth for the HiFi-GAN v1 vocoder conversion: loads the REAL `matcha.hifigan.models.Generator`
directly with the real `generator_v1` checkpoint weights and real `v1` config, runs its real forward
pass on a small random mel input.

Usage: python3 reference_forward_matcha_vocoder.py <generator_v1> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

from matcha.hifigan.config import v1
from matcha.hifigan.models import Generator


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def main():
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["generator"]

    h = AttrDict(v1)
    gen = Generator(h)
    gen.load_state_dict(sd, strict=True)
    gen.eval()
    gen.remove_weight_norm()

    torch.manual_seed(0)
    T = 4
    mel = torch.randn(1, 80, T)

    with torch.no_grad():
        wav = gen(mel)

    np.save(out_dir / "ref_vocoder_mel.npy", mel.squeeze(0).numpy().astype(np.float32))  # (80,T)
    np.save(out_dir / "ref_vocoder_wav.npy", wav.squeeze(0).squeeze(0).numpy().astype(np.float32))  # (T*256,)
    print("wav shape", wav.shape)
    print("wav[:5]", wav.flatten()[:5])


if __name__ == "__main__":
    main()
