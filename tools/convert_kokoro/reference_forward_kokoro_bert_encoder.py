"""Hand-rolled pure-PyTorch reference for Kokoro's `bert_encoder` (plain `Linear(768,512)` +
`.transpose(-1,-2)`), against the real checkpoint's own `bert_encoder` weights, used as ground truth for
tests/test_e2e_kokoro_bert_encoder.cpp.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd_all["bert_encoder"]
    w = sd["module.weight"]
    b = sd["module.bias"]

    rng = np.random.RandomState(71)
    T = 5
    bert_dur = torch.from_numpy(rng.normal(scale=0.4, size=(T, 768)).astype(np.float32))  # (T,768), time-major

    with torch.no_grad():
        d_en = F.linear(bert_dur, w, b).T  # (512,T) -- real .transpose(-1,-2)

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(arr))

    save("ref_bert_encoder_x", bert_dur.numpy())
    save("ref_bert_encoder_out", d_en.numpy())
    print(f"T={T}, d_en shape={d_en.shape} (expect (512,{T}))")


if __name__ == "__main__":
    main()
