#!/usr/bin/env python3
"""Ground truth for `add_convnext_block` (supertonic_common.py): runs the REAL `ConvNextBlock` nn.Module
directly (loaded straight out of `assets/pt/vocoder.pt`'s own `convnext.{0,1}` sub-modules, real weights)
on a synthetic input -- the real package is importable in this venv, so this uses its own `.forward()`
as ground truth rather than a hand-copied formula (a stronger reference than this project's usual
hand-rolled-Python precedent, same "import the real package" idea as Whisper's own openai-whisper use).

`convnext.0` is causal (dilation=1, the SpeechDecoder's own first block); `convnext.1` is causal with
dilation=2 -- both real, both used to check the dilated/causal-pad path, not just the plain case.

Usage: python3 reference_forward_supertonic_convnext.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    decoder = torch.load(repo_root / "assets/pt/vocoder.pt", weights_only=False, map_location="cpu")
    decoder.eval()

    torch.manual_seed(0)
    T = 12
    x = torch.randn(1, 512, T)

    for i, name in [(0, "block0_d1_causal"), (1, "block1_d2_causal")]:
        block = decoder.convnext[i]
        with torch.no_grad():
            y = block(x)
        x.numpy().astype(np.float32).tofile(out_dir / f"convnext_{name}_x.bin")
        y.numpy().astype(np.float32).tofile(out_dir / f"convnext_{name}_y.bin")
        print(f"{name}: dilation={block.dwconv.dilation}, x{tuple(x.shape)} -> y{tuple(y.shape)}, "
              f"mean_abs_y={y.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
