"""Standalone verification of `add_convnext_block` (supertonic_common.py) against the real
`vocoder.pt`'s own `convnext.{0,1}` sub-modules (see reference_forward_supertonic_convnext.py's own
docstring for why using the real module directly is possible/preferable here). Both instances are
CAUSAL (SpeechDecoder's own convention): block0 dilation=1, block1 dilation=2 -- exercises the
causal-left-pad composition at two different pad widths before trusting it in the full stack.

Usage: python3 convert_supertonic_convnext.py <supertonic-tts repo root> <out_dir>
"""
import sys
from pathlib import Path

import torch

from supertonic_common import TopologyBuilder, add_convnext_block, write_gguf

DIM = 512
INTERM_DIM = 2048
KERNEL_SIZE = 7


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <supertonic-tts root> <out_dir>", file=sys.stderr)
        sys.exit(1)
    repo_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    decoder = torch.load(repo_root / "assets/pt/vocoder.pt", weights_only=False, map_location="cpu")
    sd = decoder.state_dict()

    for i, dilation, name in [(0, 1, "block0_d1_causal"), (1, 2, "block1_d2_causal")]:
        tb = TopologyBuilder()
        out = add_convnext_block(tb, "x", "cn", sd, f"convnext.{i}", DIM, INTERM_DIM, KERNEL_SIZE,
                                  dilation, causal=True, seq_len_expr="$n_tokens", out_hint="out")
        inputs = [{"name": "x", "dtype": "f32", "shape": ["$n_tokens", str(DIM)]}]
        write_gguf(out_dir / f"supertonic_convnext_{name}.gguf", tb.topology(inputs, out), tb.weights,
                   "loom-supertonic-convnext")


if __name__ == "__main__":
    main()
