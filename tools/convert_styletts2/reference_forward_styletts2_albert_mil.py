"""Hand-rolled pure-PyTorch reference for the MIL-traced "albert" topology (export_styletts2_mil.py's
AlbertWrapper) -- literal reuse of tools/convert_kokoro/reference_forward_kokoro_albert.py's own
`albert_forward` (already verified against the bespoke Kokoro topology in test_e2e_kokoro_albert.cpp) and
convert_kokoro_albert.py's own HP, unmodified: StyleTTS2's PL-BERT config is byte-identical to Kokoro's own
(same hidden_size/n_head/n_layer/ln_eps -- see styletts2_driver.h's own top comment), and the real state
dict's "bert" subtree uses the SAME "module."-prefixed key convention `albert_forward` already expects
(confirmed directly, see convert_styletts2_reused.py's own docstring). Only the checkpoint differs, so
this is an independent ground truth (not trusting export_styletts2_mil.py's own AlbertWrapper at all,
unlike the decoder_vocoder reference below which necessarily does -- see that script's own docstring).

Output convention: (T,768) row-major/time-major -- byte-identical to AlbertWrapper's own output (no
transpose either way, see that class's own docstring for why).

Usage: python3 reference_forward_styletts2_albert_mil.py <epoch_2nd_00100.pth> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_kokoro"))
from convert_kokoro_albert import HP  # noqa: E402
from reference_forward_kokoro_albert import albert_forward  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <epoch_2nd_00100.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net = sd_all["net"] if "net" in sd_all else sd_all
    sd = net["bert"]

    # Real StyleTTS2 wraps with a SINGLE LEADING 0 token only (NOT Kokoro's leading+trailing convention --
    # see styletts2_driver.h's own docstring) -- but this reference just checks CustomAlbert's own math in
    # isolation given arbitrary valid token ids, so the exact wrapping convention doesn't matter here; a
    # longer sequence than reference_forward_kokoro_albert.py's own default also exercises position
    # embeddings a bit further.
    phoneme_ids = [43, 62, 83, 61, 62, 47, 76, 46, 76, 56, 47, 12, 5, 90, 33, 61, 2]
    token_ids = [0] + phoneme_ids

    with torch.no_grad():
        out = albert_forward(token_ids, sd, HP)  # (T,768) numpy

    np.save(out_dir / "ref_styletts2_albert_tokens.npy", np.array(token_ids, dtype=np.int32))
    np.save(out_dir / "ref_styletts2_albert_out.npy", out)
    print(f"tokens={token_ids}, out shape={out.shape}, mean={out.mean():.6f}, std={out.std():.6f}")


if __name__ == "__main__":
    main()
