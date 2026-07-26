"""Hand-rolled pure-PyTorch reference for the MIL-traced "albert_bert_encoder" combined topology
(export_kokoro_mil.py's AlbertBertEncoderWrapper): CustomAlbert (reuses reference_forward_kokoro_albert.py's
own albert_forward, already verified against the bespoke topology in test_e2e_kokoro_albert.cpp) followed
by bert_encoder's plain Linear(768,512) -- but, UNLIKE reference_forward_kokoro_bert_encoder.py's own
reference, deliberately WITHOUT the real code's final `.transpose(-1,-2)`. AlbertBertEncoderWrapper's own
docstring (export_kokoro_mil.py) explains why: a bare permute as a traced graph's own declared output is a
live, non-contiguous view this project's exporter reads in PRE-permute order (the exact bug
export_vits_mil.py's StatsWrapper already found for VITS's `stats` output), so the MIL wrapper returns the
natural (T,512) time-major layout instead and kokoro_driver_mil.lua does the transpose (as an index-order
choice, not a real transpose) on the Lua side. This reference matches that convention: output is (T,512),
NOT the (512,T) `reference_forward_kokoro_bert_encoder.py` produces.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_kokoro_albert import HP
from reference_forward_kokoro_albert import albert_forward


def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <kokoro-v1_0.pth> <out_dir>", file=sys.stderr)
        sys.exit(1)
    ckpt_path, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_all = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    bert_sd = sd_all["bert"]
    be_sd = sd_all["bert_encoder"]
    w = be_sd["module.weight"]
    b = be_sd["module.bias"]

    # Same real BOS/phoneme/EOS convention as reference_forward_kokoro_albert.py's own main(), a
    # different (longer) phoneme sequence to also exercise position embeddings beyond a handful of rows.
    phoneme_ids = [43, 62, 83, 61, 62, 47, 76, 46, 76, 56, 47, 12, 5, 90, 33, 61, 2]
    token_ids = [0] + phoneme_ids + [0]

    with torch.no_grad():
        bert_dur = albert_forward(token_ids, bert_sd, HP)  # (T,768) numpy
        d_en = F.linear(torch.from_numpy(bert_dur), w, b)  # (T,512) -- NOT transposed, see module docstring

    def save(name, arr):
        np.save(out_dir / f"{name}.npy", np.ascontiguousarray(arr))

    save("ref_albert_bert_encoder_tokens", np.array(token_ids, dtype=np.int32))
    save("ref_albert_bert_encoder_out", d_en.numpy())
    print(f"tokens={token_ids}, out shape={tuple(d_en.shape)} (expect ({len(token_ids)},512))")


if __name__ == "__main__":
    main()
