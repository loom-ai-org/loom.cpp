#!/usr/bin/env python3
"""Same real Conformer-CTC checkpoint/weights as tools/convert_nemo/convert_conformer_ctc.py, but the JSON
graph topology comes from walking a real torch.export() ATen graph of
tools/convert_generic/conformer_ctc_module.py's ConformerCTC through aten_to_loom's generic converter --
first real test of that converter's op-mapping table against a genuinely non-decoder-transformer-shaped
model (BACKLOG.md's gating-criterion note). Static shapes only (matching test_e2e_conformer_ctc.cpp's own
fixed kNSamples=10240 defaults) -- this POC does not attempt the real hand-written topology's dynamic-length
support.

Declared inputs are "mel_input"/"pos_emb_raw"/"kq_mask" (skips the mel frontend -- see
conformer_ctc_module.py's module docstring for why). Also writes mel_input.bin/pos_emb_raw.bin (same
waveform seed/n_samples as tools/convert_nemo/reference_forward_conformer.py's own fixture) into the output
directory, so the new e2e test can feed real values while comparing against that *existing* fixture's own
expected_logits.bin (same weights => byte-identical expected output, same "reuse the existing reference"
pattern as test_e2e_toy_llm_generic.cpp/test_e2e_qwen3_generic.cpp).

Usage: python3 make_conformer_ctc_gguf_generic.py <model.nemo> <out_dir>
Requires: pip install torch gguf numpy pyyaml
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "convert_nemo"))
import mel_common
import nemo_common as common
import reference_forward_conformer as ref
from aten_to_loom import Converter, _qualname_to_gguf_name
from conformer_ctc_module import ConformerCTC


def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <model.nemo> <out_dir>", file=sys.stderr)
        sys.exit(1)
    nemo_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ConformerCTC(nemo_path).eval()
    hp = model.hp
    hp.update(mel_common.mel_hparams(hp["feat_in"]))
    hp["n_mels"] = hp["feat_in"]

    n_samples = 10240  # must match test_e2e_conformer_ctc.cpp's kNSamples exactly.
    rng = np.random.default_rng(2024)  # must match reference_forward_conformer.py's own seed exactly.
    waveform = rng.normal(scale=0.1, size=n_samples).astype(np.float32)
    mel = ref.compute_mel_features(waveform, hp)  # (T_mel, n_mels)
    t_mel, n_mels = mel.shape

    n_sub = (((t_mel + 2 - 3) // 2 + 1) + 2 - 3) // 2 + 1  # same stride-2/pad-1/kernel-3 formula, twice.
    n_pos = 2 * n_sub - 1
    pos_emb_raw = ref.sinusoidal_pos_emb(n_sub, hp["n_embd"])  # (n_pos, n_embd)

    mel_input = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(torch.float32)  # [1,1,T_mel,n_mels]
    pos_emb_t = torch.from_numpy(pos_emb_raw).to(torch.float32)
    kq_mask_t = torch.zeros(n_sub, n_sub, dtype=torch.float32)

    ep = torch.export.export(model, (mel_input, pos_emb_t, kq_mask_t))
    input_specs = {
        "mel_input": ("f32", [str(n_mels), str(t_mel), "1", "1"]),
        "pos_emb_raw": ("f32", [str(hp["n_embd"]), str(n_pos)]),
        "kq_mask": ("f32", [str(n_sub), str(n_sub)]),
    }
    topo = Converter(example_n_tokens=-1).convert(ep, input_specs)

    out_path = out_dir / "conformer_ctc_generic.gguf"
    w = GGUFWriter(str(out_path), "loom-conformer-ctc-generic")
    w.add_string("loom.architecture", "conformer_ctc")
    for key in ("n_layers", "n_embd", "n_head", "head_dim", "ff_hidden", "conv_kernel_size", "conv_padding"):
        w.add_uint32(f"loom.{key}", hp[key])
    w.add_float32("loom.ln_eps", hp["ln_eps"])
    w.add_uint32("loom.n_samples", n_samples)
    w.add_uint32("loom.n_subsampled", n_sub)
    w.add_uint32("loom.n_pos", n_pos)
    w.add_uint32("loom.num_classes", hp["num_classes"])
    w.add_string("model.graph_topology", json.dumps(topo))

    for name, param in model.state_dict().items():
        gguf_name = _qualname_to_gguf_name(name)
        w.add_tensor(gguf_name, param.detach().numpy())

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    mel.astype(np.float32).tofile(out_dir / "mel_input.bin")
    pos_emb_raw.astype(np.float32).tofile(out_dir / "pos_emb_raw.bin")

    print(f"wrote {out_path} ({len(topo['nodes'])} topology nodes), t_mel={t_mel}, n_subsampled={n_sub}, n_pos={n_pos}")


if __name__ == "__main__":
    main()
