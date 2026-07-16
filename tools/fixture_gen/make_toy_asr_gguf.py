#!/usr/bin/env python3
"""Writes the Milestone-2 toy ASR encoder (see toy_asr_common.py) as a .gguf file: weights +
hyperparameters under "loom." + the JSON graph topology under "model.graph_topology", plus the synthetic
input features baked in as an ordinary named tensor ("features.data") -- same rationale as
make_toy_vision_gguf.py's baked-in image.

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_asr_common as common


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("toy_asr.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = GGUFWriter(str(out_path), "loom-toy-asr")
    w.add_string("loom.architecture", "toy_asr_encoder")
    hp = common.hparams()
    for key in ("n_embd", "n_layer", "n_head", "n_embd_head", "n_ff"):
        w.add_uint32(f"loom.{key}", hp[key])
    w.add_float32("loom.rms_norm_eps", hp["rms_norm_eps"])
    w.add_string("model.graph_topology", common.topology_json())

    for name, array in common.generate_weights().items():
        w.add_tensor(name, array)
    w.add_tensor("features.data", common.generate_features())

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
