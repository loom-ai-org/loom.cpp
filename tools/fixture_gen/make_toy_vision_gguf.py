#!/usr/bin/env python3
"""Writes the Milestone-2 toy vision encoder (see toy_vision_common.py) as a .gguf file: weights +
hyperparameters under "loom." + the JSON graph topology under "model.graph_topology", plus the synthetic
input image itself baked in as an ordinary named tensor ("image.data") -- since this fixture always runs
on the same fixed image, there's no need for the C++ test to supply it at runtime like a real input would
be; it's resolved from the Symbol Table exactly like any other weight.

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_vision_common as common


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("toy_vision.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = GGUFWriter(str(out_path), "loom-toy-vision")
    w.add_string("loom.architecture", "toy_vision_encoder")
    hp = common.hparams()
    for key in ("n_embd", "n_layer", "n_head", "n_embd_head", "n_ff"):
        w.add_uint32(f"loom.{key}", hp[key])
    w.add_float32("loom.rms_norm_eps", hp["rms_norm_eps"])
    w.add_string("model.graph_topology", common.topology_json())

    for name, array in common.generate_weights().items():
        w.add_tensor(name, array)
    w.add_tensor("image.data", common.generate_image())

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
