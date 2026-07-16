#!/usr/bin/env python3
"""Writes the Milestone-3 toy flow-matching vector-field network (see toy_ode_common.py) as a .gguf
file: weights + hyperparameters under "loom." + the JSON graph topology under "model.graph_topology",
plus the initial latent noise and conditioning embedding baked in as ordinary named tensors
("initial_latent.data", "conditioning.data") -- same rationale as Milestone 2's baked-in image/features:
test_e2e_toy_ode.cpp reads them back out of the Symbol Table via GgufModel rather than regenerating them
independently, guaranteeing byte-identical values with reference_forward_ode.py without needing a second
data-transfer mechanism.

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_ode_common as common


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("toy_ode.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = GGUFWriter(str(out_path), "loom-toy-ode")
    w.add_string("loom.architecture", "toy_flow_matching_vector_field")
    w.add_uint32("loom.n_embd", common.hparams()["n_embd"])
    w.add_string("model.graph_topology", common.topology_json())

    for name, array in common.generate_weights().items():
        w.add_tensor(name, array)
    w.add_tensor("initial_latent.data", common.generate_initial_latent())
    w.add_tensor("conditioning.data", common.generate_conditioning())

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
