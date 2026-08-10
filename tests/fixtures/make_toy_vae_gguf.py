#!/usr/bin/env python3
"""Writes the Milestone-3 toy VAE decoder (see toy_vae_common.py) as a .gguf file: weights + the JSON
graph topology under "model.graph_topology", plus the input latent baked in as an ordinary named tensor
("latent.data") -- same rationale as the ODE fixture's baked-in initial state / Milestone 2's baked-in
image/features: test_e2e_toy_vae.cpp doesn't need to supply any runtime input at all, since this topology
declares zero graph inputs (everything -- weights and the fixed latent -- is resolved from the Symbol
Table).

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_vae_common as common


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("toy_vae.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = GGUFWriter(str(out_path), "loom-toy-vae")
    w.add_string("loom.architecture", "toy_vae_decoder")
    w.add_string("model.graph_topology", common.topology_json())

    for name, array in common.generate_weights().items():
        w.add_tensor(name, array)
    w.add_tensor("latent.data", common.generate_latent())

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
