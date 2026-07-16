#!/usr/bin/env python3
"""Writes the Milestone-1 toy LLM (see toy_llm_common.py) as a .gguf file: weights + hyperparameters
under the "loom." KV namespace + the JSON graph topology under "model.graph_topology". Deterministic
(fixed seed, no network) -- used by test_e2e_toy_llm.cpp, verified numerically against
reference_forward.py's independent numpy computation of the identical weights/topology.

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_llm_common as common


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("toy_llm.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = GGUFWriter(str(out_path), "loom-toy-llm")
    w.add_string("loom.architecture", "toy_llm")
    hp = common.hparams()
    for key in ("n_vocab", "n_embd", "n_layer", "n_head", "n_head_kv", "n_embd_head_k",
                "n_embd_head_v", "n_ff", "n_ctx_train", "rope_dims"):
        w.add_uint32(f"loom.{key}", hp[key])
    for key in ("rope_freq_base", "rope_freq_scale", "rms_norm_eps"):
        w.add_float32(f"loom.{key}", hp[key])
    w.add_string("model.graph_topology", common.topology_json())

    for name, array in common.generate_weights().items():
        w.add_tensor(name, array)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
