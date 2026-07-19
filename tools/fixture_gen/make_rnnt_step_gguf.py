#!/usr/bin/env python3
"""Writes the tiny synthetic plain-RNN-T fixture (see rnnt_step_common.py) as GGUFs -- lstm_h/lstm_c per
LSTM layer, plus one joint -- sharing identical weights, for test_rnnt_decoder.cpp to drive via
loom::TdtDecoder's plain-RNN-T mode (empty `durations`).

Usage: python3 make_rnnt_step_gguf.py <out_dir>
Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rnnt_step_common as common


def write_one(out_path: Path, topology: dict, weights: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out_path), "loom-rnnt-step-test-fixture")
    w.add_string("loom.architecture", "rnnt_step_test")
    w.add_string("model.graph_topology", json.dumps(topology))
    for name, array in weights.items():
        w.add_tensor(name, array)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rnnt_step_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = common.generate_weights()
    for layer in range(common.N_LSTM_LAYERS):
        write_one(out_dir / f"rnnt_lstm_h_{layer}.gguf", common.build_lstm_topology(layer, "h_new"), weights)
        write_one(out_dir / f"rnnt_lstm_c_{layer}.gguf", common.build_lstm_topology(layer, "c_new"), weights)
    write_one(out_dir / "rnnt_joint.gguf", common.build_joint_topology(), weights)


if __name__ == "__main__":
    main()
