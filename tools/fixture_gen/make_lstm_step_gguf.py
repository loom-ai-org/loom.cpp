#!/usr/bin/env python3
"""Writes the tiny synthetic LSTM-step fixture (see lstm_step_common.py) as two GGUFs -- one per output,
since GraphTopology only supports a single declared output -- sharing identical weights.

Usage: python3 make_lstm_step_gguf.py <out_h.gguf> <out_c.gguf>
Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lstm_step_common as common


def write_one(out_path: Path, output_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out_path), "loom-lstm-step-test-fixture")
    w.add_string("loom.architecture", "lstm_step_test")
    w.add_string("model.graph_topology", json.dumps(common.build_topology(output_name)))
    for name, array in common.generate_weights().items():
        w.add_tensor(name, array)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def main() -> None:
    out_h = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("lstm_step_h.gguf")
    out_c = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("lstm_step_c.gguf")
    write_one(out_h, "h_new")
    write_one(out_c, "c_new")


if __name__ == "__main__":
    main()
