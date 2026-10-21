#!/usr/bin/env python3
"""Generates the GGUF test_cpu_repack_routing.cpp loads: one Q4_0 and one Q8_0 matrix, each read three
ways by the topology that test builds inline (MUL_MAT, GET_ROWS, and GET_ROWS through a RESHAPE view),
plus an F32 table the activations are looked up from.

32 rows of 64 so that ggml's CPU_REPACK has a layout for both types wherever it has kernels for them
(the interleave needs the row count to be a multiple of 4 or 8, and K a whole number of blocks).

Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter, quantize

N_ROWS, K, N_ACT = 32, 64, 8


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("repack_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    w = GGUFWriter(str(out_path), "loom-repack-test-fixture")
    w.add_string("loom.architecture", "repack_test")
    # The loader requires a topology; the test builds its own against these weights.
    w.add_string("model.graph_topology", json.dumps({
        "version": 1,
        "inputs": [{"name": "tokens", "dtype": "i32", "shape": ["n_tokens"]}],
        "output": "x",
        "nodes": [{"op": "GET_ROWS", "inputs": ["act.weight", "tokens"], "outputs": ["x"]}],
    }))
    w.add_tensor("act.weight", rng.normal(size=(N_ACT, K)).astype(np.float32))
    for name, qtype in (("w_q4_0.weight", GGMLQuantizationType.Q4_0), ("w_q8_0.weight", GGMLQuantizationType.Q8_0)):
        q = quantize(rng.normal(scale=0.1, size=(N_ROWS, K)).astype(np.float32), qtype)
        w.add_tensor(name, q, raw_dtype=qtype)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
