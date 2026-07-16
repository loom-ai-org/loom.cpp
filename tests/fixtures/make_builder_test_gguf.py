#!/usr/bin/env python3
"""Generates a tiny attention-free toy model GGUF for test_graph_builder_shapes.cpp.

Deliberately has no ATTENTION/KV-cache dependency (Phase 2 doesn't have those primitives yet): just an
embedding lookup, `n_layer` repeated (RMSNorm -> elementwise scale -> linear -> residual add) blocks, and
a final output projection. Exists purely to exercise GraphBuilder's repeat_for expansion, {i} tensor-name
substitution, and symbol resolution against a real GGUF-loaded model.

Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

N_VOCAB, N_EMBD, N_LAYER = 6, 4, 2


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("builder_test.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    def rnd(*shape):
        return rng.normal(scale=0.1, size=shape).astype(np.float32)

    w = GGUFWriter(str(out_path), "loom-builder-test-fixture")
    w.add_string("loom.architecture", "builder_test")
    w.add_uint32("loom.n_vocab", N_VOCAB)
    w.add_uint32("loom.n_embd", N_EMBD)
    w.add_uint32("loom.n_layer", N_LAYER)
    w.add_float32("loom.rms_norm_eps", 1e-5)

    topology = {
        "version": 1,
        "inputs": [{"name": "tokens", "dtype": "i32", "shape": ["n_tokens"]}],
        "output": "logits",
        "nodes": [
            {"op": "GET_ROWS", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"]},
            {"repeat_for": "$n_layer", "index_var": "i", "nodes": [
                {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["normed"], "attrs": {"eps": "$rms_norm_eps"}},
                {"op": "MUL", "inputs": ["normed", "blk.{i}.norm.weight"], "outputs": ["normed"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.ffn.weight", "normed"], "outputs": ["ffn_out"]},
                {"op": "ADD", "inputs": ["cur", "ffn_out"], "outputs": ["cur"]},
            ]},
            {"op": "MUL_MAT", "inputs": ["output.weight", "cur"], "outputs": ["logits"]},
        ],
    }
    w.add_string("model.graph_topology", json.dumps(topology))

    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))
    for i in range(N_LAYER):
        w.add_tensor(f"blk.{i}.norm.weight", rnd(N_EMBD))
        w.add_tensor(f"blk.{i}.ffn.weight", rnd(N_EMBD, N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
