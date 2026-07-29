#!/usr/bin/env python3
"""Generates a minimal, deterministic 1-tensor GGUF fixture for the Phase-1 GgufModel smoke test.

This does NOT represent a real model architecture -- it exists purely to exercise GgufModel::load(),
weight()/has_weight(), hparam_u32()/hparam_f32()/architecture(), topology_json(), and hparam_env(). The
full toy-LLM fixture (tools/fixture_gen/make_toy_llm_gguf.py) lands in Phase 4.

Requires: pip install gguf numpy
"""
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

# Kept in sync with the C++ test's expectation of the raw topology string (tests/test_gguf_model_load.cpp).
TOPOLOGY_JSON = '{"version": 1, "nodes": []}'
# A second, NAMED topology alongside the bare one -- exercises GgufModel's multi-topology support
# (LOOM_PROCEDURAL_GENERALIZATION.md) in the SAME file as the ordinary single-topology case.
OTHER_TOPOLOGY_JSON = '{"version": 1, "nodes": [], "note": "other"}'


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("minimal.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = GGUFWriter(str(out_path), "loom-minimal-fixture")
    w.add_string("loom.architecture", "minimal_test")
    w.add_uint32("loom.n_layer", 2)
    w.add_float32("loom.scale", 0.5)
    w.add_string("model.graph_topology", TOPOLOGY_JSON)
    w.add_string("model.graph_topology.other", OTHER_TOPOLOGY_JSON)

    # Array/bool KVs, purely to exercise GgufModel's generic kv_*/kv_arr_*() accessors (added for the
    # tokenizer.ggml.* vocab schema) -- not otherwise meaningful test data.
    w.add_bool("test.flag_true", True)
    w.add_array("test.arr_str", ["alpha", "beta", "gamma"])
    w.add_array("test.arr_f32", [1.5, 2.5, 3.5])
    w.add_array("test.arr_i32", [10, -20, 30])

    # ggml tensors store ne[0] as the fastest-varying dimension; gguf-py's add_tensor reverses a numpy
    # array's shape into ggml's ne order, so a (3, 4) numpy array (row-major, last axis fastest) lands
    # as ne = [4, 3] with byte-identical layout -- no transpose needed to keep the two conventions
    # aligned for this simple case.
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    w.add_tensor("test.weight", weight)

    # Content-addressed alias, exercising GgufModel::load's "loom.tensor_alias.*" resolution
    # (BACKLOG.md P0.2) without needing a real duplicate-payload model: "test.weight_alias" is declared
    # as an alias of "test.weight" and deliberately has NO tensor_info of its own in this file -- only
    # weight("test.weight_alias") resolving to the exact same ggml_tensor* as weight("test.weight")
    # proves the C++ read path, not just that the writer emitted the KV pair.
    w.add_array("loom.tensor_alias.names", ["test.weight_alias"])
    w.add_array("loom.tensor_alias.targets", ["test.weight"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
