#!/usr/bin/env python3
"""A one-layer ATTENTION module over a RING KV cache of 4 cells, for tests/ci/test_ring_kv_cache.cpp.

q, k and v are graph inputs. Every K is zero, so attention is uniform over whatever the step reads,
and the output is the MEAN of the V rows it attends to. The driver writes V = p + 1 at position p, one
token per step, so step p's output names its window exactly: the mean of the last min(p + 1, 4)
positions' values for a ring (ADR-066), the mean of all of them for a linear cache -- which with 4 cells
could not reach step 4 at all. `prefill` writes several tokens in one call, which is legal while the
ring has room and refused once it would wrap.

Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

DRIVER = """
-- `steps` single-token steps from position 0; returns every step's output.
function stepwise(inputs)
    local out = {}
    for p = 0, inputs.steps - 1 do
        loom.run_subgraph_and_retain('attn', {n_tokens = 1, n_past = p},
            {q = {1, 0}, k = {0, 0}, v = {p + 1, p + 1}, kq_mask = loom.causal_mask(1, math.min(p, 3))})
        out[#out + 1] = loom.get_output('attn', 1)[1]
    end
    return out
end

-- `n` tokens in ONE call at position `at`, then nothing else; returns the last row's output.
function prefill(inputs)
    local n, at = inputs.n, inputs.at
    local q, k, v = {}, {}, {}
    for i = 0, n - 1 do q[#q + 1] = 1; q[#q + 1] = 0; k[#k + 1] = 0; k[#k + 1] = 0
        v[#v + 1] = at + i + 1; v[#v + 1] = at + i + 1 end
    loom.run_subgraph_and_retain('attn', {n_tokens = n, n_past = at},
        {q = q, k = k, v = v, kq_mask = loom.causal_mask(n, at)})
    local o = loom.get_output('attn', 1)
    return {o[#o - 1]}
end
"""


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ring_kv.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out_path), "loom-ring-kv-fixture")
    w.add_string("loom.architecture", "ring_kv_test")
    for key, value in (("n_layer", 1), ("n_head", 1), ("n_head_kv", 1), ("n_embd_head_k", 2),
                       ("n_embd_head_v", 2), ("kv_cache_size", 4)):
        w.add_uint32(f"loom.{key}", value)
    w.add_bool("loom.kv_cache_ring", True)
    attn = {
        "version": 1,
        "inputs": [
            {"name": "q", "dtype": "f32", "shape": ["n_embd_head_k", "n_head", "n_tokens"]},
            {"name": "k", "dtype": "f32", "shape": ["n_embd_head_k", "n_head_kv", "n_tokens"]},
            {"name": "v", "dtype": "f32", "shape": ["n_embd_head_v", "n_head_kv", "n_tokens"]},
            {"name": "kq_mask", "dtype": "f32", "shape": ["n_kv", "n_tokens"]},
        ],
        "outputs": ["out"],
        "nodes": [{"op": "ATTENTION", "inputs": ["q", "k", "v", "kq_mask"], "outputs": ["out"],
                   "attrs": {"layer": 0, "scale": 1.0}}],
    }
    w.add_string("model.graph_topology.attn", json.dumps(attn))
    w.add_string("model.driver_script", DRIVER)
    w.add_tensor("test.placeholder", np.zeros(4, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
