#!/usr/bin/env python3
"""A bare two-layer ATTENTION module, for tests/ci/test_seed_kv.cpp.

**q, k and v are graph INPUTS**, so nothing but the cache decides what a step attends over: the test
seeds rows with `loom.seed_kv`, steps once with k = v = 0, and reads which seeded V each head picked.
Every seeded K is zero except one row per (layer, head), whose score against that head's one-hot query
is 50 -- so each head's output is one known seeded V row, to within exp(-50). The V values are all
distinct, so a row, head, layer or K/V swap in the seeding shows up as a wrong number rather than as a
plausible one.

The seed itself also ships as a weight (`seed.kv`), the form Pocket-TTS's built-in voice takes. And
beside the model, for tests/ci/test_voice_file.cpp, VOICE FILES (ADR-045) for it: `seed_kv_voice.gguf`
holds the same state with every V shifted by 2000 under the input name `kv`, and three that must be
refused -- made for other weights, for another architecture, and stored at F16.

Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

N_LAYER, N_HEAD, HEAD_DIM, KV_SIZE, N_SEED = 2, 2, 2, 8, 3
COMPAT = "5eedc0de" * 4
N_EMBD = N_HEAD * HEAD_DIM

# (layer, head) -> the seeded row whose K is hot. Different rows per head and per layer.
HOT = {(0, 0): 1, (0, 1): 2, (1, 0): 0, (1, 1): 1}

DRIVER = """
-- Seed from `inputs.kv` (a Lua array) or, without one, from the `seed.kv` weight; then one step with a
-- one-hot query per head and k = v = 0. Returns both layers' outputs, layer 0 first.
function seeded(inputs)
    local n = loom.seed_kv('attn', inputs.kv or 'seed.kv', inputs.n_rows)
    loom.run_subgraph_and_retain('attn', {n_tokens = 1, n_past = n},
        {q = {1, 0, 0, 1}, k = {0, 0, 0, 0}, v = {0, 0, 0, 0}, kq_mask = loom.causal_mask(1, n)})
    local out = {}
    for index = 1, 2 do
        local o = loom.get_output('attn', index)
        for i = 1, #o do out[#out + 1] = o[i] end
    end
    return out
end

-- The same call against a module with no ATTENTION node, which has no cache to seed.
function no_cache(inputs)
    loom.seed_kv('plain', 'seed.kv')
    return {0}
end
"""


def seed_values() -> np.ndarray:
    """`[layer][K then V][row][head * head_dim]`, flattened: the layout `loom.seed_kv` reads."""
    kv = np.zeros((N_LAYER, 2, N_SEED, N_HEAD, HEAD_DIM), dtype=np.float32)
    for (layer, head), row in HOT.items():
        kv[layer, 0, row, head, head] = 50.0          # K: head h's query is one-hot on dim h
    for layer in range(N_LAYER):
        for row in range(N_SEED):
            for head in range(N_HEAD):
                for d in range(HEAD_DIM):
                    kv[layer, 1, row, head, d] = 100 * layer + 10 * row + 2 * head + d + 1
    return kv.reshape(-1)


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("seed_kv.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(out_path), "loom-seed-kv-fixture")
    w.add_string("loom.architecture", "seed_kv_test")
    w.add_uint32("loom.n_layer", N_LAYER)
    w.add_uint32("loom.n_head", N_HEAD)
    w.add_uint32("loom.n_head_kv", N_HEAD)
    w.add_uint32("loom.n_embd_head_k", HEAD_DIM)
    w.add_uint32("loom.n_embd_head_v", HEAD_DIM)
    w.add_uint32("loom.kv_cache_size", KV_SIZE)
    # What a voice file must match to be loaded into this model.
    w.add_string("loom.voice.compat", COMPAT)

    attn = {
        "version": 1,
        "inputs": [
            {"name": "q", "dtype": "f32", "shape": ["n_embd_head_k", "n_head", "n_tokens"]},
            {"name": "k", "dtype": "f32", "shape": ["n_embd_head_k", "n_head_kv", "n_tokens"]},
            {"name": "v", "dtype": "f32", "shape": ["n_embd_head_v", "n_head_kv", "n_tokens"]},
            {"name": "kq_mask", "dtype": "f32", "shape": ["n_kv", "n_tokens"]},
        ],
        "outputs": ["out0", "out1"],
        "nodes": [
            {"op": "ATTENTION", "inputs": ["q", "k", "v", "kq_mask"], "outputs": ["out0"],
             "attrs": {"layer": 0, "scale": 1.0}},
            {"op": "ATTENTION", "inputs": ["q", "k", "v", "kq_mask"], "outputs": ["out1"],
             "attrs": {"layer": 1, "scale": 1.0}},
        ],
    }
    plain = {
        "version": 1,
        "inputs": [{"name": "x", "dtype": "f32", "shape": ["n_tokens"]}],
        "output": "y",
        "nodes": [{"op": "ADD", "inputs": ["x", "x"], "outputs": ["y"]}],
    }
    w.add_string("model.graph_topology.attn", json.dumps(attn))
    w.add_string("model.graph_topology.plain", json.dumps(plain))
    w.add_string("model.driver_script", DRIVER)
    w.add_tensor("seed.kv", seed_values())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    shifted = seed_values().reshape(N_LAYER, 2, -1).copy()
    shifted[:, 1] += 2000.0
    shifted = shifted.reshape(-1)
    for suffix, arch, compat, values in (
            ("", "seed_kv_test", COMPAT, shifted),
            ("_other_weights", "seed_kv_test", "0" * 32, shifted),
            ("_other_arch", "another_model", COMPAT, shifted),
            ("_f16", "seed_kv_test", COMPAT, shifted.astype(np.float16))):
        write_voice(out_path.parent / f"seed_kv_voice{suffix}.gguf", arch, compat, values)


def write_voice(path: Path, arch: str, compat: str, values: np.ndarray) -> None:
    w = GGUFWriter(str(path), "loom-voice")
    w.add_string("loom.voice.architecture", arch)
    w.add_string("loom.voice.compat", compat)
    w.add_string("loom.voice.name", "shifted")
    w.add_string("loom.voice.license", "CC0-1.0")
    w.add_tensor("kv", values)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
