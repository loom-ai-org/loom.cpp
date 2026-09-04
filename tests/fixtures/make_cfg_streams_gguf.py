#!/usr/bin/env python3
"""Two decode STREAMS of one model, for tests/ci/test_private_kv_cache.cpp.

**One topology, declared twice, and the second copy asks for its own KV cache.** That is the shape a
classifier-free-guidance decode has: the same decoder run once on the conditional input and once on
the unconditional one, every step, over two histories that must not see each other. Whether a module
needs a cache is derived from its graph; whether it needs its OWN cannot be, because two modules
running one topology are two streams when a driver runs them side by side and two phases when it runs
them in sequence. So the file says which, with `kv_cache_scope`.

The topology itself is `make_attention_test_gguf.py`'s, unchanged and deliberately -- what is under
test here is the cache wiring, not the graph, and a second graph would only be a second thing that
could be wrong. The driver interleaves the two streams so that a shared cache produces a WRONG answer
rather than an error.

Requires: pip install gguf numpy
"""
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

N_VOCAB, N_EMBD, N_HEAD, N_HEAD_KV, N_LAYER, N_FF, N_CTX_TRAIN = 8, 4, 2, 2, 1, 8, 16
HEAD_DIM = N_EMBD // N_HEAD
KV_SIZE = 16

# Two streams, one step at a time each, in strict alternation -- which is what makes a shared cache
# visible: at step t, stream B would write over the cell stream A just wrote and then attend to its
# own history plus A's.
#
# **Each step returns its LOGITS, not an argmax.** An argmax over a toy model this small is decided by
# the current token almost regardless of the history -- measured: with the argmax, sharing one cache
# between the two streams changed nothing at all, so the check that was supposed to prove the test
# could fail passed while the bug it describes was fully present. The logits are where the history
# actually shows.
DRIVER = """
local function step(module, tokens, n_past, out)
    local logits = loom.run_subgraph(module,
                                     {n_tokens = 1, n_past = n_past},
                                     {tokens = tokens,
                                      positions = {n_past},
                                      kq_mask = loom.causal_mask(1, n_past)})
    for i = 1, #logits do out[#out + 1] = logits[i] end
end

-- Both streams, alternating. `inputs.a` and `inputs.b` are the prompts, one token per step.
function interleaved(inputs)
    local a, b = {}, {}
    for t = 1, #inputs.a do
        step('cond', {inputs.a[t]}, t - 1, a)
        step('uncond', {inputs.b[t]}, t - 1, b)
    end
    -- Stream A's whole run, then stream B's -- the caller splits at the halfway point rather than
    -- de-interleaving, which is one fewer thing for the test to get wrong.
    for i = 1, #b do a[#a + 1] = b[i] end
    return a
end

-- One stream on its own, through whichever module the caller names. The oracle the interleaved run
-- has to match: a stream whose cache is private cannot tell that another stream ran at all.
function alone(inputs)
    local out = {}
    local module = inputs.uncond == 1 and 'uncond' or 'cond'
    for t = 1, #inputs.a do
        step(module, {inputs.a[t]}, t - 1, out)
    end
    return out
end
"""


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cfg_streams.gguf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    # scale 0.6 rather than the 0.1 the sibling attention fixture uses: at 0.1 the attention output is
    # a rounding error beside the residual, so a stream reading another stream's cells produced
    # logits that differed in the sixth decimal. The bug this fixture exists to expose has to be
    # bigger than the noise, and the weights are the knob for that.
    def rnd(*shape):
        return rng.normal(scale=0.6, size=shape).astype(np.float32)

    w = GGUFWriter(str(out_path), "loom-cfg-streams-fixture")
    w.add_string("loom.architecture", "cfg_streams_test")
    w.add_uint32("loom.n_vocab", N_VOCAB)
    w.add_uint32("loom.n_embd", N_EMBD)
    w.add_uint32("loom.n_layer", N_LAYER)
    w.add_uint32("loom.n_head", N_HEAD)
    w.add_uint32("loom.n_head_kv", N_HEAD_KV)
    w.add_uint32("loom.n_embd_head_k", HEAD_DIM)
    w.add_uint32("loom.n_embd_head_v", HEAD_DIM)
    w.add_uint32("loom.kv_cache_size", KV_SIZE)
    w.add_uint32("loom.n_ff", N_FF)
    w.add_uint32("loom.n_ctx_train", N_CTX_TRAIN)
    w.add_uint32("loom.rope_dims", HEAD_DIM)
    w.add_float32("loom.rope_freq_base", 10000.0)
    w.add_float32("loom.rope_freq_scale", 1.0)
    w.add_float32("loom.rms_norm_eps", 1e-5)

    topology = {
        "version": 1,
        "inputs": [
            {"name": "tokens", "dtype": "i32", "shape": ["n_tokens"]},
            {"name": "positions", "dtype": "i32", "shape": ["n_tokens"]},
            {"name": "kq_mask", "dtype": "f32", "shape": ["n_kv", "n_tokens"]},
        ],
        "output": "logits",
        "nodes": [
            {"op": "GET_ROWS", "inputs": ["token_embd.weight", "tokens"], "outputs": ["cur"]},
            {"repeat_for": "$n_layer", "index_var": "i", "nodes": [
                {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["attn_normed"], "attrs": {"eps": "$rms_norm_eps"}},
                {"op": "MUL", "inputs": ["attn_normed", "blk.{i}.attn_norm.weight"], "outputs": ["attn_normed"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_q.weight", "attn_normed"], "outputs": ["q"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_k.weight", "attn_normed"], "outputs": ["k"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_v.weight", "attn_normed"], "outputs": ["v"]},
                {"op": "RESHAPE", "inputs": ["q"], "outputs": ["q"], "attrs": {"shape": ["n_embd_head_k", "n_head", "n_tokens"]}},
                {"op": "RESHAPE", "inputs": ["k"], "outputs": ["k"], "attrs": {"shape": ["n_embd_head_k", "n_head_kv", "n_tokens"]}},
                {"op": "RESHAPE", "inputs": ["v"], "outputs": ["v"], "attrs": {"shape": ["n_embd_head_v", "n_head_kv", "n_tokens"]}},
                {"op": "ROPE", "inputs": ["q", "positions"], "outputs": ["q"], "attrs": {
                    "n_dims": "$rope_dims", "mode": 2, "n_ctx_orig": "$n_ctx_train",
                    "freq_base": "$rope_freq_base", "freq_scale": "$rope_freq_scale",
                    "ext_factor": 0.0, "attn_factor": 1.0, "beta_fast": 32.0, "beta_slow": 1.0,
                }},
                {"op": "ROPE", "inputs": ["k", "positions"], "outputs": ["k"], "attrs": {
                    "n_dims": "$rope_dims", "mode": 2, "n_ctx_orig": "$n_ctx_train",
                    "freq_base": "$rope_freq_base", "freq_scale": "$rope_freq_scale",
                    "ext_factor": 0.0, "attn_factor": 1.0, "beta_fast": 32.0, "beta_slow": 1.0,
                }},
                {"op": "ATTENTION", "inputs": ["q", "k", "v", "kq_mask"], "outputs": ["attn_out"],
                 "attrs": {"layer": "{i}", "scale": "1/sqrt($n_embd_head_k)"}},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.attn_output.weight", "attn_out"], "outputs": ["attn_proj"]},
                {"op": "ADD", "inputs": ["cur", "attn_proj"], "outputs": ["cur"]},
                {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["ffn_normed"], "attrs": {"eps": "$rms_norm_eps"}},
                {"op": "MUL", "inputs": ["ffn_normed", "blk.{i}.ffn_norm.weight"], "outputs": ["ffn_normed"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.ffn_gate.weight", "ffn_normed"], "outputs": ["ffn_gate"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.ffn_up.weight", "ffn_normed"], "outputs": ["ffn_up"]},
                {"op": "SWIGLU", "inputs": ["ffn_gate", "ffn_up"], "outputs": ["ffn_act"]},
                {"op": "MUL_MAT", "inputs": ["blk.{i}.ffn_down.weight", "ffn_act"], "outputs": ["ffn_out"]},
                {"op": "ADD", "inputs": ["cur", "ffn_out"], "outputs": ["cur"]},
            ]},
            {"op": "RMS_NORM", "inputs": ["cur"], "outputs": ["cur"], "attrs": {"eps": "$rms_norm_eps"}},
            {"op": "MUL", "inputs": ["cur", "output_norm.weight"], "outputs": ["cur"]},
            {"op": "MUL_MAT", "inputs": ["output.weight", "cur"], "outputs": ["logits"]},
        ],
    }
    # The conditional stream takes the session's shared cache -- the default, spelled by saying
    # nothing, which is what every file exported before `kv_cache_scope` existed does.
    w.add_string("model.graph_topology.cond", json.dumps(topology))
    # The unconditional stream is the same graph and its own cache.
    w.add_string("model.graph_topology.uncond", json.dumps(dict(topology, kv_cache_scope="private")))
    w.add_string("model.driver_script", DRIVER)


    w.add_tensor("token_embd.weight", rnd(N_VOCAB, N_EMBD))
    for i in range(N_LAYER):
        w.add_tensor(f"blk.{i}.attn_norm.weight", rnd(N_EMBD))
        w.add_tensor(f"blk.{i}.attn_q.weight", rnd(N_HEAD * HEAD_DIM, N_EMBD))
        w.add_tensor(f"blk.{i}.attn_k.weight", rnd(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(f"blk.{i}.attn_v.weight", rnd(N_HEAD_KV * HEAD_DIM, N_EMBD))
        w.add_tensor(f"blk.{i}.attn_output.weight", rnd(N_EMBD, N_HEAD * HEAD_DIM))
        w.add_tensor(f"blk.{i}.ffn_norm.weight", rnd(N_EMBD))
        w.add_tensor(f"blk.{i}.ffn_gate.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(f"blk.{i}.ffn_up.weight", rnd(N_FF, N_EMBD))
        w.add_tensor(f"blk.{i}.ffn_down.weight", rnd(N_EMBD, N_FF))
    w.add_tensor("output_norm.weight", rnd(N_EMBD))
    w.add_tensor("output.weight", rnd(N_VOCAB, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
