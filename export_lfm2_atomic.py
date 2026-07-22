#!/usr/bin/env python3
"""
Exports LFM2-350M as an Automatically Partitioned "Atomic" GGUF model.

Usage:
  ~/.venvs/piper/bin/python3 export_lfm2_atomic.py
"""
import sys
import types
from pathlib import Path

# Bypass the transformers library hf-hub bounds check to import safely
mock_dep = types.ModuleType("dependency_versions_check")
mock_dep.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = mock_dep

import os
import torch
import numpy as np
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from transformers import AutoModelForCausalLM

# 1.1 Monkey patch coremltools' PyTorch frontend cast operator to support 1-element numpy array conversion
from coremltools.converters.mil.frontend.torch import ops as mil_ops
_original_cast = mil_ops._cast

def _robust_cast(context, node, dtype, dtype_name):
    inputs = mil_ops._get_inputs(context, node, expected=1)
    x = inputs[0]
    if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray):
        if x.val.size == 1:
            scalar_val = dtype(x.val.item())
            res = mb.const(val=scalar_val, name=node.name)
            context.add(res, node.name)
            return
    _original_cast(context, node, dtype, dtype_name)

mil_ops._cast = _robust_cast

# 1.2 Monkey patch coremltools' SDPA decomposition to support GQA (Grouped Query Attention) tiling
from coremltools.converters.mil.frontend import _utils as mil_frontend_utils
_original_decompose_sdpa = mil_frontend_utils._decompose_scaled_dot_product_attention

def _robust_decompose_sdpa(q, k, v, mask, name, scale=None, before_op=None):
    q_shape = list(q.shape)
    k_shape = list(k.shape)
    rank = len(q_shape)

    if rank == 4:
        q_heads = q_shape[1]
        k_heads = k_shape[1]
        if isinstance(q_heads, int) and isinstance(k_heads, int) and q_heads != k_heads:
            ratio = q_heads // k_heads
            if ratio > 1:
                k = mb.tile(x=k, reps=[1, ratio, 1, 1], before_op=before_op)
                v = mb.tile(x=v, reps=[1, ratio, 1, 1], before_op=before_op)
    elif rank == 3:
        q_heads = q_shape[0]
        k_heads = k_shape[0]
        if isinstance(q_heads, int) and isinstance(k_heads, int) and q_heads != k_heads:
            ratio = q_heads // k_heads
            if ratio > 1:
                k = mb.tile(x=k, reps=[ratio, 1, 1], before_op=before_op)
                v = mb.tile(x=v, reps=[ratio, 1, 1], before_op=before_op)

    return _original_decompose_sdpa(q, k, v, mask, name, scale, before_op)

mil_frontend_utils._decompose_scaled_dot_product_attention = _robust_decompose_sdpa

# Add tools/ folder to search path for importing compiler
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import loom_mil_compiler  # Registers the "loom" backend

def _causal_mask(seq_len: int) -> torch.Tensor:
    # A real, already-prepared 4D additive mask short-circuits transformers' create_causal_mask /
    # _preprocess_mask_arguments entirely (`if isinstance(attention_mask, torch.Tensor) and
    # len(attention_mask.shape) == 4: return True, attention_mask, ...` -- "returned as-is"). That's
    # needed here for the SAME reason cache_position is now passed explicitly: the internal mask-building
    # path (masking_utils.py) derives kv_length from a Python-level `input_embeds.shape[1]` query when no
    # cache is used, which torch.jit.trace bakes in as the tracing dummy's fixed length regardless of
    # ct.RangeDim declared afterward (confirmed empirically -- a RESHAPE feeding off `cache_position`
    # stayed hardcoded to 128 even after cache_position itself became a genuine dynamic input).
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len)

class MonolithicModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, cache_position, attention_mask):
        outputs = self.model(tokens, cache_position=cache_position, attention_mask=attention_mask)
        return outputs.logits

def main():
    model_dir = "/home/flavio/Dev/models/lfm2-350m"
    out_path = "lfm2_350m_atomic.gguf"

    print(f"Loading LFM2-350M from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    wrapper = MonolithicModelWrapper(model)

    # Trace the entire monolithic PyTorch model at once. torch.jit.trace always needs one concrete
    # example shape -- the dynamic range is declared separately below via ct.convert's own `inputs=`,
    # matching tools/convert_lfm/make_lfm2_gguf.py's already-working per-submodule RangeDim tracing
    # (EXPORT-BACKLOG.md item 3: a fixed traced shape bakes a literal length into every exported slice,
    # forcing the driver to pad every prompt to that fixed length instead of using its real length).
    print("Tracing the complete monolithic PyTorch graph...")
    dummy_tokens = torch.zeros((1, 128), dtype=torch.long)
    dummy_cache_position = torch.arange(128, dtype=torch.long)
    dummy_attention_mask = _causal_mask(128)
    traced_model = torch.jit.trace(wrapper, (dummy_tokens, dummy_cache_position, dummy_attention_mask))

    # Compile to GGUF using Loom dynamic backend with atomic profiling. `tokens`/`cache_position`/
    # `attention_mask` share the SAME ct.RangeDim instance so coremltools ties them all to one symbolic
    # length (they must always be called with matching lengths at runtime) -- see
    # apply_monolithic_export/apply_atomic_export's own auto-generation of "cache_position"-named inputs
    # via loom.range(...) and "attention_mask"-named inputs via loom.causal_mask(...).
    print("Compiling to GGUF (Atomic profile)...")
    seq_len_dim = ct.RangeDim(1, 4096)
    mil_prog = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="tokens", shape=(1, seq_len_dim), dtype=np.int32),
            ct.TensorType(name="cache_position", shape=(seq_len_dim,), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, 1, seq_len_dim, seq_len_dim), dtype=np.float32),
        ],
        convert_to="milinternal"
    )

    backend = loom_mil_compiler.LoomGGUFBackend()
    backend(
        mil_prog,
        output_path=out_path,
        architecture="lfm2",
        profile="atomic"
    )
    print(f"SUCCESS! Atomic model exported cleanly to: {out_path}")

if __name__ == "__main__":
    main()
