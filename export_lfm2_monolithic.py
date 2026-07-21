#!/usr/bin/env python3
"""
Exports LFM2-350M as a single Monolithic GGUF model.

Usage:
  ~/.venvs/piper/bin/python3 export_lfm2_monolithic.py
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

class MonolithicModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        # Return only the logits tensor to satisfy PyTorch's tracing constraints
        outputs = self.model(tokens)
        return outputs.logits

def main():
    model_dir = "/home/flavio/Dev/models/lfm2-350m"
    out_path = "lfm2_350m_monolithic.gguf"

    print(f"Loading LFM2-350M from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    wrapper = MonolithicModelWrapper(model)

    # Trace the entire monolithic PyTorch model at once
    print("Tracing the complete monolithic PyTorch graph...")
    dummy_tokens = torch.zeros((1, 128), dtype=torch.long)
    traced_model = torch.jit.trace(wrapper, (dummy_tokens,))
    
    # Compile to GGUF using Loom dynamic backend with monolithic profiling
    print("Compiling to GGUF (Monolithic profile)...")
    mil_prog = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=(1, 128), dtype=np.int32)],
        convert_to="milinternal"
    )
    
    backend = loom_mil_compiler.LoomGGUFBackend()
    backend(
        mil_prog,
        output_path=out_path,
        architecture="lfm2",
        profile="monolithic"
    )
    print(f"SUCCESS! Monolithic model exported cleanly to: {out_path}")

if __name__ == "__main__":
    main()
