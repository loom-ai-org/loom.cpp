#!/usr/bin/env python3
"""
Conversion tool for Liquid AI's LFM2-350M model from Hugging Face format
to Loom's self-contained GGUF format (containing weights, topologies, and Lua driver).

Requires: coremltools, torch, numpy, gguf, transformers
Usage: python3 tools/convert_lfm/make_lfm2_gguf.py /home/flavio/Dev/models/lfm2-350m lfm2_350m.gguf
"""

import sys
import types
from pathlib import Path

# 1. Bypassing the transformers library hf-hub upper bound bounds-check to import safely
mock_dep = types.ModuleType("dependency_versions_check")
mock_dep.dep_version_check = lambda *args, **kwargs: None
sys.modules["transformers.dependency_versions_check"] = mock_dep

import os
import torch
import numpy as np
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Program, types as mil_types
from transformers import AutoModelForCausalLM, AutoTokenizer

# Insert loom_mil_compiler into search path. Importing it applies the coremltools torch-frontend
# patches (robust cast + GQA-aware SDPA decomposition) as an import-time side effect -- see
# tools/loom_mil_compiler/torch_patches.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loom_mil_compiler


class EmbeddingSubmodule(torch.nn.Module):
    def __init__(self, embed_tokens):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, tokens):
        return self.embed_tokens(tokens)


class OutputHeadSubmodule(torch.nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, x):
        return self.lm_head(x)


class LayerSubmodule(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states):
        # Pass dummy position embeddings with the correct shapes
        # (LFM2-350M has hidden_size=1024, 16 heads => 64 dimensions per head)
        pos_enc1 = torch.zeros(1, 1, 64)
        pos_enc2 = torch.zeros(1, 1, 64)
        return self.layer(hidden_states, position_embeddings=(pos_enc1, pos_enc2))


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/flavio/Dev/models/lfm2-350m"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "lfm2_350m.gguf"
    
    print(f"Loading LFM2-350M from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    
    master_prog = Program()
    
    # 2. Convert and attach the embedding submodule
    print("Exporting embedding submodule...")
    emb_sub = EmbeddingSubmodule(model.model.embed_tokens)
    dummy_tokens = torch.zeros((1, 4), dtype=torch.long)
    traced_emb = torch.jit.trace(emb_sub, (dummy_tokens,))
    
    emb_mil = ct.convert(
        traced_emb,
        inputs=[ct.TensorType(shape=(1, ct.RangeDim(1, 4096)), dtype=np.int32)],
        convert_to="milinternal"
    )
    master_prog.functions["embedding"] = emb_mil.functions["main"]
    
    # 3. Convert and attach layer submodules (16 blocks)
    print("Exporting 16 decoder layers...")
    dummy_states = torch.zeros((1, 4, 1024), dtype=torch.float32)
    
    for i in range(16):
        print(f"  Processing layer {i}/15...")
        layer = model.model.layers[i]
        layer_sub = LayerSubmodule(layer)
        traced_layer = torch.jit.trace(layer_sub, (dummy_states,))
        
        layer_mil = ct.convert(
            traced_layer,
            inputs=[ct.TensorType(shape=(1, ct.RangeDim(1, 4096), 1024), dtype=np.float32)],
            convert_to="milinternal"
        )
        master_prog.functions[f"layer_{i}"] = layer_mil.functions["main"]
        
    # 4. Convert and attach output head
    print("Exporting output projection head...")
    head_sub = OutputHeadSubmodule(model.lm_head)
    traced_head = torch.jit.trace(head_sub, (dummy_states,))
    
    head_mil = ct.convert(
        traced_head,
        inputs=[ct.TensorType(shape=(1, ct.RangeDim(1, 4096), 1024), dtype=np.float32)],
        convert_to="milinternal"
    )
    master_prog.functions["output_head"] = head_mil.functions["main"]
    
    # 5. Define main orchestration logic in MIL AST
    # This will be transpiled into our embedded Lua driver script.
    print("Constructing master generation orchestration function...")
    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, 4), dtype=mil_types.int32)
    ])
    def main_generation(tokens):
        # Placeholders that we will swap:
        x = mb.add(x=tokens, y=1, name="embedding")
        x0 = mb.add(x=x, y=1, name="layer_0")
        x1 = mb.add(x=x0, y=1, name="layer_1")
        x2 = mb.add(x=x1, y=1, name="layer_2")
        x3 = mb.add(x=x2, y=1, name="layer_3")
        x4 = mb.add(x=x3, y=1, name="layer_4")
        x5 = mb.add(x=x4, y=1, name="layer_5")
        x6 = mb.add(x=x5, y=1, name="layer_6")
        x7 = mb.add(x=x6, y=1, name="layer_7")
        x8 = mb.add(x=x7, y=1, name="layer_8")
        x9 = mb.add(x=x8, y=1, name="layer_9")
        x10 = mb.add(x=x9, y=1, name="layer_10")
        x11 = mb.add(x=x10, y=1, name="layer_11")
        x12 = mb.add(x=x11, y=1, name="layer_12")
        x13 = mb.add(x=x12, y=1, name="layer_13")
        x14 = mb.add(x=x13, y=1, name="layer_14")
        x15 = mb.add(x=x14, y=1, name="layer_15")
        
        logits = mb.add(x=x15, y=1, name="output_head")
        next_token = mb.add(x=logits, y=1, name="argmax")
        return next_token
        
    class MockOperation:
        def __init__(self, op_type, inputs, outputs, blocks=None):
            self.op_type = op_type
            self.inputs = inputs
            self.outputs = outputs
            self.blocks = blocks or []

    main_ops = list(main_generation.functions["main"].operations)
    for idx, op in enumerate(main_ops):
        out_name = op.outputs[0].name
        
        # Safely find the first input Var dynamically
        input_var = None
        from coremltools.converters.mil.mil import Var
        for k, v in op.inputs.items():
            if isinstance(v, Var):
                input_var = v
                break
                
        if out_name == "embedding":
            main_ops[idx] = MockOperation(op_type="embedding", inputs={"tokens": input_var}, outputs=op.outputs)
        elif out_name.startswith("layer_"):
            layer_num = out_name.split("_")[1]
            main_ops[idx] = MockOperation(op_type=f"layer_{layer_num}", inputs={"hidden_states": input_var}, outputs=op.outputs)
        elif out_name == "output_head":
            main_ops[idx] = MockOperation(op_type="output_head", inputs={"x": input_var}, outputs=op.outputs)
        elif out_name == "argmax":
            main_ops[idx] = MockOperation(op_type="argmax", inputs={"x": input_var}, outputs=op.outputs)

    main_generation.functions["main"].operations = main_ops
    master_prog.functions["main"] = main_generation.functions["main"]
    
    # 6. Compile and package everything to the final GGUF file
    print(f"Compiling MIL Program to unified GGUF target: {out_path}...")
    backend = loom_mil_compiler.LoomGGUFBackend()
    backend(
        master_prog,
        output_path=out_path,
        architecture="lfm2"
    )
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()
