#!/usr/bin/env python3
"""
Demonstration script to export and verify both "monolithic" and "atomic" GGUF profiles.
Creates a sequential multi-layered PyTorch model and compiles it to both targets.

Usage:
  ~/.venvs/piper/bin/python3 tools/convert_lfm/export_profiles_demo.py
"""

import os
import sys
from pathlib import Path

# Add tools/ folder to search path for importing compiler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import coremltools as ct
import loom_mil_compiler  # Dynamic backend

class LayeredModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Standard dynamic layers scoped as "layer_0" and "layer_1"
        self.layer_0 = torch.nn.Linear(4, 4)
        self.layer_1 = torch.nn.Linear(4, 4)
        
        # Hardcode weights for determinism
        self.layer_0.weight.data.fill_(1.0)
        self.layer_0.bias.data.fill_(0.5)
        self.layer_1.weight.data.fill_(1.5)
        self.layer_1.bias.data.fill_(0.1)

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        return x

def main():
    print("Initializing layered PyTorch model...")
    model = LayeredModel().eval()
    example_input = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    
    # Trace the PyTorch graph
    traced_model = torch.jit.trace(model, example_input)
    
    # Parse PyTorch model to CoreML MIL Program
    print("Parsing PyTorch model to MIL IR...")
    mil_prog = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=(1, ct.RangeDim(1, 4096), 4), dtype=np.float32)],
        convert_to="milinternal"
    )
    
    # 1. Export Monolithic Profile
    mono_path = "model_monolithic.gguf"
    print(f"\n--- Compiling to Monolithic GGUF: {mono_path} ---")
    backend = loom_mil_compiler.LoomGGUFBackend()
    backend(
        mil_prog,
        output_path=mono_path,
        profile="monolithic",
        architecture="profile_demo"
    )
    
    # 2. Export Atomic Profile
    atomic_path = "model_atomic.gguf"
    print(f"\n--- Compiling to Atomic GGUF: {atomic_path} ---")
    backend(
        mil_prog,
        output_path=atomic_path,
        profile="atomic",
        architecture="profile_demo"
    )
    
    print("\nSUCCESS! Both model profiles have been compiled and exported.")
    print(f"  - Monolithic: {mono_path}")
    print(f"  - Atomic:     {atomic_path}")

if __name__ == "__main__":
    main()
