import os
import unittest
import numpy as np
import torch
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Program, types
from gguf import GGUFReader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the compiler to trigger the backend registration
import loom_mil_compiler

class TestLoomMILCompiler(unittest.TestCase):
    def setUp(self):
        self.output_path = "test_output.gguf"

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_pytorch_single_module_conversion(self):
        """
        Tests converting a standard PyTorch model with a single forward pass.
        Since it only contains a 'main' function, we verify it registers and compiles.
        """
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                # Hardcode weights for determinism
                self.linear.weight.data.fill_(1.5)
                self.linear.bias.data.fill_(0.5)

            def forward(self, x):
                return self.linear(x) + 2.0

        model = SimpleModel().eval()
        example_input = torch.rand(1, 4)
        
        traced = torch.jit.trace(model, example_input)
        
        # 1. Convert PyTorch model to MIL Program via ct.convert
        mil_prog = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=example_input.shape)],
            convert_to="milinternal"
        )
        
        # 2. Invoke our custom backend to generate the GGUF file
        backend = loom_mil_compiler.LoomGGUFBackend()
        mlmodel_path = backend(mil_prog, output_path=self.output_path, architecture="simple_test")
        
        self.assertEqual(mlmodel_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        
        # Verify GGUF contents
        reader = GGUFReader(self.output_path)
        
        # Verify architecture and driver script metadata exist
        self.assertIn("loom.architecture", reader.fields)
        self.assertEqual(reader.fields["loom.architecture"].parts[-1].tobytes().decode("utf-8"), "simple_test")
        
        self.assertIn("model.driver_script", reader.fields)
        driver_script = reader.fields["model.driver_script"].parts[-1].tobytes().decode("utf-8")
        self.assertIn("function main(inputs)", driver_script)

    def test_multi_modular_program_transpilation(self):
        """
        Constructs a multi-modular MIL Program with:
          - A submodule 'dense_layer' representing a heavy network layer (static topology).
          - A 'main' function with a loop and conditional branch calling the submodule (Lua transpiled).
        """
        class MockOperation:
            def __init__(self, op_type, inputs, outputs, blocks=None):
                self.op_type = op_type
                self.inputs = inputs
                self.outputs = outputs
                self.blocks = blocks or []

        prog = Program()
        
        # 1. Define the 'dense_layer' submodule
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 4), dtype=types.fp32)])
        def dense_layer(x):
            # A heavy constant weight matrix (treated as GGUF weight)
            w = mb.const(val=np.ones((4, 4), dtype=np.float32) * 1.5, name="w")
            y = mb.matmul(x=x, y=w, name="matmul_node")
            return mb.add(x=y, y=0.5, name="add_node")

        # Add the submodule to our Program
        prog.functions["dense_layer"] = dense_layer.functions["main"]

        # 2. Define the main function with control flow and submodule dispatch
        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.int32)])
        def main_func(loop_count):
            # A simple loop variable initialized
            i = mb.const(val=0, name="i")
            total = mb.const(val=np.zeros((4, 4), dtype=np.float32), name="total")
            
            # Use a placeholder add operation that we will swap
            sub_res = mb.add(x=total, y=1.0, name="sub_res")
            
            # Simple cond block
            pred = mb.const(val=True, name="pred")
            
            def true_fn():
                return mb.add(x=sub_res, y=1.0, name="total_add")
            def false_fn():
                return mb.sub(x=sub_res, y=1.0, name="total_sub")
                
            total = mb.cond(pred=pred, _true_fn=true_fn, _false_fn=false_fn, name="cond_node")
                
            return total

        # Swap the placeholder add op named "sub_res" with our MockOperation representing dense_layer
        main_operations = list(main_func.functions["main"].operations)
        for idx, op in enumerate(main_operations):
            if op.outputs[0].name == "sub_res":
                main_operations[idx] = MockOperation(
                    op_type="dense_layer",
                    inputs={"x": op.inputs["x"]},
                    outputs=op.outputs
                )
                break

        main_func.functions["main"].operations = main_operations

        prog.functions["main"] = main_func.functions["main"]

        # Run conversion to "loom"
        exporter = loom_mil_compiler.LoomGGUFBackend()
        exporter(prog, output_path=self.output_path, architecture="multi_modular_test")

        self.assertTrue(os.path.exists(self.output_path))
        
        # Verify GGUF contents
        reader = GGUFReader(self.output_path)
        
        # Check architecture
        self.assertEqual(reader.fields["loom.architecture"].parts[-1].tobytes().decode("utf-8"), "multi_modular_test")
        
        # Check that the driver script contains the conditional and submodule dispatch
        self.assertIn("model.driver_script", reader.fields)
        driver_script = reader.fields["model.driver_script"].parts[-1].tobytes().decode("utf-8")
        self.assertIn("if pred", driver_script)
        self.assertIn("loom.run_subgraph('dense_layer'", driver_script)

        # Check that the dense_layer submodule topology is serialized as metadata
        self.assertIn("model.graph_topology.dense_layer", reader.fields)
        topo_str = reader.fields["model.graph_topology.dense_layer"].parts[-1].tobytes().decode("utf-8")
        
        # Parse and verify the topology contents
        import json
        topo = json.loads(topo_str)
        self.assertEqual(topo["version"], 1)
        self.assertTrue(any(node["op"] == "MUL_MAT" for node in topo["nodes"]))
        self.assertTrue(any(node["op"] == "ADD" for node in topo["nodes"]))

if __name__ == "__main__":
    unittest.main()
