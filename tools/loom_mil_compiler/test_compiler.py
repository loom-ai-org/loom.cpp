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
from loom_mil_compiler.exporter import LoomGGUFExporter

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

    def test_multi_output_submodule_topology(self):
        """P2 (EXPORT-ROADMAP.md / BACKLOG.md's implementation sequence): a submodule genuinely traced
        with two real, independent outputs (e.g. LFM2's rotary-embedding table returning (cos, sin) --
        modular_export.py's own aux_output_names) must serialize BOTH into its topology's "outputs"
        array, not just the first (the pre-P2 behavior, which would silently drop the second output's
        entire computation from the topology via dead-node pruning). The driver's own SubgraphCall must
        capture both Lua locals from the one loom.run_subgraph call."""
        class MockOperation:
            def __init__(self, op_type, inputs, outputs):
                self.op_type = op_type
                self.inputs = inputs
                self.outputs = outputs
                self.blocks = []

        class MockVar:
            def __init__(self, name):
                self.name = name

        prog = Program()

        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 4), dtype=types.fp32)])
        def splitter(x):
            w = mb.const(val=np.ones((4, 4), dtype=np.float32) * 2.0, name="w")
            y = mb.matmul(x=x, y=w, name="matmul_node")
            z = mb.add(x=x, y=1.0, name="add_node")
            return y, z

        prog.functions["splitter"] = splitter.functions["main"]

        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 4), dtype=types.fp32)])
        def main_func(x):
            y = mb.add(x=x, y=1.0, name="placeholder")
            return mb.add(x=y, y=0.0, name="final")

        main_ops = list(main_func.functions["main"].operations)
        # main_ops[0] is a "const" (the "placeholder" add's own y=1.0 literal); the "add" op itself
        # (the one whose "x" input is the real function input, needed by MockOperation below) is
        # main_ops[1] -- a lone op depending directly on the function's own input placeholder Var
        # otherwise ends up with an empty `.inputs` dict (a coremltools quirk unrelated to this test).
        placeholder_op = main_ops[1]
        # Reuse the placeholder op's OWN output Var for the call's first output (same trick
        # test_multi_modular_program_transpilation uses): main's own declared final output already
        # refers to this Var by name, so it resolves correctly without also having to patch
        # main_func.functions["main"].outputs itself. The second output is a genuinely NEW call-site
        # local with no other consumer -- exactly the "captured but not read further" shape a real
        # aux_output_names[1] (e.g. "sin") can have.
        second_out = MockVar("splitter_out_1")
        main_ops[1] = MockOperation(op_type="splitter", inputs={"x": placeholder_op.inputs["x"]},
                                     outputs=[placeholder_op.outputs[0], second_out])
        main_func.functions["main"].operations = main_ops

        prog.functions["main"] = main_func.functions["main"]

        exporter = loom_mil_compiler.LoomGGUFBackend()
        exporter(prog, output_path=self.output_path, architecture="multi_output_submodule_test")

        reader = GGUFReader(self.output_path)
        self.assertIn("model.graph_topology.splitter", reader.fields)
        import json
        topo = json.loads(reader.fields["model.graph_topology.splitter"].parts[-1].tobytes().decode("utf-8"))

        # Both outputs declared (plural "outputs", not singular "output") -- P2's own schema choice.
        self.assertNotIn("output", topo)
        self.assertEqual(len(topo["outputs"]), 2)
        self.assertTrue(any(n["op"] == "MUL_MAT" for n in topo["nodes"]))
        # The SECOND output's own node ("add_node" -> ADD) must survive pruning too -- pre-P2 this
        # would have been dropped as unreachable from the (only) first declared output.
        self.assertTrue(any(n["op"] == "ADD" for n in topo["nodes"]))

        driver_script = reader.fields["model.driver_script"].parts[-1].tobytes().decode("utf-8")
        self.assertIn(f"local placeholder, {second_out.name} = loom.run_subgraph('splitter'", driver_script)

    def test_single_main_function_auto_generates_main_topo(self):
        """
        Verify that a single-"main" Program automatically serializes the entire
        main function into 'main_topo' and generates a wrapper Lua driver script.
        """
        prog = Program()
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4), dtype=types.fp32)])
        def main_func(x):
            y = mb.add(x=x, y=1.0, name="add_node")
            return mb.add(x=y, y=2.0, name="add_node2")

        prog.functions["main"] = main_func.functions["main"]
        
        backend = loom_mil_compiler.LoomGGUFBackend()
        backend(prog, output_path=self.output_path, flat_namespace=True, architecture="mono_test")
        
        self.assertTrue(os.path.exists(self.output_path))
        reader = GGUFReader(self.output_path)
        
        self.assertIn("model.graph_topology.main_topo", reader.fields)
        self.assertIn("model.driver_script", reader.fields)
        
        driver_script = reader.fields["model.driver_script"].parts[-1].tobytes().decode("utf-8")
        self.assertIn("loom.run_subgraph('main_topo'", driver_script)

    def test_random_normal_in_bespoke_driver_maps_to_gaussian_array(self):
        """EXPORT-IMPROVEMENT-BACKLOG.md item 4: a random_normal op in the driver-level 'main' function
        (the bespoke workflow, which -- unlike a static submodule topology -- genuinely can call host
        functions mid-script) should translate to loom.gaussian_array, the same host RNG hand-written
        drivers already use, not raise or silently mishandle."""
        prog = Program()

        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 4), dtype=types.fp32)])
        def dense_layer(x):
            w = mb.const(val=np.ones((4, 4), dtype=np.float32), name="w")
            return mb.matmul(x=x, y=w, name="matmul_node")

        prog.functions["dense_layer"] = dense_layer.functions["main"]

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def main_func(x):
            noise = mb.random_normal(shape=np.array([4], dtype=np.int32), mean=0.0, stddev=1.0, name="noise")
            return mb.add(x=noise, y=x, name="out")

        prog.functions["main"] = main_func.functions["main"]

        backend = loom_mil_compiler.LoomGGUFBackend()
        backend(prog, output_path=self.output_path, architecture="random_test")

        reader = GGUFReader(self.output_path)
        driver_script = reader.fields["model.driver_script"].parts[-1].tobytes().decode("utf-8")
        self.assertIn("loom.gaussian_array(4)", driver_script)

    def test_random_normal_in_static_topology_raises_actionable_error(self):
        """The SAME op inside a static topology (main_topo, generated via
        generate_graph_topology) must fail loudly and specifically -- ggml has no RNG-capable compute op,
        so this can never be satisfied there, unlike the bespoke-driver case above."""
        prog = Program()

        @mb.program(input_specs=[mb.TensorSpec(shape=(1,), dtype=types.fp32)])
        def main_func(x):
            noise = mb.random_normal(shape=np.array([4], dtype=np.int32), name="noise")
            return mb.add(x=noise, y=x, name="out")

        prog.functions["main"] = main_func.functions["main"]

        backend = loom_mil_compiler.LoomGGUFBackend()
        with self.assertRaises(NotImplementedError) as ctx:
            backend(prog, output_path=self.output_path, flat_namespace=True, architecture="random_test")
        self.assertIn("ggml has no RNG-capable compute op", str(ctx.exception))


class TestInputAxisValidation(unittest.TestCase):
    """`LoomGGUFExporter`'s axis-declaration checks (BACKLOG.md P4.0.2). Both of these were silent
    before: an undeclared second dynamic axis collapsed onto `root_axis`, and a declaration naming a
    static axis did nothing at all."""

    @staticmethod
    def _prog2(shape_a, shape_b):
        """`mb.program` binds input names from the decorated function's own parameter names, so these
        two builders exist per arity rather than one variadic one."""
        @mb.program(input_specs=[mb.TensorSpec(shape=shape_a, dtype=types.fp32),
                                 mb.TensorSpec(shape=shape_b, dtype=types.fp32)])
        def main_func(root, other):
            return mb.identity(x=root, name="out")

        return main_func

    @staticmethod
    def _prog3(shape_a, shape_b, shape_c):
        @mb.program(input_specs=[mb.TensorSpec(shape=shape_a, dtype=types.fp32),
                                 mb.TensorSpec(shape=shape_b, dtype=types.fp32),
                                 mb.TensorSpec(shape=shape_c, dtype=types.fp32)])
        def main_func(tokens, cache_position, attention_mask):
            return mb.identity(x=tokens, name="out")

        return main_func

    def test_one_shared_symbol_across_inputs_is_the_root_axis(self):
        """The idiom every model already uses: one `ct.RangeDim` instance shared by inputs whose lengths
        always match (causal-LM's tokens/cache_position/attention_mask) gives ONE symbol, so there is
        nothing to declare."""
        from coremltools.converters.mil.mil import get_new_symbol

        seq = get_new_symbol()
        prog = self._prog3((1, seq), (seq,), (1, 1, seq, seq))
        exporter = LoomGGUFExporter(prog)
        self.assertEqual(exporter.root_axis, "n_tokens")

    def test_two_independent_dynamic_axes_raise_naming_both(self):
        from coremltools.converters.mil.mil import get_new_symbol

        a, b = get_new_symbol(), get_new_symbol()
        prog = self._prog2((1, a), (1, b))
        with self.assertRaises(ValueError) as ctx:
            LoomGGUFExporter(prog)
        message = str(ctx.exception)
        self.assertIn("2 independent dynamic input axes", message)
        # Names the real inputs and axis positions, not just a count.
        self.assertIn("[1]", message)
        self.assertIn("declared_axes", message)

    def test_declaring_the_second_axis_makes_it_pass(self):
        """Kokoro's `decoder_vocoder` shape: a second leaf whose length is a fixed multiple of the root
        axis, not derivable from the graph."""
        from coremltools.converters.mil.mil import get_new_symbol

        a, b = get_new_symbol(), get_new_symbol()
        prog = self._prog2((1, a), (1, b))
        names = list(prog.functions["main"].inputs)
        exporter = LoomGGUFExporter(
            prog, root_axis="n_enc_frames", declared_axes={names[1]: {1: "2*n_enc_frames"}},
        )
        self.assertEqual(list(exporter._axis_overrides.values()), ["2*n_enc_frames"])

    def test_declaring_a_static_axis_raises_instead_of_doing_nothing(self):
        from coremltools.converters.mil.mil import get_new_symbol

        seq = get_new_symbol()
        prog = self._prog2((1, seq), (1, 4000))
        names = list(prog.functions["main"].inputs)
        with self.assertRaises(ValueError) as ctx:
            LoomGGUFExporter(prog, declared_axes={names[1]: {1: "2*n_tokens"}})
        self.assertIn("is static", str(ctx.exception))

    def test_a_program_with_no_main_function_is_not_validated(self):
        """The modular-blueprint Program has one Function per submodule and no "main" -- a real limit of
        this check, stated in `_input_axis_symbols`."""
        from coremltools.converters.mil.mil import get_new_symbol

        a, b = get_new_symbol(), get_new_symbol()
        prog = Program()
        prog.functions["layer_0"] = self._prog2((1, a), (1, b)).functions["main"]
        LoomGGUFExporter(prog)  # must not raise


if __name__ == "__main__":
    unittest.main()
