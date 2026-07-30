"""
Checks `passes.py`'s MIL->MIL canonicalizing passes (EXPORT-ROADMAP.md R2a/R2): `normalize_matmul`,
`insert_explicit_broadcasts`, `canonicalize_replicate_pad`, `canonicalize_conv_transpose_dw`,
`lower_stack`, `lower_reduce_mean`. Each runs as a real rewrite over a hand-built `mb.program`, checked
directly against the rewritten graph's structure -- `fuse_gqa_repeat_kv` (the pass these all join) has no
equivalent unit test of its own, relying entirely on numerical e2e reference tests instead, but not every
pass here has a model on the current roadmap that actually exercises its rewrite (matmul's
`transpose_x=True` has never been needed by any traced model; SupertonicTTS/Matcha/Kokoro/StyleTTS2 do
exercise the rest, via e2e reference tests covering the *pipeline*, not these passes in isolation) -- so
this is the only place every rewrite itself gets verified in isolation.
"""
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from coremltools.converters.mil.mil import Builder as mb, get_new_symbol, types

from loom_mil_compiler.passes import PASS_REGISTRY


def _ops(prog):
    return [op.op_type for op in prog.functions["main"].operations if op.op_type != "const"]


class TestNormalizeMatmul(unittest.TestCase):
    def test_rewrites_transpose_x_true(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32),
                                  mb.TensorSpec(shape=(2, 4), dtype=types.fp32)])
        def prog(x, y):
            return mb.matmul(x=x, y=y, transpose_x=True, transpose_y=False)

        PASS_REGISTRY["loom::normalize_matmul"](prog)

        self.assertEqual(_ops(prog), ["transpose", "matmul"])
        transpose_op, matmul_op = (op for op in prog.functions["main"].operations if op.op_type != "const")
        self.assertEqual(list(transpose_op.perm.val), [1, 0])
        self.assertFalse(bool(matmul_op.transpose_x.val))
        self.assertFalse(bool(matmul_op.transpose_y.val))
        self.assertEqual(matmul_op.x.op, transpose_op)
        # Output shape/name are preserved -- (3,4), matching the original transpose_x=True result.
        self.assertEqual(tuple(prog.functions["main"].outputs[0].shape), (3, 4))

    def test_preserves_transpose_y(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32),
                                  mb.TensorSpec(shape=(4, 2), dtype=types.fp32)])
        def prog(x, y):
            return mb.matmul(x=x, y=y, transpose_x=True, transpose_y=True)

        PASS_REGISTRY["loom::normalize_matmul"](prog)

        matmul_op = prog.functions["main"].operations[-1]
        self.assertFalse(bool(matmul_op.transpose_x.val))
        self.assertTrue(bool(matmul_op.transpose_y.val))
        self.assertEqual(tuple(prog.functions["main"].outputs[0].shape), (3, 4))

    def test_leaves_transpose_x_false_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32),
                                  mb.TensorSpec(shape=(4, 3), dtype=types.fp32)])
        def prog(x, y):
            return mb.matmul(x=x, y=y, transpose_x=False, transpose_y=True)

        PASS_REGISTRY["loom::normalize_matmul"](prog)

        self.assertEqual(_ops(prog), ["matmul"])


class TestInsertExplicitBroadcasts(unittest.TestCase):
    def test_mutual_broadcast_gets_two_loom_broadcast_to_ops(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(32, 1, 1), dtype=types.fp32),
                                  mb.TensorSpec(shape=(1, 5, 1), dtype=types.fp32)])
        def prog(x, y):
            return mb.mul(x=x, y=y)

        PASS_REGISTRY["loom::insert_explicit_broadcasts"](prog)

        self.assertEqual(_ops(prog), ["loom_broadcast_to", "loom_broadcast_to", "mul"])
        bx, by, mul_op = prog.functions["main"].operations
        self.assertEqual(mul_op.x.op, bx)
        self.assertEqual(mul_op.y.op, by)
        self.assertEqual(tuple(prog.functions["main"].outputs[0].shape), (32, 5, 1))

    def test_mutual_broadcast_with_a_dynamic_axis(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(32, 1, 1), dtype=types.fp32),
                                  mb.TensorSpec(shape=(1, length, 1), dtype=types.fp32)])
        def prog(x, y):
            return mb.add(x=x, y=y)

        PASS_REGISTRY["loom::insert_explicit_broadcasts"](prog)

        self.assertEqual(_ops(prog), ["loom_broadcast_to", "loom_broadcast_to", "add"])
        out_shape = prog.functions["main"].outputs[0].shape
        self.assertEqual(out_shape[0], 32)
        self.assertEqual(out_shape[1], length)
        self.assertEqual(out_shape[2], 1)

    def test_single_operand_broadcast_is_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 5), dtype=types.fp32),
                                  mb.TensorSpec(shape=(3, 5), dtype=types.fp32)])
        def prog(x, y):
            return mb.add(x=x, y=y)

        PASS_REGISTRY["loom::insert_explicit_broadcasts"](prog)

        self.assertEqual(_ops(prog), ["add"])

    def test_no_broadcast_needed_is_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(3, 5), dtype=types.fp32),
                                  mb.TensorSpec(shape=(3, 5), dtype=types.fp32)])
        def prog(x, y):
            return mb.mul(x=x, y=y)

        PASS_REGISTRY["loom::insert_explicit_broadcasts"](prog)

        self.assertEqual(_ops(prog), ["mul"])


class TestCanonicalizeReplicatePad(unittest.TestCase):
    def test_rewrites_a_replicate_pad_into_loom_replicate_pad(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, length), dtype=types.fp32)])
        def prog(x):
            return mb.pad(x=x, pad=[2, 3], mode="replicate")

        PASS_REGISTRY["loom::canonicalize_replicate_pad"](prog)

        self.assertEqual(_ops(prog), ["loom_replicate_pad"])
        op = prog.functions["main"].operations[-1]
        self.assertEqual(int(op.lp.val), 2)
        self.assertEqual(int(op.rp.val), 3)

    def test_a_zero_pad_is_aliased_away_entirely(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            padded = mb.pad(x=x, pad=[0, 0], mode="replicate")
            return mb.mul(x=padded, y=1.0)

        PASS_REGISTRY["loom::canonicalize_replicate_pad"](prog)

        self.assertEqual(_ops(prog), ["mul"])
        mul_op = prog.functions["main"].operations[-1]
        self.assertEqual(mul_op.x.name, "x")

    def test_non_replicate_modes_are_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            return mb.pad(x=x, pad=[2, 3], mode="constant", constant_val=0.0)

        PASS_REGISTRY["loom::canonicalize_replicate_pad"](prog)

        self.assertEqual(_ops(prog), ["pad"])

    def test_a_non_fastest_axis_replicate_pad_raises(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            # pad=[1,1,0,0]: the FIRST (non-fastest, mil_axis=1) of the two padded axes is
            # non-zero; the last (fastest) axis is left alone.
            return mb.pad(x=x, pad=[1, 1, 0, 0], mode="replicate")

        with self.assertRaises(NotImplementedError) as cm:
            PASS_REGISTRY["loom::canonicalize_replicate_pad"](prog)
        self.assertIn("fastest-varying", str(cm.exception))


class TestCanonicalizeConvTransposeDw(unittest.TestCase):
    def test_rewrites_a_depthwise_conv_transpose(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, length), dtype=types.fp32)])
        def prog(x):
            w = mb.const(val=np.zeros((4, 1, 3), dtype=np.float32), name="w")
            return mb.conv_transpose(x=x, weight=w, strides=[2], pad_type="valid", groups=4)

        PASS_REGISTRY["loom::canonicalize_conv_transpose_dw"](prog)

        self.assertEqual(_ops(prog), ["loom_conv_transpose_dw"])
        op = prog.functions["main"].operations[-1]
        self.assertEqual(int(op.stride.val), 2)
        self.assertIsNone(op.bias)

    def test_a_biased_depthwise_conv_transpose_keeps_its_bias(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            w = mb.const(val=np.zeros((4, 1, 3), dtype=np.float32), name="w")
            b = mb.const(val=np.ones((4,), dtype=np.float32), name="b")
            return mb.conv_transpose(x=x, weight=w, bias=b, strides=[2], pad_type="valid", groups=4)

        PASS_REGISTRY["loom::canonicalize_conv_transpose_dw"](prog)

        op = prog.functions["main"].operations[-1]
        self.assertEqual(op.op_type, "loom_conv_transpose_dw")
        self.assertIsNotNone(op.bias)
        self.assertTrue(np.all(op.bias.val == 1.0))

    def test_a_non_grouped_conv_transpose_is_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8), dtype=types.fp32)])
        def prog(x):
            w = mb.const(val=np.zeros((4, 4, 3), dtype=np.float32), name="w")
            return mb.conv_transpose(x=x, weight=w, strides=[2], pad_type="valid", groups=1)

        PASS_REGISTRY["loom::canonicalize_conv_transpose_dw"](prog)

        self.assertEqual(_ops(prog), ["conv_transpose"])

    def test_a_2d_grouped_conv_transpose_raises(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 4, 8, 8), dtype=types.fp32)])
        def prog(x):
            w = mb.const(val=np.zeros((4, 1, 3, 3), dtype=np.float32), name="w")
            return mb.conv_transpose(x=x, weight=w, strides=[2, 2], pad_type="valid", groups=4)

        with self.assertRaises(NotImplementedError) as cm:
            PASS_REGISTRY["loom::canonicalize_conv_transpose_dw"](prog)
        self.assertIn("groups=4", str(cm.exception))


class TestLowerStack(unittest.TestCase):
    def test_two_operand_stack_becomes_expand_dims_and_concat(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 5), dtype=types.fp32),
                                  mb.TensorSpec(shape=(4, 5), dtype=types.fp32)])
        def prog(a, b):
            return mb.stack(values=(a, b), axis=-1)

        PASS_REGISTRY["loom::lower_stack"](prog)

        self.assertEqual(_ops(prog), ["expand_dims", "expand_dims", "concat"])
        out_var = prog.functions["main"].outputs[0]
        self.assertEqual(tuple(out_var.shape), (4, 5, 2))

    def test_single_operand_stack_becomes_a_bare_expand_dims(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 5), dtype=types.fp32)])
        def prog(a):
            return mb.stack(values=(a,), axis=0)

        PASS_REGISTRY["loom::lower_stack"](prog)

        self.assertEqual(_ops(prog), ["expand_dims"])
        out_var = prog.functions["main"].outputs[0]
        self.assertEqual(tuple(out_var.shape), (1, 4, 5))


class TestLowerReduceMean(unittest.TestCase):
    def test_static_count_becomes_reduce_sum_and_loom_scale(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 192, 1), dtype=types.fp32)])
        def prog(x):
            return mb.reduce_mean(x=x, axes=[1], keep_dims=True)

        PASS_REGISTRY["loom::lower_reduce_mean"](prog)

        self.assertEqual(_ops(prog), ["reduce_sum", "loom_scale"])
        scale_op = prog.functions["main"].operations[-1]
        self.assertEqual(int(scale_op.n.val), 192)

    def test_dynamic_count_on_the_fastest_axis_becomes_loom_mean(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(1024, length), dtype=types.fp32)])
        def prog(x):
            return mb.reduce_mean(x=x, axes=[-1], keep_dims=False)

        PASS_REGISTRY["loom::lower_reduce_mean"](prog)

        self.assertEqual(_ops(prog), ["loom_mean"])

    def test_dynamic_count_on_a_non_fastest_axis_raises(self):
        length = get_new_symbol()

        @mb.program(input_specs=[mb.TensorSpec(shape=(length, 512), dtype=types.fp32)])
        def prog(x):
            return mb.reduce_mean(x=x, axes=[0], keep_dims=False)

        with self.assertRaises(NotImplementedError) as cm:
            PASS_REGISTRY["loom::lower_reduce_mean"](prog)
        self.assertIn("only known at run time", str(cm.exception))

    def test_multi_axis_reduce_mean_raises(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(4, 8, 1), dtype=types.fp32)])
        def prog(x):
            return mb.reduce_mean(x=x, axes=[0, 1], keep_dims=False)

        with self.assertRaises(NotImplementedError) as cm:
            PASS_REGISTRY["loom::lower_reduce_mean"](prog)
        self.assertIn("single reduction axis", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
