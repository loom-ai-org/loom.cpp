"""
Checks `passes.py`'s two R2a canonicalizing passes (EXPORT-ROADMAP.md): `normalize_matmul` and
`insert_explicit_broadcasts`. Both run as real MIL->MIL rewrites over a hand-built `mb.program`,
checked directly against the rewritten graph's structure -- `fuse_gqa_repeat_kv` (the pass these two
join) has no equivalent unit test of its own, relying entirely on numerical e2e reference tests instead,
but neither new pass has a model on the current roadmap that actually exercises its rewrite (matmul's
`transpose_x=True` has never been needed by any traced model; the only model needing mutual broadcast,
SupertonicTTS, already has an e2e reference test covering the *pipeline*, not this pass in isolation) --
so this is the only place either rewrite itself gets verified.
"""
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


if __name__ == "__main__":
    unittest.main()
