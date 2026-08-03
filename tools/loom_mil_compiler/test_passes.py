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

from loom_mil_compiler.passes import PASS_REGISTRY, apply_loom_mil_passes


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


class TestFuseLoomAttention(unittest.TestCase):
    """`fuse_loom_attention` (KV-CACHE.md stage 2). Unlike the other passes here, a miss is not merely a
    missed optimization: this op is the only node type that can reach the engine's KV cache, so a
    silently-unmatched block is a model that cannot generate. The negative cases matter as much -- the
    pattern is generic SDPA, and it must not fire on graphs that only look like it."""

    N_HEAD, HEAD_DIM, SEQ = 4, 8, 6

    def _sdpa(self, scale=0.35355338, transpose_y=True, trailing=True, n_blocks=1):
        n_head, head_dim, seq = self.N_HEAD, self.HEAD_DIM, self.SEQ
        specs = [mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32) for _ in range(3)]
        specs.append(mb.TensorSpec(shape=(1, 1, seq, seq), dtype=types.fp32))

        @mb.program(input_specs=specs)
        def prog(q, k, v, mask):
            out = None
            for _ in range(n_blocks):
                qs = q if scale is None else mb.mul(x=q, y=np.float32(scale))
                scores = mb.matmul(x=qs, y=k, transpose_x=False, transpose_y=transpose_y)
                scores = mb.add(x=scores, y=mask)
                probs = mb.softmax(x=scores, axis=-1)
                ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=False)
                if not trailing:
                    out = ctx
                    continue
                ctx = mb.transpose(x=ctx, perm=[0, 2, 1, 3])
                out = mb.reshape(x=ctx, shape=[1, seq, n_head * head_dim])
            return out

        return prog

    def _fused(self, prog):
        return [op for op in prog.functions["main"].operations if op.op_type == "loom_fused_attention"]

    def test_a_whole_sdpa_block_becomes_one_op(self):
        prog = self._sdpa()
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        fused = self._fused(prog)
        self.assertEqual(len(fused), 1)
        self.assertNotIn("softmax", _ops(prog))
        # The trailing transpose+reshape are absorbed too: op_attention returns the flattened context,
        # so the op's declared type has to be [b, seq, n_head*head_dim] or it disagrees with the engine.
        self.assertEqual(tuple(fused[0].outputs[0].shape), (1, self.SEQ, self.N_HEAD * self.HEAD_DIM))
        self.assertAlmostEqual(float(fused[0].scale.val), 0.35355338, places=6)

    def test_layer_indices_are_assigned_in_occurrence_order(self):
        # The index addresses a CACHE SLOT, and the cache has one slot per attention block -- so a dense
        # occurrence index is correct even for an architecture that interleaves non-attention layers,
        # where the torch module index would address past the end of the cache.
        prog = self._sdpa(n_blocks=3)
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual([int(op.layer.val) for op in self._fused(prog)], [0, 1, 2])

    def test_an_unscaled_block_fuses_with_scale_1(self):
        # Recovered from the graph rather than recomputed as 1/sqrt(head_dim): a model with no scale (or
        # a non-default one) must not silently acquire one that was never traced.
        prog = self._sdpa(scale=None)
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        fused = self._fused(prog)
        self.assertEqual(len(fused), 1)
        self.assertAlmostEqual(float(fused[0].scale.val), 1.0, places=6)

    def test_a_matmul_that_is_not_q_at_k_transposed_is_untouched(self):
        # Built by hand rather than via _sdpa: with transpose_y=False the two operands genuinely do not
        # contract, so K has to be pre-transposed for the graph to type-check at all. That IS the shape
        # this guards -- a softmax fed by an ordinary matmul is not an attention block, and `transpose_y`
        # is how the traced pattern spells Q @ K^T.
        n_head, head_dim, seq = self.N_HEAD, self.HEAD_DIM, self.SEQ

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_head, head_dim, seq), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, 1, seq, seq), dtype=types.fp32),
        ])
        def prog(q, k_t, v, mask):
            scores = mb.matmul(x=q, y=k_t, transpose_x=False, transpose_y=False)
            scores = mb.add(x=scores, y=mask)
            probs = mb.softmax(x=scores, axis=-1)
            ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=False)
            ctx = mb.transpose(x=ctx, perm=[0, 2, 1, 3])
            return mb.reshape(x=ctx, shape=[1, seq, n_head * head_dim])

        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(self._fused(prog), [])
        self.assertIn("softmax", _ops(prog))

    def test_a_block_without_the_trailing_reshape_is_untouched(self):
        # A partial match must never half-rewrite: what does not match exports and runs as before, just
        # without a cache.
        prog = self._sdpa(trailing=False)
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(self._fused(prog), [])
        self.assertIn("softmax", _ops(prog))

    def test_a_bare_softmax_is_untouched(self):
        @mb.program(input_specs=[mb.TensorSpec(shape=(2, 3), dtype=types.fp32)])
        def prog(x):
            return mb.softmax(x=x, axis=-1)

        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        self.assertEqual(self._fused(prog), [])
        self.assertEqual(_ops(prog), ["softmax"])


class TestFuseLoomAttentionStripsGqaRepeat(unittest.TestCase):
    """KV-CACHE.md 2.3. `op_attention` reads n_head_kv off K's own shape and lets ggml_mul_mat's
    broadcast map query head i to KV head i // ratio -- the same interleaved correspondence repeat_kv()
    materializes -- so attending against the UN-repeated K/V is identical arithmetic on half the cache.
    Correctness never depends on the strip, which is why every guard bails to "leave it alone"."""

    N_HEAD, N_KV, HEAD_DIM, SEQ = 4, 2, 8, 6

    def _prog(self, expand_v=True):
        n_head, n_kv, head_dim, seq = self.N_HEAD, self.N_KV, self.HEAD_DIM, self.SEQ
        ratio = n_head // n_kv
        v_heads = n_kv if expand_v else n_head

        @mb.program(input_specs=[
            mb.TensorSpec(shape=(1, n_head, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, n_kv, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, v_heads, seq, head_dim), dtype=types.fp32),
            mb.TensorSpec(shape=(1, 1, seq, seq), dtype=types.fp32),
        ])
        def prog(q, k_kv, v_in, mask):
            def expand(x):
                # Exactly what fuse_gqa_repeat_kv leaves behind: reshape -> tile -> reshape.
                r1 = mb.reshape(x=x, shape=[n_kv, 1, seq, head_dim])
                rep = mb.tile(x=r1, reps=[1, ratio, 1, 1])
                return mb.reshape(x=rep, shape=[1, n_head, seq, head_dim])
            k = expand(k_kv)
            v = expand(v_in) if expand_v else v_in
            qs = mb.mul(x=q, y=np.float32(0.35355338))
            scores = mb.matmul(x=qs, y=k, transpose_x=False, transpose_y=True)
            scores = mb.add(x=scores, y=mask)
            probs = mb.softmax(x=scores, axis=-1)
            ctx = mb.matmul(x=probs, y=v, transpose_x=False, transpose_y=False)
            ctx = mb.transpose(x=ctx, perm=[0, 2, 1, 3])
            return mb.reshape(x=ctx, shape=[1, seq, n_head * head_dim])

        return prog

    def test_k_and_v_come_from_before_the_repeat(self):
        prog = self._prog()
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        fused = [op for op in prog.functions["main"].operations
                 if op.op_type == "loom_fused_attention"]
        self.assertEqual(len(fused), 1)
        # The stored heads are the checkpoint's own, not the expanded ones -- half the cache.
        self.assertEqual(fused[0].k.shape[1], self.N_KV)
        self.assertEqual(fused[0].v.shape[1], self.N_KV)
        self.assertEqual(fused[0].q.shape[1], self.N_HEAD)

    def test_k_is_not_stripped_alone_when_v_has_no_expansion(self):
        # The rule that matters most here: stripping one and not the other would leave the cache's K and
        # V widths disagreeing, and nothing downstream would catch it. Fusion still happens; the strip
        # declines, and both stay expanded.
        prog = self._prog(expand_v=False)
        PASS_REGISTRY["loom::fuse_loom_attention"](prog)

        fused = [op for op in prog.functions["main"].operations
                 if op.op_type == "loom_fused_attention"]
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].k.shape[1], self.N_HEAD)
        self.assertEqual(fused[0].v.shape[1], self.N_HEAD)


class TestAttentionFusionIsOptIn(unittest.TestCase):
    """Decision 4, and it is a correctness requirement rather than caution: the pattern is generic SDPA,
    so it matches the non-autoregressive TTS families' self-attention too -- and an ATTENTION node's
    `kv_cache` attr defaults to TRUE, so firing there would hand them persistent state they must never
    have (and would break their byte-identity gates)."""

    def _prog(self):
        return TestFuseLoomAttention()._sdpa()

    def test_the_pipeline_does_not_fuse_by_default(self):
        prog = self._prog()
        apply_loom_mil_passes(prog)
        self.assertEqual([op for op in prog.functions["main"].operations
                          if op.op_type == "loom_fused_attention"], [])

    def test_the_pipeline_fuses_when_asked(self):
        prog = self._prog()
        apply_loom_mil_passes(prog, fuse_attention=True)
        self.assertEqual(len([op for op in prog.functions["main"].operations
                              if op.op_type == "loom_fused_attention"]), 1)


if __name__ == "__main__":
    unittest.main()
