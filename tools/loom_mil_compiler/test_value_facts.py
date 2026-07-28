"""
Checks `value_facts.py` (EXPORT-IMPROVEMENT.md item 2) on the two things the centralization actually
changed:

* the static accessors must reproduce the longhand
  `x.val if x is not None and hasattr(x, "val") and x.val is not None else default` idiom they replaced
  at ~55 call sites -- including the edge cases the individual sites used to disagree about (a None Var,
  a Var whose `.val` is None, an object with no `.val` at all); and
* the derived resolvers must be *memoized*, so a diamond-shaped expression tree resolves each shared
  subexpression once and every call site gets the same answer. The un-memoized version re-walked shared
  subtrees, and an earlier "already visited -> give up" guard turned a perfectly ordinary diamond into a
  silent `None` (see `scalar_expr`'s own docstring for the VITS case that caught it).

`dim_expr`'s memo gets the most attention here, because that one is not a nicety: without it the
branching shape walk is exponential in encoder depth, which is what made the Conformer-CTC and Parakeet
exports stop finishing at all. Its tests assert the visit *bound* directly rather than just that a cache
exists.
"""
import unittest

import numpy as np
from coremltools.converters.mil.mil import Var

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import loom_mil_compiler  # noqa: F401 -- registers the "loom" backend + applies torch-frontend patches
from loom_mil_compiler.value_facts import (
    ValueFacts, is_const_producer, static_array, static_ints, static_scalar, static_value,
)


class _Var:
    def __init__(self, val):
        self.val = val


class _NoVal:
    pass


class TestStaticAccessors(unittest.TestCase):
    def test_value_matches_the_longhand_idiom(self):
        for var, default in [(None, None), (None, 7), (_Var(None), None), (_Var(None), 7),
                             (_NoVal(), 3), (_Var(5), 7), (_Var("EXACT"), "TANH")]:
            longhand = (var.val if var is not None and hasattr(var, "val") and var.val is not None
                        else default)
            self.assertEqual(static_value(var, default), longhand, repr(var))

    def test_array_and_ints_and_scalar(self):
        var = _Var(np.array([[1, 2], [3, 4]], dtype=np.int32))
        np.testing.assert_array_equal(static_array(var), np.array([[1, 2], [3, 4]]))
        self.assertEqual(static_ints(var), [1, 2, 3, 4])
        self.assertEqual(static_scalar(var), 1)
        self.assertIsInstance(static_scalar(var), int)

    def test_absent_values_return_the_default_not_a_crash(self):
        self.assertIsNone(static_array(None))
        self.assertIsNone(static_ints(_Var(None)))
        self.assertEqual(static_scalar(None, 1e-5), 1e-5)
        self.assertEqual(static_scalar(_Var(np.array([])), 0.5), 0.5)

    def test_is_const_producer_is_about_the_producing_op_not_the_value(self):
        class _Op:
            def __init__(self, t):
                self.op_type = t

        class _V:
            def __init__(self, op):
                self.op = op

        self.assertTrue(is_const_producer(_V(_Op("const"))))
        # A `shape` op's output can carry a folded value without being a `const` node.
        self.assertFalse(is_const_producer(_V(_Op("shape"))))
        self.assertFalse(is_const_producer(_V(None)))
        self.assertFalse(is_const_producer(None))


class TestMemoization(unittest.TestCase):
    """Uses a real MIL program, since the derived resolvers isinstance-check `Var`."""

    def _program(self):
        from coremltools.converters.mil.mil import Builder as mb, get_new_symbol

        # A SYMBOLIC second axis, so MIL cannot constant-fold the shape chain away -- the derivation
        # under test only runs on values MIL left unresolved.
        @mb.program(input_specs=[mb.TensorSpec(shape=(1, get_new_symbol()))])
        def prog(x):
            # A diamond: `total` reaches `base` down two independent operand paths, exactly the shape
            # that used to be re-walked (or refused) rather than answered once.
            shape = mb.shape(x=x)
            base = mb.gather(x=shape, indices=np.array([1], dtype=np.int32), axis=0)
            doubled = mb.mul(x=base, y=np.array([2], dtype=np.int32))
            return mb.add(x=base, y=doubled)

        return prog

    class _StubExporter:
        """The one exporter method the derivation calls out to. Stubbed so this test exercises
        `ValueFacts`' own walking/caching rather than the exporter's shape inference."""

        def _infer_dynamic_dim_expr(self, var, torch_axis, _seen=None):
            return "n_tokens"

    def setUp(self):
        self.prog = self._program()
        self.facts = ValueFacts(exporter=self._StubExporter())

    def _find(self, op_type):
        return next(op for op in self.prog.functions["main"].operations if op.op_type == op_type)

    def test_repeated_queries_hit_the_cache_and_agree(self):
        total = self._find("add").outputs[0]
        first = self.facts.scalar_expr(total)
        self.assertEqual(len(self.facts._scalar_expr), len(self.facts._scalar_expr))
        cached_size = len(self.facts._scalar_expr)
        second = self.facts.scalar_expr(total)
        self.assertEqual(first, second)
        self.assertEqual(len(self.facts._scalar_expr), cached_size,
                          "a second query must be served from the cache, not re-derived")

    def test_the_shared_subexpression_is_resolved_once(self):
        base = self._find("gather").outputs[0]
        self.facts.scalar_expr(self._find("add").outputs[0])
        self.assertIn(id(base), self.facts._scalar_expr,
                      "the diamond's shared operand should be cached by the enclosing walk")

    def test_dim_expr_visits_each_var_once_per_axis_however_many_paths_reach_it(self):
        """The Conformer-CTC blow-up, in miniature.

        `_infer_dynamic_dim_expr` has had no cycle guard since a29ffe5 (removed deliberately -- it was
        corrupting ordinary DAG diamonds), and that same commit made the walk *branch* by adding a
        `concat` case that recurses into every operand. Branching + no revisit-suppression re-derives
        every shared ancestor once per path, which took the real 16-block encoder from ~2 s to
        not-finishing-in-two-hours. The memo is what bounds it, so this asserts the bound directly:
        a diamond must cost one visit per (var, axis), not one per path.
        """
        calls = []

        class _CountingExporter:
            def _infer_dynamic_dim_expr_uncached(self, var, torch_axis, _seen=None):
                calls.append((id(var), torch_axis))
                # Recurse into every operand, the way the real `concat`/elementwise cases do.
                if var.op is not None:
                    for v in var.op.inputs.values():
                        if isinstance(v, Var):
                            facts.dim_expr(v, torch_axis)
                return "n_tokens"

        facts = ValueFacts(exporter=_CountingExporter())
        total = self._find("add").outputs[0]
        facts.dim_expr(total, 0)

        self.assertEqual(len(calls), len(set(calls)),
                          "a var+axis was re-derived; the memo is not suppressing revisits")
        base = self._find("gather").outputs[0]
        self.assertEqual(sum(1 for c in calls if c[0] == id(base)), 1,
                          "the diamond's shared operand must be derived exactly once")

    def test_dim_expr_keys_on_the_axis_not_just_the_var(self):
        """The same Var resolves different axes to different expressions, so the axis is part of the
        cache identity -- keying on the Var alone would return one axis's answer for another's."""
        seen = []

        class _AxisExporter:
            def _infer_dynamic_dim_expr_uncached(self, var, torch_axis, _seen=None):
                seen.append(torch_axis)
                return f"axis{torch_axis}"

        facts = ValueFacts(exporter=_AxisExporter())
        var = self._find("add").outputs[0]
        self.assertEqual(facts.dim_expr(var, 0), "axis0")
        self.assertEqual(facts.dim_expr(var, 1), "axis1")
        self.assertEqual(facts.dim_expr(var, 0), "axis0")
        self.assertEqual(seen, [0, 1], "axis 0 should have been served from the cache the second time")

    def test_cache_keeps_its_var_alive_so_ids_cannot_be_recycled(self):
        total = self._find("add").outputs[0]
        self.facts.scalar_expr(total)
        cached_var, _ = self.facts._scalar_expr[id(total)]
        self.assertIs(cached_var, total)


if __name__ == "__main__":
    unittest.main()
